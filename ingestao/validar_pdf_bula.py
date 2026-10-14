"""Valida o PDF de uma bula antes de qualquer uso (arquivo inteiro).

Uso (da raiz):  python -I ingestao/validar_pdf_bula.py <arquivo.pdf> [bytes_esperados]
Só LÊ o arquivo (nunca o abre num leitor de PDF nem executa nada dele).
Sai com código 1 se algo suspeito for encontrado.

Duas passagens, porque uma só não vê tudo:
1. bytes crus, fora dos streams: pega objeto solto que nenhuma tabela do PDF referencia;
2. todos os objetos pelo pdfminer: pega o que está dentro de stream de objetos comprimido.
"""

import hashlib
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFParser
from pdfminer.pdftypes import PDFStream
from pdfminer.psparser import PSLiteral

MAX_BYTES = 20_000_000  # bula passa longe disso; maior que isso não é bula
ASSINATURAS = {b"MZ": "exe", b"PK": "zip", b"\x7fELF": "elf", b"#!": "script",
               b"<html": "html", b"<!DOC": "html", b"<?xml": "xml", b"\x1f\x8b": "gzip"}
# Chaves que só existem para conteúdo ativo: script, ação automática, formulário XFA, anexo, mídia.
CHAVES_ATIVAS = {"JS", "JavaScript", "AA", "XFA", "EmbeddedFiles", "EmbeddedFile", "RichMedia", "Launch"}
# Tipos de ação (/S) que executam, enviam dados ou abrem outro arquivo. "URI" (link) e "GoTo"
# (pular de página) são normais em bula e ficam de fora; os links são listados para conferir.
ACOES_ATIVAS = {"JavaScript", "Launch", "SubmitForm", "ImportData", "GoToR", "GoToE", "Movie",
                "Sound", "Rendition", "RichMediaExecute"}
_STREAM = re.compile(rb"stream\r?\n.*?endstream", re.S)
_ESCAPE = re.compile(rb"#([0-9A-Fa-f]{2})")  # "/J#61vaScript" é "/JavaScript"
_NOME = re.compile(rb"/([A-Za-z0-9]+)")


def _nome(valor) -> str | None:
    if not isinstance(valor, PSLiteral):
        return None
    return valor.name if isinstance(valor.name, str) else valor.name.decode("latin-1")


def _texto(valor) -> str:
    return valor.decode("latin-1") if isinstance(valor, bytes) else str(valor)


def _visitar(obj, ativos: set[str], links: set[str]) -> None:
    """Desce por dicionários e listas. Referências não são seguidas: cada objeto é visitado pela tabela."""
    if isinstance(obj, PDFStream):
        obj = obj.attrs
    if isinstance(obj, dict):
        ativos.update(f"/{c}" for c in obj if c in CHAVES_ATIVAS)
        acao = _nome(obj.get("S"))
        if acao in ACOES_ATIVAS:
            ativos.add(f"ação /{acao}")
        if acao == "URI" and isinstance(obj.get("URI"), (bytes, str)):
            links.add(urlparse(_texto(obj["URI"])).netloc or _texto(obj["URI"])[:60])
        if _nome(obj.get("Type")) in ("EmbeddedFile", "Filespec"):
            ativos.add(f"/Type /{_nome(obj['Type'])}")
        for valor in obj.values():
            _visitar(valor, ativos, links)
    elif isinstance(obj, list):
        for valor in obj:
            _visitar(valor, ativos, links)


def validar(caminho: Path, bytes_esperados: int | None = None) -> tuple[dict, list[str]]:
    problemas: list[str] = []
    bruto = caminho.read_bytes()
    rel = {
        "arquivo": str(caminho),
        "bytes": len(bruto),
        "sha256": hashlib.sha256(bruto).hexdigest(),
    }
    if bytes_esperados is not None and len(bruto) != bytes_esperados:
        problemas.append(f"tamanho diferente do anunciado pelo servidor ({bytes_esperados} bytes)")
    if len(bruto) > MAX_BYTES:
        problemas.append(f"maior que {MAX_BYTES} bytes")

    assinatura = next((v for k, v in ASSINATURAS.items() if bruto.startswith(k)), None)
    if assinatura:
        problemas.append(f"assinatura de arquivo {assinatura}, não de PDF")
    if not bruto.startswith(b"%PDF-"):
        problemas.append("não começa com %PDF-")
        return rel, problemas
    rel["versao"] = bruto[5:8].decode("latin-1")
    if b"%%EOF" not in bruto[-1024:]:
        problemas.append("sem %%EOF no fim (arquivo truncado ou com dados depois do PDF)")

    fora_dos_streams = _ESCAPE.sub(lambda m: bytes([int(m[1], 16)]), _STREAM.sub(b"", bruto))
    nomes = {n.decode("latin-1") for n in _NOME.findall(fora_dos_streams)}
    crus = sorted(f"/{n}" for n in nomes & (CHAVES_ATIVAS | ACOES_ATIVAS))
    rel["nomes_ativos_nos_bytes"] = crus
    if "Encrypt" in nomes:
        problemas.append("PDF cifrado: o conteúdo não pode ser inspecionado")

    ativos: set[str] = set()
    links: set[str] = set()
    objetos = ilegiveis = paginas = 0
    try:
        with caminho.open("rb") as f:
            doc = PDFDocument(PDFParser(f))
            for xref in doc.xrefs:
                _visitar(xref.get_trailer(), ativos, links)
                for objid in xref.get_objids():
                    try:
                        obj = doc.getobj(objid)
                    except Exception:  # objeto livre ou quebrado: conta, não derruba a validação
                        ilegiveis += 1
                        continue
                    objetos += 1
                    atributos = obj.attrs if isinstance(obj, PDFStream) else obj
                    paginas += isinstance(atributos, dict) and _nome(atributos.get("Type")) == "Page"
                    _visitar(obj, ativos, links)
    except Exception as e:
        problemas.append(f"o pdfminer não conseguiu ler a estrutura: {type(e).__name__}: {e}")

    rel["objetos"] = objetos
    rel["objetos_ilegiveis"] = ilegiveis
    rel["paginas"] = paginas
    rel["conteudo_ativo"] = sorted(ativos)
    rel["links (informativo)"] = sorted(links)
    if crus or ativos:
        problemas.append(f"conteúdo ativo: {', '.join(sorted(ativos | set(crus)))}")
    if objetos and not paginas:
        problemas.append("nenhuma página encontrada")
    return rel, problemas


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    relatorio, achados = validar(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else None)
    for k, v in relatorio.items():
        print(f"{k}: {v}")
    print("\nRESULTADO:", "OK, nada suspeito" if not achados else "PROBLEMAS -> " + "; ".join(achados))
    sys.exit(1 if achados else 0)
