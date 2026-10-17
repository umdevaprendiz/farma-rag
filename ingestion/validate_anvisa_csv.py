"""Valida o CSV de medicamentos da ANVISA antes de qualquer uso (arquivo inteiro).

Uso (da raiz):  python -I ingestion/validate_anvisa_csv.py [caminho]
Só biblioteca padrão. Só LÊ o arquivo (nunca o executa nem o abre em planilha).
Sai com código 1 se algo suspeito for encontrado.
"""

import csv
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

PADRAO = Path("data/raw/anvisa/DADOS_ABERTOS_MEDICAMENTOS.csv")
COLUNAS = [
    "TIPO_PRODUTO", "NOME_PRODUTO", "DATA_FINALIZACAO_PROCESSO", "CATEGORIA_REGULATORIA",
    "NUMERO_REGISTRO_PRODUTO", "DATA_VENCIMENTO_REGISTRO", "NUMERO_PROCESSO",
    "CLASSE_TERAPEUTICA", "EMPRESA_DETENTORA_REGISTRO", "SITUACAO_REGISTRO", "PRINCIPIO_ATIVO",
]
ASSINATURAS = {b"MZ": "exe", b"PK": "zip", b"%PDF": "pdf", b"\x7fELF": "elf", b"#!": "script",
               b"<html": "html", b"<!DOC": "html", b"<?xml": "xml", b"\x1f\x8b": "gzip"}
SUSPEITO = re.compile(
    r"<\s*(script|iframe|object|embed)\b|<\?php|javascript:|vbscript:|powershell|cmd\.exe|https?://",
    re.I,
)
# "- NOME" (hífen + espaço) é prefixo de cadastro antigo, não fórmula; "-X", "=", "+", "@" são.
FORMULA = re.compile(r"[=+@\t\r]|-(?!\s)")
HIFEN_INICIAL = re.compile(r"-\s")
NUMERO = re.compile(r"-?\d+([.,]\d+)?")


def validar(caminho: Path) -> tuple[dict, list[str]]:
    problemas: list[str] = []
    bruto = caminho.read_bytes()
    rel = {
        "arquivo": str(caminho),
        "bytes": len(bruto),
        "sha256": hashlib.sha256(bruto).hexdigest(),
    }

    assinatura = next((v for k, v in ASSINATURAS.items() if bruto.startswith(k)), None)
    if assinatura:
        problemas.append(f"assinatura de arquivo {assinatura}, não de CSV")
    rel["bytes_nulos"] = bruto.count(b"\x00")
    rel["controles_estranhos"] = sum(bruto.count(bytes([b])) for b in range(32) if b not in (9, 10, 13))
    if rel["bytes_nulos"] or rel["controles_estranhos"]:
        problemas.append("bytes nulos ou de controle no arquivo")

    try:
        texto = bruto.decode("cp1252")
    except UnicodeDecodeError as e:
        problemas.append(f"não decodifica como cp1252: {e}")
        return rel, problemas

    linhas = csv.reader(texto.splitlines(), delimiter=";")
    cabecalho = next(linhas)
    if cabecalho != COLUNAS:
        problemas.append(f"cabeçalho diferente do esperado: {cabecalho}")

    campos, situacao, tipo = Counter(), Counter(), Counter()
    formulas, suspeitos, hifen = [], [], []
    for n, row in enumerate(linhas, start=2):
        campos[len(row)] += 1
        if len(row) == len(COLUNAS):
            situacao[row[9]] += 1
            tipo[row[0]] += 1
        for c in row:
            if FORMULA.match(c) and not NUMERO.fullmatch(c):
                formulas.append((n, c[:50]))
            if HIFEN_INICIAL.match(c):
                hifen.append(n)
            if SUSPEITO.search(c):
                suspeitos.append((n, c[:50]))

    rel["linhas_de_dados"] = sum(campos.values())
    rel["campos_por_linha"] = dict(campos)
    rel["situacao_registro"] = dict(situacao)
    rel["tipo_produto"] = dict(tipo)
    rel["celulas_tipo_formula"] = formulas[:10]
    rel["celulas_suspeitas"] = suspeitos[:10]
    rel["nomes_com_hifen_inicial (informativo)"] = len(hifen)
    if set(campos) != {len(COLUNAS)}:
        problemas.append("há linhas com número de campos diferente de 11")
    if formulas:
        problemas.append(f"{len(formulas)} células começam como fórmula")
    if suspeitos:
        problemas.append(f"{len(suspeitos)} células com padrões suspeitos (tag, URL, comando)")
    return rel, problemas


if __name__ == "__main__":
    alvo = Path(sys.argv[1]) if len(sys.argv) > 1 else PADRAO
    relatorio, achados = validar(alvo)
    for k, v in relatorio.items():
        print(f"{k}: {v}")
    print("\nRESULTADO:", "OK, nada suspeito" if not achados else "PROBLEMAS -> " + "; ".join(achados))
    sys.exit(1 if achados else 0)
