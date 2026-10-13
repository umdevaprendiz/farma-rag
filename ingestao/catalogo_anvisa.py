"""Gera o catálogo de medicamentos (nome comercial <-> princípio ativo) a partir do CSV da ANVISA.

Uso (da raiz):  python -I ingestao/catalogo_anvisa.py
Entrada: data/raw/anvisa/DADOS_ABERTOS_MEDICAMENTOS.csv (valide antes com validar_csv_anvisa.py)
Saída:   data/processed/catalogo_anvisa.jsonl  (uma linha por par nome/princípio ativo)
         data/processed/catalogo_anvisa.meta.json  (procedência e contagens)

A saída é determinística (sem data/hora): a mesma entrada gera exatamente os mesmos bytes,
então o Git só mostra diferença quando a ANVISA mudar os dados. Só biblioteca padrão.
"""

import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

ENTRADA = Path("data/raw/anvisa/DADOS_ABERTOS_MEDICAMENTOS.csv")
SAIDA = Path("data/processed/catalogo_anvisa.jsonl")
META = Path("data/processed/catalogo_anvisa.meta.json")

_ESPACOS = re.compile(r"\s+")
_SEPARADOR = re.compile(r",\s+")  # vírgula seguida de espaço; preserva "0,1%" e "1,2-"
_PREFIXO_NOME = re.compile(r"^[\s\-]+")  # cadastros antigos como "- LIQUIDO DE DAKIN"


def chave(texto: str) -> str:
    """Minúsculas, sem acento, espaços colapsados: chave de agrupamento."""
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return _ESPACOS.sub(" ", sem_acento.lower()).strip()


def limpar_nome(nome: str) -> str:
    return _ESPACOS.sub(" ", _PREFIXO_NOME.sub("", nome)).strip()


def separar_principios(bruto: str) -> list[str]:
    """'losartana potássica, hidroclorotiazida' -> ['losartana potássica', 'hidroclorotiazida']."""
    partes = (_ESPACOS.sub(" ", p).strip(" ,;") for p in _SEPARADOR.split(bruto.strip()))
    return [p for p in partes if p]


def construir_catalogo(registros: Iterable[dict]) -> tuple[list[dict], dict]:
    """Agrupa registros ATIVOS por (nome, princípio ativo). Devolve (itens, contagens)."""
    grupos: dict[tuple[str, str], dict] = {}
    contagens = Counter()
    for r in registros:
        contagens["linhas_lidas"] += 1
        if r["SITUACAO_REGISTRO"] != "Ativo":
            continue
        contagens["linhas_ativas"] += 1
        nome = limpar_nome(r["NOME_PRODUTO"])
        principios = separar_principios(r["PRINCIPIO_ATIVO"])
        if not nome or not principios:
            contagens["ativas_sem_nome_ou_principio"] += 1
            continue

        k = (chave(nome), chave(", ".join(principios)))
        g = grupos.setdefault(
            k,
            {"nomes": Counter(), "principios": principios, "classes": Counter(), "categorias": Counter(), "n": 0},
        )
        g["nomes"][nome] += 1
        g["n"] += 1
        if r["CLASSE_TERAPEUTICA"].strip():
            g["classes"][r["CLASSE_TERAPEUTICA"].strip()] += 1
        if r["CATEGORIA_REGULATORIA"].strip():
            g["categorias"][r["CATEGORIA_REGULATORIA"].strip()] += 1

    itens = []
    for g in grupos.values():
        associacao = len(g["principios"]) > 1
        itens.append(
            {
                # grafia mais frequente; em empate, a menor, para ser determinístico
                "nome_produto": min(g["nomes"], key=lambda n: (-g["nomes"][n], n)),
                "principios_ativos": g["principios"],
                "associacao": associacao,
                # A classe é por registro e ruidosa em associações: só guardamos nas de 1 princípio.
                "classe_terapeutica": (
                    None if associacao or not g["classes"]
                    else min(g["classes"], key=lambda c: (-g["classes"][c], c))
                ),
                "categoria_regulatoria": (
                    min(g["categorias"], key=lambda c: (-g["categorias"][c], c)) if g["categorias"] else None
                ),
                "n_registros": g["n"],
            }
        )
    itens.sort(key=lambda i: (chave(", ".join(i["principios_ativos"])), chave(i["nome_produto"])))

    contagens["itens"] = len(itens)
    contagens["itens_monocomponente"] = sum(not i["associacao"] for i in itens)
    contagens["principios_ativos_distintos_monocomponente"] = len(
        {chave(i["principios_ativos"][0]) for i in itens if not i["associacao"]}
    )
    return itens, dict(contagens)


def gerar(entrada: Path = ENTRADA, saida: Path = SAIDA, meta: Path = META) -> dict:
    bruto = entrada.read_bytes()
    texto = bruto.decode("cp1252")
    leitor = csv.DictReader(texto.splitlines(), delimiter=";")
    itens, contagens = construir_catalogo(leitor)

    saida.parent.mkdir(parents=True, exist_ok=True)
    with saida.open("w", encoding="utf-8", newline="\n") as f:
        for item in itens:
            f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    info = {
        "fonte": "https://dados.anvisa.gov.br/dados/DADOS_ABERTOS_MEDICAMENTOS.csv",
        "fonte_sha256": hashlib.sha256(bruto).hexdigest(),
        "fonte_bytes": len(bruto),
        "saida_sha256": hashlib.sha256(saida.read_bytes()).hexdigest(),
        "contagens": contagens,
    }
    meta.write_text(json.dumps(info, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return info


if __name__ == "__main__":
    if not ENTRADA.exists():
        sys.exit(f"Arquivo não encontrado: {ENTRADA} (baixe e valide antes; ver data/FONTES.md)")
    resultado = gerar()
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
