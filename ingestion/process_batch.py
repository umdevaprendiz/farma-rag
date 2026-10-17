"""Processa todas as bulas do manifesto (data/bulas_manifesto.json) de uma vez.

Uso (da raiz):
  python -m ingestion.process_batch                    (todas)
  python -m ingestion.process_batch --so losartana_cozaar_profissional.pdf

Para cada PDF, antes de gravar qualquer coisa, confere:
- o SHA-256 contra o manifesto (é o mesmo arquivo que foi verificado e registrado em FONTES.md);
- quantas bulas o PDF tem contra as `partes` do manifesto (um PDF pode trazer gotas + comprimido);
- o número de registro do manifesto contra o impresso em cada bula;
- a versão do manifesto contra a impressa (quando as duas existem).
`data_versao` vazia no manifesto é tirada do código de versão impresso ("V.01_12/2025" -> 2025-12-01).

Um PDF com problema não grava nada e o lote termina com código 1. Depois rode
python -m ingestion.generate_chunks.
"""

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import get_settings
from app.schemas.dataset import Leaflet
from ingestion.extract_pdf import extrair_paginas
from ingestion.process_leaflet import gravar_bula, montar_bula
from ingestion.split_sections import ESPERADAS, LeafletPart, separar_bulas

MANIFESTO = Path("data/bulas_manifesto.json")


class ManifestPart(BaseModel):
    medicamento: str
    forma_farmaceutica: str
    data_versao: date | None = None
    versao: str | None = None


class ManifestItem(BaseModel):
    arquivo: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fabricante: str
    numero_registro: str = Field(pattern=r"^\d\.\d{4}\.\d{4}$")
    principio_ativo: str
    nomes_comerciais: list[str] = Field(default_factory=list)
    nota: str | None = None
    partes: list[ManifestPart] = Field(min_length=1)


class Manifest(BaseModel):
    pasta: str
    bulas: list[ManifestItem]


class ManifestError(Exception):
    """O PDF não bate com o manifesto: nada é gravado."""


@dataclass
class Report:
    gravadas: list[Leaflet] = field(default_factory=list)
    erros: dict[str, str] = field(default_factory=dict)  # arquivo -> motivo
    avisos: list[str] = field(default_factory=list)


def sha256(caminho: Path) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def data_da_versao(versao: str | None) -> date | None:
    """"V.01_12/2025" -> 2025-12-01 (o fabricante só imprime mês/ano)."""
    if not versao or not versao.startswith("V."):
        return None
    mes, ano = versao.split("_")[1].split("/")
    return date(int(ano), int(mes), 1)


def bulas_do_item(item: ManifestItem, pasta: Path) -> tuple[list[Leaflet], list[str]]:
    """As bulas de um PDF, já conferidas, e os avisos (seções que não apareceram)."""
    caminho = pasta / item.arquivo
    if not caminho.exists():
        raise ManifestError("arquivo não encontrado")
    if (obtido := sha256(caminho)) != item.sha256:
        raise ManifestError(f"SHA-256 diferente do manifesto ({obtido[:12]}...): não é o arquivo verificado")

    partes = separar_bulas(extrair_paginas(caminho))
    if len(partes) != len(item.partes):
        raise ManifestError(f"o PDF tem {len(partes)} bula(s), o manifesto descreve {len(item.partes)}")

    bulas, avisos = [], []
    for i, (lida, descrita) in enumerate(zip(partes, item.partes), start=1):
        bulas.append(_bula(item, caminho, i, lida, descrita))
        faltam = sorted(ESPERADAS - {s.secao for s in lida.secoes})
        if faltam:
            avisos.append(f"{item.arquivo} [{i}]: seções não encontradas: {', '.join(faltam)}")
    return bulas, avisos


def _bula(item: ManifestItem, caminho: Path, i: int, lida: LeafletPart, descrita: ManifestPart) -> Leaflet:
    onde = f"bula {i}"
    if item.numero_registro not in lida.registros:
        raise ManifestError(f"{onde}: registro {item.numero_registro} não está no texto (achados: {lida.registros or 'nenhum'})")
    if descrita.versao and lida.versao and descrita.versao != lida.versao:
        raise ManifestError(f"{onde}: versão {descrita.versao} no manifesto, {lida.versao} no PDF")
    versao = descrita.versao or lida.versao
    data_versao = descrita.data_versao or data_da_versao(versao)
    if data_versao is None:
        raise ManifestError(f"{onde}: sem data_versao no manifesto e sem código de versão no PDF")
    return montar_bula(
        caminho, descrita.medicamento, item.principio_ativo, item.fabricante, data_versao, item.nomes_comerciais,
        numero_registro=item.numero_registro, forma_farmaceutica=descrita.forma_farmaceutica, versao=versao,
        secoes=lida.secoes,
    )


def processar(manifesto: Manifest, destino: str | Path, so: str | None = None) -> Report:
    r = Report()
    pasta = Path(manifesto.pasta)
    for item in manifesto.bulas:
        if so and item.arquivo != so:
            continue
        try:
            bulas, avisos = bulas_do_item(item, pasta)
        except ManifestError as e:
            r.erros[item.arquivo] = str(e)
            continue
        for bula in bulas:
            gravar_bula(bula, destino)
        r.gravadas += bulas
        r.avisos += avisos

    ids = [b.id for b in r.gravadas]
    repetidos = sorted({i for i in ids if ids.count(i) > 1})
    if repetidos:
        r.erros["(manifesto)"] = f"medicamento + fabricante repetido gera o mesmo id: {', '.join(repetidos)}"
    if not so:
        sobrando = sorted(p.stem for p in Path(destino).glob("*.json") if p.stem not in ids)
        if sobrando:
            r.avisos.append(f"JSON em {destino} que não veio do manifesto (apague se for antigo): {', '.join(sobrando)}")
    return r


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Todas as bulas do manifesto -> JSON do dataset")
    p.add_argument("--manifesto", type=Path, default=MANIFESTO)
    p.add_argument("--so", help="processa só este arquivo do manifesto")
    a = p.parse_args()

    manifesto = Manifest.model_validate(json.loads(a.manifesto.read_text(encoding="utf-8")))
    r = processar(manifesto, get_settings().bulas_dir, a.so)

    for b in r.gravadas:
        print(f"  ok  {b.id:<52} {b.numero_registro}  {b.versao or '':<22} {b.data_versao}  {len(b.secoes):>2} seções")
    for aviso in r.avisos:
        print(f"AVISO {aviso}")
    for arquivo, motivo in r.erros.items():
        print(f"ERRO  {arquivo}: {motivo}")
    print(f"\n{len(r.gravadas)} bula(s) gravada(s), {len(r.erros)} erro(s). Próximo passo: python -m ingestion.generate_chunks")
    if r.erros:
        sys.exit(1)


if __name__ == "__main__":
    main()
