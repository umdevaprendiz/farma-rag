"""Transforma o PDF de uma bula no JSON do dataset (data/processed/bulas/<id>.json).

Uso (da raiz):
  python -m ingestao.processar_bula data/raw/tylenol.pdf --medicamento "Tylenol 750 mg" \\
      --principio-ativo paracetamol --fabricante Kenvue --data-versao 2025-03-15 \\
      --nome-comercial Tylenol

Mostra as seções encontradas para conferir com o PDF. Depois rode: python -m ingestao.gerar_chunks
"""

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

from app.config import get_settings
from app.schemas.dataset import Bula
from app.services.medicamentos import normalizar
from ingestao.extrair_pdf import extrair_paginas
from ingestao.separar_secoes import ESPERADAS, separar_secoes


def montar_bula(
    arquivo: Path, medicamento: str, principio_ativo: str, fabricante: str, data_versao: date,
    nomes_comerciais: list[str], tipo_bula: str = "profissional",
) -> Bula:
    return Bula(
        id=normalizar(f"{medicamento} {fabricante}").replace(" ", "-"),
        medicamento=medicamento,
        principio_ativo=principio_ativo,
        nomes_comerciais=nomes_comerciais,
        fabricante=fabricante,
        tipo_bula=tipo_bula,
        data_versao=data_versao,
        fonte_pdf=arquivo.as_posix(),
        processado_em=datetime.now().astimezone().replace(microsecond=0),
        secoes=separar_secoes(extrair_paginas(arquivo)),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="PDF (ou .txt) da bula -> JSON do dataset")
    p.add_argument("arquivo", type=Path)
    p.add_argument("--medicamento", required=True, help='nome e dose, ex.: "Tylenol 750 mg"')
    p.add_argument("--principio-ativo", required=True, help="como está na bula, ex.: paracetamol")
    p.add_argument("--fabricante", required=True)
    p.add_argument("--data-versao", required=True, type=date.fromisoformat, help="AAAA-MM-DD")
    p.add_argument("--nome-comercial", action="append", default=[], help="pode repetir")
    p.add_argument("--tipo", default="profissional", choices=["profissional", "paciente"])
    a = p.parse_args()

    bula = montar_bula(
        a.arquivo, a.medicamento, a.principio_ativo, a.fabricante, a.data_versao, a.nome_comercial, a.tipo
    )
    if not bula.secoes:
        sys.exit("Nenhum texto extraído: o PDF pode ser imagem (escaneado).")
    destino = Path(get_settings().bulas_dir) / f"{bula.id}.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(bula.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"{destino}\n")
    for s in bula.secoes:
        print(f"  p.{s.pagina_inicio}-{s.pagina_fim}  {s.secao:<31} {len(s.texto):>6} caracteres  {s.titulo_original}")
    faltando = sorted(ESPERADAS - {s.secao for s in bula.secoes})
    if faltando:
        print(f"\nATENÇÃO: seções não encontradas: {', '.join(faltando)}. Confira os títulos no PDF.")


if __name__ == "__main__":
    main()
