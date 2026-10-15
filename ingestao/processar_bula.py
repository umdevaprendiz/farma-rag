"""Transforma o PDF de uma bula no JSON do dataset (data/processed/bulas/<id>.json).

Para as bulas do acervo, use o lote (python -m ingestao.processar_lote), que lê tudo de
data/bulas_manifesto.json e confere hash, registro e número de bulas por PDF. Este comando é
para uma bula avulsa ou de teste.

Uso (da raiz):
  python -m ingestao.processar_bula data/raw/tylenol.pdf --medicamento "Tylenol 750 mg" \\
      --principio-ativo paracetamol --fabricante Kenvue --data-versao 2025-03-15 \\
      --nome-comercial Tylenol --registro 1.5721.0012 --forma comprimido

Se o PDF tiver mais de uma bula (ex.: gotas + comprimido), escolha uma com --parte N.
Mostra as seções encontradas para conferir com o PDF. Depois rode: python -m ingestao.gerar_chunks
"""

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

from app.config import get_settings
from app.schemas.dataset import Bula, SecaoBula
from app.services.medicamentos import normalizar
from ingestao.extrair_pdf import extrair_paginas
from ingestao.separar_secoes import ESPERADAS, separar_bulas, separar_secoes


def montar_bula(
    arquivo: Path, medicamento: str, principio_ativo: str, fabricante: str, data_versao: date,
    nomes_comerciais: list[str], tipo_bula: str = "profissional", *, numero_registro: str | None = None,
    forma_farmaceutica: str | None = None, versao: str | None = None, secoes: list[SecaoBula] | None = None,
) -> Bula:
    """`secoes` vazio = extrair do arquivo, que precisa ter uma bula só."""
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
        secoes=secoes if secoes is not None else separar_secoes(extrair_paginas(arquivo)),
        numero_registro=numero_registro,
        forma_farmaceutica=forma_farmaceutica,
        versao=versao,
    )


def gravar_bula(bula: Bula, pasta: str | Path) -> Path:
    destino = Path(pasta) / f"{bula.id}.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(bula.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
    return destino


def mostrar_secoes(bula: Bula) -> None:
    for s in bula.secoes:
        print(f"  p.{s.pagina_inicio}-{s.pagina_fim}  {s.secao:<31} {len(s.texto):>6} caracteres  {s.titulo_original}")
    faltando = sorted(ESPERADAS - {s.secao for s in bula.secoes})
    if faltando:
        print(f"\nATENÇÃO: seções não encontradas: {', '.join(faltando)}. Confira os títulos no PDF.")


def main() -> None:
    p = argparse.ArgumentParser(description="PDF (ou .txt) da bula -> JSON do dataset")
    p.add_argument("arquivo", type=Path)
    p.add_argument("--medicamento", required=True, help='nome e dose, ex.: "Tylenol 750 mg"')
    p.add_argument("--principio-ativo", required=True, help="como está na bula, ex.: paracetamol")
    p.add_argument("--fabricante", required=True)
    p.add_argument("--data-versao", required=True, type=date.fromisoformat, help="AAAA-MM-DD")
    p.add_argument("--nome-comercial", action="append", default=[], help="pode repetir")
    p.add_argument("--tipo", default="profissional", choices=["profissional", "paciente"])
    p.add_argument("--registro", help="número de registro ANVISA, ex.: 1.0029.0007 (conferido no texto)")
    p.add_argument("--forma", help='forma farmacêutica, ex.: "comprimido revestido"')
    p.add_argument("--versao", help='versão impressa, ex.: "V.01_12/2025" (vazio = a lida do PDF)')
    p.add_argument("--parte", type=int, help="qual bula do PDF, quando ele tem mais de uma (1, 2...)")
    a = p.parse_args()

    partes = separar_bulas(extrair_paginas(a.arquivo))
    if not partes:
        sys.exit("Nenhum texto extraído: o PDF pode ser imagem (escaneado).")
    if a.parte is None and len(partes) > 1:
        sys.exit(f"O PDF tem {len(partes)} bulas: escolha uma com --parte (1 a {len(partes)}).")
    if a.parte is not None and not 1 <= a.parte <= len(partes):
        sys.exit(f"--parte {a.parte} não existe: o PDF tem {len(partes)} bula(s).")
    parte = partes[(a.parte or 1) - 1]
    if a.registro and a.registro not in parte.registros:
        sys.exit(f"Registro {a.registro} não aparece no texto da bula (achados: {parte.registros or 'nenhum'}).")

    bula = montar_bula(
        a.arquivo, a.medicamento, a.principio_ativo, a.fabricante, a.data_versao, a.nome_comercial, a.tipo,
        numero_registro=a.registro, forma_farmaceutica=a.forma, versao=a.versao or parte.versao,
        secoes=parte.secoes,
    )
    print(f"{gravar_bula(bula, get_settings().bulas_dir)}\n")
    mostrar_secoes(bula)


if __name__ == "__main__":
    main()
