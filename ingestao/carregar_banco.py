"""Copia o dataset em arquivo para o PostgreSQL: bulas, trechos e embeddings.

Uso (da raiz):  python -m ingestao.carregar_banco
Entrada: data/processed/bulas/*.json, data/processed/chunks.jsonl e o cache de embeddings
         (rode antes processar_bula e gerar_chunks)
Destino: o banco de DB_HOST/DB_PORT/DB_NAME do .env (docker compose up -d db)

O arquivo é a fonte da verdade (docs/DECISOES.md, seção 15): `bulas` e `chunks` são apagadas e
recriadas numa transação só, então quem consulta durante a carga vê o acervo antigo inteiro ou o
novo inteiro. O histórico (`consultas`, `consulta_fontes`) não é tocado.
"""

import sys

import numpy as np
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import models
from app.schemas.dataset import Bula, Chunk
from app.services.acervo import AcervoIncompletoError, carregar_dataset
from app.services.medicamentos import normalizar


# Colunas que entraram depois da primeira versão de `bulas`: create_all não altera tabela que já
# existe, então quem criou o banco antes recebe as colunas aqui.
_COLUNAS_NOVAS = ("numero_registro", "forma_farmaceutica", "versao")


def preparar_banco(engine: Engine) -> None:
    """Cria a extensão vector, as tabelas que faltarem e as colunas novas de `bulas`."""
    with engine.begin() as conexao:
        conexao.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    models.Base.metadata.create_all(engine)
    with engine.begin() as conexao:
        for coluna in _COLUNAS_NOVAS:
            conexao.execute(text(f"ALTER TABLE bulas ADD COLUMN IF NOT EXISTS {coluna} text"))


def gravar(engine: Engine, bulas: list[Bula], chunks: list[Chunk], matriz: np.ndarray) -> None:
    if matriz.shape != (len(chunks), models.EMBED_DIM):
        raise ValueError(f"matriz {matriz.shape}, esperado ({len(chunks)}, {models.EMBED_DIM})")
    with Session(engine) as sessao, sessao.begin():
        sessao.execute(text("TRUNCATE chunks, bulas"))
        sessao.add_all(
            models.Bula(
                id=b.id,
                medicamento=b.medicamento,
                principio_ativo=b.principio_ativo,
                nomes_comerciais=b.nomes_comerciais,
                fabricante=b.fabricante,
                data_versao=b.data_versao,
                fonte_pdf=b.fonte_pdf,
                numero_registro=b.numero_registro,
                forma_farmaceutica=b.forma_farmaceutica,
                versao=b.versao,
            )
            for b in bulas
        )
        sessao.flush()  # bulas antes de chunks, por causa da FK
        sessao.add_all(
            models.Chunk(
                chunk_id=c.chunk_id,
                bula_id=c.bula_id,
                principio_ativo=c.principio_ativo,
                principio_ativo_norm=normalizar(c.principio_ativo),
                secao=c.secao,
                pagina=c.pagina,
                ordem=c.ordem,
                texto=c.texto,
                embedding=vetor,
            )
            for c, vetor in zip(chunks, matriz)
        )


def main(s: Settings | None = None) -> None:
    s = s or get_settings()
    if s.embed_dim != models.EMBED_DIM:
        sys.exit(
            f"EMBED_DIM={s.embed_dim} no .env, mas a coluna chunks.embedding é vector({models.EMBED_DIM}). "
            "Ajuste EMBED_DIM em app/db/models.py e recrie a tabela chunks."
        )
    try:
        bulas, chunks, matriz = carregar_dataset(s)
    except AcervoIncompletoError as e:
        sys.exit(str(e))
    engine = create_engine(s.database_url)
    preparar_banco(engine)
    gravar(engine, bulas, chunks, matriz)
    engine.dispose()
    print(f"{len(bulas)} bula(s) e {len(chunks)} trechos gravados em {s.db_host}:{s.db_port}/{s.db_name}")


if __name__ == "__main__":
    main()
