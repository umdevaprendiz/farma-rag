"""O acervo na memória: bulas, trechos, índice vetorial e catálogo de nomes.

Duas origens, o mesmo Corpus para o serviço de RAG:
- `Corpus.carregar`: dataset em arquivo (data/processed) + cache de embeddings, busca em numpy
  (teste manual pelo terminal, benchmark);
- `Corpus.do_banco`: tabelas `bulas` e `chunks` do PostgreSQL, busca no próprio banco
  (PgvectorSearch). É o da API.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

import numpy as np
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import models
from app.db.pgvector_search import PgvectorSearch
from app.db.vector_search import InMemoryIndex, VectorSearch
from app.llm.embeddings import EmbeddingCache
from app.schemas.dataset import ROTULO_SECAO, Chunk, Leaflet
from app.services.medicines import Catalog, carregar_catalogo_anvisa, carregar_conhecidos, normalizar


class IncompleteCorpusError(Exception):
    """Falta rodar a ingestão (ou parte dela)."""


class CorpusLeaflet(Protocol):
    """O que o agente e a API leem de uma bula: os dados dela, não o texto (que está nos trechos).

    Servem `schemas.dataset.Leaflet` (do arquivo) e `db.models.Leaflet` (do banco, que não guarda as seções).
    """

    id: str
    medicamento: str
    principio_ativo: str
    nomes_comerciais: list[str]
    fabricante: str
    data_versao: date
    numero_registro: str | None
    forma_farmaceutica: str | None


def texto_para_embedding(chunk: Chunk) -> str:
    """O trecho com o medicamento e a seção na frente: "Não use na gravidez." sozinho não diz de quem é."""
    return f"{chunk.medicamento} ({chunk.principio_ativo}) | {ROTULO_SECAO[chunk.secao]}\n{chunk.texto}"


def carregar_bulas(pasta: str | Path) -> list[Leaflet]:
    return [Leaflet.model_validate_json(p.read_text(encoding="utf-8")) for p in sorted(Path(pasta).glob("*.json"))]


def carregar_chunks(arquivo: str | Path) -> list[Chunk]:
    p = Path(arquivo)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [Chunk.model_validate_json(l) for l in f if l.strip()]


def carregar_dataset(s: Settings) -> tuple[list[Leaflet], list[Chunk], np.ndarray]:
    """Bulas, trechos e a matriz de embeddings (na ordem dos trechos), conferindo que nada falta."""
    bulas = carregar_bulas(s.bulas_dir)
    chunks = carregar_chunks(s.chunks_arquivo)
    if not bulas or not chunks:
        raise IncompleteCorpusError(
            f"Sem bulas em {s.bulas_dir} ou sem trechos em {s.chunks_arquivo}. "
            "Rode: python -m ingestion.process_leaflet ... e depois python -m ingestion.generate_chunks"
        )
    if not Path(s.cache_embeddings).is_file():
        raise IncompleteCorpusError(
            f"Sem o cache de embeddings em {s.cache_embeddings} (ele vem no Git, docs/DECISOES.md §39). "
            "Rode: python -m ingestion.generate_chunks"
        )
    cache = EmbeddingCache(s.cache_embeddings, somente_leitura=True)  # só lê: o arquivo vai no Git
    vetores = cache.buscar(s.embed_modelo, [texto_para_embedding(c) for c in chunks])
    cache.fechar()
    faltam = sum(v is None for v in vetores)
    if faltam:
        raise IncompleteCorpusError(
            f"{faltam} de {len(chunks)} trechos sem embedding ({s.embed_modelo}). "
            "Rode: python -m ingestion.generate_chunks"
        )
    return bulas, chunks, np.stack(vetores)


def _catalogo(bulas: list[CorpusLeaflet], chunks: list[Chunk], s: Settings) -> Catalog:
    return Catalog.montar(
        bulas,
        carregar_conhecidos(s.medicamentos_conhecidos),
        carregar_catalogo_anvisa(s.catalogo_anvisa),
        carregar_conhecidos(s.aliases_ignorados),
        vocabulario={palavra for c in chunks for palavra in normalizar(c.texto).split(" ")},
    )


@dataclass
class Corpus:
    bulas: dict[str, CorpusLeaflet]
    chunks: dict[str, Chunk]
    indice: VectorSearch
    catalogo: Catalog

    @classmethod
    def carregar(cls, s: Settings | None = None) -> "Corpus":
        s = s or get_settings()
        bulas, chunks, matriz = carregar_dataset(s)
        return cls(
            bulas={b.id: b for b in bulas},
            chunks={c.chunk_id: c for c in chunks},
            indice=InMemoryIndex(
                [c.chunk_id for c in chunks], [c.principio_ativo for c in chunks], matriz
            ),
            catalogo=_catalogo(bulas, chunks, s),
        )

    @classmethod
    def do_banco(cls, engine: Engine, s: Settings | None = None) -> "Corpus":
        """Bulas e trechos gravados por ingestion/load_database.py; os embeddings ficam no banco.

        Lido uma vez (a API faz isso ao subir): depois de rodar carregar_banco de novo, reinicie a API.
        """
        s = s or get_settings()
        colunas = [getattr(models.Chunk, nome) for nome in Chunk.model_fields]  # tudo, menos o embedding
        with Session(engine) as sessao:
            bulas = list(sessao.scalars(select(models.Leaflet).order_by(models.Leaflet.id)))
            linhas = sessao.execute(select(*colunas).order_by(models.Chunk.chunk_id)).all()
        if not bulas or not linhas:
            raise IncompleteCorpusError("O banco não tem bulas ou trechos. Rode: python -m ingestion.load_database")
        if any(linha.medicamento is None for linha in linhas):
            raise IncompleteCorpusError(
                "Os trechos do banco foram gravados antes das colunas medicamento e n_tokens. "
                "Rode de novo: python -m ingestion.load_database"
            )
        chunks = [Chunk.model_validate(dict(linha._mapping)) for linha in linhas]
        return cls(
            bulas={b.id: b for b in bulas},
            chunks={c.chunk_id: c for c in chunks},
            indice=PgvectorSearch(engine),
            catalogo=_catalogo(bulas, chunks, s),
        )
