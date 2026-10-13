"""O acervo na memória: bulas, trechos, índice vetorial e catálogo de nomes.

Lê o dataset em arquivo (data/processed) e o cache de embeddings. Quando a API usar o MySQL,
só a origem muda: o serviço de RAG continua recebendo um Acervo.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import Settings, get_settings
from app.db.busca_vetorial import BuscaVetorial, IndiceEmMemoria
from app.llm.embeddings import CacheEmbeddings
from app.schemas.dataset import ROTULO_SECAO, Bula, Chunk
from app.services.medicamentos import Catalogo, carregar_catalogo_anvisa, carregar_conhecidos


class AcervoIncompletoError(Exception):
    """Falta rodar a ingestão (ou parte dela)."""


def texto_para_embedding(chunk: Chunk) -> str:
    """O trecho com o medicamento e a seção na frente: "Não use na gravidez." sozinho não diz de quem é."""
    return f"{chunk.medicamento} ({chunk.principio_ativo}) | {ROTULO_SECAO[chunk.secao]}\n{chunk.texto}"


def carregar_bulas(pasta: str | Path) -> list[Bula]:
    return [Bula.model_validate_json(p.read_text(encoding="utf-8")) for p in sorted(Path(pasta).glob("*.json"))]


def carregar_chunks(arquivo: str | Path) -> list[Chunk]:
    p = Path(arquivo)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [Chunk.model_validate_json(l) for l in f if l.strip()]


@dataclass
class Acervo:
    bulas: dict[str, Bula]
    chunks: dict[str, Chunk]
    indice: BuscaVetorial
    catalogo: Catalogo

    @classmethod
    def carregar(cls, s: Settings | None = None) -> "Acervo":
        s = s or get_settings()
        bulas = carregar_bulas(s.bulas_dir)
        chunks = carregar_chunks(s.chunks_arquivo)
        if not bulas or not chunks:
            raise AcervoIncompletoError(
                f"Sem bulas em {s.bulas_dir} ou sem trechos em {s.chunks_arquivo}. "
                "Rode: python -m ingestao.processar_bula ... e depois python -m ingestao.gerar_chunks"
            )
        cache = CacheEmbeddings(s.cache_embeddings)
        vetores = cache.buscar(s.embed_modelo, [texto_para_embedding(c) for c in chunks])
        cache.fechar()
        faltam = sum(v is None for v in vetores)
        if faltam:
            raise AcervoIncompletoError(
                f"{faltam} de {len(chunks)} trechos sem embedding ({s.embed_modelo}). "
                "Rode: python -m ingestao.gerar_chunks"
            )
        return cls(
            bulas={b.id: b for b in bulas},
            chunks={c.chunk_id: c for c in chunks},
            indice=IndiceEmMemoria(
                [c.chunk_id for c in chunks], [c.principio_ativo for c in chunks], np.stack(vetores)
            ),
            catalogo=Catalogo.montar(
                bulas,
                carregar_conhecidos(s.medicamentos_conhecidos),
                carregar_catalogo_anvisa(s.catalogo_anvisa),
                carregar_conhecidos(s.aliases_ignorados),
            ),
        )
