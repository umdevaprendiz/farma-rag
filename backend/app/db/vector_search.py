"""Busca por similaridade de cosseno em memória (numpy).

Usada no fluxo por arquivos (benchmark.ask e testes), sem banco. Com 30-50 bulas são
poucos milhares de trechos (~4 KB cada embedding), então a força bruta leva milissegundos.

`VectorSearch` é a interface que o serviço de RAG usa. A versão no PostgreSQL é a
`PgvectorSearch` (pgvector_search.py), com a mesma assinatura e o mesmo resultado.

`bulas` (apresentação escolhida, docs/DECISOES.md §41): o princípio ativo que tem alguma dessas bulas
só busca nelas; os outros princípios da pergunta continuam em todas as bulas deles.
"""

import math
from dataclasses import dataclass
from typing import Collection, Iterable, Protocol, Sequence

import numpy as np

from app.services.medicines import normalizar


def vetor_para_bytes(vetor: Sequence[float]) -> bytes:
    return np.asarray(vetor, dtype="<f4").tobytes()


def bytes_para_vetor(dados: bytes) -> np.ndarray:
    return np.frombuffer(dados, dtype="<f4")


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    score: float  # similaridade de cosseno, de -1 a 1


class VectorSearch(Protocol):
    def buscar(
        self,
        consulta: Sequence[float],
        k: int,
        principios_ativos: Collection[str] | None = None,
        bulas: Collection[str] | None = None,
    ) -> list[SearchHit]: ...


def bula_do_chunk(chunk_id: str) -> str:
    """O chunk_id é <bula_id>__<secao>__<ordem> (contrato do dataset)."""
    return chunk_id.split("__", 1)[0]


class InMemoryIndex:
    def __init__(self, chunk_ids: list[str], principios_ativos: list[str], matriz: np.ndarray):
        if not (len(chunk_ids) == len(principios_ativos) == len(matriz)):
            raise ValueError("chunk_ids, principios_ativos e matriz precisam ter o mesmo tamanho")
        self._ids = chunk_ids
        self._ativos = np.array([normalizar(p) for p in principios_ativos])
        self._bulas = np.array([bula_do_chunk(c) for c in chunk_ids])
        normas = np.linalg.norm(matriz, axis=1, keepdims=True)
        self._matriz = matriz / np.where(normas == 0, 1, normas)

    @classmethod
    def de_linhas(cls, linhas: Iterable[tuple[str, str, bytes]]) -> "InMemoryIndex":
        """Linhas (chunk_id, principio_ativo, embedding_bytes), como vêm da tabela chunks."""
        linhas = list(linhas)
        matriz = (
            np.stack([bytes_para_vetor(b) for _, _, b in linhas])
            if linhas
            else np.empty((0, 0), dtype="<f4")
        )
        return cls([c for c, _, _ in linhas], [p for _, p, _ in linhas], matriz)

    def __len__(self) -> int:
        return len(self._ids)

    def _top(self, q: np.ndarray, k: int, mascara: np.ndarray | None) -> list[SearchHit]:
        scores = self._matriz @ q
        indices = np.flatnonzero(mascara) if mascara is not None else np.arange(len(scores))
        if indices.size == 0:
            return []
        melhores = indices[np.argsort(-scores[indices])[:k]]
        return [SearchHit(self._ids[i], float(scores[i])) for i in melhores]

    def buscar(
        self,
        consulta: Sequence[float],
        k: int,
        principios_ativos: Collection[str] | None = None,
        bulas: Collection[str] | None = None,
    ) -> list[SearchHit]:
        """Top-k por cosseno. Com vários princípios ativos, divide o k entre eles para que
        uma pergunta de interação receba trechos das duas bulas, não só da mais parecida.
        `bulas`: só vale com princípios ativos (ver o começo do módulo)."""
        if len(self) == 0 or k <= 0:
            return []
        q = np.asarray(consulta, dtype="<f4")
        norma = np.linalg.norm(q)
        if norma == 0:
            return []
        q = q / norma

        if not principios_ativos:
            return self._top(q, k, None)

        alvos = sorted({normalizar(p) for p in principios_ativos})
        k_por_alvo = max(1, math.ceil(k / len(alvos)))
        escolhidas = np.isin(self._bulas, list(bulas or []))
        resultados: list[SearchHit] = []
        for alvo in alvos:
            mascara = self._ativos == alvo
            if (mascara & escolhidas).any():
                mascara &= escolhidas
            resultados += self._top(q, k_por_alvo, mascara)
        return sorted(resultados, key=lambda r: -r.score)[: max(k, len(alvos))]
