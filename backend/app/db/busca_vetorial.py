"""Busca por similaridade de cosseno em memória (numpy).

MySQL Community guarda vetores mas não calcula distância (o DISTANCE() é do HeatWave).
Com 30-50 bulas são poucos milhares de trechos (~4 KB cada embedding), então carregar
tudo na memória da API no startup e fazer força bruta leva milissegundos.

`BuscaVetorial` é a interface que o serviço de RAG usa. Quando o PostgreSQL estiver
disponível, basta uma implementação com pgvector (`ORDER BY embedding <=> :q`) com a
mesma assinatura; o resto do código não muda.
"""

import math
from dataclasses import dataclass
from typing import Collection, Iterable, Protocol, Sequence

import numpy as np

from app.services.medicamentos import normalizar


def vetor_para_bytes(vetor: Sequence[float]) -> bytes:
    return np.asarray(vetor, dtype="<f4").tobytes()


def bytes_para_vetor(dados: bytes) -> np.ndarray:
    return np.frombuffer(dados, dtype="<f4")


@dataclass(frozen=True)
class Resultado:
    chunk_id: str
    score: float  # similaridade de cosseno, de -1 a 1


class BuscaVetorial(Protocol):
    def buscar(
        self, consulta: Sequence[float], k: int, principios_ativos: Collection[str] | None = None
    ) -> list[Resultado]: ...


class IndiceEmMemoria:
    def __init__(self, chunk_ids: list[str], principios_ativos: list[str], matriz: np.ndarray):
        if not (len(chunk_ids) == len(principios_ativos) == len(matriz)):
            raise ValueError("chunk_ids, principios_ativos e matriz precisam ter o mesmo tamanho")
        self._ids = chunk_ids
        self._ativos = np.array([normalizar(p) for p in principios_ativos])
        normas = np.linalg.norm(matriz, axis=1, keepdims=True)
        self._matriz = matriz / np.where(normas == 0, 1, normas)

    @classmethod
    def de_linhas(cls, linhas: Iterable[tuple[str, str, bytes]]) -> "IndiceEmMemoria":
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

    def _top(self, q: np.ndarray, k: int, mascara: np.ndarray | None) -> list[Resultado]:
        scores = self._matriz @ q
        indices = np.flatnonzero(mascara) if mascara is not None else np.arange(len(scores))
        if indices.size == 0:
            return []
        melhores = indices[np.argsort(-scores[indices])[:k]]
        return [Resultado(self._ids[i], float(scores[i])) for i in melhores]

    def buscar(
        self, consulta: Sequence[float], k: int, principios_ativos: Collection[str] | None = None
    ) -> list[Resultado]:
        """Top-k por cosseno. Com vários princípios ativos, divide o k entre eles para que
        uma pergunta de interação receba trechos das duas bulas, não só da mais parecida."""
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
        resultados: list[Resultado] = []
        for alvo in alvos:
            resultados += self._top(q, k_por_alvo, self._ativos == alvo)
        return sorted(resultados, key=lambda r: -r.score)[: max(k, len(alvos))]
