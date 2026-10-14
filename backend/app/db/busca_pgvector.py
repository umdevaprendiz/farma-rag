"""Busca por similaridade de cosseno no PostgreSQL (pgvector), com a interface `BuscaVetorial`.

Mesmo resultado do `IndiceEmMemoria`, inclusive a divisão do k entre princípios ativos.
Sem índice HNSW de propósito: com 30-50 bulas (poucos milhares de trechos) a varredura exata
leva milissegundos e devolve o top-k exato. O HNSW é aproximado e, com o filtro por princípio
ativo no WHERE, pode devolver menos de k trechos. Reavaliar se o acervo passar de ~100 mil trechos.
"""

import math
from typing import Collection, Sequence

import numpy as np
from sqlalchemy import Engine, select

from app.db.busca_vetorial import Resultado
from app.db.models import Chunk
from app.services.medicamentos import normalizar


class BuscaPgvector:
    def __init__(self, engine: Engine):
        self._engine = engine

    def _top(self, conexao, q: np.ndarray, k: int, alvo: str | None) -> list[Resultado]:
        distancia = Chunk.embedding.cosine_distance(q)  # 1 - cosseno
        consulta = select(Chunk.chunk_id, distancia).order_by(distancia, Chunk.chunk_id).limit(k)
        if alvo is not None:
            consulta = consulta.where(Chunk.principio_ativo_norm == alvo)
        return [Resultado(chunk_id, 1.0 - float(d)) for chunk_id, d in conexao.execute(consulta)]

    def buscar(
        self, consulta: Sequence[float], k: int, principios_ativos: Collection[str] | None = None
    ) -> list[Resultado]:
        if k <= 0:
            return []
        q = np.asarray(consulta, dtype="<f4")
        if np.linalg.norm(q) == 0:  # o cosseno com vetor nulo é NaN no pgvector
            return []

        with self._engine.connect() as conexao:
            if not principios_ativos:
                return self._top(conexao, q, k, None)

            alvos = sorted({normalizar(p) for p in principios_ativos})
            k_por_alvo = max(1, math.ceil(k / len(alvos)))
            resultados: list[Resultado] = []
            for alvo in alvos:
                resultados += self._top(conexao, q, k_por_alvo, alvo)
        return sorted(resultados, key=lambda r: -r.score)[: max(k, len(alvos))]
