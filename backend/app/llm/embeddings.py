"""Embeddings com cache em disco, por hash do texto.

Na CPU cada trecho leva ~3 s (docs/DECISOES.md, seção 1), então indexar tudo são horas.
O cache grava a cada lote: dá para interromper e continuar, e reindexar só refaz o que mudou.
"""

import hashlib
import sqlite3
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from app.config import Settings, get_settings
from app.db.busca_vetorial import bytes_para_vetor, vetor_para_bytes
from app.llm import ollama

LOTE = 8


def _chave(modelo: str, texto: str) -> str:
    return hashlib.sha256(f"{modelo}\n{texto}".encode()).hexdigest()


class CacheEmbeddings:
    def __init__(self, caminho: str | Path):
        Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(caminho)
        self._db.execute("CREATE TABLE IF NOT EXISTS embeddings (chave TEXT PRIMARY KEY, vetor BLOB NOT NULL)")

    def buscar(self, modelo: str, textos: Sequence[str]) -> list[np.ndarray | None]:
        achados = []
        for texto in textos:
            linha = self._db.execute(
                "SELECT vetor FROM embeddings WHERE chave = ?", (_chave(modelo, texto),)
            ).fetchone()
            achados.append(bytes_para_vetor(linha[0]) if linha else None)
        return achados

    def gravar(self, modelo: str, textos: Sequence[str], vetores: Sequence[Sequence[float]]) -> None:
        self._db.executemany(
            "INSERT OR REPLACE INTO embeddings VALUES (?, ?)",
            [(_chave(modelo, t), vetor_para_bytes(v)) for t, v in zip(textos, vetores)],
        )
        self._db.commit()

    def fechar(self) -> None:
        self._db.close()


def embutir(
    textos: Sequence[str],
    cache: CacheEmbeddings,
    s: Settings | None = None,
    ao_avancar: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Matriz (n, EMBED_DIM) na ordem de `textos`; só chama o Ollama para o que falta no cache.

    `ao_avancar(feitos, faltavam)` é chamado depois de cada lote calculado.
    """
    s = s or get_settings()
    vetores = cache.buscar(s.embed_modelo, textos)
    faltam = [i for i, v in enumerate(vetores) if v is None]
    for inicio in range(0, len(faltam), LOTE):
        indices = faltam[inicio : inicio + LOTE]
        lote = [textos[i] for i in indices]
        novos = ollama.embed(lote, s)
        cache.gravar(s.embed_modelo, lote, novos)
        for i, v in zip(indices, novos):
            vetores[i] = np.asarray(v, dtype="<f4")
        if ao_avancar:
            ao_avancar(inicio + len(indices), len(faltam))
    return np.stack(vetores) if vetores else np.empty((0, s.embed_dim), dtype="<f4")
