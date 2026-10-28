"""Embeddings com cache em disco, por hash do texto.

Na CPU cada trecho leva ~3 s (docs/DECISOES.md, seção 1), então indexar tudo são horas.
O cache grava a cada lote: dá para interromper e continuar, e reindexar só refaz o que mudou.
Cada linha guarda só o SHA-256 de (modelo, texto) e o vetor, nunca o texto. O arquivo do acervo vai no
Git (§39): um clone novo carrega o banco sem recalcular nada.
"""

import hashlib
import sqlite3
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from app.config import Settings, get_settings
from app.db.vector_search import bytes_para_vetor, vetor_para_bytes
from app.llm import ollama

LOTE = 8


def _chave(modelo: str, texto: str) -> str:
    return hashlib.sha256(f"{modelo}\n{texto}".encode()).hexdigest()


class EmbeddingCache:
    def __init__(self, caminho: str | Path, somente_leitura: bool = False):
        if somente_leitura:
            # Para quem só lê (carga do banco, acervo em memória): nunca cria nem altera o arquivo, que vai
            # no Git e, no container da carga, está montado só leitura. Arquivo ausente: sqlite3.OperationalError.
            self._db = sqlite3.connect(f"{Path(caminho).absolute().as_uri()}?mode=ro", uri=True)
            return
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

    def _fora_de(self, modelo: str, textos: Sequence[str]) -> list[str]:
        manter = {_chave(modelo, t) for t in textos}
        return [k for (k,) in self._db.execute("SELECT chave FROM embeddings").fetchall() if k not in manter]

    def sobras(self, modelo: str, textos: Sequence[str]) -> int:
        """Quantas linhas não são embedding de `textos` com `modelo` (trecho antigo, outro modelo)."""
        return len(self._fora_de(modelo, textos))

    def podar(self, modelo: str, textos: Sequence[str]) -> int:
        """Deixa só os embeddings de `textos` com `modelo` e compacta o arquivo. Devolve quantas saíram.

        O VACUUM reescreve o arquivo: nada das linhas apagadas fica nas páginas livres (o arquivo vai no Git).
        """
        fora = self._fora_de(modelo, textos)
        self._db.executemany("DELETE FROM embeddings WHERE chave = ?", [(k,) for k in fora])
        self._db.commit()
        self._db.execute("VACUUM")
        return len(fora)

    def fechar(self) -> None:
        self._db.close()


def embutir(
    textos: Sequence[str],
    cache: EmbeddingCache,
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
