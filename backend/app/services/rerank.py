"""Reranking da rodada 1: dos candidatos que a busca vetorial trouxe, os top_k que vão ao agente
(docs/DECISOES.md, seção 30). Sem modelo e sem download: o i5 sem GPU já gasta 30-45 s no LLM, e a
semelhança sozinha errava a seção ("dose de ibuprofeno" trazia "resultados de eficácia").

Candidatos: os `rerank_candidatos` mais parecidos de cada remédio da pergunta, mais os
EXTRA_POR_SECAO mais parecidos de cada seção que a pergunta pede (intent.secoes_para_rerank), mesmo
que estejam mais longe: "zolpidem tem lactose?" não trazia a composição entre os 10.

Nota de cada candidato = semelhança (cosseno)
                        + PESO_SECAO se é da seção que a pergunta pede
                        + PESO_ASSUNTO × fração dos assuntos da pergunta que o trecho cita.
Escolha gulosa pela nota, com a mesma divisão do top_k entre os remédios que a busca já fazia, e
sem trecho quase igual a um já escolhido (as três bulas do Tylalgin repetem frases).

Os pesos estão na escala da semelhança do bge-m3, em que os candidatos de uma mesma bula ficam a
poucos centésimos um do outro. Medidos no benchmark de 80 perguntas (data/benchmark/).
"""

import math
import re
from collections import defaultdict
from collections.abc import Sequence

from app.db.vector_search import SearchHit
from app.schemas.dataset import Chunk
from app.services.medicines import normalizar

PESO_SECAO = 0.08
PESO_ASSUNTO = 0.08
EXTRA_POR_SECAO = 2
# Jaccard das palavras a partir do qual dois trechos contam como o mesmo texto.
QUASE_IGUAL = 0.8


def _palavras(texto: str) -> set[str]:
    return set(normalizar(texto).split())


def cobertura(texto_normalizado: str, grupos: Sequence[tuple[str, ...]]) -> float:
    """Fração dos assuntos da pergunta que o trecho cita (cada grupo = uma palavra e sinônimos)."""
    if not grupos:
        return 0.0
    achados = sum(any(re.search(r"\b" + re.escape(t), texto_normalizado) for t in g) for g in grupos)
    return achados / len(grupos)


def nota(hit: SearchHit, chunk: Chunk, secoes: Sequence[str], grupos: Sequence[tuple[str, ...]]) -> float:
    return (
        hit.score
        + (PESO_SECAO if chunk.secao in secoes else 0.0)
        + PESO_ASSUNTO * cobertura(normalizar(chunk.texto), grupos)
    )


def reordenar(
    candidatos: Sequence[SearchHit],
    chunks: dict[str, Chunk],
    secoes: Sequence[str],
    grupos: Sequence[tuple[str, ...]],
    k: int,
    ativos: Sequence[str],
) -> list[SearchHit]:
    """Os k melhores candidatos pela nota, no máximo ceil(k / nº de remédios) de cada remédio e sem
    trecho quase igual a outro já escolhido. Devolve os SearchHit com a semelhança original (é a
    nota que a tela mostra), na ordem da nota."""
    unicos = list({h.chunk_id: h for h in candidatos}.values())
    ordenados = sorted(unicos, key=lambda h: (-nota(h, chunks[h.chunk_id], secoes, grupos), h.chunk_id))
    por_ativo = max(1, math.ceil(k / max(1, len({normalizar(a) for a in ativos}))))
    contagem: dict[str, int] = defaultdict(int)
    escolhidos: list[SearchHit] = []
    palavras_escolhidas: list[set[str]] = []
    for h in ordenados:
        chunk = chunks[h.chunk_id]
        ativo = normalizar(chunk.principio_ativo)
        if contagem[ativo] >= por_ativo:
            continue
        palavras = _palavras(chunk.texto)
        if any(len(palavras & p) / max(1, len(palavras | p)) >= QUASE_IGUAL for p in palavras_escolhidas):
            continue
        escolhidos.append(h)
        palavras_escolhidas.append(palavras)
        contagem[ativo] += 1
        if len(escolhidos) == k:
            break
    return escolhidos
