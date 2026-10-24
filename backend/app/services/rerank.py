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

Seção da pergunta já na rodada 1 (F3b, docs/DECISOES.md, seção 34): o bônus de seção é pequeno, e um
trecho da seção pedida muito menos parecido que os outros ficava de fora (caso 12f61084: dois trechos
de "resultados de eficácia" e um de indicações, quando a pergunta era de indicação). Duas regras:
- SECOES_RARAS ("resultados de eficácia", "características farmacológicas"): só estudos clínicos e
  farmacologia, quase nunca a resposta no balcão e trechos longos que pesam no prompt. Só ficam na
  frente se a pergunta as pede; senão vão para o fim da fila e só entram se faltar candidato.
- Garantia: se a pergunta indica seção e nenhum trecho escolhido de um remédio é de uma delas, o
  trecho da seção mais bem colocado desse remédio entra no lugar do escolhido de menor semelhança.
`garantir_secao=False` desliga as duas (a linha de base do benchmark).
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
# F3b: seções que só entram na frente se a pergunta as pede (estudos, mecanismo de ação).
SECOES_RARAS = frozenset({"resultados_eficacia", "caracteristicas_farmacologicas"})
# Maior que qualquer bônus e que a distância entre dois candidatos: a seção rara passa para o fim.
PENALIDADE_SECAO_RARA = 0.5


def _palavras(texto: str) -> set[str]:
    return set(normalizar(texto).split())


def cobertura(texto_normalizado: str, grupos: Sequence[tuple[str, ...]]) -> float:
    """Fração dos assuntos da pergunta que o trecho cita (cada grupo = uma palavra e sinônimos)."""
    if not grupos:
        return 0.0
    achados = sum(any(re.search(r"\b" + re.escape(t), texto_normalizado) for t in g) for g in grupos)
    return achados / len(grupos)


def nota(
    hit: SearchHit, chunk: Chunk, secoes: Sequence[str], grupos: Sequence[tuple[str, ...]], penalizar_raras: bool = False
) -> float:
    """`penalizar_raras`: a seção rara que a pergunta não pediu perde PENALIDADE_SECAO_RARA (F3b)."""
    rara = penalizar_raras and chunk.secao in SECOES_RARAS and chunk.secao not in secoes
    return (
        hit.score
        + (PESO_SECAO if chunk.secao in secoes else 0.0)
        + PESO_ASSUNTO * cobertura(normalizar(chunk.texto), grupos)
        - (PENALIDADE_SECAO_RARA if rara else 0.0)
    )


def _repete(palavras: set[str], escolhidas: Sequence[set[str]]) -> bool:
    """O trecho é quase igual a algum já escolhido?"""
    return any(len(palavras & p) / max(1, len(palavras | p)) >= QUASE_IGUAL for p in escolhidas)


def reordenar(
    candidatos: Sequence[SearchHit],
    chunks: dict[str, Chunk],
    secoes: Sequence[str],
    grupos: Sequence[tuple[str, ...]],
    k: int,
    ativos: Sequence[str],
    garantir_secao: bool = True,
) -> list[SearchHit]:
    """Os k melhores candidatos pela nota, no máximo ceil(k / nº de remédios) de cada remédio e sem
    trecho quase igual a outro já escolhido. Devolve os SearchHit com a semelhança original (é a
    nota que a tela mostra), na ordem da nota.

    `garantir_secao` (F3b): seções raras só na frente se pedidas, e pelo menos um trecho da seção
    pedida por remédio, quando ele tem (ver o docstring do módulo)."""

    def ordem(h: SearchHit) -> tuple[float, str]:
        return -nota(h, chunks[h.chunk_id], secoes, grupos, garantir_secao), h.chunk_id

    unicos = list({h.chunk_id: h for h in candidatos}.values())
    ordenados = sorted(unicos, key=ordem)
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
        if _repete(palavras, palavras_escolhidas):
            continue
        escolhidos.append(h)
        palavras_escolhidas.append(palavras)
        contagem[ativo] += 1
        if len(escolhidos) == k:
            break
    if garantir_secao and secoes:
        escolhidos = _garantir_secao(escolhidos, ordenados, chunks, secoes, ativos)
        escolhidos.sort(key=ordem)
    return escolhidos


def _garantir_secao(
    escolhidos: list[SearchHit],
    ordenados: Sequence[SearchHit],
    chunks: dict[str, Chunk],
    secoes: Sequence[str],
    ativos: Sequence[str],
) -> list[SearchHit]:
    """Para cada remédio da pergunta sem trecho escolhido da seção pedida: o trecho dessa seção
    com melhor nota (`ordenados` já vem pela nota) entra no lugar do escolhido de menor semelhança
    do remédio, se não for quase igual a um que fica. Remédio sem trecho da seção entre os
    candidatos, ou sem nenhum escolhido, fica como está."""
    resultado = list(escolhidos)
    for alvo in sorted({normalizar(a) for a in ativos}):
        do_alvo = [h for h in resultado if normalizar(chunks[h.chunk_id].principio_ativo) == alvo]
        if not do_alvo or any(chunks[h.chunk_id].secao in secoes for h in do_alvo):
            continue
        sai = min(do_alvo, key=lambda h: (h.score, h.chunk_id))
        ficam = [h for h in resultado if h is not sai]
        palavras_que_ficam = [_palavras(chunks[h.chunk_id].texto) for h in ficam]
        ids = {h.chunk_id for h in resultado}
        entra = next(
            (
                h for h in ordenados
                if h.chunk_id not in ids
                and normalizar(chunks[h.chunk_id].principio_ativo) == alvo
                and chunks[h.chunk_id].secao in secoes
                and not _repete(_palavras(chunks[h.chunk_id].texto), palavras_que_ficam)
            ),
            None,
        )
        if entra is not None:
            resultado = ficam + [entra]
    return resultado
