"""Métricas de recuperação baseadas em âncoras, independentes do tamanho do trecho.

O gabarito não usa chunk_id: ele muda quando CHUNK_TOKENS muda. Em vez disso, cada
pergunta aponta (princípio ativo, seção, frase) e um trecho "acerta" se vier daquela
seção do medicamento e contiver a frase.
"""

from typing import Sequence

from app.schemas.dataset import Anchor, BenchmarkQuestion, Chunk
from app.services.medicines import normalizar


def chunk_cobre_ancora(chunk: Chunk, ancora: Anchor) -> bool:
    return (
        normalizar(chunk.principio_ativo) == normalizar(ancora.principio_ativo)
        and chunk.secao == ancora.secao
        and normalizar(ancora.frase) in normalizar(chunk.texto)
    )


def acertou_recall(recuperados: Sequence[Chunk], pergunta: BenchmarkQuestion) -> bool:
    """Todas as âncoras da pergunta aparecem entre os trechos recuperados (top-k)."""
    return all(any(chunk_cobre_ancora(c, a) for c in recuperados) for a in pergunta.ancoras)


def recall_at_k(
    recuperados_por_pergunta: dict[str, Sequence[Chunk]],
    perguntas: Sequence[BenchmarkQuestion],
    k: int,
) -> float:
    """Perguntas respondíveis com o trecho certo no top-k ÷ total respondível."""
    respondiveis = [p for p in perguntas if p.respondivel]
    if not respondiveis:
        return 0.0
    acertos = sum(
        acertou_recall(recuperados_por_pergunta.get(p.id, [])[:k], p) for p in respondiveis
    )
    return acertos / len(respondiveis)


def recusa_correta(status_por_pergunta: dict[str, str], perguntas: Sequence[BenchmarkQuestion]) -> float:
    """Recusas certas ÷ perguntas sem resposta nas bulas."""
    sem_resposta = [p for p in perguntas if not p.respondivel]
    if not sem_resposta:
        return 0.0
    certas = sum(status_por_pergunta.get(p.id) == "nao_encontrada" for p in sem_resposta)
    return certas / len(sem_resposta)
