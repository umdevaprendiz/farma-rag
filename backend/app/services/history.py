"""Histórico de consultas: toda pergunta que chega à API é gravada, inclusive as que deram erro.

As fontes são um snapshot (docs/DECISOES.md, seção 6): cópia do trecho e dos dados da bula no
momento da consulta, sem FK para `chunks`, que é recriada a cada carga do banco.
"""

from dataclasses import asdict
from uuid import UUID

from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session, selectinload

from app.db.models import Inquiry, InquirySource
from app.schemas.api import QuestionIn
from app.services.corpus import Corpus
from app.services.rag import Result, Source


def _snapshot(f: Source, acervo: Corpus) -> InquirySource:
    bula = acervo.bulas[f.chunk.bula_id]
    return InquirySource(
        posicao=f.posicao,
        chunk_id=f.chunk.chunk_id,
        medicamento=f.chunk.medicamento,
        fabricante=bula.fabricante,
        numero_registro=bula.numero_registro,
        secao=f.chunk.secao,
        pagina=f.chunk.pagina,
        data_versao=f.data_versao,
        trecho=f.chunk.texto,
        score=f.score,
        citado=f.citado,
    )


def gravar(
    engine: Engine, entrada: QuestionIn, r: Result, acervo: Corpus, modelo: str, latencia_ms: int
) -> Inquiry:
    """Grava a consulta e devolve-a com `id`, `criado_em` e as fontes, legíveis depois do commit."""
    consulta = Inquiry(
        pergunta=entrada.pergunta,
        principios_ativos_filtro=entrada.principios_ativos,
        principios_ativos=r.principios_ativos,
        resposta=r.resposta,
        conclusao=r.conclusao,
        status=r.status,
        motivo=r.motivo or (r.esclarecimento.motivo if r.esclarecimento else None),
        codigo_erro=r.codigo_erro.value if r.codigo_erro else None,
        mensagem_erro=r.detalhe_erro,
        modelo=modelo,
        latencia_ms=latencia_ms,
        rodadas=r.rodadas,
        tentativas=[asdict(t) for t in r.tentativas],
        consulta_origem_id=entrada.consulta_origem_id,
        fontes=[_snapshot(f, acervo) for f in r.fontes],
    )
    with Session(engine, expire_on_commit=False) as sessao, sessao.begin():
        sessao.add(consulta)
    return consulta


def listar(engine: Engine, limite: int, pagina: int) -> tuple[list[Inquiry], int]:
    """Uma página do histórico, da mais recente para a mais antiga, e o total de consultas."""
    with Session(engine) as sessao:
        total = sessao.scalar(select(func.count()).select_from(Inquiry))
        consultas = sessao.scalars(
            select(Inquiry)
            .order_by(Inquiry.criado_em.desc(), Inquiry.id)
            .limit(limite)
            .offset((pagina - 1) * limite)
        )
        return list(consultas), total


def buscar(engine: Engine, consulta_id: UUID) -> Inquiry | None:
    with Session(engine) as sessao:
        return sessao.scalar(
            select(Inquiry).options(selectinload(Inquiry.fontes)).where(Inquiry.id == consulta_id)
        )
