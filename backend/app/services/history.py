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
from app.services.rag import AnotherExplanation, Result, Source

# Cada elo da cadeia de "não ajudou" é um clique do balconista: o limite só evita laço num
# histórico estragado à mão.
MAX_EXPLICACOES = 20


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
        destaques=[list(d) for d in f.destaques],
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
        abertura=r.abertura,
        explicacao=r.explicacao,
        marcas=[asdict(m) for m in r.marcas],
        sugestoes=r.sugestoes,
        status=r.status,
        motivo=r.motivo or (r.esclarecimento.motivo if r.esclarecimento else None),
        codigo_erro=r.codigo_erro.value if r.codigo_erro else None,
        mensagem_erro=r.detalhe_erro,
        modelo=modelo,
        latencia_ms=latencia_ms,
        rodadas=r.rodadas,
        tentativas=[asdict(t) for t in r.tentativas],
        consulta_origem_id=entrada.consulta_origem_id,
        outra_explicacao=r.outra_explicacao,
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


def explicacoes(engine: Engine, consulta_id: UUID) -> list[Inquiry]:
    """A consulta e as explicações antes dela, da mais recente até a pergunta original, que é a
    última da lista. Vazia se a consulta não existe."""
    cadeia: list[Inquiry] = []
    proxima: UUID | None = consulta_id
    while proxima is not None and len(cadeia) < MAX_EXPLICACOES and (c := buscar(engine, proxima)):
        cadeia.append(c)
        proxima = c.consulta_origem_id if c.outra_explicacao else None
    return cadeia


def para_outra_explicacao(cadeia: list[Inquiry]) -> AnotherExplanation:
    """O que o "não ajudou" evita: os trechos citados em toda a cadeia. E o que fica para depois dos
    nunca enviados: os enviados em qualquer rodada (as fontes gravadas são só as da última; as
    outras rodadas estão em `tentativas`)."""
    citados = [f.chunk_id for c in cadeia for f in c.fontes if f.citado]
    enviados = [
        chunk_id
        for c in cadeia
        for chunk_id in [*(i for t in c.tentativas or [] for i in t.get("chunk_ids", [])), *(f.chunk_id for f in c.fontes)]
    ]
    return AnotherExplanation(evitar=list(dict.fromkeys(citados)), enviados=list(dict.fromkeys(enviados)))
