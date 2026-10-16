"""Histórico de consultas: toda pergunta que chega à API é gravada, inclusive as que deram erro.

As fontes são um snapshot (docs/DECISOES.md, seção 6): cópia do trecho e dos dados da bula no
momento da consulta, sem FK para `chunks`, que é recriada a cada carga do banco.
"""

from dataclasses import asdict
from uuid import UUID

from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session, selectinload

from app.db.models import Consulta, ConsultaFonte
from app.schemas.api import PerguntaIn
from app.services.acervo import Acervo
from app.services.rag import Fonte, Resultado


def _snapshot(f: Fonte, acervo: Acervo) -> ConsultaFonte:
    bula = acervo.bulas[f.chunk.bula_id]
    return ConsultaFonte(
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
    engine: Engine, entrada: PerguntaIn, r: Resultado, acervo: Acervo, modelo: str, latencia_ms: int
) -> Consulta:
    """Grava a consulta e devolve-a com `id`, `criado_em` e as fontes, legíveis depois do commit."""
    consulta = Consulta(
        pergunta=entrada.pergunta,
        principios_ativos_filtro=entrada.principios_ativos,
        resposta=r.resposta,
        status=r.status,
        motivo=r.motivo,
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


def listar(engine: Engine, limite: int, pagina: int) -> tuple[list[Consulta], int]:
    """Uma página do histórico, da mais recente para a mais antiga, e o total de consultas."""
    with Session(engine) as sessao:
        total = sessao.scalar(select(func.count()).select_from(Consulta))
        consultas = sessao.scalars(
            select(Consulta)
            .order_by(Consulta.criado_em.desc(), Consulta.id)
            .limit(limite)
            .offset((pagina - 1) * limite)
        )
        return list(consultas), total


def buscar(engine: Engine, consulta_id: UUID) -> Consulta | None:
    with Session(engine) as sessao:
        return sessao.scalar(
            select(Consulta).options(selectinload(Consulta.fontes)).where(Consulta.id == consulta_id)
        )
