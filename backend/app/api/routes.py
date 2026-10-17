"""Rotas /api/v1.

Os endpoints são `def`, não `async def`: o FastAPI roda cada um numa thread à parte, e a pergunta
fica dezenas de segundos esperando o Ollama de forma síncrona (app/llm/ollama.py).
"""

import threading
import time
from collections import defaultdict
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import Engine, text
from sqlalchemy.exc import OperationalError

from app.api.errors import ApiError
from app.config import Settings
from app.db.connection import preparar_banco
from app.llm import ollama
from app.schemas.api import (
    AnswerOut,
    ClarificationOut,
    ErrorCode,
    HealthOut,
    HistoryOut,
    InquiryDetail,
    InquirySummary,
    LeafletSummary,
    MedicineOut,
    QuestionIn,
    SourceOut,
)
from app.services import history
from app.services.corpus import Corpus, IncompleteCorpusError
from app.services.medicines import normalizar
from app.services.rag import PreviousQuestion, responder

router = APIRouter(prefix="/api/v1")


class Services:
    """O que as requisições compartilham: a configuração, o engine e o acervo, lido do banco uma vez.

    Se o banco estava fora do ar quando a API subiu, cada requisição tenta de novo até conseguir.
    """

    def __init__(self, s: Settings, engine: Engine):
        self.s = s
        self.engine = engine
        self._preparado = False
        self._acervo: Corpus | None = None
        self._trava = threading.Lock()  # duas primeiras requisições juntas não carregam duas vezes

    def _preparar(self) -> None:
        if not self._preparado:
            preparar_banco(self.engine)
            self._preparado = True

    def banco(self) -> Engine:
        """O engine, com as tabelas criadas (o histórico não depende de haver bulas)."""
        with self._trava:
            self._preparar()
        return self.engine

    def acervo(self) -> Corpus:
        with self._trava:
            if self._acervo is None:
                self._preparar()
                try:
                    self._acervo = Corpus.do_banco(self.engine, self.s)
                except IncompleteCorpusError as e:
                    raise ApiError(ErrorCode.BANCO_INDISPONIVEL, str(e)) from e
            return self._acervo


def servicos(request: Request) -> Services:
    return request.app.state.servicos


@router.get("/saude", response_model=HealthOut)
def saude(sv: Services = Depends(servicos)) -> HealthOut:
    """Sempre 200: o corpo diz o que está fora do ar, para a tela avisar."""
    problemas: list[str] = []
    banco, bulas, trechos = False, 0, 0
    try:
        acervo = sv.acervo()
        with sv.engine.connect() as conexao:  # o acervo fica em memória: confere se o banco segue no ar
            conexao.execute(text("SELECT 1"))
        banco, bulas, trechos = True, len(acervo.bulas), len(acervo.chunks)
    except ApiError as e:
        problemas.append(e.mensagem)
    except OperationalError:
        problemas.append(f"O banco não respondeu em {sv.s.db_host}:{sv.s.db_port}. Suba com: docker compose up -d db")

    no_ar, faltando = False, []
    try:
        faltando = ollama.faltando(sv.s, ollama.modelos(sv.s))
        no_ar = True
    except ollama.OllamaError as e:
        problemas.append(e.detalhe)
    if faltando:
        problemas.append(f"Modelo(s) não baixado(s) no Ollama: {', '.join(faltando)}")

    return HealthOut(
        status="ok" if not problemas else "degradado", banco=banco, ollama=no_ar,
        modelos_faltando=faltando, bulas=bulas, trechos=trechos, problemas=problemas,
    )


@router.get("/medicamentos", response_model=list[MedicineOut])
def medicamentos(sv: Services = Depends(servicos)) -> list[MedicineOut]:
    """Os princípios ativos com bula indexada, para o seletor (o filtro é por princípio ativo)."""
    por_ativo: dict[str, list[LeafletSummary]] = defaultdict(list)
    for b in sv.acervo().bulas.values():
        por_ativo[b.principio_ativo].append(LeafletSummary.model_validate(b, from_attributes=True))
    return [
        MedicineOut(principio_ativo=ativo, bulas=sorted(bulas, key=lambda b: normalizar(b.medicamento)))
        for ativo, bulas in sorted(por_ativo.items(), key=lambda par: normalizar(par[0]))
    ]


@router.post("/perguntas", response_model=AnswerOut)
def perguntar(entrada: QuestionIn, sv: Services = Depends(servicos)) -> AnswerOut:
    """Roda o agente (services/graph.py) e grava a consulta, inclusive quando termina em erro."""
    acervo = sv.acervo()
    anterior = None
    if entrada.consulta_origem_id and (c := history.buscar(sv.engine, entrada.consulta_origem_id)):
        anterior = PreviousQuestion(c.pergunta, c.principios_ativos or [], c.status == "precisa_esclarecimento")
    inicio = time.perf_counter()
    r = responder(entrada.pergunta, acervo, entrada.principios_ativos, sv.s, anterior)
    latencia_ms = round((time.perf_counter() - inicio) * 1000)
    modelo = "frases da bula (sem LLM)" if r.extrativa else sv.s.llm_modelo
    consulta = history.gravar(sv.engine, entrada, r, acervo, modelo, latencia_ms)
    if r.status == "erro":
        raise ApiError(r.codigo_erro, r.detalhe_erro, consulta.id)
    return AnswerOut(
        consulta_id=consulta.id,
        status=r.status,
        esclarecimento=(
            ClarificationOut.model_validate(r.esclarecimento, from_attributes=True) if r.esclarecimento else None
        ),
        rodadas=r.rodadas,
        motivo=r.motivo,
        resposta=r.resposta,
        # As mesmas fontes que o histórico vai mostrar: as do snapshot gravado.
        fontes=[SourceOut.model_validate(f, from_attributes=True) for f in consulta.fontes],
        medicamentos_nao_indexados=r.medicamentos_nao_indexados,
        classes_terapeuticas={
            nome: classe for nome in r.medicamentos_nao_indexados if (classe := acervo.catalogo.classe_de(nome))
        },
        principios_ativos=r.principios_ativos,
        continuacao=r.continuacao,
        alergia_a=r.alergia_a,
        extrativa=r.extrativa,
        modelo=consulta.modelo,
        latencia_ms=latencia_ms,
    )


@router.get("/historico", response_model=HistoryOut)
def listar_historico(
    limite: int = Query(20, ge=1, le=100), pagina: int = Query(1, ge=1), sv: Services = Depends(servicos)
) -> HistoryOut:
    consultas, total = history.listar(sv.banco(), limite, pagina)
    return HistoryOut(
        itens=[InquirySummary.model_validate(c, from_attributes=True) for c in consultas],
        total=total, pagina=pagina, limite=limite,
    )


@router.get("/historico/{consulta_id}", response_model=InquiryDetail)
def detalhar_consulta(consulta_id: UUID, sv: Services = Depends(servicos)) -> InquiryDetail:
    consulta = history.buscar(sv.banco(), consulta_id)
    if consulta is None:
        raise ApiError(ErrorCode.CONSULTA_NAO_ENCONTRADA, f"A consulta {consulta_id} não está no histórico.")
    return InquiryDetail.model_validate(consulta, from_attributes=True)
