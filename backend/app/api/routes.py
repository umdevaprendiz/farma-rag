"""Rotas /api/v1.

Os endpoints são `def`, não `async def`: o FastAPI roda cada um numa thread à parte, e a pergunta
fica dezenas de segundos esperando o Ollama de forma síncrona (app/llm/ollama.py).
"""

import threading
import time
from collections import defaultdict
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import Engine, text
from sqlalchemy.exc import OperationalError

from app.api.errors import ApiError, mensagem_banco_fora
from app.api.progress import transmitir
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
    MarkOut,
    MedicineOut,
    QuestionIn,
    SourceOut,
)
from app.services import history
from app.services.corpus import Corpus, IncompleteCorpusError
from app.services.medicines import normalizar
from app.services.rag import AnotherExplanation, PreviousQuestion, Progress, responder

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
    return _responder(sv, acervo, entrada, _anterior(sv.engine, entrada.consulta_origem_id))


@router.post("/perguntas/com-progresso")
def perguntar_com_progresso(entrada: QuestionIn, sv: Services = Depends(servicos)) -> StreamingResponse:
    """O mesmo que POST /perguntas, em fluxo NDJSON: uma linha por etapa do agente e, no fim, a
    resposta ou o erro (api/progress.py; docs/DECISOES.md, seção 36). O que se checa antes do agente
    (entrada inválida, acervo, banco fora do ar) continua saindo como erro HTTP comum; o que o agente
    descobre (filtro desconhecido, Ollama fora, rodadas esgotadas) vai como a linha de erro do fluxo."""
    acervo = sv.acervo()
    anterior = _anterior(sv.engine, entrada.consulta_origem_id)
    return transmitir(
        lambda avisar: _responder(sv, acervo, entrada, anterior, avisar=avisar), sv.s.rodadas_max, mensagem_banco_fora(sv.s)
    )


def _preparar_outra_explicacao(
    sv: Services, consulta_id: UUID
) -> tuple[Corpus, QuestionIn, PreviousQuestion | None, AnotherExplanation]:
    """O que "não ajudou" precisa antes de rodar o agente: a pergunta da consulta, sem os trechos
    citados nela e nas explicações antes dela, com o mesmo filtro e a mesma pergunta anterior da
    original. Levanta 404 (consulta que não existe) e 422 (consulta que não foi respondida)."""
    acervo = sv.acervo()
    cadeia = history.explicacoes(sv.engine, consulta_id)
    if not cadeia:
        raise ApiError(ErrorCode.CONSULTA_NAO_ENCONTRADA, f"A consulta {consulta_id} não está no histórico.")
    if cadeia[0].status != "respondida":
        raise ApiError(
            ErrorCode.ENTRADA_INVALIDA,
            f"Só uma resposta (status respondida) pode pedir outra explicação; esta consulta terminou como {cadeia[0].status}.",
        )
    original = cadeia[-1]
    entrada = QuestionIn(
        pergunta=original.pergunta, principios_ativos=original.principios_ativos_filtro, consulta_origem_id=consulta_id
    )
    # Se um elo da cadeia foi apagado, a última da lista já é uma explicação: a origem dela não é
    # a pergunta anterior da conversa.
    anterior = None if original.outra_explicacao else _anterior(sv.engine, original.consulta_origem_id)
    return acervo, entrada, anterior, history.para_outra_explicacao(cadeia)


@router.post("/perguntas/{consulta_id}/outra-explicacao", response_model=AnswerOut)
def outra_explicacao(consulta_id: UUID, sv: Services = Depends(servicos)) -> AnswerOut:
    """"Não ajudou, explique de outro jeito" (docs/DECISOES.md, seção 26): a pergunta da consulta
    roda de novo sem os trechos citados nela e nas explicações antes dela, com o mesmo filtro e a
    mesma pergunta anterior da original. A resposta vira uma consulta nova, ligada à que não
    ajudou (consulta_origem_id, com outra_explicacao)."""
    acervo, entrada, anterior, outra = _preparar_outra_explicacao(sv, consulta_id)
    return _responder(sv, acervo, entrada, anterior, outra)


@router.post("/perguntas/{consulta_id}/outra-explicacao/com-progresso")
def outra_explicacao_com_progresso(consulta_id: UUID, sv: Services = Depends(servicos)) -> StreamingResponse:
    """O "não ajudou" em fluxo NDJSON, como POST /perguntas/com-progresso."""
    acervo, entrada, anterior, outra = _preparar_outra_explicacao(sv, consulta_id)
    return transmitir(
        lambda avisar: _responder(sv, acervo, entrada, anterior, outra, avisar), sv.s.rodadas_max, mensagem_banco_fora(sv.s)
    )


def _anterior(engine: Engine, consulta_id: UUID | None) -> PreviousQuestion | None:
    """A pergunta anterior da conversa, do histórico (a API não guarda conversa)."""
    if consulta_id and (c := history.buscar(engine, consulta_id)):
        return PreviousQuestion(c.pergunta, c.principios_ativos or [], c.status == "precisa_esclarecimento")
    return None


def _responder(
    sv: Services, acervo: Corpus, entrada: QuestionIn, anterior: PreviousQuestion | None,
    outra: AnotherExplanation | None = None, avisar: Progress | None = None,
) -> AnswerOut:
    inicio = time.perf_counter()
    r = responder(entrada.pergunta, acervo, entrada.principios_ativos, sv.s, anterior, outra, avisar)
    latencia_ms = round((time.perf_counter() - inicio) * 1000)
    modelo = sv.s.llm_modelo
    if r.extrativa:
        # A tela reconhece a resposta copiada pelo começo ("frases da bula"), também no histórico.
        modelo = f"frases da bula + explicação ({sv.s.llm_modelo})" if r.explicacao else "frases da bula (sem LLM)"
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
        conclusao=r.conclusao,
        abertura=r.abertura,
        resposta=r.resposta,
        explicacao=r.explicacao,
        marcas=[MarkOut.model_validate(m, from_attributes=True) for m in r.marcas],
        sugestoes=r.sugestoes,
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
        outra_explicacao=r.outra_explicacao,
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
