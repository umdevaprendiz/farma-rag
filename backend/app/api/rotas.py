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

from app.api.erros import ErroApi
from app.config import Settings
from app.db.conexao import preparar_banco
from app.llm import ollama
from app.schemas.api import (
    BulaResumo,
    CodigoErro,
    ConsultaDetalhe,
    ConsultaResumo,
    EsclarecimentoOut,
    FonteOut,
    HistoricoOut,
    MedicamentoOut,
    PerguntaIn,
    RespostaOut,
    SaudeOut,
)
from app.services import historico
from app.services.acervo import Acervo, AcervoIncompletoError
from app.services.medicamentos import normalizar
from app.services.rag import Anterior, responder

router = APIRouter(prefix="/api/v1")


class Servicos:
    """O que as requisições compartilham: a configuração, o engine e o acervo, lido do banco uma vez.

    Se o banco estava fora do ar quando a API subiu, cada requisição tenta de novo até conseguir.
    """

    def __init__(self, s: Settings, engine: Engine):
        self.s = s
        self.engine = engine
        self._preparado = False
        self._acervo: Acervo | None = None
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

    def acervo(self) -> Acervo:
        with self._trava:
            if self._acervo is None:
                self._preparar()
                try:
                    self._acervo = Acervo.do_banco(self.engine, self.s)
                except AcervoIncompletoError as e:
                    raise ErroApi(CodigoErro.BANCO_INDISPONIVEL, str(e)) from e
            return self._acervo


def servicos(request: Request) -> Servicos:
    return request.app.state.servicos


@router.get("/saude", response_model=SaudeOut)
def saude(sv: Servicos = Depends(servicos)) -> SaudeOut:
    """Sempre 200: o corpo diz o que está fora do ar, para a tela avisar."""
    problemas: list[str] = []
    banco, bulas, trechos = False, 0, 0
    try:
        acervo = sv.acervo()
        with sv.engine.connect() as conexao:  # o acervo fica em memória: confere se o banco segue no ar
            conexao.execute(text("SELECT 1"))
        banco, bulas, trechos = True, len(acervo.bulas), len(acervo.chunks)
    except ErroApi as e:
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

    return SaudeOut(
        status="ok" if not problemas else "degradado", banco=banco, ollama=no_ar,
        modelos_faltando=faltando, bulas=bulas, trechos=trechos, problemas=problemas,
    )


@router.get("/medicamentos", response_model=list[MedicamentoOut])
def medicamentos(sv: Servicos = Depends(servicos)) -> list[MedicamentoOut]:
    """Os princípios ativos com bula indexada, para o seletor (o filtro é por princípio ativo)."""
    por_ativo: dict[str, list[BulaResumo]] = defaultdict(list)
    for b in sv.acervo().bulas.values():
        por_ativo[b.principio_ativo].append(BulaResumo.model_validate(b, from_attributes=True))
    return [
        MedicamentoOut(principio_ativo=ativo, bulas=sorted(bulas, key=lambda b: normalizar(b.medicamento)))
        for ativo, bulas in sorted(por_ativo.items(), key=lambda par: normalizar(par[0]))
    ]


@router.post("/perguntas", response_model=RespostaOut)
def perguntar(entrada: PerguntaIn, sv: Servicos = Depends(servicos)) -> RespostaOut:
    """Roda o agente (services/grafo.py) e grava a consulta, inclusive quando termina em erro."""
    acervo = sv.acervo()
    anterior = None
    if entrada.consulta_origem_id and (c := historico.buscar(sv.engine, entrada.consulta_origem_id)):
        anterior = Anterior(c.pergunta, c.principios_ativos or [], c.status == "precisa_esclarecimento")
    inicio = time.perf_counter()
    r = responder(entrada.pergunta, acervo, entrada.principios_ativos, sv.s, anterior)
    latencia_ms = round((time.perf_counter() - inicio) * 1000)
    modelo = "frases da bula (sem LLM)" if r.extrativa else sv.s.llm_modelo
    consulta = historico.gravar(sv.engine, entrada, r, acervo, modelo, latencia_ms)
    if r.status == "erro":
        raise ErroApi(r.codigo_erro, r.detalhe_erro, consulta.id)
    return RespostaOut(
        consulta_id=consulta.id,
        status=r.status,
        esclarecimento=(
            EsclarecimentoOut.model_validate(r.esclarecimento, from_attributes=True) if r.esclarecimento else None
        ),
        rodadas=r.rodadas,
        motivo=r.motivo,
        resposta=r.resposta,
        # As mesmas fontes que o histórico vai mostrar: as do snapshot gravado.
        fontes=[FonteOut.model_validate(f, from_attributes=True) for f in consulta.fontes],
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


@router.get("/historico", response_model=HistoricoOut)
def listar_historico(
    limite: int = Query(20, ge=1, le=100), pagina: int = Query(1, ge=1), sv: Servicos = Depends(servicos)
) -> HistoricoOut:
    consultas, total = historico.listar(sv.banco(), limite, pagina)
    return HistoricoOut(
        itens=[ConsultaResumo.model_validate(c, from_attributes=True) for c in consultas],
        total=total, pagina=pagina, limite=limite,
    )


@router.get("/historico/{consulta_id}", response_model=ConsultaDetalhe)
def detalhar_consulta(consulta_id: UUID, sv: Servicos = Depends(servicos)) -> ConsultaDetalhe:
    consulta = historico.buscar(sv.banco(), consulta_id)
    if consulta is None:
        raise ErroApi(CodigoErro.CONSULTA_NAO_ENCONTRADA, f"A consulta {consulta_id} não está no histórico.")
    return ConsultaDetalhe.model_validate(consulta, from_attributes=True)
