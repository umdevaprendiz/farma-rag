"""Modelos Pydantic dos endpoints /api/v1 e catálogo de erros."""

from datetime import date, datetime
from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from app.schemas.dataset import ROTULO_SECAO

MAX_MEDICAMENTOS_POR_PERGUNTA = 3


class ErrorCode(str, Enum):
    MEDICAMENTO_NAO_ENCONTRADO = "MEDICAMENTO_NAO_ENCONTRADO"
    CONSULTA_NAO_ENCONTRADA = "CONSULTA_NAO_ENCONTRADA"
    ENTRADA_INVALIDA = "ENTRADA_INVALIDA"
    SAIDA_INVALIDA = "SAIDA_INVALIDA"
    CITACAO_INVALIDA = "CITACAO_INVALIDA"
    # Nenhuma rodada produziu resposta sustentada pelos trechos (fundamentação ou verificação).
    RESPOSTA_NAO_VERIFICADA = "RESPOSTA_NAO_VERIFICADA"
    OLLAMA_INDISPONIVEL = "OLLAMA_INDISPONIVEL"
    BANCO_INDISPONIVEL = "BANCO_INDISPONIVEL"
    TEMPO_ESGOTADO = "TEMPO_ESGOTADO"


HTTP_POR_CODIGO: dict[ErrorCode, int] = {
    ErrorCode.MEDICAMENTO_NAO_ENCONTRADO: 404,
    ErrorCode.CONSULTA_NAO_ENCONTRADA: 404,
    ErrorCode.ENTRADA_INVALIDA: 422,
    ErrorCode.SAIDA_INVALIDA: 502,
    ErrorCode.CITACAO_INVALIDA: 502,
    ErrorCode.RESPOSTA_NAO_VERIFICADA: 502,
    ErrorCode.OLLAMA_INDISPONIVEL: 503,
    ErrorCode.BANCO_INDISPONIVEL: 503,
    ErrorCode.TEMPO_ESGOTADO: 504,
}


class ErrorDetail(BaseModel):
    codigo: ErrorCode
    mensagem: str


class ErrorOut(BaseModel):
    erro: ErrorDetail
    # A consulta gravada no histórico (POST /perguntas que terminou em erro): a tela manda este id
    # como consulta_origem_id na pergunta seguinte, e a conversa continua depois do erro.
    consulta_id: UUID | None = None


class QuestionIn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    pergunta: str = Field(min_length=5, max_length=500)
    # Filtro por princípio ativo (não por bula): vários fabricantes têm bula do mesmo
    # princípio ativo. Vazio = a API tenta detectar o medicamento na própria pergunta.
    principios_ativos: list[str] = Field(
        default_factory=list, max_length=MAX_MEDICAMENTOS_POR_PERGUNTA
    )
    # A pergunta anterior da mesma conversa: a tela manda a última resposta. A pergunta nova pode
    # continuar dela ("e para criança?") ou responder a um esclarecimento ("dipirona", depois de "a
    # alergia é a qual remédio?"). A API continua sem estado: busca a anterior no histórico.
    consulta_origem_id: UUID | None = None

    @field_validator("principios_ativos")
    @classmethod
    def _sem_vazios(cls, v: list[str]) -> list[str]:
        limpos = [p.strip() for p in v]
        if any(not p for p in limpos):
            raise ValueError("principio_ativo vazio")
        return limpos


class SourceOut(BaseModel):
    posicao: int = Field(ge=1)  # o [n] da resposta
    chunk_id: str
    medicamento: str
    # Nulos só nas bulas de teste e em consultas gravadas antes destes campos.
    fabricante: str | None = None
    numero_registro: str | None = None
    secao: str
    pagina: int
    data_versao: date
    trecho: str
    score: float
    # Trecho enviado ao modelo e usado na resposta. Os não citados também aparecem: são o que
    # o sistema consultou.
    citado: bool

    @computed_field
    @property
    def rotulo_secao(self) -> str:
        """A seção como o usuário lê ("Posologia e modo de usar"), sem o frontend repetir a tabela."""
        return ROTULO_SECAO.get(self.secao, self.secao)


class OptionOut(BaseModel):
    rotulo: str
    principios_ativos: list[str]


class ClarificationOut(BaseModel):
    """O frontend mostra as opções e reenvia a pergunta com o filtro escolhido."""

    # opcoes vazio (remedio_da_alergia): a pessoa responde no chat, e a resposta vem com
    # consulta_origem_id desta consulta.
    motivo: Literal["medicamento_nao_reconhecido", "remedio_da_alergia"]
    pergunta: str
    opcoes: list[OptionOut]


class AnswerOut(BaseModel):
    consulta_id: UUID
    status: Literal["respondida", "nao_encontrada", "precisa_esclarecimento"]
    # Só preenchido quando status = precisa_esclarecimento.
    esclarecimento: ClarificationOut | None = None
    rodadas: int = 0
    # Só preenchido quando status = nao_encontrada.
    motivo: Literal["sem_contexto", "medicamento_nao_indexado"] | None = None
    # Pergunta de para que serve ("paracetamol serve para ansiedade?"): o veredito montado no código
    # pela seção de indicações, antes das frases da bula. Não é afirmação do modelo.
    conclusao: str | None = None
    resposta: str | None
    fontes: list[SourceOut]
    # Medicamentos citados na pergunta que não têm bula indexada (o frontend avisa).
    medicamentos_nao_indexados: list[str] = Field(default_factory=list)
    # Para que serve cada um deles, pela classe terapêutica no registro da ANVISA (dados abertos):
    # {"dipirona": "ANALGESICOS NAO NARCOTICOS"}. Sem acento, como no CSV.
    classes_terapeuticas: dict[str, str] = Field(default_factory=dict)
    principios_ativos: list[str] = Field(default_factory=list)  # os usados na busca
    continuacao: bool = False  # usou a pergunta anterior da conversa
    alergia_a: str | None = None  # a resposta traz alternativas a este remédio
    extrativa: bool = False  # a resposta são frases copiadas da bula, sem o modelo de linguagem
    modelo: str
    latencia_ms: int


class LeafletSummary(BaseModel):
    id: str
    medicamento: str
    fabricante: str
    forma_farmaceutica: str | None = None
    data_versao: date


class MedicineOut(BaseModel):
    """Um item do seletor: um princípio ativo com as bulas indexadas dele."""

    principio_ativo: str
    bulas: list[LeafletSummary]


class InquirySummary(BaseModel):
    id: UUID
    pergunta: str
    status: Literal["respondida", "nao_encontrada", "precisa_esclarecimento", "erro"]
    criado_em: datetime


class HistoryOut(BaseModel):
    itens: list[InquirySummary]  # da mais recente para a mais antiga
    total: int
    pagina: int
    limite: int


class InquiryDetail(InquirySummary):
    """Uma consulta como foi respondida, com as fontes guardadas no momento (snapshot)."""

    principios_ativos_filtro: list[str]
    conclusao: str | None = None
    resposta: str | None
    motivo: str | None
    codigo_erro: ErrorCode | None
    mensagem_erro: str | None
    rodadas: int
    modelo: str
    latencia_ms: int
    consulta_origem_id: UUID | None
    fontes: list[SourceOut]


class HealthOut(BaseModel):
    status: Literal["ok", "degradado"]
    banco: bool
    ollama: bool
    # Modelos do .env que o Ollama ainda não tem (ollama pull).
    modelos_faltando: list[str]
    bulas: int
    trechos: int
    # O que está fora do ar e por quê, para mostrar na tela.
    problemas: list[str]
