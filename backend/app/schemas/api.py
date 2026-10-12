"""Modelos Pydantic dos endpoints /api/v1 e catálogo de erros."""

from datetime import date, datetime
from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_MEDICAMENTOS_POR_PERGUNTA = 3


class CodigoErro(str, Enum):
    MEDICAMENTO_NAO_ENCONTRADO = "MEDICAMENTO_NAO_ENCONTRADO"
    ENTRADA_INVALIDA = "ENTRADA_INVALIDA"
    SAIDA_INVALIDA = "SAIDA_INVALIDA"
    CITACAO_INVALIDA = "CITACAO_INVALIDA"
    OLLAMA_INDISPONIVEL = "OLLAMA_INDISPONIVEL"
    BANCO_INDISPONIVEL = "BANCO_INDISPONIVEL"
    TEMPO_ESGOTADO = "TEMPO_ESGOTADO"


HTTP_POR_CODIGO: dict[CodigoErro, int] = {
    CodigoErro.MEDICAMENTO_NAO_ENCONTRADO: 404,
    CodigoErro.ENTRADA_INVALIDA: 422,
    CodigoErro.SAIDA_INVALIDA: 502,
    CodigoErro.CITACAO_INVALIDA: 502,
    CodigoErro.OLLAMA_INDISPONIVEL: 503,
    CodigoErro.BANCO_INDISPONIVEL: 503,
    CodigoErro.TEMPO_ESGOTADO: 504,
}


class ErroDetalhe(BaseModel):
    codigo: CodigoErro
    mensagem: str


class ErroOut(BaseModel):
    erro: ErroDetalhe


class PerguntaIn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    pergunta: str = Field(min_length=5, max_length=500)
    # Filtro por princípio ativo (não por bula): vários fabricantes têm bula do mesmo
    # princípio ativo. Vazio = a API tenta detectar o medicamento na própria pergunta.
    principios_ativos: list[str] = Field(
        default_factory=list, max_length=MAX_MEDICAMENTOS_POR_PERGUNTA
    )

    @field_validator("principios_ativos")
    @classmethod
    def _sem_vazios(cls, v: list[str]) -> list[str]:
        limpos = [p.strip() for p in v]
        if any(not p for p in limpos):
            raise ValueError("principio_ativo vazio")
        return limpos


class FonteOut(BaseModel):
    posicao: int = Field(ge=1)
    chunk_id: str
    medicamento: str
    secao: str
    pagina: int
    data_versao: date
    trecho: str
    score: float


class RespostaOut(BaseModel):
    consulta_id: UUID
    status: Literal["respondida", "nao_encontrada"]
    # Só preenchido quando status = nao_encontrada.
    motivo: Literal["sem_contexto", "medicamento_nao_indexado"] | None = None
    resposta: str | None
    fontes: list[FonteOut]
    # Medicamentos citados na pergunta que não têm bula indexada (o frontend avisa).
    medicamentos_nao_indexados: list[str] = Field(default_factory=list)
    modelo: str
    latencia_ms: int


class BulaResumo(BaseModel):
    id: str
    medicamento: str
    fabricante: str
    data_versao: date


class MedicamentoOut(BaseModel):
    """Um item do seletor: um princípio ativo com as bulas indexadas dele."""

    principio_ativo: str
    bulas: list[BulaResumo]


class ConsultaResumo(BaseModel):
    id: UUID
    pergunta: str
    status: Literal["respondida", "nao_encontrada", "erro"]
    criado_em: datetime
