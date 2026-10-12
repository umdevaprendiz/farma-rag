"""Validação da saída do LLM: formato JSON e coerência das citações.

Limite conhecido: isto prova que as citações apontam para trechos que realmente foram
enviados, não que a afirmação está sustentada pelo trecho. Isso é medido no benchmark
(métrica de fidelidade, revisão manual).
"""

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from app.schemas.api import CodigoErro

_MARCADOR = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


class SaidaLLM(BaseModel):
    status: Literal["respondida", "nao_encontrada"]
    resposta: str | None = None
    citacoes: list[int] = Field(default_factory=list)


class SaidaInvalidaError(Exception):
    def __init__(self, codigo: CodigoErro, detalhe: str):
        super().__init__(detalhe)
        self.codigo = codigo
        self.detalhe = detalhe


def marcadores_no_texto(texto: str) -> set[int]:
    """Números citados como [n] ou [n, m] no texto."""
    numeros: set[int] = set()
    for grupo in _MARCADOR.findall(texto):
        numeros.update(int(n) for n in grupo.split(","))
    return numeros


def validar_saida(bruto: str | dict, n_trechos: int) -> SaidaLLM:
    """Valida a saída do LLM; levanta SaidaInvalidaError com o código de erro certo.

    SAIDA_INVALIDA: JSON quebrado, campos errados ou status incoerente.
    CITACAO_INVALIDA: citação para trecho que não foi enviado, ou texto e lista divergentes.
    """
    try:
        dados = json.loads(bruto) if isinstance(bruto, str) else bruto
        saida = SaidaLLM.model_validate(dados)
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        raise SaidaInvalidaError(CodigoErro.SAIDA_INVALIDA, f"JSON inválido: {e}") from e

    if saida.status == "nao_encontrada":
        if saida.resposta or saida.citacoes:
            raise SaidaInvalidaError(
                CodigoErro.SAIDA_INVALIDA, "nao_encontrada não pode trazer resposta nem citações"
            )
        return saida

    if not saida.resposta or not saida.resposta.strip():
        raise SaidaInvalidaError(CodigoErro.SAIDA_INVALIDA, "resposta vazia")
    if not saida.citacoes:
        raise SaidaInvalidaError(CodigoErro.CITACAO_INVALIDA, "resposta sem citações")

    citadas = set(saida.citacoes)
    inexistentes = {n for n in citadas if not 1 <= n <= n_trechos}
    if inexistentes:
        raise SaidaInvalidaError(
            CodigoErro.CITACAO_INVALIDA,
            f"citações {sorted(inexistentes)} não existem entre os {n_trechos} trechos enviados",
        )
    no_texto = marcadores_no_texto(saida.resposta)
    if no_texto != citadas:
        raise SaidaInvalidaError(
            CodigoErro.CITACAO_INVALIDA,
            f"marcadores no texto {sorted(no_texto)} diferem de citacoes {sorted(citadas)}",
        )
    return saida
