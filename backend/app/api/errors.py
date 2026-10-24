"""Todo erro da API sai no formato ErrorOut, com o status HTTP do catálogo (schemas/api.py)."""

from uuid import UUID

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError

from app.config import Settings
from app.schemas.api import HTTP_POR_CODIGO, ErrorCode, ErrorDetail, ErrorOut


class ApiError(Exception):
    def __init__(self, codigo: ErrorCode, mensagem: str, consulta_id: UUID | None = None):
        super().__init__(mensagem)
        self.codigo = codigo
        self.mensagem = mensagem
        self.consulta_id = consulta_id


def resposta_de_erro(codigo: ErrorCode, mensagem: str, consulta_id: UUID | None = None) -> JSONResponse:
    corpo = ErrorOut(erro=ErrorDetail(codigo=codigo, mensagem=mensagem), consulta_id=consulta_id)
    return JSONResponse(corpo.model_dump(mode="json"), status_code=HTTP_POR_CODIGO[codigo])


def mensagem_banco_fora(s: Settings) -> str:
    """O texto do BANCO_INDISPONIVEL, o mesmo no erro HTTP comum e no erro do fluxo com progresso."""
    return f"O banco não respondeu em {s.db_host}:{s.db_port}. Suba com: docker compose up -d db"


def _campo(local: tuple) -> str:
    """("body", "pergunta") -> "pergunta"; ("query", "limite") -> "limite"."""
    return ".".join(str(parte) for parte in local[1:]) or str(local[0])


def registrar(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _erro_api(_: Request, e: ApiError) -> JSONResponse:
        return resposta_de_erro(e.codigo, e.mensagem, e.consulta_id)

    @app.exception_handler(RequestValidationError)
    async def _entrada_invalida(_: Request, e: RequestValidationError) -> JSONResponse:
        partes = [f"{_campo(erro['loc'])}: {erro['msg']}" for erro in e.errors()]
        return resposta_de_erro(ErrorCode.ENTRADA_INVALIDA, "; ".join(partes))

    @app.exception_handler(OperationalError)
    async def _banco_fora(request: Request, _: OperationalError) -> JSONResponse:
        return resposta_de_erro(ErrorCode.BANCO_INDISPONIVEL, mensagem_banco_fora(request.app.state.servicos.s))
