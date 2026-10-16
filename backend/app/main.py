"""A aplicação FastAPI.

Uso (da raiz, com o banco e o Ollama de pé):
  uvicorn app.main:app --app-dir backend --reload
Documentação interativa em http://127.0.0.1:8000/docs.

Ao subir, prepara o banco e lê bulas e trechos (Acervo.do_banco). Se o banco estiver fora do ar, a
API sobe mesmo assim e tenta de novo a cada requisição; o /api/v1/saude diz o que falta.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError

from app.api import erros, rotas
from app.api.erros import ErroApi
from app.config import Settings, get_settings
from app.db.conexao import criar_engine

log = logging.getLogger("farmarag")


def criar_app(s: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    """`engine`: os testes passam um ligado a um schema temporário."""
    s = s or get_settings()
    servicos = rotas.Servicos(s, engine or criar_engine(s))

    @asynccontextmanager
    async def ciclo_de_vida(_: FastAPI):
        try:
            servicos.acervo()
        except (ErroApi, OperationalError) as e:
            log.warning("Acervo não carregado ao subir (a API tenta de novo a cada requisição): %s", e)
        yield
        servicos.engine.dispose()

    app = FastAPI(title="FarmaRAG", version="0.1.0", lifespan=ciclo_de_vida)
    app.state.servicos = servicos
    app.add_middleware(
        CORSMiddleware, allow_origins=s.cors_origens_lista, allow_methods=["GET", "POST"], allow_headers=["Content-Type"]
    )
    erros.registrar(app)
    app.include_router(rotas.router)

    @app.get("/", include_in_schema=False)
    def raiz() -> RedirectResponse:
        """Quem abre a API no navegador cai na documentação, não num "Not Found"."""
        return RedirectResponse("/docs")

    return app


app = criar_app()
