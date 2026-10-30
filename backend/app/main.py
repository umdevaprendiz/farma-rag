"""A aplicação FastAPI.

Uso (da raiz, com o banco e o Ollama de pé):
  uvicorn app.main:app --app-dir backend --reload
Documentação interativa em http://127.0.0.1:8000/docs.

Ao subir, prepara o banco e lê bulas e trechos (Corpus.do_banco). Se o banco estiver fora do ar, a
API sobe mesmo assim e tenta de novo a cada requisição; o /api/v1/saude diz o que falta. Em paralelo,
carrega os modelos no Ollama (AQUECER_MODELOS), para a primeira pergunta não pagar a carga.
"""

import logging
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError

from app.api import errors, routes
from app.api.security import SecurityHeadersMiddleware
from app.api.errors import ApiError
from app.config import Settings, get_settings
from app.db.connection import criar_engine
from app.llm import ollama

log = logging.getLogger("farmarag")
SENHAS_DE_EXEMPLO = {"senha", "troque-esta-senha", "password", "postgres", "admin", "farmarag"}


def aquecer_em_segundo_plano(s: Settings) -> threading.Thread:
    """F3c (docs/DECISOES.md, seção 35): carrega os modelos no Ollama sem atrasar a subida da API e
    sem nunca derrubá-la: Ollama fora do ar ou modelo não baixado viram só um aviso no log (o /saude
    já mostra o que falta). A thread é daemon: não segura o desligamento do servidor."""

    def trabalho() -> None:
        inicio = time.perf_counter()
        try:
            resultado = ollama.aquecer(s)
        except Exception:  # nada daqui pode matar a thread em silêncio
            log.exception("Falha inesperada ao aquecer os modelos")
            return
        for modelo, erro in resultado.items():
            if erro:
                log.warning("Modelo %s não foi aquecido: %s", modelo, erro)
        carregados = [m for m, erro in resultado.items() if not erro]
        if carregados:
            log.info("Modelos carregados no Ollama em %.1f s: %s", time.perf_counter() - inicio, ", ".join(carregados))

    fio = threading.Thread(target=trabalho, name="aquecer-modelos", daemon=True)
    fio.start()
    return fio


def criar_app(s: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    """`engine`: os testes passam um ligado a um schema temporário."""
    s = s or get_settings()
    if s.db_password in SENHAS_DE_EXEMPLO:
        # Padrão do config.py ou do .env.example: serve para desenvolver, nunca para deixar o banco de pé.
        log.warning("DB_PASSWORD é uma senha de exemplo: troque no .env (docs/SEGURANCA.md, achado 6).")
    servicos = routes.Services(s, engine or criar_engine(s))

    @asynccontextmanager
    async def ciclo_de_vida(_: FastAPI):
        if s.aquecer_modelos:
            aquecer_em_segundo_plano(s)  # em paralelo com a leitura do acervo, que também leva segundos
        try:
            servicos.acervo()
        except (ApiError, OperationalError) as e:
            log.warning("Acervo não carregado ao subir (a API tenta de novo a cada requisição): %s", e)
        yield
        servicos.engine.dispose()

    app = FastAPI(title="FarmaRAG", version="0.1.0", lifespan=ciclo_de_vida)
    app.state.servicos = servicos
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(
        CORSMiddleware, allow_origins=s.cors_origens_lista, allow_methods=["GET", "POST"], allow_headers=["Content-Type"]
    )
    errors.registrar(app)
    app.include_router(routes.router)

    @app.get("/", include_in_schema=False)
    def raiz() -> RedirectResponse:
        """Quem abre a API no navegador cai na documentação, não num "Not Found"."""
        return RedirectResponse("/docs")

    return app


app = criar_app()
