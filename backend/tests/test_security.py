"""Correções de segurança da análise de 2026-10-29 (docs/SEGURANCA.md). Sem banco e sem Ollama."""

import logging

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.security import SecurityHeadersMiddleware
from app.config import Settings
from app.main import criar_app
from app.schemas.api import HTTP_POR_CODIGO, MAX_NOME_NO_FILTRO, ErrorCode, QuestionIn


def _app_de_teste() -> TestClient:
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/api/v1/x")
    def x():
        return {"ok": True}

    @app.get("/api/v1/com-cache")
    def com_cache():
        return JSONResponse({"ok": True}, headers={"Cache-Control": "max-age=5"})

    @app.get("/api/v1/fluxo")
    def fluxo():
        return StreamingResponse((f"linha {i}\n" for i in range(3)), media_type="text/plain")

    @app.get("/fora")
    def fora():
        return {"ok": True}

    return TestClient(app)


def test_respostas_da_api_vao_sem_cache_e_sem_adivinhar_o_tipo():
    r = _app_de_teste().get("/api/v1/x")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["cache-control"] == "no-store"


def test_fora_do_prefixo_da_api_nao_mexe():
    r = _app_de_teste().get("/fora")
    assert "x-content-type-options" not in r.headers and "cache-control" not in r.headers


def test_nao_sobrescreve_um_cache_control_que_a_rota_ja_definiu():
    r = _app_de_teste().get("/api/v1/com-cache")
    assert r.headers["cache-control"] == "max-age=5" and r.headers["x-content-type-options"] == "nosniff"


def test_o_fluxo_passa_pelo_middleware_inteiro():
    # ASGI puro: não guarda a resposta em buffer (o NDJSON do progresso precisa sair linha a linha).
    r = _app_de_teste().get("/api/v1/fluxo")
    assert r.text == "linha 0\nlinha 1\nlinha 2\n" and r.headers["x-content-type-options"] == "nosniff"


def test_a_api_de_verdade_manda_os_cabecalhos_em_toda_rota(monkeypatch):
    from app.api import routes

    monkeypatch.setattr(routes.Services, "acervo", lambda self: None)
    cliente = TestClient(criar_app(Settings(_env_file=None, aquecer_modelos=False)))
    r = cliente.get("/api/v1/historico/nao-e-uuid")  # 422 do Pydantic: o erro também leva os cabeçalhos
    assert r.status_code == 422 and r.headers["cache-control"] == "no-store"


def test_nome_no_filtro_tem_tamanho_maximo():
    QuestionIn(pergunta="Pode com álcool?", principios_ativos=["a" * MAX_NOME_NO_FILTRO])
    with pytest.raises(ValidationError, match="longo demais"):
        QuestionIn(pergunta="Pode com álcool?", principios_ativos=["a" * (MAX_NOME_NO_FILTRO + 1)])


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://host/x", "localhost:11434", "gopher://x"])
def test_ollama_url_so_aceita_http_e_https(url):
    # urllib abre file:// e ftp://: sem isto, um .env adulterado leria arquivo ou falaria com outro serviço.
    with pytest.raises(ValidationError, match="http"):
        Settings(_env_file=None, ollama_url=url)


@pytest.mark.parametrize("url", ["http://localhost:11434", "https://ollama.interno:443", "http://host.docker.internal:11434"])
def test_ollama_url_http_funciona(url):
    assert Settings(_env_file=None, ollama_url=url).ollama_url == url


def test_perguntas_simultaneas_precisa_ser_uma_ou_mais():
    assert Settings(_env_file=None).perguntas_simultaneas == 2
    with pytest.raises(ValidationError):
        Settings(_env_file=None, perguntas_simultaneas=0)


def test_senha_de_exemplo_gera_aviso_ao_criar_a_api(caplog):
    with caplog.at_level(logging.WARNING, logger="farmarag"):
        criar_app(Settings(_env_file=None, db_password="troque-esta-senha", aquecer_modelos=False))
    assert "senha de exemplo" in caplog.text


def test_senha_propria_nao_gera_aviso(caplog):
    with caplog.at_level(logging.WARNING, logger="farmarag"):
        criar_app(Settings(_env_file=None, db_password="Xk9-uma-senha-longa-e-so-minha", aquecer_modelos=False))
    assert "senha de exemplo" not in caplog.text


def test_servidor_ocupado_e_429():
    assert HTTP_POR_CODIGO[ErrorCode.SERVIDOR_OCUPADO] == 429
