"""F3d: o fluxo NDJSON com o progresso do agente (api/progress.py; docs/DECISOES.md, seção 36).

Sem banco e sem Ollama: o "trabalho" é uma função de mentira que avisa etapas e devolve uma resposta.
"""

import json
import logging
import threading
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel
from sqlalchemy.exc import OperationalError

from app.api import progress
from app.api.errors import ApiError
from app.schemas.api import ErrorCode


class Resposta(BaseModel):
    texto: str


def _eventos(trabalho, **kw) -> list[dict]:
    return [json.loads(linha) for linha in progress.eventos(trabalho, 3, "banco fora do ar", **kw)]


def test_etapas_na_ordem_e_a_resposta_no_fim():
    def trabalho(avisar):
        for etapa in ("buscando", "escrevendo", "conferindo"):
            avisar(etapa, 1)
        return Resposta(texto="ok")

    assert _eventos(trabalho) == [
        {"tipo": "etapa", "etapa": "buscando", "rodada": 1, "de": 3},
        {"tipo": "etapa", "etapa": "escrevendo", "rodada": 1, "de": 3},
        {"tipo": "etapa", "etapa": "conferindo", "rodada": 1, "de": 3},
        {"tipo": "resposta", "resposta": {"texto": "ok"}},
    ]


def test_erro_da_api_vai_no_fluxo_com_o_status_http_que_ele_teria():
    consulta = uuid4()

    def trabalho(avisar):
        avisar("buscando", 1)
        raise ApiError(ErrorCode.RESPOSTA_NAO_VERIFICADA, "Nenhuma resposta passou em 3 rodadas", consulta)

    assert _eventos(trabalho)[1:] == [{
        "tipo": "erro", "status": 502, "consulta_id": str(consulta),
        "erro": {"codigo": "RESPOSTA_NAO_VERIFICADA", "mensagem": "Nenhuma resposta passou em 3 rodadas"},
    }]


def test_banco_que_cai_no_meio_vira_banco_indisponivel():
    def trabalho(avisar):
        raise OperationalError("INSERT", {}, Exception("conexão perdida"))

    assert _eventos(trabalho) == [{
        "tipo": "erro", "status": 503, "consulta_id": None,
        "erro": {"codigo": "BANCO_INDISPONIVEL", "mensagem": "banco fora do ar"},
    }]


def test_falha_inesperada_termina_o_fluxo_sem_resposta_e_fica_no_log(caplog):
    # Quem lê o fluxo trata "acabou sem linha final" como resposta interrompida (frontend/src/api.ts).
    def trabalho(avisar):
        avisar("buscando", 1)
        raise RuntimeError("bug")

    with caplog.at_level(logging.ERROR, logger="farmarag"):
        eventos = _eventos(trabalho)
    assert [e["tipo"] for e in eventos] == ["etapa"]
    assert "Falha inesperada" in caplog.text


def test_a_linha_chega_antes_de_o_trabalho_terminar():
    # O ponto da F3d: a etapa sai enquanto a geração ainda roda, não junto com a resposta.
    liberar, terminou = threading.Event(), threading.Event()

    def trabalho(avisar):
        avisar("buscando", 1)
        liberar.wait(timeout=5)
        terminou.set()
        return Resposta(texto="ok")

    fluxo = progress.eventos(trabalho, 3, "banco fora do ar")
    assert json.loads(next(fluxo))["etapa"] == "buscando"
    assert not terminou.is_set()  # o trabalho ainda está esperando
    liberar.set()
    resto = [json.loads(linha) for linha in fluxo]
    assert resto == [{"tipo": "resposta", "resposta": {"texto": "ok"}}] and terminou.is_set()


def test_batimento_enquanto_nada_acontece():
    # Uma geração longa não manda nada por dezenas de segundos: o "vivo" segura proxies que fecham
    # conexões paradas.
    liberar = threading.Event()

    def trabalho(avisar):
        liberar.wait(timeout=5)
        return Resposta(texto="ok")

    fluxo = progress.eventos(trabalho, 3, "banco fora do ar", heartbeat_s=0.05)
    assert json.loads(next(fluxo)) == {"tipo": "vivo"}
    liberar.set()
    assert json.loads(list(fluxo)[-1])["tipo"] == "resposta"


def test_resposta_http_e_ndjson_sem_cache():
    app = FastAPI()

    @app.post("/x")
    def x():
        def trabalho(avisar):
            avisar("buscando", 1)
            return Resposta(texto="é")

        return progress.transmitir(trabalho, 3, "banco fora do ar")

    r = TestClient(app).post("/x")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    assert r.headers["cache-control"] == "no-cache" and r.headers["x-accel-buffering"] == "no"
    linhas = r.text.splitlines()
    assert [json.loads(l)["tipo"] for l in linhas] == ["etapa", "resposta"]
    assert json.loads(linhas[1])["resposta"] == {"texto": "é"}  # acento preservado (ensure_ascii=False)
