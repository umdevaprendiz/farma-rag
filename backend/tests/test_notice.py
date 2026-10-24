"""F2: o aviso de saúde da API (docs/DECISOES.md, seção 33). Sem banco e sem Ollama."""

from datetime import datetime
from uuid import uuid4

import pytest

from app.schemas.api import AVISO_SAUDE, AnswerOut, InquiryDetail


def _resposta(status: str) -> AnswerOut:
    return AnswerOut(
        consulta_id=uuid4(), status=status, resposta="Pode causar sono [1]." if status == "respondida" else None,
        fontes=[], modelo="qwen2.5:1.5b", latencia_ms=1200,
    )


def _consulta(status: str) -> InquiryDetail:
    return InquiryDetail(
        id=uuid4(), pergunta="Loratadina dá sono?", status=status, criado_em=datetime(2026, 10, 24, 9, 0),
        principios_ativos_filtro=[], resposta="Pode causar sono [1]." if status == "respondida" else None,
        motivo=None, codigo_erro=None, mensagem_erro=None, rodadas=1, modelo="qwen2.5:1.5b", latencia_ms=1200,
        consulta_origem_id=None, fontes=[],
    )


def test_aviso_so_na_resposta_respondida():
    assert _resposta("respondida").aviso == AVISO_SAUDE
    assert _resposta("nao_encontrada").aviso is None
    assert _resposta("precisa_esclarecimento").aviso is None


@pytest.mark.parametrize("status", ["respondida", "nao_encontrada", "precisa_esclarecimento", "erro"])
def test_aviso_no_historico_segue_o_status(status):
    assert _consulta(status).aviso == (AVISO_SAUDE if status == "respondida" else None)


def test_aviso_sai_no_json_da_api():
    # É um campo calculado: precisa aparecer no que o FastAPI serializa, senão a tela não o vê.
    assert _resposta("respondida").model_dump(mode="json")["aviso"] == AVISO_SAUDE
    assert _consulta("respondida").model_dump(mode="json")["aviso"] == AVISO_SAUDE
    assert _resposta("nao_encontrada").model_dump(mode="json")["aviso"] is None


def test_aviso_nao_entra_no_texto_da_resposta():
    # A resposta que passa pela fundamentação e vai para o histórico continua só com frases da bula.
    r = _resposta("respondida")
    assert AVISO_SAUDE not in r.resposta
    assert "farmacêutico" in AVISO_SAUDE and "bula" in AVISO_SAUDE
