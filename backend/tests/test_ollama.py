"""O pedido de geração ao Ollama."""

import logging

import pytest

from app import main
from app.config import Settings
from app.llm import ollama
from app.schemas.api import ErrorCode


def _pedido(monkeypatch, s: Settings) -> dict:
    enviados = []

    def pedir(caminho, corpo, timeout, s):
        enviados.append(corpo)
        return {"message": {"content": "{}"}}

    monkeypatch.setattr(ollama, "_pedir", pedir)
    ollama.chat("sistema", "usuário", s=s)
    return enviados[0]


def test_sem_semente_o_ollama_sorteia(monkeypatch):
    assert "seed" not in _pedido(monkeypatch, Settings(_env_file=None))["options"]


def test_semente_fixa_vai_nas_opcoes(monkeypatch):
    # Com a semente, o mesmo prompt dá a mesma resposta: o benchmark completo compara configurações.
    opcoes = _pedido(monkeypatch, Settings(_env_file=None, llm_semente=42))["options"]
    assert opcoes["seed"] == 42 and opcoes["temperature"] == 0.1


# --- F3c: keep_alive configurável e modelo aquecido (docs/DECISOES.md, seção 35) -----------------

@pytest.mark.parametrize("texto, esperado", [
    ("30m", "30m"), ("2h", "2h"), (" 10m ", "10m"), ("300", 300), ("-1", -1),
])
def test_keep_alive_no_formato_que_o_ollama_aceita(texto, esperado):
    # "-1" (para sempre) tem de ir como número: o Ollama recusa o texto "-1".
    assert ollama._keep_alive(Settings(_env_file=None, keep_alive=texto)) == esperado


def test_chat_e_embed_usam_o_keep_alive_do_env(monkeypatch):
    enviados = []

    def pedir(caminho, corpo, timeout, s):
        enviados.append((caminho, corpo["keep_alive"]))
        return {"message": {"content": "{}"}, "embeddings": [[0.0] * 1024]}

    monkeypatch.setattr(ollama, "_pedir", pedir)
    s = Settings(_env_file=None, keep_alive="-1")
    ollama.chat("sistema", "usuário", s=s)
    ollama.embed(["texto"], s)
    assert enviados == [("/api/chat", -1), ("/api/embed", -1)]


def _registrar(monkeypatch, falhas: dict[str, str] | None = None):
    """Troca o _pedir: guarda (caminho, modelo, corpo) e falha nos modelos de `falhas`."""
    chamadas = []

    def pedir(caminho, corpo, timeout, s):
        chamadas.append((caminho, corpo["model"], corpo))
        if corpo["model"] in (falhas or {}):
            raise ollama.OllamaError(ErrorCode.OLLAMA_INDISPONIVEL, falhas[corpo["model"]])
        return {}

    monkeypatch.setattr(ollama, "_pedir", pedir)
    return chamadas


def test_aquecer_carrega_embeddings_e_geracao_sem_gerar_nada(monkeypatch):
    chamadas = _registrar(monkeypatch)
    s = Settings(_env_file=None, llm_modelo="qwen2.5:1.5b", embed_modelo="bge-m3", keep_alive="45m")
    assert ollama.aquecer(s) == {"bge-m3": None, "qwen2.5:1.5b": None}
    # Embeddings primeiro (toda pergunta começa por ele); geração sem prompt e sem stream: o Ollama só carrega.
    assert [(c[0], c[1]) for c in chamadas] == [("/api/embed", "bge-m3"), ("/api/generate", "qwen2.5:1.5b")]
    assert chamadas[1][2] == {"model": "qwen2.5:1.5b", "stream": False, "keep_alive": "45m"}
    assert all(c[2]["keep_alive"] == "45m" for c in chamadas)
    assert "prompt" not in chamadas[1][2]


def test_aquecer_inclui_o_modelo_de_verificacao_so_quando_e_outro(monkeypatch):
    chamadas = _registrar(monkeypatch)
    s = Settings(_env_file=None, llm_modelo="qwen2.5:1.5b", llm_modelo_verificacao="llama3.2:3b")
    assert list(ollama.aquecer(s)) == ["bge-m3", "qwen2.5:1.5b", "llama3.2:3b"]
    chamadas.clear()
    igual = Settings(_env_file=None, llm_modelo="qwen2.5:1.5b", llm_modelo_verificacao="qwen2.5:1.5b")
    assert list(ollama.aquecer(igual)) == ["bge-m3", "qwen2.5:1.5b"]
    assert len(chamadas) == 2  # o mesmo modelo não é carregado duas vezes


def test_aquecer_nao_levanta_e_um_modelo_que_falha_nao_impede_os_outros(monkeypatch):
    _registrar(monkeypatch, {"bge-m3": "Ollama respondeu 404: model not found"})
    resultado = ollama.aquecer(Settings(_env_file=None))
    assert resultado["bge-m3"] == "Ollama respondeu 404: model not found"
    assert resultado["qwen2.5:1.5b"] is None


def test_subida_aquece_em_segundo_plano_e_loga(monkeypatch, caplog):
    monkeypatch.setattr(ollama, "aquecer", lambda s: {"bge-m3": None, "qwen2.5:1.5b": "fora do ar"})
    with caplog.at_level(logging.INFO, logger="farmarag"):
        main.aquecer_em_segundo_plano(Settings(_env_file=None)).join(timeout=5)
    mensagens = " | ".join(r.getMessage() for r in caplog.records)
    assert "qwen2.5:1.5b não foi aquecido: fora do ar" in mensagens and "carregados no Ollama" in mensagens


def test_erro_inesperado_no_aquecimento_nao_derruba_a_api(monkeypatch, caplog):
    def estoura(s):
        raise RuntimeError("bug")

    monkeypatch.setattr(ollama, "aquecer", estoura)
    fio = main.aquecer_em_segundo_plano(Settings(_env_file=None))
    fio.join(timeout=5)
    assert not fio.is_alive() and "Falha inesperada" in caplog.text  # registrou e terminou


@pytest.mark.parametrize("aquecer, chamou", [(True, 1), (False, 0)])
def test_a_api_so_aquece_ao_subir_se_AQUECER_MODELOS(monkeypatch, aquecer, chamou):
    from fastapi.testclient import TestClient

    from app.api import routes

    chamadas = []
    monkeypatch.setattr(main, "aquecer_em_segundo_plano", lambda s: chamadas.append(s))
    monkeypatch.setattr(routes.Services, "acervo", lambda self: None)  # sem banco
    with TestClient(main.criar_app(Settings(_env_file=None, aquecer_modelos=aquecer))):
        pass
    assert len(chamadas) == chamou


def test_nenhum_teste_aquece_o_ollama_de_verdade():
    # O conftest.py desliga AQUECER_MODELOS para todo Settings criado nos testes.
    assert Settings(_env_file=None).aquecer_modelos is False
