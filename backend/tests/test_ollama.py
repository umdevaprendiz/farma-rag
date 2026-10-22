"""O pedido de geração ao Ollama."""

from app.config import Settings
from app.llm import ollama


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
