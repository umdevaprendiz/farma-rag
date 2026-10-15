"""Cliente mínimo do Ollama: embeddings e chat. Só biblioteca padrão."""

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.schemas.api import CodigoErro

NS = 1e9
# Mantém o modelo na memória entre perguntas: recarregar custa vários segundos na CPU.
KEEP_ALIVE = "30m"


class OllamaError(Exception):
    def __init__(self, codigo: CodigoErro, detalhe: str):
        super().__init__(detalhe)
        self.codigo = codigo
        self.detalhe = detalhe


@dataclass(frozen=True)
class RespostaLLM:
    texto: str
    prompt_tokens: int
    tokens_gerados: int
    prompt_s: float
    geracao_s: float


def _post(caminho: str, corpo: dict, timeout: float, s: Settings) -> dict:
    req = urllib.request.Request(
        s.ollama_url + caminho, json.dumps(corpo).encode(), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:  # ex.: 404 quando o modelo não foi baixado
        detalhe = e.read().decode("utf-8", "replace")[:300]
        raise OllamaError(CodigoErro.OLLAMA_INDISPONIVEL, f"Ollama respondeu {e.code}: {detalhe}") from e
    except (TimeoutError, socket.timeout) as e:
        raise OllamaError(CodigoErro.TEMPO_ESGOTADO, f"Ollama não respondeu em {timeout:.0f} s") from e
    except urllib.error.URLError as e:
        if isinstance(e.reason, (TimeoutError, socket.timeout)):
            raise OllamaError(CodigoErro.TEMPO_ESGOTADO, f"Ollama não respondeu em {timeout:.0f} s") from e
        raise OllamaError(
            CodigoErro.OLLAMA_INDISPONIVEL, f"Ollama fora do ar em {s.ollama_url}: {e.reason}"
        ) from e


def embed(textos: list[str], s: Settings | None = None) -> list[list[float]]:
    s = s or get_settings()
    # Na CPU cada trecho leva ~3 s; o tempo limite cresce com o lote.
    timeout = s.timeout_llm_s + 10 * len(textos)
    r = _post("/api/embed", {"model": s.embed_modelo, "input": textos, "keep_alive": KEEP_ALIVE}, timeout, s)
    vetores = r["embeddings"]
    if vetores and len(vetores[0]) != s.embed_dim:
        raise OllamaError(
            CodigoErro.OLLAMA_INDISPONIVEL,
            f"{s.embed_modelo} devolveu {len(vetores[0])} dimensões, mas EMBED_DIM={s.embed_dim}",
        )
    return vetores


def chat(
    sistema: str, usuario: str, formato: dict | str = "json", s: Settings | None = None, modelo: str | None = None
) -> RespostaLLM:
    """`formato`: "json" ou um JSON Schema (o Ollama restringe a saída a ele).
    `modelo`: vazio = LLM_MODELO."""
    s = s or get_settings()
    corpo = {
        "model": modelo or s.llm_modelo,
        "messages": [{"role": "system", "content": sistema}, {"role": "user", "content": usuario}],
        "stream": False,
        "format": formato,
        "keep_alive": KEEP_ALIVE,
        "options": {"temperature": s.temperatura, "num_ctx": s.num_ctx, "num_predict": s.num_predict},
    }
    r = _post("/api/chat", corpo, s.timeout_llm_s, s)
    return RespostaLLM(
        texto=r["message"]["content"],
        prompt_tokens=r.get("prompt_eval_count", 0),
        tokens_gerados=r.get("eval_count", 0),
        prompt_s=r.get("prompt_eval_duration", 0) / NS,
        geracao_s=r.get("eval_duration", 0) / NS,
    )
