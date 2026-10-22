"""Cliente mínimo do Ollama: embeddings, chat e lista de modelos. Só biblioteca padrão."""

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.schemas.api import ErrorCode

NS = 1e9
# Mantém o modelo na memória entre perguntas: recarregar custa vários segundos na CPU.
KEEP_ALIVE = "30m"


class OllamaError(Exception):
    def __init__(self, codigo: ErrorCode, detalhe: str):
        super().__init__(detalhe)
        self.codigo = codigo
        self.detalhe = detalhe


@dataclass(frozen=True)
class LLMResponse:
    texto: str
    prompt_tokens: int
    tokens_gerados: int
    prompt_s: float
    geracao_s: float


def _pedir(caminho: str, corpo: dict | None, timeout: float, s: Settings) -> dict:
    """POST com `corpo` em JSON; GET sem corpo."""
    dados = json.dumps(corpo).encode() if corpo is not None else None
    req = urllib.request.Request(s.ollama_url + caminho, dados, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:  # ex.: 404 quando o modelo não foi baixado
        detalhe = e.read().decode("utf-8", "replace")[:300]
        raise OllamaError(ErrorCode.OLLAMA_INDISPONIVEL, f"Ollama respondeu {e.code}: {detalhe}") from e
    except (TimeoutError, socket.timeout) as e:
        raise OllamaError(ErrorCode.TEMPO_ESGOTADO, f"Ollama não respondeu em {timeout:.0f} s") from e
    except urllib.error.URLError as e:
        if isinstance(e.reason, (TimeoutError, socket.timeout)):
            raise OllamaError(ErrorCode.TEMPO_ESGOTADO, f"Ollama não respondeu em {timeout:.0f} s") from e
        raise OllamaError(
            ErrorCode.OLLAMA_INDISPONIVEL, f"Ollama fora do ar em {s.ollama_url}: {e.reason}"
        ) from e


def embed(textos: list[str], s: Settings | None = None) -> list[list[float]]:
    s = s or get_settings()
    # Na CPU cada trecho leva ~3 s; o tempo limite cresce com o lote.
    timeout = s.timeout_llm_s + 10 * len(textos)
    r = _pedir("/api/embed", {"model": s.embed_modelo, "input": textos, "keep_alive": KEEP_ALIVE}, timeout, s)
    vetores = r["embeddings"]
    if vetores and len(vetores[0]) != s.embed_dim:
        raise OllamaError(
            ErrorCode.OLLAMA_INDISPONIVEL,
            f"{s.embed_modelo} devolveu {len(vetores[0])} dimensões, mas EMBED_DIM={s.embed_dim}",
        )
    return vetores


def chat(
    sistema: str, usuario: str, formato: dict | str = "json", s: Settings | None = None, modelo: str | None = None
) -> LLMResponse:
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
    if s.llm_semente is not None:
        corpo["options"]["seed"] = s.llm_semente
    r = _pedir("/api/chat", corpo, s.timeout_llm_s, s)
    return LLMResponse(
        texto=r["message"]["content"],
        prompt_tokens=r.get("prompt_eval_count", 0),
        tokens_gerados=r.get("eval_count", 0),
        prompt_s=r.get("prompt_eval_duration", 0) / NS,
        geracao_s=r.get("eval_duration", 0) / NS,
    )


def modelos(s: Settings | None = None, timeout: float = 3) -> list[str]:
    """Modelos baixados, como o Ollama lista ("qwen2.5:1.5b", "bge-m3:latest"). Para o /saude."""
    s = s or get_settings()
    return [m["name"] for m in _pedir("/api/tags", None, timeout, s)["models"]]


def faltando(s: Settings, baixados: list[str]) -> list[str]:
    """Modelos do .env que não estão em `baixados`. Sem tag, o Ollama entende ":latest"."""
    com_tag = {n if ":" in n else f"{n}:latest" for n in baixados}
    pedidos = [s.embed_modelo, s.llm_modelo] + ([s.llm_modelo_verificacao] if s.llm_modelo_verificacao else [])
    return [m for m in dict.fromkeys(pedidos) if (m if ":" in m else f"{m}:latest") not in com_tag]
