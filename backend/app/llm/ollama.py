"""Cliente mínimo do Ollama: embeddings, chat e lista de modelos. Só biblioteca padrão."""

import json
import re
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.schemas.api import ErrorCode

NS = 1e9


def _keep_alive(s: Settings) -> str | int:
    """KEEP_ALIVE do .env no formato que o Ollama aceita: duração ("30m") como texto, segundos e
    "-1" (para sempre) como número. O Ollama recusa "-1" escrito como texto."""
    v = s.keep_alive.strip()
    return int(v) if re.fullmatch(r"-?\d+", v) else v


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
    r = _pedir("/api/embed", {"model": s.embed_modelo, "input": textos, "keep_alive": _keep_alive(s)}, timeout, s)
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
        "keep_alive": _keep_alive(s),
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


def aquecer(s: Settings | None = None) -> dict[str, str | None]:
    """Carrega no Ollama os modelos que uma pergunta usa, sem gerar nada (F3c, docs/DECISOES.md,
    seção 35): sem isto, a primeira pergunta depois de subir o Ollama paga a carga do modelo.

    O de embeddings vem primeiro (toda pergunta começa por ele), depois o de geração e, se for outro,
    o de verificação. Devolve {modelo: None se carregou, ou o motivo da falha}: um modelo que falha
    (ex.: não baixado) não impede os outros, e nada aqui levanta erro (quem chama é a subida da API).

    Geração: /api/generate sem prompt só carrega o modelo. Embeddings: /api/embed com um texto curto
    (o /api/embed não tem "só carregar")."""
    s = s or get_settings()
    ka = _keep_alive(s)
    pedidos: dict[str, tuple[str, dict]] = {
        s.embed_modelo: ("/api/embed", {"model": s.embed_modelo, "input": ["aquecer"], "keep_alive": ka}),
    }
    for modelo in [s.llm_modelo, s.llm_modelo_verificacao]:
        if modelo and modelo not in pedidos:
            pedidos[modelo] = ("/api/generate", {"model": modelo, "stream": False, "keep_alive": ka})
    resultado: dict[str, str | None] = {}
    for modelo, (caminho, corpo) in pedidos.items():
        try:
            _pedir(caminho, corpo, s.timeout_llm_s, s)
            resultado[modelo] = None
        except OllamaError as e:
            resultado[modelo] = e.detalhe
    return resultado


def modelos(s: Settings | None = None, timeout: float = 3) -> list[str]:
    """Modelos baixados, como o Ollama lista ("qwen2.5:1.5b", "bge-m3:latest"). Para o /saude."""
    s = s or get_settings()
    return [m["name"] for m in _pedir("/api/tags", None, timeout, s)["models"]]


def faltando(s: Settings, baixados: list[str]) -> list[str]:
    """Modelos do .env que não estão em `baixados`. Sem tag, o Ollama entende ":latest"."""
    com_tag = {n if ":" in n else f"{n}:latest" for n in baixados}
    pedidos = [s.embed_modelo, s.llm_modelo] + ([s.llm_modelo_verificacao] if s.llm_modelo_verificacao else [])
    return [m for m in dict.fromkeys(pedidos) if (m if ":" in m else f"{m}:latest") not in com_tag]
