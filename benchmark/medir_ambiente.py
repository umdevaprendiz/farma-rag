"""Etapa 0: mede velocidade e memória do Ollama nesta máquina.

Uso (da raiz):  python -m benchmark.medir_ambiente [modelo ...]
Sem argumentos usa LLM_MODELO do .env. Só biblioteca padrão + app.config.

O prompt é SINTÉTICO (texto de bula inventado, ~4 trechos de ~300 tokens + regras + pergunta,
saída em JSON), com o mesmo formato do prompt real. Serve para medir tempo, não qualidade.
"""

import json
import statistics
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from app.config import get_settings

NS = 1e9
RAIZ = Path(__file__).resolve().parents[1]

PARAGRAFO = (
    "O uso concomitante deste medicamento com outros fármacos que atuam sobre a pressão arterial "
    "pode alterar o efeito esperado do tratamento. Recomenda-se monitorar o paciente durante as "
    "primeiras semanas e ajustar a dose conforme a resposta clínica. Em pacientes idosos ou com "
    "insuficiência renal, a eliminação pode ser mais lenta e o acompanhamento deve ser mais próximo. "
    "Não foram observados efeitos relevantes sobre a capacidade de dirigir veículos ou operar máquinas. "
)

REGRAS = (
    "Responda em português, somente com base nos trechos. Cite cada afirmação com [n]. "
    "Se os trechos não respondem, devolva status nao_encontrada. Não recomende dose para um paciente "
    'específico. Devolva só JSON: {"status": "respondida|nao_encontrada", "resposta": "...", "citacoes": [1]}.'
)


def montar_prompt(n_trechos: int = 4, variante: int = 0) -> str:
    """`variante` muda o início do prompt: o Ollama reaproveita o cache de prompts com o mesmo
    prefixo (prompt_eval cai para ~0 s), o que esconderia o custo real de cada pergunta nova."""
    trechos = "\n\n".join(
        f"[{i}] Medicamento exemplo 50 mg | seção: interacoes | página {i}\n{PARAGRAFO * 3}"
        for i in range(1, n_trechos + 1)
    )
    return f"Consulta {variante}.\n{REGRAS}\n\nTRECHOS:\n{trechos}\n\nPERGUNTA: Este medicamento interage com anti-inflamatórios?"


def _post(url: str, payload: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.load(r)


def medir_geracao(modelo: str, n_trechos: int, rodadas: int = 3) -> dict:
    s = get_settings()
    resultados = []
    for i in range(rodadas):  # a 1ª rodada inclui o carregamento do modelo (load_duration)
        corpo = {
            "model": modelo,
            "prompt": montar_prompt(n_trechos, variante=i),
            "stream": False,
            "format": "json",
            "options": {
                "temperature": s.temperatura,
                "num_ctx": s.num_ctx,
                "num_predict": s.num_predict,
            },
        }
        t0 = time.perf_counter()
        r = _post(f"{s.ollama_url}/api/generate", corpo)
        parede = time.perf_counter() - t0
        resultados.append(
            {
                "rodada": i + 1,
                "parede_s": round(parede, 1),
                "carregamento_s": round(r["load_duration"] / NS, 1),
                "prompt_tokens": r["prompt_eval_count"],
                "prompt_s": round(r["prompt_eval_duration"] / NS, 1),
                "gerados": r["eval_count"],
                "tokens_por_s": round(r["eval_count"] / (r["eval_duration"] / NS), 1),
                "json_valido": _json_ok(r["response"]),
            }
        )
        print(f"  {resultados[-1]}", flush=True)
    quentes = resultados[1:] or resultados
    ps = next((m for m in _get(f"{s.ollama_url}/api/ps")["models"] if m["name"].startswith(modelo)), {})
    return {
        "modelo": modelo,
        "rodadas": resultados,
        "mediana_parede_s_apos_aquecer": statistics.median(r["parede_s"] for r in quentes),
        "tokens_por_s_mediana": statistics.median(r["tokens_por_s"] for r in quentes),
        "memoria_gb": round(ps.get("size", 0) / 1e9, 2),
        "memoria_vram_gb": round(ps.get("size_vram", 0) / 1e9, 2),
    }


def _json_ok(texto: str) -> bool:
    try:
        json.loads(texto)
        return True
    except json.JSONDecodeError:
        return False


def medir_embeddings(modelo: str, n: int = 8) -> dict:
    s = get_settings()
    textos = [PARAGRAFO * 3 + f" (trecho {i})" for i in range(n)]
    t0 = time.perf_counter()
    r = _post(f"{s.ollama_url}/api/embed", {"model": modelo, "input": textos})
    total = time.perf_counter() - t0
    return {
        "modelo": modelo,
        "dimensao": len(r["embeddings"][0]),
        "trechos": n,
        "total_s": round(total, 1),
        "s_por_trecho": round(total / n, 2),
        "carregamento_s": round(r.get("load_duration", 0) / NS, 1),
    }


def main() -> None:
    s = get_settings()
    modelos = sys.argv[1:] or [s.llm_modelo]
    saida = {"data": datetime.now().isoformat(timespec="seconds"), "top_k": s.top_k, "geracao": []}

    print(f"Ollama {_get(s.ollama_url + '/api/version')['version']} em {s.ollama_url}")
    print(f"\nEmbeddings: {s.embed_modelo}")
    saida["embeddings"] = medir_embeddings(s.embed_modelo)
    print(f"  {saida['embeddings']}")
    if saida["embeddings"]["dimensao"] != s.embed_dim:
        print(f"  ATENÇÃO: dimensão {saida['embeddings']['dimensao']} != EMBED_DIM={s.embed_dim}")

    for m in modelos:
        print(f"\nGeração: {m} (top_k={s.top_k}, num_ctx={s.num_ctx}, num_predict={s.num_predict})")
        saida["geracao"].append(medir_geracao(m, s.top_k))

    destino = RAIZ / "data/benchmark/resultados" / f"etapa0_{datetime.now():%Y-%m-%d}.json"
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSalvo em {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
