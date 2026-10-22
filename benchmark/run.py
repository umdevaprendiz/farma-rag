"""Benchmark do agente com o gabarito de data/benchmark/perguntas.jsonl (docs/DECISOES.md, seções 2 e 30).

Uso (da raiz, com o banco e o Ollama no ar):
  python -m benchmark.run                         (modo busca, com o reranking)
  python -m benchmark.run --sem-rerank            (modo busca, só semelhança: a linha de base)
  python -m benchmark.run --modo completo         (com o LLM de verdade: pesado, para a quinta)

Modo busca: o LLM é trocado por um que recusa, sem custo. Mede o que o agente recebe na rodada 1
(os trechos enviados ou copiados): acerto@k = todas as âncoras da pergunta estão nesses trechos. Mede
também os 10 candidatos da busca vetorial (os mesmos que o reranking recebe), para separar "o banco
não trouxe" de "o reranking não escolheu".

Modo completo: status, resposta, rodadas, latência, recusa certa nas perguntas sem resposta e quantas
respostas tiveram a citação corrigida no código (docs/DECISOES.md, seção 31). A fidelidade da resposta continua sendo revisão manual (a resposta esperada está no gabarito).

Saída: tabela na tela e data/benchmark/resultados/<modo>_<data>_<rotulo>.json.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from app.config import get_settings
from app.db.connection import criar_engine
from app.llm import ollama
from app.schemas.dataset import BenchmarkQuestion, Chunk
from app.services import rag
from app.services.corpus import Corpus
from benchmark.metrics import acertou_recall, chunk_cobre_ancora

PERGUNTAS = Path("data/benchmark/perguntas.jsonl")
RESULTADOS = Path("data/benchmark/resultados")
CANDIDATOS = 10


def carregar(caminho: Path) -> list[BenchmarkQuestion]:
    return [BenchmarkQuestion.model_validate_json(l) for l in caminho.read_text(encoding="utf-8").splitlines() if l.strip()]


def _recusa(*args, **kwargs) -> ollama.LLMResponse:
    return ollama.LLMResponse('{"status": "nao_encontrada", "afirmacoes": []}', 0, 0, 0.0, 0.0)


def caminho(r: rag.Result) -> str:
    """Por onde a pergunta passou: o que explica o acerto ou o erro."""
    if r.status == "precisa_esclarecimento":
        return "esclarecimento"
    if r.conclusao is not None:
        return "indicacao" if r.fontes else "sem_bula"
    if r.extrativa:
        return "copiada"
    if not r.tentativas:
        return "sem_trechos"
    return "busca_geral" if not r.principios_ativos else "semelhanca"


def posicao_nos_candidatos(candidatos: list[Chunk], p: BenchmarkQuestion) -> int | None:
    """Posição (1..10) do primeiro candidato que cobre uma âncora; None = nenhum dos 10 cobre."""
    for i, c in enumerate(candidatos, start=1):
        if any(chunk_cobre_ancora(c, a) for a in p.ancoras):
            return i
    return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Benchmark do agente pelo gabarito de âncoras")
    ap.add_argument("--modo", choices=["busca", "completo"], default="busca")
    ap.add_argument("--sem-rerank", action="store_true", help="linha de base: só semelhança na rodada 1")
    ap.add_argument("--rotulo", default="")
    ap.add_argument("--perguntas", type=Path, default=PERGUNTAS)
    a = ap.parse_args()

    s = get_settings()
    if a.sem_rerank:
        s = s.model_copy(update={"rerank_candidatos": 0})
    if a.modo == "busca":
        ollama.chat = _recusa  # a busca não depende do LLM; a rodada 1 é a mesma
    acervo = Corpus.do_banco(criar_engine(s))
    perguntas = carregar(a.perguntas)

    linhas = []
    for p in perguntas:
        inicio = time.perf_counter()
        r = rag.responder(p.pergunta, acervo, s=s)
        segundos = time.perf_counter() - inicio
        enviados = [acervo.chunks[c] for c in (r.tentativas[0].chunk_ids if r.tentativas else []) if c in acervo.chunks]
        candidatos: list[Chunk] = []
        if p.respondivel and r.principios_ativos:
            vetor = ollama.embed([p.pergunta], s)[0]
            k = CANDIDATOS * len(r.principios_ativos)
            candidatos = [acervo.chunks[h.chunk_id] for h in acervo.indice.buscar(vetor, k, r.principios_ativos)]
        linhas.append({
            "id": p.id, "categoria": p.categoria, "respondivel": p.respondivel, "pergunta": p.pergunta,
            "caminho": caminho(r), "status": r.status, "rodadas": r.rodadas, "segundos": round(segundos, 2),
            "citacoes_corrigidas": sum(len(t.correcoes) for t in r.tentativas),
            "acerto": acertou_recall(enviados, p) if p.respondivel else None,
            "enviados": [f"{c.chunk_id}" for c in enviados],
            "posicao_nos_candidatos": posicao_nos_candidatos(candidatos, p) if candidatos else None,
            "principios_ativos": r.principios_ativos,
            "conclusao": r.conclusao, "resposta": r.resposta if a.modo == "completo" else None,
            "resposta_esperada": p.resposta_esperada,
        })
        marca = "-" if not p.respondivel else ("ok" if linhas[-1]["acerto"] else "ERRO")
        print(f"{p.id:7} {marca:4} {linhas[-1]['caminho']:14} {segundos:5.1f}s  {p.pergunta[:70]}")

    resumo = resumir(linhas, a.modo)
    print(json.dumps(resumo, ensure_ascii=False, indent=1))
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    nome = f"{a.modo}_{datetime.now():%Y-%m-%d_%H%M}{'_' + a.rotulo if a.rotulo else ''}{'_sem-rerank' if a.sem_rerank else ''}.json"
    (RESULTADOS / nome).write_text(
        json.dumps({"resumo": resumo, "config": {"top_k": s.top_k, "rerank_candidatos": s.rerank_candidatos,
                                                 "llm": s.llm_modelo}, "perguntas": linhas},
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"\nresultado em {RESULTADOS / nome}")


def resumir(linhas: list[dict], modo: str) -> dict:
    resp = [l for l in linhas if l["respondivel"]]
    por_cat, por_caminho = defaultdict(list), defaultdict(list)
    for l in resp:
        por_cat[l["categoria"]].append(l["acerto"])
        por_caminho[l["caminho"]].append(l["acerto"])

    def taxa(v: list[bool]) -> str:
        return f"{sum(v)}/{len(v)}"

    com_candidatos = [l for l in resp if l["caminho"] in ("semelhanca",)]
    resumo = {
        "acerto_rodada1": taxa([l["acerto"] for l in resp]),
        "por_categoria": {c: taxa(v) for c, v in sorted(por_cat.items())},
        "por_caminho": {c: taxa(v) for c, v in sorted(por_caminho.items())},
        # Só nas perguntas que passaram pela semelhança: o que o reranking pode mudar.
        "semelhanca_ancora_nos_10_candidatos": taxa([l["posicao_nos_candidatos"] is not None for l in com_candidatos]),
        "semelhanca_ancora_nos_3_primeiros_candidatos": taxa(
            [(l["posicao_nos_candidatos"] or 99) <= 3 for l in com_candidatos]
        ),
        "segundos_mediana": sorted(l["segundos"] for l in linhas)[len(linhas) // 2],
    }
    if modo == "completo":
        sem = [l for l in linhas if not l["respondivel"]]
        resumo["recusa_certa"] = taxa([l["status"] == "nao_encontrada" for l in sem])
        resumo["respondidas"] = taxa([l["status"] == "respondida" for l in resp])
        resumo["rodadas_media"] = round(sum(l["rodadas"] for l in resp) / max(1, len(resp)), 2)
        resumo["com_citacao_corrigida"] = taxa([l["citacoes_corrigidas"] > 0 for l in linhas])
    return resumo


if __name__ == "__main__":
    main()
