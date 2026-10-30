"""Benchmark do agente com o gabarito de data/benchmark/perguntas.jsonl (docs/DECISOES.md, seções 2, 30 e 32).

Uso (da raiz, com o banco e o Ollama no ar):
  python -m benchmark.run                              (modo busca, com o reranking)
  python -m benchmark.run --sem-rerank                 (modo busca, só semelhança: a linha de base)
  python -m benchmark.run --candidatos 10 25 50 100    (compara o número de candidatos do reranking)
  python -m benchmark.run --modo completo              (com o LLM de verdade: pesado, para a quinta)

Modo busca: o LLM é trocado por um que recusa, sem custo. Mede o que o agente recebe na rodada 1
(os trechos enviados ou copiados): acerto@k = todas as âncoras da pergunta estão nesses trechos. Mede
também os candidatos da busca vetorial (os mesmos que o reranking recebe), para separar "o banco
não trouxe" de "o reranking não escolheu".

Modo completo: status, resposta, rodadas, latência, recusa certa nas perguntas sem resposta e quantas
respostas tiveram a citação corrigida no código (seção 31) e, desde a seção 47, `resposta_confere` e
`resposta_completa` (benchmark/quality.py: polaridade, termos e números contra a resposta esperada; erro grosseiro,
não nota). A leitura humana da resposta continua valendo. Usa a semente fixa SEMENTE (LLM_SEMENTE do .env,
se houver): o mesmo prompt dá a mesma resposta, então rodar de novo dá o mesmo resultado.

Vários valores de --candidatos (RERANK_CANDIDATOS; 0 = sem reranking): roda cada um e compara com o
primeiro (a base). O número de candidatos só muda a rodada 1 das perguntas com remédio indexado; a
pergunta que recebe os mesmos trechos, na mesma ordem, recebe o mesmo prompt e a mesma resposta. Por
isso, no modo completo, o LLM só roda de novo nas perguntas cujos trechos da rodada 1 mudaram (as
outras reaproveitam a resposta da base, marcadas com "reaproveitada"). O embedding de cada pergunta é
calculado uma vez só: do segundo valor em diante, "segundos" não inclui o embedding.

Saída: tabela na tela e data/benchmark/resultados/<modo>_<data>_<rotulo>.json (um por valor) e, com
vários valores, comparacao_<modo>_<data>_<rotulo>.json.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from app.config import Settings, get_settings
from app.db.connection import criar_engine
from app.llm import ollama
from app.schemas.dataset import BenchmarkQuestion, Chunk
from app.services import rag
from app.services.corpus import Corpus
from benchmark.metrics import acertou_recall, chunk_cobre_ancora
from benchmark.quality import confere

PERGUNTAS = Path("data/benchmark/perguntas.jsonl")
RESULTADOS = Path("data/benchmark/resultados")
# Candidatos olhados por remédio quando o reranking está desligado (só para a métrica da âncora).
CANDIDATOS = 10
# Semente do modo completo (Settings.llm_semente), quando o .env não define uma.
SEMENTE = 42


def carregar(caminho: Path) -> list[BenchmarkQuestion]:
    return [BenchmarkQuestion.model_validate_json(l) for l in caminho.read_text(encoding="utf-8").splitlines() if l.strip()]


def taxa(v: list[bool]) -> str:
    return f"{sum(v)}/{len(v)}"


def _recusa(*args, **kwargs) -> ollama.LLMResponse:
    return ollama.LLMResponse('{"status": "nao_encontrada", "afirmacoes": []}', 0, 0, 0.0, 0.0)


@contextmanager
def _sem_llm() -> Iterator[None]:
    """O LLM trocado por um que recusa: a rodada 1 não depende dele, e a busca sai sem custo."""
    original = ollama.chat
    ollama.chat = _recusa
    try:
        yield
    finally:
        ollama.chat = original


def _embeddings_em_cache() -> None:
    """Cada texto vira vetor uma vez só (~2 s na CPU): com vários valores de candidatos, a mesma
    pergunta é respondida várias vezes."""
    original = ollama.embed
    cache: dict[tuple[str, ...], list[list[float]]] = {}

    def embed(textos: list[str], s: Settings | None = None) -> list[list[float]]:
        chave = tuple(textos)
        if chave not in cache:
            cache[chave] = original(textos, s)
        return cache[chave]

    ollama.embed = embed


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
    """Posição (1..N) do primeiro candidato que cobre uma âncora; None = nenhum dos N cobre."""
    for i, c in enumerate(candidatos, start=1):
        if any(chunk_cobre_ancora(c, a) for a in p.ancoras):
            return i
    return None


def candidatos_por_remedio(s: Settings) -> int:
    return s.rerank_candidatos or CANDIDATOS


def escolher_apresentacao(r: rag.Result, p: BenchmarkQuestion, acervo: Corpus) -> rag.Option:
    """A pergunta de apresentação (DECISOES §41), respondida como o balconista faria: a opção cuja bula
    tem a âncora do gabarito (a primeira, se nenhuma tem, ou se a pergunta não tem resposta)."""
    opcoes = r.esclarecimento.opcoes
    for o in opcoes:
        da_bula = [c for c in acervo.chunks.values() if c.bula_id == o.bula_id]
        if any(chunk_cobre_ancora(c, a) for c in da_bula for a in p.ancoras):
            return o
    return opcoes[0]


def responder(p: BenchmarkQuestion, acervo: Corpus, s: Settings, modo: str) -> dict:
    """Uma pergunta do gabarito: a linha do resultado. Se o agente pergunta qual apresentação, o
    benchmark escolhe como o balconista e mede a resposta (o tempo conta as duas)."""
    inicio = time.perf_counter()
    r = rag.responder(p.pergunta, acervo, s=s)
    apresentacao = None
    if r.status == "precisa_esclarecimento" and r.esclarecimento.motivo == "apresentacao":
        opcao = escolher_apresentacao(r, p, acervo)
        apresentacao = opcao.bula_id
        anterior = rag.PreviousQuestion(p.pergunta, r.principios_ativos, pediu_esclarecimento=True)
        r = rag.responder(p.pergunta, acervo, opcao.principios_ativos, s, anterior, bula_id=opcao.bula_id)
    segundos = time.perf_counter() - inicio
    nota = confere(r.resposta, p) if modo == "completo" and p.respondivel and r.status == "respondida" and r.resposta else None
    enviados = [acervo.chunks[c] for c in (r.tentativas[0].chunk_ids if r.tentativas else []) if c in acervo.chunks]
    candidatos: list[Chunk] = []
    if p.respondivel and r.principios_ativos:
        vetor = ollama.embed([p.pergunta], s)[0]
        k = candidatos_por_remedio(s) * len(r.principios_ativos)
        candidatos = [acervo.chunks[h.chunk_id] for h in acervo.indice.buscar(vetor, k, r.principios_ativos)]
    return {
        "id": p.id, "categoria": p.categoria, "respondivel": p.respondivel, "pergunta": p.pergunta,
        "caminho": caminho(r), "apresentacao_escolhida": apresentacao, "status": r.status, "rodadas": r.rodadas, "segundos": round(segundos, 2),
        "citacoes_corrigidas": sum(len(t.correcoes) for t in r.tentativas),
        "acerto": acertou_recall(enviados, p) if p.respondivel else None,
        "enviados": [c.chunk_id for c in enviados],
        "posicao_nos_candidatos": posicao_nos_candidatos(candidatos, p) if candidatos else None,
        "principios_ativos": r.principios_ativos,
        "conclusao": r.conclusao, "resposta": r.resposta if modo == "completo" else None,
        "resposta_esperada": p.resposta_esperada,
        # benchmark/quality.py: só nas respondíveis que o sistema respondeu, no modo completo.
        "confere": nota["confere"] if nota else None,
        "completa": nota["completa"] if nota else None,
        "problemas_da_resposta": nota["problemas"] if nota else None,
    }


def mostrar(linha: dict, mudou: bool = False) -> None:
    marca = "-" if not linha["respondivel"] else ("ok" if linha["acerto"] else "ERRO")
    aviso = (" (trechos mudaram)" if mudou else "") + (" (a resposta não confere)" if linha.get("confere") is False else "")
    print(f"{linha['id']:7} {marca:4} {linha['caminho']:14} {linha['segundos']:5.1f}s  {linha['pergunta'][:70]}{aviso}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Benchmark do agente pelo gabarito de âncoras")
    ap.add_argument("--modo", choices=["busca", "completo"], default="busca")
    ap.add_argument("--sem-rerank", action="store_true", help="linha de base: só semelhança na rodada 1 (= --candidatos 0)")
    ap.add_argument(
        "--candidatos", type=int, nargs="+", metavar="N",
        help="RERANK_CANDIDATOS a testar (0 = sem reranking); com vários, compara cada um com o primeiro",
    )
    ap.add_argument("--rotulo", default="")
    ap.add_argument("--perguntas", type=Path, default=PERGUNTAS)
    a = ap.parse_args()
    if a.sem_rerank and a.candidatos:
        ap.error("--sem-rerank é o mesmo que --candidatos 0: use um dos dois")

    s = get_settings()
    if a.modo == "completo" and s.llm_semente is None:
        s = s.model_copy(update={"llm_semente": SEMENTE})
    valores = list(dict.fromkeys(a.candidatos or ([0] if a.sem_rerank else [s.rerank_candidatos])))
    configs = {n: s.model_copy(update={"rerank_candidatos": n}) for n in valores}
    _embeddings_em_cache()
    acervo = Corpus.do_banco(criar_engine(s))
    perguntas = carregar(a.perguntas)
    base = valores[0]

    # Rodada 1 de cada valor, sem LLM: no modo busca, é o resultado; no completo com vários valores,
    # diz quais perguntas mudam (no completo com um valor só, não precisa).
    busca: dict[int, list[dict]] = {}
    if a.modo == "busca" or len(valores) > 1:
        for n in valores:
            if len(valores) > 1:
                print(f"\n== {n} candidatos por remédio (busca{', base' if n == base else ''}) ==")
            linhas = []
            with _sem_llm():
                for i, p in enumerate(perguntas):
                    linhas.append(responder(p, acervo, configs[n], "busca"))
                    mudou = n != base and linhas[-1]["enviados"] != busca[base][i]["enviados"]
                    if n == base or mudou:
                        mostrar(linhas[-1], mudou)
            busca[n] = linhas

    resultados = busca
    if a.modo == "completo":
        resultados = {}
        for n in valores:
            print(f"\n== {n} candidatos por remédio (completo, com o LLM) ==")
            linhas = []
            for i, p in enumerate(perguntas):
                if n != base and busca[n][i]["enviados"] == busca[base][i]["enviados"]:
                    # Mesmos trechos, na mesma ordem: mesmo prompt e, com a semente, a mesma resposta.
                    da_busca = {"posicao_nos_candidatos": busca[n][i]["posicao_nos_candidatos"]}
                    linhas.append(resultados[base][i] | da_busca | {"reaproveitada": True})
                    continue
                linhas.append(responder(p, acervo, configs[n], "completo"))
                mostrar(linhas[-1])
            resultados[n] = linhas

    RESULTADOS.mkdir(parents=True, exist_ok=True)
    data = f"{datetime.now():%Y-%m-%d_%H%M}{'_' + a.rotulo if a.rotulo else ''}"
    resumos = {}
    for n in valores:
        resumos[n] = resumir(resultados[n], a.modo, candidatos_por_remedio(configs[n]))
        sufixo = ("_sem-rerank" if n == 0 else "") + (f"_cand{n}" if len(valores) > 1 and n else "")
        gravar(RESULTADOS / f"{a.modo}_{data}{sufixo}.json", {
            "resumo": resumos[n],
            "config": {"top_k": s.top_k, "rerank_candidatos": n, "llm": s.llm_modelo, "llm_semente": s.llm_semente},
            "perguntas": resultados[n],
        })
    if len(valores) == 1:
        print(json.dumps(resumos[base], ensure_ascii=False, indent=1))
        return

    comparacoes = {n: comparar(resultados[base], resultados[n], a.modo) for n in valores[1:]}
    mostrar_comparacao(valores, resumos, comparacoes, a.modo)
    gravar(RESULTADOS / f"comparacao_{a.modo}_{data}.json", {
        "base": base, "valores": valores, "resumos": resumos, "comparacoes": comparacoes,
    })


def gravar(arquivo: Path, conteudo: dict) -> None:
    arquivo.write_text(json.dumps(conteudo, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"resultado em {arquivo}")


def resumir(linhas: list[dict], modo: str, candidatos: int = CANDIDATOS) -> dict:
    resp = [l for l in linhas if l["respondivel"]]
    por_cat, por_caminho = defaultdict(list), defaultdict(list)
    for l in resp:
        por_cat[l["categoria"]].append(l["acerto"])
        por_caminho[l["caminho"]].append(l["acerto"])

    com_candidatos = [l for l in resp if l["caminho"] in ("semelhanca",)]
    resumo = {
        "acerto_rodada1": taxa([l["acerto"] for l in resp]),
        "por_categoria": {c: taxa(v) for c, v in sorted(por_cat.items())},
        "por_caminho": {c: taxa(v) for c, v in sorted(por_caminho.items())},
        # Só nas perguntas que passaram pela semelhança: o que o reranking pode mudar.
        f"semelhanca_ancora_nos_{candidatos}_candidatos": taxa(
            [l["posicao_nos_candidatos"] is not None for l in com_candidatos]
        ),
        "semelhanca_ancora_nos_3_primeiros_candidatos": taxa(
            [(l["posicao_nos_candidatos"] or 99) <= 3 for l in com_candidatos]
        ),
        "segundos_mediana": sorted(l["segundos"] for l in linhas)[len(linhas) // 2],
    }
    if modo == "completo":
        sem = [l for l in linhas if not l["respondivel"]]
        resumo["recusa_certa"] = taxa([l["status"] == "nao_encontrada" for l in sem])
        resumo["respondidas"] = taxa([l["status"] == "respondida" for l in resp])
        dadas = [l for l in resp if l.get("confere") is not None]
        resumo["resposta_confere"] = taxa([l["confere"] for l in dadas])
        resumo["resposta_completa"] = taxa([l["completa"] for l in dadas])
        resumo["respostas_que_nao_conferem"] = [l["id"] for l in dadas if not l["confere"]]
        resumo["fora_do_escopo_respondidas"] = [l["id"] for l in sem if l["status"] == "respondida"]
        resumo["rodadas_media"] = round(sum(l["rodadas"] for l in resp) / max(1, len(resp)), 2)
        resumo["com_citacao_corrigida"] = taxa([l["citacoes_corrigidas"] > 0 for l in linhas])
    return resumo


def comparar(base: list[dict], linhas: list[dict], modo: str) -> dict:
    """O que mudou de um valor de candidatos para a base, pergunta por pergunta (mesma ordem)."""
    pares = list(zip(base, linhas))
    resp = [(b, l) for b, l in pares if b["respondivel"]]
    comparacao = {
        # Mesmos trechos na mesma ordem = mesmo prompt para o modelo.
        "mesmos_trechos_rodada1": taxa([b["enviados"] == l["enviados"] for b, l in pares]),
        "trechos_mudaram": [l["id"] for b, l in pares if b["enviados"] != l["enviados"]],
        "passou_a_acertar": [l["id"] for b, l in resp if l["acerto"] and not b["acerto"]],
        "deixou_de_acertar": [l["id"] for b, l in resp if b["acerto"] and not l["acerto"]],
    }
    if modo == "completo":
        def mesma(b: dict, l: dict) -> bool:
            return (b["status"], b["resposta"]) == (l["status"], l["resposta"])

        comparacao["mesma_resposta"] = taxa([mesma(b, l) for b, l in pares])
        comparacao["respostas_que_mudaram"] = [
            {"id": l["id"], "pergunta": l["pergunta"], "base": b["resposta"] or b["status"], "agora": l["resposta"] or l["status"]}
            for b, l in pares if not mesma(b, l)
        ]
    return comparacao


def mostrar_comparacao(valores: list[int], resumos: dict, comparacoes: dict, modo: str) -> None:
    base = valores[0]
    print(f"\nCOMPARAÇÃO COM {base} CANDIDATOS POR REMÉDIO")
    cabecalho = f"{'candidatos':>10} | {'acerto':>6} | {'âncora nos N':>12} | {'mesmos trechos':>14}"
    print(cabecalho + (f" | {'mesma resposta':>14}" if modo == "completo" else "") + " | passou a acertar | deixou de acertar")
    for n in valores:
        r = resumos[n]
        ancora = r[f"semelhanca_ancora_nos_{n or CANDIDATOS}_candidatos"]
        c = comparacoes.get(n)
        linha = f"{n:>10} | {r['acerto_rodada1']:>6} | {ancora:>12} | {c['mesmos_trechos_rodada1'] if c else 'base':>14}"
        if modo == "completo":
            linha += f" | {c['mesma_resposta'] if c else 'base':>14}"
        if c:
            linha += f" | {', '.join(c['passou_a_acertar']) or '-'} | {', '.join(c['deixou_de_acertar']) or '-'}"
        print(linha)


if __name__ == "__main__":
    main()
