"""O agente como grafo do LangGraph: 7 nós, rodadas adaptativas (docs/DECISOES.md, seções 14 e 16).

    entender ─┬─► esclarecer ───────────────────────────────┐
              ├─► (não indexado / filtro desconhecido) ─────┤
              └─► buscar ─► gerar ─► validar ─┬─ ok ───────►├─► finalizar ─► END
                     ▲                        │             │
                     └── falhou, cabe outra ──┘             │
                         rodada 3 ok ─► verificar_llm ──────┘

Só `gerar` e `verificar_llm` chamam o LLM. As decisões ficam nas funções `depois_de_*`
(arestas condicionais), em código: com um modelo de 1,5B na CPU, cada chamada custa 30-45 s
e o modelo não é confiável para escolher caminho.
"""

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import Settings, get_settings
from app.db.busca_vetorial import Resultado as Achado
from app.llm import ollama
from app.schemas.api import MAX_MEDICAMENTOS_POR_PERGUNTA, CodigoErro
from app.services import fundamentacao
from app.services.acervo import Acervo
from app.services.citacoes import SaidaInvalidaError, validar_saida
from app.services.intencao import inferir_secoes
from app.services.medicamentos import detectar, normalizar
from app.services.rag import (
    SISTEMA,
    Esclarecimento,
    Fonte,
    Opcao,
    Resultado,
    Tentativa,
    _para_saida,
    afirmacoes,
    formato,
    montar_prompt,
)

# A verificação por LLM só entra a partir desta rodada (com RODADAS_MAX < 3, nunca entra).
RODADA_VERIFICACAO = 3
# Quantos candidatos por princípio ativo a busca traz antes de filtrar por seção e por
# trechos já usados. Uma bula tem algumas dezenas de trechos: 200 cobre a bula inteira.
K_AMPLO = 200

# O que a próxima rodada diz ao modelo sobre a falha anterior. Genérico de propósito: repetir
# a afirmação errada no prompt faz o modelo pequeno copiá-la de novo.
DICA_FORMATO = "A tentativa anterior veio em formato inválido. Devolva só o JSON pedido."
DICA_CITACAO = "A tentativa anterior citou trechos de forma incoerente. Cada afirmação deve indicar o trecho de onde saiu."
DICA_FUNDAMENTACAO = (
    "A tentativa anterior trouxe informação que não está escrita nos trechos. Use só números, doses "
    'e termos que estão nos trechos; se eles não respondem à pergunta, devolva "nao_encontrada".'
)

SISTEMA_VERIFICACAO = (
    "Você confere respostas sobre bulas. Para cada afirmação numerada, leia o trecho que vem com ela "
    'e responda "sim" se o trecho diz a mesma coisa, ou "nao" se o trecho não diz isso ou diz outra '
    "coisa. Não use conhecimento próprio. Devolva só JSON."
)


@dataclass(frozen=True)
class Falha:
    codigo: CodigoErro
    detalhe: str


class Estado(TypedDict, total=False):
    # Entrada
    pergunta: str
    filtro: list[str]
    inicio: float  # time.perf_counter() no começo, para o orçamento de tempo
    # entender
    principios_ativos: list[str]
    nao_indexados: list[str]
    secoes: list[str]  # seções que a pergunta indica (intencao.inferir_secoes)
    # Rodadas
    rodada: int
    vetor: list[float]  # embedding da pergunta, calculado uma vez só
    fontes: list[Fonte]  # os trechos da rodada atual
    usados: list[str]  # chunk_ids já enviados ao modelo em qualquer rodada
    saida_bruta: str | None
    falha: Falha | None  # o que reprovou a rodada atual (None = passou)
    dica: str | None  # o que a próxima geração vai ouvir sobre a falha
    recusou: bool  # o modelo disse que os trechos não respondem
    resposta: str | None
    citacoes: list[int]
    afirmacoes: list[tuple[str, int]]
    tentativas: list[Tentativa]
    tempos_s: dict[str, float]
    prompt_tokens: int
    # Desfechos que não passam por validar
    desfecho: Literal["erro", "nao_indexado", "esclarecer"]
    codigo_erro: CodigoErro
    detalhe_erro: str
    esclarecimento: Esclarecimento
    # Saída
    resultado: Resultado


def _erro(codigo: CodigoErro, detalhe: str) -> Estado:
    return {"desfecho": "erro", "codigo_erro": codigo, "detalhe_erro": detalhe}


def _somar_tempo(e: Estado, nome: str, inicio: float) -> dict[str, float]:
    tempos = dict(e["tempos_s"])
    tempos[nome] = tempos.get(nome, 0.0) + time.perf_counter() - inicio
    return tempos


def _cabe_outra_rodada(e: Estado, s: Settings) -> bool:
    """Ainda há rodada no limite e a próxima (estimada pela média das anteriores) cabe no orçamento."""
    if e["rodada"] >= s.rodadas_max:
        return False
    gasto = time.perf_counter() - e["inicio"]
    return gasto + gasto / e["rodada"] <= s.orcamento_s


def _secao_nao_consultada(e: Estado) -> bool:
    """A pergunta indica seção(ões) e nenhum trecho enviado era delas."""
    return bool(e["secoes"]) and not any(f.chunk.secao in e["secoes"] for f in e["fontes"])


def _explicar(e: Estado, s: Settings) -> str:
    """A mensagem do 502: o que cada rodada tentou e por que nada foi aceito."""
    partes = [
        f"rodada {t.rodada}: " + (t.falha if t.resultado == "reprovada" else "o modelo disse que os trechos não respondem")
        for t in e["tentativas"]
    ]
    parada = "limite de rodadas" if e["rodada"] >= s.rodadas_max else f"orçamento de {s.orcamento_s} s"
    return (
        f"Nenhuma resposta passou na verificação em {e['rodada']} rodada(s) ({parada}). "
        + "; ".join(partes)
    )


def montar_grafo(acervo: Acervo, s: Settings):
    # ---- Nós ---------------------------------------------------------------------------

    def entender(e: Estado) -> Estado:
        """Rodada 0, sem LLM: qual medicamento e qual seção."""
        inicial: Estado = {
            "secoes": inferir_secoes(e["pergunta"]), "rodada": 0, "usados": [], "tentativas": [],
            "tempos_s": {}, "prompt_tokens": 0, "principios_ativos": [], "nao_indexados": [],
        }
        if e["filtro"]:
            # O filtro aceita qualquer nome que o catálogo conheça ("losartana", "Cozaar").
            achados = {p: acervo.catalogo.indexados.get(normalizar(p)) for p in e["filtro"]}
            desconhecidos = [p for p, ativo in achados.items() if ativo is None]
            if desconhecidos:
                return inicial | _erro(
                    CodigoErro.MEDICAMENTO_NAO_ENCONTRADO, f"sem bula indexada para: {', '.join(desconhecidos)}"
                )
            return inicial | {"principios_ativos": sorted(set(achados.values()))}

        deteccao = detectar(e["pergunta"], acervo.catalogo)
        inicial |= {
            "principios_ativos": deteccao.indexados[:MAX_MEDICAMENTOS_POR_PERGUNTA],
            "nao_indexados": deteccao.nao_indexados,
        }
        if not deteccao.indexados and deteccao.nao_indexados:
            inicial["desfecho"] = "nao_indexado"
        return inicial

    def esclarecer(_: Estado) -> Estado:
        """Nenhum medicamento reconhecido e sem filtro: pergunta de volta, sem gastar o LLM."""
        nomes: dict[str, set[str]] = defaultdict(set)
        for bula in acervo.bulas.values():
            nomes[bula.principio_ativo].add(bula.medicamento)
        opcoes = [Opcao(f"{ativo} ({', '.join(sorted(meds))})", [ativo]) for ativo, meds in sorted(nomes.items())]
        return {
            "desfecho": "esclarecer",
            "esclarecimento": Esclarecimento("medicamento_nao_reconhecido", "Sobre qual medicamento é a pergunta?", opcoes),
        }

    def buscar(e: Estado) -> Estado:
        """Rodada 1: top_k por semelhança. Depois: trechos ainda não usados das seções indicadas."""
        rodada = e["rodada"] + 1
        t = time.perf_counter()
        try:
            vetor = e.get("vetor") or ollama.embed([e["pergunta"]], s)[0]
        except ollama.OllamaError as erro:
            return {"rodada": rodada} | _erro(erro.codigo, erro.detalhe)
        if rodada == 1:
            achados = acervo.indice.buscar(vetor, s.top_k, e["principios_ativos"])
        else:
            achados = _nova_busca(vetor, e)
        if s.limiar_similaridade is not None:
            achados = [a for a in achados if a.score >= s.limiar_similaridade]
        fontes = [
            Fonte(i, acervo.chunks[a.chunk_id], acervo.bulas[acervo.chunks[a.chunk_id].bula_id].data_versao, a.score)
            for i, a in enumerate(achados, start=1)
        ]
        novos = [f.chunk.chunk_id for f in fontes if f.chunk.chunk_id not in e["usados"]]
        return {
            "rodada": rodada, "vetor": vetor, "fontes": fontes, "usados": e["usados"] + novos,
            "tempos_s": _somar_tempo(e, "busca", t),
        }

    def _nova_busca(vetor: list[float], e: Estado) -> list[Achado]:
        """Com seção indicada: só trechos dessa seção, os ainda não usados primeiro (o trecho
        certo pode ter vindo na rodada 1 e o erro ter sido do modelo). Sem seção: trechos ainda
        não usados. Sem nada novo: os melhores de novo (a rodada ainda muda a observação ao modelo)."""
        
        ativos = e["principios_ativos"]
        candidatos = acervo.indice.buscar(vetor, K_AMPLO * len(ativos), ativos)
        novos = [a for a in candidatos if a.chunk_id not in e["usados"]]
        da_secao = [a for a in candidatos if acervo.chunks[a.chunk_id].secao in e["secoes"]]
        # sorted é estável: dentro de cada grupo (novo / já usado) continua a ordem de score.
        escolhidos = sorted(da_secao, key=lambda a: a.chunk_id in e["usados"]) or novos or candidatos
        # Mesma divisão do k entre princípios ativos que o índice faz na rodada 1.
        k_por_ativo = max(1, math.ceil(s.top_k / len(ativos)))
        por_ativo: dict[str, list[Achado]] = defaultdict(list)
        for a in escolhidos:  # já vêm em ordem de score
            ativo = normalizar(acervo.chunks[a.chunk_id].principio_ativo)
            if len(por_ativo[ativo]) < k_por_ativo:
                por_ativo[ativo].append(a)
        juntos = sorted((a for lista in por_ativo.values() for a in lista), key=lambda a: -a.score)
        return juntos[: max(s.top_k, len(ativos))]

    def gerar(e: Estado) -> Estado:
        t = time.perf_counter()
        prompt = montar_prompt(e["pergunta"], e["fontes"], e.get("dica"))
        try:
            llm = ollama.chat(SISTEMA, prompt, formato(len(e["fontes"])), s)
        except ollama.OllamaError as erro:
            return _erro(erro.codigo, erro.detalhe)
        return {
            "saida_bruta": llm.texto, "prompt_tokens": e["prompt_tokens"] + llm.prompt_tokens,
            "tempos_s": _somar_tempo(e, "geracao", t),
        }

    def validar(e: Estado) -> Estado:
        """Sem LLM: formato e citações (validar_saida), depois a fundamentação de cada afirmação."""
        bruto, fontes = e["saida_bruta"], e["fontes"]
        tentativa = {"rodada": e["rodada"], "chunk_ids": [f.chunk.chunk_id for f in fontes], "saida_bruta": bruto}
        limpo: Estado = {"falha": None, "recusou": False, "resposta": None, "citacoes": [], "afirmacoes": []}

        def reprovar(codigo: CodigoErro, detalhe: str, dica: str) -> Estado:
            return limpo | {
                "falha": Falha(codigo, detalhe), "dica": dica,
                "tentativas": e["tentativas"] + [Tentativa(**tentativa, resultado="reprovada", falha=detalhe)],
            }

        try:
            saida = validar_saida(_para_saida(bruto), len(fontes))
        except SaidaInvalidaError as erro:
            dica = DICA_FORMATO if erro.codigo == CodigoErro.SAIDA_INVALIDA else DICA_CITACAO
            return reprovar(erro.codigo, erro.detalhe, dica)
        if saida.status == "nao_encontrada":
            return limpo | {"recusou": True, "tentativas": e["tentativas"] + [Tentativa(**tentativa, resultado="recusa")]}

        frases = afirmacoes(bruto)
        problemas = fundamentacao.checar(frases, {f.posicao: f.chunk for f in fontes}, s.fundamentacao_min_termos)
        if problemas:
            return reprovar(CodigoErro.RESPOSTA_NAO_VERIFICADA, "; ".join(problemas), DICA_FUNDAMENTACAO)
        return limpo | {
            "resposta": saida.resposta, "citacoes": saida.citacoes, "afirmacoes": frases,
            "tentativas": e["tentativas"] + [Tentativa(**tentativa, resultado="aprovada")],
        }

    def verificar_llm(e: Estado) -> Estado:
        """Rodada 3: o LLM confere, afirmação por afirmação, se o trecho citado diz aquilo."""
        chunks = {f.posicao: f.chunk for f in e["fontes"]}
        frases = e["afirmacoes"]
        usuario = "\n\n".join(
            f"AFIRMAÇÃO {i}: {texto}\nTRECHO: {chunks[trecho].texto}" for i, (texto, trecho) in enumerate(frases, start=1)
        )
        # Uma chave por afirmação, todas obrigatórias: o modelo não pode pular nenhuma.
        chaves = [f"a{i}" for i in range(1, len(frases) + 1)]
        esquema = {
            "type": "object",
            "properties": {c: {"type": "string", "enum": ["sim", "nao"]} for c in chaves},
            "required": chaves,
        }
        t = time.perf_counter()
        try:
            llm = ollama.chat(SISTEMA_VERIFICACAO, usuario, esquema, s, s.llm_modelo_verificacao)
        except ollama.OllamaError as erro:
            return _erro(erro.codigo, erro.detalhe)
        try:
            votos = json.loads(llm.texto)
            negadas = [i for i, c in enumerate(chaves, start=1) if votos.get(c) != "sim"]
        except (json.JSONDecodeError, AttributeError):
            negadas = list(range(1, len(chaves) + 1))

        anteriores, ultima = e["tentativas"][:-1], e["tentativas"][-1]
        atualizacao: Estado = {
            "prompt_tokens": e["prompt_tokens"] + llm.prompt_tokens,
            "tempos_s": _somar_tempo(e, "verificacao", t),
        }
        if not negadas:
            return atualizacao | {"tentativas": anteriores + [replace(ultima, verificacao_bruta=llm.texto)]}
        detalhe = f"a verificação não confirmou a(s) afirmação(ões) {', '.join(map(str, negadas))}"
        return atualizacao | {
            "falha": Falha(CodigoErro.RESPOSTA_NAO_VERIFICADA, detalhe), "dica": DICA_FUNDAMENTACAO,
            "tentativas": anteriores + [replace(ultima, resultado="reprovada", falha=detalhe, verificacao_bruta=llm.texto)],
        }

    def finalizar(e: Estado) -> Estado:
        r = Resultado(
            status="nao_encontrada",
            principios_ativos=e.get("principios_ativos", []),
            medicamentos_nao_indexados=e.get("nao_indexados", []),
            fontes=e.get("fontes", []),
            saida_bruta=e.get("saida_bruta"),
            rodadas=e.get("rodada", 0),
            tentativas=e.get("tentativas", []),
            tempos_s=e.get("tempos_s", {}),
            prompt_tokens=e.get("prompt_tokens", 0),
        )
        desfecho = e.get("desfecho")
        if desfecho == "erro":
            r.status, r.codigo_erro, r.detalhe_erro = "erro", e["codigo_erro"], e["detalhe_erro"]
        elif desfecho == "nao_indexado":
            r.motivo = "medicamento_nao_indexado"
        elif desfecho == "esclarecer":
            r.status, r.esclarecimento = "precisa_esclarecimento", e["esclarecimento"]
        elif e.get("falha"):
            # Decisão de 2026-10-13: rodadas esgotadas viram erro 502 com explicação.
            r.status, r.codigo_erro, r.detalhe_erro = "erro", e["falha"].codigo, _explicar(e, s)
        elif e.get("resposta"):
            r.status, r.resposta = "respondida", e["resposta"]
            r.fontes = [replace(f, citado=f.posicao in e["citacoes"]) for f in r.fontes]
        else:
            r.motivo = "sem_contexto"  # recusa do modelo ou nenhum trecho acima do limiar
        return {"resultado": r}

    # ---- Arestas condicionais (as decisões) -------------------------------------------

    def depois_de_entender(e: Estado) -> str:
        if e.get("desfecho"):
            return "finalizar"
        return "buscar" if e["principios_ativos"] else "esclarecer"

    def depois_de_buscar(e: Estado) -> str:
        return "gerar" if e.get("desfecho") != "erro" and e["fontes"] else "finalizar"

    def depois_de_gerar(e: Estado) -> str:
        return "finalizar" if e.get("desfecho") == "erro" else "validar"

    def depois_de_validar(e: Estado) -> str:
        if e["falha"]:
            return "buscar" if _cabe_outra_rodada(e, s) else "finalizar"
        if e["recusou"]:
            # Recusar com trechos da seção errada é falha de busca, não falta de informação:
            # vale uma rodada com a seção que a pergunta indica, uma vez só.
            if e["rodada"] == 1 and _secao_nao_consultada(e) and _cabe_outra_rodada(e, s):
                return "buscar"
            return "finalizar"
        return "verificar_llm" if e["rodada"] >= RODADA_VERIFICACAO else "finalizar"

    def depois_de_verificar(e: Estado) -> str:
        if e.get("desfecho") != "erro" and e["falha"] and _cabe_outra_rodada(e, s):
            return "buscar"
        return "finalizar"

    g = StateGraph(Estado)
    for nome, no in [
        ("entender", entender), ("esclarecer", esclarecer), ("buscar", buscar), ("gerar", gerar),
        ("validar", validar), ("verificar_llm", verificar_llm), ("finalizar", finalizar),
    ]:
        g.add_node(nome, no)
    g.add_edge(START, "entender")
    g.add_conditional_edges("entender", depois_de_entender, ["esclarecer", "buscar", "finalizar"])
    g.add_edge("esclarecer", "finalizar")
    g.add_conditional_edges("buscar", depois_de_buscar, ["gerar", "finalizar"])
    g.add_conditional_edges("gerar", depois_de_gerar, ["validar", "finalizar"])
    g.add_conditional_edges("validar", depois_de_validar, ["buscar", "verificar_llm", "finalizar"])
    g.add_conditional_edges("verificar_llm", depois_de_verificar, ["buscar", "finalizar"])
    g.add_edge("finalizar", END)
    return g.compile()


def executar(
    pergunta: str, acervo: Acervo, principios_ativos: list[str] | None = None, s: Settings | None = None
) -> Resultado:
    s = s or get_settings()
    final = montar_grafo(acervo, s).invoke(
        {"pergunta": pergunta, "filtro": principios_ativos or [], "inicio": time.perf_counter()}
    )
    return final["resultado"]
