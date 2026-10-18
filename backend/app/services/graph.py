"""O agente como grafo do LangGraph: 6 nós, rodadas adaptativas (docs/DECISOES.md, seções 14, 16 e 20).

    entender ─┬─► (filtro desconhecido) ────────────────────┐
              └─► buscar ─► gerar ─► validar ─┬─ ok ───────►├─► finalizar ─► END
                     ▲                        │             │
                     └── falhou, cabe outra ──┘             │
                         rodada 3 ok ─► verificar_llm ──────┘

`buscar` tem três caminhos: com medicamento indexado na pergunta (ou no filtro), nas bulas dele;
sem, em todas as bulas, um trecho por princípio ativo (busca geral: "melhor remédio pra dor de
cabeça", remédio sem bula no acervo); e alergia a um remédio ("sou alérgico a dipirona, tem
outro?"): a contraindicação dele e remédios para o mesmo uso. Toda pergunta segue para a resposta:
a recusa só vem quando nenhum trecho tem relação com ela.

Para que serve (docs/DECISOES.md, seção 29): "paracetamol serve para ansiedade?" é respondida pela
seção de indicações, sem LLM: as frases dela e a conclusão montada no código (citado, geral ou
ausente: o que não está nas indicações não é indicado pela bula). Remédio sem bula: a classe da
ANVISA, sem busca e sem concluir.

Conversa (docs/DECISOES.md, seção 23): a API recebe a pergunta anterior (`PreviousQuestion`). Sem remédio
na pergunta nova ("e para criança?"), continua sobre o da anterior; "alergia a um remédio" sem dizer
qual pergunta de volta, e a resposta da pessoa chega junto com a pergunta que pediu o esclarecimento.

"Não ajudou" (docs/DECISOES.md, seção 26): o balconista pede outra explicação (`AnotherExplanation`). A
pergunta roda de novo sem os trechos citados na resposta que não ajudou; os enviados e não citados
ainda valem, depois dos nunca enviados. Sem frases copiadas da bula (a cópia pode ter sido o que não
ajudou). Sem trecho novo, "não há outra explicação", sem chamar o LLM.

Só `gerar` e `verificar_llm` chamam o LLM. As decisões ficam nas funções `depois_de_*`
(arestas condicionais), em código: com um modelo de 1,5B na CPU, cada chamada custa 30-45 s
e o modelo não é confiável para escolher caminho.
"""

import json
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import Settings, get_settings
from app.db.vector_search import SearchHit
from app.llm import ollama
from app.schemas.api import MAX_MEDICAMENTOS_POR_PERGUNTA, ErrorCode
from app.schemas.dataset import Chunk
from app.services import grounding, indication, rerank
from app.services.citations import InvalidOutputError, validar_saida
from app.services.corpus import Corpus
from app.services.extract import INTERACAO, PERGUNTA_DE_DOSE, frase_da_alternativa, frases_de_dose, frases_sobre
from app.services.intent import (
    eliptica,
    grupos_do_assunto,
    inferir_secoes,
    menciona,
    pede_remedio,
    responde_ao_assunto,
    secoes_para_rerank,
    termos_do_assunto,
    texto_apos_alergia,
)
from app.services.medicines import Suggestion, detectar, formas_do_nome, normalizar, sugerir
from app.services.rag import (
    SISTEMA,
    AnotherExplanation,
    Attempt,
    Clarification,
    Option,
    PreviousQuestion,
    Result,
    Source,
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
# Regra 3 da fundamentação: o modelo pequeno lista "Tylalgin Sachet [1]. Ibuprofeno [2]." e repete
# isso em toda rodada se o aviso for o genérico.
DICA_ASSUNTO = (
    "A tentativa anterior não respondeu ao que foi perguntado. Responda exatamente à pergunta, com a "
    "frase do trecho que fala disso; se nenhum trecho fala disso, devolva \"nao_encontrada\"."
)
DICA_SO_NOME = (
    "A tentativa anterior respondeu só com nomes de medicamentos. Cada afirmação deve ser uma frase "
    "que diz, com as palavras do trecho, o que o trecho diz sobre aquele medicamento."
)
# Regra 5 da fundamentação: "não é indicado para dor no ouvido" citando um trecho que diz "é indicado".
DICA_NEGACAO = (
    "A tentativa anterior negou o que o trecho não nega. Escreva só o que o trecho diz, sem acrescentar "
    "\"não\"; se nenhum trecho responde à pergunta, devolva \"nao_encontrada\"."
)

# Pergunta de para que serve: quantas frases das indicações, no total (no máximo 2 por bula).
MAX_FRASES_INDICACAO = 4

# Alternativas por semelhança (sem remédio da mesma classe no acervo): só as que ficam a até esta
# distância da melhor. Calibrar com o benchmark.
MARGEM_PARECIDOS = 0.05
PERGUNTA_REMEDIO = "Sobre qual remédio é a pergunta? Me diga o nome."
PERGUNTA_ALERGIA = "A alergia é a qual remédio? Me diga o nome, que eu procuro nas bulas o que serve para o mesmo uso."


def esclarecer_nome(parecidos: list[Suggestion]) -> Clarification:
    """"Tomei rocutan...": nenhum remédio reconhecido, mas "rocutan" parece Roacutan no registro da
    ANVISA. Pergunta em vez de adivinhar; cada opção reenvia a pergunta com o princípio ativo como
    filtro (docs/DECISOES.md, seção 28)."""
    rotulos = [f"{p.nome} ({p.principio_ativo})" for p in parecidos]
    qual = f"É o {rotulos[0]}?" if len(rotulos) == 1 else "É um destes?"
    return Clarification(
        "medicamento_nao_reconhecido",
        f'Não reconheci "{parecidos[0].palavra}". {qual} Se não for, escreva o nome do remédio.',
        [Option(r, [p.principio_ativo]) for r, p in zip(rotulos, parecidos)],
    )

SISTEMA_VERIFICACAO = (
    "Você confere respostas sobre bulas. Para cada afirmação numerada, leia o trecho que vem com ela "
    'e responda "sim" se o trecho diz a mesma coisa, ou "nao" se o trecho não diz isso ou diz outra '
    "coisa. Não use conhecimento próprio. Devolva só JSON."
)


@dataclass(frozen=True)
class Failure:
    codigo: ErrorCode
    detalhe: str


class State(TypedDict, total=False):
    # Entrada
    pergunta: str
    filtro: list[str]
    inicio: float  # time.perf_counter() no começo, para o orçamento de tempo
    anterior: PreviousQuestion | None  # a pergunta anterior da conversa, se a tela mandou
    outra: AnotherExplanation | None  # "não ajudou": refazer sem os trechos citados (seção 26)
    # entender
    texto: str  # o que vai para a busca: a pergunta, ou a anterior + a nova quando continua
    assunto: list[str]  # palavras que o trecho precisa ter na busca geral (termos_do_assunto)
    continuacao: bool
    principios_ativos: list[str]
    nao_indexados: list[str]
    secoes: list[str]  # seções que a pergunta indica (intent.inferir_secoes)
    alergia_a: str | None  # remédio da alergia: a busca traz alternativas a ele
    uso: str | None  # pergunta de para que serve: o uso perguntado ("" = geral); None = outra pergunta
    conclusao: str | None  # o veredito da pergunta de uso (services/indication.py)
    # Rodadas
    rodada: int
    vetor: list[float]  # embedding da pergunta, calculado uma vez só
    vetor_uso: list[float] | None  # embedding do uso do remédio da alergia (indicação ou classe)
    extrato: list[tuple[int, str]]  # (trecho, frase copiada da bula): a resposta sem o LLM
    extrativa: bool  # a resposta desta rodada veio do extrato
    fontes: list[Source]  # os trechos da rodada atual
    usados: list[str]  # chunk_ids já enviados ao modelo em qualquer rodada (e na resposta que não ajudou)
    saida_bruta: str | None
    falha: Failure | None  # o que reprovou a rodada atual (None = passou)
    dica: str | None  # o que a próxima geração vai ouvir sobre a falha
    recusou: bool  # o modelo disse que os trechos não respondem
    resposta: str | None
    citacoes: list[int]
    afirmacoes: list[tuple[str, int]]
    tentativas: list[Attempt]
    tempos_s: dict[str, float]
    prompt_tokens: int
    # Fim antes das rodadas: erro (filtro desconhecido, Ollama fora do ar) ou pergunta de volta
    desfecho: Literal["erro", "esclarecer"]
    codigo_erro: ErrorCode
    detalhe_erro: str
    esclarecimento: Clarification
    # Saída
    resultado: Result


def _erro(codigo: ErrorCode, detalhe: str) -> State:
    return {"desfecho": "erro", "codigo_erro": codigo, "detalhe_erro": detalhe}


def _sem_os_citados(achados: list[SearchHit], e: State) -> list[SearchHit]:
    """"Não ajudou": os trechos citados na resposta que não ajudou ficam fora da busca."""
    if not (outra := e.get("outra")):
        return achados
    evitar = set(outra.evitar)
    return [a for a in achados if a.chunk_id not in evitar]


def _citada(f: Source, afirmacoes: list[tuple[str, int]]) -> Source:
    """A fonte citada na resposta, com as frases do trecho que sustentam cada afirmação dela."""
    deste = [texto for texto, trecho in afirmacoes if trecho == f.posicao]
    return replace(f, citado=True, destaques=tuple(grounding.destaques(deste, f.chunk)))


def _somar_tempo(e: State, nome: str, inicio: float) -> dict[str, float]:
    tempos = dict(e["tempos_s"])
    tempos[nome] = tempos.get(nome, 0.0) + time.perf_counter() - inicio
    return tempos


def _cabe_outra_rodada(e: State, s: Settings) -> bool:
    """Ainda há rodada no limite e a próxima (estimada pela média das anteriores) cabe no orçamento."""
    if e["rodada"] >= s.rodadas_max:
        return False
    gasto = time.perf_counter() - e["inicio"]
    return gasto + gasto / e["rodada"] <= s.orcamento_s


def _secao_nao_consultada(e: State) -> bool:
    """A pergunta indica seção(ões) e nenhum trecho enviado era delas."""
    return bool(e["secoes"]) and not any(f.chunk.secao in e["secoes"] for f in e["fontes"])


def _explicar(e: State, s: Settings) -> str:
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


def montar_grafo(acervo: Corpus, s: Settings):
    # ---- Nós ---------------------------------------------------------------------------

    def entender(e: State) -> State:
        """Rodada 0, sem LLM: qual medicamento, qual seção e se a pergunta continua a anterior."""
        anterior: PreviousQuestion | None = e.get("anterior")
        pergunta = e["pergunta"]
        # Resposta a um esclarecimento ("A alergia é a qual remédio?" -> "dipirona"): vale a
        # pergunta que pediu o esclarecimento junto com a resposta.
        responde = anterior is not None and anterior.pediu_esclarecimento
        # O botão de uma opção reenvia a mesma pergunta (com o filtro): não repetir o texto.
        texto = f"{anterior.pergunta} {pergunta}" if responde and pergunta != anterior.pergunta else pergunta
        inicial: State = {
            "texto": texto, "assunto": termos_do_assunto(texto), "secoes": inferir_secoes(texto),
            "continuacao": responde, "alergia_a": None, "vetor_uso": None, "rodada": 0, "uso": None, "conclusao": None,
            "usados": list(e["outra"].enviados) if e.get("outra") else [],
            "tentativas": [], "tempos_s": {}, "prompt_tokens": 0, "principios_ativos": [], "nao_indexados": [],
        }
        if e["filtro"]:
            # O filtro aceita qualquer nome que o catálogo conheça ("losartana", "Cozaar"). Remédio
            # conhecido sem bula ("isotretinoína", escolhido no "é o Roacutan?") segue como na
            # pergunta escrita: busca nas bulas que falam dele (seção 28; antes era 404).
            achados = {p: acervo.catalogo.resolver(p) for p in e["filtro"]}
            desconhecidos = [p for p, nomes in achados.items() if not any(nomes)]
            if desconhecidos:
                return inicial | _erro(
                    ErrorCode.MEDICAMENTO_NAO_ENCONTRADO, f"medicamento desconhecido: {', '.join(desconhecidos)}"
                )
            return inicial | {
                "principios_ativos": sorted({a for a, _ in achados.values() if a}),
                "nao_indexados": sorted({n for _, n in achados.values() if n}),
                # "Tomei rocutan para dor no ouvido" -> "É o Roacutan?" -> o botão reenvia a pergunta.
                "uso": indication.uso_perguntado(pergunta) if len(achados) == 1 else None,
            }

        # Sem medicamento indexado (nenhum nome, ou só remédio sem bula), principios_ativos fica
        # vazio e a busca vai a todas as bulas.
        deteccao = detectar(texto, acervo.catalogo)
        ativos, nao_indexados = deteccao.indexados[:MAX_MEDICAMENTOS_POR_PERGUNTA], deteccao.nao_indexados
        if not responde and not ativos and not nao_indexados and (parecidos := sugerir(pergunta, acervo.catalogo)):
            # Antes da conversa: "e o rocutan?" não é continuação da pergunta anterior.
            return inicial | {"desfecho": "esclarecer", "esclarecimento": esclarecer_nome(parecidos)}
        continua = False
        if anterior and not responde and not pede_remedio(pergunta):
            if not ativos and not nao_indexados and (anterior.principios_ativos or eliptica(pergunta)):
                # "Pode na gravidez?" depois de uma pergunta sobre paracetamol: continua sobre ele. Se
                # a anterior foi geral ("melhor remédio pra dor de cabeça"), só continua a pergunta que
                # começa com "e"/"mas" ("e para criança?").
                ativos, continua = anterior.principios_ativos[:MAX_MEDICAMENTOS_POR_PERGUNTA], True
            elif eliptica(pergunta) and anterior.principios_ativos:
                # "E com varfarina?" depois de "omeprazol pode com clopidogrel?": a pergunta é sobre
                # omeprazol e varfarina, não só sobre a varfarina.
                ativos = list(dict.fromkeys(anterior.principios_ativos + ativos))[:MAX_MEDICAMENTOS_POR_PERGUNTA]
                continua = True
        if continua:
            inicial |= {
                "texto": f"{anterior.pergunta} {pergunta}", "continuacao": True,
                "secoes": inferir_secoes(pergunta) or inferir_secoes(anterior.pergunta),
                "assunto": termos_do_assunto(anterior.pergunta),
            }
        estado = inicial | {"principios_ativos": ativos, "nao_indexados": nao_indexados}
        if anterior is None and not ativos and not nao_indexados and eliptica(pergunta):
            # "E para criança?" sem pergunta anterior: não há de quem falar. Pergunta de volta, em vez
            # de buscar "criança" em todas as bulas (respondia com a frase de outro remédio).
            return estado | {
                "desfecho": "esclarecer",
                "esclarecimento": Clarification("medicamento_nao_reconhecido", PERGUNTA_REMEDIO, []),
            }

        if (depois := texto_apos_alergia(inicial["texto"])) is None:
            # Pergunta de para que serve; "e para dor de cabeça?" depois de uma, também. Só com um
            # remédio: "toma losartana, pode tomar ibuprofeno para dor?" é de uso junto (seção 30).
            if len(ativos) + len(nao_indexados) != 1:
                return estado
            return estado | {"uso": indication.uso_perguntado(pergunta, anterior.pergunta if continua else None)}
        # Na resposta ao esclarecimento, o remédio está na mensagem nova ("Exemplol", "é a dipirona").
        alvo = detectar(pergunta if responde else depois, acervo.catalogo)
        nomes = alvo.indexados + alvo.nao_indexados
        if not nomes:
            # "Tenho alergia a um remédio, tem outro?": pergunta qual, uma vez só (se a pessoa já
            # respondeu e não deu nome, segue com a busca geral). Com remédio na pergunta, ela é
            # sobre ele: "alérgico a penicilina, pode amoxicilina?", "rinite alérgica com coriza,
            # fexofenadina serve?" (benchmark, seção 30).
            if responde or ativos or nao_indexados:
                return estado
            return estado | {
                "desfecho": "esclarecer",
                "esclarecimento": Clarification("remedio_da_alergia", PERGUNTA_ALERGIA, []),
            }
        outros = [a for a in ativos if a not in alvo.indexados]
        if outros:
            # "Sou alérgico a dipirona, posso tomar paracetamol?": a pergunta é sobre o outro.
            return estado | {"principios_ativos": outros}
        return estado | {"alergia_a": nomes[0], "principios_ativos": []}

    def buscar(e: State) -> State:
        """Com medicamento: rodada 1 por semelhança, depois os trechos ainda não usados das seções
        indicadas. Sem medicamento: busca geral em todas as bulas.

        "Não ajudou": já começa como uma rodada seguinte (a resposta que não ajudou foi a primeira),
        sem os trechos citados nela."""
        rodada = e["rodada"] + 1
        t = time.perf_counter()
        # Os caminhos da primeira tentativa: frases copiadas da bula e o top-k direto.
        primeira = rodada == 1 and not e.get("outra")
        uso = e.get("uso")
        if primeira and uso is not None and not e["principios_ativos"] and e["nao_indexados"]:
            # Para que serve um remédio sem bula: as outras bulas só o citam de passagem (numa
            # interação), e isso não diz para que ele serve. A classe da ANVISA, sem busca nem LLM.
            return {
                "rodada": rodada, "fontes": [], "extrato": [], "tempos_s": _somar_tempo(e, "busca", t),
                "conclusao": indication.conclusao(_vereditos_sem_bula(e)),
            }
        vetor_uso = e.get("vetor_uso")
        try:
            vetor = e.get("vetor") or ollama.embed([e["texto"]], s)[0]
            if e.get("alergia_a") and vetor_uso is None and (uso := _uso_do_remedio(e["alergia_a"])):
                vetor_uso = ollama.embed([uso], s)[0]
        except ollama.OllamaError as erro:
            return {"rodada": rodada} | _erro(erro.codigo, erro.detalhe)
        extrato: list[tuple[str, str]] = []  # (chunk_id, frase): só na primeira; depois, o LLM tenta
        conclusao: str | None = None
        if e.get("alergia_a"):
            achados = _busca_alternativas(vetor, vetor_uso, e)
            if primeira:
                extrato = [(a.chunk_id, frase_da_alternativa(acervo.chunks[a.chunk_id])) for a in achados]
        elif primeira and uso is not None and (indicacao := _frases_da_indicacao(vetor, e)):
            achados, extrato, conclusao = indicacao
        elif primeira and (interacao := _frases_da_interacao(vetor, e)):
            achados, extrato = interacao
        elif primeira and (dose := _frases_da_dose(vetor, e)):
            achados, extrato = dose
        elif not e["principios_ativos"]:
            achados = _busca_geral(vetor, e)
        elif primeira:
            achados = _primeira_busca(vetor, e)
        else:
            achados = _nova_busca(vetor, e)
        if s.limiar_similaridade is not None:
            achados = [a for a in achados if a.score >= s.limiar_similaridade]
        fontes = [
            Source(i, acervo.chunks[a.chunk_id], acervo.bulas[acervo.chunks[a.chunk_id].bula_id].data_versao, a.score)
            for i, a in enumerate(achados, start=1)
        ]
        novos = [f.chunk.chunk_id for f in fontes if f.chunk.chunk_id not in e["usados"]]
        posicao = {f.chunk.chunk_id: f.posicao for f in fontes}
        return {
            "rodada": rodada, "vetor": vetor, "vetor_uso": vetor_uso, "fontes": fontes,
            "usados": e["usados"] + novos, "tempos_s": _somar_tempo(e, "busca", t),
            "extrato": [(posicao[c], frase) for c, frase in extrato if c in posicao],
            # Só vale com o extrato desta rodada; numa rodada seguinte, o LLM responde sem veredito.
            "conclusao": conclusao,
        }

    def _vereditos_sem_bula(e: State) -> list[indication.Verdict]:
        return [
            indication.Verdict(indication.nome_curto(n), e["uso"], "sem_bula", acervo.catalogo.classe_de(n))
            for n in e["nao_indexados"]
        ]

    def _frases_da_indicacao(
        vetor: list[float], e: State
    ) -> tuple[list[SearchHit], list[tuple[str, str]], str | None] | None:
        """"Paracetamol serve para ansiedade?": as frases da seção de indicações de cada remédio
        (services/indication.py), no máximo 2 por bula e MAX_FRASES_INDICACAO no total, e a conclusão
        (citado, geral, ausente). Frase repetida entre bulas do mesmo remédio entra uma vez. None =
        nenhum remédio da pergunta tem a seção de indicações (segue o caminho normal)."""
        uso = e["uso"]
        vereditos: list[indication.Verdict] = []
        escolhidas: list[tuple[Chunk, str]] = []
        for ativo in e["principios_ativos"]:
            alvo = normalizar(ativo)
            da_secao = sorted(
                (c for c in acervo.chunks.values() if c.secao == "indicacoes" and normalizar(c.principio_ativo) == alvo),
                key=lambda c: c.chunk_id,
            )
            if not da_secao:
                continue
            sit = indication.situacao(uso, [c.texto for c in da_secao]) if uso else "lista"
            # O nome como a bula escreve (o filtro chega normalizado: "isotretinoina").
            vereditos.append(indication.Verdict(indication.nome_curto(da_secao[0].principio_ativo), uso, sit))
            vistas: set[str] = set()
            por_bula: dict[str, list[Chunk]] = defaultdict(list)
            for c in da_secao:
                por_bula[c.bula_id].append(c)
            for chunks_da_bula in por_bula.values():
                da_bula = [(c, f) for c in chunks_da_bula for f in indication.frases(uso, c, sit)]
                novas = [(c, f) for c, f in da_bula if normalizar(f) not in vistas][:2]
                vistas |= {normalizar(f) for _, f in novas}
                escolhidas += novas
        if not escolhidas:
            return None
        escolhidas = escolhidas[:MAX_FRASES_INDICACAO]
        # Mais de uma bula na resposta: cada frase diz de qual é ("Tylalgin 750 mg: ...").
        varias = len({c.bula_id for c, _ in escolhidas}) > 1
        extrato = [(c.chunk_id, f"{c.medicamento}: {f}" if varias else f) for c, f in escolhidas]
        ativos = e["principios_ativos"]
        semelhanca = {a.chunk_id: a.score for a in acervo.indice.buscar(vetor, K_AMPLO * len(ativos), ativos)}
        trechos = list(dict.fromkeys(c.chunk_id for c, _ in escolhidas))
        conclusao = indication.conclusao(vereditos + _vereditos_sem_bula(e))
        return [SearchHit(c, semelhanca.get(c, 0.0)) for c in trechos], extrato, conclusao

    def _frases_da_dose(vetor: list[float], e: State) -> tuple[list[SearchHit], list[tuple[str, str]]] | None:
        """"Qual a dose máxima de dipirona?": as frases da posologia com número e unidade, uma por
        bula (services/extract.py). None = não é pergunta de dose sobre remédio do acervo, ou a
        posologia não tem frase que fale do pedido."""
        ativos = e["principios_ativos"]
        if not ativos or not PERGUNTA_DE_DOSE.search(normalizar(e["texto"])):
            return None
        alvos = {normalizar(a) for a in ativos}
        das_bulas = [c for c in acervo.chunks.values() if normalizar(c.principio_ativo) in alvos]
        frases = frases_de_dose(das_bulas, _termos_da_pergunta(e))
        if not frases:
            return None
        semelhanca = {a.chunk_id: a.score for a in acervo.indice.buscar(vetor, K_AMPLO * len(ativos), ativos)}
        return [SearchHit(c.chunk_id, semelhanca.get(c.chunk_id, 0.0)) for c, _ in frases], [(c.chunk_id, f) for c, f in frases]

    def _frases_da_interacao(vetor: list[float], e: State) -> tuple[list[SearchHit], list[tuple[str, str]]] | None:
        """"Omeprazol pode com clopidogrel?": o que a bula de cada remédio citado diz do outro
        (services/extract.py). None = a pergunta não é de uso junto, ou nenhuma bula cita o outro."""
        ativos = e["principios_ativos"]
        if not ativos or not INTERACAO.search(normalizar(e["texto"])):
            return None
        frases = []
        for ativo in ativos:
            da_bula = [c for c in acervo.chunks.values() if normalizar(c.principio_ativo) == normalizar(ativo)]
            for outro in [a for a in ativos if a != ativo] + e["nao_indexados"]:
                frases += frases_sobre(da_bula, outro)
        if not frases:
            return None
        semelhanca = {a.chunk_id: a.score for a in acervo.indice.buscar(vetor, K_AMPLO * len(ativos), ativos)}
        trechos = list(dict.fromkeys(chunk.chunk_id for chunk, _ in frases))
        return [SearchHit(c, semelhanca.get(c, 0.0)) for c in trechos], [(chunk.chunk_id, f) for chunk, f in frases]

    def _primeira_busca(vetor: list[float], e: State) -> list[SearchHit]:
        """Rodada 1 com remédio indexado: os `rerank_candidatos` mais parecidos de cada remédio,
        mais os melhores das seções que a pergunta pede, e o reranking escolhe os top_k
        (services/rerank.py). `rerank_candidatos=0`: só a semelhança (a linha de base)."""
        ativos = e["principios_ativos"]
        if not s.rerank_candidatos:
            return acervo.indice.buscar(vetor, s.top_k, ativos)
        secoes = secoes_para_rerank(e["texto"], len(ativos) + len(e["nao_indexados"]))
        candidatos = acervo.indice.buscar(vetor, s.rerank_candidatos * len(ativos), ativos)
        if secoes:
            vistos: dict[tuple[str, str], int] = defaultdict(int)
            for a in acervo.indice.buscar(vetor, K_AMPLO * len(ativos), ativos):  # já em ordem de score
                c = acervo.chunks[a.chunk_id]
                chave = (normalizar(c.principio_ativo), c.secao)
                if c.secao in secoes and vistos[chave] < rerank.EXTRA_POR_SECAO:
                    vistos[chave] += 1
                    candidatos.append(a)
        return rerank.reordenar(candidatos, acervo.chunks, secoes, _grupos_da_pergunta(e), s.top_k, ativos)

    def _grupos_da_pergunta(e: State) -> list[tuple[str, ...]]:
        """Os assuntos da pergunta nova sem os que são só nome: do princípio ativo, da marca ou do
        fabricante ("Nisoflan", "Geolab" não dizem do que a pergunta trata)."""
        alvos = {normalizar(a) for a in e["principios_ativos"]}
        das_bulas = [
            nome for b in acervo.bulas.values() if normalizar(b.principio_ativo) in alvos
            for nome in (b.medicamento, b.fabricante, *b.nomes_comerciais)
        ]
        nomes = [*e["principios_ativos"], *e["nao_indexados"], *das_bulas]
        palavras = set(re.findall(r"[a-z]{3,}", normalizar(" ".join(nomes))))
        return [g for g in grupos_do_assunto(e["pergunta"]) if not all(any(p.startswith(t) for p in palavras) for t in g)]

    def _nova_busca(vetor: list[float], e: State) -> list[SearchHit]:
        """Com seção indicada: só trechos dessa seção, os ainda não usados primeiro (o trecho
        certo pode ter vindo na rodada 1 e o erro ter sido do modelo). Sem seção: trechos ainda
        não usados. Sem nada novo: os melhores de novo (a rodada ainda muda a observação ao modelo).

        "Não ajudou": sem os trechos citados. Se a bula tem a seção da pergunta e todos os trechos dela
        já foram citados, nenhum (não há outra explicação; o resto da bula não fala disso)."""

        ativos = e["principios_ativos"]
        todos = acervo.indice.buscar(vetor, K_AMPLO * len(ativos), ativos)
        tem_secao = any(acervo.chunks[a.chunk_id].secao in e["secoes"] for a in todos)
        candidatos = _sem_os_citados(todos, e)
        novos = [a for a in candidatos if a.chunk_id not in e["usados"]]
        da_secao = [a for a in candidatos if acervo.chunks[a.chunk_id].secao in e["secoes"]]
        if tem_secao and not da_secao:
            return []
        # sorted é estável: dentro de cada grupo (novo / já usado) continua a ordem de score.
        escolhidos = sorted(da_secao, key=lambda a: a.chunk_id in e["usados"]) or novos or candidatos
        # Mesma divisão do k entre princípios ativos que o índice faz na rodada 1.
        k_por_ativo = max(1, math.ceil(s.top_k / len(ativos)))
        por_ativo: dict[str, list[SearchHit]] = defaultdict(list)
        for a in escolhidos:  # já vêm em ordem de score
            ativo = normalizar(acervo.chunks[a.chunk_id].principio_ativo)
            if len(por_ativo[ativo]) < k_por_ativo:
                por_ativo[ativo].append(a)
        juntos = sorted((a for lista in por_ativo.values() for a in lista), key=lambda a: -a.score)
        return juntos[: max(s.top_k, len(ativos))]

    def _busca_geral(vetor: list[float], e: State) -> list[SearchHit]:
        """Todas as bulas, só nas seções que a pergunta indica (se indica alguma), só trechos que
        falam do assunto, e um trecho por princípio ativo: cada um é uma opção na resposta, em vez
        de três trechos do mesmo remédio. Da rodada 2 em diante, os ainda não usados vêm primeiro.
        No "não ajudou", os citados saem depois de escolhida a seção: sem trecho novo nela, não
        passa a valer a seção errada (o remédio que causa tontura como se a tratasse).

        "Fala do assunto" (busca híbrida: semelhança + palavra): se a pergunta cita remédio sem bula,
        o trecho tem de mencioná-lo, em qualquer seção (os da seção da pergunta primeiro); senão, ter
        uma palavra do assunto ("tonto" -> tontura, vertigem) dentro da seção da pergunta, para não
        listar o remédio que causa tontura como se a tratasse. Nenhum trecho assim = nenhuma bula fala
        disso, e a resposta é "não encontrei", sem chamar o LLM."""
        candidatos = acervo.indice.buscar(vetor, len(acervo.chunks))  # já em ordem de score

        def na_secao(a: SearchHit) -> bool:
            return acervo.chunks[a.chunk_id].secao in e["secoes"]

        def texto(a: SearchHit) -> str:
            return normalizar(acervo.chunks[a.chunk_id].texto)

        if nomes := [f for nome in e["nao_indexados"] for f in formas_do_nome(nome)]:
            candidatos = sorted((a for a in candidatos if menciona(texto(a), nomes)), key=lambda a: not na_secao(a))
        else:
            if e["secoes"]:
                candidatos = [a for a in candidatos if na_secao(a)] or candidatos
            if e["assunto"]:
                candidatos = [a for a in candidatos if menciona(texto(a), e["assunto"])]
        # Os ainda não usados primeiro (na primeira rodada, nenhum foi usado). sorted é estável: cada
        # grupo continua na ordem de score.
        candidatos = sorted(_sem_os_citados(candidatos, e), key=lambda a: a.chunk_id in e["usados"])
        escolhidos: dict[str, SearchHit] = {}
        for a in candidatos:
            escolhidos.setdefault(normalizar(acervo.chunks[a.chunk_id].principio_ativo), a)
            if len(escolhidos) == s.top_k_geral:
                break
        return sorted(escolhidos.values(), key=lambda a: -a.score)

    def _uso_do_remedio(nome: str) -> str | None:
        """Para que o remédio serve: a indicação da bula dele, ou a classe terapêutica da ANVISA."""
        alvo = normalizar(nome)
        indicacao = next(
            (c.texto for c in acervo.chunks.values() if c.secao == "indicacoes" and normalizar(c.principio_ativo) == alvo),
            None,
        )
        if indicacao:
            return indicacao
        classe = acervo.catalogo.classe_de(nome)
        return f"Medicamento da classe {classe.lower()}" if classe else None

    def _busca_alternativas(vetor: list[float], vetor_uso: list[float] | None, e: State) -> list[SearchHit]:
        """Alergia a um remédio: a contraindicação da bula dele (se está no acervo), depois a indicação
        de outros remédios, um por princípio ativo.

        Alternativas: os remédios do acervo da mesma classe terapêutica no registro da ANVISA (dipirona
        e ibuprofeno são "analgésicos não narcóticos": entra o paracetamol). Só eles, quando há: outro
        anti-inflamatório para quem tem alergia a um anti-inflamatório é arriscado (alergia cruzada).
        Sem nenhum da mesma classe, as indicações mais parecidas com o uso do remédio, só as próximas
        da melhor (a semelhança sozinha trazia o albendazol para a dipirona). O score das
        alternativas é a semelhança com o uso do remédio, não com a pergunta. No "não ajudou", sem os
        trechos citados: vem outro trecho da contraindicação e da indicação, se a bula tiver."""
        nome = e["alergia_a"]
        alvo = normalizar(nome)

        def da_secao(referencia: list[float], ativo: str, secao: str) -> SearchHit | None:
            achados = _sem_os_citados(acervo.indice.buscar(referencia, K_AMPLO, [ativo]), e)
            return next((a for a in achados if acervo.chunks[a.chunk_id].secao == secao), None)

        achados = [a] if (a := da_secao(vetor, alvo, "contraindicacoes")) else []
        referencia = vetor_uso if vetor_uso is not None else vetor
        vagas = s.top_k_geral - len(achados)

        def indicacao(ativo: str) -> SearchHit | None:
            return da_secao(referencia, ativo, "indicacoes")

        classe = acervo.catalogo.classe_de(nome)
        do_acervo = sorted({normalizar(b.principio_ativo) for b in acervo.bulas.values()} - {alvo})
        irmaos = [a for p in do_acervo if classe and acervo.catalogo.classe_de(p) == classe and (a := indicacao(p))]
        if irmaos:
            return achados + sorted(irmaos, key=lambda a: -a.score)[:vagas]
        if vetor_uso is None:
            return achados
        parecidos: dict[str, SearchHit] = {}
        for a in _sem_os_citados(acervo.indice.buscar(vetor_uso, len(acervo.chunks)), e):
            chunk = acervo.chunks[a.chunk_id]
            if chunk.secao == "indicacoes" and normalizar(chunk.principio_ativo) != alvo:
                parecidos.setdefault(normalizar(chunk.principio_ativo), a)
        melhores = sorted(parecidos.values(), key=lambda a: -a.score)
        return achados + [a for a in melhores[:vagas] if a.score >= melhores[0].score - MARGEM_PARECIDOS]

    def gerar(e: State) -> State:
        t = time.perf_counter()
        if e.get("extrato"):
            # Frases copiadas da bula (services/extract.py): o modelo não entra.
            saida = {"status": "respondida", "afirmacoes": [{"texto": f, "trecho": p} for p, f in e["extrato"]]}
            return {"saida_bruta": json.dumps(saida, ensure_ascii=False), "extrativa": True, "tempos_s": _somar_tempo(e, "geracao", t)}
        avisos = [e["dica"]] if e.get("dica") else []
        if nome := e.get("alergia_a"):
            da_bula = any(normalizar(f.chunk.principio_ativo) == normalizar(nome) for f in e["fontes"])
            avisos.insert(0, (
                f"A pessoa tem alergia a {nome}." + (f" O trecho [1] é da bula de {nome}." if da_bula else "")
                + " Os outros trechos são de medicamentos para um uso parecido: diga para que cada um é indicado."
            ))
        if e["nao_indexados"]:
            # Na busca geral os trechos são de outros remédios: o modelo não pode atribuir a um
            # remédio sem bula o que a bula de outro diz.
            avisos.insert(0, (
                f"Não há trechos da bula de {', '.join(e['nao_indexados'])}. Não atribua a "
                f"{'eles' if len(e['nao_indexados']) > 1 else 'ele'} o que a bula de outro medicamento diz."
            ))
        anterior = e["anterior"].pergunta if e.get("continuacao") and e.get("anterior") else None
        prompt = montar_prompt(e["pergunta"], e["fontes"], " ".join(avisos) or None, anterior)
        try:
            llm = ollama.chat(SISTEMA, prompt, formato(len(e["fontes"])), s)
        except ollama.OllamaError as erro:
            return _erro(erro.codigo, erro.detalhe)
        return {
            "saida_bruta": llm.texto, "prompt_tokens": e["prompt_tokens"] + llm.prompt_tokens,
            "tempos_s": _somar_tempo(e, "geracao", t), "extrativa": False,
        }

    def validar(e: State) -> State:
        """Sem LLM: formato e citações (validar_saida), depois a fundamentação de cada afirmação."""
        bruto, fontes = e["saida_bruta"], e["fontes"]
        tentativa = {"rodada": e["rodada"], "chunk_ids": [f.chunk.chunk_id for f in fontes], "saida_bruta": bruto}
        limpo: State = {"falha": None, "recusou": False, "resposta": None, "citacoes": [], "afirmacoes": []}

        def reprovar(codigo: ErrorCode, detalhe: str, dica: str) -> State:
            return limpo | {
                "falha": Failure(codigo, detalhe), "dica": dica,
                "tentativas": e["tentativas"] + [Attempt(**tentativa, resultado="reprovada", falha=detalhe)],
            }

        try:
            saida = validar_saida(_para_saida(bruto), len(fontes))
        except InvalidOutputError as erro:
            dica = DICA_FORMATO if erro.codigo == ErrorCode.SAIDA_INVALIDA else DICA_CITACAO
            return reprovar(erro.codigo, erro.detalhe, dica)
        if saida.status == "nao_encontrada":
            return limpo | {"recusou": True, "tentativas": e["tentativas"] + [Attempt(**tentativa, resultado="recusa")]}

        frases = afirmacoes(bruto)
        problemas = grounding.checar(
            frases, {f.posicao: f.chunk for f in fontes}, s.fundamentacao_min_termos, acervo.catalogo,
            permitidos=[e["alergia_a"]] if e.get("alergia_a") else None,
        )
        if problemas:
            if all(p.endswith(grounding.SO_NOME) for p in problemas):
                dica = DICA_SO_NOME
            elif all(p.endswith(grounding.NEGA) for p in problemas):
                dica = DICA_NEGACAO
            else:
                dica = DICA_FUNDAMENTACAO
            return reprovar(ErrorCode.RESPOSTA_NAO_VERIFICADA, "; ".join(problemas), dica)
        termos = _termos_da_pergunta(e)
        citadas = {f.chunk.secao for f in fontes if f.posicao in saida.citacoes}
        if not e.get("extrativa") and termos and not responde_ao_assunto([f for f, _ in frases], termos, citadas, e["secoes"]):
            detalhe = f"a resposta não fala do que foi perguntado ({', '.join(termos[:5])})"
            return reprovar(ErrorCode.RESPOSTA_NAO_VERIFICADA, detalhe, DICA_ASSUNTO)
        return limpo | {
            "resposta": saida.resposta, "citacoes": saida.citacoes, "afirmacoes": frases,
            "tentativas": e["tentativas"] + [Attempt(**tentativa, resultado="aprovada")],
        }

    def _termos_da_pergunta(e: State) -> list[str]:
        """O assunto da pergunta (a nova, não a anterior), sem os nomes de remédio: "dose máxima de
        dipirona?" -> dose, maxim. A resposta tem de falar disso (intent.responde_ao_assunto)."""
        nomes = [*e["principios_ativos"], *e["nao_indexados"], *(f.chunk.medicamento for f in e.get("fontes", []))]
        palavras = set(re.findall(r"[a-z]{3,}", normalizar(" ".join(nomes))))
        return [t for t in termos_do_assunto(e["pergunta"]) if not any(p.startswith(t) for p in palavras)]

    def verificar_llm(e: State) -> State:
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
        atualizacao: State = {
            "prompt_tokens": e["prompt_tokens"] + llm.prompt_tokens,
            "tempos_s": _somar_tempo(e, "verificacao", t),
        }
        if not negadas:
            return atualizacao | {"tentativas": anteriores + [replace(ultima, verificacao_bruta=llm.texto)]}
        detalhe = f"a verificação não confirmou a(s) afirmação(ões) {', '.join(map(str, negadas))}"
        return atualizacao | {
            "falha": Failure(ErrorCode.RESPOSTA_NAO_VERIFICADA, detalhe), "dica": DICA_FUNDAMENTACAO,
            "tentativas": anteriores + [replace(ultima, resultado="reprovada", falha=detalhe, verificacao_bruta=llm.texto)],
        }

    def finalizar(e: State) -> State:
        r = Result(
            status="nao_encontrada",
            principios_ativos=e.get("principios_ativos", []),
            medicamentos_nao_indexados=e.get("nao_indexados", []),
            fontes=e.get("fontes", []),
            saida_bruta=e.get("saida_bruta"),
            rodadas=e.get("rodada", 0),
            tentativas=e.get("tentativas", []),
            tempos_s=e.get("tempos_s", {}),
            prompt_tokens=e.get("prompt_tokens", 0),
            continuacao=e.get("continuacao", False),
            alergia_a=e.get("alergia_a"),
            extrativa=bool(e.get("resposta")) and e.get("extrativa", False),
            outra_explicacao=bool(e.get("outra")),
        )
        # O veredito da pergunta de uso vale com a resposta copiada das indicações, ou sozinho (remédio
        # sem bula, sem trechos). Resposta do LLM numa rodada seguinte não leva veredito.
        if (r.extrativa or not r.fontes) and not e.get("desfecho") and not e.get("falha"):
            r.conclusao = e.get("conclusao")
        if e.get("desfecho") == "erro":
            r.status, r.codigo_erro, r.detalhe_erro = "erro", e["codigo_erro"], e["detalhe_erro"]
        elif e.get("desfecho") == "esclarecer":
            r.status, r.esclarecimento = "precisa_esclarecimento", e["esclarecimento"]
        elif e.get("falha"):
            # Decisão de 2026-10-13: rodadas esgotadas viram erro 502 com explicação.
            r.status, r.codigo_erro, r.detalhe_erro = "erro", e["falha"].codigo, _explicar(e, s)
        elif e.get("resposta"):
            r.status, r.resposta = "respondida", e["resposta"]
            r.fontes = [_citada(f, e["afirmacoes"]) if f.posicao in e["citacoes"] else f for f in r.fontes]
        elif e.get("outra"):
            # Nenhum trecho além dos citados, ou o modelo disse que os outros não respondem.
            r.motivo = "sem_outra_explicacao"
        elif e.get("nao_indexados") and not e.get("principios_ativos"):
            # A busca geral não achou nada sobre o remédio sem bula: a falta é da bula dele.
            r.motivo = "medicamento_nao_indexado"
        else:
            r.motivo = "sem_contexto"  # recusa do modelo ou nenhum trecho acima do limiar
        return {"resultado": r}

    # ---- Arestas condicionais (as decisões) -------------------------------------------

    def depois_de_entender(e: State) -> str:
        return "finalizar" if e.get("desfecho") else "buscar"

    def depois_de_buscar(e: State) -> str:
        return "gerar" if e.get("desfecho") != "erro" and e["fontes"] else "finalizar"

    def depois_de_gerar(e: State) -> str:
        return "finalizar" if e.get("desfecho") == "erro" else "validar"

    def depois_de_validar(e: State) -> str:
        if e["falha"]:
            return "buscar" if _cabe_outra_rodada(e, s) else "finalizar"
        if e["recusou"]:
            # Recusar com trechos da seção errada é falha de busca, não falta de informação:
            # vale uma rodada com a seção que a pergunta indica, uma vez só.
            if e["rodada"] == 1 and _secao_nao_consultada(e) and _cabe_outra_rodada(e, s):
                return "buscar"
            return "finalizar"
        return "verificar_llm" if e["rodada"] >= RODADA_VERIFICACAO else "finalizar"

    def depois_de_verificar(e: State) -> str:
        if e.get("desfecho") != "erro" and e["falha"] and _cabe_outra_rodada(e, s):
            return "buscar"
        return "finalizar"

    g = StateGraph(State)
    for nome, no in [
        ("entender", entender), ("buscar", buscar), ("gerar", gerar), ("validar", validar),
        ("verificar_llm", verificar_llm), ("finalizar", finalizar),
    ]:
        g.add_node(nome, no)
    g.add_edge(START, "entender")
    g.add_conditional_edges("entender", depois_de_entender, ["buscar", "finalizar"])
    g.add_conditional_edges("buscar", depois_de_buscar, ["gerar", "finalizar"])
    g.add_conditional_edges("gerar", depois_de_gerar, ["validar", "finalizar"])
    g.add_conditional_edges("validar", depois_de_validar, ["buscar", "verificar_llm", "finalizar"])
    g.add_conditional_edges("verificar_llm", depois_de_verificar, ["buscar", "finalizar"])
    g.add_edge("finalizar", END)
    return g.compile()


def executar(
    pergunta: str,
    acervo: Corpus,
    principios_ativos: list[str] | None = None,
    s: Settings | None = None,
    anterior: PreviousQuestion | None = None,
    outra: AnotherExplanation | None = None,
) -> Result:
    s = s or get_settings()
    final = montar_grafo(acervo, s).invoke({
        "pergunta": pergunta, "filtro": principios_ativos or [], "inicio": time.perf_counter(),
        "anterior": anterior, "outra": outra,
    })
    return final["resultado"]
