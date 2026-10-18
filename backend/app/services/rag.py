"""Peças do RAG: prompt, JSON Schema da saída e o Result devolvido a quem pergunta.

A orquestração (detecção -> busca -> geração -> validação, em rodadas) é um grafo do LangGraph
em services/graph.py. `responder` continua sendo a porta de entrada.
"""

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from app.config import Settings
from app.schemas.api import ErrorCode
from app.schemas.dataset import ROTULO_SECAO, Chunk
from app.services.corpus import Corpus

SISTEMA = (
    "Você responde perguntas de profissionais de farmácia sobre medicamentos consultando bulas. "
    "Use SOMENTE os trechos numerados enviados; não use conhecimento próprio. "
    "Responda em português, em até 4 frases curtas, sempre dizendo de qual medicamento é cada "
    "informação; cada frase diz o que o trecho diz, nunca só o nome do medicamento. "
    "Responda toda pergunta com o que os trechos dizem: se ela pede o melhor remédio "
    "para um sintoma, diga quais medicamentos dos trechos são indicados para ele (os trechos não "
    "dizem qual é o melhor); se fala de alergia a um remédio, diga o que os trechos dizem sobre "
    "hipersensibilidade e quais medicamentos dos trechos servem para o mesmo uso. Devolva só JSON: "
    '{"status": "respondida" ou "nao_encontrada", "afirmacoes": [{"texto": uma frase da resposta, '
    '"trecho": número do trecho de onde ela saiu}]}. '
    'Só devolva "status": "nao_encontrada" e "afirmacoes": [] se nenhum trecho tiver relação com a pergunta.'
)
# Lições do teste de 2026-10-13 com o qwen2.5:1.5b:
# - ele não escreve "[1]" dentro do texto, mesmo pedindo; por isso cada frase vem num objeto com
#   o seu trecho, e o "[n]" é montado aqui (ver _para_saida);
# - exemplo de resposta pronta no prompt é copiado como se fosse a resposta. Não coloque.
# 2026-10-16 (pedido do usuário: responder qualquer pergunta sobre remédio, com base nas bulas): saiu
# "não recomende dose nem conduta para um paciente específico", e a recusa ficou só para trechos sem
# relação nenhuma com a pergunta.


def formato(n_trechos: int) -> dict:
    """JSON Schema da saída: o Ollama só deixa o modelo citar trechos que foram enviados."""
    return {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["respondida", "nao_encontrada"]},
            "afirmacoes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "texto": {"type": "string"},
                        "trecho": {"type": "integer", "enum": list(range(1, n_trechos + 1))},
                    },
                    "required": ["texto", "trecho"],
                },
            },
        },
        "required": ["status", "afirmacoes"],
    }


@dataclass(frozen=True)
class Source:
    posicao: int  # o [n] do prompt
    chunk: Chunk
    data_versao: date
    score: float
    citado: bool = False
    # (início, fim) no texto do trecho das frases que sustentam a resposta (grounding.destaques).
    destaques: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class Attempt:
    """Uma rodada: o que foi enviado ao modelo, o que voltou e por que foi aceito ou não."""

    rodada: int
    chunk_ids: list[str]
    saida_bruta: str | None
    resultado: Literal["aprovada", "recusa", "reprovada"]
    falha: str | None = None  # só em "reprovada"
    verificacao_bruta: str | None = None  # saída do verificar_llm, só na rodada 3


@dataclass(frozen=True)
class Option:
    rotulo: str
    principios_ativos: list[str]


@dataclass(frozen=True)
class Clarification:
    """`opcoes` vazio = resposta livre: a pessoa responde no chat, e a resposta volta com a
    pergunta anterior (ver PreviousQuestion)."""

    motivo: Literal["medicamento_nao_reconhecido", "remedio_da_alergia"]
    pergunta: str
    opcoes: list[Option]


@dataclass(frozen=True)
class PreviousQuestion:
    """A pergunta anterior da mesma conversa, carregada do histórico pela API. A API continua sem
    estado: quem diz qual é a anterior é a tela (consulta_origem_id)."""

    pergunta: str
    principios_ativos: list[str]  # os usados na busca dela (detectados, herdados ou do filtro)
    pediu_esclarecimento: bool  # terminou perguntando algo de volta: a nova é a resposta


@dataclass
class Result:
    status: Literal["respondida", "nao_encontrada", "precisa_esclarecimento", "erro"]
    # sem_outra_explicacao: "não ajudou" sem outro trecho que responda, além dos já citados.
    motivo: Literal["sem_contexto", "medicamento_nao_indexado", "sem_outra_explicacao"] | None = None
    resposta: str | None = None
    fontes: list[Source] = field(default_factory=list)
    principios_ativos: list[str] = field(default_factory=list)  # filtro usado na busca
    medicamentos_nao_indexados: list[str] = field(default_factory=list)
    esclarecimento: Clarification | None = None
    codigo_erro: ErrorCode | None = None
    detalhe_erro: str | None = None
    saida_bruta: str | None = None  # o que o modelo devolveu na última rodada, para depurar
    rodadas: int = 0
    tentativas: list[Attempt] = field(default_factory=list)
    tempos_s: dict[str, float] = field(default_factory=dict)  # somados entre as rodadas
    prompt_tokens: int = 0  # somados entre as rodadas
    continuacao: bool = False  # usou a pergunta anterior (remédio herdado ou resposta a esclarecimento)
    alergia_a: str | None = None  # remédio da alergia: a resposta traz alternativas a ele
    extrativa: bool = False  # a resposta são frases copiadas da bula, sem o LLM (services/extract.py)
    # Pergunta de para que serve (services/indication.py): o veredito montado no código pela seção
    # de indicações ("Não. Ansiedade não está entre as indicações da bula de paracetamol...").
    conclusao: str | None = None
    outra_explicacao: bool = False  # refez uma resposta que não ajudou, sem os trechos dela


@dataclass(frozen=True)
class AnotherExplanation:
    """O balconista disse que a resposta não ajudou (docs/DECISOES.md, seção 26)."""

    evitar: list[str]  # chunk_ids citados na resposta que não ajudou (e nas explicações antes dela)
    enviados: list[str]  # chunk_ids consultados nela: continuam valendo, mas depois dos novos


def montar_prompt(
    pergunta: str, fontes: list[Source], observacao: str | None = None, anterior: str | None = None
) -> str:
    """`anterior`: a pergunta anterior da conversa, quando esta continua dela ("e para criança?")."""
    trechos = "\n\n".join(
        f"[{f.posicao}] {f.chunk.medicamento} | {ROTULO_SECAO[f.chunk.secao]} | página {f.chunk.pagina}\n{f.chunk.texto}"
        for f in fontes
    )
    extra = f"\n\nOBSERVAÇÃO: {observacao}" if observacao else ""
    antes = f"\n\nPERGUNTA ANTERIOR: {anterior}" if anterior else ""
    return f"TRECHOS:\n{trechos}{extra}{antes}\n\nPERGUNTA: {pergunta}"


def responder(
    pergunta: str,
    acervo: Corpus,
    principios_ativos: list[str] | None = None,
    s: Settings | None = None,
    anterior: PreviousQuestion | None = None,
    outra: AnotherExplanation | None = None,
) -> Result:
    """`principios_ativos` vazio = detectar o medicamento na própria pergunta.
    `anterior`: a pergunta anterior da conversa, se a tela mandou.
    `outra`: refazer uma resposta que não ajudou, sem os trechos que ela citou."""
    from app.services.graph import executar  # o grafo importa as peças deste módulo

    return executar(pergunta, acervo, principios_ativos, s, anterior, outra)


def afirmacoes(bruto: str) -> list[tuple[str, int]]:
    """As afirmações não vazias da saída do modelo, como (texto sem ponto final, trecho)."""
    try:
        dados = json.loads(bruto)
        pares = [(a["texto"].strip().rstrip("."), a["trecho"]) for a in dados["afirmacoes"]]
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return []
    return [(texto, trecho) for texto, trecho in pares if texto]


def _para_saida(bruto: str) -> str | dict:
    """Converte {"status", "afirmacoes"} no formato que validar_saida confere.

    Cada afirmação vira "frase [n]." Na recusa, o que o modelo escreveu a mais é descartado:
    modelo pequeno recusa e ainda explica, e a recusa é o que vale.
    """
    try:
        dados = json.loads(bruto)
        if dados["status"] == "nao_encontrada":
            return {"status": "nao_encontrada"}
        dados["afirmacoes"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return bruto  # validar_saida devolve SAIDA_INVALIDA
    frases = afirmacoes(bruto)
    return {
        "status": dados["status"],
        "resposta": " ".join(f"{texto} [{trecho}]." for texto, trecho in frases),
        "citacoes": sorted({trecho for _, trecho in frases}),
    }
