"""Peças do RAG: prompt, JSON Schema da saída e o Resultado devolvido a quem pergunta.

A orquestração (detecção -> busca -> geração -> validação, em rodadas) é um grafo do LangGraph
em services/grafo.py. `responder` continua sendo a porta de entrada.
"""

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from app.config import Settings
from app.schemas.api import CodigoErro
from app.schemas.dataset import ROTULO_SECAO, Chunk
from app.services.acervo import Acervo

SISTEMA = (
    "Você responde perguntas de profissionais de farmácia consultando bulas. "
    "Use SOMENTE os trechos numerados enviados; não use conhecimento próprio. "
    "Responda em português, em até 4 frases curtas. Não recomende dose nem conduta para um "
    "paciente específico. Devolva só JSON: "
    '{"status": "respondida" ou "nao_encontrada", "afirmacoes": [{"texto": uma frase da resposta, '
    '"trecho": número do trecho de onde ela saiu}]}. '
    'Se os trechos não respondem à pergunta, devolva "status": "nao_encontrada" e "afirmacoes": [].'
)
# Lições do teste de 2026-10-13 com o qwen2.5:1.5b:
# - ele não escreve "[1]" dentro do texto, mesmo pedindo; por isso cada frase vem num objeto com
#   o seu trecho, e o "[n]" é montado aqui (ver _para_saida);
# - exemplo de resposta pronta no prompt é copiado como se fosse a resposta. Não coloque.


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
class Fonte:
    posicao: int  # o [n] do prompt
    chunk: Chunk
    data_versao: date
    score: float
    citado: bool = False


@dataclass(frozen=True)
class Tentativa:
    """Uma rodada: o que foi enviado ao modelo, o que voltou e por que foi aceito ou não."""

    rodada: int
    chunk_ids: list[str]
    saida_bruta: str | None
    resultado: Literal["aprovada", "recusa", "reprovada"]
    falha: str | None = None  # só em "reprovada"
    verificacao_bruta: str | None = None  # saída do verificar_llm, só na rodada 3


@dataclass(frozen=True)
class Opcao:
    rotulo: str
    principios_ativos: list[str]


@dataclass(frozen=True)
class Esclarecimento:
    motivo: Literal["medicamento_nao_reconhecido"]
    pergunta: str
    opcoes: list[Opcao]


@dataclass
class Resultado:
    status: Literal["respondida", "nao_encontrada", "precisa_esclarecimento", "erro"]
    motivo: Literal["sem_contexto", "medicamento_nao_indexado"] | None = None
    resposta: str | None = None
    fontes: list[Fonte] = field(default_factory=list)
    principios_ativos: list[str] = field(default_factory=list)  # filtro usado na busca
    medicamentos_nao_indexados: list[str] = field(default_factory=list)
    esclarecimento: Esclarecimento | None = None
    codigo_erro: CodigoErro | None = None
    detalhe_erro: str | None = None
    saida_bruta: str | None = None  # o que o modelo devolveu na última rodada, para depurar
    rodadas: int = 0
    tentativas: list[Tentativa] = field(default_factory=list)
    tempos_s: dict[str, float] = field(default_factory=dict)  # somados entre as rodadas
    prompt_tokens: int = 0  # somados entre as rodadas


def montar_prompt(pergunta: str, fontes: list[Fonte], observacao: str | None = None) -> str:
    trechos = "\n\n".join(
        f"[{f.posicao}] {f.chunk.medicamento} | {ROTULO_SECAO[f.chunk.secao]} | página {f.chunk.pagina}\n{f.chunk.texto}"
        for f in fontes
    )
    extra = f"\n\nOBSERVAÇÃO: {observacao}" if observacao else ""
    return f"TRECHOS:\n{trechos}{extra}\n\nPERGUNTA: {pergunta}"


def responder(
    pergunta: str, acervo: Acervo, principios_ativos: list[str] | None = None, s: Settings | None = None
) -> Resultado:
    """`principios_ativos` vazio = detectar o medicamento na própria pergunta."""
    from app.services.grafo import executar  # o grafo importa as peças deste módulo

    return executar(pergunta, acervo, principios_ativos, s)


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
