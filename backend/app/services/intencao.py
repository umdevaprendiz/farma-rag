"""Em que seção(ões) da bula a pergunta deve estar respondida. Sem LLM.

Usado pelo grafo a partir da rodada 2: quando a primeira busca (só por semelhança) não deu
resposta verificada, a próxima busca fica restrita às seções que a pergunta indica.
Vazio = a pergunta não indica seção, e a busca continua sem esse filtro.
"""

import re

from app.schemas.dataset import Secao
from app.services.medicamentos import normalizar

# Início de palavra, já normalizado (sem acento, minúsculo). Uma pergunta pode cair em mais
# de uma seção: "gravidez" está em contraindicações numa bula e em advertências noutra.
_GATILHOS: dict[Secao, tuple[str, ...]] = {
    "interacoes": ("interac", "interag", "junto", "associa", "misturar", "alcool", "bebida"),
    "contraindicacoes": ("contraindic", "gravid", "gestant", "gestac", "amament", "lactant", "nao pode"),
    "advertencias": (
        "gravid", "gestant", "amament", "idoso", "renal", "rins", "hepat", "figado", "dirigir",
        "precauc", "cuidado", "diabet",
    ),
    "posologia": (
        "dose", "dosagem", "posologia", "quantos", "quantas", "vezes ao dia", "intervalo",
        "de quanto em quanto", "como tomar", "como usar", "jejum", "aliment", "maxim",
    ),
    "reacoes_adversas": ("efeito colateral", "efeitos colaterais", "reac", "advers", "causa"),
    "armazenamento": ("guardar", "armazen", "conserva", "geladeira", "temperatura", "validade"),
    "composicao": ("lactose", "gluten", "excipiente", "composic", "contem", "acucar", "corante"),
    "indicacoes": ("para que serve", "serve para", "indicad", "indicac", "tratar", "trata "),
    "superdose": ("superdose", "overdose", "intoxic", "tomou demais", "dose excessiva"),
    "caracteristicas_farmacologicas": ("mecanismo", "como age", "como funciona", "meia vida", "efeito maximo"),
}


def inferir_secoes(pergunta: str) -> list[Secao]:
    texto = f" {normalizar(pergunta)} "
    return [
        secao
        for secao, gatilhos in _GATILHOS.items()
        if any(re.search(r"\b" + re.escape(g), texto) for g in gatilhos)
    ]
