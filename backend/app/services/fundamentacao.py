"""Checagem de fundamentação, determinística: a afirmação está escrita no trecho que ela cita?

Complementa citacoes.validar_saida, que só prova que o trecho citado existe. Duas regras por
afirmação, sempre contra o trecho citado (com o nome do medicamento, que o modelo vê no
cabeçalho do trecho):
1. Todo número tem de aparecer no trecho, e todo número com unidade de dose ("100 mg",
   "15 gotas", "30 °C") tem de aparecer com a mesma unidade. É o erro mais perigoso em bula.
2. Pelo menos `min_termos` dos termos de conteúdo (palavras de 4+ letras fora da lista de
   palavras comuns) aparecem no trecho, comparando pelo começo da palavra para aceitar
   flexão ("contraindicado" x "contraindicada"). Arredondado para baixo e no mínimo 1: em frase
   curta ("A tontura ocorre em 1% a 10%") um verbo trocado não pode reprovar sozinho.

É de propósito conservadora: paráfrase demais reprova, e a reprovação só custa outra rodada.
"""

import math
import re
import unicodedata
from dataclasses import dataclass

from app.schemas.dataset import Chunk

_PALAVRAS_NUMERO = {
    "um": "1", "uma": "1", "dois": "2", "duas": "2", "tres": "3", "quatro": "4", "cinco": "5",
    "seis": "6", "sete": "7", "oito": "8", "nove": "9", "dez": "10", "doze": "12",
    "quinze": "15", "vinte": "20", "trinta": "30", "cem": "100",
}
# Unidade -> forma canônica. Só as de dose e medida: "8 horas" já é coberto pela regra do número.
_UNIDADES = {
    "mg": "mg", "mcg": "mcg", "ug": "mcg", "g": "g", "kg": "kg", "ml": "ml", "l": "l", "ui": "ui",
    "%": "%", "c": "c", "mmhg": "mmhg", "gota": "gota", "gotas": "gota",
}
_NUMERO = re.compile(r"\d+(?:[.,]\d+)*")
_NUMERO_UNIDADE = re.compile(r"(\d+(?:[.,]\d+)*)\s*(mg|mcg|ug|g|kg|ml|l|ui|%|c|mmhg|gotas?)(?![a-z])")
_PALAVRA = re.compile(r"[a-z]{4,}")
_PREFIXO = 5
# Palavras de 4+ letras que não carregam informação da bula.
_COMUNS = {
    "para", "como", "pode", "podem", "deve", "devem", "mais", "menos", "pelo", "pela", "pelos",
    "pelas", "este", "esta", "esse", "essa", "isso", "isto", "entre", "sobre", "quando", "tambem",
    "muito", "muita", "seja", "sendo", "ser", "estar", "sao", "esta", "pois", "porque", "cada",
    "outro", "outra", "outros", "outras", "mesmo", "mesma", "apos", "antes", "depois", "ainda",
    "sempre", "nunca", "deste", "desta", "neste", "nesta", "medicamento", "uso", "usar", "usado",
    "usada", "tomar", "tomado", "tomada", "paciente", "pacientes", "bula", "trecho", "informa",
    "recomenda", "recomendado", "recomendada", "segundo", "acordo", "conforme",
}


def _preparar(texto: str) -> str:
    """Minúsculo, sem acento, com µ virando "mc" e "°C" virando "c" (mantém %, vírgula e ponto)."""
    texto = texto.replace("µ", "mc").replace("μ", "mc").replace("°", "").replace("º", "")
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()


def _variantes(numero: str) -> set[str]:
    """"2,5" = "2.5"; "1.000" = "1000"."""
    return {numero.replace(",", "."), numero.replace(".", "").replace(",", ".")}


def _com_numeros_por_extenso(texto: str) -> str:
    return re.sub(r"[a-z]+", lambda m: _PALAVRAS_NUMERO.get(m.group(), m.group()), texto)


@dataclass(frozen=True)
class Referencia:
    """O que o trecho citado permite afirmar, já pré-processado."""

    numeros: set[str]
    pares: set[tuple[str, str]]  # (número, unidade)
    prefixos: set[str]

    @classmethod
    def de_chunk(cls, chunk: Chunk) -> "Referencia":
        texto = _com_numeros_por_extenso(_preparar(f"{chunk.medicamento} {chunk.principio_ativo} {chunk.texto}"))
        return cls(
            numeros={v for n in _NUMERO.findall(texto) for v in _variantes(n)},
            pares={(v, _UNIDADES[u]) for n, u in _NUMERO_UNIDADE.findall(texto) for v in _variantes(n)},
            prefixos={p[:_PREFIXO] for p in _PALAVRA.findall(texto)},
        )


def problemas_da_afirmacao(texto: str, ref: Referencia, min_termos: float) -> list[str]:
    """Lista vazia = a afirmação está sustentada pelo trecho."""
    afirmacao = _preparar(texto)
    problemas = []

    for numero, unidade in _NUMERO_UNIDADE.findall(afirmacao):
        if not _variantes(numero) & {n for n, u in ref.pares if u == _UNIDADES[unidade]}:
            problemas.append(f"'{numero} {unidade}' não está no trecho")
    ja_citados = {n for n, _ in _NUMERO_UNIDADE.findall(afirmacao)}
    for numero in _NUMERO.findall(afirmacao):
        if numero not in ja_citados and not _variantes(numero) & ref.numeros:
            problemas.append(f"o número '{numero}' não está no trecho")

    termos = {p for p in _PALAVRA.findall(afirmacao) if p not in _COMUNS}
    if termos:
        faltam = sorted(t for t in termos if t[:_PREFIXO] not in ref.prefixos)
        if len(termos) - len(faltam) < max(1, math.floor(min_termos * len(termos))):
            problemas.append(f"termos fora do trecho: {', '.join(faltam)}")
    return problemas


def checar(afirmacoes: list[tuple[str, int]], chunks_por_posicao: dict[int, Chunk], min_termos: float) -> list[str]:
    """Problemas de todas as afirmações, já com o número da afirmação e do trecho."""
    refs = {pos: Referencia.de_chunk(c) for pos, c in chunks_por_posicao.items()}
    problemas = []
    for i, (texto, trecho) in enumerate(afirmacoes, start=1):
        for p in problemas_da_afirmacao(texto, refs[trecho], min_termos):
            problemas.append(f"afirmação {i}, trecho [{trecho}]: {p}")
    return problemas
