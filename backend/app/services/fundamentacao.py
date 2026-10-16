"""Checagem de fundamentação, determinística: a afirmação está escrita no trecho que ela cita?

Complementa citacoes.validar_saida, que só prova que o trecho citado existe. Regras por
afirmação, sempre contra o trecho citado (com o nome do medicamento, que o modelo vê no
cabeçalho do trecho):
1. Todo número tem de aparecer no trecho, e todo número com unidade de dose ("100 mg",
   "15 gotas", "30 °C") tem de aparecer com a mesma unidade. É o erro mais perigoso em bula.
2. Pelo menos `min_termos` dos termos de conteúdo (palavras de 4+ letras fora da lista de
   palavras comuns) aparecem no trecho, comparando pelo começo da palavra para aceitar
   flexão ("contraindicado" x "contraindicada"). Arredondado para baixo e no mínimo 1: em frase
   curta ("A tontura ocorre em 1% a 10%") um verbo trocado não pode reprovar sozinho.
3. Algum termo além do nome do medicamento está no texto do trecho. Sem isso, "Ibuprofeno 400 mg
   [2]" passava como resposta a "estou tonto, qual o melhor?" (o nome está no cabeçalho).
4. (com o catálogo) Remédio citado na afirmação é o do trecho ou aparece no texto dele. Sem isso,
   "a dipirona é contraindicada para crianças" passava citando a bula do omeprazol.
Regras 3 e 4 entraram em 2026-10-16, com a busca em todas as bulas (docs/DECISOES.md, seção 20).

É de propósito conservadora: paráfrase demais reprova, e a reprovação só custa outra rodada.
"""

import math
import re
import unicodedata
from dataclasses import dataclass

from app.schemas.dataset import Chunk
from app.services.medicamentos import Catalogo, detectar, formas_do_nome, normalizar

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
# Termos que a bula escreve de outro jeito: "alergia" na resposta, "hipersensibilidade" na bula.
# (começo do termo na afirmação, começo da palavra que vale no trecho)
_EQUIVALENTES = (("alerg", "hipersensib"), ("hipersensib", "alerg"))
# Mensagem da regra 3 (o grafo reconhece por ela e manda um aviso específico na próxima rodada).
SO_NOME = "não diz nada do texto do trecho, só o nome do medicamento"
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
    prefixos: set[str]  # do nome e do texto
    prefixos_nome: set[str]  # só do medicamento e do princípio ativo (o cabeçalho do trecho)
    prefixos_texto: set[str]  # só do texto da bula
    palavras: set[str]  # palavras do nome e do texto, para os termos equivalentes

    @classmethod
    def de_chunk(cls, chunk: Chunk) -> "Referencia":
        nome = _preparar(f"{chunk.medicamento} {chunk.principio_ativo}")
        corpo = _com_numeros_por_extenso(_preparar(chunk.texto))
        texto = f"{_com_numeros_por_extenso(nome)} {corpo}"
        prefixos_nome = {p[:_PREFIXO] for p in _PALAVRA.findall(nome)}
        prefixos_texto = {p[:_PREFIXO] for p in _PALAVRA.findall(corpo)}
        return cls(
            numeros={v for n in _NUMERO.findall(texto) for v in _variantes(n)},
            pares={(v, _UNIDADES[u]) for n, u in _NUMERO_UNIDADE.findall(texto) for v in _variantes(n)},
            prefixos=prefixos_nome | prefixos_texto,
            prefixos_nome=prefixos_nome,
            prefixos_texto=prefixos_texto,
            palavras=set(_PALAVRA.findall(texto)),
        )

    def tem(self, termo: str, prefixos: set[str] | None = None) -> bool:
        """O termo (ou um equivalente, como alergia/hipersensibilidade) está no trecho?"""
        if termo[:_PREFIXO] in (self.prefixos if prefixos is None else prefixos):
            return True
        return any(termo.startswith(a) and any(p.startswith(b) for p in self.palavras) for a, b in _EQUIVALENTES)


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
        faltam = sorted(t for t in termos if not ref.tem(t))
        if len(termos) - len(faltam) < max(1, math.floor(min_termos * len(termos))):
            problemas.append(f"termos fora do trecho: {', '.join(faltam)}")
        conteudo = [t for t in termos if t[:_PREFIXO] not in ref.prefixos_nome]
        if not any(ref.tem(t, ref.prefixos_texto) for t in conteudo):
            problemas.append(SO_NOME)
    return problemas


def remedios_de_outra_bula(texto: str, chunk: Chunk, catalogo: Catalogo) -> list[str]:
    """Remédios citados na afirmação que não são o do trecho nem aparecem no texto dele."""
    deteccao = detectar(texto, catalogo, tolerar_erros=False)  # o modelo escreve o nome certo
    do_trecho = normalizar(chunk.principio_ativo)
    corpo = normalizar(chunk.texto)
    return [
        nome
        for nome in deteccao.indexados + deteccao.nao_indexados
        if normalizar(nome) != do_trecho
        and not any(re.search(rf"\b{re.escape(f)}\b", corpo) for f in formas_do_nome(nome))
    ]


def checar(
    afirmacoes: list[tuple[str, int]],
    chunks_por_posicao: dict[int, Chunk],
    min_termos: float,
    catalogo: Catalogo | None = None,
    permitidos: list[str] | None = None,
) -> list[str]:
    """Problemas de todas as afirmações, já com o número da afirmação e do trecho.
    `catalogo`: liga a regra 4 (remédio certo); o grafo sempre passa.
    `permitidos`: remédios que podem aparecer citando a bula de outro ("para quem tem alergia a
    dipirona, o paracetamol é indicado para dor [2]")."""
    refs = {pos: Referencia.de_chunk(c) for pos, c in chunks_por_posicao.items()}
    livres = {normalizar(p) for p in permitidos or []}
    problemas = []
    for i, (texto, trecho) in enumerate(afirmacoes, start=1):
        achados = problemas_da_afirmacao(texto, refs[trecho], min_termos)
        if catalogo is not None:
            chunk = chunks_por_posicao[trecho]
            achados += [
                f"fala de {nome}, mas o trecho é da bula de {chunk.medicamento}"
                for nome in remedios_de_outra_bula(texto, chunk, catalogo)
                if normalizar(nome) not in livres
            ]
        problemas += [f"afirmação {i}, trecho [{trecho}]: {p}" for p in achados]
    return problemas
