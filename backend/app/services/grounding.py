"""Checagem de fundamentação, determinística: a afirmação está escrita no trecho que ela cita?

Complementa citations.validar_saida, que só prova que o trecho citado existe. Regras por
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
5. Afirmação que nega ("não", "nunca", "nem") só passa se a frase do trecho que a sustenta também
   nega ("não", "contraindicado", "evitar"...). As regras 1 a 3 comparam palavras, e "não é
   indicado para dor no ouvido" tem as mesmas palavras de "é indicado para... ouvido".
Regras 3 e 4 entraram em 2026-10-16, com a busca em todas as bulas (docs/DECISOES.md, seção 20);
a 5, em 2026-10-18 (seção 28).

É de propósito conservadora: paráfrase demais reprova, e a reprovação só custa outra rodada.

`corrigir_citacoes` roda antes da checagem: a frase certa citando o trecho errado ganha o [n] do
trecho que a sustenta, em vez de custar uma rodada do LLM (docs/DECISOES.md, seção 31).

`destaques` usa a mesma comparação para o caminho inverso: as frases inteiras do trecho que
sustentam a afirmação, para a tela destacar na fonte (docs/DECISOES.md, seção 26).
"""

import math
import re
import unicodedata
from dataclasses import dataclass

from app.schemas.dataset import Chunk
from app.services.medicines import Catalog, detectar, formas_do_nome, normalizar

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
# Mensagens das regras 3 e 5 (o grafo reconhece por elas e manda um aviso específico na próxima rodada).
SO_NOME = "não diz nada do texto do trecho, só o nome do medicamento"
NEGA = "nega o que o trecho não nega"
# Regra 5. Na afirmação, só a negação explícita; no trecho, também as formas da bula de proibir
# ("Não use na gravidez" diz o mesmo que "contraindicado na gravidez").
_NEGA_NA_AFIRMACAO = re.compile(r"\b(?:nao|nunca|jamais|nem|nenhum|nenhuma)\b")
_NEGA_NO_TRECHO = re.compile(r"\b(?:nao|nunca|jamais|nem|nenhum|nenhuma|contra-?indic\w*|evit\w*|proib\w*)\b")
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
class Reference:
    """O que o trecho citado permite afirmar, já pré-processado."""

    numeros: set[str]
    pares: set[tuple[str, str]]  # (número, unidade)
    prefixos: set[str]  # do nome e do texto
    prefixos_nome: set[str]  # só do medicamento e do princípio ativo (o cabeçalho do trecho)
    prefixos_texto: set[str]  # só do texto da bula
    palavras: set[str]  # palavras do nome e do texto, para os termos equivalentes

    @classmethod
    def de_chunk(cls, chunk: Chunk) -> "Reference":
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


def problemas_da_afirmacao(texto: str, ref: Reference, min_termos: float) -> list[str]:
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


def remedios_de_outra_bula(texto: str, chunk: Chunk, catalogo: Catalog) -> list[str]:
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
    catalogo: Catalog | None = None,
    permitidos: list[str] | None = None,
) -> list[str]:
    """Problemas de todas as afirmações, já com o número da afirmação e do trecho.
    `catalogo`: liga a regra 4 (remédio certo); o grafo sempre passa.
    `permitidos`: remédios que podem aparecer citando a bula de outro ("para quem tem alergia a
    dipirona, o paracetamol é indicado para dor [2]")."""
    refs = {pos: Reference.de_chunk(c) for pos, c in chunks_por_posicao.items()}
    livres = {normalizar(p) for p in permitidos or []}
    problemas = []
    for i, (texto, trecho) in enumerate(afirmacoes, start=1):
        achados = _problemas(texto, chunks_por_posicao[trecho], refs[trecho], min_termos, catalogo, livres)
        problemas += [f"afirmação {i}, trecho [{trecho}]: {p}" for p in achados]
    return problemas


def _problemas(
    texto: str, chunk: Chunk, ref: Reference, min_termos: float, catalogo: Catalog | None, livres: set[str]
) -> list[str]:
    """As regras 1 a 5 para uma afirmação contra um trecho. Lista vazia = sustentada."""
    achados = problemas_da_afirmacao(texto, ref, min_termos)
    if nega_sem_o_trecho(texto, chunk):
        achados.append(NEGA)
    if catalogo is not None:
        achados += [
            f"fala de {nome}, mas o trecho é da bula de {chunk.medicamento}"
            for nome in remedios_de_outra_bula(texto, chunk, catalogo)
            if normalizar(nome) not in livres
        ]
    return achados


# ---- Citação corrigida no código -------------------------------------------------------------


def corrigir_citacoes(
    afirmacoes: list[tuple[str, int]],
    chunks_por_posicao: dict[int, Chunk],
    min_termos: float,
    catalogo: Catalog | None = None,
    permitidos: list[str] | None = None,
) -> tuple[list[tuple[str, int]], list[str]]:
    """A citação trocada corrigida antes da checagem (docs/DECISOES.md, seção 31).

    O modelo pequeno copia a frase certa de um trecho e cita outro (consulta 12f61084: a frase do [3]
    citando [1] e [2]), e isso custava uma rodada inteira do LLM. Se a afirmação não passa no trecho
    que citou, mas passa em todas as regras num outro trecho enviado, o [n] passa a ser o desse trecho
    (o primeiro, na ordem enviada). A checagem depois continua sendo contra o trecho citado, então a
    fonte mostrada é a que sustenta a frase. Afirmação repetida (o mesmo texto) sai.

    O outro trecho tem de ser do mesmo remédio do citado, ou de um remédio que a afirmação nomeia:
    "é indicado para dor de cabeça", escrita sobre o ibuprofeno, não pode virar a bula do paracetamol.

    Devolve as afirmações corrigidas e o que mudou, para o histórico (Attempt.correcoes)."""
    refs = {pos: Reference.de_chunk(c) for pos, c in chunks_por_posicao.items()}
    livres = {normalizar(p) for p in permitidos or []}

    def passa(texto: str, pos: int) -> bool:
        return not _problemas(texto, chunks_por_posicao[pos], refs[pos], min_termos, catalogo, livres)

    corrigidas: list[tuple[str, int]] = []
    mudancas: list[str] = []
    vistas: set[str] = set()
    for i, (texto, trecho) in enumerate(afirmacoes, start=1):
        if (chave := normalizar(texto)) in vistas:
            mudancas.append(f"afirmação {i} repetida, saiu")
            continue
        vistas.add(chave)
        if not passa(texto, trecho):
            citado = chunks_por_posicao[trecho]
            outro = next(
                (
                    pos for pos in sorted(chunks_por_posicao)
                    if pos != trecho
                    and _pode_citar(texto, citado, chunks_por_posicao[pos], catalogo)
                    and passa(texto, pos)
                ),
                None,
            )
            if outro is not None:
                mudancas.append(f"afirmação {i}: trecho [{trecho}] -> [{outro}]")
                trecho = outro
        corrigidas.append((texto, trecho))
    return corrigidas, mudancas


def _pode_citar(texto: str, citado: Chunk, outro: Chunk, catalogo: Catalog | None) -> bool:
    """O outro trecho é do mesmo remédio do citado, ou de um remédio que a afirmação nomeia."""
    if normalizar(outro.principio_ativo) == normalizar(citado.principio_ativo):
        return True
    if catalogo is None:
        return False
    deteccao = detectar(texto, catalogo, tolerar_erros=False)
    cabecalho = normalizar(f"{outro.medicamento} {outro.principio_ativo}")
    return any(
        re.search(rf"\b{re.escape(forma)}\b", cabecalho)
        for nome in deteccao.indexados + deteccao.nao_indexados
        for forma in formas_do_nome(nome)
    )


# ---- Destaque das frases que sustentam a resposta ------------------------------------------

# Fim de frase como em extract.unidades: ponto (ou ! ?) e a próxima começando por maiúscula ou número.
_FIM_DE_FRASE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ú0-9])")
# Frases destacadas por afirmação, no máximo: uma afirmação junta no máximo umas poucas frases da bula.
MAX_FRASES_POR_AFIRMACAO = 3


def frases_com_posicao(texto: str) -> list[tuple[int, int]]:
    """(início, fim) de cada frase do texto, nas posições do texto original (o fim é exclusivo)."""
    limites, inicio = [], 0
    for m in _FIM_DE_FRASE.finditer(texto):
        limites.append((inicio, m.start()))
        inicio = m.end()
    limites.append((inicio, len(texto)))
    saida = []
    for i, f in limites:
        pedaco = texto[i:f]
        if pedaco.strip():
            saida.append((i + len(pedaco) - len(pedaco.lstrip()), f - (len(pedaco) - len(pedaco.rstrip()))))
    return saida


def _literal(afirmacao: str, chunk: Chunk) -> re.Match | None:
    """A afirmação copiada do trecho (resposta extrativa: "Exemplol 50 mg: <frase da bula>"), com
    qualquer espaço entre as palavras."""
    frase = afirmacao.strip()
    prefixo = f"{chunk.medicamento}: "
    if frase.startswith(prefixo):
        frase = frase[len(prefixo):]
    palavras = frase.split()
    if len(palavras) < 5:  # poucas palavras em sequência acham o lugar errado ("a dose máxima é")
        return None
    return re.search(r"\s+".join(map(re.escape, palavras)), chunk.texto, re.IGNORECASE)


def _itens_da_afirmacao(afirmacao: str, chunk: Chunk) -> list[tuple[str, str]]:
    """O que a afirmação diz, para procurar nas frases: termos de conteúdo (sem o nome do remédio,
    que não ajuda a escolher a frase) e números."""
    texto = _com_numeros_por_extenso(_preparar(afirmacao))
    nome = {p[:_PREFIXO] for p in _PALAVRA.findall(_preparar(f"{chunk.medicamento} {chunk.principio_ativo}"))}
    termos = sorted({p for p in _PALAVRA.findall(texto) if p not in _COMUNS and p[:_PREFIXO] not in nome})
    numeros = sorted(set(_NUMERO.findall(texto)))
    return [("termo", t) for t in termos] + [("numero", n) for n in numeros]


def _itens_cobertos(itens: list[tuple[str, str]], frase: str) -> set[tuple[str, str]]:
    """Os itens da afirmação que a frase do trecho tem (termo pelo começo da palavra, ou equivalente)."""
    texto = _com_numeros_por_extenso(_preparar(frase))
    palavras = set(_PALAVRA.findall(texto))
    prefixos = {p[:_PREFIXO] for p in palavras}
    numeros = {v for n in _NUMERO.findall(texto) for v in _variantes(n)}

    def cobre(tipo: str, valor: str) -> bool:
        if tipo == "numero":
            return bool(_variantes(valor) & numeros)
        if valor[:_PREFIXO] in prefixos:
            return True
        return any(valor.startswith(a) and any(p.startswith(b) for p in palavras) for a, b in _EQUIVALENTES)

    return {it for it in itens if cobre(*it)}


def destaques_da_afirmacao(afirmacao: str, chunk: Chunk) -> list[tuple[int, int]]:
    """As frases inteiras do trecho que sustentam a afirmação, em posições do texto do trecho.

    Afirmação copiada da bula: as frases onde ela está. Senão, as frases que mais cobrem os termos e
    números da afirmação, uma de cada vez, até nenhuma acrescentar nada (no máximo
    MAX_FRASES_POR_AFIRMACAO). A primeira frase precisa cobrir dois itens (ou todos, se a afirmação
    tem um só): uma palavra solta em comum não basta para destacar. Sem LLM."""
    frases = frases_com_posicao(chunk.texto)
    if not frases:
        return []
    if m := _literal(afirmacao, chunk):
        return [(i, f) for i, f in frases if i < m.end() and f > m.start()]
    faltam = _itens_da_afirmacao(afirmacao, chunk)
    cobertura = {(i, f): _itens_cobertos(faltam, chunk.texto[i:f]) for i, f in frases}
    escolhidas: list[tuple[int, int]] = []
    pendentes = set(faltam)
    while pendentes and len(escolhidas) < MAX_FRASES_POR_AFIRMACAO:
        # Empate: a frase que vem antes no trecho (max devolve a primeira).
        melhor = max(frases, key=lambda fr: len(cobertura[fr] & pendentes))
        ganho = len(cobertura[melhor] & pendentes)
        if ganho == 0 or (not escolhidas and ganho < min(2, len(faltam))):
            break
        escolhidas.append(melhor)
        pendentes -= cobertura[melhor]
    return sorted(escolhidas)


def destaques(afirmacoes: list[str], chunk: Chunk) -> list[tuple[int, int]]:
    """Os destaques de todas as afirmações que citam o trecho, sem repetir frase."""
    return sorted({d for a in afirmacoes for d in destaques_da_afirmacao(a, chunk)})


def nega_sem_o_trecho(afirmacao: str, chunk: Chunk) -> bool:
    """Regra 5: uma frase da afirmação nega, mas nenhuma frase do trecho que sustenta ela nega.

    Frase por frase: com a afirmação inteira, a outra frase dela puxa outras frases do trecho para o
    apoio, e um "não" qualquer delas salvava a negação inventada. As frases que sustentam são as do
    destaque; sem nenhuma (pouco em comum com o trecho), vale o trecho inteiro."""
    for i, f in frases_com_posicao(afirmacao):
        frase = afirmacao[i:f]
        if not _NEGA_NA_AFIRMACAO.search(_preparar(frase)):
            continue
        apoio = [chunk.texto[a:b] for a, b in destaques_da_afirmacao(frase, chunk)] or [chunk.texto]
        if not any(_NEGA_NO_TRECHO.search(_preparar(t)) for t in apoio):
            return True
    return False
