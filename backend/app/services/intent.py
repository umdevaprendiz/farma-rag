"""Em que seção(ões) da bula a pergunta deve estar respondida. Sem LLM.

Usado pelo grafo de dois jeitos:
- pergunta sobre um medicamento indexado: a partir da rodada 2, quando a primeira busca (só por
  semelhança) não deu resposta verificada, a próxima fica restrita às seções indicadas;
- pergunta sem medicamento indexado ("melhor remédio pra dor de cabeça"): a busca em todas as
  bulas já começa nessas seções, senão trechos de reações adversas que falam de dor de cabeça
  viriam antes das indicações.
Vazio = a pergunta não indica seção, e a busca continua sem esse filtro.
"""

import re

from app.schemas.dataset import Section
from app.services.medicines import normalizar

# Início de palavra, já normalizado (sem acento, minúsculo). Uma pergunta pode cair em mais
# de uma seção: "gravidez" está em contraindicações numa bula e em advertências noutra.
_GATILHOS: dict[Section, tuple[str, ...]] = {
    "interacoes": ("interac", "interag", "junto", "associa", "misturar", "alcool", "bebida"),
    "contraindicacoes": (
        "contraindic", "gravid", "gestant", "gestac", "amament", "lactant", "nao pode", "alergi", "hipersensib",
    ),
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
    "indicacoes": (
        "para que serve", "serve para", "indicad", "indicac", "tratar", "trata ",
        # Pergunta de escolha ("qual o melhor remédio para...") e sintoma: a resposta são os
        # medicamentos cuja bula indica aquilo.
        "melhor remedio", "melhor medicamento", "remedio para", "remedio pra", "medicamento para",
        "medicamento pra", "o que tomar", "o que eu tomo", "o que posso tomar", "outro que",
        "para dor", "pra dor", "dor de", "dores", "febre", "gripe", "resfriad", "tosse", "catarro",
        "secrec", "alergi", "coceira", "rinite", "espirr", "azia", "gastrite", "ulcera", "refluxo",
        "pressao alta", "hipertens", "verme", "piolho", "sarna", "assadura", "tontura", "tonto",
        "enjoo", "nausea", "anemia",
    ),
    "superdose": ("superdose", "overdose", "intoxic", "tomou demais", "dose excessiva"),
    "caracteristicas_farmacologicas": ("mecanismo", "como age", "como funciona", "meia vida", "efeito maximo"),
}


def inferir_secoes(pergunta: str) -> list[Section]:
    texto = f" {normalizar(pergunta)} "
    return [
        secao
        for secao, gatilhos in _GATILHOS.items()
        if any(re.search(r"\b" + re.escape(g), texto) for g in gatilhos)
    ]


# --- Assunto da pergunta, para a busca geral -------------------------------------------------
# A busca por semelhança sempre devolve os trechos "menos distantes", mesmo quando nenhuma bula
# fala do assunto: "estou tonto" trazia as indicações do paracetamol (2026-10-16). Na busca
# geral, o trecho precisa ter uma destas palavras para entrar.

# Palavras da pergunta que não dizem o assunto. Normalizadas (sem acento).
_GENERICAS = {
    "qual", "quais", "melhor", "remedio", "remedios", "medicamento", "medicamentos", "tomar", "tomo",
    "posso", "pode", "podem", "estou", "estava", "tenho", "tive", "para", "outro", "outra", "algum",
    "alguma", "usar", "usado", "usada", "dado", "dada", "quero", "preciso", "devo", "serve", "como",
    "quando", "quanto", "quantos", "quantas", "sobre", "esse", "essa", "este", "esta", "isso", "fazer",
    "ajuda", "voce", "junto", "mesmo", "muito", "mais", "existe", "indicado", "indicada", "bula",
    "eles", "elas", "dele", "dela", "sinto", "sentindo", "minha", "meu", "favor", "possa", "possam",
    "poderia", "deveria", "tenha", "tomei", "tomou", "tomando", "estar", "ficar", "fiquei", "coisa",
    "algo", "nada", "alguem", "pessoa", "gente", "hoje", "agora", "ainda", "certo", "seria",
    "alta", "baixa",  # "pressão alta": o assunto é a pressão
}
# Sintoma dito de um jeito na pergunta e de outro na bula. Valores: começos de palavra.
_SINONIMOS = {
    "dor": ("dor", "dores", "analges"),
    "tonto": ("tontura", "vertig", "labirint"),
    "tontura": ("tontura", "vertig", "labirint"),
    "cabeca": ("cabeca", "cefale", "enxaquec"),
    "enjoo": ("enjoo", "nause", "vomit"),
    "enjoado": ("enjoo", "nause", "vomit"),
    "vomito": ("enjoo", "nause", "vomit"),
    "alergia": ("alergi", "alergic", "hipersensib", "histamin", "rinite", "urticar", "prurid"),
    "alergico": ("alergi", "alergic", "hipersensib", "histamin", "rinite", "urticar", "prurid"),
    "coceira": ("coceira", "prurid"),
    "febre": ("febre", "antipiret", "antitermic"),
    "gripe": ("gripe", "resfriad"),
    "resfriado": ("gripe", "resfriad"),
    "pressao": ("pressao", "hipertens", "hipotens"),
    "azia": ("azia", "acidez", "refluxo", "gastrit", "ulcera", "dispeps"),
    "tosse": ("tosse", "expector", "secrec", "mucolit", "catarro"),
    "catarro": ("catarro", "expector", "secrec", "mucolit"),
    "verme": ("verme", "helmint", "parasit"),
    "vermes": ("verme", "helmint", "parasit"),
    "piolho": ("piolho", "pedicul"),
    "sarna": ("sarna", "escabi"),
    "assadura": ("assadur", "dermatit"),
    "diabetes": ("diabet",),
    "rins": ("rins", "rim", "renal", "renais", "nefro"),
    "rim": ("rins", "rim", "renal", "renais", "nefro"),
    "figado": ("figado", "hepat"),
}
_PREFIXO = 5  # o mesmo da fundamentação: aceita flexão ("criança" x "crianças")


def termos_do_assunto(pergunta: str) -> list[str]:
    """Começos de palavra que um trecho precisa ter (ao menos um) para falar do assunto da
    pergunta. Vazio = a pergunta não tem assunto além do pedido ("qual o melhor remédio?")."""
    grupos: list[tuple[str, ...]] = []
    for palavra in re.findall(r"[a-z]{3,}", normalizar(pergunta)):
        if palavra in _SINONIMOS:
            grupos.append(_SINONIMOS[palavra])
        elif len(palavra) >= 4 and palavra not in _GENERICAS:
            grupos.append((palavra[:_PREFIXO],))
    # "Dor" só é o assunto quando não há outro: em "dor de cabeça" o assunto é a cabeça, senão
    # qualquer bula com "dores" entraria.
    if len(grupos) > 1:
        grupos = [g for g in grupos if g != _SINONIMOS["dor"]]
    return list(dict.fromkeys(t for g in grupos for t in g))


def menciona(texto_normalizado: str, termos: list[str]) -> bool:
    """O texto (já normalizado) tem alguma palavra começando por um dos termos?"""
    return any(re.search(r"\b" + re.escape(t), texto_normalizado) for t in termos)


# Pergunta de dose: a bula responde com número e unidade ("½ a 1 comprimido até 4 vezes ao dia"),
# muitas vezes sem escrever a palavra "dose".
_DE_DOSE = ("dose", "dosag", "posol", "maxim", "quant")
_UNIDADE_DE_DOSE = ["comprim", "gota", "mg", "ml", "capsul", "sache", "vezes", "dose", "ui"]


def responde_ao_assunto(
    afirmacoes: list[str], termos: list[str], secoes_citadas: set[str] = frozenset(), secoes_da_pergunta: list[str] = ()
) -> bool:
    """A resposta fala do que foi perguntado? `termos`: os do assunto da pergunta, sem os nomes de
    remédio (ver grafo). A fundamentação garante que a frase está no trecho, não que ela responde:
    "dose máxima de dipirona?" -> "não existe experiência em insuficiência renal" passava (2026-10-16).

    Fala do assunto: usa uma palavra dele; ou cita trecho da seção que a pergunta indica ("tem
    interação?" -> "pode reduzir o efeito anti-hipertensivo", de Interações), menos posologia, onde a
    frase errada da dipirona estava; ou, em pergunta de dose, traz número com unidade de dose."""
    texto = normalizar(" ".join(afirmacoes))
    if menciona(texto, termos) or (set(secoes_da_pergunta) - {"posologia"}) & set(secoes_citadas):
        return True
    return any(t.startswith(_DE_DOSE) for t in termos) and bool(re.search(r"\d", texto)) and menciona(texto, _UNIDADE_DE_DOSE)


# --- Conversa: alergia a um remédio, pedido de remédio, pergunta que continua a anterior -------

# "alergia a", "alérgica à", "hipersensibilidade ao": o que vem depois é o remédio da alergia.
# "Loratadina serve para alergia?" e "tenho alergia, o que tomar?" não casam: lá a alergia é a
# indicação, não um remédio a evitar.
_ALERGIA_A = re.compile(r"\b(?:alergi|alergic|hipersensib)[a-z]*\s+(?:a|ao|aos|as|de|do|da|dos|das|com|pelo|pela)\b")
_PEDE_REMEDIO = re.compile(
    r"\b(?:melhor (?:remedio|medicamento)|(?:remedio|medicamento)s? (?:para|pra)|o que (?:eu )?(?:posso )?tomar"
    r"|qual (?:remedio|medicamento))\b"
)
_ELIPTICA = re.compile(r"^(?:e|mas|tambem)\b")


# Onde termina o nome do remédio da alergia: "alergia a dipirona, posso tomar Exemplol?" é alergia
# a dipirona, não ao Exemplol.
_FIM_DO_NOME = {
    "posso", "pode", "podem", "tem", "tenho", "tomar", "usar", "qual", "quais", "que", "e", "mas", "ou",
    "para", "pra", "se", "existe", "algum", "alguma", "outro", "outra",
}


def texto_apos_alergia(pergunta: str) -> str | None:
    """"Tenho alergia a dipirona, posso tomar outro?" -> "dipirona" (normalizado): onde procurar o
    remédio da alergia. "" = alergia a um remédio sem dizer qual. None = não fala de alergia a algo."""
    texto = normalizar(pergunta)
    achado = _ALERGIA_A.search(texto)
    if achado is None:
        return None
    nome = []
    for palavra in texto[achado.end():].split():
        if palavra in _FIM_DO_NOME:
            break
        nome.append(palavra)
    return " ".join(nome)


def pede_remedio(pergunta: str) -> bool:
    """"Qual o melhor remédio para tosse?": pergunta nova, mesmo sem nome de remédio."""
    return bool(_PEDE_REMEDIO.search(normalizar(pergunta)))


def eliptica(pergunta: str) -> bool:
    """"E para criança?", "mas e na gravidez?": só faz sentido junto com a pergunta anterior."""
    return bool(_ELIPTICA.match(normalizar(pergunta)))
