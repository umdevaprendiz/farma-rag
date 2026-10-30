"""Perguntas que uma bula não responde por natureza, decididas antes da busca e sem LLM (docs/DECISOES.md, seção 47).

O benchmark completo de 2026-10-29 mostrou o motivo: "Quanto custa o Cozaar?" e "dose de amoxicilina para um
cachorro" citam um remédio do acervo, e o caminho de resposta copiada da bula devolvia a posologia dele sem
conferir se a pergunta era sobre dose. A bula de remédio humano não traz preço nem uso em animais."""

import re

from app.services.medicines import normalizar

PRECO = "preco"
ANIMAL = "animal"

# Normalizadas (sem acento, minúsculas). "valor" e "barato" ficam de fora: "valor desejado" e "barata" existem em bula.
_PRECO = re.compile(
    r"\b(?:quanto (?:custa|custam|sai|cobra|cobram|fica)|custa|custam|precos?|mais barat[oa]s?|descontos?"
    r"|promocao|promocoes|onde (?:comprar|encontrar|vende)|farmacia popular|plano de saude)\b"
)
# "animais" sozinho fica de fora: "os estudos em animais mostraram risco?" é pergunta sobre a bula.
_ANIMAL = re.compile(
    r"\b(?:cachorr\w+|caes|cao|gat[oa]s?|cavalos?|bovinos?|coelhos?|hamsters?|passaros?|pets?|veterinari\w+"
    r"|animal de estimacao|animais de estimacao)\b"
)

# "chá de boldo", "erva-doce", "suco de toranja": a substância que não é remédio. Depois dela, o nome.
_NAO_REMEDIO = re.compile(r"\b(?:cha|chas|suco|sucos|erva|ervas|planta|plantas|infusao|fitoterapico)[\s-]+(?:(?:de|da|do)[\s-]+)?([a-z]{4,})")
# Palavras que seguem "chá"/"erva" sem ser o nome da substância.
_NAO_E_NOME = {"para", "como", "medicinal", "medicinais", "natural", "naturais", "tipo", "qualquer", "algum", "alguma"}
_PREFIXO = 5


def fora_do_escopo(pergunta: str) -> str | None:
    """PRECO ou ANIMAL se a pergunta é sobre isso; None se é assunto de bula."""
    texto = normalizar(pergunta)
    if _PRECO.search(texto):
        return PRECO
    if _ANIMAL.search(texto):
        return ANIMAL
    return None


def substancia_fora_do_remedio(pergunta: str) -> str | None:
    """"Omeprazol pode com chá de boldo?" -> "boldo" (normalizado). None = a pergunta não cita chá, erva ou suco."""
    for achado in _NAO_REMEDIO.finditer(normalizar(pergunta)):
        if achado.group(1) not in _NAO_E_NOME:
            return achado.group(1)
    return None


def citada_em(substancia: str, textos: list[str]) -> bool:
    """Algum dos textos (já normalizados) começa uma palavra com as primeiras letras da substância?"""
    return any(re.search(r"\b" + re.escape(substancia[:_PREFIXO]), t) for t in textos)
