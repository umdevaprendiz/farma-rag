"""Para que serve o remédio, pela seção de indicações da bula. Sem LLM (docs/DECISOES.md, seção 29).

Pedido do usuário (2026-10-18): "o que cada medicamento faz, pra que serve; se não está escrito para
que serve, não é indicado". A bula do profissional lista as indicações aprovadas pela ANVISA; o que
não está lá é uso fora da bula. Para cada remédio com bula, uma de três situações:
- citado: o uso perguntado aparece nas indicações ("ibuprofeno serve para febre?");
- geral: só aparece o termo geral ("dor no ouvido" para o paracetamol, que a bula indica para "dores
  leves a moderadas" com outros exemplos): a bula não cita, mas também não exclui;
- ausente: nada do uso aparece ("paracetamol serve para ansiedade?"): pela bula, não é indicado.
Remédio sem bula no acervo: a classe da ANVISA, sem concluir. A classe é grosseira (o clonazepam é
"anticonvulsivante", e a bula dele indica pânico), e concluir por ela seria inventar.

Pergunta que junta o uso com outra coisa ("ibuprofeno para dor se tenho dengue?", "dipirona para
criança?", "quantas gotas para febre?") não entra aqui: segue o caminho normal, que lê a seção de
contraindicação, advertência ou posologia.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from app.schemas.dataset import Chunk
from app.services.extract import unidades
from app.services.intent import _GENERICAS, _SINONIMOS, inferir_secoes
from app.services.medicines import _SAIS, normalizar

# ressalva: o uso só aparece em frase com "não" nas indicações ("não é recomendado trocar ... na
# insuficiência cardíaca"): mostra a frase, sem dizer sim nem não.
# lista: pergunta geral ("para que serve?"), só as indicações.
Situation = Literal["citado", "ressalva", "geral", "ausente", "sem_bula", "lista"]

# Pergunta geral: "para que serve?", "serve pra quê?", "qual a indicação?".
_GERAL = re.compile(
    r"\b(?:para|pra) (?:que|q|oque|o que) (?:serve|servem|e|eh|sao|ele e|ela e|e indicad[oa])\b"
    r"|\bserve[m]? (?:para|pra) (?:que|q|oque|o que)\b"
    r"|\b(?:qual|quais) (?:e |sao )?(?:a |as )?indicac(?:ao|oes)\b"
    r"|\bindicad[oa] (?:para|pra) (?:que|o que)\b"
    r"|\bo que (?:o |a )?[a-z]+ trata\b"
)
# Verbo de uso seguido de "para X": "serve para", "é pra", "é bom para", "tomei ... para".
_VERBO = (
    r"(?:serve[m]?|e|eh|sao|seria|funciona|bom|boa|indicad[oa]s?|recomendad[oa]s?|tomar|tomei|tomo|tomou"
    r"|tomando|tomaria|usar|usei|uso|usou|usando|dar|dei|pode|posso|ajuda|trata|cura|alivia)"
)
_USO = re.compile(rf"\b{_VERBO}\b[^,.;?!]{{0,40}}?\b(?:para|pra|contra)\s+([^,.;?!]+)")
# "E para dor de cabeça?" depois de uma pergunta de uso: o uso novo, do mesmo remédio.
_USO_ELIPTICO = re.compile(r"^(?:e|mas e|e se for) (?:para|pra|contra)\s+([^,.;?!]+)")
# O que pode fechar a pergunta sem mudar o assunto: "tomei X para Y, fiz certo?".
_FECHO = re.compile(r"\b(?:fiz certo|faz sentido|esta certo|ta certo|certo|e verdade|ne|mesmo|ok)\b")
# Começo de outra condição: a pergunta não é só de uso ("para dor se tenho dengue").
_OUTRA_CONDICAO = re.compile(
    r"\b(?:se|sendo|estando|tendo|tenho|com|junto|mas|porque|pois|quando|depois|antes|durante|enquanto|sem"
    r"|na gravidez|gravida|amamentando|posso|pode|devo|qual|quanto|quantos|quantas)\b"
)
# "Para criança", "pra idoso": é a pergunta de quem pode usar, não de para que serve.
_POPULACAO = {
    "crianca", "criancas", "bebe", "bebes", "nenem", "idoso", "idosos", "idosa", "idosas", "gestante",
    "gestantes", "gravida", "gravidas", "adulto", "adultos", "lactante", "lactantes", "filho", "filha",
    "mulher", "mulheres", "homem", "homens", "pessoa", "pessoas", "quem", "recem",
}
# Palavras do uso que não dizem o que é ("dor muito forte no ouvido" -> dor, ouvido).
_VAZIAS = _GENERICAS | {
    "no", "na", "nos", "nas", "de", "da", "do", "das", "dos", "em", "um", "uma", "uns", "umas", "meu",
    "minha", "seu", "sua", "forte", "fraca", "fraco", "leve", "muito", "muita", "pouco", "pouca",
    "constante", "sempre", "toda", "todo", "dia", "dias", "noite", "hoje", "evitar", "prevenir", "tratar",
    "curar", "aliviar", "combater", "melhorar", "parar", "acabar", "tirar", "passar", "baixar", "isso",
    "aquilo", "coisa", "problema", "problemas", "sintoma", "sintomas", "crise", "crises",
}
# Sinônimos além dos da busca (intent._SINONIMOS): como a pessoa fala e como a bula escreve.
_MAIS_SINONIMOS = {
    "ouvido": ("ouvid", "otit", "auricul"), "garganta": ("gargant", "faring", "amigdal"),
    "dente": ("dente", "dentar", "odont"), "costas": ("costas", "lombal", "lombar"),
    "colica": ("colic", "dismenor", "espasm"), "menstrual": ("menstru", "dismenor"),
    "menstruacao": ("menstru", "dismenor"), "insonia": ("insoni",), "dormir": ("insoni", "sono"),
    "sono": ("insoni", "sono"), "espinha": ("acne", "espinh"), "espinhas": ("acne", "espinh"),
    "ansiedade": ("ansied", "ansios"), "depressao": ("depress",), "nausea": ("nause", "enjoo", "vomit"),
    "diarreia": ("diarr",), "intestino": ("intestin", "constip", "obstip", "laxat"),
    "prisao": ("constip", "obstip", "laxat"), "corrimento": ("corriment", "vagin", "candid"),
    "candidiase": ("candid",), "micose": ("micose", "fung", "dermatofit", "tinea"),
    "nariz": ("nariz", "nasal", "congest"), "entupido": ("congest", "obstru", "entupid"),
    "coracao": ("cardi", "coracao"), "infarto": ("infart", "cardi"), "gravidez": ("gravid", "contracep", "anticoncep"),
    "engravidar": ("gravid", "contracep", "anticoncep"), "infeccao": ("infec",), "inflamacao": ("inflam",),
    "queda": ("queda", "alopec", "calvic"), "cabelo": ("cabelo", "capilar", "alopec", "calvic"),
}
_DOR = _SINONIMOS["dor"]
# Expressões da bula em que a palavra não quer dizer o que parece: "febre do feno" é rinite alérgica.
_FALSOS_AMIGOS = [
    (re.compile(r"\bfebre do feno\b"), "rinite alergica"),
    (re.compile(r"\bfebre ou outras alergias\b"), "outras alergias"),  # oximetazolina: a febre do feno
]
# "Não" que nega a indicação ("não é recomendado trocar..."), e não o que descreve a doença
# ("reumatismo não articular", "diabetes não dependente de insulina").
_NEGA = re.compile(
    r"\bnao (?:e|sao|deve\w*|esta\w*|recomend\w*|indicad\w*|use|usar|utiliz\w*|administr\w*)\b|\bcontra-?indic\w*"
)
# Frase que diz para que o remédio é: a primeira que tiver uma destas palavras.
_FRASE_DE_INDICACAO = re.compile(r"\b(?:indica\w*|destina\w*|tratamento|alivio|usad[oa]|profilaxia|prevenc\w*)\b")


def _sem_acento(texto: str) -> str:
    """Tira o acento letra a letra, sem mudar o tamanho: a posição no texto sem acento é a mesma no
    original (o uso volta como a pessoa escreveu, "pressão alta")."""
    return "".join(unicodedata.normalize("NFKD", c)[:1] or c for c in texto)


def uso_perguntado(pergunta: str, anterior: str | None = None) -> str | None:
    """O uso que a pergunta quer saber se o remédio trata, em minúsculas, como a pessoa escreveu:
    "paracetamol serve para ansiedade?" -> "ansiedade"; "tomei rocutan para dor no ouvido, fiz certo?"
    -> "dor no ouvido". "" = pergunta geral ("para que serve a dipirona?"). None = não é pergunta só de
    para que serve (dose, criança, gravidez, interação, outra condição junto): segue o caminho normal.

    `anterior`: a pergunta anterior da conversa. "E para dor de cabeça?" depois de uma pergunta de uso
    pergunta outro uso do mesmo remédio."""
    # Com a pontuação (normalizar tira; aqui ela separa as orações).
    original = re.sub(r"\s+", " ", pergunta.lower()).strip()
    texto = _sem_acento(original)
    if eliptico := _USO_ELIPTICO.match(texto):
        # Sem anterior de uso ("pode na gravidez?" -> "e para dor de cabeça?"), não é pergunta de uso.
        if anterior is None or uso_perguntado(anterior) is None:
            return None
        return _limpar(original, texto, eliptico.span(1))
    if _GERAL.search(texto):
        resto = _GERAL.sub(" ", texto)
        return "" if _so_indicacao(resto) else None
    achado = _USO.search(texto)
    if achado is None:
        return None
    return _limpar(original, texto, achado.span(1))


def _limpar(original: str, texto: str, span: tuple[int, int]) -> str | None:
    """Corta o fecho ("fiz certo"); outra condição no meio ou fora do uso -> None."""
    inicio, fim = span
    if fecho := _FECHO.search(texto, inicio, fim):
        fim = fecho.start()
    uso = texto[inicio:fim]
    if _OUTRA_CONDICAO.search(uso):
        return None
    palavras = re.findall(r"[a-z]+", uso)
    if not palavras or palavras[0] in _POPULACAO or any(p in _POPULACAO for p in palavras[:2]):
        return None
    resto = texto[:inicio] + " " + texto[span[1]:]
    if not _so_indicacao(_FECHO.sub(" ", resto)):
        return None
    return original[inicio:fim].strip() if any(p not in _VAZIAS for p in palavras) else ""


def _so_indicacao(resto: str) -> bool:
    """O resto da pergunta (sem o uso) não pede outra seção da bula nem fala de outra condição."""
    if set(inferir_secoes(resto)) - {"indicacoes"}:
        return False
    return not _OUTRA_CONDICAO.search(re.sub(r"\b(?:posso|pode|devo|qual)\b", " ", resto)) and not any(
        p in _POPULACAO for p in re.findall(r"[a-z]+", resto)
    )


def grupos(uso: str) -> tuple[list[tuple[str, ...]], bool]:
    """Os começos de palavra que dizem o uso, um grupo por palavra (com sinônimos), e se o uso fala de
    dor. "dor no ouvido" -> ([("ouvid", "otit", "auricul")], True)."""
    especificos: list[tuple[str, ...]] = []
    dor = False
    for p in re.findall(r"[a-z]{3,}", normalizar(uso)):
        if p in ("dor", "dores", "doi", "doendo"):
            dor = True
        elif p not in _VAZIAS:
            especificos.append(_MAIS_SINONIMOS.get(p) or _SINONIMOS.get(p) or (p[:5],))
    return list(dict.fromkeys(especificos)), dor


def _menciona(texto: str, termos: tuple[str, ...]) -> bool:
    return any(re.search(r"\b" + re.escape(t), texto) for t in termos)


def _sem_falsos_amigos(texto: str) -> str:
    texto = normalizar(texto)
    for padrao, troca in _FALSOS_AMIGOS:
        texto = padrao.sub(troca, texto)
    return texto


def _termos(uso: str) -> tuple[list[tuple[str, ...]], bool]:
    """Os grupos que decidem "citado" e se ainda vale o "geral" (dor). Uso só de dor ("serve para
    dor?"): a dor é o específico."""
    especificos, dor = grupos(uso)
    return (especificos, dor) if especificos else (([_DOR] if dor else []), False)


def _cita(frase: str, especificos: list[tuple[str, ...]]) -> bool:
    """Metade ou mais das palavras específicas do uso aparecem ("dor de cabeça forte": cabeça)."""
    texto = _sem_falsos_amigos(frase)
    return bool(especificos) and sum(_menciona(texto, g) for g in especificos) * 2 >= len(especificos)


def situacao(uso: str, indicacoes: list[str]) -> Situation:
    """O uso está nas indicações (os textos da seção, de todas as bulas do remédio)? Frase a frase:
    citado se uma frase sem "não" cita o uso; ressalva se só frase com "não" cita; geral se nenhuma
    cita, mas o uso é uma dor e a bula indica para dor; ausente no resto."""
    especificos, dor = _termos(uso)
    frases_ = [f for texto in indicacoes for f in unidades(texto)]
    que_citam = [f for f in frases_ if _cita(f, especificos)]
    if any(not _NEGA.search(normalizar(f)) for f in que_citam):
        return "citado"
    if que_citam:
        return "ressalva"
    if dor and any(_menciona(_sem_falsos_amigos(f), _DOR) for f in frases_):
        return "geral"
    return "ausente"


def frases(uso: str, chunk: Chunk, situacao_: Situation, limite: int = 2) -> list[str]:
    """As frases do trecho de indicações que vão na resposta, copiadas como estão: as que citam o uso
    (citado, ressalva) ou as que dizem para que o remédio é (geral, ausente, pergunta geral)."""
    todas = [f for f in unidades(chunk.texto) if len(f) > 15]
    if situacao_ in ("citado", "ressalva"):
        # Só as que citam: o trecho sem o uso (outro pedaço da mesma seção) não entra.
        especificos, _ = _termos(uso)
        return [f for f in todas if _cita(f, especificos)][:limite]
    escolhidas = [f for f in todas if _FRASE_DE_INDICACAO.search(normalizar(f))]
    return (escolhidas or todas[:1])[:limite]


def nome_curto(principio_ativo: str) -> str:
    """"cloridrato de sertralina" -> "sertralina", "amoxicilina tri-hidratada" -> "amoxicilina";
    "ácido acetilsalicílico" fica como está (com acento)."""
    palavras = [p for p in principio_ativo.split() if not all(q in _SAIS for q in normalizar(p).split())]
    return " ".join(palavras) or principio_ativo


@dataclass(frozen=True)
class Verdict:
    medicamento: str  # nome curto do princípio ativo
    uso: str  # como a pessoa escreveu, sem acento ("" = pergunta geral)
    situacao: Situation
    classe: str | None = None  # remédio sem bula: classe terapêutica da ANVISA

    def frase(self) -> str:
        """A conclusão, montada no código: não é afirmação do modelo, é o que a comparação achou."""
        m, u = self.medicamento, self.uso
        if self.situacao == "sem_bula":
            base = f"A bula de {m} não está no acervo, então não dá para dizer"
            base += f" se é indicado para {u}." if u else " para que é indicado pela bula."
            if self.classe:
                base += f" Pelo registro na ANVISA, é da classe {self.classe.lower()}."
            return base
        if not u:
            return f"Para que serve, pela bula de {m}:"
        # "Sim" sem dizer que a palavra está na bula: "espinha" casa com "acne" pelo sinônimo.
        if self.situacao == "citado":
            return f"Sim. Pela bula de {m}, é indicado para isso:"
        if self.situacao == "ressalva":
            return f"A bula de {m} fala disso nas indicações, com ressalva:"
        if self.situacao == "geral":
            return f"A bula de {m} não cita {u}; ela indica o remédio para dor de forma geral:"
        return (
            f"Não. {u[0].upper() + u[1:]} não está entre as indicações da bula de {m}: pela bula, não é "
            "indicado para isso. A bula indica:"
        )


def conclusao(vereditos: list[Verdict]) -> str | None:
    return " ".join(v.frase() for v in vereditos) or None
