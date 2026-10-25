"""Glossário do balcão: o termo técnico da bula e a palavra do dia a dia (docs/DECISOES.md, seção 37).

Uma lista só, com dois usos:
- na tela (`anotar`): depois do termo técnico da resposta, a explicação entre parênteses ("cefaleia
  (dor de cabeça)"). A frase da bula não muda: as posições vão à parte, e a tela monta;
- na fundamentação (`cobertos`): a resposta pode trocar o termo técnico pela expressão do dia a dia
  ("dor de cabeça" contra um trecho que diz "cefaleia"). Só a expressão inteira ("pressão alta", nunca
  "pressão" sozinha, que passaria em "pressão baixa") e só se o trecho tem o termo técnico.

Vocabulário geral de saúde escrito por Claude em 2026-10-24, não tirado das bulas: **revisar antes de
confiar.** As explicações são curtas e conservadoras; termo sem explicação segura fica de fora. Entrada
com `explicacao=None` só vale para a fundamentação (a tela não mostra nada).

Os padrões são expressões regulares sobre o texto minúsculo e sem acento (`dobrar`). A ordem importa
na tela: a entrada mais específica ("prurido ocular") vem antes da geral ("prurido").
"""

import re
import unicodedata
from dataclasses import dataclass

# Explicações por texto, no máximo: mais que isso enche a resposta de parênteses.
MAX_ANOTACOES = 6
_PALAVRA = re.compile(r"[a-z]{4,}")  # como a fundamentação conta termos (grounding._PALAVRA)


@dataclass(frozen=True)
class Term:
    tecnico: re.Pattern[str]  # como a bula escreve
    explicacao: str | None  # o que a tela mostra entre parênteses; None = só para a fundamentação
    comum: re.Pattern[str] | None  # como se fala no balcão; None = a fundamentação não troca


def _t(tecnico: str, explicacao: str | None, comum: str | None = None) -> Term:
    return Term(re.compile(rf"\b(?:{tecnico})\b"), explicacao, re.compile(rf"\b(?:{comum})\b") if comum else None)


TERMOS: tuple[Term, ...] = (
    # Dor
    _t(r"cefaleias?(?: tensiona(?:l|is))?", "dor de cabeça", r"dor(?:es)? (?:de|na) cabeca"),
    _t(r"lombalgias?", "dor na parte baixa das costas", r"dor(?:es)? (?:nas|na) costas|dor(?:es)? lomba(?:r|res)"),
    _t(r"dismenorreias?(?: primaria)?", "cólica menstrual", r"colicas? menstrua(?:l|is)"),
    _t(r"mialgias?", "dor muscular", r"dor(?:es)? muscula(?:r|res)|dor(?:es)? nos musculos"),
    _t(r"artralgias?", "dor nas articulações", r"dor(?:es)? nas articulacoes|dor(?:es)? nas juntas"),
    _t(r"odontalgias?", "dor de dente", r"dor(?:es)? de dentes?"),
    _t(r"epigastralgias?", "dor na boca do estômago", r"dor(?:es)? no estomago|dor(?:es)? na boca do estomago"),
    _t(r"analgesic[oa]s?", "alivia a dor", r"alivi[a-z]* (?:a |as )?dor(?:es)?|contra (?:a )?dor"),
    _t(r"antitermic[oa]s?|antipiretic[oa]s?", "baixa a febre", r"(?:baix|abaix|reduz)[a-z]* (?:a )?febre"),
    _t(r"reducao da febre", None, r"(?:baix|abaix|reduz)[a-z]* (?:a )?febre"),
    _t(r"antiespasmodic[oa]s?", "alivia cólicas", r"alivi[a-z]* (?:a |as )?colicas?"),
    # Pele e alergia
    _t(r"prurido ocular", "coceira nos olhos", r"coceiras? nos olhos"),
    _t(r"prurido nasal", "coceira no nariz", r"coceiras? no nariz"),
    _t(r"prurido do palato", "coceira no céu da boca", r"coceiras? no ceu da boca"),
    _t(r"pruridos?|pruriginos[oa]s?", "coceira", r"coceiras?"),
    _t(r"palato", "céu da boca", r"ceu da boca"),
    _t(r"urticarias?", "placas vermelhas na pele, com coceira"),
    _t(r"eczemas?", "inflamação da pele, com coceira"),
    _t(r"dermatites? de fraldas?", "assadura", r"assaduras?"),
    _t(r"dermatites?(?: atopica)?", "inflamação da pele", r"inflamac(?:ao|oes) (?:da|na) pele"),
    _t(r"eritemas?", "vermelhidão na pele", r"vermelhid(?:ao|oes)"),
    _t(r"angioedemas?|edema de quincke", "inchaço em lábios, língua, rosto ou garganta"),
    _t(r"edemas?", "inchaço", r"inchac(?:o|os)"),
    _t(r"hipersensibilidade", "alergia"),  # na fundamentação, já vale (grounding._EQUIVALENTES)
    _t(r"anafilaxias?|reac(?:ao|oes) anafilatic[ao]s?|choque anafilatico", "reação alérgica grave"),
    _t(r"fotossensibilidade", "sensibilidade da pele ao sol"),
    _t(r"acnes?", None, r"espinhas?"),
    _t(r"alopecias?(?: androgenetica)?", "queda de cabelo", r"queda (?:de |do )?cabelos?"),
    _t(r"pediculose", "piolho", r"piolhos?"),
    _t(r"escabiose", "sarna", r"sarnas?"),
    # Nariz, garganta, pulmão
    _t(r"coriza|rinorreia", "nariz escorrendo", r"nariz escorrendo"),
    _t(r"congestao nasal|obstrucao nasal", "nariz entupido", r"nariz entupido"),
    _t(r"descongestionantes?(?: nasa(?:l|is))?", "desentope o nariz", r"desentup[a-z]* (?:o )?nariz"),
    _t(r"anti-?histaminic[oa]s?", "antialérgico", r"antialergic[oa]s?"),
    _t(r"expectorantes?", "ajuda a soltar o catarro", r"solt[a-z]* (?:o )?catarro"),
    _t(r"expectorar", "soltar o catarro", r"solt[a-z]* (?:o )?catarro"),
    _t(r"mucolitic[oa]s?|secretolitic[oa]s?", "deixa o catarro mais fino, mais fácil de soltar",
       r"(?:afin|fluidific)[a-z]* (?:o )?catarro"),
    _t(r"broncopulmona(?:r|res)", "dos brônquios e pulmões"),
    _t(r"broncoespasmos?", "aperto nos brônquios, com falta de ar"),
    _t(r"dispneia", "falta de ar", r"falta de ar"),
    _t(r"faringoamigdalites?", "inflamação da garganta e das amígdalas"),
    _t(r"faringites?", "inflamação da garganta", r"inflamac(?:ao|oes) (?:na|da) garganta"),
    _t(r"otites?", "inflamação no ouvido", r"inflamac(?:ao|oes) (?:no|do) ouvido"),
    _t(r"exacerbac(?:ao|oes)", "piora dos sintomas"),
    # Estômago e intestino
    _t(r"nauseas?", "enjoo", r"enjoos?"),
    _t(r"pirose", "azia", r"azia"),
    _t(r"dispepsias?", "má digestão", r"ma digestao"),
    _t(r"constipacao(?: intestinal)?|obstipacao", "intestino preso", r"intestino preso|prisao de ventre"),
    _t(r"flatulencias?", "gases", r"gases"),
    _t(r"gastroenterites?", "inflamação do estômago e do intestino"),
    _t(r"disenterias?(?: amebiana)?", "diarreia com sangue ou muco"),
    _t(r"helmintiases?", "verminose", r"verminoses?"),
    _t(r"anti-?helmintic[oa]s?", "contra vermes", r"contra (?:os )?vermes"),
    _t(r"amebiase", "infecção por ameba"),
    _t(r"giardiase", "infecção por giárdia"),
    # Infecção
    _t(r"candidiase(?: vaginal| vulvovaginal)?", "infecção por fungo (cândida)", r"infecc(?:ao|oes) por fungos?"),
    _t(r"micoses?", None, r"infecc(?:ao|oes) por fungos?"),
    _t(r"antifungic[oa]s?|antimicotic[oa]s?", "contra fungos", r"contra (?:os )?fungos?"),
    _t(r"vaginite", "inflamação na vagina"),
    _t(r"cistites?", "inflamação da bexiga"),
    _t(r"bactericidas?", "mata bactérias"),
    # Coração, pressão e sangue
    _t(r"hipertensao(?: arterial)?(?: essencial)?", "pressão alta", r"pressao alta"),
    _t(r"hipotensao (?:postural|ortostatica)", "pressão que cai ao levantar", r"pressao baixa"),
    _t(r"hipotensao", "pressão baixa", r"pressao baixa"),
    _t(r"insuficiencia cardiaca(?: congestiva)?", "o coração não bombeia o sangue como deveria"),
    _t(r"cardiovascula(?:r|res)", "do coração e dos vasos sanguíneos"),
    _t(r"taquicardias?", "coração acelerado", r"coracao acelerado|batimentos? acelerados?"),
    _t(r"bradicardias?", "coração lento", r"coracao lento|batimentos? lentos?"),
    _t(r"palpitac(?:ao|oes)", "sentir o coração bater forte ou rápido"),
    _t(r"tromboses?", "coágulo dentro de um vaso sanguíneo"),
    _t(r"agregacao plaquetaria", "as plaquetas se juntarem para formar o coágulo"),
    _t(r"hemorragias?", "sangramento", r"sangramentos?"),
    _t(r"equimoses?", "mancha roxa na pele", r"manchas? roxas?"),
    _t(r"hematuria", "sangue na urina", r"sangue na urina"),
    _t(r"hipoglicemia", "açúcar baixo no sangue", r"acucar baixo"),
    _t(r"hiperglicemia", "açúcar alto no sangue", r"acucar alto"),
    _t(r"hipercalemia|hiperpotassemia", "potássio alto no sangue", r"potassio alto"),
    _t(r"hipocalemia|hipopotassemia", "potássio baixo no sangue", r"potassio baixo"),
    # Fígado e rins
    _t(r"insuficiencia hepatica", "o fígado não funciona bem", r"problemas? (?:no|de) figado|doencas? (?:no|do) figado"),
    _t(r"insuficiencia renal", "os rins não funcionam bem", r"problemas? (?:nos|de) rins?|doencas? (?:nos|dos) rins"),
    _t(r"hepatotoxicidade", "dano ao fígado"),
    _t(r"nefrotoxicidade", "dano aos rins"),
    _t(r"ictericia", "pele e olhos amarelados", r"pele amarelada|olhos amarelados"),
    # Nervoso
    _t(r"vertigens?", "tontura, sensação de que tudo gira", r"tonturas?"),
    # Gravidez, idade
    _t(r"gestantes?", "grávidas", r"gravidas?"),
    _t(r"lactantes?", "mulheres que amamentam", r"amament[a-z]*"),
    _t(r"lactacao", "amamentação", r"amamentac(?:ao|oes)"),
    _t(r"lactentes?", "bebês", r"bebes?"),
    _t(r"neonatos?", "recém-nascidos", r"recem-?nascidos?"),
    _t(r"pediatric[oa]s?", "crianças", r"criancas?"),
    _t(r"geriatric[oa]s?", "idosos", r"idos[oa]s?"),
    _t(r"teratogenic[oa]s?|teratogenicidade", "pode causar malformação no bebê"),
    _t(r"contraceptiv[oa]s?|anticoncepciona(?:l|is)", "para evitar gravidez", r"evit[a-z]* (?:a )?gravidez"),
    _t(r"profilaxia", "prevenção", r"prevenc(?:ao|oes)|prevenir"),
    # Uso
    _t(r"via oral", "pela boca", r"pela boca"),
    _t(r"concomitante(?:mente)?", "ao mesmo tempo", r"ao mesmo tempo|juntos?"),
    _t(r"osteoporose", "ossos fracos, que quebram com facilidade"),
)


def dobrar(texto: str) -> str:
    """Minúsculo e sem acento, letra a letra, sem mudar o tamanho: a posição no texto dobrado é a mesma
    no original (o Python conta caracteres como a tela conta com Array.from)."""
    return "".join(unicodedata.normalize("NFKD", c)[:1].lower()[:1] or c for c in texto)


@dataclass(frozen=True)
class Annotation:
    inicio: int  # posição do termo técnico no texto (caracteres)
    fim: int  # exclusivo: a explicação vai logo depois daqui
    termo: str  # como está no texto
    explicacao: str


def anotar(texto: str, limite: int = MAX_ANOTACOES) -> list[Annotation]:
    """Os termos técnicos do texto que ganham explicação na tela, a primeira vez que aparecem.

    Fica sem explicação o termo que a própria bula já explica: entre parênteses ("nariz escorrendo
    (coriza)"), seguido de parênteses ("dermatite de fraldas (assaduras)") ou com a explicação escrita
    em outro lugar do texto."""
    dobrado = dobrar(texto)
    ocupados: list[tuple[int, int]] = []
    achados: list[Annotation] = []
    for termo in TERMOS:
        if termo.explicacao is None:
            continue
        ja_escrita = dobrar(termo.explicacao) in dobrado
        for m in termo.tecnico.finditer(dobrado):
            i, f = m.span()
            if any(i < fim and f > ini for ini, fim in ocupados):
                continue
            # Ocupado mesmo quando a bula já explica: "dermatite de fraldas (assaduras)" não pode
            # ganhar, pela entrada geral, "dermatite (inflamação da pele)".
            ocupados.append((i, f))
            entre_parenteses = dobrado[:i].rstrip().endswith("(") or dobrado[f:].lstrip().startswith("(")
            if not ja_escrita and not entre_parenteses:
                achados.append(Annotation(i, f, texto[i:f], termo.explicacao))
            break
    return sorted(achados, key=lambda a: a.inicio)[:limite]


def dicas(texto: str, limite: int = MAX_ANOTACOES) -> list[tuple[str, str]]:
    """(termo como a bula escreve, palavra do dia a dia), para o prompt: o modelo que troca o termo pela
    palavra do glossário passa na fundamentação."""
    return [(a.termo, a.explicacao) for a in anotar(texto, limite)]


def cobertos(afirmacao: str, trecho: str) -> set[str]:
    """As palavras da afirmação que contam como estando no trecho, porque fazem parte de uma expressão do
    dia a dia cujo termo técnico o trecho tem: "dores de cabeça" contra "cefaleia" -> {"dores", "cabeca"}.

    Os dois textos já minúsculos e sem acento (grounding._preparar ou `dobrar`)."""
    saida: set[str] = set()
    for termo in TERMOS:
        if termo.comum is None:
            continue
        achados = [m.group() for m in termo.comum.finditer(afirmacao)]
        if achados and termo.tecnico.search(trecho):
            saida |= {p for a in achados for p in _PALAVRA.findall(a)}
    return saida
