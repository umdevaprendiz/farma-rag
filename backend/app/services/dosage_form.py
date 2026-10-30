"""Mais de uma apresentação do mesmo remédio no acervo: qual bula a pergunta quer (docs/DECISOES.md §41).

O acervo tem bulas diferentes do mesmo princípio ativo: paracetamol em gotas, comprimido de 750 mg e sachê
(Tylalgin); permetrina em loção e loção cremosa; AAS de 100 e de 500 mg; isotretinoína de 10 e de 20 mg;
colecalciferol em duas bulas de doses diferentes. Para dose, modo de usar, superdose, armazenamento e uso em
criança a resposta muda de uma para outra, e o modelo pequeno, com trechos de duas bulas, pode juntar a dose
de uma com a forma da outra. Sem LLM:
- `escolhidas`: a pergunta já diz qual ("Tylalgin gotas", "AAS 100", "loção cremosa"): a busca fica nela;
- `depende`: a pergunta é de um assunto em que a apresentação muda a resposta;
- `esclarecimento`: a pergunta de volta, com um botão por bula, que reenvia a pergunta com o `bula_id`.
O grafo só pergunta quando a resposta iria para o LLM: a dose copiada da bula já vem uma frase por
apresentação (§25), e para que serve, interação e contraindicação não mudam com a forma.
"""

import re
from typing import Protocol

from app.services.medicines import normalizar
from app.services.rag import Clarification, Option

# Seções em que a resposta muda com a apresentação (intent.inferir_secoes).
SECOES = frozenset({"posologia", "superdose", "armazenamento"})
# "bebê" só como substantivo: em "cliente bebe álcool" é o verbo (o normalizar tira o acento).
_CRIANCA = re.compile(
    r"\b(?:crianc\w*|infant\w*|bebes|pediatr\w*|lactentes?|recem nascid\w*|filh[oa]s?)\b"
    r"|\b(?:o|um|meu|minha|seu|sua|para|pra|de|do|no|ao) bebe\b"
)
_MODO_DE_USAR = re.compile(r"\b(?:aplic\w*|dilu\w*|dissolv\w*|prepar\w*|misturar na agua)\b")

# Formas do mesmo jeito de dizer, depois do plural tirado ("gotinhas" -> "gotinha" -> "gota").
_SINONIMOS = {
    "sachet": "sache", "envelope": "sache", "gotinha": "gota", "cp": "comprimido", "comp": "comprimido",
    "comprimidinho": "comprimido", "capsul": "capsula", "pilula": "comprimido",
}
# Não distinguem uma bula de outra: unidade sozinha, ligação e o que toda forma oral tem.
_IGNORAR = {"mg", "ml", "ui", "mcg", "g", "de", "da", "do", "para", "com", "e", "oral", "uso", "por"}
# Palavras de forma farmacêutica no nome da bula ("Tylalgin 200 mg/mL gotas"), depois de _palavra.
_FORMAS = {
    "gota", "comprimido", "capsula", "sache", "locao", "cremosa", "creme", "gel", "pomada", "xarope",
    "solucao", "suspensao", "spray", "aerossol", "injetavel", "ampola", "supositorio", "efervescente",
    "mastigavel", "revestido", "mole", "dura", "po", "granulado", "emulsao", "colirio", "shampoo",
    "xampu", "adesivo", "ovulo", "pastilha", "infantil", "pediatrico", "adulto",
}
# Número que não é da apresentação: "criança de 10 kg", "há 3 dias", "500 mg a cada 6 horas" (o 6).
_NAO_E_APRESENTACAO = re.compile(r"^(?:kg|kgs|kilos?|quilos?|anos?|mes|meses|dias?|horas?|h|vezes|x|semanas?)$")


class _LeafletLike(Protocol):
    id: str
    medicamento: str
    principio_ativo: str
    forma_farmaceutica: str | None


def _palavra(p: str) -> str:
    if len(p) > 4 and p.endswith("s") and not p.isdigit():
        p = p[:-1]
    return _SINONIMOS.get(p, p)


def _sem_milhar(texto: str) -> str:
    """"5.000 UI" -> "5000 UI" (o normalizar faria "5 000")."""
    return re.sub(r"(?<=\d)\.(?=\d{3}(?!\d))", "", texto)


def palavras(texto: str) -> set[str]:
    """As palavras e números da apresentação, como a comparação usa."""
    return {_palavra(p) for p in normalizar(_sem_milhar(texto)).split() if p not in _IGNORAR}


def _palavras_da_pergunta(pergunta: str) -> set[str]:
    """Como `palavras`, sem os números que medem outra coisa (peso, idade, intervalo)."""
    tokens = normalizar(_sem_milhar(pergunta)).split()
    saida = set()
    for i, p in enumerate(tokens):
        if p.isdigit() and i + 1 < len(tokens) and _NAO_E_APRESENTACAO.match(tokens[i + 1]):
            continue
        if p not in _IGNORAR:
            saida.add(_palavra(p))
    return saida


def _da_bula(b: _LeafletLike) -> set[str]:
    return palavras(f"{b.medicamento} {b.forma_farmaceutica or ''}")


def apresentacao(b: _LeafletLike) -> frozenset[str]:
    """O que diz a apresentação: a forma farmacêutica e, do nome, os números e as palavras de forma. Marca e
    fabricante ficam de fora: o Exemplol 50 mg de dois laboratórios é a mesma apresentação."""
    do_nome = {p for p in palavras(b.medicamento) if p.isdigit() or p in _FORMAS}
    return frozenset(palavras(b.forma_farmaceutica or "") | do_nome)


def diferentes(bulas: list[_LeafletLike]) -> bool:
    """As bulas têm mais de uma apresentação (perguntar faz sentido). Mesmo produto de dois laboratórios, não."""
    return len({apresentacao(b) for b in bulas}) > 1


def escolhidas(pergunta: str, bulas: list[_LeafletLike]) -> list[_LeafletLike]:
    """As bulas que a pergunta nomeia pelo que só elas têm ("gotas", "750", "cremosa", "sachê"). Vazio =
    a pergunta não escolheu (ou nomeou todas): quem decide é quem chama."""
    if len(bulas) < 2:
        return []
    proprias = {b.id: _da_bula(b) for b in bulas}
    comuns = set.intersection(*proprias.values())
    pedidas = _palavras_da_pergunta(pergunta)
    achadas = [b for b in bulas if (proprias[b.id] - comuns) & pedidas]
    return achadas if len(achadas) < len(bulas) else []


def depende(texto: str, secoes: list[str]) -> bool:
    """A pergunta é de um assunto em que a resposta muda com a apresentação: dose e modo de usar, superdose,
    armazenamento, uso em criança."""
    norm = normalizar(texto)
    return bool(SECOES & set(secoes)) or bool(_CRIANCA.search(norm)) or bool(_MODO_DE_USAR.search(norm))


def rotulo(b: _LeafletLike) -> str:
    """O nome da bula, com a forma entre parênteses quando o nome não a diz ("Tylalgin 750 mg (comprimido)")."""
    forma = b.forma_farmaceutica or ""
    if not forma or palavras(forma) & palavras(b.medicamento):
        return b.medicamento
    return f"{b.medicamento} ({forma})"


def esclarecimento(nome: str, bulas: list[_LeafletLike]) -> Clarification:
    """A pergunta de volta: um botão por bula, que reenvia a pergunta com o `bula_id` dela."""
    return Clarification(
        "apresentacao",
        f"Tenho {len(bulas)} bulas de {nome}, e a resposta muda conforme a apresentação. Qual delas?",
        [Option(rotulo(b), [b.principio_ativo], b.id) for b in bulas],
    )
