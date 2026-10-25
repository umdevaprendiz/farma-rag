"""A resposta menos robótica, da abertura ao fim, sem LLM (F4, docs/DECISOES.md, seção 37).

Pedido do usuário (2026-10-24): "que a resposta inteira seja menos robotizada, da abertura até o final".
O que o modelo escreve (ou a frase copiada da bula) continua igual e conferido; em volta dela, o código
monta o que deixa a resposta com cara de conversa:
- **abertura**: uma frase antes da resposta, pelo caminho que a pergunta seguiu ("Pela bula de
  Ibuprofeno 400 mg:", "Para quem tem alergia a dipirona, pelas bulas do acervo:"). Pergunta de
  opinião ("o que acha...?") ganha "Opinião eu não dou...". Na de para que serve, a conclusão de
  services/indication.py já abre a resposta;
- **marcas** na resposta, como posições (a frase não muda): o que foi perguntado em negrito ("foco") e
  a explicação do termo técnico pelo glossário ("termo", services/glossary.py);
- **sugestões** de pergunta seguinte sobre o mesmo remédio, que a tela mostra como botões no fim.
Nada disso passa pelo LLM nem pela fundamentação: é texto fixo do código ou posição no texto aprovado.
"""

import re
from dataclasses import dataclass
from typing import Literal

from app.services import glossary
from app.services.medicines import normalizar

# Por onde a pergunta passou no `buscar` (services/graph.py).
Route = Literal["indicacao", "dose", "interacao", "alternativas", "geral", "remedio"]

# Palavras em negrito por resposta, no máximo; e a fração do texto que elas podem cobrir (mais que
# isso, o negrito deixa de mostrar o que importa).
MAX_FOCO = 6
MAX_FRACAO_FOCO = 0.5

_OPINIAO = re.compile(
    r"\b(?:o que|oque|q) (?:voce |vc |tu )?(?:acha|achas|pensa)\b|\bvale a pena\b"
    r"|\b(?:voce|vc) (?:recomenda|indica|aconselha)\b|\bna sua opiniao\b"
)
_MELHOR = re.compile(r"\bmelhor(?:es)?\b")
ABERTURA_OPINIAO = "Opinião eu não dou, mas posso dizer o que a bula diz."


def pede_opiniao(pergunta: str) -> bool:
    """"O que acha do ibuprofeno pra dor de cabeça?": a resposta diz o que a bula diz, não opina."""
    return bool(_OPINIAO.search(normalizar(pergunta)))


def em_lista(nomes: list[str]) -> str:
    """["a"] -> "a"; ["a", "b", "c"] -> "a, b e c"."""
    return nomes[0] if len(nomes) == 1 else f"{', '.join(nomes[:-1])} e {nomes[-1]}"


def _pela(medicamentos: list[str], nomes: list[str]) -> str:
    """De onde vem a resposta: uma bula pelo nome dela; várias do mesmo remédio ou de vários, pelo
    princípio ativo."""
    if len(medicamentos) == 1:
        return f"pela bula de {medicamentos[0]}"
    return f"pelas bulas de {em_lista(nomes)}"


def abertura(
    caminho: Route,
    pergunta: str,
    medicamentos: list[str],
    nomes: list[str],
    alergia_a: str | None = None,
    nomes_da_pergunta: list[str] = (),
    sem_bula: bool = False,
) -> str | None:
    """A frase antes da resposta. `medicamentos`: as bulas citadas, na ordem; `nomes`: os princípios
    ativos delas, sem o sal; `nomes_da_pergunta`: os remédios da pergunta de interação."""
    opiniao = ABERTURA_OPINIAO if pede_opiniao(pergunta) else None
    if caminho == "indicacao" or not medicamentos:
        return opiniao  # a conclusão (services/indication.py) já abre a resposta
    if caminho == "alternativas":
        base = f"Para quem tem alergia a {alergia_a}, pelas bulas do acervo:"
    elif caminho == "interacao" and len(nomes_da_pergunta) >= 2:
        juntos = (
            f"{nomes_da_pergunta[0]} com {nomes_da_pergunta[1]}" if len(nomes_da_pergunta) == 2
            else f"{em_lista(nomes_da_pergunta)} juntos"
        )
        base = f"Sobre usar {juntos}, {_pela(medicamentos, nomes)}:"
    elif caminho == "dose":
        base = f"Pela posologia da bula de {medicamentos[0]}:" if len(medicamentos) == 1 else "Pela posologia de cada bula:"
    elif caminho == "geral":
        if sem_bula:
            base = "O que as outras bulas do acervo dizem:"
        elif _MELHOR.search(normalizar(pergunta)):
            base = "As bulas não dizem qual é o melhor, mas dizem isto:"
        else:
            base = "Pelas bulas do acervo:"
    else:
        pela = _pela(medicamentos, nomes)
        base = f"{pela[0].upper()}{pela[1:]}:"
    return f"{opiniao} {base}" if opiniao else base


@dataclass(frozen=True)
class Mark:
    """Uma marca na resposta, em posições do texto (caracteres, como grounding.destaques)."""

    inicio: int
    fim: int
    tipo: Literal["foco", "termo"]  # foco: o que foi perguntado (negrito); termo: explicação do glossário
    explicacao: str | None = None  # só no termo: o que vai entre parênteses depois dele


# "Dor de cabeça", "dores nas costas": a dor vem junto com a palavra do assunto.
_DOR_ANTES = re.compile(r"\bdor(?:es)?(?: (?:de|do|da|no|na|nos|nas))? $")


def foco(texto: str, grupos: list[tuple[str, ...]], limite: int = MAX_FOCO) -> list[tuple[int, int]]:
    """As palavras da resposta que dizem o que foi perguntado: as que começam por um termo dos `grupos`
    (os do assunto da pergunta, com sinônimos: "cabeça" -> cabeça, cefaleia, enxaqueca). Palavras
    vizinhas viram uma marca só ("cólica menstrual"). Vazio se nada casa, ou se o negrito cobriria
    boa parte do texto (aí não destaca nada)."""
    termos = tuple(t for g in grupos for t in g)
    if not termos:
        return []
    dobrado = glossary.dobrar(texto)
    marcas: list[tuple[int, int]] = []
    for m in re.finditer(r"[a-z]+", dobrado):
        if len(m.group()) < 3 or not m.group().startswith(termos):
            continue
        inicio = m.start()
        if (antes := _DOR_ANTES.search(dobrado, 0, inicio)) and not m.group().startswith("dor"):
            inicio = antes.start()
        if marcas and inicio <= marcas[-1][1] + 1 and dobrado[marcas[-1][1]:inicio].strip() == "":
            marcas[-1] = (marcas[-1][0], m.end())
        else:
            marcas.append((inicio, m.end()))
    letras = sum(1 for c in dobrado if c.isalpha())
    if not marcas or sum(f - i for i, f in marcas) > MAX_FRACAO_FOCO * letras:
        return []
    return marcas[:limite]


def marcas(texto: str, grupos: list[tuple[str, ...]]) -> list[Mark]:
    """O foco e os termos explicados pelo glossário, em ordem de posição."""
    saida = [Mark(i, f, "foco") for i, f in foco(texto, grupos)]
    saida += [Mark(a.inicio, a.fim, "termo", a.explicacao) for a in glossary.anotar(texto)]
    return sorted(saida, key=lambda m: (m.inicio, m.tipo))


# Perguntas seguintes, na ordem em que aparecem; sai a da seção que a pergunta já pediu.
_SEGUINTES: tuple[tuple[str, str], ...] = (
    ("indicacoes", "Para que serve {nome}?"),
    ("posologia", "Qual a dose de {nome}?"),
    ("contraindicacoes", "Quem não pode tomar {nome}?"),
    ("reacoes_adversas", "Quais os efeitos colaterais de {nome}?"),
)
MAX_SUGESTOES = 3


def sugestoes(nome: str, secoes_da_pergunta: set[str], limite: int = MAX_SUGESTOES) -> list[str]:
    """Perguntas seguintes sobre o mesmo remédio (`nome`: o princípio ativo sem o sal), que o sistema
    responde bem: cada uma cai no caminho certo (para que serve e dose copiados da bula, contraindicação e
    efeitos pela seção). Sai a da seção que a pergunta já pediu."""
    return [p.format(nome=nome) for secao, p in _SEGUINTES if secao not in secoes_da_pergunta][:limite]
