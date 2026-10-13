"""Detecta medicamentos na pergunta antes de chamar o LLM.

Três resultados possíveis por pergunta:
- medicamentos indexados encontrados -> busca filtrada por eles;
- só medicamentos conhecidos, mas sem bula indexada -> nao_encontrada, sem chamar o LLM;
- nenhum medicamento reconhecido -> segue sem filtro e o limiar de similaridade decide.
"""

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

# Palavras de sal/hidratação que o catálogo da ANVISA junta ao princípio ativo
# ("cloridrato de loperamida", "diclofenaco sódico", "amoxicilina tri-hidratada"),
# mas que ninguém diz na pergunta. Já normalizadas (sem acento, sem hífen).
_SAIS = {
    "de", "da", "do", "e", "cloridrato", "dicloridrato", "maleato", "sulfato", "fumarato",
    "succinato", "tartarato", "acetato", "citrato", "nitrato", "mesilato", "fosfato", "besilato",
    "bromidrato", "hidrobrometo", "brometo", "butilbrometo", "sodico", "sodica", "potassico",
    "potassica", "calcico", "dietilamonio", "arginina", "micronizado", "micronizada",
    "microgranulados", "monoidratado", "monoidratada", "hidratado", "hidratada", "tri", "di",
    "tetra", "hepta", "triidratado", "heptaidratado", "heptaidratada", "anidro", "anidra",
}
# Palavras de nome químico que, sozinhas, não identificam o medicamento.
_PALAVRAS_GENERICAS = {"acido", "vitamina", "sodio", "potassio", "calcio"}
_MIN_ALIAS_PALAVRA = 5
_MIN_ALIAS_FUZZY = 6
_CUTOFF_FUZZY = 0.88


def normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", sem_acento.lower())).strip()


class _BulaLike(Protocol):
    principio_ativo: str
    nomes_comerciais: list[str]


def _aliases(nome: str, principio_ativo: bool = True) -> set[str]:
    """Formas pelas quais o nome pode aparecer numa pergunta.

    "cloridrato de loperamida" -> {"cloridrato de loperamida", "loperamida"}.
    Para princípios ativos, também cada palavra específica ("ácido ascórbico" -> "ascorbico").
    Marcas (principio_ativo=False) só ganham a forma sem sal: quebrar "ADVIL FORTE" em palavras
    geraria falsos positivos ("forte").
    """
    completo = normalizar(nome)
    if not completo:
        return set()
    aliases = {completo}
    nucleo = [t for t in completo.split(" ") if t not in _SAIS]
    if nucleo:
        aliases.add(" ".join(nucleo))
    if principio_ativo and len(nucleo) > 1:
        aliases.update(t for t in nucleo if len(t) >= _MIN_ALIAS_PALAVRA and t not in _PALAVRAS_GENERICAS)
    return aliases


@dataclass
class Catalogo:
    # alias normalizado -> princípio ativo (normalizado)
    indexados: dict[str, str] = field(default_factory=dict)
    # alias normalizado -> nome conhecido que NÃO tem bula indexada
    nao_indexados: dict[str, str] = field(default_factory=dict)

    @classmethod
    def montar(cls, bulas: Iterable[_BulaLike], conhecidos: Iterable[str] = ()) -> "Catalogo":
        cat = cls()
        for b in bulas:
            ativo = normalizar(b.principio_ativo)
            for nome in [b.principio_ativo, *b.nomes_comerciais]:
                for alias in _aliases(nome, principio_ativo=(nome == b.principio_ativo)):
                    cat.indexados[alias] = ativo
        for nome in conhecidos:
            for alias in _aliases(nome):
                if alias not in cat.indexados:
                    cat.nao_indexados[alias] = normalizar(nome)
        return cat


def carregar_conhecidos(caminho: str | Path) -> list[str]:
    """Lê a lista de medicamentos conhecidos (um por linha; '#' comenta)."""
    p = Path(caminho)
    if not p.exists():
        return []
    linhas = (l.strip() for l in p.read_text(encoding="utf-8").splitlines())
    return [l for l in linhas if l and not l.startswith("#")]


@dataclass(frozen=True)
class Deteccao:
    indexados: list[str]  # princípios ativos (normalizados) com bula indexada
    nao_indexados: list[str]  # nomes conhecidos sem bula indexada


def _achar(pergunta_norm: str, tokens: list[str], aliases: dict[str, str]) -> set[str]:
    achados: set[str] = set()
    for alias, alvo in aliases.items():
        if re.search(rf"\b{re.escape(alias)}\b", pergunta_norm):
            achados.add(alvo)
    # Erros de digitação: só para aliases de uma palavra e longos, para evitar falso positivo.
    unicos = {a: t for a, t in aliases.items() if " " not in a and len(a) >= _MIN_ALIAS_FUZZY}
    for token in tokens:
        if len(token) < _MIN_ALIAS_FUZZY:
            continue
        for alias in difflib.get_close_matches(token, unicos, n=1, cutoff=_CUTOFF_FUZZY):
            achados.add(unicos[alias])
    return achados


def detectar(pergunta: str, catalogo: Catalogo) -> Deteccao:
    pergunta_norm = normalizar(pergunta)
    tokens = pergunta_norm.split(" ")
    return Deteccao(
        indexados=sorted(_achar(pergunta_norm, tokens, catalogo.indexados)),
        nao_indexados=sorted(_achar(pergunta_norm, tokens, catalogo.nao_indexados)),
    )
