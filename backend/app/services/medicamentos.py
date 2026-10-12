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

# Primeiras palavras de nomes compostos que não identificam o medicamento sozinhas.
_PALAVRAS_GENERICAS = {"acido", "sulfato", "cloridrato", "sodio", "potassio", "calcio", "de", "e"}
_MIN_ALIAS_ABREVIADO = 5
_MIN_ALIAS_FUZZY = 6
_CUTOFF_FUZZY = 0.88


def normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", sem_acento.lower())).strip()


class _BulaLike(Protocol):
    principio_ativo: str
    nomes_comerciais: list[str]


def _aliases(nome: str) -> set[str]:
    """O nome inteiro e, em nomes compostos, a primeira palavra específica ("losartana")."""
    completo = normalizar(nome)
    aliases = {completo} if completo else set()
    primeira = completo.split(" ")[0] if completo else ""
    if " " in completo and len(primeira) >= _MIN_ALIAS_ABREVIADO and primeira not in _PALAVRAS_GENERICAS:
        aliases.add(primeira)
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
                for alias in _aliases(nome):
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
