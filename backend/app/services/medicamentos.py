"""Detecta medicamentos na pergunta antes de chamar o LLM.

Três resultados possíveis por pergunta:
- medicamentos indexados encontrados -> busca filtrada por eles;
- só medicamentos conhecidos, mas sem bula indexada -> nao_encontrada, sem chamar o LLM;
- nenhum medicamento reconhecido -> o grafo pede esclarecimento (services/grafo.py, nó esclarecer).
"""

import difflib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from functools import cached_property
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
# Hidratação e forma física: não mudam a bula ("amoxicilina" = "amoxicilina tri-hidratada").
# O sal muda (diclofenaco sódico e potássico têm bulas diferentes), por isso fica fora daqui.
_HIDRATACAO = re.compile(
    r"\b(?:(?:mono|di|tri|tetra|penta|hexa|hepta|deca|hemi|sesqui) )?"
    r"(?:\w*idratad[oa]|anidr[oa]|microniz\w*|microgranul\w*)\b"
)
# Palavras de nome químico que, sozinhas, não identificam o medicamento.
_PALAVRAS_GENERICAS = {"acido", "vitamina", "sodio", "potassio", "calcio"}
_MIN_ALIAS_PALAVRA = 5
_MIN_ALIAS_FUZZY = 6
_CUTOFF_FUZZY = 0.88


def normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", sem_acento.lower())).strip()


def chave_principio(nome: str) -> str:
    """Princípio ativo sem hidratação: "amoxicilina tri-hidratada" -> "amoxicilina"."""
    return re.sub(r"\s+", " ", _HIDRATACAO.sub(" ", normalizar(nome))).strip()


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
    nucleo = [t for t in chave_principio(nome).split(" ") if t and t not in _SAIS]
    if nucleo:
        aliases.add(" ".join(nucleo))
    if principio_ativo and len(nucleo) > 1:
        aliases.update(t for t in nucleo if len(t) >= _MIN_ALIAS_PALAVRA and t not in _PALAVRAS_GENERICAS)
    return aliases


def _serve(alias: str, ignorar: set[str]) -> bool:
    """Alias tirado do catálogo da ANVISA sem revisão humana: curto ou genérico vira falso positivo."""
    return (
        len(alias.replace(" ", "")) >= _MIN_ALIAS_PALAVRA
        and alias not in _PALAVRAS_GENERICAS
        and alias not in ignorar
    )


def _contem(tokens: list[str], aliases: dict[str, str], max_palavras: int) -> bool:
    return any(
        " ".join(tokens[i : i + n]) in aliases
        for n in range(1, min(max_palavras, len(tokens)) + 1)
        for i in range(len(tokens) - n + 1)
    )


@dataclass
class Catalogo:
    # alias normalizado -> princípio ativo (normalizado)
    indexados: dict[str, str] = field(default_factory=dict)
    # alias normalizado -> nome conhecido que NÃO tem bula indexada
    nao_indexados: dict[str, str] = field(default_factory=dict)
    # Marcas vindas do catálogo da ANVISA: só casam escritas certo. Marca imita sintoma
    # ("INTESTIN"), e tolerar erro de digitação faria "intestino" virar loperamida.
    so_exato: set[str] = field(default_factory=set)

    @cached_property
    def max_palavras(self) -> int:
        return max((a.count(" ") + 1 for a in (*self.indexados, *self.nao_indexados)), default=1)

    @cached_property
    def candidatos_fuzzy(self) -> list[str]:
        # Só aliases de uma palavra e longos, para evitar falso positivo.
        return [
            a
            for a in (*self.indexados, *self.nao_indexados)
            if " " not in a and len(a) >= _MIN_ALIAS_FUZZY and a not in self.so_exato
        ]

    @classmethod
    def montar(
        cls,
        bulas: Iterable[_BulaLike],
        conhecidos: Iterable[str] = (),
        anvisa: Iterable[dict] = (),
        ignorados: Iterable[str] = (),
    ) -> "Catalogo":
        """`anvisa`: itens de catalogo_anvisa.jsonl; `ignorados`: aliases do catálogo a descartar.

        Do catálogo entram, para os indexados, as marcas de produtos só com aquele princípio
        ativo; para os demais, só o princípio ativo (marca curta ou ambígua, como "CORUS" ou
        "DIP", daria falso positivo em pergunta comum). Ver docs/DECISOES.md, seção 13.
        """
        indexados: dict[str, str] = {}
        por_chave: dict[str, str] = {}  # princípio sem hidratação -> princípio ativo indexado
        for b in bulas:
            ativo = normalizar(b.principio_ativo)
            por_chave[chave_principio(b.principio_ativo)] = ativo
            for nome in [b.principio_ativo, *b.nomes_comerciais]:
                for alias in _aliases(nome, principio_ativo=(nome == b.principio_ativo)):
                    indexados[alias] = ativo

        ignorar = {normalizar(i) for i in ignorados}
        marcas: dict[str, set[str]] = {}  # alias de marca -> princípios indexados que a usam
        outros: dict[str, dict] = {}  # nome do produto -> item que NÃO é de princípio indexado
        for item in anvisa:
            ativo = None if item["associacao"] else por_chave.get(chave_principio(item["principios_ativos"][0]))
            if ativo is None:
                outros[normalizar(item["nome_produto"])] = item
                continue
            for alias in _aliases(item["principios_ativos"][0]):
                indexados.setdefault(alias, ativo)
            for alias in _aliases(item["nome_produto"], principio_ativo=False):
                if _serve(alias, ignorar):
                    marcas.setdefault(alias, set()).add(ativo)
        so_exato: set[str] = set()
        for alias, ativos in marcas.items():
            # Mesma marca em dois princípios ativos, ou também numa associação: ambígua, fica fora.
            if len(ativos) == 1 and alias not in outros and alias not in indexados:
                indexados[alias] = next(iter(ativos))
                so_exato.add(alias)

        nao_indexados: dict[str, str] = {}
        for nome in conhecidos:
            for alias in _aliases(nome):
                if alias not in indexados:
                    nao_indexados[alias] = normalizar(nome)
        livres = {normalizar(i["principios_ativos"][0]) for i in outros.values() if not i["associacao"]}
        # Grafia mais curta primeiro: "hidrocortisona" responde por "acetato de hidrocortisona".
        for principio in sorted(livres, key=lambda p: (len(p), p)):
            for alias in _aliases(principio, principio_ativo=False):
                if alias not in indexados and _serve(alias, ignorar):
                    nao_indexados.setdefault(alias, principio)
        # Produto que leva o nome de um indexado mas é outro remédio ("BUSCOPAN COMPOSTO" tem
        # dipirona): vira não indexado, e na pergunta o nome mais longo vence (ver detectar).
        max_palavras = max((a.count(" ") + 1 for a in indexados), default=1)
        for nome in outros:
            if nome not in indexados and _contem(nome.split(" "), indexados, max_palavras):
                nao_indexados.setdefault(nome, nome)
        return cls(indexados, nao_indexados, so_exato)


def carregar_conhecidos(caminho: str | Path) -> list[str]:
    """Lê a lista de medicamentos conhecidos (um por linha; '#' comenta)."""
    p = Path(caminho)
    if not p.exists():
        return []
    linhas = (l.strip() for l in p.read_text(encoding="utf-8").splitlines())
    return [l for l in linhas if l and not l.startswith("#")]


def carregar_catalogo_anvisa(caminho: str | Path) -> list[dict]:
    """Lê catalogo_anvisa.jsonl (gerado por ingestao/catalogo_anvisa.py)."""
    p = Path(caminho)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


@dataclass(frozen=True)
class Deteccao:
    indexados: list[str]  # princípios ativos (normalizados) com bula indexada
    nao_indexados: list[str]  # nomes conhecidos sem bula indexada


def detectar(pergunta: str, catalogo: Catalogo) -> Deteccao:
    tokens = normalizar(pergunta).split(" ")
    indexados: set[str] = set()
    nao_indexados: set[str] = set()

    def registrar(alias: str) -> None:
        if alias in catalogo.indexados:
            indexados.add(catalogo.indexados[alias])
        else:
            nao_indexados.add(catalogo.nao_indexados[alias])

    i = 0
    while i < len(tokens):
        # Maior nome primeiro: "buscopan composto" não pode casar como "buscopan".
        for n in range(min(catalogo.max_palavras, len(tokens) - i), 0, -1):
            trecho = " ".join(tokens[i : i + n])
            if trecho in catalogo.indexados or trecho in catalogo.nao_indexados:
                registrar(trecho)
                break
        else:
            n = 1
            # Erro de digitação: só em palavra longa que não é, ela mesma, um nome conhecido
            # ("prednisona" não pode virar "prednisolona").
            if len(tokens[i]) >= _MIN_ALIAS_FUZZY:
                for alias in difflib.get_close_matches(
                    tokens[i], catalogo.candidatos_fuzzy, n=1, cutoff=_CUTOFF_FUZZY
                ):
                    registrar(alias)
        i += n
    return Deteccao(indexados=sorted(indexados), nao_indexados=sorted(nao_indexados))
