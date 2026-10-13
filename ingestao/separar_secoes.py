"""Separa o texto de uma bula (uma string por página) nas seções do contrato.

Os títulos são padronizados pela ANVISA (RDC 47/2009), mas cada fabricante varia numeração,
acento e caixa ("1. INDICAÇÕES", "1 - Indicações", "CONTRA-INDICAÇÕES"). Só conta como título a
linha inteira, numerada ou toda em maiúsculas, e as seções só andam para a frente: assim um
subtítulo "Posologia" ou uma remissão a uma seção anterior não abre seção nova.
"""

import math
import re
from collections import Counter

from app.schemas.dataset import SecaoBula
from app.services.medicamentos import normalizar

# (ordem, seção, título normalizado). Primeiro a bula do profissional, depois a do paciente.
_TITULOS: list[tuple[int, str, str]] = [
    (-3, "outras", r"identificacao do (?:medicamento|produto)"),
    (-2, "outras", r"apresentac(?:ao|oes)"),
    (-1, "composicao", r"composic(?:ao|oes)"),
    (0, "outras", r"informacoes (?:tecnicas )?ao(?:s)? (?:profissiona(?:l|is) de saude|paciente)"),
    (1, "indicacoes", r"indicac(?:ao|oes)"),
    (2, "resultados_eficacia", r"resultados? de eficacia"),
    (3, "caracteristicas_farmacologicas", r"caracteristicas farmacologicas"),
    (4, "contraindicacoes", r"contra ?indicac(?:ao|oes)"),
    (5, "advertencias", r"advertencias(?: e precaucoes)?|precaucoes e advertencias"),
    (6, "interacoes", r"interac(?:ao|oes) medicamentosas?"),
    (7, "armazenamento", r"cuidados de (?:armazenamento|conservacao)(?: do medicamento)?"),
    (8, "posologia", r"posologia(?: e modo de usar)?|modo de usar e posologia"),
    (9, "reacoes_adversas", r"reac(?:ao|oes) adversas(?: a medicamentos)?"),
    (10, "superdose", r"superdos(?:e|agem)"),
    (99, "outras", r"dizeres legais"),
    # Títulos longos do paciente quebram em duas linhas no PDF: basta o começo.
    (1, "indicacoes", r"para que este medicamento e indicado"),
    (2, "caracteristicas_farmacologicas", r"como este medicamento funciona"),
    (3, "contraindicacoes", r"quando nao devo usar este medicamento"),
    (4, "advertencias", r"o que devo saber antes de usar.*"),
    (5, "armazenamento", r"onde como e por quanto tempo posso guardar.*"),
    (6, "posologia", r"como devo usar este medicamento"),
    (7, "posologia", r"o que devo fazer quando eu me esquecer.*"),
    (8, "reacoes_adversas", r"quais os males que este medicamento.*"),
    (9, "superdose", r"o que fazer se alguem usar uma quantidade maior.*"),
]
_TITULOS_RE = [(ordem, secao, re.compile(rf"(\d{{1,2}} )?(?:{padrao})")) for ordem, secao, padrao in _TITULOS]
_SO_PAGINA = re.compile(r"(?:pagina |pag )?\d{1,3}(?: de \d{1,3}| \d{1,3})?")
_MAX_TITULO = 100
_BORDA = 3  # linhas do topo e do pé da página onde procuramos cabeçalho/rodapé
_FIM_DE_FRASE = ".:;!?"

ESPERADAS = {
    "indicacoes", "contraindicacoes", "advertencias", "interacoes", "armazenamento", "posologia",
    "reacoes_adversas", "superdose",
}


def _titulo(linha: str) -> tuple[int, str] | None:
    """(ordem, seção) se a linha é um título de seção."""
    if not linha or len(linha) > _MAX_TITULO:
        return None
    norm = normalizar(linha)
    for ordem, secao, padrao in _TITULOS_RE:
        m = padrao.fullmatch(norm)
        if m and (m.group(1) or linha == linha.upper()):
            return ordem, secao
    return None


def _molde(linha: str) -> str:
    return re.sub(r"\d+", "#", normalizar(linha))


def limpar_paginas(paginas: list[str]) -> list[list[str]]:
    """Linhas de cada página, sem cabeçalho/rodapé repetido nem número de página.

    Só mexe nas bordas da página: no meio, uma linha repetida pode ser conteúdo (tabela).
    Linha vazia fica, porque separa parágrafos.
    """
    linhas = [[l.strip() for l in p.splitlines()] for p in paginas]

    def bordas(pagina: list[str]) -> set[int]:
        cheias = [i for i, l in enumerate(pagina) if l]
        return set(cheias[:_BORDA] + cheias[-_BORDA:])

    vezes = Counter(m for pag in linhas for m in {_molde(pag[i]) for i in bordas(pag)})
    minimo = max(3, math.ceil(len(paginas) * 0.6))
    repetidas = {m for m, n in vezes.items() if n >= minimo}

    limpas = []
    for pag in linhas:
        borda = bordas(pag)
        limpas.append([
            l for i, l in enumerate(pag)
            if not (i in borda and not _titulo(l) and (_molde(l) in repetidas or _SO_PAGINA.fullmatch(normalizar(l))))
        ])
    return limpas


def _juntar(linhas: list[tuple[int, str]]) -> str:
    """Desfaz a quebra de linha do PDF: "\\n" entre parágrafos, "\\f" na virada de página."""
    texto = ""
    pagina_anterior = None
    paragrafo_novo = False
    for pagina, linha in linhas:
        if not linha:
            paragrafo_novo = True
            continue
        if texto:
            if pagina != pagina_anterior:
                texto += "\f"
            elif paragrafo_novo or texto[-1] in _FIM_DE_FRASE:
                texto += "\n"
            else:
                texto += " "
        texto += linha
        pagina_anterior, paragrafo_novo = pagina, False
    return texto


def separar_secoes(paginas: list[str]) -> list[SecaoBula]:
    secoes: list[SecaoBula] = []
    ordem_atual, secao_atual, titulo_atual = -100, "outras", "(antes da primeira seção)"
    corpo: list[tuple[int, str]] = []

    def fechar() -> None:
        com_texto = [p for p, l in corpo if l]
        if com_texto:
            secoes.append(SecaoBula(
                secao=secao_atual, titulo_original=titulo_atual,
                pagina_inicio=com_texto[0], pagina_fim=com_texto[-1], texto=_juntar(corpo),
            ))

    for numero, pagina in enumerate(limpar_paginas(paginas), start=1):
        for linha in pagina:
            achado = _titulo(linha)
            if achado and achado[0] > ordem_atual:
                fechar()
                (ordem_atual, secao_atual), titulo_atual, corpo = achado, linha, []
            else:
                corpo.append((numero, linha))
    fechar()
    return secoes
