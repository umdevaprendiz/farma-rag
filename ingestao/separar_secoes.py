"""Separa o texto de um PDF de bula (uma string por página) em bulas e cada bula em seções.

Os títulos são padronizados pela ANVISA (RDC 47/2009), mas cada fabricante varia numeração,
acento e caixa ("1. INDICAÇÕES", "1 - Indicações", "CONTRA-INDICAÇÕES"). Só conta como título a
linha inteira, numerada ou toda em maiúsculas, sem parênteses, e as seções só andam para a
frente: assim um subtítulo "Posologia" ou uma remissão a uma seção anterior não abre seção nova.

Um PDF pode trazer **mais de uma bula** (Tylalgin gotas + comprimido, Kaodine 10 mg/mL + 50 mg/mL).
Cada bula termina na tabela "Histórico de Alteração da Bula" (exigida pela ANVISA), que não é
conteúdo clínico e é descartada; a próxima começa em "MODELO DE BULA PARA O PROFISSIONAL...".
"""

import math
import re
from collections import Counter
from dataclasses import dataclass

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
    # "ADVERTÊNCIA E PRECAUÇÕES" no singular (Cisteil).
    (5, "advertencias", r"advertencias?(?: e precauc(?:ao|oes))?|precaucoes e advertencias"),
    (6, "interacoes", r"interac(?:ao|oes) medicamentosas?"),
    # "ARMANEZAMENTO" (Kaodine): erro de digitação na bula aprovada.
    (7, "armazenamento", r"cuidados de (?:arma\w+|conservacao)(?: do medicamento)?"),
    (8, "posologia", r"posologia(?: e modo de usar)?|modo de usar e posologia"),
    (9, "reacoes_adversas", r"reac(?:ao|oes) adversas(?: a medicamentos)?"),
    # "SUSPERDOSE" (Kaodine).
    (10, "superdose", r"sus?perdos(?:e|agem)"),
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
_FECHA_PARENTESE = re.compile(r"\)\W*$")
_BORDA = 3  # linhas do topo e do pé da página onde procuramos cabeçalho/rodapé
_FIM_DE_FRASE = ".:;!?"

# Rodapé que o pdfplumber cola no começo da linha seguinte ("V.14_02/20265. ADVERTÊNCIA...").
# O da Geolab é o código de versão da bula: guardado em `versao`.
_VERSAO_GEOLAB = re.compile(r"^V\.?\s?(\d{1,2})\s?_?\s?(\d{2})/(\d{4})")
_RODAPES_COLADOS = [
    _VERSAO_GEOLAB,
    re.compile(r"^Bula_Profissional de sa[uú]de \d+"),  # Vitamedic
    re.compile(r"^[A-Z]+_BU\d+_\d+_VPS ?\d*"),  # Organon ("DESALEX_BU27_112017_VPS 4")
]
_FIM_DE_BULA = re.compile(r"historico de alteracao")
_INICIO_DE_BULA = re.compile(r"modelo de bula (?:para|do) (?:o )?profissiona")
_REGISTRO = re.compile(
    r"(?:M\.?\s?S\.?|Reg(?:istro)?\.?)[^\d\n]{0,25}(\d\.\s?\d{4}\.\s?\d{4})(?:\.\d{3}-\d)?", re.IGNORECASE
)
# Lista de referências bibliográficas (no fim de "Resultados de eficácia" ou depois dos dizeres
# legais): não ajuda no balcão e polui a busca. Pulada até o próximo título de seção.
_REFERENCIAS = re.compile(r"referencias(?: ?bibliograficas)?")
_HIFEN_NO_FIM = re.compile(r"[^\W\d_]-$")
# Bula sem o título "DIZERES LEGAIS" (ambroxol): depois da superdose, estas linhas abrem a seção.
_DIZERES_SEM_TITULO = re.compile(r"venda (?:sem|sob) prescricao.*|registro m ?s .*")
_ORDEM_SUPERDOSE, _ORDEM_DIZERES = 10, 99
_FORMA = re.compile(r"forma farmaceutica|^apresentac(?:ao|oes)$")

ESPERADAS = {
    "indicacoes", "contraindicacoes", "advertencias", "interacoes", "armazenamento", "posologia",
    "reacoes_adversas", "superdose",
}


@dataclass(frozen=True)
class ParteBula:
    """Uma bula de dentro do PDF, com o que dá para conferir contra o manifesto."""

    secoes: list[SecaoBula]
    versao: str | None  # código impresso pelo fabricante ("V.01_12/2025"), se houver
    registros: list[str]  # números de registro ANVISA achados no texto ("1.5423.0342")
    apresentacao: str | None  # 1ª linha depois de "FORMA FARMACÊUTICA..." / "APRESENTAÇÕES"


def _titulo(linha: str) -> tuple[int, str] | None:
    """(ordem, seção) se a linha é um título de seção."""
    if not linha or len(linha) > _MAX_TITULO or "(" in linha or _FECHA_PARENTESE.search(linha):
        # Remissão quebrada em duas linhas: "...(veja o item 8." + "POSOLOGIA E MODO DE USAR)."
        # O ")" de numeração ("6) Interação medicamentosa") continua valendo.
        return None
    norm = normalizar(linha)
    for ordem, secao, padrao in _TITULOS_RE:
        m = padrao.fullmatch(norm)
        if m and (m.group(1) or linha == linha.upper()):
            return ordem, secao
    return None


def _molde(linha: str) -> str:
    return re.sub(r"\d+", "#", normalizar(linha))


def _sem_rodape_colado(linha: str) -> tuple[str, str | None]:
    """(linha sem o rodapé colado, código de versão Geolab que estava nela)."""
    m = _VERSAO_GEOLAB.match(linha)
    # Forma única para "V. 03 12/2025", "V.0409/2025", "V03_05/2025": "V.03_12/2025".
    versao = f"V.{int(m[1]):02d}_{m[2]}/{m[3]}" if m else None
    for padrao in _RODAPES_COLADOS:
        linha = padrao.sub("", linha, count=1).strip()
    return linha, versao


def _limpar(paginas: list[str]) -> list[list[tuple[str, str | None]]]:
    """Como limpar_paginas, guardando o código de versão que cada linha trazia."""
    marcadas = [[_sem_rodape_colado(l.strip()) for l in p.splitlines()] for p in paginas]
    linhas = [[l for l, _ in pag] for pag in marcadas]

    def bordas(pagina: list[str]) -> set[int]:
        cheias = [i for i, l in enumerate(pagina) if l]
        return set(cheias[:_BORDA] + cheias[-_BORDA:])

    vezes = Counter(m for pag in linhas for m in {_molde(pag[i]) for i in bordas(pag)})
    minimo = max(3, math.ceil(len(paginas) * 0.6))
    repetidas = {m for m, n in vezes.items() if n >= minimo}

    limpas = []
    for pag, marcas in zip(linhas, marcadas):
        borda = bordas(pag)
        limpas.append([
            marcas[i] for i, l in enumerate(pag)
            if not (i in borda and not _titulo(l) and (_molde(l) in repetidas or _SO_PAGINA.fullmatch(normalizar(l))))
        ])
    return limpas


def limpar_paginas(paginas: list[str]) -> list[list[str]]:
    """Linhas de cada página, sem cabeçalho/rodapé repetido nem número de página.

    Só mexe nas bordas da página: no meio, uma linha repetida pode ser conteúdo (tabela).
    Linha vazia fica, porque separa parágrafos. O rodapé colado no começo de uma linha de
    conteúdo sai em qualquer posição (ver _RODAPES_COLADOS).
    """
    return [[l for l, _ in pag] for pag in _limpar(paginas)]


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
            elif not _HIFEN_NO_FIM.search(texto):
                texto += " "
            # Hífen no fim da linha fica e cola na próxima: nas bulas medidas é sempre palavra
            # composta ("anti-" + "histamínico", "observou-" + "se"), não separação de sílaba.
        texto += linha
        pagina_anterior, paragrafo_novo = pagina, False
    return texto


def _secoes(linhas: list[tuple[int, str]]) -> list[SecaoBula]:
    secoes: list[SecaoBula] = []
    ordem_atual, secao_atual, titulo_atual = -100, "outras", "(antes da primeira seção)"
    corpo: list[tuple[int, str]] = []
    pulando = False  # dentro de uma lista de referências

    def fechar() -> None:
        com_texto = [p for p, l in corpo if l]
        if com_texto:
            secoes.append(SecaoBula(
                secao=secao_atual, titulo_original=titulo_atual,
                pagina_inicio=com_texto[0], pagina_fim=com_texto[-1], texto=_juntar(corpo),
            ))

    for numero, linha in linhas:
        achado = _titulo(linha)
        if achado and achado[0] > ordem_atual:
            fechar()
            (ordem_atual, secao_atual), titulo_atual, corpo, pulando = achado, linha, [], False
        elif _ORDEM_SUPERDOSE <= ordem_atual < _ORDEM_DIZERES and _DIZERES_SEM_TITULO.fullmatch(normalizar(linha)):
            fechar()
            ordem_atual, secao_atual, titulo_atual = _ORDEM_DIZERES, "outras", "(dizeres legais sem título)"
            corpo, pulando = [(numero, linha)], False
        elif _REFERENCIAS.fullmatch(normalizar(linha)):
            pulando = True
        elif not pulando:
            corpo.append((numero, linha))
    fechar()
    return secoes


def _partes(paginas: list[str]) -> list[tuple[list[tuple[int, str]], str | None]]:
    """Linhas (página, texto) de cada bula do PDF, com o código de versão Geolab dela.

    Do "Histórico de Alteração" até o próximo "MODELO DE BULA..." tudo é descartado. Um novo
    "MODELO DE BULA..." no meio de uma bula também abre outra (PDF sem a tabela entre elas).
    """
    partes: list[tuple[list[tuple[int, str]], str | None]] = []
    atual: list[tuple[int, str]] = []
    versao: str | None = None
    dentro = True
    for numero, pagina in enumerate(_limpar(paginas), start=1):
        for linha, codigo in pagina:
            norm = normalizar(linha)
            if _INICIO_DE_BULA.search(norm):
                if dentro and any(_titulo(l) for _, l in atual):
                    partes.append((atual, versao))
                    atual, versao = [], None
                dentro = True
            if _FIM_DE_BULA.search(norm):
                if atual:
                    partes.append((atual, versao))
                atual, versao, dentro = [], None, False
                continue
            if dentro:
                atual.append((numero, linha))
                # A Geolab imprime a versão colada em "MODELO DE BULA..." e no rodapé de cada página.
                versao = versao or codigo
    if atual and any(l for _, l in atual):
        partes.append((atual, versao))
    return partes


def _apresentacao(linhas: list[tuple[int, str]]) -> str | None:
    cheias = [l for _, l in linhas if l]
    for i, linha in enumerate(cheias[:-1]):
        if _FORMA.search(normalizar(linha)):
            return cheias[i + 1]
    return None


def separar_bulas(paginas: list[str]) -> list[ParteBula]:
    """Cada bula do PDF, na ordem em que aparece, já separada em seções."""
    resultado = []
    for linhas, versao in _partes(paginas):
        texto = "\n".join(l for _, l in linhas)
        registros = sorted({re.sub(r"\s", "", r) for r in _REGISTRO.findall(texto)})
        secoes = _secoes(linhas)
        if secoes:
            resultado.append(ParteBula(secoes, versao, registros, _apresentacao(linhas)))
    return resultado


def separar_secoes(paginas: list[str]) -> list[SecaoBula]:
    """Seções de um PDF com uma bula só. Com mais de uma, use separar_bulas."""
    partes = separar_bulas(paginas)
    if len(partes) > 1:
        raise ValueError(f"o PDF tem {len(partes)} bulas: use separar_bulas")
    return partes[0].secoes if partes else []
