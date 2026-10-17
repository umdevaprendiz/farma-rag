"""Extrai o texto de uma bula, uma string por página.

Aceita .pdf (pdfplumber, só no host: ver backend/requirements-dev.txt) e .txt, em que as páginas
são separadas por form feed ("\\f"); o .txt serve para teste e para bula colada à mão.

Tabela de interações em duas colunas (remédio | "Impacto clínico:" / "Intervenção:" + texto, como
a do Omenax): lida como tabela. Pela extração de texto, os rótulos da coluna da esquerda caíam no
meio das frases da direita ("pode reduzir o efeito antiviral e promover o Impacto clínico:
desenvolvimento de resistência"), e um trecho juntava saquinavir com clopidogrel. Cada célula vira
uma linha que leva o nome do remédio ("Clopidogrel, intervenção: Evitar o uso..."): o trecho que
pegar só aquela linha ainda diz de quem ela fala. As outras tabelas continuam pela extração de
texto (docs/DECISOES.md, seção 22).
"""

import re
from pathlib import Path

_ROTULO = re.compile(r"^(impacto cl[ií]nico|interven[cç][aã]o)\s*:?$", re.IGNORECASE)


def _e_tabela_de_interacoes(tabela: list[list[str | None]]) -> bool:
    return bool(tabela) and len(tabela[0]) == 2 and any(_ROTULO.match((linha[0] or "").strip()) for linha in tabela)


def _celula(texto: str | None) -> str:
    return " ".join((texto or "").split())


class _InteractionTableReader:
    """Transforma as linhas da tabela em texto. Guarda o remédio e o rótulo da última linha:
    a tabela continua de uma página para a outra."""

    def __init__(self) -> None:
        self.remedio = ""
        self.rotulo = ""
        self.sem_conteudo = False  # o cabeçalho atual ainda não teve linha de conteúdo

    def linhas(self, tabela: list[list[str | None]]) -> list[str]:
        saida = []
        for esquerda, direita in ((_celula(a), _celula(b)) for a, b in tabela):
            if esquerda and not direita:  # "Varfarina" ou "Substratos de CYP2C19 (...)"
                if self.sem_conteudo:  # cabeçalho de grupo, seguido do remédio: vira texto
                    saida.append(self.remedio)
                self.remedio, self.rotulo, self.sem_conteudo = esquerda, "", True
                continue
            self.sem_conteudo = False
            if _ROTULO.match(esquerda):
                self.rotulo = esquerda.rstrip(":").strip().lower()
                saida.append(f"{self.remedio}, {self.rotulo}: {direita}")
            elif direita:  # célula que continua da página anterior
                rotulo = f"{self.rotulo} (continuação)" if self.rotulo else "continuação"
                saida.append(f"{self.remedio}, {rotulo}: {direita}" if self.remedio else direita)
        return saida


def _texto_da_pagina(pagina, leitor: _InteractionTableReader) -> str:
    tabelas = [t for t in pagina.find_tables() if _e_tabela_de_interacoes(t.extract())]
    if not tabelas:
        return pagina.extract_text() or ""
    caixas = [t.bbox for t in tabelas]

    def fora_das_tabelas(obj) -> bool:
        meio_x, meio_y = (obj["x0"] + obj["x1"]) / 2, (obj["top"] + obj["bottom"]) / 2
        return not any(x0 <= meio_x <= x1 and topo <= meio_y <= base for x0, topo, x1, base in caixas)

    # Texto de fora das tabelas e linhas das tabelas, na ordem em que aparecem na página.
    blocos = [(linha["top"], linha["text"]) for linha in pagina.filter(fora_das_tabelas).extract_text_lines()]
    blocos += [(t.bbox[1], "\n".join(leitor.linhas(t.extract()))) for t in tabelas]
    return "\n".join(texto for _, texto in sorted(blocos, key=lambda b: b[0]) if texto)


def extrair_paginas(caminho: str | Path) -> list[str]:
    caminho = Path(caminho)
    if caminho.suffix.lower() == ".txt":
        return caminho.read_text(encoding="utf-8").split("\f")
    try:
        import pdfplumber
    except ImportError as e:
        raise RuntimeError("Falta o pdfplumber: pip install -r backend/requirements-dev.txt") from e
    leitor = _InteractionTableReader()
    with pdfplumber.open(caminho) as pdf:
        return [_texto_da_pagina(pagina, leitor) for pagina in pdf.pages]
