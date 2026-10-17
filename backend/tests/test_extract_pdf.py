"""Leitura da tabela de interações (duas colunas, rótulos "Impacto clínico:"/"Intervenção:"), sem PDF:
as linhas são como as que o pdfplumber devolve da bula do Omenax."""

from ingestion.extract_pdf import _e_tabela_de_interacoes, _InteractionTableReader

PAGINA_1 = [
    ["Antirretrovirais", None],
    ["Impacto clínico:", "O efeito dos IBPs nos medicamentos\nantirretrovirais é variável."],
    ["Intervenção:", "Atazanavir: Evitar o uso concomitante de omeprazol. Ver"],
]
PAGINA_2 = [
    ["", "informação de posologia na bula de atazanavir."],  # a célula continua da página anterior
    ["Substratos de CYP2C19 (por exemplo, clopidogrel, citalopram)", ""],
    ["Clopidogrel", ""],
    ["Impacto clínico:", "Redução do metabólito ativo do clopidogrel."],
    ["Intervenção", "Evitar o uso concomitante de omeprazol."],  # sem dois-pontos, como na bula
]


def test_cada_linha_leva_o_nome_do_remedio_mesmo_entre_paginas():
    leitor = _InteractionTableReader()
    assert leitor.linhas(PAGINA_1) == [
        "Antirretrovirais, impacto clínico: O efeito dos IBPs nos medicamentos antirretrovirais é variável.",
        "Antirretrovirais, intervenção: Atazanavir: Evitar o uso concomitante de omeprazol. Ver",
    ]
    assert leitor.linhas(PAGINA_2) == [
        "Antirretrovirais, intervenção (continuação): informação de posologia na bula de atazanavir.",
        "Substratos de CYP2C19 (por exemplo, clopidogrel, citalopram)",  # cabeçalho de grupo vira texto
        "Clopidogrel, impacto clínico: Redução do metabólito ativo do clopidogrel.",
        "Clopidogrel, intervenção: Evitar o uso concomitante de omeprazol.",
    ]


def test_so_a_tabela_de_interacoes_e_lida_como_tabela():
    assert _e_tabela_de_interacoes(PAGINA_1)
    # Histórico de alteração (10 colunas) e tabela de eficácia: continuam pela extração de texto.
    assert not _e_tabela_de_interacoes([["Data", "Nº", "Assunto"] + [""] * 7])
    assert not _e_tabela_de_interacoes([["Dose", "Resposta"], ["50 mg", "12 mmHg"]])
