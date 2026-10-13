from pathlib import Path

import pytest

from ingestao.extrair_pdf import extrair_paginas
from ingestao.separar_secoes import ESPERADAS, _titulo, separar_secoes

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("linha, secao", [
    ("1. INDICAÇÕES", "indicacoes"),
    ("1 - Indicações", "indicacoes"),
    ("1.INDICACOES", "indicacoes"),
    ("INDICAÇÕES", "indicacoes"),
    ("3. CARACTERÍSTICAS FARMACOLÓGICAS", "caracteristicas_farmacologicas"),
    ("4. CONTRA-INDICAÇÕES", "contraindicacoes"),
    ("5. ADVERTÊNCIAS", "advertencias"),
    ("6) Interação medicamentosa", "interacoes"),
    ("7. CUIDADOS DE CONSERVAÇÃO", "armazenamento"),
    ("8. Posologia e modo de usar", "posologia"),
    ("9. REAÇÕES ADVERSAS A MEDICAMENTOS", "reacoes_adversas"),
    ("10. SUPERDOSAGEM", "superdose"),
    ("COMPOSIÇÃO", "composicao"),
    ("1. PARA QUE ESTE MEDICAMENTO É INDICADO?", "indicacoes"),
    ("9. O QUE FAZER SE ALGUÉM USAR UMA QUANTIDADE MAIOR DO QUE A", "superdose"),
])
def test_variacoes_de_titulo(linha, secao):
    assert _titulo(linha)[1] == secao


@pytest.mark.parametrize("linha", [
    "Indicações",  # subtítulo: sem número e sem caixa alta
    "Posologia em idosos: não é necessário ajuste (ver item 8. Posologia e modo de usar).",
    "As reações adversas mais comuns são tontura e tosse.",
    "",
])
def test_linha_que_nao_e_titulo(linha):
    assert _titulo(linha) is None


def test_bula_de_exemplo_inteira():
    secoes = separar_secoes(extrair_paginas(FIXTURES / "bula_exemplo.txt"))
    chaves = [s.secao for s in secoes]
    assert chaves == [
        "outras", "outras", "composicao", "indicacoes", "resultados_eficacia",
        "caracteristicas_farmacologicas", "contraindicacoes", "advertencias", "interacoes",
        "armazenamento", "posologia", "reacoes_adversas", "superdose", "outras",
    ]
    assert ESPERADAS <= set(chaves)
    por_chave = {s.secao: s for s in secoes}
    assert (por_chave["indicacoes"].pagina_inicio, por_chave["interacoes"].pagina_inicio) == (1, 2)
    assert por_chave["armazenamento"].pagina_inicio == 3
    assert por_chave["interacoes"].titulo_original == "6. INTERAÇÕES MEDICAMENTOSAS"
    # a linha quebrada no PDF volta a ser uma frase só
    assert "pode reduzir o efeito anti-hipertensivo da exemplina" in por_chave["interacoes"].texto
    # o subtítulo "Posologia" e a remissão "ver item 8" não abriram seção nova
    assert por_chave["posologia"].texto.startswith("Posologia")
    assert "ver item 8" in por_chave["advertencias"].texto
    # cabeçalho repetido e número de página saem
    texto = " ".join(s.texto for s in secoes)
    assert "Bula do profissional" not in texto and "Página" not in texto


def test_remissao_a_secao_anterior_nao_abre_secao():
    secoes = separar_secoes(["5. ADVERTÊNCIAS E PRECAUÇÕES\nVer também:\n1. INDICAÇÕES\ne o item 4."])
    assert [s.secao for s in secoes] == ["advertencias"]
    assert "1. INDICAÇÕES" in secoes[0].texto


def test_secao_que_atravessa_paginas_marca_a_virada():
    secoes = separar_secoes(["9. REAÇÕES ADVERSAS\nTontura e", "tosse seca.\n10. SUPERDOSE\nHipotensão."])
    reacoes, superdose = secoes
    assert reacoes.texto == "Tontura e\ftosse seca."
    assert (reacoes.pagina_inicio, reacoes.pagina_fim) == (1, 2)
    assert superdose.pagina_inicio == 2


def test_texto_sem_titulo_vira_uma_secao_so():
    secoes = separar_secoes(["Texto qualquer sem título reconhecido."])
    assert [(s.secao, s.pagina_inicio) for s in secoes] == [("outras", 1)]
    assert separar_secoes(["", "   "]) == []
