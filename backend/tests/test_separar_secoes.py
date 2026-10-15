from pathlib import Path

import pytest

from ingestao.extrair_pdf import extrair_paginas
from ingestao.separar_secoes import ESPERADAS, _titulo, separar_bulas, separar_secoes

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


@pytest.mark.parametrize("linha, secao", [
    ("5. ADVERTÊNCIA E PRECAUÇÕES", "advertencias"),  # singular (Cisteil)
    ("7. CUIDADOS DE ARMANEZAMENTO DO MEDICAMENTO", "armazenamento"),  # erro de digitação (Kaodine)
    ("10. SUSPERDOSE", "superdose"),  # idem
])
def test_titulos_com_variacao_das_bulas_reais(linha, secao):
    assert _titulo(linha)[1] == secao


def test_remissao_quebrada_em_duas_linhas_nao_e_titulo():
    # "...(veja o item 8." + "POSOLOGIA E MODO DE USAR)." abria a posologia no meio das advertências.
    assert _titulo("POSOLOGIA E MODO DE USAR).") is None
    assert _titulo("CONTRAINDICAÇÕES e 8. POSOLOGIA E MODO DE USAR).") is None


def test_rodape_colado_sai_e_vira_a_versao():
    partes = separar_bulas([
        "V.14_02/2026MODELO DE BULA PARA O PROFISSIONAL DE SAÚDE\n4. CONTRAINDICAÇÕES\nNão use.\n"
        "V.14_02/20265. ADVERTÊNCIA E PRECAUÇÕES\nCuidado.\nV. 03 12/2025Registro 1.5423.0140"
    ])
    assert len(partes) == 1
    p = partes[0]
    assert [s.secao for s in p.secoes] == ["outras", "contraindicacoes", "advertencias"]
    assert p.versao == "V.14_02/2026" and p.registros == ["1.5423.0140"]
    assert "V." not in " ".join(s.texto for s in p.secoes)


def test_pdf_com_duas_bulas_separadas_pelo_historico():
    paginas = [
        "TYLALGIN\nV.05_05/2022MODELO DE BULA PARA O PROFISSIONAL DE SAÚDE\nFORMA FARMACÊUTICA E APRESENTAÇÃO:\n"
        "Solução gotas de 200mg/mL\n1. INDICAÇÕES\nDor e febre.\n10. SUPERDOSE\nSuporte.\nRegistro M.S. nº 1.5423.0008",
        "V.05_05/2022Anexo B\nHistórico de Alteração para a Bula\n15/06/2018 Notificação 9. Reações Adversas\n"
        "V.04_05/2022MODELO DE BULA PARA O PROFISSIONAL DE SAÚDE\nFORMA FARMACÊUTICA E APRESENTAÇÃO:\n"
        "Comprimido de 750mg\n1. INDICAÇÕES\nDor.",
        "8. POSOLOGIA E MODO DE USAR\n1 comprimido.\nRegistro M.S. nº 1.5423.0008\nHistórico de Alteração da Bula\nlixo",
    ]
    gotas, comprimido = separar_bulas(paginas)
    assert (gotas.versao, comprimido.versao) == ("V.05_05/2022", "V.04_05/2022")
    assert (gotas.apresentacao, comprimido.apresentacao) == ("Solução gotas de 200mg/mL", "Comprimido de 750mg")
    assert gotas.registros == comprimido.registros == ["1.5423.0008"]
    assert [s.secao for s in comprimido.secoes] == ["outras", "indicacoes", "posologia"]
    # as páginas continuam as do PDF, e a tabela do histórico não entra em nenhuma bula
    assert (comprimido.secoes[1].pagina_inicio, comprimido.secoes[2].pagina_inicio) == (2, 3)
    texto = " ".join(s.texto for p in (gotas, comprimido) for s in p.secoes)
    assert "Notificação" not in texto and "lixo" not in texto
    with pytest.raises(ValueError, match="2 bulas"):
        separar_secoes(paginas)


def test_referencias_bibliograficas_sao_puladas_ate_o_proximo_titulo():
    secoes = separar_secoes([
        "2. RESULTADOS DE EFICÁCIA\nReduziu a dor.\nReferências bibliográficas:\n1. Silva A. J Med 2001.\n"
        "3. CARACTERÍSTICAS FARMACOLÓGICAS\nAge rápido.\nDIZERES LEGAIS\nSAC.\nREFERÊNCIAS BIBLIOGRÁFICAS\nZambon S.p.A."
    ])
    assert [(s.secao, s.texto) for s in secoes] == [
        ("resultados_eficacia", "Reduziu a dor."), ("caracteristicas_farmacologicas", "Age rápido."), ("outras", "SAC."),
    ]


def test_hifen_de_palavra_composta_no_fim_da_linha_cola_na_proxima():
    secoes = separar_secoes(["4. CONTRAINDICAÇÕES\nAlergia a outros anti-\nhistamínicos e uso do cirurgião-\ndentista."])
    assert secoes[0].texto == "Alergia a outros anti-histamínicos e uso do cirurgião-dentista."


def test_dizeres_legais_sem_titulo_saem_da_superdose():
    secoes = separar_secoes([
        "10. SUPERDOSE\nTratamento de suporte.\nVENDA SEM PRESCRIÇÃO MÉDICA\nRegistro M.S. nº 1.5423.0067\nCNPJ"
    ])
    assert [(s.secao, s.texto) for s in secoes] == [
        ("superdose", "Tratamento de suporte."), ("outras", "VENDA SEM PRESCRIÇÃO MÉDICA Registro M.S. nº 1.5423.0067 CNPJ"),
    ]
