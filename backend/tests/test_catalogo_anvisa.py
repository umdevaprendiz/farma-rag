from ingestao.catalogo_anvisa import chave, construir_catalogo, limpar_nome, separar_principios


def _r(nome, principio, situacao="Ativo", classe="ANTI-HIPERTENSIVOS", categoria="Genérico"):
    return {
        "NOME_PRODUTO": nome, "PRINCIPIO_ATIVO": principio, "SITUACAO_REGISTRO": situacao,
        "CLASSE_TERAPEUTICA": classe, "CATEGORIA_REGULATORIA": categoria,
    }


def test_separar_principios_respeita_numeros_decimais_e_virgula_final():
    assert separar_principios("losartana potássica, hidroclorotiazida") == ["losartana potássica", "hidroclorotiazida"]
    assert separar_principios("cianocobalamina 0,1%") == ["cianocobalamina 0,1%"]
    assert separar_principios("hepar sulfuris,") == ["hepar sulfuris"]
    assert separar_principios("cloridrato  de ondansetrona") == ["cloridrato de ondansetrona"]
    assert separar_principios("   ") == []


def test_limpar_nome_remove_prefixo_de_cadastro_antigo():
    assert limpar_nome("- LIQUIDO DE DAKIN") == "LIQUIDO DE DAKIN"
    assert limpar_nome("  ADVIL   400 MG ") == "ADVIL 400 MG"


def test_so_registros_ativos_e_agrupa_por_nome_e_principio():
    itens, c = construir_catalogo([
        _r("LOSARTANA POTASSICA", "losartana potássica"),
        _r("Losartana Potássica", "losartana potássica"),  # mesma chave, outra caixa/acento
        _r("COZAAR", "losartana potássica", situacao="Inativo"),
        _r("LOSARTANA", "losartana potássica", classe=""),
    ])
    assert c["linhas_lidas"] == 4 and c["linhas_ativas"] == 3
    nomes = {i["nome_produto"]: i for i in itens}
    assert set(nomes) == {"LOSARTANA POTASSICA", "LOSARTANA"}
    assert nomes["LOSARTANA POTASSICA"]["n_registros"] == 2
    assert nomes["LOSARTANA"]["classe_terapeutica"] is None  # sem classe no registro


def test_associacao_nao_recebe_classe_terapeutica():
    itens, _ = construir_catalogo([_r("ADVIL MULTI", "ibuprofeno, paracetamol", classe="ANALGESICOS NARCOTICOS")])
    assert itens[0]["associacao"] is True
    assert itens[0]["classe_terapeutica"] is None
    assert itens[0]["principios_ativos"] == ["ibuprofeno", "paracetamol"]


def test_classe_mais_frequente_vence_e_empate_e_deterministico():
    itens, _ = construir_catalogo([
        _r("X", "ibuprofeno", classe="B"), _r("X", "ibuprofeno", classe="A"),
        _r("X", "ibuprofeno", classe="B"),
    ])
    assert itens[0]["classe_terapeutica"] == "B"
    itens, _ = construir_catalogo([_r("X", "ibuprofeno", classe="B"), _r("X", "ibuprofeno", classe="A")])
    assert itens[0]["classe_terapeutica"] == "A"


def test_ativa_sem_principio_ou_nome_e_descartada_e_contada():
    itens, c = construir_catalogo([_r("ARNICA", ""), _r("-", "x"), _r("OK", "dipirona")])
    assert [i["nome_produto"] for i in itens] == ["OK"]
    assert c["ativas_sem_nome_ou_principio"] == 2


def test_saida_ordenada_e_independente_da_ordem_de_entrada():
    a = [_r("B", "zinco"), _r("A", "ácido fólico"), _r("C", "zinco")]
    assert construir_catalogo(a)[0] == construir_catalogo(list(reversed(a)))[0]
    assert [i["nome_produto"] for i in construir_catalogo(a)[0]] == ["A", "B", "C"]


def test_chave():
    assert chave("  Ácido  ACETILSALICÍLICO ") == "acido acetilsalicilico"
