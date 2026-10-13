from types import SimpleNamespace

from app.services.medicamentos import Catalogo, detectar, normalizar

BULAS = [
    SimpleNamespace(principio_ativo="losartana potássica", nomes_comerciais=["Cozaar"]),
    SimpleNamespace(principio_ativo="ibuprofeno", nomes_comerciais=["Advil", "Alivium"]),
    SimpleNamespace(principio_ativo="ácido acetilsalicílico", nomes_comerciais=["Aspirina"]),
]
CONHECIDOS = ["paracetamol", "losartana", "ibuprofeno", "ácido ascórbico", "dipirona"]
CAT = Catalogo.montar(BULAS, CONHECIDOS)


def test_normalizar():
    assert normalizar("Losartana  Potássica, 50 mg!") == "losartana potassica 50 mg"


def test_detecta_dois_medicamentos_na_pergunta_de_interacao():
    d = detectar("Pode tomar ibuprofeno junto com losartana?", CAT)
    assert d.indexados == ["ibuprofeno", "losartana potassica"]
    assert d.nao_indexados == []


def test_detecta_por_nome_comercial_e_nome_parcial():
    assert detectar("Advil dá sono?", CAT).indexados == ["ibuprofeno"]
    assert detectar("Dose de Cozaar para adulto", CAT).indexados == ["losartana potassica"]


def test_medicamento_conhecido_sem_bula_e_nao_indexado():
    d = detectar("Qual a dose de paracetamol para adulto?", CAT)
    assert d.indexados == []
    assert d.nao_indexados == ["paracetamol"]


def test_conhecido_que_esta_indexado_nao_vira_nao_indexado():
    d = detectar("losartana na gravidez?", CAT)
    assert d.nao_indexados == []


def test_mistura_indexado_e_nao_indexado():
    d = detectar("ibuprofeno com dipirona pode?", CAT)
    assert d.indexados == ["ibuprofeno"]
    assert d.nao_indexados == ["dipirona"]


def test_erro_de_digitacao_em_nome_longo():
    assert detectar("ibuprofemo causa sono?", CAT).indexados == ["ibuprofeno"]


def test_pergunta_sem_medicamento():
    d = detectar("Qual o preço médio deste remédio?", CAT)
    assert d.indexados == [] and d.nao_indexados == []


def test_palavra_comum_nao_vira_falso_positivo():
    # "acido" sozinho é genérico demais para identificar um princípio ativo
    assert detectar("o suco ácido faz mal?", CAT).indexados == []


def test_nome_de_sal_do_catalogo_da_anvisa_vira_o_nome_que_o_usuario_diz():
    bulas = [
        SimpleNamespace(principio_ativo="cloridrato de loperamida", nomes_comerciais=["Imosec"]),
        SimpleNamespace(principio_ativo="dicloridrato de cetirizina", nomes_comerciais=[]),
        SimpleNamespace(principio_ativo="butilbrometo de escopolamina", nomes_comerciais=["Buscopan"]),
        SimpleNamespace(principio_ativo="dipirona monoidratada", nomes_comerciais=[]),
        SimpleNamespace(principio_ativo="ácido ascórbico", nomes_comerciais=[]),
    ]
    cat = Catalogo.montar(bulas)
    assert detectar("loperamida dá sono?", cat).indexados == ["cloridrato de loperamida"]
    assert detectar("cetirizina na gravidez", cat).indexados == ["dicloridrato de cetirizina"]
    assert detectar("escopolamina e Buscopan são iguais?", cat).indexados == ["butilbrometo de escopolamina"]
    assert detectar("dose de dipirona", cat).indexados == ["dipirona monoidratada"]
    assert detectar("vitamina ascórbica em excesso", cat).indexados == ["acido ascorbico"]  # flexão (fuzzy)
    assert detectar("ascorbico demais faz mal?", cat).indexados == ["acido ascorbico"]


def test_marca_nao_e_quebrada_em_palavras():
    cat = Catalogo.montar([SimpleNamespace(principio_ativo="ibuprofeno", nomes_comerciais=["Advil Forte"])])
    assert detectar("Advil Forte tem sal?", cat).indexados == ["ibuprofeno"]
    assert detectar("um remédio forte para dor", cat).indexados == []  # "forte" sozinho não é alias


def test_sulfato_nao_e_confundido_com_outro_sulfato():
    cat = Catalogo.montar([SimpleNamespace(principio_ativo="sulfato ferroso heptaidratado", nomes_comerciais=[])])
    assert detectar("sulfato de magnésio", cat).indexados == []
    assert detectar("ferroso em jejum", cat).indexados == ["sulfato ferroso heptaidratado"]
