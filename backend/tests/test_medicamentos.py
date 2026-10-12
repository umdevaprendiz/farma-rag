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
