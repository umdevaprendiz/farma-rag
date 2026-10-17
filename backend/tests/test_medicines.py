import json
from pathlib import Path
from types import SimpleNamespace

from app.services.medicines import (
    Catalog, carregar_catalogo_anvisa, carregar_conhecidos, chave_principio, detectar, normalizar,
)

RAIZ = Path(__file__).resolve().parents[2]

BULAS = [
    SimpleNamespace(principio_ativo="losartana potássica", nomes_comerciais=["Cozaar"]),
    SimpleNamespace(principio_ativo="ibuprofeno", nomes_comerciais=["Advil", "Alivium"]),
    SimpleNamespace(principio_ativo="ácido acetilsalicílico", nomes_comerciais=["Aspirina"]),
]
CONHECIDOS = ["paracetamol", "losartana", "ibuprofeno", "ácido ascórbico", "dipirona"]
CAT = Catalog.montar(BULAS, CONHECIDOS)


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
    cat = Catalog.montar(bulas)
    assert detectar("loperamida dá sono?", cat).indexados == ["cloridrato de loperamida"]
    assert detectar("cetirizina na gravidez", cat).indexados == ["dicloridrato de cetirizina"]
    assert detectar("escopolamina e Buscopan são iguais?", cat).indexados == ["butilbrometo de escopolamina"]
    assert detectar("dose de dipirona", cat).indexados == ["dipirona monoidratada"]
    assert detectar("vitamina ascórbica em excesso", cat).indexados == ["acido ascorbico"]  # flexão (fuzzy)
    assert detectar("ascorbico demais faz mal?", cat).indexados == ["acido ascorbico"]


def test_marca_nao_e_quebrada_em_palavras():
    cat = Catalog.montar([SimpleNamespace(principio_ativo="ibuprofeno", nomes_comerciais=["Advil Forte"])])
    assert detectar("Advil Forte tem sal?", cat).indexados == ["ibuprofeno"]
    assert detectar("um remédio forte para dor", cat).indexados == []  # "forte" sozinho não é alias


def test_sulfato_nao_e_confundido_com_outro_sulfato():
    cat = Catalog.montar([SimpleNamespace(principio_ativo="sulfato ferroso heptaidratado", nomes_comerciais=[])])
    assert detectar("sulfato de magnésio", cat).indexados == []
    assert detectar("ferroso em jejum", cat).indexados == ["sulfato ferroso heptaidratado"]


# --- Catálogo da ANVISA (docs/DECISOES.md, seção 13) ---

def _bula(principio, *marcas):
    return SimpleNamespace(principio_ativo=principio, nomes_comerciais=list(marcas))


def _item(nome, *principios):
    return {"nome_produto": nome, "principios_ativos": list(principios), "associacao": len(principios) > 1}


BULAS_ANVISA = [
    _bula("dipirona monoidratada"), _bula("cloridrato de loperamida"), _bula("butilbrometo de escopolamina"),
    _bula("diclofenaco sódico"), _bula("prednisolona"), _bula("ibuprofeno"),
]
ANVISA = [
    _item("NOVALGINA", "dipirona monoidratada"),
    _item("DIPIRONA", "dipirona"),  # grafia antiga do mesmo princípio ativo
    _item("DIP", "dipirona monoidratada"),
    _item("INTESTIN", "cloridrato de loperamida"),
    _item("BUSCOPAN", "butilbrometo de escopolamina"),
    _item("BUSCOPAN COMPOSTO", "butilbrometo de escopolamina", "dipirona monoidratada"),
    _item("CATAFLAM", "diclofenaco potássico"),
    _item("METICORTEN", "prednisona"),
    _item("GLICOSE", "glicose"),
    _item("ALIVIUM", "ibuprofeno"),
    _item("ALIVIUM", "dipirona monoidratada"),
    _item("ADVIL", "ibuprofeno"),
    _item("ADVIL", "ibuprofeno", "cafeína"),
]
CAT_ANVISA = Catalog.montar(BULAS_ANVISA, anvisa=ANVISA)
NADA = detectar("", CAT_ANVISA)


def test_chave_principio_tira_hidratacao_mas_nao_o_sal():
    assert chave_principio("Amoxicilina tri-hidratada") == "amoxicilina"
    assert chave_principio("dipirona monoidratada") == chave_principio("dipirona")
    assert chave_principio("omeprazol microgranulados") == "omeprazol"
    assert chave_principio("diclofenaco sódico") != chave_principio("diclofenaco potássico")
    assert chave_principio("cloridrato de loperamida") == "cloridrato de loperamida"


def test_marca_do_catalogo_leva_ao_principio_ativo_indexado():
    assert detectar("Novalgina dá sono?", CAT_ANVISA).indexados == ["dipirona monoidratada"]
    assert detectar("Buscopan serve para cólica?", CAT_ANVISA).indexados == ["butilbrometo de escopolamina"]


def test_marca_curta_do_catalogo_fica_fora_mas_a_escrita_na_bula_vale():
    assert detectar("tomei um dip", CAT_ANVISA) == NADA
    cat = Catalog.montar([_bula("dipirona monoidratada", "Dip")], anvisa=ANVISA)
    assert detectar("tomei um dip", cat).indexados == ["dipirona monoidratada"]


def test_marca_do_catalogo_nao_tolera_erro_de_digitacao():
    assert detectar("Intestin é forte?", CAT_ANVISA).indexados == ["cloridrato de loperamida"]
    assert detectar("isso prende o intestino?", CAT_ANVISA) == NADA


def test_marca_ambigua_fica_fora():
    # ALIVIUM em dois princípios ativos; ADVIL também numa associação
    assert detectar("Alivium e Advil", CAT_ANVISA) == NADA


def test_associacao_com_nome_de_indexado_vira_nao_indexada():
    d = detectar("Buscopan Composto pode para criança?", CAT_ANVISA)
    assert d.indexados == [] and d.nao_indexados == ["buscopan composto"]


def test_nao_indexado_entra_pelo_principio_ativo_e_nao_pela_marca():
    assert detectar("Cataflam e Meticorten", CAT_ANVISA) == NADA
    d = detectar("prednisona com dipirona pode?", CAT_ANVISA)
    assert d.indexados == ["dipirona monoidratada"]  # "prednisona" não vira "prednisolona"
    assert d.nao_indexados == ["prednisona"]


def test_outro_sal_do_mesmo_farmaco_e_nao_indexado():
    assert detectar("diclofenaco dá azia?", CAT_ANVISA).indexados == ["diclofenaco sodico"]
    d = detectar("diclofenaco potássico dá azia?", CAT_ANVISA)
    assert d.indexados == [] and d.nao_indexados == ["diclofenaco potassico"]


def test_aliases_ignorados():
    assert detectar("dipirona altera a glicose?", CAT_ANVISA).nao_indexados == ["glicose"]
    cat = Catalog.montar(BULAS_ANVISA, anvisa=ANVISA, ignorados=["Glicose"])
    assert detectar("dipirona altera a glicose?", cat).nao_indexados == []


def test_catalogo_real_com_os_medicamentos_do_mvp():
    mvp = json.loads((RAIZ / "data/medicamentos_mvp.json").read_text(encoding="utf-8"))["medicamentos"]
    bulas = [_bula(m["principio_ativo_catalogo"].split(" | ")[0].split(" (")[0]) for m in mvp]
    cat = Catalog.montar(
        bulas,
        carregar_conhecidos(RAIZ / "data/medicamentos_conhecidos.txt"),
        carregar_catalogo_anvisa(RAIZ / "data/processed/catalogo_anvisa.jsonl"),
        carregar_conhecidos(RAIZ / "data/aliases_ignorados.txt"),
    )
    casos = {
        "Tylenol bebê qual a dose?": (["paracetamol"], []),
        "Tylenol Sinus pode com pressão alta?": ([], ["tylenol sinus"]),
        "Captopril e enalapril juntos?": (["maleato de enalapril"], ["captopril"]),
        "Insulina pode junto com metformina?": (["cloridrato de metformina"], ["insulina"]),
        "Metformina baixa a glicose?": (["cloridrato de metformina"], []),
        "Losartana aumenta o potássio?": (["losartana potassica"], []),
        "Sulfato ferroso prende o intestino?": (["sulfato ferroso heptaidratado"], []),
        "Pode misturar com cafeína ou café?": ([], []),
        "Tomar em jejum ou depois da refeição com proteína e gordura?": ([], []),
        "Qual é o mecanismo de ação e a contraindicação?": ([], []),
        "Idoso com diabetes e hipertensão pode usar?": ([], []),
    }
    for pergunta, (indexados, nao_indexados) in casos.items():
        d = detectar(pergunta, cat)
        assert (d.indexados, d.nao_indexados) == (indexados, nao_indexados), pergunta


def test_palavra_das_bulas_nao_vira_remedio_por_erro_de_digitacao():
    # "cefaleia" fica a 0,88 de "cefalexina": sem o vocabulário das bulas, virava o antibiótico.
    assert detectar("remédio para cefaleia", Catalog.montar([], ["cefalexina"])).nao_indexados == ["cefalexina"]
    cat = Catalog.montar([], ["cefalexina"], vocabulario=["cefaleia"])
    assert detectar("remédio para cefaleia", cat).nao_indexados == []
    assert detectar("cefalexna serve para que?", cat).nao_indexados == ["cefalexina"]  # erro de verdade


def test_sem_tolerancia_so_o_nome_exato():
    cat = Catalog.montar([], ["cefalexina"])
    assert detectar("cefalexna", cat, tolerar_erros=False).nao_indexados == []
    assert detectar("cefalexina", cat, tolerar_erros=False).nao_indexados == ["cefalexina"]


def test_classe_terapeutica_do_registro_anvisa():
    anvisa = [
        {"associacao": False, "classe_terapeutica": "ANALGESICOS NAO NARCOTICOS", "n_registros": 1018,
         "nome_produto": "NOVALGINA", "principios_ativos": ["dipirona"]},
        {"associacao": False, "classe_terapeutica": "ANALGESICOS", "n_registros": 4,
         "nome_produto": "DIPIRONA X", "principios_ativos": ["dipirona monoidratada"]},
        {"associacao": False, "classe_terapeutica": "ANTIDIARREICOS", "n_registros": 9,
         "nome_produto": "IMOSEC", "principios_ativos": ["cloridrato de loperamida"]},
        {"associacao": True, "classe_terapeutica": "ANTIGRIPAIS", "n_registros": 50,
         "nome_produto": "COMBINADO", "principios_ativos": ["dipirona", "cafeína"]},  # associação não conta
    ]
    cat = Catalog.montar([], [], anvisa)
    assert cat.classe_de("dipirona") == "ANALGESICOS NAO NARCOTICOS"  # a mais registrada
    assert cat.classe_de("Loperamida") == "ANTIDIARREICOS"  # pelo núcleo, sem o sal
    assert cat.classe_de("captopril") is None


def test_nome_com_sal_do_catalogo_e_reconhecido_sem_o_sal():
    # No registro da ANVISA é "bissulfato de clopidogrel"; na pergunta, "clopidogrel".
    anvisa = [{"associacao": False, "classe_terapeutica": "ANTIAGREGANTE PLAQUETARIO", "n_registros": 3,
               "nome_produto": "PLAVIX", "principios_ativos": ["bissulfato de clopidogrel"]}]
    cat = Catalog.montar([], [], anvisa)
    assert detectar("Omeprazol pode com clopidogrel?", cat).nao_indexados == ["bissulfato de clopidogrel"]
