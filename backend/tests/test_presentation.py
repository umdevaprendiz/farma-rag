"""A resposta menos robótica, montada no código (services/presentation.py; docs/DECISOES.md, seção 37)."""

import pytest

from app.services import presentation
from app.services.indication import grupos_do_uso
from app.services.presentation import ABERTURA_OPINIAO, abertura, foco, marcas, sugestoes

PARACETAMOL = (
    "Este medicamento é indicado, em adultos, para a redução da febre e o alívio temporário de dores leves a "
    "moderadas, tais como: dores associadas a resfriados comuns, dor de cabeça, dor no corpo, dor de dente [1]."
)
IBUPROFENO = (
    "O ibuprofeno 400 mg cápsulas moles está indicado no alívio temporário da febre e de dores de leve a "
    "moderada intensidade como cefaleia tensional, lombalgia, dor muscular, enxaqueca, dismenorreia [1]."
)


@pytest.mark.parametrize("caminho, pergunta, medicamentos, nomes, extra, esperado", [
    ("remedio", "Ibuprofeno pode na gravidez?", ["Ibuprofeno 400 mg"], ["ibuprofeno"], {},
     "Pela bula de Ibuprofeno 400 mg:"),
    ("remedio", "paracetamol pode na gravidez?", ["Tylalgin 750 mg", "Tylalgin gotas"], ["paracetamol"], {},
     "Pelas bulas de paracetamol:"),
    ("dose", "dose máxima de paracetamol?", ["Tylalgin 750 mg", "Tylalgin gotas"], ["paracetamol"], {},
     "Pela posologia de cada bula:"),
    ("dose", "dose máxima de dipirona?", ["Dorageo 1 g"], ["dipirona"], {}, "Pela posologia da bula de Dorageo 1 g:"),
    ("interacao", "Omeprazol pode com clopidogrel?", ["Omenax 20 mg"], ["omeprazol"],
     {"nomes_da_pergunta": ["omeprazol", "clopidogrel"]}, "Sobre usar omeprazol com clopidogrel, pela bula de Omenax 20 mg:"),
    ("alternativas", "sou alérgico a dipirona, tem outro?", ["Tylalgin 750 mg", "Ibuprofeno 400 mg"],
     ["paracetamol", "ibuprofeno"], {"alergia_a": "dipirona"}, "Para quem tem alergia a dipirona, pelas bulas do acervo:"),
    ("geral", "qual o melhor remédio pra dor de cabeça?", ["Tylalgin 750 mg", "Ibuprofeno 400 mg"],
     ["paracetamol", "ibuprofeno"], {}, "As bulas não dizem qual é o melhor, mas dizem isto:"),
    ("geral", "zolpidem pode com álcool?", ["Omenax 20 mg"], ["omeprazol"], {"sem_bula": True},
     "O que as outras bulas do acervo dizem:"),
    ("indicacao", "paracetamol serve para febre?", ["Tylalgin 750 mg"], ["paracetamol"], {}, None),
    # Opinião: diz que não opina e segue; no para que serve, a conclusão vem logo depois.
    ("indicacao", "oque acha do ibuprofeno pra dor de cabeça?", ["Ibuprofeno 400 mg"], ["ibuprofeno"], {},
     ABERTURA_OPINIAO),
    ("remedio", "o que você acha de ibuprofeno na gravidez?", ["Ibuprofeno 400 mg"], ["ibuprofeno"], {},
     f"{ABERTURA_OPINIAO} Pela bula de Ibuprofeno 400 mg:"),
])
def test_abertura_pelo_caminho_da_pergunta(caminho, pergunta, medicamentos, nomes, extra, esperado):
    assert abertura(caminho, pergunta, medicamentos, nomes, **extra) == esperado


def test_foco_marca_o_que_foi_perguntado_com_a_dor_junto():
    grupos = grupos_do_uso("dor de cabeça")
    (inicio, fim), = foco(PARACETAMOL, grupos)
    assert PARACETAMOL[inicio:fim] == "dor de cabeça"


def test_foco_pelos_sinonimos_da_bula_e_palavras_vizinhas_juntas():
    texto = IBUPROFENO
    achadas = [texto[i:f] for i, f in foco(texto, grupos_do_uso("dor de cabeça"))]
    assert achadas == ["cefaleia", "enxaqueca"]
    texto = "Indicado para cólica menstrual e dor de dente [1]."
    assert [texto[i:f] for i, f in foco(texto, grupos_do_uso("cólica menstrual"))] == ["cólica menstrual"]


def test_foco_nao_marca_quando_cobriria_quase_tudo():
    assert foco("Dor e dores [1].", [("dor", "dores")]) == []
    assert foco(PARACETAMOL, []) == []


def test_marcas_juntam_foco_e_glossario_em_ordem():
    achadas = marcas(IBUPROFENO, grupos_do_uso("dor nas costas"))
    assert [(m.tipo, IBUPROFENO[m.inicio:m.fim], m.explicacao) for m in achadas] == [
        ("termo", "cefaleia tensional", "dor de cabeça"),
        ("foco", "lombalgia", None),
        ("termo", "lombalgia", "dor na parte baixa das costas"),
        ("termo", "dismenorreia", "cólica menstrual"),
    ]


def test_sugestoes_sem_a_secao_que_a_pergunta_ja_pediu():
    assert sugestoes("dipirona", {"indicacoes"}) == [
        "Qual a dose de dipirona?", "Quem não pode tomar dipirona?", "Quais os efeitos colaterais de dipirona?",
    ]
    assert sugestoes("dipirona", {"posologia", "contraindicacoes"}) == [
        "Para que serve dipirona?", "Quais os efeitos colaterais de dipirona?",
    ]


def test_pede_opiniao():
    assert presentation.pede_opiniao("O que você acha do ibuprofeno?")
    assert presentation.pede_opiniao("vale a pena tomar dipirona?")
    assert not presentation.pede_opiniao("paracetamol é bom pra dor de cabeça?")
