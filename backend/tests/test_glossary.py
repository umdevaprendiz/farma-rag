"""Glossário do balcão (services/glossary.py; docs/DECISOES.md, seção 37): a tela e a fundamentação."""

import pytest

from app.schemas.dataset import Chunk
from app.services import glossary
from app.services.grounding import checar, destaques

# Textos reais da seção de indicações (data/processed/bulas), encurtados.
IBUPROFENO = (
    "O ibuprofeno 400 mg cápsulas moles está indicado no alívio temporário da febre e de dores de leve a "
    "moderada intensidade como cefaleia tensional, lombalgia, dor muscular, enxaqueca, dismenorreia, de gripes "
    "e resfriados comuns, de artrite e dor de dente."
)
LORATADINA = (
    "Alívio temporário dos sintomas associados com rinite alérgica (por exemplo: febre do feno), como: coceira "
    "nasal, nariz escorrendo (coriza), espirros, ardor e coceira nos olhos."
)
PANTODEX = "Pantodex é indicado para tratamento e prevenção de dermatite de fraldas (assaduras)."
COZAAR = "COZAAR® é indicado para o tratamento da hipertensão."


def _chunk(texto: str, medicamento: str = "Ibuprofeno 400 mg", principio: str = "ibuprofeno") -> Chunk:
    return Chunk(
        chunk_id="b__indicacoes__000", bula_id="b", medicamento=medicamento, principio_ativo=principio,
        secao="indicacoes", pagina=1, ordem=0, texto=texto, n_tokens=50,
    )


def test_dobrar_tira_acento_sem_mudar_as_posicoes():
    texto = "Cefaleia, dismenorréia e lombalgia — ação"
    assert len(glossary.dobrar(texto)) == len(texto)
    assert glossary.dobrar(texto) == "cefaleia, dismenorreia e lombalgia — acao"


def test_anota_os_termos_tecnicos_na_primeira_vez():
    anotados = glossary.anotar(IBUPROFENO + " Em caso de cefaleia, suspenda.")
    assert [(a.termo, a.explicacao) for a in anotados] == [
        ("cefaleia tensional", "dor de cabeça"),
        ("lombalgia", "dor na parte baixa das costas"),
        ("dismenorreia", "cólica menstrual"),
    ]
    assert IBUPROFENO[anotados[0].inicio:anotados[0].fim] == "cefaleia tensional"


@pytest.mark.parametrize("texto", [
    LORATADINA,  # "nariz escorrendo (coriza)": a bula já explica, entre parênteses
    PANTODEX,  # "dermatite de fraldas (assaduras)": explicada logo depois
    "Prurido e coceira na pele.",  # a explicação já está escrita no texto
])
def test_termo_que_a_bula_ja_explica_fica_sem_parenteses(texto):
    assert [a.termo for a in glossary.anotar(texto)] == []


def test_termo_mais_especifico_vem_antes_do_geral():
    # O geral explica o primeiro "prurido" que sobrou, não o de dentro de "prurido ocular".
    assert [(a.termo, a.explicacao) for a in glossary.anotar("Alívio do prurido ocular e do prurido.")] == [
        ("prurido ocular", "coceira nos olhos"), ("prurido", "coceira"),
    ]


def test_limite_de_anotacoes():
    texto = "cefaleia, lombalgia, mialgia, artralgia, odontalgia, dismenorreia, pirose, dispepsia"
    assert len(glossary.anotar(texto)) == glossary.MAX_ANOTACOES


def test_cobertos_so_a_expressao_inteira_e_so_com_o_termo_no_trecho():
    assert glossary.cobertos("serve para dores de cabeca", "indicado para cefaleia") == {"dores", "cabeca"}
    assert glossary.cobertos("serve para dor de cabeca", "indicado para febre") == set()
    # "pressão" sozinha não é a expressão: não vira hipertensão.
    assert glossary.cobertos("cuidado com a pressao", "tratamento da hipertensao") == set()


@pytest.mark.parametrize("afirmacao, trecho, passa", [
    ("O ibuprofeno serve para dor de cabeça, dor nas costas e cólica menstrual", IBUPROFENO, True),
    ("Cozaar é indicado para pressão alta", COZAAR, True),
    # Sem o glossário a frase nem teria termo do trecho além do nome (regra 3).
    ("Cozaar serve para pressão alta", COZAAR, True),
    ("Ibuprofeno serve para pressão alta", IBUPROFENO, False),  # o trecho não fala de hipertensão
])
def test_fundamentacao_aceita_a_palavra_do_dia_a_dia(afirmacao, trecho, passa):
    chunk = _chunk(trecho, "Cozaar 50 mg", "losartana potássica") if trecho is COZAAR else _chunk(trecho)
    assert (not checar([(afirmacao, 1)], {1: chunk}, 0.6)) is passa


def test_destaque_acha_a_frase_pelo_termo_tecnico():
    chunk = _chunk(IBUPROFENO + " Não use por mais de 3 dias.")
    assert destaques(["O ibuprofeno serve para dor nas costas"], chunk) == [(0, len(IBUPROFENO))]
