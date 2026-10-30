import pytest

from app.schemas.dataset import BenchmarkQuestion
from benchmark.quality import avaliar, confere, polaridade


def _pergunta(esperada: str, **extra) -> BenchmarkQuestion:
    return BenchmarkQuestion(
        id="q1", pergunta="Pergunta de teste?", respondivel=True, categoria="posologia", resposta_esperada=esperada,
        ancoras=[{"principio_ativo": "x", "secao": "posologia", "frase": "frase de teste"}], **extra,
    )


@pytest.mark.parametrize("texto, esperado", [
    ("Sim, pode.", "sim"), ("Não, não pode.", "nao"), ("  Não é recomendado", "nao"),
    ("Simplesmente isso", None), ("Dose de 50 mg", None),
])
def test_polaridade_so_olha_a_abertura(texto, esperado):
    assert polaridade(texto) == esperado


def test_sim_onde_o_gabarito_diz_nao_nao_confere():
    nota = confere("Sim, o ácido acetilsalicílico pode ser usado [1].", _pergunta("Não: úlcera ativa é contraindicação."))
    assert not nota["confere"] and "polaridade" in nota["problemas"][0]


def test_mesma_polaridade_confere():
    assert confere("Não é recomendado para menores de 12 anos [2].", _pergunta("Não: contraindicada abaixo de 12 anos."))["confere"]


def test_numeros_que_faltam_deixam_a_resposta_incompleta_mas_nao_errada():
    nota = confere("Não precisa ficar na geladeira [1].", _pergunta("Não: em temperatura ambiente, entre 15 e 30 °C."))
    assert nota["confere"] and not nota["completa"]
    assert nota["problemas"] == ["números que faltam: 15, 30"]


@pytest.mark.parametrize("resposta", ["Até 4.000 mg por dia [1].", "Até 4000mg por dia [1].", "Dose máxima de 4000 mg [1]."])
def test_milhar_com_ou_sem_ponto_e_o_mesmo_numero(resposta):
    assert confere(resposta, _pergunta("Até 4000 mg por dia."))["completa"]


def test_virgula_decimal_conta():
    assert confere("A dose inicial é de 12,5 mg [1].", _pergunta("12,5 mg uma vez ao dia."))["completa"]
    assert not confere("A dose inicial é de 12 mg [1].", _pergunta("12,5 mg uma vez ao dia."))["completa"]


def test_deve_conter_aceita_alternativas_e_ignora_acento():
    q = _pergunta("Tratamento por 3 dias.", deve_conter=["3 dias|três dias"])
    assert confere("Use por três dias [1].", q)["confere"] is True
    assert confere("Use a dose pediátrica de 2 gotas [1].", q)["confere"] is False


def test_nao_deve_conter_reprova_o_que_a_bula_proibe():
    q = _pergunta("Dose única, sem readministrar.", nao_deve_conter=["pode tomar outro"])
    assert not confere("Sim, ele pode tomar outro comprimido [1].", q)["confere"]
    assert confere("Dose única [1].", q)["confere"]


def test_avaliar_so_olha_as_respondidas_com_gabarito():
    perguntas = {"q1": _pergunta("Sim.")}
    linhas = [
        {"id": "q1", "respondivel": True, "status": "respondida", "resposta": "Sim [1]."},
        {"id": "q2", "respondivel": True, "status": "nao_encontrada", "resposta": None},
        {"id": "q3", "respondivel": False, "status": "respondida", "resposta": "algo"},
    ]
    assert list(avaliar(linhas, perguntas)) == ["q1"]


def test_pergunta_sem_resposta_nao_aceita_termos():
    with pytest.raises(ValueError):
        BenchmarkQuestion(id="q", pergunta="Pergunta de teste?", respondivel=False, categoria="fora_do_escopo", deve_conter=["x"])
