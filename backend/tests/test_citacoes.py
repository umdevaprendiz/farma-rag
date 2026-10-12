import pytest

from app.schemas.api import CodigoErro
from app.services.citacoes import SaidaInvalidaError, marcadores_no_texto, validar_saida


def _ok(resposta="Segundo a bula, reduz o efeito [1]. Também [2].", citacoes=(1, 2)):
    return {"status": "respondida", "resposta": resposta, "citacoes": list(citacoes)}


def _codigo(bruto, n=3):
    with pytest.raises(SaidaInvalidaError) as e:
        validar_saida(bruto, n)
    return e.value.codigo


def test_aceita_saida_valida_dict_e_string():
    assert validar_saida(_ok(), 3).citacoes == [1, 2]
    assert validar_saida('{"status":"respondida","resposta":"x [1]","citacoes":[1]}', 3).status == "respondida"


def test_aceita_nao_encontrada():
    assert validar_saida({"status": "nao_encontrada"}, 3).status == "nao_encontrada"


def test_json_quebrado_e_saida_invalida():
    assert _codigo("{nao é json") == CodigoErro.SAIDA_INVALIDA


def test_campos_errados_e_saida_invalida():
    assert _codigo({"status": "talvez"}) == CodigoErro.SAIDA_INVALIDA
    assert _codigo({"status": "respondida", "resposta": "  ", "citacoes": [1]}) == CodigoErro.SAIDA_INVALIDA


def test_nao_encontrada_com_resposta_e_incoerente():
    assert _codigo({"status": "nao_encontrada", "resposta": "x [1]", "citacoes": [1]}) == CodigoErro.SAIDA_INVALIDA


def test_citacao_inventada_e_rejeitada():
    assert _codigo(_ok("Texto [1] e [7].", (1, 7))) == CodigoErro.CITACAO_INVALIDA
    assert _codigo(_ok("Texto [0].", (0,))) == CodigoErro.CITACAO_INVALIDA


def test_resposta_sem_citacao_e_rejeitada():
    assert _codigo(_ok("Texto sem marcador.", ())) == CodigoErro.CITACAO_INVALIDA


def test_texto_e_lista_precisam_coincidir():
    assert _codigo(_ok("Texto [1] e [2].", (1,))) == CodigoErro.CITACAO_INVALIDA  # [2] não listado
    assert _codigo(_ok("Texto [1].", (1, 2))) == CodigoErro.CITACAO_INVALIDA  # 2 listado sem aparecer


def test_marcadores_agrupados():
    assert marcadores_no_texto("Ver [1, 3] e [2].") == {1, 2, 3}
    assert validar_saida(_ok("Ver [1, 2].", (1, 2)), 3).citacoes == [1, 2]
