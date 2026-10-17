import numpy as np
import pytest

from app.db.vector_search import InMemoryIndex, bytes_para_vetor, vetor_para_bytes


def _indice():
    # 2 dimensões, para dar para raciocinar: ângulo define a similaridade
    linhas = [
        ("los__interacoes__000", "losartana potássica", vetor_para_bytes([1.0, 0.0])),
        ("los__posologia__000", "losartana potássica", vetor_para_bytes([0.7, 0.7])),
        ("ibu__interacoes__000", "ibuprofeno", vetor_para_bytes([0.9, 0.1])),
        ("ibu__posologia__000", "ibuprofeno", vetor_para_bytes([0.0, 1.0])),
    ]
    return InMemoryIndex.de_linhas(linhas)


def test_roundtrip_dos_bytes():
    v = [0.1, -0.5, 3.0]
    assert np.allclose(bytes_para_vetor(vetor_para_bytes(v)), v)


def test_top_k_ordenado_por_cosseno():
    r = _indice().buscar([1.0, 0.0], k=2)
    assert [x.chunk_id for x in r] == ["los__interacoes__000", "ibu__interacoes__000"]
    assert r[0].score == pytest.approx(1.0)


def test_cosseno_ignora_o_tamanho_do_vetor():
    a = _indice().buscar([1.0, 0.0], k=1)[0].score
    b = _indice().buscar([50.0, 0.0], k=1)[0].score
    assert a == pytest.approx(b)


def test_filtro_por_principio_ativo_com_acento_e_caixa():
    r = _indice().buscar([1.0, 0.0], k=4, principios_ativos=["Losartana Potassica"])
    assert {x.chunk_id.split("__")[0] for x in r} == {"los"}


def test_dois_medicamentos_recebem_trechos_dos_dois():
    # sem a divisão, o top-2 seriam los e ibu "interacoes"; aqui garantimos 1+ de cada
    r = _indice().buscar([0.0, 1.0], k=2, principios_ativos=["losartana potassica", "ibuprofeno"])
    assert {x.chunk_id.split("__")[0] for x in r} == {"los", "ibu"}
    assert r[0].score >= r[1].score


def test_casos_vazios():
    assert InMemoryIndex.de_linhas([]).buscar([1.0, 0.0], k=3) == []
    assert _indice().buscar([0.0, 0.0], k=3) == []
    assert _indice().buscar([1.0, 0.0], k=3, principios_ativos=["inexistente"]) == []
