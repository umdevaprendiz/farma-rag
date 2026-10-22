"""Comparação do número de candidatos do reranking no benchmark (docs/DECISOES.md, seção 32)."""

from benchmark.run import comparar, resumir


def _linha(id: str, enviados: list[str], acerto: bool | None, resposta: str | None = None, respondivel: bool = True) -> dict:
    return {
        "id": id, "pergunta": f"pergunta {id}", "categoria": "posologia", "respondivel": respondivel,
        "caminho": "semelhanca", "status": "respondida" if resposta else "nao_encontrada", "rodadas": 1,
        "segundos": 1.0, "citacoes_corrigidas": 0, "acerto": acerto, "enviados": enviados,
        "posicao_nos_candidatos": 2, "resposta": resposta,
    }


BASE = [
    _linha("p1", ["a", "b", "c"], True, "Resposta 1 [1]."),
    _linha("p2", ["d", "e", "f"], False, "Resposta 2 [1]."),
    _linha("p3", ["g", "h", "i"], True, "Resposta 3 [2]."),
    _linha("x1", ["j"], None, respondivel=False),
]


def test_compara_trechos_e_acerto_com_a_base():
    agora = [
        BASE[0],
        _linha("p2", ["d", "x", "f"], True, "Resposta 2 [1]."),  # trecho novo, passou a acertar
        _linha("p3", ["h", "g", "i"], False, "Resposta 3 [2]."),  # mesma lista em outra ordem = outro prompt
        BASE[3],
    ]
    c = comparar(BASE, agora, "busca")
    assert c == {
        "mesmos_trechos_rodada1": "2/4", "trechos_mudaram": ["p2", "p3"],
        "passou_a_acertar": ["p2"], "deixou_de_acertar": ["p3"],
    }


def test_no_modo_completo_compara_a_resposta():
    agora = [BASE[0], _linha("p2", ["d", "x", "f"], True, "Outra resposta [1]."), BASE[2], BASE[3]]
    c = comparar(BASE, agora, "completo")
    assert c["mesma_resposta"] == "3/4"
    assert c["respostas_que_mudaram"] == [
        {"id": "p2", "pergunta": "pergunta p2", "base": "Resposta 2 [1].", "agora": "Outra resposta [1]."}
    ]


def test_resumo_diz_entre_quantos_candidatos_a_ancora_esta():
    assert "semelhanca_ancora_nos_25_candidatos" in resumir(BASE, "busca", 25)
    assert "semelhanca_ancora_nos_10_candidatos" in resumir(BASE, "busca")  # o nome de antes, para comparar
