from app.schemas.dataset import BenchmarkQuestion, Chunk
from benchmark.metrics import acertou_recall, recall_at_k, recusa_correta


def _chunk(ordem, texto, ativo="losartana potássica", secao="interacoes"):
    return Chunk(
        chunk_id=f"b1__{secao}__{ordem:03d}", bula_id="b1", medicamento="Losartana",
        principio_ativo=ativo, secao=secao, pagina=4, ordem=ordem, texto=texto, n_tokens=10,
    )


def _pergunta(id, ancoras, respondivel=True):
    return BenchmarkQuestion(
        id=id, pergunta="pergunta de teste aqui", respondivel=respondivel, ancoras=ancoras,
        resposta_esperada="x" if respondivel else None,
        categoria="interacao" if respondivel else "fora_do_escopo",
    )


ANC = {"principio_ativo": "losartana potássica", "secao": "interacoes",
       "frase": "anti-inflamatórios não esteroidais"}
ANC_IBU = {"principio_ativo": "ibuprofeno", "secao": "interacoes", "frase": "anti-hipertensivos podem ter efeito reduzido"}


def test_acerto_independe_do_id_e_do_tamanho_do_trecho():
    q = _pergunta("q1", [ANC])
    grande = _chunk(0, "Texto longo. O uso com Anti-inflamatórios não esteroidais reduz o efeito. Mais texto.")
    pequeno = _chunk(7, "o uso com anti-inflamatórios não esteroidais reduz")
    assert acertou_recall([grande], q) and acertou_recall([pequeno], q)


def test_erra_se_secao_ou_medicamento_forem_outros():
    q = _pergunta("q1", [ANC])
    texto = "anti-inflamatórios não esteroidais"
    assert not acertou_recall([_chunk(0, texto, secao="posologia")], q)
    assert not acertou_recall([_chunk(0, texto, ativo="ibuprofeno")], q)
    assert not acertou_recall([_chunk(0, "nada a ver")], q)


def test_pergunta_com_dois_medicamentos_exige_as_duas_ancoras():
    q = _pergunta("q2", [ANC, ANC_IBU])
    losartana = _chunk(0, "anti-inflamatórios não esteroidais")
    ibuprofeno = _chunk(0, "anti-hipertensivos podem ter efeito reduzido", ativo="ibuprofeno")
    assert not acertou_recall([losartana], q)
    assert acertou_recall([losartana, ibuprofeno], q)


def test_recall_at_k_respeita_o_k():
    qs = [_pergunta("q1", [ANC]), _pergunta("q2", [ANC])]
    certo = _chunk(0, "anti-inflamatórios não esteroidais")
    ruido = _chunk(1, "irrelevante")
    recuperados = {"q1": [certo, ruido], "q2": [ruido, certo]}
    assert recall_at_k(recuperados, qs, k=1) == 0.5
    assert recall_at_k(recuperados, qs, k=2) == 1.0


def test_recusa_correta_so_conta_perguntas_sem_resposta():
    qs = [_pergunta("q1", [ANC]), _pergunta("q41", [], respondivel=False), _pergunta("q42", [], respondivel=False)]
    assert recusa_correta({"q1": "respondida", "q41": "nao_encontrada", "q42": "respondida"}, qs) == 0.5
