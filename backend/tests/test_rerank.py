"""Reranking da rodada 1 (services/rerank.py; docs/DECISOES.md, seção 30)."""

from datetime import date

import pytest

from app.db.vector_search import SearchHit
from app.schemas.dataset import Chunk
from app.services import rerank
from app.services.intent import grupos_do_assunto, secoes_para_rerank


def _chunk(chunk_id: str, secao: str, texto: str, ativo: str = "ibuprofeno") -> Chunk:
    bula = chunk_id.split("__")[0]
    return Chunk(
        chunk_id=f"{bula}__{secao}__000", bula_id=bula, medicamento="Ibuprofeno 400 mg", principio_ativo=ativo,
        data_versao=date(2025, 1, 1), secao=secao, pagina=1, ordem=0, texto=texto, n_tokens=20,
    )


EFICACIA = _chunk("a", "resultados_eficacia", "Em estudos, o ibuprofeno reduziu a febre em crianças com resposta rápida.")
CONTRA = _chunk("b", "contraindicacoes", "Este medicamento é contraindicado em caso de suspeita de dengue.")
POSO = _chunk("c", "posologia", "Tomar 1 cápsula a cada 6 horas.")
CHUNKS = {c.chunk_id: c for c in (EFICACIA, CONTRA, POSO)}


def test_secao_e_assunto_viram_o_mais_parecido():
    # Semelhança sozinha: eficácia na frente. A seção pedida e o assunto ("dengue") viram o jogo.
    candidatos = [SearchHit(EFICACIA.chunk_id, 0.70), SearchHit(CONTRA.chunk_id, 0.66), SearchHit(POSO.chunk_id, 0.60)]
    grupos = grupos_do_assunto("suspeita de dengue")
    escolhidos = rerank.reordenar(candidatos, CHUNKS, ["contraindicacoes"], grupos, 1, ["ibuprofeno"])
    assert [h.chunk_id for h in escolhidos] == [CONTRA.chunk_id]
    assert escolhidos[0].score == 0.66  # a tela continua mostrando a semelhança, não a nota


def test_sem_secao_nem_assunto_fica_a_ordem_da_semelhanca():
    # A linha de base (garantir_secao=False, F3b desligada): só a semelhança.
    candidatos = [SearchHit(POSO.chunk_id, 0.60), SearchHit(EFICACIA.chunk_id, 0.70), SearchHit(CONTRA.chunk_id, 0.66)]
    escolhidos = rerank.reordenar(candidatos, CHUNKS, [], [], 3, ["ibuprofeno"], garantir_secao=False)
    assert [h.chunk_id for h in escolhidos] == [EFICACIA.chunk_id, CONTRA.chunk_id, POSO.chunk_id]


def test_trecho_quase_igual_de_outra_bula_nao_ocupa_vaga():
    # As bulas do Tylalgin repetem as mesmas frases: a segunda cópia daria um trecho a menos de informação.
    gotas = _chunk("gotas", "posologia", "A dose diária máxima de paracetamol é de 4000 mg em doses fracionadas.", "paracetamol")
    comprimido = _chunk("comp", "posologia", "A dose diária máxima de paracetamol é de 4000 mg em doses fracionadas.", "paracetamol")
    outro = _chunk("comp", "advertencias", "Usuários crônicos de álcool têm risco aumentado.", "paracetamol")
    chunks = {c.chunk_id: c for c in (gotas, comprimido, outro)}
    hits = [SearchHit(gotas.chunk_id, 0.8), SearchHit(comprimido.chunk_id, 0.79), SearchHit(outro.chunk_id, 0.5)]
    assert [h.chunk_id for h in rerank.reordenar(hits, chunks, [], [], 2, ["paracetamol"])] == [gotas.chunk_id, outro.chunk_id]


def test_divide_as_vagas_entre_os_remedios():
    los = _chunk("los", "interacoes", "Anti-inflamatórios podem reduzir o efeito anti-hipertensivo.", "losartana potássica")
    hits = [SearchHit(EFICACIA.chunk_id, 0.9), SearchHit(CONTRA.chunk_id, 0.85), SearchHit(POSO.chunk_id, 0.8), SearchHit(los.chunk_id, 0.4)]
    chunks = {**CHUNKS, los.chunk_id: los}
    escolhidos = rerank.reordenar(hits, chunks, [], [], 3, ["ibuprofeno", "losartana potássica"])
    assert los.chunk_id in [h.chunk_id for h in escolhidos]  # 2 por remédio, no máximo


@pytest.mark.parametrize("pergunta, remedios, tem, nao_tem", [
    # "Pode tomar?" é pergunta de segurança, mesmo sem a palavra "contraindicado".
    ("Paciente com febre e suspeita de dengue pode tomar ibuprofeno 400 mg?", 1, {"contraindicacoes", "advertencias"}, set()),
    # Dois remédios: uso junto, e não contraindicação de cada um.
    ("Paciente que toma lítio pode usar diclofenaco?", 2, {"interacoes"}, {"contraindicacoes"}),
    # Pergunta de horário é de posologia, mesmo com "pode tomar".
    ("Em que momento deve tomar o zolpidem? Pode tomar outro de madrugada?", 1, {"posologia"}, {"contraindicacoes"}),
    ("Por quantos dias no máximo dá para usar a oximetazolina?", 1, {"advertencias"}, set()),
    ("Isotretinoína precisa ser protegida da luz?", 1, {"armazenamento"}, set()),
])
def test_secoes_para_rerank(pergunta, remedios, tem, nao_tem):
    secoes = set(secoes_para_rerank(pergunta, remedios))
    assert tem <= secoes and not (nao_tem & secoes)


# --- F3b: a seção da pergunta já na rodada 1 (docs/DECISOES.md, seção 34) ----------------------

ADVERT = _chunk("d", "advertencias", "Em pacientes com insuficiência renal, monitorar a creatinina.")
REACAO = _chunk("e", "reacoes_adversas", "Reação comum: tontura. Reação incomum: tosse seca.")
INDICA = _chunk("f", "indicacoes", "O ibuprofeno é indicado para dor de cabeça e febre.")
FARMACO = _chunk("g", "caracteristicas_farmacologicas", "O ibuprofeno inibe a ciclo-oxigenase; o efeito começa em 30 minutos.")
MAIS = {c.chunk_id: c for c in (EFICACIA, CONTRA, POSO, ADVERT, REACAO, INDICA, FARMACO)}


def _ids(hits):
    return [h.chunk_id for h in hits]


def test_secao_rara_sem_ser_pedida_vai_para_o_fim_da_fila():
    # Caso 12f61084: dois trechos de eficácia na frente da indicação, para uma pergunta de indicação.
    hits = [SearchHit(EFICACIA.chunk_id, 0.80), SearchHit(FARMACO.chunk_id, 0.78), SearchHit(INDICA.chunk_id, 0.70),
            SearchHit(POSO.chunk_id, 0.60)]
    com = rerank.reordenar(hits, MAIS, ["indicacoes"], [], 2, ["ibuprofeno"])
    assert _ids(com) == [INDICA.chunk_id, POSO.chunk_id]
    sem = rerank.reordenar(hits, MAIS, ["indicacoes"], [], 2, ["ibuprofeno"], garantir_secao=False)
    assert EFICACIA.chunk_id in _ids(sem) or FARMACO.chunk_id in _ids(sem)  # a linha de base mantinha


def test_secao_rara_pedida_fica_na_frente():
    hits = [SearchHit(POSO.chunk_id, 0.80), SearchHit(EFICACIA.chunk_id, 0.60)]
    assert _ids(rerank.reordenar(hits, MAIS, ["resultados_eficacia"], [], 1, ["ibuprofeno"])) == [EFICACIA.chunk_id]


def test_secao_rara_entra_se_faltar_candidato():
    # Rebaixada, não excluída: sem outros candidatos, é melhor um trecho de eficácia que nenhum.
    hits = [SearchHit(EFICACIA.chunk_id, 0.60), SearchHit(POSO.chunk_id, 0.50)]
    assert _ids(rerank.reordenar(hits, MAIS, [], [], 3, ["ibuprofeno"])) == [POSO.chunk_id, EFICACIA.chunk_id]


def test_garante_um_trecho_da_secao_pedida_no_lugar_do_menos_parecido():
    # O bônus (0,08) não alcança a distância de 0,3: sem a garantia, a contraindicação ficava de fora.
    hits = [SearchHit(POSO.chunk_id, 0.80), SearchHit(ADVERT.chunk_id, 0.78), SearchHit(REACAO.chunk_id, 0.75),
            SearchHit(CONTRA.chunk_id, 0.40)]
    sem = rerank.reordenar(hits, MAIS, ["contraindicacoes"], [], 3, ["ibuprofeno"], garantir_secao=False)
    assert CONTRA.chunk_id not in _ids(sem)
    com = rerank.reordenar(hits, MAIS, ["contraindicacoes"], [], 3, ["ibuprofeno"])
    assert set(_ids(com)) == {POSO.chunk_id, ADVERT.chunk_id, CONTRA.chunk_id}  # saiu a de menor semelhança
    assert next(h for h in com if h.chunk_id == CONTRA.chunk_id).score == 0.40  # a tela mostra a semelhança


def test_nao_troca_quando_a_secao_pedida_ja_esta_entre_os_escolhidos():
    hits = [SearchHit(POSO.chunk_id, 0.80), SearchHit(CONTRA.chunk_id, 0.70), SearchHit(REACAO.chunk_id, 0.60),
            SearchHit(ADVERT.chunk_id, 0.30)]
    com = rerank.reordenar(hits, MAIS, ["contraindicacoes", "advertencias"], [], 3, ["ibuprofeno"])
    sem = rerank.reordenar(hits, MAIS, ["contraindicacoes", "advertencias"], [], 3, ["ibuprofeno"], garantir_secao=False)
    assert _ids(com) == _ids(sem)


def test_remedio_sem_trecho_da_secao_fica_como_esta():
    hits = [SearchHit(POSO.chunk_id, 0.80), SearchHit(REACAO.chunk_id, 0.70)]
    assert _ids(rerank.reordenar(hits, MAIS, ["interacoes"], [], 2, ["ibuprofeno"])) == [POSO.chunk_id, REACAO.chunk_id]


def test_cada_remedio_recebe_um_trecho_da_secao():
    los_poso = _chunk("los", "posologia", "A dose inicial é de 50 mg.", "losartana potássica")
    los_inter = _chunk("los", "interacoes", "Anti-inflamatórios podem reduzir o efeito anti-hipertensivo.", "losartana potássica")
    ibu_inter = _chunk("ibu", "interacoes", "O uso com losartana pode reduzir o efeito anti-hipertensivo.", "ibuprofeno")
    chunks = {**MAIS, los_poso.chunk_id: los_poso, los_inter.chunk_id: los_inter, ibu_inter.chunk_id: ibu_inter}
    hits = [SearchHit(POSO.chunk_id, 0.90), SearchHit(los_poso.chunk_id, 0.85), SearchHit(ibu_inter.chunk_id, 0.30),
            SearchHit(los_inter.chunk_id, 0.20)]
    escolhidos = rerank.reordenar(hits, chunks, ["interacoes"], [], 2, ["ibuprofeno", "losartana potássica"])
    assert set(_ids(escolhidos)) == {ibu_inter.chunk_id, los_inter.chunk_id}  # uma interação de cada


def test_trecho_da_secao_quase_igual_a_um_que_fica_nao_entra():
    repetido = _chunk("r", "contraindicacoes", "Tomar 1 cápsula a cada 6 horas.")
    hits = [SearchHit(POSO.chunk_id, 0.80), SearchHit(ADVERT.chunk_id, 0.70), SearchHit(repetido.chunk_id, 0.40)]
    chunks = {**MAIS, repetido.chunk_id: repetido}
    # O único candidato da seção repete a posologia que fica: não vale a vaga.
    assert _ids(rerank.reordenar(hits, chunks, ["contraindicacoes"], [], 2, ["ibuprofeno"])) == [POSO.chunk_id, ADVERT.chunk_id]


@pytest.mark.parametrize("pergunta, seção", [
    ("O que dizem os estudos sobre a eficácia do ibuprofeno?", "resultados_eficacia"),
    ("Qual a evidência de que o remédio funciona?", "resultados_eficacia"),
    ("Quanto tempo o ibuprofeno demora para fazer efeito?", "caracteristicas_farmacologicas"),
    ("Como é a farmacocinética e a absorção da loratadina?", "caracteristicas_farmacologicas"),
    ("Qual o mecanismo de ação do omeprazol?", "caracteristicas_farmacologicas"),
])
def test_perguntas_que_pedem_as_secoes_raras(pergunta, seção):
    assert seção in secoes_para_rerank(pergunta, 1)


@pytest.mark.parametrize("pergunta", [
    "Para que serve o ibuprofeno?", "Qual a dose máxima de paracetamol?", "Pode tomar ibuprofeno com dengue?",
    "Ibuprofeno causa sono?",
])
def test_perguntas_comuns_nao_pedem_as_secoes_raras(pergunta):
    assert not (rerank.SECOES_RARAS & set(secoes_para_rerank(pergunta, 1)))
