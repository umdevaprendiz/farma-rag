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
    candidatos = [SearchHit(POSO.chunk_id, 0.60), SearchHit(EFICACIA.chunk_id, 0.70), SearchHit(CONTRA.chunk_id, 0.66)]
    escolhidos = rerank.reordenar(candidatos, CHUNKS, [], [], 3, ["ibuprofeno"])
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
