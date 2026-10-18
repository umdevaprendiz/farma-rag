"""Para que serve, pela seção de indicações (services/indication.py; docs/DECISOES.md, seção 29)."""

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.db.vector_search import InMemoryIndex
from app.llm import ollama
from app.services import rag
from app.services.corpus import Corpus
from app.services.indication import Verdict, nome_curto, situacao, uso_perguntado
from app.services.medicines import Catalog
from ingestion.generate_chunks import gerar_chunks
from ingestion.process_leaflet import montar_bula

FIXTURES = Path(__file__).parent / "fixtures"
S = Settings(_env_file=None, top_k=2)

# Textos reais da seção de indicações (data/processed/bulas, 18/10/2026), encurtados.
PARACETAMOL = (
    "Este medicamento é indicado, em adultos, para a redução da febre e o alívio temporário de dores leves a "
    "moderadas, tais como: dores associadas a resfriados comuns, dor de cabeça, dor no corpo, dor de dente, dor "
    "nas costas, dores musculares, dores leves associadas a artrites e dismenorreia."
)
LORATADINA = (
    "Alívio temporário dos sintomas associados com rinite alérgica (por exemplo: febre do feno), como: coceira "
    "nasal, nariz escorrendo (coriza), espirros, ardor e coceira nos olhos."
)
DICLOFENACO = (
    "Este medicamento está indicado para o tratamento de: Formas degenerativas e inflamatórias de reumatismo: "
    "reumatismo não articular; Como auxiliar no tratamento de processos infecciosos acompanhados de dor e "
    "inflamação de ouvido, nariz ou garganta, como por exemplo, faringoamigdalites, otites."
)
COZAAR = (
    "COZAAR® é indicado para o tratamento da hipertensão. Não é recomendado trocar inibidores da ECA por COZAAR® "
    "no tratamento de pacientes com insuficiência cardíaca que estejam estabilizados com a terapia atual."
)


@pytest.mark.parametrize("pergunta, uso", [
    # As três perguntas reais de 18/10 que voltaram "não encontrei", e a do rocutan.
    ("oque acha de eu tomar zolpidem para nauséa?", "nauséa"),
    ("certralina é pra vomito?", "vomito"),
    ("paracetamol serve para  ansiedade?", "ansiedade"),
    ("tomei rocutan para dor no ouvido, fiz certo?", "dor no ouvido"),
    ("paracetamol é bom pra dor de cabeça?", "dor de cabeça"),
    ("Loratadina serve para alergia?", "alergia"),
    ("pozato serve para evitar gravidez?", "evitar gravidez"),
    ("Para que serve a dipirona?", ""),
    ("dipirona serve pra que?", ""),
    ("o que a isotretinoína trata?", ""),
])
def test_pergunta_de_uso(pergunta, uso):
    assert uso_perguntado(pergunta) == uso


@pytest.mark.parametrize("pergunta", [
    # Outra condição junto: a resposta tem de vir da contraindicação, advertência ou posologia.
    "posso tomar ibuprofeno para dor se tenho dengue?",
    "posso tomar dipirona para dor de cabeça estando grávida?",
    "quantas gotas de dipirona para febre?",
    "posso dar dipirona para criança?",
    "Qual a dose máxima de paracetamol?",
    "omeprazol pode com clopidogrel?",
    "Exemplol tem interações?",
    "Qual o melhor remédio para pressão alta?",  # sem remédio: o grafo nem usa (busca geral)
    "e para dor de cabeça?",  # elíptica sem pergunta de uso antes
])
def test_nao_e_pergunta_so_de_uso(pergunta):
    assert uso_perguntado(pergunta) is None


def test_eliptica_depois_de_pergunta_de_uso():
    assert uso_perguntado("e para dor de cabeça?", "paracetamol serve para ansiedade?") == "dor de cabeça"
    assert uso_perguntado("e para dor de cabeça?", "dipirona pode na gravidez?") is None
    assert uso_perguntado("e para criança?", "paracetamol serve para ansiedade?") is None


@pytest.mark.parametrize("uso, indicacoes, esperado", [
    ("ansiedade", PARACETAMOL, "ausente"),
    ("dor de cabeca", PARACETAMOL, "citado"),
    ("febre", PARACETAMOL, "citado"),
    ("dor no ouvido", PARACETAMOL, "geral"),  # "dores leves a moderadas", sem citar ouvido
    ("dor no ouvido", DICLOFENACO, "citado"),  # "não articular" não nega a indicação
    ("febre", LORATADINA, "ausente"),  # "febre do feno" é rinite alérgica
    ("alergia", LORATADINA, "citado"),
    ("insuficiencia cardiaca", COZAAR, "ressalva"),  # só aparece em frase que nega
    ("pressao alta", COZAAR, "citado"),
])
def test_situacao(uso, indicacoes, esperado):
    assert situacao(uso, [indicacoes]) == esperado


def test_conclusao_montada_no_codigo():
    assert nome_curto("cloridrato de sertralina") == "sertralina"
    assert nome_curto("ácido acetilsalicílico") == "ácido acetilsalicílico"
    assert nome_curto("amoxicilina tri-hidratada") == "amoxicilina"
    assert Verdict("paracetamol", "ansiedade", "ausente").frase() == (
        "Não. Ansiedade não está entre as indicações da bula de paracetamol: pela bula, não é indicado para "
        "isso. A bula indica:"
    )
    assert Verdict("paracetamol", "dor no ouvido", "geral").frase().startswith("A bula de paracetamol não cita dor no ouvido")
    assert Verdict("dipirona", "", "lista").frase() == "Para que serve, pela bula de dipirona:"


@pytest.fixture
def acervo(monkeypatch):
    bula = montar_bula(
        FIXTURES / "sample_leaflet.txt", "Exemplol 50 mg", "maleato de exemplina", "Laboratório Teste",
        date(2025, 3, 15), ["Exemplol"],
    )
    chunks = gerar_chunks(bula, 300, 50)
    indice = InMemoryIndex([c.chunk_id for c in chunks], [c.principio_ativo for c in chunks], np.eye(len(chunks)))
    monkeypatch.setattr(ollama, "embed", lambda textos, s=None: [np.full(len(chunks), 0.1).tolist()])
    monkeypatch.setattr(ollama, "chat", lambda *a, **k: pytest.fail("pergunta de uso não chama o LLM"))
    return Corpus({bula.id: bula}, {c.chunk_id: c for c in chunks}, indice, Catalog.montar([bula], ["captopril"]))


INDICACAO_EXEMPLOL = "Exemplol é indicado para o tratamento da hipertensão arterial em adultos"


def test_uso_que_a_bula_nao_cita_nao_e_indicado(acervo):
    r = rag.responder("Exemplol serve para ansiedade?", acervo, s=S)
    assert (r.status, r.extrativa, r.rodadas) == ("respondida", True, 1)
    assert r.resposta == f"{INDICACAO_EXEMPLOL} [1]."
    assert [(f.chunk.secao, f.citado) for f in r.fontes] == [("indicacoes", True)]
    assert r.conclusao.startswith("Não. Ansiedade não está entre as indicações da bula de exemplina")


def test_uso_citado_nas_indicacoes(acervo):
    r = rag.responder("Exemplol serve para pressão alta?", acervo, s=S)
    assert (r.status, r.resposta) == ("respondida", f"{INDICACAO_EXEMPLOL} [1].")
    assert r.conclusao == "Sim. Pela bula de exemplina, é indicado para isso:"


def test_para_que_serve(acervo):
    r = rag.responder("Para que serve o Exemplol?", acervo, s=S)
    assert (r.status, r.conclusao) == ("respondida", "Para que serve, pela bula de exemplina:")


def test_remedio_sem_bula_nao_busca_nem_conclui(acervo, monkeypatch):
    monkeypatch.setattr(ollama, "embed", lambda *a, **k: pytest.fail("não busca: as outras bulas só o citam de passagem"))
    r = rag.responder("Captopril serve para ansiedade?", acervo, s=S)
    assert (r.status, r.motivo, r.fontes) == ("nao_encontrada", "medicamento_nao_indexado", [])
    assert r.conclusao == "A bula de captopril não está no acervo, então não dá para dizer se é indicado para ansiedade."


def test_pergunta_com_outra_condicao_segue_o_caminho_normal(acervo, monkeypatch):
    chamadas = []
    monkeypatch.setattr(ollama, "chat", lambda *a, **k: chamadas.append(a) or ollama.LLMResponse(
        '{"status": "nao_encontrada", "afirmacoes": []}', 1, 1, 0, 0))
    r = rag.responder("Posso tomar Exemplol para pressão alta estando grávida?", acervo, s=S)
    assert chamadas and r.conclusao is None
