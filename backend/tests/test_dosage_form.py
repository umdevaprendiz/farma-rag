"""Mais de uma apresentação do mesmo remédio: qual bula a pergunta quer e quando perguntar (DECISOES §41).
Ollama falso, como em test_rag.py."""

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.db.vector_search import InMemoryIndex
from app.llm import ollama
from app.services import dosage_form, rag
from app.services.corpus import Corpus
from app.services.intent import inferir_secoes
from app.services.medicines import Catalog
from app.services.rag import PreviousQuestion
from ingestion.generate_chunks import gerar_chunks
from ingestion.process_leaflet import montar_bula

FIXTURES = Path(__file__).parent / "fixtures"
S = Settings(_env_file=None, top_k=2)
COMPRIMIDO, GOTAS = "exemplol-50-mg-laboratorio-teste", "exemplol-10-mg-ml-gotas-laboratorio-teste"
DOSE_INICIAL = "A dose inicial usual é de 50 mg uma vez ao dia"


@dataclass
class _Bula:
    id: str
    medicamento: str
    principio_ativo: str
    forma_farmaceutica: str | None


TYLALGIN = [
    _Bula("gotas", "Tylalgin 200 mg/mL gotas", "paracetamol", "solução oral gotas"),
    _Bula("cp", "Tylalgin 750 mg", "paracetamol", "comprimido"),
    _Bula("sache", "Tylalgin Sachet 500 mg", "paracetamol", "pó para solução oral (sachê)"),
]


# ---- A regra, sem o grafo -----------------------------------------------------------------


@pytest.mark.parametrize(("pergunta", "esperadas"), [
    ("Tylalgin gotas pode para criança?", ["gotas"]),
    ("quantas gotinhas de tylalgin?", ["gotas"]),
    ("Como tomar o Tylalgin sachê?", ["sache"]),
    ("E o sachet?", ["sache"]),
    ("Tylalgin 750 a cada quantas horas?", ["cp"]),
    ("Posso tomar 2 comprimidos de Tylalgin?", ["cp"]),
    ("Tylalgin para criança de 20 kg?", []),  # 20 kg é o peso, não a apresentação
    ("Como guardar o Tylalgin?", []),
    ("Tylalgin gotas ou comprimido ou sachê?", []),  # nomeou todas: não escolheu
])
def test_a_pergunta_escolhe_a_bula_pelo_que_so_ela_tem(pergunta, esperadas):
    assert [b.id for b in dosage_form.escolhidas(pergunta, TYLALGIN)] == esperadas


def test_numero_com_ponto_de_milhar():
    delumi = [
        _Bula("a", "Delumi 5.000, 10.000 e 15.000 UI", "colecalciferol", "comprimido revestido"),
        _Bula("b", "Delumi 7.000 e 50.000 UI", "colecalciferol", "comprimido revestido"),
    ]
    assert [b.id for b in dosage_form.escolhidas("Delumi 50.000 como tomar?", delumi)] == ["b"]


@pytest.mark.parametrize(("pergunta", "depende"), [
    ("Como tomar o Tylalgin?", True),
    ("Qual a dose máxima de Tylalgin?", True),
    ("Como guardar o Tylalgin?", True),
    ("Tylalgin pode para criança?", True),
    ("Como aplicar a loção?", True),
    ("Tomei Tylalgin demais, superdose?", True),
    ("Tylalgin tem interação com álcool?", False),
    ("Cliente bebe álcool todo dia, precisa de cuidado com Tylalgin?", False),  # o verbo, não o bebê
    ("Posso dar Tylalgin para o bebê?", True),
    ("Tylalgin é contraindicado na gravidez?", False),
    ("Para que serve o Tylalgin?", False),
])
def test_assuntos_que_mudam_com_a_apresentacao(pergunta, depende):
    assert dosage_form.depende(pergunta, inferir_secoes(pergunta)) is depende


def test_mesmo_produto_de_dois_laboratorios_nao_e_outra_apresentacao():
    dois_labs = [
        _Bula("a", "Exemplol 50 mg", "exemplina", "comprimido"),
        _Bula("b", "Exemplol 50 mg", "exemplina", "comprimido"),
    ]
    assert not dosage_form.diferentes(dois_labs)
    assert dosage_form.diferentes(TYLALGIN)


def test_botoes_com_a_forma_quando_o_nome_nao_diz():
    c = dosage_form.esclarecimento("paracetamol", TYLALGIN)
    assert c.motivo == "apresentacao"
    assert c.pergunta == "Tenho 3 bulas de paracetamol, e a resposta muda conforme a apresentação. Qual delas?"
    assert [(o.rotulo, o.principios_ativos, o.bula_id) for o in c.opcoes] == [
        ("Tylalgin 200 mg/mL gotas", ["paracetamol"], "gotas"),
        ("Tylalgin 750 mg (comprimido)", ["paracetamol"], "cp"),
        ("Tylalgin Sachet 500 mg", ["paracetamol"], "sache"),
    ]


# ---- No grafo -------------------------------------------------------------------------------


@pytest.fixture
def acervo():
    """O Exemplol em comprimido e em gotas (mesmo princípio ativo, mesmo texto), um eixo por trecho."""
    bulas = [
        montar_bula(FIXTURES / "sample_leaflet.txt", nome, "maleato de exemplina", "Laboratório Teste",
                    date(2025, 3, 15), ["Exemplol"], forma_farmaceutica=forma)
        for nome, forma in [("Exemplol 50 mg", "comprimido"), ("Exemplol 10 mg/mL gotas", "solução oral gotas")]
    ]
    assert [b.id for b in bulas] == [COMPRIMIDO, GOTAS]
    chunks = [c for b in bulas for c in gerar_chunks(b, 300, 50)]
    indice = InMemoryIndex([c.chunk_id for c in chunks], [c.principio_ativo for c in chunks], np.eye(len(chunks)))
    return Corpus({b.id: b for b in bulas}, {c.chunk_id: c for c in chunks}, indice, Catalog.montar(bulas, ["captopril"]))


@pytest.fixture
def modelo(monkeypatch, acervo):
    """A busca acha primeiro a posologia das duas bulas; a geração responde com a dose inicial, citando
    o [1]. Devolve a lista de prompts de geração (vazia = o modelo não foi chamado)."""
    ids = list(acervo.chunks)
    vetor = np.full(len(ids), 0.01)
    for i, chunk_id in enumerate(ids):
        if "__posologia__" in chunk_id:
            vetor[i] = 1.0
    monkeypatch.setattr(ollama, "embed", lambda textos, s=None: [vetor.tolist()])
    chamadas = []

    def chat(sistema, usuario, formato, s=None, modelo=None):
        chamadas.append(usuario)
        saida = {"status": "respondida", "afirmacoes": [{"texto": DOSE_INICIAL, "trecho": 1}]}
        return ollama.LLMResponse(json.dumps(saida), 400, 40, 8.0, 4.0)

    monkeypatch.setattr(ollama, "chat", chat)
    return chamadas


def _bulas_enviadas(r: rag.Result) -> set[str]:
    """As bulas dos trechos que foram ao modelo (ou à resposta copiada) em qualquer rodada."""
    return {c.split("__")[0] for t in r.tentativas for c in t.chunk_ids} or {f.chunk.bula_id for f in r.fontes}


def test_sem_apresentacao_na_pergunta_pergunta_qual_sem_chamar_o_modelo(acervo, modelo):
    r = rag.responder("Como tomar o Exemplol?", acervo, s=S)
    assert (r.status, r.rodadas, r.fontes, modelo) == ("precisa_esclarecimento", 0, [], [])
    assert r.esclarecimento.motivo == "apresentacao"
    assert [(o.rotulo, o.bula_id) for o in r.esclarecimento.opcoes] == [
        ("Exemplol 10 mg/mL gotas", GOTAS), ("Exemplol 50 mg (comprimido)", COMPRIMIDO),
    ]


def test_o_botao_reenvia_com_a_bula_e_so_ela_vai_ao_modelo(acervo, modelo):
    anterior = PreviousQuestion("Como tomar o Exemplol?", ["maleato de exemplina"], pediu_esclarecimento=True)
    r = rag.responder("Como tomar o Exemplol?", acervo, ["maleato de exemplina"], S, anterior, bula_id=GOTAS)
    assert r.status == "respondida" and len(modelo) == 1
    assert _bulas_enviadas(r) == {GOTAS} and r.bulas == [GOTAS]


def test_apresentacao_na_pergunta_escolhe_sem_perguntar(acervo, modelo):
    r = rag.responder("Como tomar o Exemplol gotas?", acervo, s=S)
    assert r.status == "respondida"
    assert _bulas_enviadas(r) == {GOTAS} and r.bulas == [GOTAS]


def test_resposta_escrita_ao_esclarecimento_escolhe_e_nunca_pergunta_de_novo(acervo, modelo):
    anterior = PreviousQuestion("Como tomar o Exemplol?", ["maleato de exemplina"], pediu_esclarecimento=True)
    r = rag.responder("o comprimido", acervo, s=S, anterior=anterior)
    assert (r.status, _bulas_enviadas(r)) == ("respondida", {COMPRIMIDO})
    # Sem dizer qual: segue com as duas (no máximo um esclarecimento por pergunta, §14).
    r = rag.responder("tanto faz", acervo, s=S, anterior=anterior)
    assert r.status != "precisa_esclarecimento" and modelo
    assert _bulas_enviadas(r) == {COMPRIMIDO, GOTAS} and r.bulas == []


def test_pergunta_seguinte_continua_na_apresentacao_escolhida(acervo, modelo):
    anterior = PreviousQuestion("Como tomar o Exemplol gotas?", ["maleato de exemplina"], False, [GOTAS])
    r = rag.responder("E para criança?", acervo, s=S, anterior=anterior)
    # O modelo falso responde a dose, que não fala de criança: aqui importa que não perguntou e só as gotas foram.
    assert r.status != "precisa_esclarecimento" and modelo
    assert _bulas_enviadas(r) == {GOTAS} and r.bulas == [GOTAS]


def test_assunto_que_nao_muda_com_a_apresentacao_nao_pergunta(acervo, modelo):
    r = rag.responder("O Exemplol tem interações?", acervo, s=S)
    assert r.status != "precisa_esclarecimento" and modelo
    assert _bulas_enviadas(r) == {COMPRIMIDO, GOTAS} and r.bulas == []


def test_dose_copiada_da_bula_vem_por_apresentacao_sem_perguntar(acervo, modelo):
    # §25: a frase da posologia de cada bula, sem o LLM, já é o "depende da apresentação".
    r = rag.responder("Qual a dose máxima do Exemplol?", acervo, s=S)
    assert (r.status, r.extrativa, modelo) == ("respondida", True, [])
