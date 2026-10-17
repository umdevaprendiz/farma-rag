"""Conversa (docs/DECISOES.md, seção 23): pergunta que continua a anterior, esclarecimento do remédio
da alergia e alternativas a ele. Ollama falso, como em test_rag.py."""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.db.vector_search import InMemoryIndex
from app.llm import ollama
from app.services import graph, rag
from app.services.corpus import Corpus
from app.services.intent import eliptica, pede_remedio, responde_ao_assunto, texto_apos_alergia
from app.services.medicines import Catalog
from app.services.rag import PreviousQuestion
from ingestion.generate_chunks import gerar_chunks
from ingestion.process_leaflet import montar_bula

FIXTURES = Path(__file__).parent / "fixtures"
S = Settings(_env_file=None, top_k=2)
EXEMPLINA, OUTRINICO = "maleato de exemplina", "ácido outrínico"
DIPIRONA_ANVISA = {
    "associacao": False, "classe_terapeutica": "ANALGESICOS NAO NARCOTICOS", "n_registros": 10,
    "nome_produto": "NOVALGINA", "principios_ativos": ["dipirona"],
}


@pytest.fixture
def acervo():
    """Exemplol e Outrol (mesmo texto, princípios ativos diferentes), um eixo por trecho; dipirona é
    conhecida, sem bula, com classe terapêutica."""
    bulas = [
        montar_bula(FIXTURES / "sample_leaflet.txt", nome, ativo, "Laboratório Teste", date(2025, 3, 15), [nome.split()[0]])
        for nome, ativo in [("Exemplol 50 mg", EXEMPLINA), ("Outrol 10 mg", OUTRINICO)]
    ]
    chunks = [c for b in bulas for c in gerar_chunks(b, 300, 50)]
    indice = InMemoryIndex([c.chunk_id for c in chunks], [c.principio_ativo for c in chunks], np.eye(len(chunks)))
    catalogo = Catalog.montar(bulas, ["dipirona"], [DIPIRONA_ANVISA])
    return Corpus({b.id: b for b in bulas}, {c.chunk_id: c for c in chunks}, indice, catalogo)


@pytest.fixture
def modelo(monkeypatch, acervo):
    """A "pergunta" fica parecida com o trecho (princípio ativo, seção) de cada peso; a geração devolve
    `frases`. Devolve as chamadas: {"embed": [textos], "chat": [prompts]}."""
    chamadas = {"embed": [], "chat": []}

    def preparar(pesos: dict[tuple[str, str], float], *frases: tuple[str, int]):
        chunks = list(acervo.chunks.values())
        vetor = np.full(len(chunks), 0.01)
        for (ativo, secao), peso in pesos.items():
            vetor[next(i for i, c in enumerate(chunks) if c.principio_ativo == ativo and c.secao == secao)] = peso

        def embed(textos, s=None):
            chamadas["embed"] += textos
            return [vetor.tolist()]

        def chat(sistema, usuario, formato, s=None, modelo=None):
            chamadas["chat"].append(usuario)
            saida = {"status": "respondida", "afirmacoes": [{"texto": t, "trecho": n} for t, n in frases]}
            return ollama.LLMResponse(json.dumps(saida), 1, 1, 0, 0)

        monkeypatch.setattr(ollama, "embed", embed)
        monkeypatch.setattr(ollama, "chat", chat)
        return chamadas

    return preparar


def test_pergunta_seguinte_sem_remedio_continua_o_da_anterior(acervo, modelo):
    chamadas = modelo({(EXEMPLINA, "contraindicacoes"): 1.0}, ("O Exemplol é contraindicado durante a gravidez", 1))
    anterior = PreviousQuestion("Qual a dose máxima de Exemplol?", [EXEMPLINA], False)
    r = rag.responder("Pode usar na gravidez?", acervo, s=S, anterior=anterior)
    assert (r.status, r.principios_ativos, r.continuacao) == ("respondida", [EXEMPLINA], True)
    assert {f.chunk.principio_ativo for f in r.fontes} == {EXEMPLINA}  # só a bula do remédio herdado
    assert "PERGUNTA ANTERIOR: Qual a dose máxima de Exemplol?" in chamadas["chat"][0]
    assert chamadas["embed"][0] == "Qual a dose máxima de Exemplol? Pode usar na gravidez?"


def test_pedido_de_remedio_novo_nao_herda_a_anterior(acervo, modelo):
    chamadas = modelo({(EXEMPLINA, "reacoes_adversas"): 1.0}, ("Algo", 1))
    anterior = PreviousQuestion("Qual a dose máxima de Exemplol?", [EXEMPLINA], False)
    r = rag.responder("Qual o melhor remédio para tosse?", acervo, s=S, anterior=anterior)
    # Nenhuma indicação fala de tosse (o Exemplol só tem "tosse seca" como reação adversa).
    assert (r.status, r.continuacao, r.principios_ativos, chamadas["chat"]) == ("nao_encontrada", False, [], [])


def test_alergia_sem_dizer_o_remedio_pergunta_qual(acervo, monkeypatch):
    monkeypatch.setattr(ollama, "embed", lambda *a, **k: pytest.fail("não devia buscar"))
    r = rag.responder("Tenho alergia a um remédio, tem outro que eu possa tomar?", acervo, s=S)
    assert r.status == "precisa_esclarecimento"
    assert (r.esclarecimento.motivo, r.esclarecimento.pergunta, r.esclarecimento.opcoes) == (
        "remedio_da_alergia", graph.PERGUNTA_ALERGIA, [],
    )


def test_resposta_ao_esclarecimento_traz_contraindicacao_e_alternativas(acervo, modelo):
    chamadas = modelo(
        {(EXEMPLINA, "contraindicacoes"): 1.0, (OUTRINICO, "indicacoes"): 0.5},
        ("O Exemplol é contraindicado para pacientes com hipersensibilidade à exemplina", 1),
        # Fala do Exemplol citando a bula do Outrol: pode, é o remédio da alergia.
        ("Para quem tem alergia ao Exemplol, o Outrol é indicado para o tratamento da hipertensão arterial", 2),
    )
    anterior = PreviousQuestion("Tenho alergia a um remédio, tem outro que eu possa tomar?", [], True)
    r = rag.responder("Exemplol", acervo, s=S, anterior=anterior)
    assert (r.status, r.alergia_a, r.continuacao, r.extrativa) == ("respondida", EXEMPLINA, True, True)
    assert [(f.chunk.principio_ativo, f.chunk.secao) for f in r.fontes] == [
        (EXEMPLINA, "contraindicacoes"), (OUTRINICO, "indicacoes"),
    ]
    # A resposta são as frases das bulas, sem o LLM: a de hipersensibilidade e a de indicação.
    assert chamadas["chat"] == []
    assert r.resposta == (
        "Exemplol 50 mg: Exemplol é contraindicado para pacientes com hipersensibilidade à exemplina ou a "
        "qualquer componente da fórmula [1]. "
        "Outrol 10 mg: Exemplol é indicado para o tratamento da hipertensão arterial em adultos [2]."
    )
    # O uso do Exemplol, para achar alternativas, é a indicação da bula dele.
    assert chamadas["embed"][1].startswith("Exemplol é indicado para o tratamento da hipertensão")


def test_alergia_a_remedio_sem_bula_usa_a_classe_da_anvisa(acervo, modelo):
    chamadas = modelo(
        {(OUTRINICO, "indicacoes"): 1.0},
        ("Para quem tem alergia a dipirona, o Outrol é indicado para o tratamento da hipertensão arterial", 1),
    )
    r = rag.responder("Sou alérgica à dipirona, tem outro?", acervo, s=S)
    assert (r.status, r.alergia_a, r.medicamentos_nao_indexados) == ("respondida", "dipirona", ["dipirona"])
    # Nenhum remédio do acervo é da classe da dipirona: vale a semelhança com a classe, só os próximos
    # do melhor (o Exemplol, a 0,01 contra 1,0, fica fora). Sem bula, sem contraindicação.
    assert [(f.chunk.principio_ativo, f.chunk.secao) for f in r.fontes] == [(OUTRINICO, "indicacoes")]
    assert chamadas["embed"][1] == "Medicamento da classe analgesicos nao narcoticos"


def test_alternativas_sao_os_remedios_da_mesma_classe(acervo, modelo):
    da_classe = {"associacao": False, "n_registros": 5, "nome_produto": "X"}
    acervo.catalogo = Catalog.montar(list(acervo.bulas.values()), ["dipirona"], [
        DIPIRONA_ANVISA,
        da_classe | {"classe_terapeutica": "ANALGESICOS NAO NARCOTICOS", "principios_ativos": [EXEMPLINA]},
        da_classe | {"classe_terapeutica": "ANTI-HIPERTENSIVOS", "principios_ativos": [OUTRINICO]},
    ])
    modelo(
        {(OUTRINICO, "indicacoes"): 1.0},  # o Outrol é o mais parecido, mas é de outra classe
        ("Para quem tem alergia a dipirona, o Exemplol é indicado para o tratamento da hipertensão arterial", 1),
    )
    r = rag.responder("Sou alérgica à dipirona, tem outro?", acervo, s=S)
    assert [(f.chunk.principio_ativo, f.chunk.secao) for f in r.fontes] == [(EXEMPLINA, "indicacoes")]


def test_e_com_outro_remedio_junta_com_o_da_anterior(acervo, modelo):
    modelo({(EXEMPLINA, "interacoes"): 1.0}, ("O uso com anti-inflamatórios pode piorar a função renal", 1))
    anterior = PreviousQuestion("Exemplol pode com ibuprofeno?", [EXEMPLINA], False)
    r = rag.responder("E com Outrol?", acervo, s=S, anterior=anterior)
    # Os detectados vêm normalizados (sem acento), como o filtro da busca compara.
    assert (sorted(r.principios_ativos), r.continuacao) == (["acido outrinico", "maleato de exemplina"], True)


def test_alergia_e_outro_remedio_na_pergunta_e_sobre_o_outro(acervo, modelo):
    modelo({(EXEMPLINA, "contraindicacoes"): 1.0}, ("O Exemplol é contraindicado para pacientes com hipersensibilidade à exemplina", 1))
    r = rag.responder("Sou alérgico a dipirona, posso tomar Exemplol?", acervo, s=S)
    assert (r.alergia_a, r.principios_ativos, r.status) == (None, [EXEMPLINA], "respondida")


@pytest.mark.parametrize("pergunta, depois", [
    ("Tenho alergia a dipirona, posso tomar outro?", "dipirona"),
    ("Sou alérgico a dipirona, posso tomar Exemplol?", "dipirona"),
    ("Tenho alergia a algum remédio, tem outro?", ""),  # alergia sem dizer a qual remédio
    ("Sou alérgica à amoxicilina", "amoxicilina"),
    ("Loratadina serve para alergia?", None),  # alergia como indicação, não remédio a evitar
    ("Tenho alergia, o que posso tomar?", None),
])
def test_texto_apos_alergia(pergunta, depois):
    assert texto_apos_alergia(pergunta) == depois


def test_pede_remedio_e_eliptica():
    assert pede_remedio("Qual o melhor remédio para tosse?") and pede_remedio("O que posso tomar pra febre?")
    assert not pede_remedio("Pode usar na gravidez?")
    assert eliptica("E para criança?") and eliptica("Mas e na gravidez?") and not eliptica("Qual a dose?")


def test_responde_ao_assunto():
    # A frase errada da dipirona: está na bula, mas não fala de dose.
    assert not responde_ao_assunto(["Não existe experiência em insuficiência renal ou hepática"], ["dose", "maxim"])
    # A bula dá a dose sem a palavra "dose": número com unidade vale.
    assert responde_ao_assunto(["½ a 1 comprimido até 4 vezes ao dia"], ["dose", "maxim"])
    # "Tem interação?" respondida com uma frase de Interações, sem a palavra "interação".
    assert responde_ao_assunto(["Pode reduzir o efeito anti-hipertensivo"], ["inter"], {"interacoes"}, ["interacoes"])
    # Posologia não vale por seção: a frase errada da dipirona era de lá.
    assert not responde_ao_assunto(["Não usar por via retal"], ["dose"], {"posologia"}, ["posologia"])
    assert responde_ao_assunto(["Pode piorar a função renal"], ["rins", "rim", "renal"])


def test_resposta_fora_do_assunto_vai_para_outra_rodada(acervo, modelo, monkeypatch):
    chamadas = {"n": 0}
    respostas = [
        ("A dose inicial usual é de 50 mg uma vez ao dia", 1),  # está no trecho, mas não fala de alimentos
        ("Os comprimidos podem ser tomados com ou sem alimentos", 1),
    ]
    registro = modelo({(EXEMPLINA, "posologia"): 1.0}, respostas[0])

    def chat(sistema, usuario, formato, s=None, modelo=None):
        registro["chat"].append(usuario)
        texto, n = respostas[min(chamadas["n"], 1)]
        chamadas["n"] += 1
        return ollama.LLMResponse(json.dumps({"status": "respondida", "afirmacoes": [{"texto": texto, "trecho": n}]}), 1, 1, 0, 0)

    monkeypatch.setattr(ollama, "chat", chat)
    r = rag.responder("O Exemplol pode ser tomado junto das refeições, com alimentos?", acervo, s=S)
    assert (r.status, r.rodadas) == ("respondida", 2)
    assert "não fala do que foi perguntado" in r.tentativas[0].falha
    assert graph.DICA_ASSUNTO in registro["chat"][1]


def test_dose_vem_da_frase_da_posologia_sem_llm(acervo, modelo):
    chamadas = modelo({(EXEMPLINA, "posologia"): 1.0}, ("Algo", 1))
    r = rag.responder("Qual a dose máxima de Exemplol?", acervo, s=S)
    assert (r.status, r.extrativa, chamadas["chat"]) == ("respondida", True, [])
    # Uma frase por bula, a que mais fala do pedido ("máxima"), com o nome do medicamento na frente.
    assert r.resposta == "Exemplol 50 mg: A dose máxima é de 100 mg uma vez ao dia [1]."


def test_pergunta_eliptica_sem_anterior_pergunta_o_remedio(acervo, monkeypatch):
    monkeypatch.setattr(ollama, "embed", lambda *a, **k: pytest.fail("não devia buscar"))
    r = rag.responder("E para criança?", acervo, s=S)
    assert (r.status, r.esclarecimento.pergunta, r.esclarecimento.opcoes) == ("precisa_esclarecimento", graph.PERGUNTA_REMEDIO, [])
