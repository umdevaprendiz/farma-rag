"""F4: o "para que serve" copiado da bula, explicado em palavras simples pelo modelo e conferido
(services/explanation.py, nó explicar do grafo; docs/DECISOES.md, seção 37). O Ollama é falso."""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.db.vector_search import InMemoryIndex
from app.llm import ollama
from app.schemas.api import ErrorCode
from app.services import explanation, rag
from app.services.corpus import Corpus
from app.services.medicines import Catalog
from app.services.presentation import ABERTURA_OPINIAO
from ingestion.generate_chunks import gerar_chunks
from ingestion.process_leaflet import montar_bula

FIXTURES = Path(__file__).parent / "fixtures"
S = Settings(_env_file=None, top_k=2)
INDICACAO = "Exemplol é indicado para o tratamento da hipertensão arterial em adultos"
SIMPLES = "O Exemplol serve para tratar a pressão alta em adultos"


@pytest.fixture
def acervo(monkeypatch):
    bula = montar_bula(
        FIXTURES / "sample_leaflet.txt", "Exemplol 50 mg", "maleato de exemplina", "Laboratório Teste",
        date(2025, 3, 15), ["Exemplol"],
    )
    chunks = gerar_chunks(bula, 300, 50)
    indice = InMemoryIndex([c.chunk_id for c in chunks], [c.principio_ativo for c in chunks], np.eye(len(chunks)))
    monkeypatch.setattr(ollama, "embed", lambda textos, s=None: [np.full(len(chunks), 0.1).tolist()])
    return Corpus({bula.id: bula}, {c.chunk_id: c for c in chunks}, indice, Catalog.montar([bula], ["captopril"]))


@pytest.fixture
def explicacao(monkeypatch):
    """O modelo falso devolve as frases dadas (texto, trecho), ou o texto cru se vier uma string. Devolve a
    lista de chamadas como (prompt, formato, settings)."""
    chamadas = []

    def preparar(*frases: tuple[str, int] | str):
        def chat(sistema, usuario, formato, s=None, modelo=None):
            assert sistema == explanation.SISTEMA, "no para que serve, só a explicação chama o LLM"
            chamadas.append((usuario, formato, s))
            if frases and isinstance(frases[0], str):
                return ollama.LLMResponse(frases[0], 120, 30, 2.0, 3.0)
            saida = {"afirmacoes": [{"texto": t, "trecho": n} for t, n in frases]}
            return ollama.LLMResponse(json.dumps(saida, ensure_ascii=False), 120, 30, 2.0, 3.0)

        monkeypatch.setattr(ollama, "chat", chat)
        return chamadas

    return preparar


def test_para_que_serve_ganha_a_explicacao_em_palavras_simples(acervo, explicacao):
    chamadas = explicacao((SIMPLES, 1))
    etapas = []
    r = rag.responder("Exemplol serve para pressão alta?", acervo, s=S, avisar=lambda etapa, rodada: etapas.append(etapa))
    assert (r.status, r.extrativa, r.rodadas) == ("respondida", True, 1)
    assert r.conclusao == "Sim. Pela bula de exemplina, é indicado para isso:"
    assert r.resposta == f"{INDICACAO} [1]."  # a cópia da bula continua igual
    assert r.explicacao == f"{SIMPLES} [1]."
    assert etapas == ["buscando", "conferindo", "explicando"]
    assert "explicacao" in r.tempos_s and r.tentativas[-1].explicacao_bruta

    prompt, formato, s = chamadas[0]
    assert f"[1] Exemplol 50 mg: {INDICACAO}" in prompt
    assert "PALAVRAS DO DIA A DIA: hipertensão arterial = pressão alta" in prompt
    assert "serve para" not in prompt  # sem a pergunta: o modelo explica as frases, não responde de novo
    assert formato["properties"]["afirmacoes"]["items"]["properties"]["trecho"]["enum"] == [1]
    assert s.num_predict == 160  # até 3 frases curtas

    # Em volta da resposta, sem LLM: o negrito no que foi perguntado, o termo explicado e o que perguntar depois.
    assert [(m.tipo, r.resposta[m.inicio:m.fim]) for m in r.marcas] == [
        ("foco", "hipertensão"), ("termo", "hipertensão arterial"),
    ]
    assert r.sugestoes == [
        "Qual a dose de exemplina?", "Quem não pode tomar exemplina?", "Quais os efeitos colaterais de exemplina?",
    ]
    assert r.abertura is None  # a conclusão abre a resposta


def test_frase_que_inventa_sai_e_as_outras_ficam(acervo, explicacao):
    explicacao(("O Exemplol também alivia dor de cabeça e enxaqueca", 1), ("Ele serve para tratar a pressão alta em adultos", 1))
    r = rag.responder("Exemplol serve para pressão alta?", acervo, s=S)
    assert r.explicacao == "Ele serve para tratar a pressão alta em adultos [1]."
    descartadas = r.tentativas[-1].explicacao_descartada
    assert len(descartadas) == 1 and descartadas[0].startswith("frase 1 [1]: termos fora do trecho")


def test_conclusao_negativa_a_explicacao_nao_fala_do_uso_perguntado(acervo, explicacao):
    # Passaria pela fração de termos (5 de 6 no trecho); "ansiedade" é o uso que a bula não indica.
    explicacao(("O Exemplol trata a pressão alta em adultos e a ansiedade", 1))
    r = rag.responder("Exemplol serve para ansiedade?", acervo, s=S)
    assert r.conclusao.startswith("Não. Ansiedade não está entre as indicações")
    assert (r.status, r.explicacao) == ("respondida", None)
    assert r.tentativas[-1].explicacao_descartada == ["frase 1 [1]: fala do uso que a bula não indica"]


def test_repetir_a_frase_da_bula_nao_e_explicacao(acervo, explicacao):
    explicacao((INDICACAO, 1))
    r = rag.responder("Para que serve o Exemplol?", acervo, s=S)
    assert r.explicacao is None
    assert r.tentativas[-1].explicacao_descartada == ["frase 1 [1]: repete a frase da bula"]


def test_saida_invalida_fica_sem_explicacao(acervo, explicacao):
    explicacao("isso não é JSON")
    r = rag.responder("Para que serve o Exemplol?", acervo, s=S)
    assert (r.status, r.explicacao) == ("respondida", None)
    assert r.tentativas[-1].explicacao_descartada == ["o modelo não devolveu explicação"]


def test_ollama_fora_nunca_vira_erro(acervo, monkeypatch):
    def cair(*a, **k):
        raise ollama.OllamaError(ErrorCode.TEMPO_ESGOTADO, "Ollama não respondeu em 120 s")

    monkeypatch.setattr(ollama, "chat", cair)
    r = rag.responder("Para que serve o Exemplol?", acervo, s=S)
    assert (r.status, r.resposta, r.explicacao) == ("respondida", f"{INDICACAO} [1].", None)
    assert r.tentativas[-1].explicacao_descartada == ["sem explicação: Ollama não respondeu em 120 s"]


def test_desligada_nao_chama_o_llm(acervo, monkeypatch):
    monkeypatch.setattr(ollama, "chat", lambda *a, **k: pytest.fail("EXPLICAR_INDICACAO=false"))
    r = rag.responder("Para que serve o Exemplol?", acervo, s=Settings(_env_file=None, top_k=2, explicar_indicacao=False))
    assert (r.status, r.explicacao) == ("respondida", None)


def test_pergunta_de_opiniao_e_de_para_que_serve(acervo, explicacao):
    # Como a consulta 12f61084 ("oque acha do ibuprofeno pra dor de cabeça?", 98 s pelo LLM em 2 rodadas).
    explicacao((SIMPLES, 1))
    r = rag.responder("o que acha do Exemplol para pressão alta?", acervo, s=S)
    assert (r.status, r.extrativa) == ("respondida", True)
    assert r.abertura == ABERTURA_OPINIAO
    assert r.conclusao == "Sim. Pela bula de exemplina, é indicado para isso:"
    assert r.explicacao == f"{SIMPLES} [1]."
