import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.db.busca_vetorial import IndiceEmMemoria
from app.llm import ollama
from app.schemas.api import CodigoErro
from app.services import rag
from app.services.acervo import Acervo
from app.services.medicamentos import Catalogo
from ingestao.gerar_chunks import gerar_chunks
from ingestao.processar_bula import montar_bula

FIXTURES = Path(__file__).parent / "fixtures"
S = Settings(_env_file=None, top_k=2)


@pytest.fixture
def acervo():
    bula = montar_bula(
        FIXTURES / "bula_exemplo.txt", "Exemplol 50 mg", "maleato de exemplina", "Laboratório Teste",
        date(2025, 3, 15), ["Exemplol"],
    )
    chunks = gerar_chunks(bula, 300, 50)
    # Um eixo por trecho: a "pergunta" aponta para o trecho que o teste quiser recuperar.
    indice = IndiceEmMemoria([c.chunk_id for c in chunks], [c.principio_ativo for c in chunks], np.eye(len(chunks)))
    return Acervo({bula.id: bula}, {c.chunk_id: c for c in chunks}, indice, Catalogo.montar([bula], ["captopril"]))


@pytest.fixture
def modelo(monkeypatch, acervo):
    """Troca o Ollama: a busca devolve primeiro o trecho de `secao`; o chat devolve `saida`."""
    chamadas = []

    def preparar(secao: str, saida: dict | str):
        alvo = next(i for i, c in enumerate(acervo.chunks.values()) if c.secao == secao)
        vetor = np.full(len(acervo.chunks), 0.01)
        vetor[alvo] = 1.0
        monkeypatch.setattr(ollama, "embed", lambda textos, s=None: [vetor.tolist()])

        def chat(sistema, usuario, formato, s=None):
            chamadas.append(usuario)
            texto = saida if isinstance(saida, str) else json.dumps(saida)
            return ollama.RespostaLLM(texto, prompt_tokens=400, tokens_gerados=40, prompt_s=8.0, geracao_s=4.0)

        monkeypatch.setattr(ollama, "chat", chat)
        return chamadas

    return preparar


def test_resposta_com_citacao_por_frase(acervo, modelo):
    chamadas = modelo("interacoes", {"status": "respondida", "afirmacoes": [
        {"texto": "Pode reduzir o efeito anti-hipertensivo.", "trecho": 1},
        {"texto": "Pode piorar a função renal", "trecho": 1},
    ]})
    r = rag.responder("Exemplol pode com ibuprofeno?", acervo, s=S)
    assert r.status == "respondida"
    assert r.resposta == "Pode reduzir o efeito anti-hipertensivo [1]. Pode piorar a função renal [1]."
    assert r.principios_ativos == ["maleato de exemplina"]
    assert [(f.posicao, f.chunk.secao, f.citado) for f in r.fontes][0] == (1, "interacoes", True)
    assert len(r.fontes) == 2 and not r.fontes[1].citado
    assert r.fontes[0].data_versao == date(2025, 3, 15)
    assert "[1] Exemplol 50 mg | Interações medicamentosas | página 2" in chamadas[0]
    assert chamadas[0].endswith("PERGUNTA: Exemplol pode com ibuprofeno?")


def test_medicamento_sem_bula_nao_chama_o_modelo(acervo, monkeypatch):
    monkeypatch.setattr(ollama, "embed", lambda *a, **k: pytest.fail("não devia buscar"))
    r = rag.responder("Captopril pode com álcool?", acervo, s=S)
    assert (r.status, r.motivo, r.medicamentos_nao_indexados) == ("nao_encontrada", "medicamento_nao_indexado", ["captopril"])
    assert r.fontes == []


def test_mistura_segue_com_o_indexado_e_avisa_do_outro(acervo, modelo):
    modelo("interacoes", {"status": "respondida", "afirmacoes": [{"texto": "Sem dados.", "trecho": 2}]})
    r = rag.responder("Exemplol com captopril", acervo, s=S)
    assert r.status == "respondida" and r.medicamentos_nao_indexados == ["captopril"]


def test_recusa_do_modelo_vale_mesmo_com_texto_junto(acervo, modelo):
    modelo("posologia", {"status": "nao_encontrada", "afirmacoes": [{"texto": "Os trechos não informam.", "trecho": 1}]})
    r = rag.responder("Exemplol causa queda de cabelo?", acervo, s=S)
    assert (r.status, r.motivo, r.resposta) == ("nao_encontrada", "sem_contexto", None)
    assert len(r.fontes) == 2  # os trechos consultados continuam visíveis


@pytest.mark.parametrize("saida, codigo", [
    ("isto não é json", CodigoErro.SAIDA_INVALIDA),
    ({"status": "respondida"}, CodigoErro.SAIDA_INVALIDA),
    ({"status": "respondida", "afirmacoes": []}, CodigoErro.SAIDA_INVALIDA),
    ({"status": "respondida", "afirmacoes": [{"texto": "Algo.", "trecho": 9}]}, CodigoErro.CITACAO_INVALIDA),
])
def test_saida_invalida_vira_erro_com_a_saida_crua(acervo, modelo, saida, codigo):
    modelo("posologia", saida)
    r = rag.responder("Qual a dose de Exemplol?", acervo, s=S)
    assert (r.status, r.codigo_erro, r.resposta) == ("erro", codigo, None)
    assert r.saida_bruta


def test_ollama_fora_do_ar(acervo, monkeypatch):
    def cair(*a, **k):
        raise ollama.OllamaError(CodigoErro.OLLAMA_INDISPONIVEL, "fora do ar")

    monkeypatch.setattr(ollama, "embed", cair)
    r = rag.responder("Qual a dose de Exemplol?", acervo, s=S)
    assert (r.status, r.codigo_erro) == ("erro", CodigoErro.OLLAMA_INDISPONIVEL)


def test_filtro_aceita_nome_comercial_e_recusa_desconhecido(acervo, modelo):
    modelo("posologia", {"status": "respondida", "afirmacoes": [{"texto": "50 mg ao dia.", "trecho": 1}]})
    r = rag.responder("Qual a dose inicial?", acervo, principios_ativos=["Exemplol"], s=S)
    assert r.status == "respondida" and r.principios_ativos == ["maleato de exemplina"]
    r = rag.responder("Qual a dose inicial?", acervo, principios_ativos=["captopril"], s=S)
    assert (r.status, r.codigo_erro) == ("erro", CodigoErro.MEDICAMENTO_NAO_ENCONTRADO)


def test_abaixo_do_limiar_nao_chama_o_modelo(acervo, modelo):
    chamadas = modelo("posologia", {"status": "respondida", "afirmacoes": []})
    r = rag.responder("Qual a dose de Exemplol?", acervo, s=Settings(_env_file=None, limiar_similaridade=1.5))
    assert (r.status, r.motivo, r.fontes, chamadas) == ("nao_encontrada", "sem_contexto", [], [])


def test_formato_so_permite_citar_trechos_enviados():
    assert rag.formato(3)["properties"]["afirmacoes"]["items"]["properties"]["trecho"]["enum"] == [1, 2, 3]
