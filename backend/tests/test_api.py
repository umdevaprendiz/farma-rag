"""A API de ponta a ponta: HTTP -> agente -> PostgreSQL, com o Ollama trocado por um falso.

Usa a bula inventada "Exemplol" gravada num schema temporário (fixture `engine`, conftest.py) e
o catálogo de nomes real de data/. Pulado sem o banco, menos o teste de banco fora do ar.
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.config import Settings
from app.db.models import EMBED_DIM
from app.llm import ollama
from app.main import criar_app
from app.schemas.api import ErrorCode
from app.services import graph
from ingestion.generate_chunks import gerar_chunks
from ingestion.load_database import gravar
from ingestion.process_leaflet import montar_bula

FIXTURES = Path(__file__).parent / "fixtures"
S = Settings(_env_file=None, top_k=2)
RECUSA = {"status": "nao_encontrada", "afirmacoes": []}


@pytest.fixture(scope="module")
def dados():
    bula = montar_bula(
        FIXTURES / "sample_leaflet.txt", "Exemplol 50 mg", "maleato de exemplina", "Laboratório Teste",
        date(2025, 3, 15), ["Exemplol"],
    )
    chunks = gerar_chunks(bula, 300, 50)
    # Um eixo por trecho: a "pergunta" aponta para o trecho que o teste quiser recuperar.
    return [bula], chunks, np.eye(len(chunks), EMBED_DIM, dtype="<f4")


@pytest.fixture(scope="module")
def api(engine, dados):
    gravar(engine, *dados)
    with TestClient(criar_app(S, engine)) as cliente:
        yield cliente


@pytest.fixture
def cliente(api, engine):
    with engine.begin() as conexao:
        conexao.execute(text("TRUNCATE consultas CASCADE"))
    return api


@pytest.fixture
def modelo(monkeypatch, dados):
    """Troca o Ollama: a busca devolve primeiro o trecho de `secao` e a geração devolve `saida` em
    toda rodada. Devolve a lista de prompts de geração."""
    _, chunks, _ = dados
    chamadas = []

    def preparar(secao: str, saida: dict):
        vetor = np.zeros(EMBED_DIM)
        vetor[: len(chunks)] = 0.01
        vetor[next(i for i, c in enumerate(chunks) if c.secao == secao)] = 1.0
        monkeypatch.setattr(ollama, "embed", lambda textos, s=None: [vetor.tolist()])

        def chat(sistema, usuario, formato, s=None, modelo=None):
            assert sistema != graph.SISTEMA_VERIFICACAO, "nenhum teste daqui chega à rodada 3 aprovado"
            chamadas.append(usuario)
            return ollama.LLMResponse(json.dumps(saida), 400, 40, 8.0, 4.0)

        monkeypatch.setattr(ollama, "chat", chat)
        return chamadas

    return preparar


def _resposta(*frases: tuple[str, int]) -> dict:
    return {"status": "respondida", "afirmacoes": [{"texto": t, "trecho": n} for t, n in frases]}


def _erro(r) -> tuple[int, str]:
    return r.status_code, r.json()["erro"]["codigo"]


def test_saude(cliente, monkeypatch):
    monkeypatch.setattr(ollama, "modelos", lambda s=None: ["qwen2.5:1.5b", "bge-m3:latest"])
    corpo = cliente.get("/api/v1/saude").json()
    assert corpo["status"] == "ok" and corpo["banco"] and corpo["ollama"]
    assert corpo["bulas"] == 1 and corpo["trechos"] > 1 and corpo["problemas"] == []

    monkeypatch.setattr(ollama, "modelos", lambda s=None: ["bge-m3:latest"])
    corpo = cliente.get("/api/v1/saude").json()
    assert (corpo["status"], corpo["modelos_faltando"]) == ("degradado", ["qwen2.5:1.5b"])

    def fora(s=None):
        raise ollama.OllamaError(ErrorCode.OLLAMA_INDISPONIVEL, "Ollama fora do ar em http://localhost:11434")

    monkeypatch.setattr(ollama, "modelos", fora)
    corpo = cliente.get("/api/v1/saude").json()
    assert (corpo["status"], corpo["banco"], corpo["ollama"]) == ("degradado", True, False)
    assert corpo["problemas"] == ["Ollama fora do ar em http://localhost:11434"]


def test_medicamentos(cliente):
    assert cliente.get("/api/v1/medicamentos").json() == [
        {
            "principio_ativo": "maleato de exemplina",
            "bulas": [{
                "id": "exemplol-50-mg-laboratorio-teste", "medicamento": "Exemplol 50 mg",
                "fabricante": "Laboratório Teste", "forma_farmaceutica": None, "data_versao": "2025-03-15",
            }],
        }
    ]


def test_resposta_com_fontes_fica_no_historico_igual(cliente, modelo, engine):
    modelo("interacoes", _resposta(("Pode reduzir o efeito anti-hipertensivo", 1)))
    # Sem segundo remédio na pergunta: vai pelo LLM (com outro remédio citado, a resposta é a frase
    # da bula: ver test_interacao_vem_da_frase_da_bula_sem_llm).
    r = cliente.post("/api/v1/perguntas", json={"pergunta": "Exemplol tem interação medicamentosa?"})
    assert r.status_code == 200
    corpo = r.json()
    assert (corpo["status"], corpo["rodadas"], corpo["modelo"]) == ("respondida", 1, "qwen2.5:1.5b")
    assert corpo["resposta"] == "Pode reduzir o efeito anti-hipertensivo [1]."
    primeira = corpo["fontes"][0]
    assert (primeira["posicao"], primeira["citado"], primeira["rotulo_secao"]) == (1, True, "Interações medicamentosas")
    assert (primeira["fabricante"], primeira["numero_registro"], primeira["data_versao"]) == ("Laboratório Teste", None, "2025-03-15")
    assert [f["citado"] for f in corpo["fontes"]] == [True, False]  # o não citado também foi consultado

    detalhe = cliente.get(f"/api/v1/historico/{corpo['consulta_id']}").json()
    assert detalhe["fontes"] == corpo["fontes"]
    assert (detalhe["status"], detalhe["resposta"], detalhe["rodadas"]) == ("respondida", corpo["resposta"], 1)
    assert detalhe["consulta_origem_id"] is None and detalhe["codigo_erro"] is None
    with engine.connect() as conexao:
        tentativas = conexao.execute(text("SELECT tentativas FROM consultas")).scalar_one()
    assert [t["resultado"] for t in tentativas] == ["aprovada"] and tentativas[0]["saida_bruta"]


def test_pergunta_sem_remedio_responde_e_o_reenvio_fica_ligado(cliente, modelo):
    modelo("posologia", _resposta(("A dose máxima do Exemplol é de 100 mg uma vez ao dia", 1)))
    primeira = cliente.post("/api/v1/perguntas", json={"pergunta": "Qual a dose máxima por dia?"}).json()
    assert (primeira["status"], primeira["esclarecimento"]) == ("respondida", None)  # busca em todas as bulas
    assert primeira["fontes"][0]["rotulo_secao"] == "Posologia e modo de usar"

    reenvio = cliente.post("/api/v1/perguntas", json={
        "pergunta": "Qual a dose máxima por dia?", "principios_ativos": ["maleato de exemplina"],
        "consulta_origem_id": primeira["consulta_id"],
    }).json()
    assert reenvio["status"] == "respondida"
    detalhe = cliente.get(f"/api/v1/historico/{reenvio['consulta_id']}").json()
    assert detalhe["consulta_origem_id"] == primeira["consulta_id"]
    assert detalhe["principios_ativos_filtro"] == ["maleato de exemplina"]


def test_rodadas_esgotadas_dao_502_explicado_e_ficam_no_historico(cliente, modelo):
    chamadas = modelo("posologia", _resposta(("A dose máxima é de 200 mg uma vez ao dia", 1)))
    r = cliente.post("/api/v1/perguntas", json={"pergunta": "Como tomar o Exemplol?"})
    assert _erro(r) == (502, "RESPOSTA_NAO_VERIFICADA") and len(chamadas) == 3
    mensagem = r.json()["erro"]["mensagem"]
    assert "3 rodada(s)" in mensagem and "rodada 1: " in mensagem

    item = cliente.get("/api/v1/historico").json()["itens"][0]
    assert item["status"] == "erro"
    detalhe = cliente.get(f"/api/v1/historico/{item['id']}").json()
    assert (detalhe["codigo_erro"], detalhe["mensagem_erro"], detalhe["rodadas"]) == ("RESPOSTA_NAO_VERIFICADA", mensagem, 3)
    assert detalhe["resposta"] is None and not any(f["citado"] for f in detalhe["fontes"])


def test_ollama_fora_do_ar_da_503_e_fica_no_historico(cliente, monkeypatch):
    def fora(textos, s=None):
        raise ollama.OllamaError(ErrorCode.OLLAMA_INDISPONIVEL, "Ollama fora do ar")

    monkeypatch.setattr(ollama, "embed", fora)
    r = cliente.post("/api/v1/perguntas", json={"pergunta": "Exemplol pode com ibuprofeno?"})
    assert _erro(r) == (503, "OLLAMA_INDISPONIVEL")
    assert cliente.get("/api/v1/historico").json()["itens"][0]["status"] == "erro"


def test_remedio_sem_bula_que_nenhuma_bula_cita(cliente, modelo):
    modelo("interacoes", RECUSA)
    corpo = cliente.post("/api/v1/perguntas", json={"pergunta": "Captopril pode com álcool?"}).json()
    assert (corpo["status"], corpo["motivo"], corpo["medicamentos_nao_indexados"], corpo["fontes"]) == (
        "nao_encontrada", "medicamento_nao_indexado", ["captopril"], [],
    )


def test_filtro_desconhecido_da_404(cliente):
    r = cliente.post("/api/v1/perguntas", json={"pergunta": "Qual a dose máxima?", "principios_ativos": ["remediox"]})
    assert _erro(r) == (404, "MEDICAMENTO_NAO_ENCONTRADO")


def test_nome_parecido_pergunta_qual_e_e_a_opcao_traz_a_classe_da_anvisa(cliente, modelo):
    # Caso real (consulta b14b2d1a, 18/10; seção 28), com o catálogo real de data/.
    pergunta = "Tomei rocutan para dor no ouvido, fiz certo?"
    corpo = cliente.post("/api/v1/perguntas", json={"pergunta": pergunta}).json()
    assert (corpo["status"], corpo["esclarecimento"]["opcoes"]) == (
        "precisa_esclarecimento", [{"rotulo": "Roacutan (isotretinoína)", "principios_ativos": ["isotretinoína"]}],
    )

    # O botão: a mesma pergunta, com o princípio ativo da opção (sem bula: antes era 404).
    modelo("indicacoes", RECUSA)
    opcao = cliente.post("/api/v1/perguntas", json={
        "pergunta": pergunta, "principios_ativos": ["isotretinoína"], "consulta_origem_id": corpo["consulta_id"],
    }).json()
    assert (opcao["status"], opcao["motivo"], opcao["classes_terapeuticas"]) == (
        "nao_encontrada", "medicamento_nao_indexado", {"isotretinoina": "PRODUTOS ANTI-ACNE"},
    )


@pytest.mark.parametrize("corpo, campo", [
    ({"pergunta": "oi"}, "pergunta"),
    ({"pergunta": "Pode com álcool?", "principios_ativos": ["a", "b", "c", "d"]}, "principios_ativos"),
    ({"pergunta": "Pode com álcool?", "consulta_origem_id": "nao-e-uuid"}, "consulta_origem_id"),
])
def test_entrada_invalida_da_422_no_formato_de_erro(cliente, corpo, campo):
    r = cliente.post("/api/v1/perguntas", json=corpo)
    assert _erro(r) == (422, "ENTRADA_INVALIDA") and r.json()["erro"]["mensagem"].startswith(f"{campo}: ")


def test_historico_paginado_do_mais_recente(cliente, modelo):
    modelo("interacoes", RECUSA)
    for i in range(3):
        cliente.post("/api/v1/perguntas", json={"pergunta": f"Captopril pergunta {i}"})
    pagina1 = cliente.get("/api/v1/historico", params={"limite": 2}).json()
    assert (pagina1["total"], pagina1["pagina"], pagina1["limite"]) == (3, 1, 2)
    assert [c["pergunta"] for c in pagina1["itens"]] == ["Captopril pergunta 2", "Captopril pergunta 1"]
    pagina2 = cliente.get("/api/v1/historico", params={"limite": 2, "pagina": 2}).json()
    assert [c["pergunta"] for c in pagina2["itens"]] == ["Captopril pergunta 0"]
    assert _erro(cliente.get("/api/v1/historico", params={"limite": 0})) == (422, "ENTRADA_INVALIDA")


def test_consulta_que_nao_existe_da_404(cliente):
    r = cliente.get("/api/v1/historico/00000000-0000-0000-0000-000000000000")
    assert _erro(r) == (404, "CONSULTA_NAO_ENCONTRADA")


def test_raiz_leva_para_a_documentacao(cliente):
    r = cliente.get("/", follow_redirects=False)
    assert (r.status_code, r.headers["location"]) == (307, "/docs")


def test_cors_libera_o_frontend(cliente):
    r = cliente.options("/api/v1/perguntas", headers={
        "Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST",
    })
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_banco_fora_do_ar_da_503_e_a_api_sobe_mesmo_assim(monkeypatch):
    monkeypatch.setattr(ollama, "modelos", lambda s=None: ["qwen2.5:1.5b", "bge-m3:latest"])
    s = Settings(_env_file=None, db_port=1, db_timeout_s=1)  # porta onde nada escuta
    with TestClient(criar_app(s)) as cliente:
        assert _erro(cliente.get("/api/v1/medicamentos")) == (503, "BANCO_INDISPONIVEL")
        corpo = cliente.get("/api/v1/saude").json()
        assert (corpo["status"], corpo["banco"]) == ("degradado", False)
        assert "127.0.0.1:1" in corpo["problemas"][0]


def test_pergunta_seguinte_herda_o_remedio_da_anterior(cliente, modelo):
    modelo("posologia", _resposta(("A dose máxima do Exemplol é de 100 mg uma vez ao dia", 1)))
    primeira = cliente.post("/api/v1/perguntas", json={"pergunta": "Como tomar o Exemplol?"}).json()
    assert (primeira["principios_ativos"], primeira["continuacao"]) == (["maleato de exemplina"], False)

    seguinte = cliente.post("/api/v1/perguntas", json={
        "pergunta": "E qual a dose inicial?", "consulta_origem_id": primeira["consulta_id"],
    }).json()
    assert (seguinte["status"], seguinte["principios_ativos"], seguinte["continuacao"]) == (
        "respondida", ["maleato de exemplina"], True,
    )


def test_alergia_pergunta_o_remedio_e_a_resposta_vem_no_chat(cliente, modelo):
    modelo("contraindicacoes", _resposta(("O Exemplol é contraindicado para pacientes com hipersensibilidade à exemplina", 1)))
    pergunta = cliente.post("/api/v1/perguntas", json={"pergunta": "Tenho alergia a um remédio, tem outro?"}).json()
    assert pergunta["status"] == "precisa_esclarecimento"
    assert (pergunta["esclarecimento"]["motivo"], pergunta["esclarecimento"]["opcoes"]) == ("remedio_da_alergia", [])

    resposta = cliente.post("/api/v1/perguntas", json={
        "pergunta": "Exemplol", "consulta_origem_id": pergunta["consulta_id"],
    }).json()
    assert (resposta["status"], resposta["alergia_a"], resposta["continuacao"]) == ("respondida", "maleato de exemplina", True)
    assert resposta["fontes"][0]["rotulo_secao"] == "Contraindicações"


def test_remedio_sem_bula_vem_com_a_classe_terapeutica_da_anvisa(cliente, modelo):
    modelo("interacoes", RECUSA)
    corpo = cliente.post("/api/v1/perguntas", json={"pergunta": "Dipirona pode para criança?"}).json()
    # O catálogo real (data/processed/catalogo_anvisa.jsonl) diz para que a dipirona serve.
    assert corpo["classes_terapeuticas"] == {"dipirona": "ANALGESICOS NAO NARCOTICOS"}


def test_interacao_vem_da_frase_da_bula_sem_llm(cliente, monkeypatch, modelo):
    modelo("interacoes", RECUSA)
    monkeypatch.setattr(ollama, "chat", lambda *a, **k: pytest.fail("a frase vem da bula, sem o LLM"))
    corpo = cliente.post("/api/v1/perguntas", json={"pergunta": "Exemplol pode ser usado com ibuprofeno?"}).json()
    assert (corpo["status"], corpo["extrativa"], corpo["modelo"]) == ("respondida", True, "frases da bula (sem LLM)")
    assert corpo["resposta"] == (
        "O uso com anti-inflamatórios não esteroidais, como o ibuprofeno, pode reduzir o efeito "
        "anti-hipertensivo da exemplina e piorar a função renal [1]."
    )


def test_erro_traz_o_id_da_consulta_para_a_conversa_continuar(cliente, monkeypatch):
    def fora(textos, s=None):
        raise ollama.OllamaError(ErrorCode.OLLAMA_INDISPONIVEL, "Ollama fora do ar")

    monkeypatch.setattr(ollama, "embed", fora)
    corpo = cliente.post("/api/v1/perguntas", json={"pergunta": "Exemplol pode com álcool?"}).json()
    assert corpo["erro"]["codigo"] == "OLLAMA_INDISPONIVEL"
    assert cliente.get(f"/api/v1/historico/{corpo['consulta_id']}").json()["status"] == "erro"
