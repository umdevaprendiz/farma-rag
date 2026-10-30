"""A API de ponta a ponta: HTTP -> agente -> PostgreSQL, com o Ollama trocado por um falso.

Usa a bula inventada "Exemplol" gravada num schema temporário (fixture `engine`, conftest.py) e
o catálogo de nomes real de data/. Pulado sem o banco, menos o teste de banco fora do ar.
"""

import json
from datetime import date
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.config import Settings
from app.db.models import EMBED_DIM
from app.llm import ollama
from app.main import criar_app
from app.schemas.api import AVISO_SAUDE, ErrorCode
from app.services import explanation, graph
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
    toda rodada. Devolve a lista de prompts de geração.

    Com listas: a busca devolve os trechos dessas seções nessa ordem, e cada chamada de geração
    recebe a saída seguinte (a última se repete)."""
    _, chunks, _ = dados
    chamadas = []

    def preparar(secao: str | list[str], saida: dict | list[dict]):
        vetor = np.zeros(EMBED_DIM)
        vetor[: len(chunks)] = 0.01
        for n, nome in enumerate([secao] if isinstance(secao, str) else secao):
            vetor[next(i for i, c in enumerate(chunks) if c.secao == nome)] = 1.0 - 0.1 * n
        saidas = [saida] if isinstance(saida, dict) else saida
        monkeypatch.setattr(ollama, "embed", lambda textos, s=None: [vetor.tolist()])

        def chat(sistema, usuario, formato, s=None, modelo=None):
            assert sistema != graph.SISTEMA_VERIFICACAO, "nenhum teste daqui chega à rodada 3 aprovado"
            if sistema == explanation.SISTEMA:
                # F4: a explicação opcional do "para que serve" também chama o LLM. Aqui ela é descartada e
                # não gasta uma das saídas combinadas (senão as rodadas seguintes recebem a saída errada).
                return ollama.LLMResponse("{}", 10, 1, 1.0, 1.0)
            chamadas.append(usuario)
            return ollama.LLMResponse(json.dumps(saidas[min(len(chamadas), len(saidas)) - 1]), 400, 40, 8.0, 4.0)

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

    # Estourou o tempo da sondagem = o Ollama está no ar, ocupado gerando uma resposta (CPU): não é problema.
    def ocupado(s=None):
        raise ollama.OllamaError(ErrorCode.TEMPO_ESGOTADO, "Ollama não respondeu em 3 s")

    monkeypatch.setattr(ollama, "modelos", ocupado)
    corpo = cliente.get("/api/v1/saude").json()
    assert (corpo["status"], corpo["ollama"], corpo["problemas"]) == ("ok", True, [])


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
    # F2: o aviso vem da API, fora do texto da resposta, e a consulta do histórico o traz igual.
    assert corpo["aviso"] == AVISO_SAUDE and AVISO_SAUDE not in corpo["resposta"]
    primeira = corpo["fontes"][0]
    assert (primeira["posicao"], primeira["citado"], primeira["rotulo_secao"]) == (1, True, "Interações medicamentosas")
    assert (primeira["fabricante"], primeira["numero_registro"], primeira["data_versao"]) == ("Laboratório Teste", None, "2025-03-15")
    assert [f["citado"] for f in corpo["fontes"]] == [True, False]  # o não citado também foi consultado

    detalhe = cliente.get(f"/api/v1/historico/{corpo['consulta_id']}").json()
    assert detalhe["fontes"] == corpo["fontes"]
    assert (detalhe["status"], detalhe["resposta"], detalhe["rodadas"]) == ("respondida", corpo["resposta"], 1)
    assert detalhe["consulta_origem_id"] is None and detalhe["codigo_erro"] is None
    assert detalhe["aviso"] == AVISO_SAUDE
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
    assert corpo["aviso"] is None  # F2: sem resposta da bula, sem aviso


def test_filtro_desconhecido_da_404(cliente):
    r = cliente.post("/api/v1/perguntas", json={"pergunta": "Qual a dose máxima?", "principios_ativos": ["remediox"]})
    assert _erro(r) == (404, "MEDICAMENTO_NAO_ENCONTRADO")


def test_bula_desconhecida_da_404_antes_do_agente(cliente):
    # DECISOES §41: o botão da pergunta de apresentação manda bula_id; uma que não está no acervo é 404,
    # também na rota com progresso (antes do fluxo, como erro HTTP comum).
    corpo = {"pergunta": "Como tomar o Exemplol?", "bula_id": "bula-que-nao-existe"}
    assert _erro(cliente.post("/api/v1/perguntas", json=corpo)) == (404, "BULA_NAO_ENCONTRADA")
    assert _erro(cliente.post("/api/v1/perguntas/com-progresso", json=corpo)) == (404, "BULA_NAO_ENCONTRADA")


def test_bula_escolhida_fica_no_historico(cliente, modelo):
    modelo("interacoes", _resposta(("Pode reduzir o efeito anti-hipertensivo", 1)))
    bula = "exemplol-50-mg-laboratorio-teste"
    r = cliente.post("/api/v1/perguntas", json={"pergunta": "Exemplol tem interação medicamentosa?", "bula_id": bula})
    assert r.status_code == 200 and r.json()["principios_ativos"] == ["maleato de exemplina"]
    detalhe = cliente.get(f"/api/v1/historico/{r.json()['consulta_id']}").json()
    assert detalhe["bula_id_filtro"] == bula


def test_nome_parecido_pergunta_qual_e_e_a_opcao_traz_a_classe_da_anvisa(cliente, modelo):
    # Caso real (consulta b14b2d1a, 18/10; seção 28), com o catálogo real de data/.
    pergunta = "Tomei rocutan para dor no ouvido, fiz certo?"
    corpo = cliente.post("/api/v1/perguntas", json={"pergunta": pergunta}).json()
    assert (corpo["status"], corpo["esclarecimento"]["opcoes"]) == (
        "precisa_esclarecimento",
        [{"rotulo": "Roacutan (isotretinoína)", "principios_ativos": ["isotretinoína"], "bula_id": None}],
    )

    # O botão: a mesma pergunta, com o princípio ativo da opção (sem bula: antes era 404).
    modelo("indicacoes", RECUSA)
    opcao = cliente.post("/api/v1/perguntas", json={
        "pergunta": pergunta, "principios_ativos": ["isotretinoína"], "consulta_origem_id": corpo["consulta_id"],
    }).json()
    assert (opcao["status"], opcao["motivo"], opcao["classes_terapeuticas"]) == (
        "nao_encontrada", "medicamento_nao_indexado", {"isotretinoina": "PRODUTOS ANTI-ACNE"},
    )
    # Pergunta de para que serve, sem a bula no acervo: a classe, sem concluir (seção 29).
    assert opcao["conclusao"] == (
        "A bula de isotretinoina não está no acervo, então não dá para dizer se é indicado para dor no ouvido. "
        "Pelo registro na ANVISA, é da classe produtos anti-acne."
    )


def test_pergunta_de_uso_traz_a_conclusao_e_o_historico_guarda(cliente, monkeypatch, modelo):
    modelo("indicacoes", RECUSA)
    # F4: a resposta é a cópia das indicações (sem LLM), mas depois o LLM é chamado UMA vez para a explicação
    # em palavras simples (opcional). Aqui ele devolve lixo: a explicação é descartada e a resposta sai igual.
    # Não usar pytest.fail aqui: ele roda na thread da API, mata o portal do TestClient e derruba o módulo todo.
    chamadas = []

    def llm_com_defeito(sistema, usuario, *a, **k):
        chamadas.append(sistema)
        return ollama.LLMResponse("isto não é json", 10, 1, 1.0, 1.0)

    monkeypatch.setattr(ollama, "chat", llm_com_defeito)
    corpo = cliente.post("/api/v1/perguntas", json={"pergunta": "Exemplol serve para ansiedade?"}).json()
    assert (corpo["status"], corpo["extrativa"]) == ("respondida", True)
    assert len(chamadas) == 1 and corpo["explicacao"] is None  # só a explicação opcional chamou o LLM
    assert corpo["conclusao"].startswith("Não. Ansiedade não está entre as indicações da bula de exemplina")
    assert corpo["fontes"][0]["rotulo_secao"] == "Indicações"
    detalhe = cliente.get(f"/api/v1/historico/{corpo['consulta_id']}").json()
    assert detalhe["conclusao"] == corpo["conclusao"]


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
    chamadas = []  # sem pytest.fail aqui: ele roda na thread da API e derruba o módulo (ver o teste do "para que serve")
    monkeypatch.setattr(ollama, "chat", lambda *a, **k: chamadas.append(a) or ollama.LLMResponse("{}", 0, 0, 0.0, 0.0))
    corpo = cliente.post("/api/v1/perguntas", json={"pergunta": "Exemplol pode ser usado com ibuprofeno?"}).json()
    assert (corpo["status"], corpo["extrativa"], corpo["modelo"]) == ("respondida", True, "frases da bula (sem LLM)")
    assert corpo["aviso"] == AVISO_SAUDE  # também na resposta copiada da bula
    assert corpo["resposta"] == (
        "O uso com anti-inflamatórios não esteroidais, como o ibuprofeno, pode reduzir o efeito "
        "anti-hipertensivo da exemplina e piorar a função renal [1]."
    )
    assert chamadas == []  # a interação é cópia da bula: o modelo não foi chamado


def test_erro_traz_o_id_da_consulta_para_a_conversa_continuar(cliente, monkeypatch):
    def fora(textos, s=None):
        raise ollama.OllamaError(ErrorCode.OLLAMA_INDISPONIVEL, "Ollama fora do ar")

    monkeypatch.setattr(ollama, "embed", fora)
    corpo = cliente.post("/api/v1/perguntas", json={"pergunta": "Exemplol pode com álcool?"}).json()
    assert corpo["erro"]["codigo"] == "OLLAMA_INDISPONIVEL"
    assert cliente.get(f"/api/v1/historico/{corpo['consulta_id']}").json()["status"] == "erro"


# ---- "Não ajudou" (docs/DECISOES.md, seção 26) ----------------------------------------------

INDICACAO = "Exemplol é indicado para o tratamento da hipertensão arterial em adultos"
CONTRAINDICACAO = "Exemplol é contraindicado para pacientes com hipersensibilidade à exemplina"
CONSERVAR = "Conservar em temperatura ambiente, entre 15 °C e 30 °C"
POSOLOGIA = "A dose inicial usual é de 50 mg uma vez ao dia"
# Ordem de semelhança de "O que é o Exemplol?" nos testes do "não ajudou". "Características farmacológicas" fica por
# último de propósito: desde a F3b (docs/DECISOES.md, seção 34) ela é rebaixada na rodada 1 quando a pergunta não a pede,
# então não é um dos dois primeiros trechos. Os testes de 20/10 a esperavam em segundo e só voltaram a rodar com o banco.
ORDEM_NAO_AJUDOU = [
    "indicacoes", "contraindicacoes", "armazenamento", "advertencias", "posologia", "caracteristicas_farmacologicas",
]


def _destacado(fonte: dict) -> list[str]:
    return [fonte["trecho"][i:f] for i, f in fonte["destaques"]]


def _outra(cliente, consulta_id: str):
    return cliente.post(f"/api/v1/perguntas/{consulta_id}/outra-explicacao")


def _secoes(corpo: dict) -> list[str]:
    return [f["secao"] for f in corpo["fontes"]]


def test_nao_ajudou_refaz_sem_os_trechos_citados_e_fica_ligado(cliente, modelo):
    # "O que é o Exemplol?" não indica seção: vale a semelhança, na ordem de ORDEM_NAO_AJUDOU.
    chamadas = modelo(ORDEM_NAO_AJUDOU, [_resposta((INDICACAO, 1)), _resposta((CONSERVAR, 1)), _resposta((POSOLOGIA, 1))])
    primeira = cliente.post("/api/v1/perguntas", json={"pergunta": "O que é o Exemplol?"}).json()
    assert (primeira["status"], primeira["outra_explicacao"]) == ("respondida", False)
    assert _secoes(primeira) == ["indicacoes", "contraindicacoes"]
    # A fonte citada vem com a frase que sustenta a resposta; a não citada, sem destaque.
    assert _destacado(primeira["fontes"][0]) == [f"{INDICACAO}."] and primeira["fontes"][1]["destaques"] == []

    # Sem o trecho citado (indicações). A contraindicação, enviada e não citada, fica depois dos nunca enviados.
    segunda = _outra(cliente, primeira["consulta_id"])
    assert segunda.status_code == 200
    segunda = segunda.json()
    assert (segunda["status"], segunda["outra_explicacao"], segunda["resposta"]) == ("respondida", True, f"{CONSERVAR} [1].")
    assert _secoes(segunda) == ["armazenamento", "advertencias"]
    assert segunda["consulta_id"] != primeira["consulta_id"] and len(chamadas) == 2

    # O histórico liga a outra explicação à que não ajudou, com a mesma pergunta e os destaques.
    detalhe = cliente.get(f"/api/v1/historico/{segunda['consulta_id']}").json()
    assert (detalhe["pergunta"], detalhe["consulta_origem_id"], detalhe["outra_explicacao"]) == (
        "O que é o Exemplol?", primeira["consulta_id"], True,
    )
    assert detalhe["fontes"] == segunda["fontes"] and _destacado(detalhe["fontes"][0])
    itens = cliente.get("/api/v1/historico").json()["itens"]
    assert [i["outra_explicacao"] for i in itens] == [True, False]

    # De novo: sem os citados nas duas respostas anteriores; sobram os nunca enviados.
    terceira = _outra(cliente, segunda["consulta_id"]).json()
    assert (terceira["status"], _secoes(terceira), terceira["resposta"]) == (
        "respondida", ["posologia", "caracteristicas_farmacologicas"], f"{POSOLOGIA} [1].",
    )


def test_nao_ajudou_sem_outro_trecho_da_secao_nao_chama_o_modelo(cliente, modelo):
    # A bula de teste tem um trecho de interações só, e ele já foi citado.
    chamadas = modelo("interacoes", _resposta(("Pode reduzir o efeito anti-hipertensivo", 1)))
    primeira = cliente.post("/api/v1/perguntas", json={"pergunta": "Exemplol tem interação medicamentosa?"}).json()
    outra = _outra(cliente, primeira["consulta_id"]).json()
    assert (outra["status"], outra["motivo"], outra["fontes"], outra["outra_explicacao"]) == (
        "nao_encontrada", "sem_outra_explicacao", [], True,
    )
    assert len(chamadas) == 1  # só a primeira resposta
    assert cliente.get(f"/api/v1/historico/{outra['consulta_id']}").json()["motivo"] == "sem_outra_explicacao"

    # O que não é resposta não pede outra explicação.
    assert _erro(_outra(cliente, outra["consulta_id"])) == (422, "ENTRADA_INVALIDA")


def test_nao_ajudou_de_consulta_que_nao_existe_da_404(cliente):
    assert _erro(_outra(cliente, "00000000-0000-0000-0000-000000000000")) == (404, "CONSULTA_NAO_ENCONTRADA")


def test_nao_ajudou_mantem_a_pergunta_anterior_da_conversa(cliente, modelo):
    # "E como conservar?" só é sobre o Exemplol por causa da pergunta anterior: a outra explicação
    # também precisa dela.
    modelo(["armazenamento", "posologia", "advertencias"], _resposta((CONSERVAR, 1)))
    antes = cliente.post("/api/v1/perguntas", json={"pergunta": "O que é o Exemplol?"}).json()
    seguinte = cliente.post("/api/v1/perguntas", json={
        "pergunta": "E como conservar?", "consulta_origem_id": antes["consulta_id"],
    }).json()
    assert (seguinte["status"], seguinte["continuacao"]) == ("respondida", True)
    outra = _outra(cliente, seguinte["consulta_id"]).json()
    # Com o Exemplol herdado, a seção de armazenamento dele acabou: não há outra explicação.
    assert (outra["status"], outra["motivo"], outra["principios_ativos"], outra["continuacao"]) == (
        "nao_encontrada", "sem_outra_explicacao", ["maleato de exemplina"], True,
    )


# --- F3d: o mesmo agente em fluxo NDJSON, com o progresso (docs/DECISOES.md, seção 36) ------------


def _fluxo(r) -> list[dict]:
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    return [json.loads(linha) for linha in r.text.splitlines()]


def _etapas(eventos) -> list[tuple[str, int, int]]:
    return [(e["etapa"], e["rodada"], e["de"]) for e in eventos if e["tipo"] == "etapa"]


def test_pergunta_com_progresso_manda_as_etapas_e_a_mesma_resposta(cliente, modelo):
    modelo("interacoes", _resposta(("Pode reduzir o efeito anti-hipertensivo", 1)))
    eventos = _fluxo(cliente.post("/api/v1/perguntas/com-progresso", json={"pergunta": "Exemplol tem interação medicamentosa?"}))
    assert _etapas(eventos) == [("buscando", 1, 3), ("escrevendo", 1, 3), ("conferindo", 1, 3)]
    final = eventos[-1]
    assert final["tipo"] == "resposta"
    resposta = final["resposta"]
    assert (resposta["status"], resposta["resposta"], resposta["aviso"]) == (
        "respondida", "Pode reduzir o efeito anti-hipertensivo [1].", AVISO_SAUDE,
    )
    # A consulta foi gravada como na rota comum.
    detalhe = cliente.get(f"/api/v1/historico/{resposta['consulta_id']}").json()
    assert (detalhe["status"], detalhe["resposta"]) == ("respondida", resposta["resposta"])


def test_rodadas_esgotadas_vem_como_erro_no_fluxo_com_as_tres_rodadas(cliente, modelo):
    modelo("posologia", _resposta(("A dose máxima é de 200 mg uma vez ao dia", 1)))
    eventos = _fluxo(cliente.post("/api/v1/perguntas/com-progresso", json={"pergunta": "Como tomar o Exemplol?"}))
    assert [(e, r) for e, r, _ in _etapas(eventos)] == [
        ("buscando", 1), ("escrevendo", 1), ("conferindo", 1),
        ("buscando", 2), ("escrevendo", 2), ("conferindo", 2),
        ("buscando", 3), ("escrevendo", 3), ("conferindo", 3),
    ]
    erro = eventos[-1]
    assert (erro["tipo"], erro["status"], erro["erro"]["codigo"]) == ("erro", 502, "RESPOSTA_NAO_VERIFICADA")
    assert "3 rodada(s)" in erro["erro"]["mensagem"] and erro["consulta_id"]  # a conversa continua depois do erro


def test_entrada_invalida_e_http_comum_e_erro_do_agente_vai_no_fluxo(cliente):
    # Entrada inválida (Pydantic) sai como no POST /perguntas, antes do fluxo.
    assert _erro(cliente.post("/api/v1/perguntas/com-progresso", json={"pergunta": "oi"})) == (422, "ENTRADA_INVALIDA")
    # Filtro desconhecido só o agente descobre: vai como evento de erro, com o 404 que ele teria.
    eventos = _fluxo(
        cliente.post("/api/v1/perguntas/com-progresso", json={"pergunta": "Como tomar o Exemplol?", "principios_ativos": ["xyzabc"]})
    )
    erro = eventos[-1]
    assert (erro["tipo"], erro["status"], erro["erro"]["codigo"]) == ("erro", 404, "MEDICAMENTO_NAO_ENCONTRADO")
    assert _etapas(eventos) == []  # nem chegou a buscar


def test_nao_ajudou_com_progresso_manda_as_etapas_e_a_resposta_nova(cliente, modelo):
    modelo(ORDEM_NAO_AJUDOU, [_resposta((INDICACAO, 1)), _resposta((CONSERVAR, 1))])
    primeira = cliente.post("/api/v1/perguntas", json={"pergunta": "O que é o Exemplol?"}).json()
    eventos = _fluxo(cliente.post(f"/api/v1/perguntas/{primeira['consulta_id']}/outra-explicacao/com-progresso"))
    # O "não ajudou" é uma pergunta nova para o agente: a contagem de rodadas recomeça em 1.
    assert _etapas(eventos) == [("buscando", 1, 3), ("escrevendo", 1, 3), ("conferindo", 1, 3)]
    final = eventos[-1]["resposta"]
    assert (eventos[-1]["tipo"], final["outra_explicacao"], final["resposta"]) == ("resposta", True, f"{CONSERVAR} [1].")
    # Consulta que não existe: 404 comum, antes do fluxo.
    r = cliente.post(f"/api/v1/perguntas/{uuid4()}/outra-explicacao/com-progresso")
    assert _erro(r) == (404, "CONSULTA_NAO_ENCONTRADA")


# --- Segurança: teto de perguntas em andamento (docs/SEGURANCA.md, achado 4) ----------------------


def test_perguntas_demais_ao_mesmo_tempo_dao_429_e_depois_volta_a_responder(api, cliente, modelo):
    modelo("interacoes", _resposta(("Pode reduzir o efeito anti-hipertensivo", 1)))
    sv = api.app.state.servicos
    tomadas = 0
    while sv.vagas.acquire(blocking=False):  # ocupa todas as vagas, como se houvesse perguntas em andamento
        tomadas += 1
    try:
        pedido = {"pergunta": "Exemplol tem interação medicamentosa?"}
        r = cliente.post("/api/v1/perguntas", json=pedido)
        assert _erro(r) == (429, "SERVIDOR_OCUPADO") and "outras perguntas" in r.json()["erro"]["mensagem"]
        # Sem vaga não há consulta gravada: nada foi processado.
        assert cliente.get("/api/v1/historico").json()["total"] == 0
        # O fluxo com progresso devolve o mesmo erro como linha do fluxo, com o status 429.
        eventos = _fluxo(cliente.post("/api/v1/perguntas/com-progresso", json=pedido))
        assert (eventos[-1]["tipo"], eventos[-1]["status"], eventos[-1]["erro"]["codigo"]) == ("erro", 429, "SERVIDOR_OCUPADO")
    finally:
        for _ in range(tomadas):
            sv.vagas.release()
    assert tomadas == S.perguntas_simultaneas
    # Soltas as vagas, a pergunta passa; e a vaga volta depois dela (não vaza).
    assert cliente.post("/api/v1/perguntas", json=pedido).status_code == 200
    assert sv.vagas.acquire(blocking=False)
    sv.vagas.release()


def test_vaga_volta_mesmo_quando_a_pergunta_termina_em_erro(api, cliente, monkeypatch):
    def fora(textos, s=None):
        raise ollama.OllamaError(ErrorCode.OLLAMA_INDISPONIVEL, "Ollama fora do ar")

    monkeypatch.setattr(ollama, "embed", fora)
    sv = api.app.state.servicos
    for _ in range(S.perguntas_simultaneas + 2):  # mais erros que vagas: se vazasse, a 3ª daria 429
        assert _erro(cliente.post("/api/v1/perguntas", json={"pergunta": "Exemplol tem interação?"})) == (503, "OLLAMA_INDISPONIVEL")
    assert sv.vagas.acquire(blocking=False)
    sv.vagas.release()


def test_filtro_com_nome_gigante_da_422(cliente):
    r = cliente.post("/api/v1/perguntas", json={"pergunta": "Pode com álcool?", "principios_ativos": ["a" * 5000]})
    assert _erro(r) == (422, "ENTRADA_INVALIDA")
