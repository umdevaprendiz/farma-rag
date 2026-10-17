import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.db.vector_search import InMemoryIndex
from app.llm import ollama
from app.schemas.api import ErrorCode
from app.services import graph, rag
from app.services.corpus import Corpus
from app.services.medicines import Catalog
from ingestion.generate_chunks import gerar_chunks
from ingestion.process_leaflet import montar_bula

FIXTURES = Path(__file__).parent / "fixtures"
S = Settings(_env_file=None, top_k=2)


@pytest.fixture
def acervo():
    bula = montar_bula(
        FIXTURES / "sample_leaflet.txt", "Exemplol 50 mg", "maleato de exemplina", "Laboratório Teste",
        date(2025, 3, 15), ["Exemplol"],
    )
    chunks = gerar_chunks(bula, 300, 50)
    # Um eixo por trecho: a "pergunta" aponta para o trecho que o teste quiser recuperar.
    indice = InMemoryIndex([c.chunk_id for c in chunks], [c.principio_ativo for c in chunks], np.eye(len(chunks)))
    return Corpus({bula.id: bula}, {c.chunk_id: c for c in chunks}, indice, Catalog.montar([bula], ["captopril"]))


@pytest.fixture
def modelo(monkeypatch, acervo):
    """Troca o Ollama: a busca devolve primeiro o trecho de `secao`.

    A geração devolve `saidas` em ordem, uma por rodada (a última se repete); o verificar_llm
    devolve `verificacao`. Devolve a lista de prompts de geração; os da verificação ficam em
    `preparar.verificacoes`, como (prompt, modelo).
    """
    chamadas = []

    def preparar(secao: str, *saidas: dict | str, verificacao: dict | None = None):
        alvo = next(i for i, c in enumerate(acervo.chunks.values()) if c.secao == secao)
        vetor = np.full(len(acervo.chunks), 0.01)
        vetor[alvo] = 1.0
        monkeypatch.setattr(ollama, "embed", lambda textos, s=None: [vetor.tolist()])

        def chat(sistema, usuario, formato, s=None, modelo=None):
            if sistema == graph.SISTEMA_VERIFICACAO:
                preparar.verificacoes.append((usuario, modelo))
                return ollama.LLMResponse(json.dumps(verificacao), 200, 10, 4.0, 1.0)
            chamadas.append(usuario)
            saida = saidas[min(len(chamadas), len(saidas)) - 1]
            texto = saida if isinstance(saida, str) else json.dumps(saida)
            return ollama.LLMResponse(texto, prompt_tokens=400, tokens_gerados=40, prompt_s=8.0, geracao_s=4.0)

        monkeypatch.setattr(ollama, "chat", chat)
        return chamadas

    preparar.verificacoes = []
    return preparar


def _resposta(*frases: tuple[str, int]) -> dict:
    return {"status": "respondida", "afirmacoes": [{"texto": t, "trecho": n} for t, n in frases]}


def test_resposta_com_citacao_por_frase(acervo, modelo):
    chamadas = modelo("interacoes", _resposta(
        ("Pode reduzir o efeito anti-hipertensivo.", 1), ("Pode piorar a função renal", 1),
    ))
    # Sem o nome do outro remédio: com ele, a resposta é a frase da bula, sem o LLM (test_extract).
    r = rag.responder("Exemplol tem interações?", acervo, s=S)
    assert r.status == "respondida"
    assert r.resposta == "Pode reduzir o efeito anti-hipertensivo [1]. Pode piorar a função renal [1]."
    assert r.principios_ativos == ["maleato de exemplina"]
    assert [(f.posicao, f.chunk.secao, f.citado) for f in r.fontes][0] == (1, "interacoes", True)
    assert len(r.fontes) == 2 and not r.fontes[1].citado
    assert r.fontes[0].data_versao == date(2025, 3, 15)
    assert "[1] Exemplol 50 mg | Interações medicamentosas | página 2" in chamadas[0]
    assert chamadas[0].endswith("PERGUNTA: Exemplol tem interações?")
    assert (r.rodadas, [t.resultado for t in r.tentativas]) == (1, ["aprovada"])
    assert "OBSERVAÇÃO" not in chamadas[0]


def test_remedio_sem_bula_busca_nas_bulas_que_falam_dele(acervo, modelo):
    acervo.catalogo = Catalog.montar(list(acervo.bulas.values()), ["captopril", "ibuprofeno"])
    chamadas = modelo("interacoes", _resposta(
        ("A bula do Exemplol diz que o uso com anti-inflamatórios como o ibuprofeno pode piorar a função renal", 1),
    ))
    r = rag.responder("Ibuprofeno faz mal para os rins?", acervo, s=S)
    assert (r.status, r.medicamentos_nao_indexados, r.principios_ativos) == ("respondida", ["ibuprofeno"], [])
    assert [f.chunk.secao for f in r.fontes] == ["interacoes"]  # o único trecho que fala de ibuprofeno
    assert "OBSERVAÇÃO: Não há trechos da bula de ibuprofeno. Não atribua a ele" in chamadas[0]


def test_remedio_sem_bula_que_nenhuma_bula_cita_nem_chama_o_modelo(acervo, modelo):
    chamadas = modelo("interacoes", _resposta(("Algo", 1)))
    r = rag.responder("Captopril pode com álcool?", acervo, s=S)
    assert (r.status, r.motivo, r.fontes, chamadas) == ("nao_encontrada", "medicamento_nao_indexado", [], [])


def test_assunto_que_nenhuma_bula_trata_nao_chama_o_modelo(acervo, modelo):
    chamadas = modelo("indicacoes", _resposta(("O Exemplol é indicado para hipertensão", 1)))
    r = rag.responder("Estou tonto, qual o melhor medicamento?", acervo, s=S)
    # Só as indicações valem, e nenhuma fala de tontura (a do Exemplol está em reações adversas).
    assert (r.status, r.motivo, r.fontes, chamadas) == ("nao_encontrada", "sem_contexto", [], [])


def test_resposta_que_e_so_o_nome_do_remedio_reprova(acervo, modelo):
    chamadas = modelo("indicacoes", _resposta(("Exemplol 50 mg", 1)), _resposta(("O Exemplol é indicado para o tratamento da hipertensão arterial", 1)))
    r = rag.responder("Qual o melhor remédio para pressão alta?", acervo, s=S)
    assert (r.status, r.rodadas, len(chamadas)) == ("respondida", 2, 2)
    assert "só o nome do medicamento" in r.tentativas[0].falha
    assert graph.DICA_SO_NOME in chamadas[1]  # o aviso diz o que corrigir, não o genérico


def test_mistura_segue_com_o_indexado_e_avisa_do_outro(acervo, modelo):
    modelo("interacoes", _resposta(("Pode piorar a função renal", 1)))
    r = rag.responder("Exemplol com captopril", acervo, s=S)
    assert r.status == "respondida" and r.medicamentos_nao_indexados == ["captopril"]


def test_sem_medicamento_responde_com_a_busca_em_todas_as_bulas(acervo, modelo):
    chamadas = modelo("posologia", _resposta(("A dose máxima do Exemplol é de 100 mg uma vez ao dia", 1)))
    r = rag.responder("Qual a dose máxima por dia?", acervo, s=S)
    assert (r.status, r.principios_ativos, r.esclarecimento) == ("respondida", [], None)
    assert [f.chunk.secao for f in r.fontes] == ["posologia"] and "OBSERVAÇÃO" not in chamadas[0]


@pytest.fixture
def dois_remedios(monkeypatch):
    """Exemplol e Outrol (mesmo texto, princípios ativos diferentes), com um eixo por trecho.
    Devolve `apontar(pesos)`: a "pergunta" fica parecida com cada (princípio ativo, seção) pelo peso."""
    bulas = [
        montar_bula(FIXTURES / "sample_leaflet.txt", nome, ativo, "Laboratório Teste", date(2025, 3, 15), [nome.split()[0]])
        for nome, ativo in [("Exemplol 50 mg", "maleato de exemplina"), ("Outrol 10 mg", "ácido outrínico")]
    ]
    chunks = [c for b in bulas for c in gerar_chunks(b, 300, 50)]
    indice = InMemoryIndex([c.chunk_id for c in chunks], [c.principio_ativo for c in chunks], np.eye(len(chunks)))
    acervo = Corpus({b.id: b for b in bulas}, {c.chunk_id: c for c in chunks}, indice, Catalog.montar(bulas, []))
    monkeypatch.setattr(ollama, "chat", lambda *a, **k: ollama.LLMResponse(
        json.dumps(_resposta(("O potássio no sangue deve ser monitorado", 1))), 1, 1, 0, 0))

    def apontar(pesos: dict[tuple[str, str], float]) -> Corpus:
        vetor = np.full(len(chunks), 0.01)
        for (ativo, secao), peso in pesos.items():
            vetor[next(i for i, c in enumerate(chunks) if c.principio_ativo == ativo and c.secao == secao)] = peso
        monkeypatch.setattr(ollama, "embed", lambda textos, s=None: [vetor.tolist()])
        return acervo

    return apontar


EXEMPLINA, OUTRINICO = "maleato de exemplina", "ácido outrínico"


def test_busca_geral_um_trecho_por_remedio_e_so_os_que_falam_do_assunto(dois_remedios):
    acervo = dois_remedios({
        (EXEMPLINA, "advertencias"): 1.0,  # "...monitorar a creatinina e o potássio no sangue"
        (EXEMPLINA, "interacoes"): 0.9,  # também fala de potássio, mas é o 2º do mesmo remédio
        (OUTRINICO, "interacoes"): 0.5,
        (OUTRINICO, "advertencias"): 0.4,
        (EXEMPLINA, "reacoes_adversas"): 2.0,  # o mais parecido, mas não fala de potássio
    })
    r = rag.responder("Que remédio pode alterar o potássio no sangue?", acervo, s=S)
    assert [(f.posicao, f.chunk.principio_ativo, f.chunk.secao) for f in r.fontes] == [
        (1, EXEMPLINA, "advertencias"), (2, OUTRINICO, "interacoes"),
    ]


def test_busca_geral_fica_na_secao_da_pergunta(dois_remedios):
    acervo = dois_remedios({
        (EXEMPLINA, "resultados_eficacia"): 2.0,  # fala de "pressão arterial", mas não é indicação
        (EXEMPLINA, "indicacoes"): 1.0,
        (OUTRINICO, "indicacoes"): 0.5,
    })
    r = rag.responder("Qual o melhor remédio para pressão alta?", acervo, s=S)
    assert [(f.chunk.principio_ativo, f.chunk.secao) for f in r.fontes] == [(EXEMPLINA, "indicacoes"), (OUTRINICO, "indicacoes")]


def test_recusa_sem_secao_indicada_vale_na_primeira_rodada(acervo, modelo):
    chamadas = modelo("posologia", {"status": "nao_encontrada", "afirmacoes": [{"texto": "Os trechos não informam.", "trecho": 1}]})
    r = rag.responder("Exemplol tem registro?", acervo, s=S)
    assert (r.status, r.motivo, r.resposta, r.rodadas) == ("nao_encontrada", "sem_contexto", None, 1)
    assert len(r.fontes) == 2 and len(chamadas) == 1  # os trechos consultados continuam visíveis


def test_recusa_com_trechos_da_secao_errada_busca_a_secao_da_pergunta(acervo, modelo):
    recusa = {"status": "nao_encontrada", "afirmacoes": []}
    chamadas = modelo("posologia", recusa)
    r = rag.responder("Exemplol causa queda de cabelo?", acervo, s=S)  # "causa" -> reações adversas
    assert (r.status, r.motivo, r.rodadas) == ("nao_encontrada", "sem_contexto", 2)
    assert [f.chunk.secao for f in r.fontes] == ["reacoes_adversas"]
    assert [t.resultado for t in r.tentativas] == ["recusa", "recusa"] and len(chamadas) == 2


def test_numero_sem_fonte_reprova_e_a_rodada_2_corrige(acervo, modelo):
    chamadas = modelo(
        "posologia",
        _resposta(("A dose máxima é de 200 mg uma vez ao dia", 1)),
        _resposta(("A dose máxima é de 100 mg uma vez ao dia", 1)),
    )
    r = rag.responder("Como tomar o Exemplol?", acervo, s=S)
    assert (r.status, r.rodadas) == ("respondida", 2)
    assert r.resposta == "A dose máxima é de 100 mg uma vez ao dia [1]."
    primeira = r.tentativas[0]
    assert primeira.resultado == "reprovada" and "'200 mg' não está no trecho" in primeira.falha
    assert "OBSERVAÇÃO" in chamadas[1] and "200" not in chamadas[1]  # a dica não repete o erro
    assert r.fontes[0].chunk.secao == "posologia"  # a rodada 2 ficou na seção da pergunta
    assert r.tempos_s["geracao"] > 0 and r.prompt_tokens == 800


def test_rodada_3_passa_pela_verificacao_do_llm(acervo, modelo):
    errada = _resposta(("A dose máxima é de 300 mg", 1))
    certa = _resposta(("A dose máxima é de 100 mg uma vez ao dia", 1))
    modelo("posologia", errada, errada, certa, verificacao={"a1": "sim"})
    s = Settings(_env_file=None, top_k=2, llm_modelo_verificacao="modelo-maior")
    r = rag.responder("Como tomar o Exemplol?", acervo, s=s)
    assert (r.status, r.rodadas) == ("respondida", 3)
    assert [t.resultado for t in r.tentativas] == ["reprovada", "reprovada", "aprovada"]
    assert r.tentativas[2].verificacao_bruta == '{"a1": "sim"}'
    prompt, modelo_usado = modelo.verificacoes[0]
    assert modelo_usado == "modelo-maior" and "AFIRMAÇÃO 1: A dose máxima é de 100 mg uma vez ao dia" in prompt
    assert "verificacao" in r.tempos_s


def test_verificacao_negada_vira_502_com_explicacao(acervo, modelo):
    errada = _resposta(("A dose máxima é de 300 mg", 1))
    certa = _resposta(("A dose máxima é de 100 mg uma vez ao dia", 1))
    modelo("posologia", errada, errada, certa, verificacao={"a1": "nao"})
    r = rag.responder("Como tomar o Exemplol?", acervo, s=S)
    assert (r.status, r.codigo_erro, r.resposta) == ("erro", ErrorCode.RESPOSTA_NAO_VERIFICADA, None)
    assert r.detalhe_erro.startswith("Nenhuma resposta passou na verificação em 3 rodada(s) (limite de rodadas).")
    assert "rodada 1: afirmação 1, trecho [1]: '300 mg' não está no trecho" in r.detalhe_erro
    assert "rodada 3: a verificação não confirmou a(s) afirmação(ões) 1" in r.detalhe_erro


@pytest.mark.parametrize("saida, codigo", [
    ("isto não é json", ErrorCode.SAIDA_INVALIDA),
    ({"status": "respondida"}, ErrorCode.SAIDA_INVALIDA),
    ({"status": "respondida", "afirmacoes": []}, ErrorCode.SAIDA_INVALIDA),
    ({"status": "respondida", "afirmacoes": [{"texto": "Algo.", "trecho": 9}]}, ErrorCode.CITACAO_INVALIDA),
])
def test_saida_invalida_em_todas_as_rodadas_vira_erro_com_a_saida_crua(acervo, modelo, saida, codigo):
    chamadas = modelo("posologia", saida)
    r = rag.responder("Como tomar o Exemplol?", acervo, s=S)
    assert (r.status, r.codigo_erro, r.resposta, r.rodadas) == ("erro", codigo, None, 3)
    assert r.saida_bruta and len(chamadas) == 3 and not modelo.verificacoes


def test_orcamento_esgotado_nao_abre_outra_rodada(acervo, modelo):
    chamadas = modelo("posologia", _resposta(("A dose máxima é de 300 mg", 1)))
    r = rag.responder("Como tomar o Exemplol?", acervo, s=Settings(_env_file=None, top_k=2, orcamento_s=0))
    assert (r.status, r.codigo_erro, r.rodadas, len(chamadas)) == ("erro", ErrorCode.RESPOSTA_NAO_VERIFICADA, 1, 1)
    assert "(orçamento de 0 s)" in r.detalhe_erro


def test_ollama_fora_do_ar(acervo, monkeypatch):
    def cair(*a, **k):
        raise ollama.OllamaError(ErrorCode.OLLAMA_INDISPONIVEL, "fora do ar")

    monkeypatch.setattr(ollama, "embed", cair)
    r = rag.responder("Como tomar o Exemplol?", acervo, s=S)
    assert (r.status, r.codigo_erro) == ("erro", ErrorCode.OLLAMA_INDISPONIVEL)


def test_filtro_aceita_nome_comercial_e_recusa_desconhecido(acervo, modelo):
    modelo("posologia", _resposta(("50 mg ao dia.", 1)))
    r = rag.responder("Qual a dose inicial?", acervo, principios_ativos=["Exemplol"], s=S)
    assert r.status == "respondida" and r.principios_ativos == ["maleato de exemplina"]
    r = rag.responder("Qual a dose inicial?", acervo, principios_ativos=["captopril"], s=S)
    assert (r.status, r.codigo_erro) == ("erro", ErrorCode.MEDICAMENTO_NAO_ENCONTRADO)


def test_abaixo_do_limiar_nao_chama_o_modelo(acervo, modelo):
    chamadas = modelo("posologia", {"status": "respondida", "afirmacoes": []})
    r = rag.responder("Como tomar o Exemplol?", acervo, s=Settings(_env_file=None, limiar_similaridade=1.5))
    assert (r.status, r.motivo, r.fontes, chamadas) == ("nao_encontrada", "sem_contexto", [], [])


def test_formato_so_permite_citar_trechos_enviados():
    assert rag.formato(3)["properties"]["afirmacoes"]["items"]["properties"]["trecho"]["enum"] == [1, 2, 3]
