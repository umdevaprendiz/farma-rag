import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.db.busca_vetorial import IndiceEmMemoria
from app.llm import ollama
from app.schemas.api import CodigoErro
from app.services import grafo, rag
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
            if sistema == grafo.SISTEMA_VERIFICACAO:
                preparar.verificacoes.append((usuario, modelo))
                return ollama.RespostaLLM(json.dumps(verificacao), 200, 10, 4.0, 1.0)
            chamadas.append(usuario)
            saida = saidas[min(len(chamadas), len(saidas)) - 1]
            texto = saida if isinstance(saida, str) else json.dumps(saida)
            return ollama.RespostaLLM(texto, prompt_tokens=400, tokens_gerados=40, prompt_s=8.0, geracao_s=4.0)

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
    r = rag.responder("Exemplol pode com ibuprofeno?", acervo, s=S)
    assert r.status == "respondida"
    assert r.resposta == "Pode reduzir o efeito anti-hipertensivo [1]. Pode piorar a função renal [1]."
    assert r.principios_ativos == ["maleato de exemplina"]
    assert [(f.posicao, f.chunk.secao, f.citado) for f in r.fontes][0] == (1, "interacoes", True)
    assert len(r.fontes) == 2 and not r.fontes[1].citado
    assert r.fontes[0].data_versao == date(2025, 3, 15)
    assert "[1] Exemplol 50 mg | Interações medicamentosas | página 2" in chamadas[0]
    assert chamadas[0].endswith("PERGUNTA: Exemplol pode com ibuprofeno?")
    assert (r.rodadas, [t.resultado for t in r.tentativas]) == (1, ["aprovada"])
    assert "OBSERVAÇÃO" not in chamadas[0]


def test_medicamento_sem_bula_nao_chama_o_modelo(acervo, monkeypatch):
    monkeypatch.setattr(ollama, "embed", lambda *a, **k: pytest.fail("não devia buscar"))
    r = rag.responder("Captopril pode com álcool?", acervo, s=S)
    assert (r.status, r.motivo, r.medicamentos_nao_indexados) == ("nao_encontrada", "medicamento_nao_indexado", ["captopril"])
    assert r.fontes == [] and r.rodadas == 0


def test_mistura_segue_com_o_indexado_e_avisa_do_outro(acervo, modelo):
    modelo("interacoes", _resposta(("Pode piorar a função renal", 1)))
    r = rag.responder("Exemplol com captopril", acervo, s=S)
    assert r.status == "respondida" and r.medicamentos_nao_indexados == ["captopril"]


def test_sem_medicamento_pede_esclarecimento_sem_buscar(acervo, monkeypatch, modelo):
    monkeypatch.setattr(ollama, "embed", lambda *a, **k: pytest.fail("não devia buscar"))
    r = rag.responder("Qual a dose máxima por dia?", acervo, s=S)
    assert r.status == "precisa_esclarecimento" and r.esclarecimento.motivo == "medicamento_nao_reconhecido"
    assert [o.rotulo for o in r.esclarecimento.opcoes] == ["maleato de exemplina (Exemplol 50 mg)"]

    # O reenvio com a opção escolhida passa pelo filtro e responde.
    modelo("posologia", _resposta(("A dose máxima é de 100 mg uma vez ao dia", 1)))
    r = rag.responder("Qual a dose máxima por dia?", acervo, r.esclarecimento.opcoes[0].principios_ativos, s=S)
    assert r.status == "respondida"


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
    r = rag.responder("Qual a dose máxima de Exemplol?", acervo, s=S)
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
    r = rag.responder("Qual a dose máxima de Exemplol?", acervo, s=s)
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
    r = rag.responder("Qual a dose máxima de Exemplol?", acervo, s=S)
    assert (r.status, r.codigo_erro, r.resposta) == ("erro", CodigoErro.RESPOSTA_NAO_VERIFICADA, None)
    assert r.detalhe_erro.startswith("Nenhuma resposta passou na verificação em 3 rodada(s) (limite de rodadas).")
    assert "rodada 1: afirmação 1, trecho [1]: '300 mg' não está no trecho" in r.detalhe_erro
    assert "rodada 3: a verificação não confirmou a(s) afirmação(ões) 1" in r.detalhe_erro


@pytest.mark.parametrize("saida, codigo", [
    ("isto não é json", CodigoErro.SAIDA_INVALIDA),
    ({"status": "respondida"}, CodigoErro.SAIDA_INVALIDA),
    ({"status": "respondida", "afirmacoes": []}, CodigoErro.SAIDA_INVALIDA),
    ({"status": "respondida", "afirmacoes": [{"texto": "Algo.", "trecho": 9}]}, CodigoErro.CITACAO_INVALIDA),
])
def test_saida_invalida_em_todas_as_rodadas_vira_erro_com_a_saida_crua(acervo, modelo, saida, codigo):
    chamadas = modelo("posologia", saida)
    r = rag.responder("Qual a dose de Exemplol?", acervo, s=S)
    assert (r.status, r.codigo_erro, r.resposta, r.rodadas) == ("erro", codigo, None, 3)
    assert r.saida_bruta and len(chamadas) == 3 and not modelo.verificacoes


def test_orcamento_esgotado_nao_abre_outra_rodada(acervo, modelo):
    chamadas = modelo("posologia", _resposta(("A dose máxima é de 300 mg", 1)))
    r = rag.responder("Qual a dose de Exemplol?", acervo, s=Settings(_env_file=None, top_k=2, orcamento_s=0))
    assert (r.status, r.codigo_erro, r.rodadas, len(chamadas)) == ("erro", CodigoErro.RESPOSTA_NAO_VERIFICADA, 1, 1)
    assert "(orçamento de 0 s)" in r.detalhe_erro


def test_ollama_fora_do_ar(acervo, monkeypatch):
    def cair(*a, **k):
        raise ollama.OllamaError(CodigoErro.OLLAMA_INDISPONIVEL, "fora do ar")

    monkeypatch.setattr(ollama, "embed", cair)
    r = rag.responder("Qual a dose de Exemplol?", acervo, s=S)
    assert (r.status, r.codigo_erro) == ("erro", CodigoErro.OLLAMA_INDISPONIVEL)


def test_filtro_aceita_nome_comercial_e_recusa_desconhecido(acervo, modelo):
    modelo("posologia", _resposta(("50 mg ao dia.", 1)))
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
