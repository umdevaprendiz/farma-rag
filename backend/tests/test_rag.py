import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.db.vector_search import InMemoryIndex
from app.llm import ollama
from app.schemas.api import ErrorCode
from app.services import graph, grounding, rag
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


def test_resposta_que_nega_o_trecho_reprova(acervo, modelo):
    chamadas = modelo(
        "indicacoes",
        _resposta(("O Exemplol não é indicado para o tratamento da hipertensão arterial", 1)),
        _resposta(("O Exemplol é indicado para o tratamento da hipertensão arterial em adultos", 1)),
    )
    r = rag.responder("Qual o melhor remédio para pressão alta?", acervo, s=S)
    assert (r.status, r.rodadas, len(chamadas)) == ("respondida", 2, 2)
    assert r.tentativas[0].falha.endswith(grounding.NEGA)
    assert graph.DICA_NEGACAO in chamadas[1]


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
    # Sem reranking (a linha de base), a rodada 1 é só semelhança: veio a posologia, e a rodada 2
    # foi buscar a seção da pergunta.
    recusa = {"status": "nao_encontrada", "afirmacoes": []}
    chamadas = modelo("posologia", recusa)
    sem_rerank = S.model_copy(update={"rerank_candidatos": 0})
    r = rag.responder("Exemplol causa queda de cabelo?", acervo, s=sem_rerank)  # "causa" -> reações adversas
    assert (r.status, r.motivo, r.rodadas) == ("nao_encontrada", "sem_contexto", 2)
    assert [f.chunk.secao for f in r.fontes] == ["reacoes_adversas"]
    assert [t.resultado for t in r.tentativas] == ["recusa", "recusa"] and len(chamadas) == 2


def test_reranking_traz_a_secao_da_pergunta_ja_na_rodada_1(acervo, modelo):
    # Com o reranking (seção 30), a seção pedida entra entre os candidatos e ganha a nota da seção:
    # a recusa vem na rodada 1, com o trecho certo consultado, sem a segunda chamada ao LLM.
    chamadas = modelo("posologia", {"status": "nao_encontrada", "afirmacoes": []})
    r = rag.responder("Exemplol causa queda de cabelo?", acervo, s=S)
    assert (r.status, r.rodadas, len(chamadas)) == ("nao_encontrada", 1, 1)
    # A posologia continua primeiro: no teste ela tem semelhança 1,0 contra 0,01 das outras.
    assert "reacoes_adversas" in [f.chunk.secao for f in r.fontes]


def test_secao_da_pergunta_distante_entra_na_rodada_1_e_poupa_a_rodada_2(acervo, monkeypatch):
    # F3b (docs/DECISOES.md, seção 34). A semelhança traz posologia (1,0) e advertências (0,5); as reações
    # adversas, que a pergunta pede ("causa"), ficam a 0,01: o bônus de seção (0,08) não as alcança.
    trechos = list(acervo.chunks.values())
    vetor = np.full(len(trechos), 0.01)
    vetor[next(i for i, c in enumerate(trechos) if c.secao == "posologia")] = 1.0
    vetor[next(i for i, c in enumerate(trechos) if c.secao == "advertencias")] = 0.5
    chamadas = []

    def chat(sistema, usuario, formato, s=None, modelo=None):
        chamadas.append(usuario)
        return ollama.LLMResponse('{"status": "nao_encontrada", "afirmacoes": []}', 400, 0, 8.0, 0.0)

    monkeypatch.setattr(ollama, "embed", lambda textos, s=None: [vetor.tolist()])
    monkeypatch.setattr(ollama, "chat", chat)

    com = rag.responder("Exemplol causa queda de cabelo?", acervo, s=S)
    primeira = [acervo.chunks[c].secao for c in com.tentativas[0].chunk_ids]
    assert "reacoes_adversas" in primeira and len(primeira) == 2  # entrou no lugar da menos parecida
    assert "advertencias" not in primeira
    assert (com.status, com.rodadas, len(chamadas)) == ("nao_encontrada", 1, 1)

    chamadas.clear()
    sem = rag.responder("Exemplol causa queda de cabelo?", acervo, s=S.model_copy(update={"rerank_garante_secao": False}))
    assert "reacoes_adversas" not in [acervo.chunks[c].secao for c in sem.tentativas[0].chunk_ids]
    assert (sem.rodadas, len(chamadas)) == (2, 2)  # a linha de base paga a segunda rodada inteira


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


def test_citacao_trocada_e_corrigida_sem_outra_rodada(acervo, modelo):
    # Caso real (consulta 12f61084, 17/10): a frase certa citando o trecho errado, duas vezes. Antes
    # custava a rodada 2 (~45 s); agora o código troca o [n] e tira a repetida (DECISOES, seção 31).
    frase = "A dose máxima é de 100 mg uma vez ao dia"
    chamadas = modelo("posologia", _resposta((frase, 2), (frase, 1)))
    r = rag.responder("Como tomar o Exemplol?", acervo, s=S)
    assert (r.status, r.rodadas, len(chamadas)) == ("respondida", 1, 1)
    assert r.resposta == f"{frase} [1]."
    assert r.tentativas[0].correcoes == ["afirmação 1: trecho [2] -> [1]", "afirmação 2 repetida, saiu"]
    assert r.fontes[0].chunk.secao == "posologia" and [f.citado for f in r.fontes] == [True, False]
    i, f = r.fontes[0].destaques[0]
    assert r.fontes[0].chunk.texto[i:f] == f"{frase}."  # a fonte mostra a frase no trecho certo


IBUPROFENO_NA_BULA = (
    "O uso com anti-inflamatórios não esteroidais, como o ibuprofeno, pode reduzir o efeito anti-hipertensivo da exemplina"
)


@pytest.mark.parametrize("frase, aprovada", [
    ("Diuréticos poupadores de potássio podem aumentar o potássio no sangue", False),  # fala de outra coisa
    (IBUPROFENO_NA_BULA, True),
])
def test_resposta_tem_de_falar_do_remedio_da_pergunta_que_o_trecho_cita(acervo, modelo, frase, aprovada):
    # Caso real (consulta 72e05926): "E com varfarina?" respondida com frases do clopidogrel passava,
    # porque a regra do assunto tira os nomes de remédio da pergunta (DECISOES, seção 31).
    acervo.catalogo = Catalog.montar(list(acervo.bulas.values()), ["captopril", "ibuprofeno"])
    modelo("interacoes", _resposta((frase, 1)))
    anterior = rag.PreviousQuestion("Como conservar o Exemplol?", ["maleato de exemplina"], False)
    r = rag.responder("E o ibuprofeno?", acervo, s=S, anterior=anterior)
    primeira = r.tentativas[0]
    assert "__interacoes__" in primeira.chunk_ids[0]  # o trecho [1] fala do ibuprofeno
    if aprovada:
        assert (r.status, r.rodadas, primeira.resultado) == ("respondida", 1, "aprovada")
    else:
        assert primeira.resultado == "reprovada" and "a resposta não fala de ibuprofeno" in primeira.falha


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


def test_filtro_aceita_nome_comercial_e_sem_bula_e_recusa_desconhecido(acervo, modelo):
    modelo("posologia", _resposta(("50 mg ao dia.", 1)))
    r = rag.responder("Qual a dose inicial?", acervo, principios_ativos=["Exemplol"], s=S)
    assert r.status == "respondida" and r.principios_ativos == ["maleato de exemplina"]
    # Conhecido sem bula: como na pergunta escrita (docs/DECISOES.md, seção 28; antes era erro).
    r = rag.responder("Qual a dose inicial?", acervo, principios_ativos=["captopril"], s=S)
    assert (r.status, r.motivo, r.medicamentos_nao_indexados) == ("nao_encontrada", "medicamento_nao_indexado", ["captopril"])
    r = rag.responder("Qual a dose inicial?", acervo, principios_ativos=["remediox"], s=S)
    assert (r.status, r.codigo_erro) == ("erro", ErrorCode.MEDICAMENTO_NAO_ENCONTRADO)


def test_abaixo_do_limiar_nao_chama_o_modelo(acervo, modelo):
    chamadas = modelo("posologia", {"status": "respondida", "afirmacoes": []})
    r = rag.responder("Como tomar o Exemplol?", acervo, s=Settings(_env_file=None, limiar_similaridade=1.5))
    assert (r.status, r.motivo, r.fontes, chamadas) == ("nao_encontrada", "sem_contexto", [], [])


def test_formato_so_permite_citar_trechos_enviados():
    assert rag.formato(3)["properties"]["afirmacoes"]["items"]["properties"]["trecho"]["enum"] == [1, 2, 3]


# --- F3d: o agente avisa cada etapa que começa (docs/DECISOES.md, seção 36) -----------------------


def _etapas():
    avisos: list[tuple[str, int]] = []
    return avisos, lambda etapa, rodada: avisos.append((etapa, rodada))


def test_avisa_cada_etapa_de_uma_resposta_na_rodada_1(acervo, modelo):
    modelo("posologia", _resposta(("A dose máxima é de 100 mg uma vez ao dia", 1)))
    avisos, avisar = _etapas()
    r = rag.responder("Como tomar o Exemplol?", acervo, s=S, avisar=avisar)
    assert r.rodadas == 1
    assert avisos == [("buscando", 1), ("escrevendo", 1), ("conferindo", 1)]


def test_avisa_a_rodada_2_quando_a_primeira_reprova(acervo, modelo):
    modelo(
        "posologia",
        _resposta(("A dose máxima é de 200 mg uma vez ao dia", 1)),
        _resposta(("A dose máxima é de 100 mg uma vez ao dia", 1)),
    )
    avisos, avisar = _etapas()
    assert rag.responder("Como tomar o Exemplol?", acervo, s=S, avisar=avisar).rodadas == 2
    assert avisos == [
        ("buscando", 1), ("escrevendo", 1), ("conferindo", 1),
        ("buscando", 2), ("escrevendo", 2), ("conferindo", 2),
    ]


def test_avisa_a_verificacao_na_rodada_3(acervo, modelo):
    errada = _resposta(("A dose máxima é de 300 mg", 1))
    certa = _resposta(("A dose máxima é de 100 mg uma vez ao dia", 1))
    modelo("posologia", errada, errada, certa, verificacao={"a1": "sim"})
    avisos, avisar = _etapas()
    assert rag.responder("Como tomar o Exemplol?", acervo, s=S, avisar=avisar).rodadas == 3
    assert avisos[-2:] == [("conferindo", 3), ("verificando", 3)]
    assert [a for a in avisos if a[0] == "verificando"] == [("verificando", 3)]  # só uma vez, só na 3


def test_frase_copiada_da_bula_nao_avisa_escrevendo(acervo, monkeypatch, modelo):
    # Sem LLM não há "escrevendo": a frase vem pronta, e o aviso ficaria piscando por 1 ms.
    modelo("posologia", {"status": "nao_encontrada", "afirmacoes": []})
    monkeypatch.setattr(ollama, "chat", lambda *a, **k: pytest.fail("a frase vem da bula, sem o LLM"))
    avisos, avisar = _etapas()
    r = rag.responder("Qual a dose máxima do Exemplol?", acervo, s=S, avisar=avisar)  # dose: frase da posologia
    assert r.extrativa and avisos == [("buscando", 1), ("conferindo", 1)]


def test_quem_ouve_o_aviso_nunca_muda_a_resposta(acervo, modelo):
    # O aviso é só informação: se o ouvinte quebrar (conexão fechada), a pergunta segue igual.
    modelo("posologia", _resposta(("A dose máxima é de 100 mg uma vez ao dia", 1)))

    def quebrado(etapa, rodada):
        raise RuntimeError("conexão fechada")

    r = rag.responder("Como tomar o Exemplol?", acervo, s=S, avisar=quebrado)
    assert (r.status, r.resposta) == ("respondida", "A dose máxima é de 100 mg uma vez ao dia [1].")


def test_conselho_que_o_trecho_nao_traz_sai_da_resposta(acervo, modelo):
    """DECISOES §42: "consulte um médico" que a bula não diz sai; o aviso fixo da F2 já diz isso."""
    modelo("interacoes", _resposta(
        ("Pode reduzir o efeito anti-hipertensivo, consulte um médico.", 1), ("Converse com seu médico antes de usar", 1),
    ))
    r = rag.responder("Exemplol tem interações?", acervo, s=S)
    assert (r.status, r.rodadas) == ("respondida", 1)
    assert r.resposta == "Pode reduzir o efeito anti-hipertensivo [1]."
    assert r.tentativas[0].correcoes == [
        "afirmação 1: conselho que o trecho [1] não traz, cortado",
        "afirmação 2: conselho que o trecho [1] não traz, saiu",
    ]


def test_resposta_que_e_so_conselho_reprova(acervo, modelo):
    chamadas = modelo(
        "interacoes", _resposta(("Consulte um médico", 1)), _resposta(("Pode reduzir o efeito anti-hipertensivo", 1)),
    )
    r = rag.responder("Exemplol tem interações?", acervo, s=S)
    assert (r.status, r.rodadas) == ("respondida", 2)
    assert r.tentativas[0].falha == "a resposta só traz conselho que os trechos não trazem"
    assert graph.DICA_ASSUNTO in chamadas[1]


# --- Seção 47: escopo, substância que a bula não cita e "sim" sem permissão ---------------------------------


@pytest.mark.parametrize("pergunta", [
    "Quanto custa o Exemplol 50 mg com 30 comprimidos?",
    "Qual a dose do Exemplol para um cachorro de 10 kg?",
    "Onde comprar o Exemplol mais barato?",
])
def test_pergunta_de_preco_ou_animal_nao_busca_nem_chama_o_modelo(acervo, modelo, pergunta):
    chamadas = modelo("posologia", _resposta(("A dose inicial usual é de 50 mg uma vez ao dia", 1)))
    r = rag.responder(pergunta, acervo, s=S)
    assert (r.status, r.motivo, r.fontes, r.rodadas, chamadas) == ("nao_encontrada", "fora_do_escopo", [], 0, [])


def test_cha_que_nenhuma_bula_do_remedio_cita_nao_chama_o_modelo(acervo, modelo):
    chamadas = modelo("interacoes", _resposta(("O uso com anti-inflamatórios pode piorar a função renal", 1)))
    r = rag.responder("Exemplol pode ser tomado junto com chá de boldo?", acervo, s=S)
    assert (r.status, r.motivo, r.fontes, chamadas) == ("nao_encontrada", "fora_da_bula", [], [])


def test_cha_que_a_bula_cita_segue_para_a_resposta(acervo, modelo):
    # "alimentos" não é chá, erva nem suco: a guarda não olha; e "suco de" só barra o que nenhuma bula do remédio cita.
    modelo("posologia", _resposta(("Os comprimidos podem ser tomados com ou sem alimentos", 1)))
    r = rag.responder("Exemplol pode ser tomado com suco de laranja?", acervo, s=S)
    assert r.motivo == "fora_da_bula"
    modelo("posologia", _resposta(("Os comprimidos podem ser tomados com ou sem alimentos", 1)))
    r = rag.responder("Exemplol pode ser tomado com alimentos?", acervo, s=S)
    assert r.status == "respondida"


def test_sim_sem_dizer_que_pode_reprova_e_a_proxima_rodada_recebe_o_aviso(acervo, modelo):
    chamadas = modelo(
        "advertencias",
        _resposta(("Sim, é recomendado monitorar a creatinina e o potássio no sangue", 1)),
        _resposta(("Em pacientes com insuficiência renal, recomenda-se monitorar a creatinina e o potássio", 1)),
    )
    r = rag.responder("Exemplol pode ser usado em pacientes com insuficiência renal?", acervo, s=S)
    assert r.status == "respondida" and r.rodadas == 2
    assert r.tentativas[0].resultado == "reprovada" and "sim" in r.tentativas[0].falha
    assert graph.DICA_SIM in chamadas[1]
