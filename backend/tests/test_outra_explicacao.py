""""Não ajudou, explique de outro jeito" e o destaque das frases na fonte (docs/DECISOES.md, seção 26).
Ollama falso, como em test_rag.py."""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.db.busca_vetorial import IndiceEmMemoria
from app.llm import ollama
from app.services import rag
from app.services.acervo import Acervo
from app.services.medicamentos import Catalogo
from app.services.rag import OutraExplicacao
from ingestao.gerar_chunks import gerar_chunks
from ingestao.processar_bula import montar_bula

FIXTURES = Path(__file__).parent / "fixtures"
S = Settings(_env_file=None, top_k=2)
# Bulas: o Exemplol de dois laboratórios (mesmo princípio ativo) e o Outrol.
A, B, O = "exemplol-50-mg-laboratorio-teste", "exemplol-50-mg-laboratorio-outro", "outrol-10-mg-laboratorio-teste"
CONSERVAR = "Conservar em temperatura ambiente, entre 15 °C e 30 °C"


def trecho(bula: str, secao: str) -> str:
    return f"{bula}__{secao}__000"


@pytest.fixture
def acervo():
    """As três bulas com o texto da bula de teste, um eixo por trecho."""
    bulas = [
        montar_bula(FIXTURES / "bula_exemplo.txt", nome, ativo, fabricante, date(2025, 3, 15), [nome.split()[0]])
        for nome, ativo, fabricante in [
            ("Exemplol 50 mg", "maleato de exemplina", "Laboratório Teste"),
            ("Exemplol 50 mg", "maleato de exemplina", "Laboratório Outro"),
            ("Outrol 10 mg", "ácido outrínico", "Laboratório Teste"),
        ]
    ]
    chunks = [c for b in bulas for c in gerar_chunks(b, 300, 50)]
    indice = IndiceEmMemoria([c.chunk_id for c in chunks], [c.principio_ativo for c in chunks], np.eye(len(chunks)))
    return Acervo({b.id: b for b in bulas}, {c.chunk_id: c for c in chunks}, indice, Catalogo.montar(bulas, ["captopril"]))


@pytest.fixture
def modelo(monkeypatch, acervo):
    """Troca o Ollama: a busca acha primeiro os trechos de `ordem`, nessa ordem; a geração devolve
    `saida`. Devolve a lista de prompts de geração (vazia = o modelo não foi chamado)."""
    ids = list(acervo.chunks)
    chamadas = []

    def preparar(ordem: list[str], saida: dict):
        vetor = np.full(len(ids), 0.01)
        for i, chunk_id in enumerate(ordem):
            vetor[ids.index(chunk_id)] = 1.0 - 0.1 * i
        monkeypatch.setattr(ollama, "embed", lambda textos, s=None: [vetor.tolist()])

        def chat(sistema, usuario, formato, s=None, modelo=None):
            chamadas.append(usuario)
            return ollama.RespostaLLM(json.dumps(saida), prompt_tokens=400, tokens_gerados=40, prompt_s=8.0, geracao_s=4.0)

        monkeypatch.setattr(ollama, "chat", chat)
        return chamadas

    return preparar


def _resposta(*frases: tuple[str, int]) -> dict:
    return {"status": "respondida", "afirmacoes": [{"texto": t, "trecho": n} for t, n in frases]}


def _destacado(fonte: rag.Fonte) -> list[str]:
    return [fonte.chunk.texto[i:f] for i, f in fonte.destaques]


# ---- Destaque ------------------------------------------------------------------------------

def test_fonte_citada_traz_a_frase_que_sustenta_a_resposta(acervo, modelo):
    modelo([trecho(A, "armazenamento")], _resposta((CONSERVAR, 1)))
    r = rag.responder("Como conservar o Exemplol?", acervo, s=S)
    assert (r.status, r.outra_explicacao) == ("respondida", False)
    assert r.fontes[0].citado and _destacado(r.fontes[0]) == [
        "Conservar em temperatura ambiente (entre 15 °C e 30 °C), protegido da luz e da umidade."
    ]
    assert [f.destaques for f in r.fontes if not f.citado] == [()]


def test_frase_copiada_da_bula_vem_destacada(acervo, modelo):
    chamadas = modelo([trecho(A, "posologia")], _resposta(("não usada", 1)))
    r = rag.responder("Qual a dose máxima do Exemplol?", acervo, s=S)
    assert (r.status, r.extrativa, chamadas) == ("respondida", True, [])
    citadas = [f for f in r.fontes if f.citado]
    assert citadas and all(_destacado(f) == ["A dose máxima é de 100 mg uma vez ao dia."] for f in citadas)


# ---- Não ajudou ----------------------------------------------------------------------------

def test_evita_os_citados_e_manda_primeiro_os_nunca_enviados(acervo, modelo):
    # "O que é o Exemplol?" não indica seção: vale a semelhança, sem o citado e depois do só enviado.
    citado, enviado = trecho(A, "indicacoes"), trecho(A, "caracteristicas_farmacologicas")
    novos = [trecho(B, "indicacoes"), trecho(A, "composicao")]
    chamadas = modelo(
        [citado, enviado, *novos], _resposta(("Exemplol é indicado para o tratamento da hipertensão arterial em adultos", 1))
    )
    r = rag.responder("O que é o Exemplol?", acervo, s=S, outra=OutraExplicacao(evitar=[citado], enviados=[citado, enviado]))
    assert [f.chunk.chunk_id for f in r.fontes] == novos
    assert (r.status, r.outra_explicacao, len(chamadas)) == ("respondida", True, 1)


def test_usa_o_trecho_da_mesma_secao_em_outra_bula(acervo, modelo):
    citado = trecho(A, "armazenamento")
    chamadas = modelo([citado, trecho(A, "posologia"), trecho(B, "armazenamento")], _resposta((CONSERVAR, 1)))
    r = rag.responder("Como conservar o Exemplol?", acervo, s=S, outra=OutraExplicacao([citado], [citado]))
    assert [f.chunk.chunk_id for f in r.fontes] == [trecho(B, "armazenamento")]  # a seção antes da semelhança
    assert (r.status, r.fontes[0].citado, len(chamadas)) == ("respondida", True, 1)


def test_sem_outro_trecho_da_secao_nao_chama_o_modelo(acervo, modelo):
    armazenamento = [trecho(A, "armazenamento"), trecho(B, "armazenamento")]
    chamadas = modelo(armazenamento, _resposta((CONSERVAR, 1)))
    r = rag.responder("Como conservar o Exemplol?", acervo, s=S, outra=OutraExplicacao(armazenamento, armazenamento))
    assert (r.status, r.motivo, r.fontes, chamadas) == ("nao_encontrada", "sem_outra_explicacao", [], [])


def test_recusa_do_modelo_vira_sem_outra_explicacao(acervo, modelo):
    citado = trecho(A, "armazenamento")
    modelo([citado, trecho(B, "armazenamento")], {"status": "nao_encontrada", "afirmacoes": []})
    r = rag.responder("Como conservar o Exemplol?", acervo, s=S, outra=OutraExplicacao([citado], [citado]))
    assert (r.status, r.motivo, r.rodadas) == ("nao_encontrada", "sem_outra_explicacao", 1)


def test_pergunta_de_dose_passa_pelo_modelo_em_vez_de_copiar_a_bula(acervo, modelo):
    # A primeira resposta foi a frase copiada; a outra explicação é do modelo, com outro trecho.
    citado = trecho(A, "posologia")
    chamadas = modelo([citado, trecho(B, "posologia")], _resposta(("A dose máxima é de 100 mg uma vez ao dia", 1)))
    r = rag.responder("Qual a dose máxima do Exemplol?", acervo, s=S, outra=OutraExplicacao([citado], [citado]))
    assert (r.status, r.extrativa, len(chamadas)) == ("respondida", False, 1)
    assert [f.chunk.chunk_id for f in r.fontes] == [trecho(B, "posologia")]


def test_busca_geral_traz_outro_trecho_da_secao(acervo, modelo):
    citados = [trecho(A, "indicacoes"), trecho(O, "indicacoes")]
    chamadas = modelo(
        [*citados, trecho(B, "indicacoes")],
        _resposta(("O Exemplol é indicado para o tratamento da hipertensão arterial em adultos", 1)),
    )
    r = rag.responder("Qual o melhor remédio para pressão alta?", acervo, s=S, outra=OutraExplicacao(citados, citados))
    assert [f.chunk.chunk_id for f in r.fontes] == [trecho(B, "indicacoes")]
    assert (r.status, len(chamadas)) == ("respondida", 1)


def test_busca_geral_sem_trecho_novo_na_secao_nao_passa_para_outra(acervo, modelo):
    # Os resultados de eficácia falam de pressão arterial, mas não são indicação: não contam.
    indicacoes = [trecho(A, "indicacoes"), trecho(B, "indicacoes"), trecho(O, "indicacoes")]
    chamadas = modelo([*indicacoes, trecho(A, "resultados_eficacia")], _resposta(("Algo", 1)))
    r = rag.responder("Qual o melhor remédio para pressão alta?", acervo, s=S, outra=OutraExplicacao(indicacoes, indicacoes))
    assert (r.status, r.motivo, chamadas) == ("nao_encontrada", "sem_outra_explicacao", [])
