"""BuscaPgvector e carregar_banco contra um PostgreSQL de verdade.

Pulado se o pgvector/psycopg não estiver instalado ou se o banco do .env não responder
(suba com: docker compose up -d db). Roda num schema temporário, apagado no fim: não mexe
nas tabelas do banco de desenvolvimento.
"""

from datetime import date
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

pytest.importorskip("pgvector")
pytest.importorskip("psycopg")

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.exc import OperationalError  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db.busca_pgvector import BuscaPgvector  # noqa: E402
from app.db.busca_vetorial import IndiceEmMemoria  # noqa: E402
from app.db.models import EMBED_DIM  # noqa: E402
from ingestao.carregar_banco import gravar, preparar_banco  # noqa: E402
from ingestao.gerar_chunks import gerar_chunks  # noqa: E402
from ingestao.processar_bula import montar_bula  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def engine():
    url = get_settings().database_url
    schema = f"teste_{uuid4().hex[:12]}"
    try:
        with create_engine(url).begin() as conexao:
            conexao.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conexao.execute(text(f"CREATE SCHEMA {schema}"))
    except OperationalError:
        pytest.skip("PostgreSQL não responde (docker compose up -d db)")
    # "public" no fim do search_path para achar o tipo vector, que a extensão criou lá.
    eng = create_engine(url, connect_args={"options": f"-csearch_path={schema},public"})
    preparar_banco(eng)
    yield eng
    eng.dispose()
    with create_engine(url).begin() as conexao:
        conexao.execute(text(f"DROP SCHEMA {schema} CASCADE"))


@pytest.fixture(scope="module")
def dados():
    bulas = [
        montar_bula(
            FIXTURES / "bula_exemplo.txt", nome, ativo, "Laboratório Teste", date(2025, 3, 15), [nome.split()[0]]
        )
        for nome, ativo in [("Exemplol 50 mg", "maleato de exemplina"), ("Outrol 10 mg", "Ácido Outrínico")]
    ]
    chunks = [c for b in bulas for c in gerar_chunks(b, 300, 50)]
    matriz = np.random.default_rng(42).normal(size=(len(chunks), EMBED_DIM)).astype("<f4")
    return bulas, chunks, matriz


@pytest.fixture(scope="module")
def buscas(engine, dados):
    bulas, chunks, matriz = dados
    gravar(engine, bulas, chunks, matriz)
    memoria = IndiceEmMemoria([c.chunk_id for c in chunks], [c.principio_ativo for c in chunks], matriz)
    return memoria, BuscaPgvector(engine)


def _iguais(a, b):
    assert [r.chunk_id for r in a] == [r.chunk_id for r in b]
    assert [r.score for r in a] == pytest.approx([r.score for r in b], abs=1e-5)


@pytest.mark.parametrize(
    "filtro", [None, ["maleato de exemplina"], ["acido outrinico"], ["Maleato de Exemplina", "ácido outrínico"]]
)
def test_mesmo_resultado_que_a_busca_em_memoria(buscas, dados, filtro):
    memoria, pg = buscas
    _, chunks, matriz = dados
    for i in (0, len(chunks) - 1):  # a "pergunta" é um trecho com um pouco de ruído
        q = matriz[i] + 0.1 * np.random.default_rng(i).normal(size=EMBED_DIM)
        _iguais(pg.buscar(q, 3, filtro), memoria.buscar(q, 3, filtro))


def test_casos_vazios(buscas):
    _, pg = buscas
    assert pg.buscar(np.zeros(EMBED_DIM), 3) == []
    assert pg.buscar(np.ones(EMBED_DIM), 0) == []
    assert pg.buscar(np.ones(EMBED_DIM), 3, ["inexistente"]) == []


def test_recarregar_substitui_o_acervo(engine, dados, buscas):
    bulas, chunks, matriz = dados
    gravar(engine, bulas[:1], chunks[: len(chunks) // 2], matriz[: len(chunks) // 2])
    with engine.connect() as conexao:
        assert conexao.execute(text("SELECT count(*) FROM bulas")).scalar() == 1
        assert conexao.execute(text("SELECT count(*) FROM chunks")).scalar() == len(chunks) // 2
    gravar(engine, bulas, chunks, matriz)  # deixa como os outros testes esperam


def test_matriz_com_dimensao_errada(engine, dados):
    bulas, chunks, _ = dados
    with pytest.raises(ValueError, match="esperado"):
        gravar(engine, bulas, chunks, np.zeros((len(chunks), 3), dtype="<f4"))
