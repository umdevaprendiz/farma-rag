"""PgvectorSearch, carregar_banco e Corpus.do_banco contra um PostgreSQL de verdade.

Pulado se o pgvector/psycopg não estiver instalado ou se o banco do .env não responder
(suba com: docker compose up -d db). Roda num schema temporário (fixture `engine`, conftest.py).
"""

from datetime import date
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pgvector")
pytest.importorskip("psycopg")

from sqlalchemy import text  # noqa: E402

from app.config import Settings  # noqa: E402
from app.db.connection import preparar_banco  # noqa: E402
from app.db.models import EMBED_DIM  # noqa: E402
from app.db.pgvector_search import PgvectorSearch  # noqa: E402
from app.db.vector_search import InMemoryIndex  # noqa: E402
from app.services.corpus import Corpus, IncompleteCorpusError  # noqa: E402
from ingestion.generate_chunks import gerar_chunks  # noqa: E402
from ingestion.load_database import gravar  # noqa: E402
from ingestion.process_leaflet import montar_bula  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def dados():
    bulas = [
        montar_bula(
            FIXTURES / "sample_leaflet.txt", nome, ativo, "Laboratório Teste", date(2025, 3, 15), [nome.split()[0]]
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
    memoria = InMemoryIndex([c.chunk_id for c in chunks], [c.principio_ativo for c in chunks], matriz)
    return memoria, PgvectorSearch(engine)


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


def test_acervo_do_banco_tem_os_mesmos_trechos_dos_arquivos(engine, dados, buscas):
    bulas, chunks, _ = dados
    acervo = Corpus.do_banco(engine, Settings(_env_file=None))
    assert list(acervo.chunks.values()) == sorted(chunks, key=lambda c: c.chunk_id)
    assert sorted(acervo.bulas) == sorted(b.id for b in bulas)
    b = acervo.bulas[bulas[0].id]
    assert (b.medicamento, b.fabricante, b.data_versao) == (bulas[0].medicamento, "Laboratório Teste", date(2025, 3, 15))
    assert isinstance(acervo.indice, PgvectorSearch)
    assert acervo.catalogo.indexados["exemplol"] == "maleato de exemplina"


def test_acervo_do_banco_vazio_ou_antigo(engine, dados, buscas):
    gravar(engine, [], [], np.empty((0, EMBED_DIM), dtype="<f4"))
    with pytest.raises(IncompleteCorpusError, match="não tem bulas"):
        Corpus.do_banco(engine, Settings(_env_file=None))

    gravar(engine, *dados)
    with engine.begin() as conexao:  # como um banco criado antes da coluna
        conexao.execute(text("ALTER TABLE chunks DROP COLUMN medicamento"))
    preparar_banco(engine)  # devolve a coluna, vazia nos trechos que já estavam lá
    with pytest.raises(IncompleteCorpusError, match="Rode de novo"):
        Corpus.do_banco(engine, Settings(_env_file=None))
    gravar(engine, *dados)  # deixa como os outros testes esperam


def test_tabelas_ficam_no_schema_temporario(engine):
    """O pytest não pode tocar no banco de desenvolvimento (public): ver criar_engine."""
    with engine.connect() as conexao:
        schema = conexao.execute(text("SELECT current_schema()")).scalar()
        tabelas = set(conexao.execute(
            text("SELECT table_name FROM information_schema.tables WHERE table_schema = :s"), {"s": schema}
        ).scalars())
    assert schema.startswith("teste_")
    assert {"bulas", "chunks", "consultas", "consulta_fontes"} <= tabelas
