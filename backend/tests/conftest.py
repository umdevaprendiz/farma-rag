from uuid import uuid4

import pytest


@pytest.fixture(scope="module")
def engine():
    """Engine num schema temporário do banco do .env, um por módulo de teste, apagado no fim: não
    mexe nas tabelas do banco de desenvolvimento.

    Pula o módulo se o PostgreSQL não responder (suba com: docker compose up -d db). Com o
    DB_TIMEOUT_S, isso leva segundos, não os ~130 s que o Windows espera sem limite.
    """
    pytest.importorskip("pgvector")
    pytest.importorskip("psycopg")
    from sqlalchemy import text
    from sqlalchemy.exc import OperationalError

    from app.config import get_settings
    from app.db.connection import criar_engine, preparar_banco

    s = get_settings()
    schema = f"teste_{uuid4().hex[:12]}"
    admin = criar_engine(s)
    try:
        with admin.begin() as conexao:
            conexao.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conexao.execute(text(f"CREATE SCHEMA {schema}"))
    except OperationalError:
        admin.dispose()
        pytest.skip("PostgreSQL não responde (docker compose up -d db)")
    eng = criar_engine(s, schema)
    preparar_banco(eng)
    yield eng
    eng.dispose()
    with admin.begin() as conexao:
        conexao.execute(text(f"DROP SCHEMA {schema} CASCADE"))
    admin.dispose()
