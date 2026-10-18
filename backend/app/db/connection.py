"""Conexão com o PostgreSQL e preparo do esquema, para a API e para a ingestão.

Fica em `app` (e não em `ingestion/`) porque a imagem Docker da API só leva o pacote `app`, e a API
também precisa preparar o banco ao subir: as tabelas do histórico podem ainda não existir.
"""

from sqlalchemy import Engine, create_engine, text

from app.config import Settings
from app.db import models

# Colunas que entraram depois da primeira versão de cada tabela: create_all não altera tabela que
# já existe, então quem criou o banco antes recebe as colunas aqui (sem Alembic: DECISOES §11).
_COLUNAS_NOVAS = (
    ("bulas", "numero_registro", "text"),
    ("bulas", "forma_farmaceutica", "text"),
    ("bulas", "versao", "text"),
    # A API monta os trechos a partir do banco, sem ler chunks.jsonl.
    ("chunks", "medicamento", "text"),
    ("chunks", "n_tokens", "integer"),
    ("consultas", "rodadas", "integer NOT NULL DEFAULT 0"),
    ("consultas", "tentativas", "jsonb"),
    ("consultas", "consulta_origem_id", "uuid"),
    ("consultas", "mensagem_erro", "text"),
    ("consultas", "principios_ativos", "text[]"),
    ("consulta_fontes", "fabricante", "text"),
    ("consulta_fontes", "numero_registro", "text"),
    ("consulta_fontes", "destaques", "jsonb"),
    ("consultas", "conclusao", "text"),
)


def criar_engine(s: Settings, schema: str | None = None) -> Engine:
    """`schema`: usar esse schema em vez do public (os testes criam um temporário)."""
    args: dict = {"connect_timeout": s.db_timeout_s}
    if schema:
        # "public" no fim do search_path para achar o tipo vector, que a extensão criou lá. Vale
        # para o SQL escrito à mão (TRUNCATE, ALTER TABLE), depois que as tabelas existem no schema.
        args["options"] = f"-csearch_path={schema},public"
    # pre_ping: a API fica no ar por dias; se o banco reiniciar, a conexão velha é trocada.
    engine = create_engine(s.database_url, pool_pre_ping=True, connect_args=args)
    if schema:
        # O search_path sozinho não isola: o create_all vê as tabelas do public pelo caminho, não
        # cria as do schema, e tudo cai no banco de desenvolvimento (aconteceu até 2026-10-16:
        # o pytest regravava bulas, chunks e consultas do public). Com o mapa, o create_all e
        # todo comando do SQLAlchemy usam o schema pelo nome.
        engine = engine.execution_options(schema_translate_map={None: schema})
    return engine


def preparar_banco(engine: Engine) -> None:
    """Cria a extensão vector, as tabelas que faltarem e as colunas novas. Pode rodar sempre."""
    with engine.begin() as conexao:
        conexao.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    models.Base.metadata.create_all(engine)
    with engine.begin() as conexao:
        for tabela, coluna, tipo in _COLUNAS_NOVAS:
            conexao.execute(text(f"ALTER TABLE {tabela} ADD COLUMN IF NOT EXISTS {coluna} {tipo}"))
