"""Modelos SQLAlchemy para PostgreSQL 17 + pgvector.

- listas em ARRAY(Text), UUID nativo, datas com fuso (timestamptz, gravadas em UTC);
- o embedding é `vector(EMBED_DIM)` e a similaridade é calculada no banco (ver busca_pgvector.py);
- antes de criar as tabelas é preciso `CREATE EXTENSION vector` (db/conexao.py, preparar_banco);
- coluna nova em tabela que já existe também entra em `preparar_banco` (create_all não altera tabela).
"""

from datetime import date, datetime, timezone
from uuid import UUID, uuid4

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, Boolean, Date, DateTime, Float, ForeignKey, Integer, Text, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Dimensão do bge-m3. Precisa bater com EMBED_DIM do .env (carregar_banco.py confere):
# trocar de modelo de embedding exige recriar a tabela chunks.
EMBED_DIM = 1024


def _agora_utc() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Bula(Base):
    """Espelho de data/processed/bulas/<id>.json. Recriada por carregar_banco.py."""

    __tablename__ = "bulas"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    medicamento: Mapped[str] = mapped_column(Text)
    principio_ativo: Mapped[str] = mapped_column(Text, index=True)
    nomes_comerciais: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    fabricante: Mapped[str] = mapped_column(Text)
    data_versao: Mapped[date] = mapped_column(Date)
    fonte_pdf: Mapped[str] = mapped_column(Text)
    numero_registro: Mapped[str | None] = mapped_column(Text)
    forma_farmaceutica: Mapped[str | None] = mapped_column(Text)
    versao: Mapped[str | None] = mapped_column(Text)


class Chunk(Base):
    """Espelho de data/processed/chunks.jsonl. Recriada por carregar_banco.py."""

    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(Text, primary_key=True)
    bula_id: Mapped[str] = mapped_column(ForeignKey("bulas.id"), index=True)
    # Como no chunks.jsonl: o prompt e a fundamentação usam o nome do medicamento do trecho.
    medicamento: Mapped[str] = mapped_column(Text)
    # Denormalizado para filtrar a busca por princípio ativo sem JOIN.
    principio_ativo: Mapped[str] = mapped_column(Text)
    # normalizar(principio_ativo): o filtro da busca compara assim, igual ao IndiceEmMemoria.
    principio_ativo_norm: Mapped[str] = mapped_column(Text, index=True)
    secao: Mapped[str] = mapped_column(Text, index=True)
    pagina: Mapped[int] = mapped_column(Integer)
    ordem: Mapped[int] = mapped_column(Integer)
    texto: Mapped[str] = mapped_column(Text)
    n_tokens: Mapped[int] = mapped_column(Integer)
    # Sem índice HNSW: ver busca_pgvector.py.
    embedding: Mapped[np.ndarray] = mapped_column(Vector(EMBED_DIM))


class Consulta(Base):
    __tablename__ = "consultas"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    pergunta: Mapped[str] = mapped_column(Text)
    principios_ativos_filtro: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    # Os usados na busca (detectados, herdados da anterior ou do filtro): a pergunta seguinte da
    # conversa herda daqui ("e para criança?"). Nulo nas consultas gravadas antes da coluna.
    principios_ativos: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    resposta: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Text)  # respondida | nao_encontrada | precisa_esclarecimento | erro
    motivo: Mapped[str | None] = mapped_column(Text, nullable=True)
    codigo_erro: Mapped[str | None] = mapped_column(Text, nullable=True)
    # A mensagem que o erro levou (no 502, o que cada rodada tentou).
    mensagem_erro: Mapped[str | None] = mapped_column(Text, nullable=True)
    modelo: Mapped[str] = mapped_column(Text)
    latencia_ms: Mapped[int] = mapped_column(Integer)
    rodadas: Mapped[int] = mapped_column(Integer, default=0)
    # rag.Tentativa de cada rodada: trechos enviados, saída crua do modelo e por que passou ou não.
    tentativas: Mapped[list[dict]] = mapped_column(JSONB, default=list)
    # A pergunta anterior da mesma conversa (pergunta seguinte, ou resposta a um esclarecimento).
    # Sem FK, como o chunk_id das fontes: o histórico não pode quebrar se uma consulta antiga for apagada.
    consulta_origem_id: Mapped[UUID | None] = mapped_column(Uuid, nullable=True)
    criado_em: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_agora_utc, index=True)

    fontes: Mapped[list["ConsultaFonte"]] = relationship(
        back_populates="consulta", cascade="all, delete-orphan", order_by="ConsultaFonte.posicao"
    )


class ConsultaFonte(Base):
    """Fonte usada numa resposta, guardada como SNAPSHOT.

    Não há FK para `chunks`: essa tabela é recriada quando o dataset muda, e o histórico
    precisa continuar mostrando exatamente o texto e a versão da bula que o profissional viu.
    """

    __tablename__ = "consulta_fontes"

    consulta_id: Mapped[UUID] = mapped_column(
        ForeignKey("consultas.id", ondelete="CASCADE"), primary_key=True
    )
    posicao: Mapped[int] = mapped_column(Integer, primary_key=True)  # [1], [2]...
    chunk_id: Mapped[str] = mapped_column(Text)  # texto simples, sem FK
    medicamento: Mapped[str] = mapped_column(Text)
    # Da bula, no momento da consulta. Nulos só nas bulas de teste, que não têm registro.
    fabricante: Mapped[str | None] = mapped_column(Text, nullable=True)
    numero_registro: Mapped[str | None] = mapped_column(Text, nullable=True)
    secao: Mapped[str] = mapped_column(Text)
    pagina: Mapped[int] = mapped_column(Integer)
    data_versao: Mapped[date] = mapped_column(Date)
    trecho: Mapped[str] = mapped_column(Text)
    score: Mapped[float] = mapped_column(Float)
    citado: Mapped[bool] = mapped_column(Boolean, default=False)
    # Partes do trecho que sustentam a resposta, como posições início/fim. Coluna criada já
    # (decisão de 2026-10-15), preenchida quando entrar o destaque na fonte; até lá, nula.
    destaques: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    consulta: Mapped[Consulta] = relationship(back_populates="fontes")
