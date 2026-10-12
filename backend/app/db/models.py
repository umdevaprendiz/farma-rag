"""Modelos SQLAlchemy para MySQL 8.

Escolhas por causa do MySQL (o PDF previa PostgreSQL + pgvector):
- sem ARRAY/UUID/vector nativos: listas viram JSON, UUID usa o tipo genérico `Uuid`
  (CHAR(32)) e o embedding é guardado como BLOB de float32 (ver busca_vetorial.py);
- chaves e índices em TEXT exigem tamanho no MySQL, então usamos String(n);
- DATETIME não guarda fuso: gravamos sempre em UTC.
"""

from datetime import date, datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _agora_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Bula(Base):
    """Espelho de data/processed/bulas/<id>.json. Recriada por carregar_banco.py."""

    __tablename__ = "bulas"

    id: Mapped[str] = mapped_column(String(200), primary_key=True)
    medicamento: Mapped[str] = mapped_column(String(255))
    principio_ativo: Mapped[str] = mapped_column(String(255), index=True)
    nomes_comerciais: Mapped[list[str]] = mapped_column(JSON, default=list)
    fabricante: Mapped[str] = mapped_column(String(255))
    data_versao: Mapped[date] = mapped_column(Date)
    fonte_pdf: Mapped[str] = mapped_column(String(500))


class Chunk(Base):
    """Espelho de data/processed/chunks.jsonl. Recriada por carregar_banco.py."""

    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    bula_id: Mapped[str] = mapped_column(ForeignKey("bulas.id"), index=True)
    # Denormalizado para filtrar a busca por princípio ativo sem JOIN.
    principio_ativo: Mapped[str] = mapped_column(String(255), index=True)
    secao: Mapped[str] = mapped_column(String(50), index=True)
    pagina: Mapped[int] = mapped_column(Integer)
    ordem: Mapped[int] = mapped_column(Integer)
    texto: Mapped[str] = mapped_column(Text)
    # float32 little-endian, EMBED_DIM valores (4 KB para 1024). Ver busca_vetorial.py.
    embedding: Mapped[bytes] = mapped_column(LargeBinary)


class Consulta(Base):
    __tablename__ = "consultas"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    pergunta: Mapped[str] = mapped_column(Text)
    principios_ativos_filtro: Mapped[list[str]] = mapped_column(JSON, default=list)
    resposta: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20))  # respondida | nao_encontrada | erro
    motivo: Mapped[str | None] = mapped_column(String(40), nullable=True)
    codigo_erro: Mapped[str | None] = mapped_column(String(40), nullable=True)
    modelo: Mapped[str] = mapped_column(String(100))
    latencia_ms: Mapped[int] = mapped_column(Integer)
    criado_em: Mapped[datetime] = mapped_column(DateTime, default=_agora_utc, index=True)  # UTC

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
    chunk_id: Mapped[str] = mapped_column(String(255))  # texto simples, sem FK
    medicamento: Mapped[str] = mapped_column(String(255))
    secao: Mapped[str] = mapped_column(String(50))
    pagina: Mapped[int] = mapped_column(Integer)
    data_versao: Mapped[date] = mapped_column(Date)
    trecho: Mapped[str] = mapped_column(Text)
    score: Mapped[float] = mapped_column(Float)
    citado: Mapped[bool] = mapped_column(Boolean, default=False)

    consulta: Mapped[Consulta] = relationship(back_populates="fontes")
