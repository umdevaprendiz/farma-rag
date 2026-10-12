from functools import lru_cache
from urllib.parse import quote_plus

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Lê o .env da raiz do repositório. Nomes das variáveis não diferenciam maiúsculas."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_url: str = "http://localhost:11434"
    llm_modelo: str = "qwen2.5:1.5b"
    embed_modelo: str = "bge-m3"
    embed_dim: int = 1024

    # MySQL. O banco é montado a partir de partes: DB_HOST é "localhost" nos scripts
    # rodados no host (ingestão, benchmark) e "db" dentro do Docker Compose.
    db_user: str = "farmarag"
    db_password: str = "senha"
    db_name: str = "farmarag"
    db_host: str = "localhost"
    db_port: int = 3306

    top_k: int = 4
    limiar_similaridade: float | None = None
    chunk_tokens: int = 300
    chunk_sobreposicao: int = 50

    temperatura: float = 0.1
    num_ctx: int = 4096
    num_predict: int = 250
    timeout_llm_s: int = 90

    cors_origens: str = "http://localhost:5173"
    medicamentos_conhecidos: str = "data/medicamentos_conhecidos.txt"

    @property
    def database_url(self) -> str:
        senha = quote_plus(self.db_password)
        return (
            f"mysql+pymysql://{self.db_user}:{senha}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}?charset=utf8mb4"
        )

    @property
    def cors_origens_lista(self) -> list[str]:
        return [o.strip() for o in self.cors_origens.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
