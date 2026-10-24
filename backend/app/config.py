from functools import lru_cache
from urllib.parse import quote_plus

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Lê o .env da raiz do repositório. Nomes das variáveis não diferenciam maiúsculas."""

    # env_ignore_empty: "LIMIAR_SIMILARIDADE=" vazio no .env vale "não definido", não erro.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)

    ollama_url: str = "http://localhost:11434"
    llm_modelo: str = "qwen2.5:1.5b"
    # Modelo do nó verificar_llm (só roda na rodada 3). Vazio = o mesmo de LLM_MODELO.
    # Permite um modelo maior só onde ele pesa pouco no tempo total.
    llm_modelo_verificacao: str | None = None
    embed_modelo: str = "bge-m3"
    embed_dim: int = 1024
    # F3c (docs/DECISOES.md, seção 35). Quanto tempo o Ollama mantém cada modelo na memória depois do
    # último uso: "30m", "2h", segundos ("300") ou "-1" (para sempre). Recarregar custa vários
    # segundos na CPU.
    keep_alive: str = "30m"
    # Ao subir, a API carrega no Ollama os modelos que a pergunta usa, em segundo plano.
    aquecer_modelos: bool = True

    # PostgreSQL. O banco é montado a partir de partes: DB_HOST é "127.0.0.1" nos scripts
    # rodados no host (ingestão, benchmark) e "db" dentro do Docker Compose.
    db_user: str = "farmarag"
    db_password: str = "senha"
    db_name: str = "farmarag"
    db_host: str = "127.0.0.1"  # não "localhost": ver .env.example
    db_port: int = 5432
    # Sem limite, conectar ao banco desligado espera ~130 s no Windows antes de falhar.
    db_timeout_s: int = 5

    top_k: int = 3
    # Reranking (services/rerank.py): a rodada 1 olha os N mais parecidos por remédio e escolhe os
    # top_k pela seção da pergunta, pelas palavras dela e sem repetir trecho. 0 = só semelhança.
    rerank_candidatos: int = 10
    # F3b (docs/DECISOES.md, seção 34): a seção que a pergunta indica já na rodada 1 (um trecho dela por
    # remédio) e as seções "resultados de eficácia" e "características farmacológicas" só na frente
    # quando pedidas. False = o reranking da seção 30 sozinho (a linha de base do benchmark).
    rerank_garante_secao: bool = True
    # Pergunta sem medicamento indexado: busca em todas as bulas, um trecho por princípio ativo.
    # Cada trecho é uma opção na resposta ("melhor remédio pra dor de cabeça").
    top_k_geral: int = 4
    limiar_similaridade: float | None = None
    chunk_tokens: int = 300
    chunk_sobreposicao: int = 50

    temperatura: float = 0.1
    # Semente da geração (None = sorteada a cada chamada). O benchmark completo fixa uma: o mesmo
    # prompt dá a mesma resposta, e a comparação entre configurações mede só a configuração.
    llm_semente: int | None = None
    num_ctx: int = 4096
    num_predict: int = 250
    timeout_llm_s: int = 120

    # Rodadas de verificação (docs/DECISOES.md, seção 14). Uma nova rodada só começa se a
    # anterior falhou e se ainda cabe no orçamento de tempo da pergunta.
    rodadas_max: int = 3
    orcamento_s: int = 150
    # Fração mínima dos termos de cada afirmação que precisa aparecer no trecho citado.
    fundamentacao_min_termos: float = 0.6

    cors_origens: str = "http://localhost:5173"
    medicamentos_conhecidos: str = "data/medicamentos_conhecidos.txt"
    catalogo_anvisa: str = "data/processed/catalogo_anvisa.jsonl"
    aliases_ignorados: str = "data/aliases_ignorados.txt"

    # Dataset em arquivo (fonte da verdade) e cache de embeddings (fora do Git).
    bulas_dir: str = "data/processed/bulas"
    chunks_arquivo: str = "data/processed/chunks.jsonl"
    cache_embeddings: str = "data/cache/embeddings.sqlite"

    @property
    def database_url(self) -> str:
        senha = quote_plus(self.db_password)
        return (
            f"postgresql+psycopg://{self.db_user}:{senha}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def cors_origens_lista(self) -> list[str]:
        return [o.strip() for o in self.cors_origens.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
