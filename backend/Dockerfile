# API (docs/DECISOES.md §38 e §44). Endurecida: imagem base fixada por digest, dependências com versão
# exata (requirements.lock), processo sem privilégios e healthcheck.
#
# Imagem oficial "library/python:3.12-slim", verificada no Docker Hub em 2026-10-29. Para atualizar, troque
# o digest de propósito (docker buildx imagetools inspect python:3.12-slim) e rode os testes.
FROM python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159

# Sem .pyc (o sistema de arquivos do container é somente leitura em produção), log sem buffer, pip sem cache.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Usuário e grupo de sistema, sem shell e sem home: o processo da API nunca roda como root.
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

WORKDIR /srv
# Primeiro só o que define as dependências: muda pouco, e o Docker reaproveita a camada do pip.
# requirements.lock = versões exatas (gerado por lock_deps.py), não o requirements.txt com ">=".
COPY requirements.lock .
RUN pip install -r requirements.lock
COPY app ./app

USER 10001:10001
EXPOSE 8000

# Vivo = o servidor responde. Não usa o /saude: ele sonda o banco e o Ollama, e uma pergunta longa (CPU
# ocupada) o deixaria lento sem que o container estivesse com defeito.
HEALTHCHECK --interval=30s --timeout=8s --start-period=60s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/docs', timeout=6).read()"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
