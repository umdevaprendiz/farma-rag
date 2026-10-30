# Segurança do FarmaRAG

Análise de 2026-10-29, feita sobre o código, os containers e as dependências do repositório, com correção do
que se achou. As decisões estão em [`DECISOES.md`](DECISOES.md) §44; aqui ficam o modelo de ameaça, o que foi
verificado (e como repetir), os achados e os riscos que ficaram de propósito.

## Modelo de ameaça

O FarmaRAG é um assistente **local**, para uma pessoa no balcão, na própria máquina: tela, API, banco e Ollama
respondem só em `127.0.0.1`, sem login, e nada da pergunta sai da máquina (a única chamada de rede que o código
faz é ao Ollama). Os dados são as bulas (públicas) e o **histórico de perguntas**, que pode conter, mesmo sem
querer, algo sensível digitado em texto livre.

O pior dano aqui não é o de um sistema comum: é uma **resposta de saúde errada**. Por isso a defesa mais
importante é a do projeto inteiro (citação verificada, fundamentação, recusa), e não só a de rede.

| Ameaça | Controle |
|---|---|
| Alguém da rede acessando o sistema | Todas as portas publicadas só em `127.0.0.1` (tela, API e banco) |
| Entrada maliciosa (SQL, HTML, comando, prompt) | Pydantic com limites; SQLAlchemy parametrizado; React escapa o texto; sem `eval`/shell; a saída do modelo é validada e conferida contra a bula (inclui a regra 6: "pode ser usado" onde a bula proíbe é reprovado, DECISOES §45) |
| Dependência ou imagem comprometida | Versões exatas (`requirements.lock`, `package.json`), imagens por digest, `ignore-scripts`, auditoria e scan |
| Segredo ou dado pessoal no repositório público | `.env` ignorado, varredura do que vai ao Git, `.dockerignore` |
| Abuso (muitas perguntas) | Teto de perguntas em andamento (429) e `client_max_body_size` no nginx |
| Dado de paciente no histórico | Aviso na tela, histórico sem nome por desenho, como apagar (abaixo) |

## O que foi verificado

| Verificação | Resultado |
|---|---|
| Varredura de chaves, tokens e senhas nos arquivos que vão ao Git (~180) | Nenhum. O `.env` nunca esteve no histórico |
| Caminhos locais, e-mails, CPF e telefone fora dos dados de bula | Nenhum; metadados do MP4 e do JPG da demonstração limpos |
| `eval`, `exec`, `subprocess`, `pickle`, `shell=True` no código | Nenhum |
| SQL montada com texto | Uma (migração de colunas, com constantes do próprio código, nunca entrada do usuário) |
| `dangerouslySetInnerHTML`, `innerHTML`, links externos e `localStorage` na tela | Nenhum |
| Chamadas de saída do backend | Só o Ollama (`OLLAMA_URL`) |
| `npm audit` (frontend, com dev) | 0 vulnerabilidades |
| `pip-audit` 2.10.1 sobre `requirements.lock` | 1 (`certifi`), corrigida: 0 |
| `docker scout` nas duas imagens | 0 críticas e 0 altas **com correção disponível** (ver riscos aceitos) |
| Containers em execução | Backend uid 10001 e nginx uid 101, raiz somente leitura, sem capabilities, `no-new-privileges`; escrita em `/` falha |
| Tela com a CSP ativa | Monta, responde uma pergunta de verdade e não gera nenhuma violação no console |
| nginx | `/.env` dá 404; corpo de 20 KB dá 413 |
| Clone novo (pasta limpa, `.env` do exemplo, banco novo) | Banco, carga de 37 bulas e 1.384 trechos, API e tela saudáveis, só com o que vai ao Git |
| Testes | 524 passam com o banco ligado, 0 pulados |

## Achados e correções

| # | Achado | Gravidade | Correção |
|---|---|---|---|
| 1 | **Cadeia de suprimentos sem versão fixa.** `requirements.txt` usava `>=` e as imagens eram escolhidas por tag: cada build podia trazer versão nova, nunca testada | Média | `backend/requirements.lock` (54 pacotes exatos, gerado por `backend/lock_deps.py`); as três imagens por **digest** |
| 2 | **Containers como root, com disco gravável e todas as capabilities** | Média | Backend sem root (uid 10001), nginx sem privilégios (`nginx-unprivileged`, uid 101), raiz somente leitura (só `/tmp` em memória), `cap_drop: ALL`, `no-new-privileges`, teto de memória e de processos, healthchecks, reinício automático |
| 3 | **Tela sem Content-Security-Policy** (só tinha 3 cabeçalhos) | Média | CSP restritiva (`default-src 'none'`, scripts e estilos só `'self'`, `connect-src 'self'`), `Permissions-Policy`, COOP e CORP. A CSP é a segunda barreira: o texto do modelo já é renderizado como texto |
| 4 | **Sem limite de perguntas simultâneas.** Cada uma segura uma thread e o Ollama por 30 a 90 s; abas demais empilhavam chamadas até todas estourarem | Média | `PERGUNTAS_SIMULTANEAS` (padrão 2): acima disso, 429 `SERVIDOR_OCUPADO` na hora, sem gravar consulta, e a vaga volta mesmo se a pergunta termina em erro |
| 5 | **Respostas da API cacheáveis**, sem `nosniff` | Baixa | Middleware ASGI (`api/security.py`): `X-Content-Type-Options: nosniff` e `Cache-Control: no-store` em `/api/` (ASGI puro, para não travar o fluxo NDJSON) |
| 6 | **Senha de exemplo aceita em silêncio** (`senha`, `troque-esta-senha`...) | Baixa | A API avisa no log ao subir; o Compose já exigia `DB_PASSWORD` |
| 7 | **Entrada sem limite e URL sem validação:** itens de `principios_ativos` sem tamanho máximo; `OLLAMA_URL` aceitava `file://` e `ftp://` (o `urllib` abre os dois) | Baixa | Limite de 200 caracteres por item (422); só `http://` e `https://` |
| 8 | **`certifi` 2026.7.22 vulnerável** (GHSA-6q48-6c7w-9gw2, validação incorreta de certificado, CVSS 9.1) | Alta no papel; baixa na prática (o cliente do Ollama usa HTTP puro) | Atualizada para 2026.10.19 |
| 9 | **Pacotes do Alpine da imagem do nginx com falha já corrigida**, entre elas `brotli` (CVE-2026-111497, na lista de explorados da CISA), `curl` e `zlib` | Alta | `apk upgrade` na construção; `curl` removido (a tela não o usa) |
| 10 | **Dado de paciente em texto livre** | Média | Aviso na caixa de pergunta ("Não escreva nome nem dado de paciente"); o histórico guarda só pergunta e resposta |

Também fechado (não é segurança, mas apareceu na mesma conferência): o `/saude` tratava "o Ollama demorou a
responder" como "fora do ar" e mostrava um falso alarme durante toda pergunta longa; agora estouro de tempo
conta como "ocupado" (§44).

## Riscos que ficaram de propósito

| Risco | Por que ficou | Se for expor fora da máquina |
|---|---|---|
| **Sem autenticação** | O controle é o bind em `127.0.0.1` | Proxy reverso com TLS, login e limite de taxa; não publicar portas fora do loopback |
| **HTTP, sem TLS** | Tráfego só no loopback | TLS no proxy (HSTS só depois) |
| **zlib (Debian) CVE-2026-85091 e libxml2 (Alpine) CVE-2026-86140** | Sem versão corrigida publicada. O libxml2 só é usado pelo módulo XSLT do nginx, que não está configurado | Refazer `docker scout` ao trocar os digests e quando as correções saírem |
| **Mensagens de erro mostram `host:porta` e a URL do Ollama** | Ajuda a diagnosticar numa máquina só | Trocar por mensagens genéricas e deixar o detalhe só no log |
| **O app conecta ao PostgreSQL como o usuário dono (superusuário da imagem)** | A criação da extensão `vector` e das tabelas pede isso | Criar um papel sem superusuário para a API, depois que o administrador criar a extensão |
| **Ollama sem autenticação na 11434 do host** | Escuta só em `127.0.0.1` por padrão | Nunca definir `OLLAMA_HOST=0.0.0.0` |
| **Sem retenção do histórico** | Decisão de produto ainda em aberto | Ver "Apagar o histórico" abaixo |
| **PDF malicioso na ingestão** | A ingestão roda só na máquina de quem mantém, em PDFs baixados à mão de fonte oficial, com Defender e `validate_leaflet_pdf.py` antes; o parser é Python puro | Nunca ingerir PDF de origem não verificada |
| **Prompt injection** | As bulas são conteúdo confiável (procedência em `data/FONTES.md`); a pergunta não é, mas o modelo só pode devolver afirmações `{texto, trecho}` que passam por citação válida, fundamentação (números, termos, negação, remédio certo) e conferência. O efeito mais provável de uma injeção é "não encontrei" | Manter essas regras antes de qualquer mudança no prompt |
| **Sem hashes nas dependências, sem assinatura de imagem, sem scan em CI** | Versões e digests exatos já cobrem o essencial | `pip install --require-hashes` (via `pip-compile`), `docker scout` ou Trivy no GitHub Actions, e atualização mensal dos digests |

## Apagar o histórico (LGPD)

O histórico não tem nome nem identificador de paciente por desenho, mas texto livre pode trazer algo. Para
apagar o que for antigo (as fontes saem junto, em cascata):

```powershell
docker compose exec db psql -U $env:DB_USER -d $env:DB_NAME -c "DELETE FROM consultas WHERE criado_em < now() - interval '90 days'"
```

Para apagar tudo, inclusive o banco: `docker compose down -v` (o acervo volta com `docker compose run --rm carga`).

## Como repetir as verificações

```powershell
# Dependências
cd frontend; npm.cmd audit; cd ..
python -m venv $env:TEMP\audit; & $env:TEMP\audit\Scripts\pip install pip-audit
& $env:TEMP\audit\Scripts\pip-audit -r backend\requirements.lock --no-deps --disable-pip

# Imagens (Docker Desktop traz o Scout)
docker compose build
docker scout quickview local://farma-rag-backend:latest
docker scout quickview local://farma-rag-frontend:latest

# Containers: usuário, disco e capabilities
docker exec farma-rag-backend-1 id
docker inspect farma-rag-backend-1 --format 'readonly={{.HostConfig.ReadonlyRootfs}} cap_drop={{.HostConfig.CapDrop}}'

# Regenerar o lock depois de atualizar uma dependência (com a .venv do projeto)
python backend\lock_deps.py
```

Antes de trocar o digest de uma imagem base, confira no Docker Hub que a tag é da conta oficial
(`library/...` ou `nginxinc/...`), copie o digest e rode `docker scout` de novo.
