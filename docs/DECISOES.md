# Decisões de projeto (correções ao FarmaRAG.pdf)

O PDF é o ponto de partida. Onde este arquivo discorda dele, **este arquivo vale**.

## 1. Hardware: tudo na CPU
A máquina tem Intel UHD (sem NVIDIA), i5-10210U (4 núcleos) e ~16 GB de RAM.
- Modelos iniciais: 1–2B (`qwen2.5:1.5b`); 3B só como comparação. 7–8B fora do uso diário.
- Padrões: `TOP_K=3`, `NUM_PREDICT=250`, `TIMEOUT_LLM_S=120`.
- **Medido na etapa 0** (2026-10-11, CPU, prompt sem cache; `data/benchmark/resultados/etapa0_*.json`):
  - leitura do prompt ~50 tokens/s (qwen2.5:1.5b) e ~27 tokens/s (llama3.2:3b);
  - geração ~9 tokens/s (1.5b) e ~4 tokens/s (3b);
  - 3 trechos de ~330 tokens (~1250 tokens de prompt): ~25 s só para ler + 5–15 s para gerar
    = **30–45 s** com o 1.5b; o 3b passa de 90 s na primeira pergunta.
  - embeddings bge-m3: ~2,5–3,4 s por trecho de 300 tokens.
- A meta de 30 s **não fecha** com trechos de 300 tokens. Meta revisada: mediana < 60 s com o
  1.5b. Para baixar: trechos de ~200 tokens, resposta em streaming e `keep_alive` no Ollama.
- A leitura do prompt domina a latência, então `CHUNK_TOKENS` e `TOP_K` mandam mais que o modelo.
- Ingestão: ~4000 trechos x ~3 s = ~3 h de CPU, uma vez por configuração de trecho. Guardar o
  embedding em cache por hash do texto, para não refazer o que não mudou.

## 2. Gabarito do benchmark por âncora, não por `chunk_id`
`chunk_id` muda quando `CHUNK_TOKENS` muda, o que quebraria o experimento de 2 tamanhos de trecho.
Cada pergunta traz `ancoras: [{principio_ativo, secao, frase}]`; um trecho acerta se for daquela
seção/medicamento e contiver a frase (copiada da bula, curta, dentro de uma única frase).
Pergunta de interação tem **duas** âncoras e só conta se as duas forem recuperadas.
Código: `benchmark/metrics.py`; contrato: `BenchmarkQuestion` em `schemas/dataset.py`.

## 3. Pergunta com mais de um medicamento
Suportamos até 3 (`MAX_MEDICAMENTOS_POR_PERGUNTA`). A busca divide o `k` entre os medicamentos
(`InMemoryIndex.buscar`), então "ibuprofeno + losartana" recebe trechos das duas bulas.

## 4. Medicamento não indexado
`services/medicines.py` detecta, antes do LLM, nomes na pergunta (princípio ativo, nome
comercial, erro de digitação leve) contra o catálogo indexado e contra
`data/medicamentos_conhecidos.txt`:
- só não-indexados -> `nao_encontrada` com `motivo=medicamento_nao_indexado`, **sem chamar o LLM**;
- misturados -> segue com os indexados e devolve `medicamentos_nao_indexados` para o frontend avisar;
- nenhum reconhecido -> segue sem filtro e o limiar de similaridade decide.
Limite: só reconhece o que está nas duas listas. Mantenha `medicamentos_conhecidos.txt` crescendo.
**Mudou em 2026-10-16 (seção 20):** só não-indexados e nenhum reconhecido vão para a busca geral.

## 5. Filtro por princípio ativo
A API recebe `principios_ativos: list[str]`, não `bula_id`. Vários fabricantes têm bula do mesmo
princípio ativo, e o seletor do frontend já era por princípio ativo. Vazio = detectar na pergunta.

## 6. Histórico como snapshot, sem FK para `chunks`
`consulta_fontes` guarda cópia do trecho, medicamento, seção, página e `data_versao`.
`chunks` pode ser apagada e recriada à vontade sem quebrar o histórico.

## 7. Ingestão fora do Docker
Uma só `.env`, montada em partes: `DB_HOST=localhost` para scripts no host; o Compose
sobrescreve para `db` no container da API. A porta do banco é publicada em `127.0.0.1`.

## 8. Código compartilhado
`ingestion/` e `benchmark/` importam `app` (cliente do Ollama, schemas, config). Instale com
`pip install -e backend` e rode da raiz: `python -m ingestion.extract_pdf`.

## 9. Códigos de erro
Adicionado `SAIDA_INVALIDA` (502) para JSON quebrado; `CITACAO_INVALIDA` fica para citação
inexistente/incoerente. `BULA_NAO_ENCONTRADA` virou `MEDICAMENTO_NAO_ENCONTRADO`.

## 10. Limite conhecido da validação de citação
`services/citations.py` garante que as citações apontam para trechos enviados. Não garante que a
afirmação está sustentada: isso é a métrica de fidelidade (revisão manual) do benchmark.

## 11. Banco: PostgreSQL 17 + pgvector (2026-10-14)
Volta ao que o PDF previa. O MySQL ficou só no código (nunca foi instalado nem recebeu dados), então
não houve dado a migrar: o banco é preenchido a partir dos arquivos por `ingestion/load_database.py`.
- Container `pgvector/pgvector:pg17` (serviço `db` do `docker-compose.yml`, volume `pgdata`). No
  Windows nativo o pgvector precisa ser compilado com Visual Studio; o container evita isso.
- Driver `psycopg[binary]` 3 (`postgresql+psycopg://`) e o pacote `pgvector` para o tipo `vector`.
- Modelos: listas em `ARRAY(Text)`, UUID nativo, `timestamptz` em UTC, `Text` sem tamanho,
  `chunks.embedding` em `vector(1024)` (`EMBED_DIM` em `db/models.py`, conferido contra o `.env`).
- `chunks.principio_ativo_norm` = `normalizar(principio_ativo)`: o filtro da busca no banco compara
  igual ao `InMemoryIndex`.
- `PgvectorSearch` (`db/pgvector_search.py`) implementa `VectorSearch` com `<=>` (distância de
  cosseno) e a mesma divisão do k entre princípios ativos. O teste `test_pgvector_search.py` confere
  que devolve os mesmos trechos e scores do numpy. O `InMemoryIndex` continua no fluxo por arquivos.
- **Sem índice HNSW**: com poucos milhares de trechos a varredura exata leva milissegundos e dá o
  top-k exato; o HNSW é aproximado e, com o filtro por princípio ativo, pode devolver menos de k.
- `load_database.py` cria a extensão e as tabelas que faltam (`create_all`, sem Alembic por ora) e
  troca `bulas` e `chunks` com `TRUNCATE` numa transação só; `consultas` não é tocada. Mudar colunas
  de tabela já criada exige apagá-la ou adotar Alembic (decidir quando o histórico tiver dado real).

## 12. Ambiente Python
O Python local é o 3.15, sem wheels para várias libs (ex.: PyYAML não compila sem C++ Build
Tools). A imagem Docker usa 3.12. Para desenvolver no host, instale o **Python 3.12** e crie a
`.venv` com ele.

## 13. CSV da ANVISA (registros de medicamentos)
Procedência, SHA-256 e validação em `data/FONTES.md`. O que ele é e não é:
- **Serve para:** catálogo de nome comercial <-> princípio ativo (detecção na pergunta), filtro
  `Ativo`, número de registro (chave para achar a bula no Bulário) e classe terapêutica.
- **Não tem** bula, indicação nem link para bula. As bulas continuam manuais (o Bulário usa
  Cloudflare Turnstile; não automatizar nem contornar).
- Números (2026-10-11): 228.773 linhas; 85.091 ativas; ~8,8 mil pares (nome, princípio) únicos entre
  as ativas; ~2,5 mil princípios ativos; 12.268 ativas sem princípio ativo (homeopáticos etc.).
- **Cuidados:** `PRINCIPIO_ATIVO` traz combinações separadas por vírgula ("losartana potássica,
  hidroclorotiazida"); o catálogo precisa separar monocomponentes de associações. `CLASSE_TERAPEUTICA`
  é por registro e ruidosa (317 registros de paracetamol aparecem como "ANALGESICOS NARCOTICOS" por
  serem associações com codeína): usar só como dica secundária, rotulada "classe registrada na
  ANVISA", e só para produtos de um único princípio ativo. Não substitui as indicações da bula.

### Catálogo derivado e aliases (consequência da seção 13)
`data/processed/catalogo_anvisa.jsonl` (6.999 itens; 5.216 de um só princípio; 1.525 princípios
ativos distintos) é versionado. Ao adaptar `Catalog.montar` para usá-lo:
- Para os medicamentos **indexados**, usar princípio ativo + nomes comerciais do catálogo.
- Para os **não indexados** (`medicamentos_conhecidos`), usar só princípios ativos de produtos
  de um único princípio, não nomes comerciais: há marcas curtas ou ambíguas (ex.: "CORUS", "DIP")
  que gerariam falso positivo em perguntas normais.
- Sais: o catálogo traz "losartana potássica", a pergunta diz "losartana"; o alias da primeira
  palavra já cobre isso (ver `_aliases` em `services/medicines.py`).

**Implementado em 2026-10-13** (`Catalog.montar(bulas, conhecidos, anvisa, ignorados)`):
- Bula e catálogo casam pelo princípio ativo **sem a hidratação** ("dipirona" = "dipirona
  monoidratada"), mas **com o sal**: diclofenaco sódico e potássico são bulas diferentes. Quem
  pergunta por "diclofenaco potássico" tendo só o sódico indexado recebe "não indexado".
- Marca do catálogo só entra com 5+ letras, se não estiver em `data/aliases_ignorados.txt`, se
  pertencer a um único princípio ativo e não der nome também a uma associação. Só casa escrita
  certo, sem tolerância a erro de digitação ("INTESTIN" é marca de loperamida; "intestino" não pode
  virar loperamida). Marca curta e conhecida ("AAS", "Vitamina C") vai escrita à mão em
  `nomes_comerciais` da bula, que sempre vale.
- Produto que leva o nome de um indexado mas é outro remédio ("BUSCOPAN COMPOSTO", "TYLENOL SINUS")
  entra como não indexado, e na pergunta **o nome mais longo vence**. Sem isso, a pergunta sobre
  Buscopan Composto seria respondida com a bula do Buscopan simples.
- Palavra que já é um nome conhecido nunca é "corrigida" para outro: "prednisona" não vira
  "prednisolona".
- `data/aliases_ignorados.txt`: nomes do catálogo que são palavra comum na pergunta (glicose,
  cafeína). Cresce conforme o benchmark mostrar falso positivo.
- **Limites:** marca de medicamento não indexado não é reconhecida ("Dorflex", "Cataflam",
  "Clavulin"): a pergunta segue sem filtro. **Mudou em 2026-10-18 (seção 28):** sem nenhum remédio
  reconhecido, uma marca parecida com palavra da pergunta vira pergunta de volta ("é o Roacutan?"). Princípio ativo que só existe em associação
  ("clavulanato") também não. Formas diferentes do mesmo princípio ativo ("Voltaren gel") caem na
  mesma bula: quem resolve é o esclarecimento da rodada 0 (seção 14).

## 14. Rodadas de verificação e perguntas de esclarecimento (desenho)
Motivação: em saúde, "não sei" é melhor que resposta errada. Mas cada chamada ao LLM custa
30-45 s nesta CPU (seção 1), então as rodadas são **adaptativas**: só repetem se a anterior falhar.

**Rodada 0, sem LLM (instantânea).** `detectar()` + catálogo. Sem medicamento reconhecido e sem
filtro, ou nome ambíguo (várias bulas do mesmo princípio ativo: ex. diclofenaco sódico x gel),
a API devolve `status = precisa_esclarecimento` com a pergunta e as opções, sem gastar o LLM.

**Rodada 1.** Busca (top_k) + geração em JSON + `validar_saida` + **checagem de fundamentação**
determinística: todo número, dose e unidade da resposta tem de aparecer nos trechos citados, e a
maioria dos termos-chave também. Passou = responde.

**Rodada 2 (só se a 1 falhou).** Nova busca com o que a falha indicou (filtro pela seção inferida
da pergunta; trechos diferentes dos já usados) + geração, informando ao modelo o que falhou.

**Rodada 3 (só se a 2 falhou).** Última tentativa, com verificação por LLM ("cada afirmação está
nos trechos? sim/não"). Se não bater, **erro 502 `RESPOSTA_NAO_VERIFICADA` com mensagem explicando
o que falhou em cada rodada** (decisão do usuário em 2026-10-13; antes estava
`nao_encontrada`/`resposta_nao_verificada`). Nunca devolve resposta não verificada. Cada tentativa
fica registrada em `consultas`. Implementado como grafo do LangGraph: ver seção 16.

Por que não "rodar 3 vezes e comparar": com temperatura 0,1 as três saídas saem quase iguais, então
concordância não prova correção (o erro sistemático se repete), e triplica a latência sempre.

**Esclarecimento.** `status = precisa_esclarecimento`, HTTP 200, com
`esclarecimento: {motivo, pergunta, opcoes: [{rotulo, principios_ativos | bula_id}]}`.
O frontend mostra botões e reenvia a pergunta com o filtro escolhido e `consulta_origem_id`
(a API continua sem estado). No máximo **1** esclarecimento por pergunta, para não entrar em laço.
O sistema só pergunta sobre medicamento, forma farmacêutica ou seção; **nunca** por dado do
paciente (idade, gestação, doenças), por causa da LGPD e da regra "não personaliza".

Mudanças de contrato decorrentes: `status` ganha `precisa_esclarecimento` e `ErrorCode` ganha
`RESPOSTA_NAO_VERIFICADA` (502), já feitos; env `RODADAS_MAX=3` e `ORCAMENTO_S=150`, já feitos.
Com a API (2026-10-16, seção 18): `consultas` ganhou `rodadas` (int), `tentativas` (JSON) e
`consulta_origem_id` (UUID, sem FK).

## 15. Fluxo de ponta a ponta por arquivos, para teste manual (2026-10-13)
Do PDF à resposta sem banco, API ou tela. Da raiz, com `pip install -e backend` feito (ou
`PYTHONPATH=backend` no terminal):

1. `python -m ingestion.process_leaflet <pdf> --medicamento ... --principio-ativo ... --fabricante ...
   --data-versao AAAA-MM-DD` -> `data/processed/bulas/<id>.json` (lista as seções para conferir).
2. `python -m ingestion.generate_chunks` -> `data/processed/chunks.jsonl` + embeddings no cache.
3. `python -m benchmark.ask ["pergunta"]` -> resposta, trechos usados e tempos.

Decisões tomadas ao implementar:
- **O dataset em arquivo é a fonte da verdade** e o `Corpus` (`services/corpus.py`) carrega dele. O
  PostgreSQL fica para o histórico de consultas e para a API; `rag.responder` não sabe de onde o acervo veio.
- **Cache de embeddings** em SQLite (`data/cache/embeddings.sqlite`, fora do Git), chave = hash de
  modelo + texto. Grava a cada lote de 8: dá para interromper e continuar.
- O texto que vira embedding leva medicamento e seção na frente (`texto_para_embedding`): mudar
  esse cabeçalho invalida o cache.
- **Seções novas** em `Section`: `composicao` ("tem lactose?"), `resultados_eficacia` e
  `caracteristicas_farmacologicas` (mecanismo de ação). Antes caíam em `outras`.
- **Página do trecho:** o texto da seção guarda `\f` onde a página vira; o trecho usa a página em
  que começa.
- **Tokens são estimados** (3,3 caracteres por token, medido com o qwen2.5:1.5b). Não há tokenizador
  local sem baixar mais coisa.
- **Saída do modelo por afirmações**, não mais `{resposta, citacoes}`:
  `{"status", "afirmacoes": [{"texto", "trecho"}]}`, com JSON Schema que só aceita números de
  trechos enviados. O `[n]` é montado no código e `validar_saida` continua conferindo. Motivo,
  medido com a bula de teste: pedindo `[n]` dentro do texto, o qwen2.5:1.5b citou só na lista e
  5 de 5 respostas certas foram recusadas; no formato novo, 7 de 7 passaram.
- **Sem exemplo de resposta no prompt:** o modelo copiou o exemplo como resposta ("Deve ser tomado
  em jejum [2]") numa pergunta sobre armazenamento, e a validação de citações deixou passar.
- Cliente do Ollama com `urllib` (biblioteca padrão), síncrono.

**Limites de hoje:** testado só com a bula inventada de `backend/tests/fixtures/sample_leaflet.txt`;
a leitura de PDF (`pdfplumber`) ainda não rodou nesta máquina, porque depende do Python 3.12
(seção 12). As rodadas e a checagem de fundamentação entraram em 2026-10-15 (seção 16).

## 16. Agente em grafo com LangGraph (2026-10-15)
Pedido do usuário: o agente decidir melhor o que fazer quando a resposta é fraca, em vez de uma
tentativa só. A seção 14 virou um grafo de **7 nós** (`app/services/graph.py`): `entender`,
`esclarecer`, `buscar`, `gerar`, `validar`, `verificar_llm`, `finalizar`. `rag.responder`
continua sendo a porta de entrada, com o mesmo `Result` (mais `rodadas` e `tentativas`).

Decisões:
- **Só 2 nós chamam o LLM** (`gerar` e `verificar_llm`). As decisões ficam nas arestas
  condicionais (`depois_de_*`), em código. Motivo: 30-45 s por chamada nesta CPU, e o qwen2.5:1.5b
  não é confiável para escolher caminho; o `detectar()` determinístico já faz isso melhor. Nada
  de roteador, planejador ou reescritor de pergunta por LLM.
- **Seção da pergunta sem LLM** (`services/intent.py`): palavras-gatilho por seção. A rodada 1
  não usa (busca só por semelhança, como antes); da rodada 2 em diante a busca fica nas seções
  indicadas, com os trechos ainda não usados primeiro. Os já usados continuam elegíveis: o trecho
  certo pode ter vindo na rodada 1 e o erro ter sido do modelo.
- **Recusa do modelo também pode gerar nova rodada**, uma vez: se a pergunta indica seção e nenhum
  trecho enviado era dela, a recusa é falha de busca, não falta de informação.
- **Checagem de fundamentação** (`services/grounding.py`): todo número da afirmação tem de
  estar no trecho citado, e número com unidade de dose (mg, mL, gotas, %, °C…) com a mesma unidade;
  pelo menos `FUNDAMENTACAO_MIN_TERMOS` (0,6, arredondado para baixo, mínimo 1) dos termos de 4+
  letras também, comparando pelos 5 primeiros caracteres para aceitar flexão. Números por extenso
  no trecho ("uma vez") valem como dígito. Conservadora: paráfrase demais reprova e custa uma rodada.
- **A observação para a próxima rodada é genérica** ("trouxe informação que não está nos trechos"),
  sem repetir a afirmação errada: o modelo pequeno copia o que estiver no prompt.
- **Verificação por LLM só na rodada 3**, uma chave `aN: "sim"|"nao"` por afirmação no JSON Schema
  (todas obrigatórias, o modelo não pula nenhuma). `LLM_MODELO_VERIFICACAO` permite um modelo maior
  só nesse nó, que roda pouco: é o jeito de ganhar capacidade sem deixar a pergunta comum mais lenta.
- **Orçamento de tempo:** nova rodada só se `gasto + gasto/rodadas <= ORCAMENTO_S`.
- **Esclarecimento:** por ora só "nenhum medicamento reconhecido e sem filtro"; as opções são os
  princípios ativos indexados. **Saiu em 2026-10-16 (seção 20):** esse caso virou busca geral. Antes desse caso a busca rodava sem filtro. A ambiguidade de forma
  farmacêutica precisa do campo `forma` na bula.
- **Erro do Ollama não gera nova rodada**: fora do ar ou tempo esgotado não se resolve tentando de
  novo dentro da mesma pergunta.
- **Sem checkpointer e sem `langchain-ollama`**: a API continua sem estado (seção 14) e os nós
  chamam o cliente próprio (`app/llm/ollama.py`), que já usa JSON Schema no `format`.
  Dependência nova: `langgraph>=1.2,<2` (instalado 1.2.15; traz `langchain-core` e `langsmith`,
  que só envia algo se `LANGSMITH_TRACING` estiver ligado).

## 17. Bulas reais: lote pelo manifesto, várias bulas por PDF, registro e forma (2026-10-15)
Primeira ingestão das bulas reais (lote 1 de `FONTES.md`: 20 PDFs). `python -m ingestion.process_batch`
lê `data/bulas_manifesto.json` e grava 24 bulas em `data/processed/bulas/`.

Encontrado nos PDFs e decidido:
- **Um PDF pode ter mais de uma bula.** Tylalgin traz gotas, comprimido e sachê; Kaodine, loção
  10 mg/mL e loção cremosa 50 mg/mL; Delumi, duas faixas de concentração. Cada bula termina na
  tabela "Histórico de Alteração da Bula" (descartada: não é conteúdo clínico) e a próxima começa
  em "MODELO DE BULA PARA O PROFISSIONAL...". Por isso o manifesto descreve as `partes` de cada
  PDF, e o lote recusa o PDF se o número de bulas lidas não bater.
- **O manifesto é a entrada do lote**, não 20 comandos à mão: arquivo, SHA-256, fabricante, registro,
  princípio ativo (grafia do catálogo), nomes comerciais e, por parte, medicamento e forma. Nada é
  gravado para um PDF cujo SHA-256 difere do `FONTES.md` (há teste que confere os dois) ou cujo
  registro não aparece impresso na bula.
- **`Leaflet` ganha `numero_registro`, `forma_farmaceutica` e `versao`** (opcionais só por causa das
  bulas de teste). O registro é para a fonte completa na resposta (item "fonte em toda resposta");
  a forma é o que vai permitir perguntar "gotas ou comprimido?" quando houver duas bulas do mesmo
  princípio ativo. No banco, `carregar_banco` acrescenta as colunas com `ADD COLUMN IF NOT EXISTS`.
- **Versão:** a Geolab imprime um código ("V.01_12/2025") colado no rodapé de cada página; ele vira
  `versao`, e `data_versao` = dia 01 daquele mês. Organon e Vitamedic não imprimem código: a data vem
  do manifesto (data de aprovação ou a mais recente do histórico).
- **Nome:** `medicamento` segue o que o PDF diz. Cisteil e Albel se apresentam como "genérico" no PDF,
  mas o registro impresso é o do similar com marca no CSV; a marca fica em `nomes_comerciais`.
- **Ajustes no `separar_secoes`**, todos vistos nas bulas: rodapé colado no começo da linha
  ("V.14_02/20265. ADVERTÊNCIA...") é retirado antes de procurar título; título no singular
  ("ADVERTÊNCIA") e com erro de digitação da própria bula ("ARMANEZAMENTO", "SUSPERDOSE"); linha
  com "(" ou terminando em ")" não é título (a remissão "(veja o item 8." + "POSOLOGIA E MODO DE
  USAR)." abria a posologia no meio das advertências do Cozaar e do Omenax, e 4 seções sumiam);
  "Referências bibliográficas" pula até o próximo título (15 dos 20 PDFs têm, a maioria no fim de
  "Resultados de eficácia"); hífen no fim da linha é mantido e cola na linha seguinte (os 30 casos
  medidos eram palavras compostas: "anti-histamínico", "observou-se"); bula sem o título "DIZERES
  LEGAIS" (ambroxol) abre essa seção em "VENDA SEM PRESCRIÇÃO" / "Registro M.S." depois da superdose.
- **Resultado:** as 24 bulas têm as 8 seções esperadas e o registro conferido. Limite conhecido:
  tabelas (doses por peso, por exemplo) saem como texto corrido.

## 18. API FastAPI (2026-10-16)
Cinco rotas sob `/api/v1` (`app/main.py`, `app/api/routes.py`): `saude`, `medicamentos`, `perguntas`,
`historico` e `historico/{id}`. O agente é o mesmo do terminal (`rag.responder`); a API acrescenta a
leitura do acervo no banco, o histórico e o formato de erro.

Decisões:
- **O acervo da API vem do banco** (`Corpus.do_banco`): bulas e trechos são lidos uma vez, ao subir,
  e a busca é a `PgvectorSearch`. Os arquivos continuam sendo a fonte da verdade (seção 15); o banco é
  o que a API enxerga. Depois de `carregar_banco`, a API precisa ser reiniciada.
- **`chunks` ganhou `medicamento` e `n_tokens`**, para o banco espelhar o `chunks.jsonl` inteiro e a
  API montar os trechos sem ler arquivo. Banco carregado antes delas: a API recusa subir o acervo e
  pede para rodar `carregar_banco` de novo (não adivinha valor).
- **A bula do banco não tem as seções** (o texto está nos trechos), então o `Corpus` passou a aceitar
  qualquer objeto com os dados da bula (`CorpusLeaflet`, um `Protocol`): a `Leaflet` do arquivo e a do
  banco servem.
- **`preparar_banco` saiu de `ingestion/` para `app/db/connection.py`**: a imagem Docker da API só leva o
  pacote `app`, e a API prepara o banco ao subir (cria tabelas e colunas que faltam). As colunas
  novas de tabela já existente continuam entrando por `ADD COLUMN IF NOT EXISTS`, sem Alembic.
- **Toda pergunta fica no histórico**, inclusive esclarecimento e erro. Colunas novas: `rodadas`,
  `tentativas` (JSONB com cada rodada e a saída crua do modelo), `mensagem_erro` (no 502, a
  explicação das rodadas) e `consulta_origem_id` (sem FK, como o `chunk_id` das fontes). A entrada
  inválida (422) não é gravada: não chegou a ser pergunta.
- **Snapshot da fonte com fabricante e nº de registro**, e a coluna `destaques` já criada, nula
  (decisão de 2026-10-15 para a feature "fonte completa", que segue em espera).
- **A resposta do `POST` é montada a partir do snapshot gravado**, não do resultado em memória: o que
  o balconista vê na hora é exatamente o que o histórico mostra depois.
- **Todo erro sai em `ErrorOut`** (`app/api/errors.py`), inclusive a validação de entrada do FastAPI
  (422 `ENTRADA_INVALIDA`, com o campo e o motivo) e o banco fora do ar (503 `BANCO_INDISPONIVEL`).
  Código novo: `CONSULTA_NAO_ENCONTRADA` (404).
- **`/saude` sempre responde 200**, com `status` `ok` ou `degradado` e a lista do que falta (banco,
  Ollama, modelo não baixado): é para a tela avisar, não para derrubar a requisição.
- **A API sobe sem o banco** e tenta de novo a cada requisição. `DB_TIMEOUT_S=5`: sem limite, conectar
  ao banco desligado esperava ~130 s no Windows (medido; era o que deixava o `pytest` em 2 min 16 s).
- **Endpoints síncronos** (`def`): o FastAPI roda cada um numa thread, e a pergunta espera o Ollama
  (cliente síncrono, `urllib`) por dezenas de segundos sem travar as outras rotas.
- **Docker:** o serviço `backend` monta `data/` só para leitura. Bulas e trechos vêm do banco, mas o
  catálogo de nomes (`medicamentos_conhecidos.txt`, catálogo da ANVISA, aliases ignorados) é lido de lá.
  A porta 8000 foi publicada só em `127.0.0.1`, como a do banco. A imagem ainda não foi construída.

**Defeito encontrado nos testes e corrigido.** O schema temporário dos testes não isolava nada: com
`search_path=teste_x,public`, o `create_all` enxergava as tabelas do `public` pelo caminho, não criava
as do schema e os testes gravavam no banco de desenvolvimento (o `TRUNCATE chunks, bulas` do
`carregar_banco` apagava o acervo do `public` a cada `pytest`). Agora `criar_engine(schema=...)`
também usa `schema_translate_map`, e um teste confere que as tabelas existem no schema temporário.

## 19. Frontend React (2026-10-16)
Vite + React 19 + TypeScript em `frontend/`, duas abas (Perguntar e Histórico), sobre a API da seção 18.

Decisões:
- **Dependências mínimas e fixadas**: só `react`, `react-dom`, `vite`, `@vitejs/plugin-react`,
  `typescript` e os tipos do React, com versão exata (`save-exact`), sem roteador, sem biblioteca de
  componentes, sem fonte externa (a tela funciona sem internet). 24 pacotes, 73 MB em `node_modules`.
- **Instalação conferida antes** (regra do projeto para qualquer download): versões, datas e tamanhos
  lidos do registro do npm sem baixar nada; proveniência SLSA de React, Vite e plugin (publicados pelo
  GitHub Actions dos repositórios oficiais; o React passou de `facebook/react` para `react/react` com
  a React Foundation, em 2026); `package-lock.json` resolvido antes do download; `npm ci` (instala
  exatamente o lockfile e confere o SHA-512 de cada pacote); `npm audit` sem vulnerabilidades;
  Defender sem ameaças. `.npmrc` com `ignore-scripts=true`: nenhum pacote roda script ao instalar.
- **Proxy do Vite em desenvolvimento**: a tela chama `/api` na própria origem e o Vite repassa para
  `127.0.0.1:8000` (`FARMARAG_API` troca o destino). Não depende de CORS e evita o `localhost` → IPv6.
- **O Vite escuta em `127.0.0.1:5173`**, não em `localhost`: aqui o `localhost` resolve primeiro para o
  IPv6 (`::1`), o Vite escutava só nele, e o navegador que tentava `127.0.0.1` dava conexão recusada
  (erro -102). No PowerShell, `npm.cmd run dev`: o `npm.ps1` é barrado pela política de scripts.
- **Tipos espelhados à mão** (`src/api.ts`) dos contratos de `schemas/api.py`, sem gerador a partir do
  OpenAPI: são poucos modelos, e um gerador seria mais uma dependência. Mudou o contrato, muda lá.
- **A resposta e o histórico usam os mesmos componentes** (texto com `[n]` clicável e lista de fontes).
  O clique numa citação destaca a fonte e abre o trecho inteiro; o mesmo `[n]` clicado de novo reabre.
- **As duas abas ficam montadas** (`hidden`): trocar de aba não perde a resposta na tela, e o
  histórico recarrega ao voltar a aparecer. A aba fica no endereço (`#historico`).
- **Erro com título claro por código** e, no 502, a mensagem da API quebrada em uma linha por rodada.
  Os três erros de verificação (`RESPOSTA_NAO_VERIFICADA`, `SAIDA_INVALIDA`, `CITACAO_INVALIDA`)
  dizem que o sistema preferiu parar a mostrar algo que não confere com a bula.
- **Sem testes automáticos de tela por enquanto** (exigiriam Vitest e Testing Library, mais
  dependências). Conferido com `tsc` estrito, `vite build` e capturas do Edge sem janela em todos os
  estados, com uma API de teste (o mesmo `criar_app`, Ollama simulado, schema temporário) para os
  estados que dependem do LLM.
- **Falta**: serviço `frontend` no `docker-compose.yml` e o botão "não ajudou" (feature em espera).

## 20. Qualquer pergunta sobre remédio, com base nas bulas (2026-10-16)
Pedido do usuário depois de testar a tela: "qual o melhor remédio pra dor de cabeça?" voltava com
"sobre qual medicamento é a pergunta?" e parecia não responder nada. Ele quer que **qualquer
pergunta sobre remédio seja respondida** ("estou tonto, qual o melhor medicamento?", "tenho alergia a
um remédio, tem outro que eu possa tomar?"), **com base nas pesquisas**, isto é, nas bulas
recuperadas. Não com o conhecimento próprio do modelo.

Decisões:
- **Busca geral** (`_busca_geral` em `graph.py`) quando a pergunta não tem medicamento indexado
  (nenhum nome, ou só remédio sem bula no acervo): todas as bulas, só nas seções que a pergunta indica
  e **um trecho por princípio ativo**, até `TOP_K_GERAL=4`. Cada trecho é uma opção na resposta; sem a
  regra, os três melhores trechos podiam ser do mesmo remédio.
- **A seção vale já na rodada 1 da busca geral.** Sem isso, "melhor remédio pra dor de cabeça" traz
  reações adversas que citam cefaleia antes das indicações. `intent.py` ganhou gatilhos de escolha e
  de sintoma ("melhor remédio", "remédio pra", "o que tomar", "dor de", "febre", "tontura",
  "alergia"...) para Indicações, e "alergia"/"hipersensibilidade" para Contraindicações.
- **O nó `esclarecer` saiu** (o grafo tem 6 nós): ele perguntava o medicamento, o que agora vira busca
  geral. O contrato `precisa_esclarecimento` fica na API e na tela para a ambiguidade de forma
  farmacêutica (Tylalgin gotas × comprimido), que ainda não tem nó.
- **Remédio sem bula também busca**: "captopril pode com álcool?" procura o assunto nas outras bulas.
  O prompt ganha a observação "não há trechos da bula de captopril; não atribua a ele o que a bula de
  outro medicamento diz", e a tela avisa que a resposta veio de outras bulas. Se nenhuma bula fala do
  assunto, continua `nao_encontrada` com `medicamento_nao_indexado`.
- **Prompt**: saiu "não recomende dose nem conduta para um paciente específico"; entrou "responda toda
  pergunta com o que os trechos dizem, dizendo de qual medicamento é cada informação". Para "o
  melhor", ele lista os indicados (as bulas não dizem qual é o melhor); para alergia, o que as bulas
  dizem de hipersensibilidade e quais servem para o mesmo uso. A recusa ficou só para trechos sem
  relação nenhuma com a pergunta. Continua sem exemplo de resposta (o modelo pequeno copia).
- **Não muda**: toda frase cita o trecho, a checagem de fundamentação e as rodadas continuam, e
  resposta que não passa vira 502. É isso que mantém "com base nas pesquisas" verdadeiro.
- **Limite**: o acervo tem 20 dos 40 medicamentos do MVP. Para um sintoma que nenhuma bula trata
  (tontura: o dimenidrinato ainda não foi baixado), a resposta certa é "nenhuma bula do acervo fala
  disso", com os trechos consultados ao lado.

**Primeiro teste com o LLM real e as 24 bulas (mesmo dia), e o que mudou por causa dele.** Das 11
perguntas, as 7 sobre um remédio específico saíram certas, mas a busca geral deu três respostas
erradas com cara de verificadas: "estou tonto" → "Tylalgin gotas [1]. Ibuprofeno [2]."; "dipirona
para criança" → uma frase da bula do omeprazol atribuída à dipirona; "dor de cabeça" → só nomes.
Correções, todas determinísticas:
- **Fundamentação, regra 3**: algum termo da afirmação, além do nome do remédio, tem de estar no
  texto do trecho. O nome está no cabeçalho do trecho, então "Ibuprofeno 400 mg [2]" passava.
- **Fundamentação, regra 4**: remédio citado na afirmação (pelo catálogo, sem tolerar erro de
  digitação) tem de ser o do trecho ou aparecer no texto dele.
- **Busca híbrida na busca geral**: além da semelhança e da seção, o trecho precisa ter uma palavra
  do assunto da pergunta (`intent.termos_do_assunto`, com sinônimos de sintoma: "tonto" →
  tontura, vertigem, labirintite; "dor de cabeça" → cabeça, cefaleia, enxaqueca). Se a pergunta cita
  remédio sem bula, o trecho tem de mencioná-lo (em qualquer seção, os da seção da pergunta
  primeiro). Nenhum trecho assim = "não encontrei", sem chamar o LLM.
- **"Cefaleia" não é cefalexina**: a tolerância a erro de digitação de `detectar` lia "cefaleia" como o
  antibiótico (0,88 de semelhança). Palavra que aparece no texto das bulas (`Catalog.vocabulario`)
  é palavra de verdade e não passa pela tolerância; no texto do modelo a tolerância fica desligada.
- **Aviso específico** para a próxima rodada quando o modelo responde só com nomes
  (`DICA_SO_NOME`); com o aviso genérico ele repetia os nomes nas 3 rodadas. O prompt também pede
  "cada frase diz o que o trecho diz, nunca só o nome do medicamento".

Resultado depois das correções: 10 de 11 respostas certas ("tonto" e "dipirona" viram "não encontrei",
que é o certo com este acervo; "dor de cabeça" responde ibuprofeno e paracetamol com a indicação de
cada um). A de alergia continua fraca: sem saber a que remédio é a alergia, cita as contraindicações
por hipersensibilidade da nimesulida e uma do ibuprofeno sem relação. Omeprazol + clopidogrel acertou
nas duas últimas execuções, mas errou na primeira ("aumenta a toxicidade"): a tabela de interações do
Omenax saiu embaralhada do PDF e precisa de conferência.

## 21. Tela em formato de chat (2026-10-16)
Pedido do usuário: tirar o filtro de medicamento e deixar a tela "parecida com a do Claude, do
ChatGPT ou do Gemini". A tela de pergunta + aba de histórico (seção 19) virou um chat:
- Barra lateral com "Nova pergunta" e o histórico da API agrupado por data (Hoje, Ontem, Últimos 7
  dias, Anteriores); no celular, menu. Abrir uma consulta antiga mostra a pergunta e a resposta como
  estavam na hora (snapshot).
- Conversa com a pergunta em balão e a resposta com `[n]` clicável; as fontes viram chips embaixo da
  resposta e o clique abre o trecho (no lugar da coluna lateral de fontes).
- Sem filtro de medicamento: com a busca geral (seção 20), a pergunta sozinha basta. O contrato
  `principios_ativos` continua na API (o esclarecimento de forma farmacêutica vai usá-lo).
- A conversa vive só na tela (`conversation.ts`): a API continua sem estado, cada pergunta é independente.
  **Mudou no mesmo dia (seção 23):** a tela manda a última resposta, e "e para criança?" passa a
  continuar a pergunta anterior.
- Conferido com capturas do Edge sem janela: API de teste (Ollama simulado) para todos os estados, e
  a resposta real da dor de cabeça aberta pelo histórico.

## 22. Tabela de interações lida como tabela (2026-10-16)
A bula do Omenax traz as interações numa tabela de duas colunas (remédio | "Impacto clínico:" /
"Intervenção:" + texto). Pela extração de texto do `pdfplumber`, os rótulos da coluna da esquerda caíam
no meio das frases da direita ("pode reduzir o efeito antiviral e promover o Impacto clínico:
desenvolvimento de resistência") e um trecho juntava saquinavir com clopidogrel: no teste real, a
resposta disse que omeprazol + clopidogrel "aumenta a toxicidade" (a bula diz que reduz o efeito).
- `ingestion/extract_pdf.py` lê **só esse formato** como tabela (duas colunas, rótulo "Impacto clínico"
  ou "Intervenção" na esquerda). Cada célula vira uma linha com o nome do remédio na frente
  ("Clopidogrel, intervenção: Evitar o uso concomitante de omeprazol..."), para o trecho que pegar só
  aquela linha ainda dizer de quem fala. A tabela continua de uma página para a outra ("(continuação)");
  o cabeçalho de grupo ("Substratos de CYP2C19 (por exemplo, clopidogrel, ...)") vira uma linha.
- As outras tabelas continuam pela extração de texto: comparando página a página, só as páginas 26 a 30
  do Omenax mudaram nas 20 bulas. As tabelas de 10 colunas são o histórico de alteração (descartado) e
  as de eficácia e dose por peso ficam como estavam (limite conhecido).
- Reprocessado só o Omenax (`processar_lote --so`); 11 trechos novos com embedding (22 s); 866 trechos.

## 23. Conversa: pergunta seguinte, alergia e "o que o remédio faz" (2026-10-16)
Pedido do usuário depois do teste real: guardar a última pergunta para a pessoa perguntar outra coisa
depois, pedir o nome do remédio na pergunta de alergia, e saber o que cada remédio faz.

**Pergunta seguinte.** A tela manda a última resposta da conversa em `consulta_origem_id`; a API busca
a consulta no histórico (`rag.PreviousQuestion`: pergunta, princípios ativos usados, se pediu esclarecimento) e
continua sem estado. `consultas` ganhou `principios_ativos` (os usados na busca). No `entender`:
- sem remédio na pergunta nova ("pode na gravidez?"), herda o da anterior; se a anterior foi geral,
  só herda a pergunta que começa com "e"/"mas" ("e para criança?"), e o assunto vem da anterior;
- "e com varfarina?" junta o remédio novo com o da anterior (a pergunta é omeprazol + varfarina);
- pedido de remédio ("qual o melhor remédio para tosse?") é assunto novo e não herda nada;
- a busca usa as duas perguntas juntas, e o prompt ganha "PERGUNTA ANTERIOR: ...".

**Alergia.** "Alergia a / alérgico à / hipersensibilidade ao" + remédio (`intent.texto_apos_alergia`,
que corta o nome na primeira palavra de ligação: "alérgico a dipirona, posso tomar Exemplol?" é alergia
a dipirona e pergunta sobre o Exemplol). "Loratadina serve para alergia?" não casa: lá a alergia é a
indicação.
- Sem dizer o remédio: `precisa_esclarecimento` com motivo `remedio_da_alergia`, sem opções (a pessoa
  responde no chat; a resposta volta com a pergunta que pediu o esclarecimento). Uma vez só.
- Com o remédio (e nenhum outro): a contraindicação por hipersensibilidade da bula dele, se está no
  acervo, e as **alternativas**: os remédios do acervo da **mesma classe terapêutica** no registro da
  ANVISA (dipirona e ibuprofeno são "analgésicos não narcóticos": entra o paracetamol). Só eles, quando
  há: outro anti-inflamatório para quem tem alergia a um anti-inflamatório é arriscado (alergia
  cruzada). Sem nenhum da mesma classe, as indicações mais parecidas com o uso do remédio (indicação
  da bula dele ou a classe), só as que ficam a até 0,05 da melhor: a semelhança sozinha trouxe o
  albendazol para a dipirona.
- A regra 4 da fundamentação deixa o remédio da alergia aparecer citando a bula de outro ("para quem
  tem alergia a dipirona, o paracetamol é indicado para..."), e "alergia" vale como
  "hipersensibilidade" (a bula escreve um, o modelo o outro).
- A tela avisa: antes de trocar, conferir na bula de cada alternativa a hipersensibilidade e a alergia
  cruzada.

**O que o remédio faz.** O CSV da ANVISA, que já estava no projeto, traz a classe terapêutica de cada
registro. `Catalog.classes` guarda, por princípio ativo (e pelos nomes sem o sal), a classe mais
registrada ("dipirona" → ANALGESICOS NAO NARCOTICOS, sem acento como no CSV). A resposta traz
`classes_terapeuticas` dos remédios sem bula, e a tela mostra "para que serve, pelo registro na ANVISA".
Não é texto de bula, por isso não entra na resposta do modelo, só como informação ao lado.

Junto: "bissulfato", "hemifumarato", "trometamol" e outros sais entraram na lista de palavras que o
nome do remédio perde ("clopidogrel" não era reconhecido porque no registro é "bissulfato de
clopidogrel").

## 24. Respostas copiadas da bula, sem LLM (2026-10-16)
Pedido do usuário: melhorar as respostas. No teste real, o modelo de 1,5B errou onde a resposta certa
é uma frase que a bula já escreve: juntou a linha do clopidogrel com a da varfarina ("deve ser
evitado", quando a bula manda monitorar o INR) e recusou as alternativas para a dipirona.
`services/extract.py` monta a resposta com as frases da bula, e o `gerar` não chama o modelo, em dois
casos:
- **interação entre remédios citados** (a pergunta tem "com", "junto", "interação"...): as frases da
  bula de um que citam o outro, em interações, advertências e contraindicações, até 3. Na tabela de
  interações vai a linha inteira; linha de outro remédio que só cita este de passagem fica fora, e o
  cabeçalho de grupo (sem ponto final) também. Os trechos se sobrepõem: a linha cortada num e inteira
  no seguinte fica só a inteira. Nenhuma frase assim = segue pelo LLM.
- **alternativas na alergia**: a frase de hipersensibilidade da bula do remédio e a frase de indicação
  de cada alternativa, com o nome do medicamento na frente.
A resposta passa pela mesma validação (o texto é o do trecho, a fundamentação confere); só vale na
rodada 1: se reprovar, a rodada 2 vai pelo LLM. `extrativa` na resposta e "frases da bula (sem LLM)"
no `modelo` do histórico; a tela mostra "frases copiadas da bula, sem resumo do modelo".

Resultado no teste real, com o LLM de verdade: alergia à dipirona em 4 s (antes, recusa), alergia ao
ibuprofeno em 5 s (antes, 3 rodadas e 68 s), omeprazol + varfarina em 2 s com as duas linhas exatas
da tabela (antes, errado).

Limite: a resposta copiada é longa quando a frase da bula é longa (a contraindicação do ibuprofeno é
uma lista), e não resume nem junta informação de duas frases. Para isso continua o LLM.

## 25. A resposta tem de responder, dose copiada da bula, e o lote 2 (2026-10-16)
Com a bula da dipirona (Dorageo, lote 2 em `data/FONTES.md`), o teste real mostrou: "qual a dose
máxima de dipirona?" -> "não existe experiência com o uso de dipirona em longo prazo em pacientes com
insuficiência renal". A frase está na bula (passa na fundamentação), mas não responde.

- **Regra "fala do assunto"** (`intent.responde_ao_assunto`, no `validar`): a resposta do LLM precisa
  usar uma palavra do assunto da pergunta (tirados os nomes de remédio); ou citar trecho da seção que
  a pergunta indica (menos posologia, onde a frase errada estava); ou, em pergunta de dose, trazer
  número com unidade ("½ a 1 comprimido até 4 vezes ao dia" não tem a palavra "dose"). Reprovou = nova
  rodada com o aviso `DICA_ASSUNTO`. Medida antes de entrar: das 12 respostas certas do LLM gravadas
  no dia, nenhuma seria reprovada. Sinônimos novos: rins/renal/rim, fígado/hepático.
- **Dose copiada da bula** (`extract.frases_de_dose`): pergunta de dose sobre remédio do acervo traz,
  de cada bula dele, a frase da posologia com número e unidade que mais fala do pedido ("máxima",
  "inicial", "insuficiência cardíaca"); criança só se a pergunta é de criança. "Paracetamol, qual a
  dose máxima?" traz uma frase por apresentação (gotas, comprimido), que era o "depende da
  apresentação" esperado desde a seção 14. Sem frase assim, segue pelo LLM.
- **"E para criança?" sem pergunta anterior** pergunta de volta "sobre qual remédio?" em vez de buscar
  "criança" em todas as bulas (respondeu com uma frase da bula da permetrina).
- **O erro também traz o id da consulta** (`ErrorOut.consulta_id`): a consulta já era gravada; agora a tela
  manda o id na pergunta seguinte, e a conversa continua depois de um 502.

Resultado no teste real: dose máxima da dipirona em 3 s ("½ a 1 comprimido até 4 vezes ao dia"),
paracetamol por apresentação em 2 s, e as outras perguntas do dia iguais ou melhores.

**Lote 2 das bulas.** Só o Dorageo passou: a verificação de segurança e o que foi recusado estão no
`FONTES.md`. Os sites dos outros detentores publicam só a bula do paciente (Cimed, Aché), não têm link
direto (Prati-Donaduzzi) ou recusam cliente automático (Bayer, Kenvue, Eurofarma, Neo Química). As
19 que faltam ficam para download manual no Bulário Eletrônico (Cloudflare: não automatizar).

## 26. "Não ajudou" e o destaque na fonte (2026-10-17)
Pedido do usuário em 2026-10-14, desenho decidido em 2026-10-15. Quem pede outra explicação é o
balconista (botão "não ajudou, explique de outro jeito"); o agente não pede sozinho (resposta fraca
continua sendo a verificação em rodadas da seção 14). Toda resposta segue com fonte; a única pergunta
de volta continua sendo a de esclarecimento.

**Destaque** (`grounding.destaques`, sem LLM): de cada fonte citada, as frases inteiras do trecho
que sustentam as afirmações que a citam, como posições (início, fim) no texto do trecho. Frase copiada
da bula (resposta extrativa): as frases onde ela está. Paráfrase do modelo: as frases que mais cobrem
os termos e números da afirmação, uma de cada vez, até nenhuma acrescentar nada (no máximo 3); a
primeira precisa cobrir dois itens, porque uma palavra solta em comum não basta. O nome do remédio
não conta (está no trecho todo).

**Não ajudou** (`rag.AnotherExplanation`, grafo):
- A pergunta roda de novo **sem os trechos citados** na resposta que não ajudou (e nas outras
  explicações antes dela). Os enviados e não citados continuam valendo, mas depois dos que nunca foram
  enviados: o grafo começa como numa rodada seguinte, com eles já "usados".
- Com seção indicada, a seção vem antes da semelhança, como nas rodadas 2 e 3. Se a bula tem a seção
  e todos os trechos dela já foram citados, **não há outra explicação**: `nao_encontrada` com motivo
  `sem_outra_explicacao`, sem chamar o LLM. Na busca geral, os citados saem depois de escolhida a
  seção, para não voltar o erro da seção 20 (o remédio que causa tontura listado como se a tratasse).
- **Sem frases copiadas**: a cópia da bula pode ter sido o que não ajudou, e com outros trechos a
  frase copiada seria a mesma (os trechos se sobrepõem). A outra explicação passa pelo LLM, com a
  mesma validação de sempre.
- Recusa do modelo também vira `sem_outra_explicacao` (não `sem_contexto`: a primeira resposta mostrou
  que a bula fala disso).
- O prompt não recebe a resposta anterior nem aviso de "explique de outro jeito": o modelo pequeno
  copia o que vê no prompt (seções 15 e 16). A diferença vem dos trechos.

Feito em 2026-10-17: destaque e grafo (`Result.outra_explicacao`, `Source.destaques`), 16 testes.
Falta: a API (pedido de outra explicação a partir da consulta, destaques/fabricante/registro na
fonte, gravar em `consulta_fontes`) e a tela (botão, trecho com a parte destacada).

## 27. Classes, tipos e arquivos em inglês (2026-10-17)
Pedido do usuário: nomes em inglês. Escopo escolhido por ele: **classes do Python (e os aliases de
tipo `Section` e `Category`), tipos/interfaces do TypeScript, e os arquivos e pastas de código**
(`ingestao/` → `ingestion/`, `componentes/` → `components/`); o componente principal de cada arquivo
do frontend segue o nome dele (`Sidebar.tsx` exporta `Sidebar`). Continua em português todo o resto:
funções, métodos, variáveis, campos dos modelos e do JSON, colunas e tabelas do banco, os outros
componentes React, os textos da tela, e os nomes em `data/` e `docs/` (dados e documentos, não código).

- O contrato da API não muda: o JSON é o mesmo. Só os nomes dos schemas no OpenAPI (`/docs`) mudam.
- O banco não muda: as tabelas têm `__tablename__` explícito.
- Classe, tipo ou arquivo de código novo nasce em inglês.
- Os comandos mudam: `python -m ingestion.process_batch`, `python -m ingestion.generate_chunks`,
  `python -m ingestion.load_database`, `python -m benchmark.ask`.

| Onde | Antes → depois |
|---|---|
| `schemas/dataset.py` | Bula → Leaflet, SecaoBula → LeafletSection, Secao → Section, Ancora → Anchor, PerguntaBenchmark → BenchmarkQuestion, Categoria → Category (Chunk fica) |
| `schemas/api.py` | CodigoErro → ErrorCode, ErroDetalhe → ErrorDetail, ErroOut → ErrorOut, PerguntaIn → QuestionIn, FonteOut → SourceOut, OpcaoOut → OptionOut, EsclarecimentoOut → ClarificationOut, RespostaOut → AnswerOut, BulaResumo → LeafletSummary, MedicamentoOut → MedicineOut, ConsultaResumo → InquirySummary, HistoricoOut → HistoryOut, ConsultaDetalhe → InquiryDetail, SaudeOut → HealthOut |
| `db/` | Bula → Leaflet, Consulta → Inquiry, ConsultaFonte → InquirySource, BuscaVetorial → VectorSearch, IndiceEmMemoria → InMemoryIndex, Resultado (da busca) → SearchHit, BuscaPgvector → PgvectorSearch |
| `services/` | Acervo → Corpus, AcervoIncompletoError → IncompleteCorpusError, BulaDoAcervo → CorpusLeaflet, Catalogo → Catalog, Deteccao → Detection, Referencia → Reference, SaidaLLM → LLMOutput, SaidaInvalidaError → InvalidOutputError, Resultado → Result, Fonte → Source, Tentativa → Attempt, Opcao → Option, Esclarecimento → Clarification, Anterior → PreviousQuestion, OutraExplicacao → AnotherExplanation, Estado → State, Falha → Failure |
| `api/`, `llm/` | ErroApi → ApiError, Servicos → Services, RespostaLLM → LLMResponse, CacheEmbeddings → EmbeddingCache |
| `ingestion/` | ParteBula → LeafletPart, Manifesto → Manifest, ItemManifesto → ManifestItem, ParteManifesto → ManifestPart, ManifestoError → ManifestError, Relatorio → Report, _LeitorDeInteracoes → _InteractionTableReader |
| `frontend/src/` | CodigoErro → ErrorCode, ErroApi → ApiError, Fonte → Source, Opcao → Option, Esclarecimento → Clarification, Resposta → Answer, BulaResumo → LeafletSummary, Medicamento → Medicine, StatusConsulta → InquiryStatus, ConsultaResumo → InquirySummary, Historico → History, ConsultaDetalhe → InquiryDetail, Saude → Health, PerguntaIn → QuestionIn, FonteAtiva → ActiveSource, Vista → View, Mensagem (tipo) → Message, props dos componentes (TextProps, SourcesProps, AnswerProps) |

**Arquivos:**

| Onde | Antes → depois |
|---|---|
| `backend/app/api/` | erros → errors, rotas → routes |
| `backend/app/db/` | busca_vetorial → vector_search, busca_pgvector → pgvector_search, conexao → connection |
| `backend/app/services/` | acervo → corpus, citacoes → citations, extrato → extract, fundamentacao → grounding, grafo → graph, historico → history, intencao → intent, medicamentos → medicines |
| `ingestao/` → `ingestion/` | carregar_banco → load_database, catalogo_anvisa → anvisa_catalog, extrair_pdf → extract_pdf, gerar_chunks → generate_chunks, processar_bula → process_leaflet, processar_lote → process_batch, separar_secoes → split_sections, validar_csv_anvisa → validate_anvisa_csv, validar_pdf_bula → validate_leaflet_pdf |
| `benchmark/` | medir_ambiente → measure_environment, metricas → metrics, perguntar → ask |
| `backend/tests/` | cada `test_*.py` segue o módulo que testa (test_fundamentacao → test_grounding...); fixture bula_exemplo → sample_leaflet |
| `frontend/src/` | componentes/ → components/: AvisoErro → ErrorNotice, BarraLateral → Sidebar, Citacoes → Citations, Conversa → Conversation, Entrada → MessageInput, Icones → Icons, Mensagem → Message (componente MensagemVista → MessageView); conversa.ts → conversation.ts, formato.ts → format.ts, estilo.css → styles.css |

O arquivo de dados `data/processed/catalogo_anvisa.jsonl` mantém o nome (gerado por `ingestion/anvisa_catalog.py`).

As seções anteriores e o README foram atualizados onde citam classe, arquivo ou comando. Onde "bula", "fonte" ou
"consulta" aparecem como palavra (e não como classe), continuam em português.

## 28. Negação na fundamentação e nome parecido com marca da ANVISA (2026-10-18)
Caso real (consulta `b14b2d1a`, teste pela tela): "tomei rocutan para dor no ouvido, fiz certo?" foi
respondida com "Não, RoctoN não é indicado para dor no ouvido. É recomendado consultar um médico [...]
[1]", citando as indicações do **diclofenaco**, que dizem o contrário ("auxiliar no tratamento de
[...] inflamação de ouvido [...] otites"). Três falhas juntas:
1. "rocutan" (Roacutan, isotretinoína) não foi reconhecido: nem "roacutan" escrito certo é, porque
   marca de remédio sem bula fica fora do catálogo (seção 13). Sem remédio, a busca geral achou
   "ouvido" no diclofenaco, e o modelo inventou o nome "RoctoN".
2. A fundamentação compara palavras: "não é indicado para dor no ouvido" tem as palavras do trecho e
   passava, assim como o contrário dela. "Médico" casava com "medicamento" pelo prefixo de 5 letras.
3. A regra 4 (remédio certo) só pega nome que o catálogo conhece; "RoctoN" não é.

**Regra 5 da fundamentação** (`grounding.nega_sem_o_trecho`): frase da afirmação que nega ("não",
"nunca", "jamais", "nem", "nenhum") só passa se alguma frase do trecho que a sustenta (as do destaque,
seção 26) também nega; no trecho também valem "contraindicado" (e "contra-indicado"), "evitar" e
"proibido". A checagem é frase por frase da afirmação: com a afirmação inteira, a outra frase dela
puxava outras frases do trecho para o apoio, e o "não" de qualquer uma ("Febre isolada não é uma
indicação") salvava a negação inventada. Reprovação só por ela leva o aviso `DICA_NEGACAO`.
- Conferida nas 81 respostas aprovadas do histórico (125 afirmações, 44 com negação): reprova 2, e as
  2 estavam erradas ("não há dados [...] em crianças" citando estudo em ratos; "omeprazol com
  clopidogrel não é recomendado" citando um trecho que não fala de clopidogrel).
- **Limite:** o caminho inverso (a bula nega, a afirmação não: "pode usar na gravidez" citando "não
  use na gravidez") não é checado; com cláusulas longas, reprovaria paráfrase certa demais. Fica para
  o benchmark.

**Nome parecido com marca da ANVISA** (`medicines.sugerir`, `graph.esclarecer_nome`): quando nenhum
remédio é reconhecido e a pergunta não responde a um esclarecimento, palavras de 6+ letras que não
são nome conhecido nem palavra das bulas são comparadas às marcas de uma palavra do registro da ANVISA
(`Catalog.marcas`: 3.245, de um só princípio ativo, fora de associação, inclusive de remédio sem
bula). Parecida (difflib, 0,85) → `precisa_esclarecimento` com motivo `medicamento_nao_reconhecido`:
'Não reconheci "rocutan". É o Roacutan (isotretinoína)? Se não for, escreva o nome do remédio.', com
uma opção por princípio ativo (até 3). Pergunta em vez de adivinhar: a marca não decide nada sozinha,
e por isso a seção 13 continua valendo para a detecção.
- 0,85 porque uma letra trocada numa marca de 8 letras dá 0,875 ("buscopam"). Medido com 159
  perguntas (110 do histórico, 49 de balcão): o único falso positivo era palavra + sufixo ("herpes
  labial" → HERPESIL), descartado (marca que começa com a palavra e tem 2+ letras a mais).
- A sugestão vem antes da conversa: "e o rocutan?" não continua a pergunta anterior.
- **Filtro (seção 5):** o botão reenvia a pergunta com o princípio ativo da opção. O filtro passa a
  aceitar remédio conhecido sem bula, que segue como na pergunta escrita (busca nas bulas que falam
  dele, `medicamentos_nao_indexados`, classe terapêutica da ANVISA); só nome desconhecido dá 404
  `MEDICAMENTO_NAO_ENCONTRADO`. Antes, todo nome sem bula dava 404.
- O reenvio pelo botão traz a mesma pergunta com `consulta_origem_id` do esclarecimento: o texto não
  é repetido na busca.
- Resultado no caso real: depois do botão, "não encontrei nas bulas" (nenhuma bula fala de
  isotretinoína, sem LLM) e "Para que serve, pelo registro na ANVISA: isotretinoina: PRODUTOS
  ANTI-ACNE", que é o porquê de não servir para dor de ouvido.
- Ainda não: o conselho genérico do modelo ("consulte um médico") passa pelas regras 1 a 3; fica para
  o aviso fixo da tela (F2) e o prompt.

## 29. Para que serve: o que não está nas indicações não é indicado, e o lote 3 (2026-10-18)

**Caso real (18/10, depois do §28):** "oque acha de eu tomar zolpidem para nauséa?", "certralina é pra
vomito?" e "paracetamol serve para ansiedade?" voltaram "não encontrei". Os dois primeiros não tinham
bula; no terceiro, a rodada 2 enviou as três indicações do Tylalgin e o modelo recusou, porque
nenhum trecho fala de ansiedade. Pedido do usuário: "o que cada medicamento faz, pra que serve; e se
não está escrito para que serve, não é indicado".

**A regra** (`services/indication.py`, caminho novo no `buscar`, sem LLM): a bula do profissional lista
as indicações aprovadas; o que não está nela é uso fora da bula. Pergunta de uso ("X serve para Y?",
"é pra", "é bom para", "tomei X para Y", "para que serve X?") é respondida com as frases da seção de
indicações, copiadas como estão (no máximo 2 por bula, 4 no total), e uma **conclusão montada no
código** (`Result.conclusao`, `AnswerOut.conclusao`, coluna `consultas.conclusao`), que a tela mostra
em negrito antes das frases:
- **citado** (metade ou mais das palavras do uso aparecem numa frase sem negação): "Sim. Pela bula
  de paracetamol, é indicado para isso:" (sem dizer que a palavra está na bula: "espinha" casa com
  "acne" pelo sinônimo);
- **ressalva** (só aparece em frase que nega: "não é recomendado trocar ... na insuficiência
  cardíaca", Cozaar): mostra a frase, sem sim nem não;
- **geral** (o uso é uma dor que a bula não cita, mas ela indica para dor: "dor no ouvido" e o
  paracetamol, "dores leves a moderadas"): "não cita dor no ouvido; indica para dor de forma geral";
- **ausente**: "Não. Ansiedade não está entre as indicações da bula de paracetamol: pela bula, não é
  indicado para isso. A bula indica:" e as frases.
- **Remédio sem bula no acervo**: sem busca (as outras bulas só o citam de passagem, numa interação)
  e sem concluir: "A bula de X não está no acervo, então não dá para dizer se é indicado para Y. Pelo
  registro na ANVISA, é da classe ...". Decisão do usuário: a classe é grosseira (o clonazepam é
  "anticonvulsivante" e a bula dele indica pânico), concluir por ela seria inventar.

Cuidados:
- **Pergunta com outra condição não entra** ("ibuprofeno para dor se tenho dengue?", "dipirona para
  dor de cabeça estando grávida?", "para criança?", dose, interação): segue o caminho normal, que lê
  contraindicação, advertência ou posologia. Senão o "sim, é indicado" esconderia a contraindicação.
- Comparação por começo de palavra, como a busca, com sinônimos de como a pessoa fala ("ouvido" →
  otite, "insônia", "espinha" → acne). Falsos amigos tratados: "febre do feno" (loratadina) e "febre
  ou outras alergias" (oximetazolina) são rinite alérgica, não febre.
- "Não" que descreve a doença ("reumatismo não articular", "diabetes não dependente de insulina") não
  é negação da indicação; só "não é", "não deve", "não é recomendado", "contraindicado".
- "E para dor de cabeça?" depois de uma pergunta de uso é outra pergunta de uso do mesmo remédio;
  depois de outra pergunta, não.
- O uso volta como a pessoa escreveu ("pressão alta", "nauséa"); a conclusão repete o nome do
  princípio ativo sem o sal ("sertralina").
- **Limite**: sinônimo que a lista não tem dá "ausente" para um uso que a bula cita com outra palavra.
  A conclusão mostra as indicações logo abaixo, e o balconista vê a frase da bula. Medir no benchmark.

**Lote 3** (procedência analisada antes de baixar, a pedido do usuário; detalhes em `data/FONTES.md`):
10 PDFs, 12 bulas (sertralina e zolpidem da Geolab; isotretinoína 10 e 20 mg, AAS 500 e 100 mg,
amoxicilina, clotrimazol, nitazoxanida, oximetazolina e nafazolina da EMS pelo sara.com.br, portal de
bulas da EMS; Pozato Uni da Libbs). Banco com 37 bulas. Os PDFs da EMS sem "MODELO DE BULA..." trazem
duas bulas: o separador abre a seguinte em "I - IDENTIFICAÇÃO DO MEDICAMENTO" depois do histórico
(as 25 anteriores, reprocessadas, ficaram idênticas). Pendentes: Abbott, Germed e Sandoz (ressalvas
de procedência, decisão do usuário), Natulab (domínio em nome de outra empresa, CNPJ baixado) e
Eurofarma (bloqueia cliente automático); cetirizina, bisacodil, escopolamina, dimenidrinato, ácido
ascórbico, guaifenesina e sulfato ferroso pelo Bulário, à mão.

## 30. Benchmark de 80 perguntas e reranking da rodada 1 (2026-10-18)

Pedido do usuário: reranking ("o banco vetorial traz 10, escolhe cirurgicamente os 3 para o agente") e
um benchmark para medir. A divisão por seção com metadados (a outra parte da ideia) **já existia**:
cada trecho nasce dentro de uma seção e carrega seção, medicamento, princípio ativo, página, versão,
fabricante e registro; o limite de ~300 tokens só corta seção grande (201 das 442 cabem num trecho só;
as advertências do Omenax dariam ~9 mil tokens num trecho, inviável no qwen 1.5B na CPU). O passo
seguinte nessa linha são as subseções (gravidez, idosos, rins, crianças) como metadado: etapa 2.

**Gabarito** (`data/benchmark/perguntas.jsonl`, formato da seção 2): 80 perguntas, 70 com resposta nas
bulas (posologia 12, contraindicação 11, advertência 10, interação 10, reação adversa 9, indicação 9
com as 4 reais de 18/10, conservação 5, superdose 2, composição 2) e 10 sem resposta (remédio sem
bula, preço, vacina, unha encravada). Cada uma com âncora (princípio ativo, seção, frase copiada da
bula; as 81 conferidas contra os trechos) e resposta esperada detalhada. Categorias `advertencia`,
`superdose` e `composicao` entraram no contrato.

**Executor** (`python -m benchmark.run`): modo `busca` troca o LLM por um que recusa e mede o que a
rodada 1 entrega ao agente (acerto = todas as âncoras nos trechos enviados ou copiados), por categoria
e por caminho (semelhança, copiada, indicação, busca geral), e a posição da âncora entre os 10
candidatos da busca vetorial. `--sem-rerank` é a linha de base. Modo `completo` (LLM de verdade:
quinta) mede também respondida, recusa certa, rodadas e latência. Resultados em
`data/benchmark/resultados/`.

**Reranking** (`services/rerank.py`, chamado em `graph._primeira_busca`; `RERANK_CANDIDATOS=10`, 0
desliga), sem modelo e sem download:
- candidatos: os 10 mais parecidos de cada remédio + os 2 mais parecidos de cada seção que a pergunta
  pede, mesmo fora dos 10 ("zolpidem tem lactose?" não trazia a composição);
- nota = semelhança + 0,08 se é da seção pedida + 0,08 × fração dos assuntos da pergunta que o trecho
  cita (`intent.grupos_do_assunto`: uma palavra e seus sinônimos contam uma vez; nome de princípio
  ativo, marca e fabricante não contam);
- escolha gulosa dos top_k pela nota, com a divisão por remédio que a busca já fazia e sem trecho quase
  igual a um já escolhido (Jaccard ≥ 0,8: as bulas do Tylalgin repetem frases);
- a semelhança original continua sendo a nota mostrada na tela.
- Seções do reranking (`intent.secoes_para_rerank`): as de `inferir_secoes` e mais: "pode
  tomar/usar/ser dado/começar" com um remédio pede contraindicações e advertências (pergunta de
  segurança sem a palavra "contraindicado"), se não for de posologia; dois ou mais remédios pedem
  interações; "luz/umidade" armazenamento, "resseca" reações, "cerveja/beber" interações, "quantos
  dias/quanto tempo" advertências, "momento/horário" posologia. Fica fora de `inferir_secoes` porque a
  busca geral usa: lá "o que posso tomar para dor?" é pedido de remédio.
- Pesos: grade em 51 perguntas (as que passam pela semelhança) deu um platô de 39-40/51 para seção
  0,08-0,10 e assunto 0,04-0,10 (com 2 extras por seção); escolhido o meio do platô, sem caçar o
  máximo, para não ajustar às 80 perguntas.

**Falhas de reconhecimento que o benchmark achou** (não eram de ranking), corrigidas:
- "AAS" não era reconhecido (3 letras: o filtro de alias descarta). Lista curta de siglas de balcão
  revisadas à mão (`medicines._SIGLAS`).
- "Alérgico a penicilina, pode amoxicilina?" e "rinite alérgica com coriza, fexofenadina serve?"
  caíam no fluxo de alergia e perguntavam "a alergia é a qual remédio?". Agora, se o alvo da alergia
  não é remédio conhecido e há remédio na pergunta, ela segue sobre ele.
- "Toma losartana, pode tomar ibuprofeno para dor?" caía na regra de indicação (seção 29). A regra
  agora só vale com um remédio na pergunta.

**Resultado** (modo busca, mesmo gabarito, rodada 1):

| | linha de base | reranking + correções |
|---|---|---|
| Acerto geral | 44/70 | **57/70** |
| Pelo caminho da semelhança | 27/45 | 40/51 (top-3 só por semelhança, nas mesmas: 29/51) |
| Contraindicação | 3/11 | 6/11 |
| Reação adversa | 4/9 | 7/9 |
| Advertência | 7/10 | 9/10 |
| Composição / conservação | 0/2, 4/5 | 2/2, 5/5 |
| Tempo por pergunta (mediana) | 2,25 s | 2,26 s |

O reranking em si custa milissegundos; o tempo é o embedding da pergunta. O ganho que mais importa
para o balcão vem na quinta: trecho certo na rodada 1 = menos rodadas de 30-45 s do LLM.

Ainda erram (13): contraindicações em que advertências parecidas ficam na frente (angioedema, IMAO,
úlcera com AAS, mulher que pode engravidar e isotretinoína); interações (losartana + ibuprofeno,
metformina + cerveja, varfarina + AAS); reações (lábios ressecados, tosse do enalapril); álcool com
paracetamol; superdose de paracetamol (10º candidato, entre 6 trechos da mesma seção); e dois casos do
caminho copiado, que não passa pelo reranking (posologia da loção para piolho escolheu a bula da
sarna; alisquireno ficou atrás de frases de interação). **Limite do método**: pesos e gatilhos foram ajustados olhando essas 80 perguntas;
perguntas novas, escritas sem olhar o resultado, são o teste honesto (próximo lote do gabarito).
