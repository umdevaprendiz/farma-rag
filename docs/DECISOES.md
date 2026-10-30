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
  no trecho ("uma vez") valem como dígito. Conservadora: paráfrase demais reprova e custa uma rodada
  (a citação trocada é corrigida no código desde 2026-10-22: seção 31).
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

**API e tela (2026-10-20):**
- Rota própria, `POST /api/v1/perguntas/{consulta_id}/outra-explicacao`, sem corpo. A tela não
  reenvia a pergunta: a API a tira do histórico, e assim pergunta, filtro e conversa são os da
  original, sem a tela poder mandar outra coisa.
- Coluna nova `consultas.outra_explicacao` (boolean, padrão false). A consulta nova guarda em
  `consulta_origem_id` a resposta que não ajudou; seguindo os `outra_explicacao = true` para trás,
  chega-se à pergunta original (`history.explicacoes`, no máximo 20 elos).
- `AnotherExplanation` do histórico (`history.para_outra_explicacao`): `evitar` = fontes citadas em
  toda a cadeia; `enviados` = os chunk_ids de todas as rodadas (`tentativas`) e das fontes gravadas
  (que são só as da última rodada).
- A rodada roda com a pergunta anterior da conversa da original (o `consulta_origem_id` dela):
  "e como conservar?" só é sobre o remédio por causa dela.
- Só resposta `respondida` pede outra explicação; senão, 422 `ENTRADA_INVALIDA` (não há resposta
  para explicar de outro jeito).
- `SourceOut.destaques` e `consulta_fontes.destaques` com `[início, fim]` em caracteres do Python
  (code points); a tela fatia com `Array.from`. Fonte não citada: lista vazia; consulta gravada antes:
  nula no banco, lista vazia na API.
- Tela: o botão fica só na última resposta da conversa (a outra explicação aparece logo embaixo),
  também quando ela foi aberta do histórico. Fabricante e registro já estavam no cartão da fonte.

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

## 31. Citação trocada corrigida no código e remédio da pergunta ignorado (2026-10-22)
Parte da melhoria de tempo (F3) pedida pelo usuário em 2026-10-17. Caso real (consulta `12f61084`,
"oque acha do ibuprofeno pra dor de cabeça?"): o modelo copiou a frase certa do trecho [3] e citou
[1] e [2], a mesma frase duas vezes. A fundamentação reprovou (certo) e a rodada 2 repetiu tudo:
98 s em 2 rodadas, ~45 s jogados fora. Até aqui, citação errada = nova rodada (seções 14 e 16).

**Correção da citação** (`grounding.corrigir_citacoes`, no `validar`, antes da checagem):
- se a afirmação não passa no trecho que citou, mas passa em todas as regras (1 a 5) num outro
  trecho enviado, o `[n]` passa a ser o desse trecho (o primeiro, na ordem enviada). O `[n]` já era
  montado no código (seção 15), então nada muda no formato;
- a checagem continua sendo contra o trecho citado: a fonte mostrada (e o destaque) é a que sustenta
  a frase. Afirmação que nenhum trecho sustenta fica como veio e reprova como antes;
- o outro trecho tem de ser do mesmo remédio do citado, ou de um remédio que a afirmação nomeia:
  "é indicado para dor de cabeça", escrita sobre o ibuprofeno, não pode virar a bula do paracetamol;
- afirmação repetida (o mesmo texto, sem contar caixa e pontuação) sai;
- o que mudou fica em `Attempt.correcoes` (histórico, `benchmark.ask`, e `com_citacao_corrigida`
  no `benchmark.run --modo completo`).

Medido no histórico (113 consultas, 118 rodadas do LLM gravadas): das 42 reprovadas na fundamentação,
11 passariam, todas com a frase escrita no trecho para onde a citação foi (omeprazol + clopidogrel,
losartana na insuficiência cardíaca, o caso do ibuprofeno). Nas aprovadas, nenhuma citação muda; numa,
a frase repetida sai.

**Remédio da pergunta ignorado** (`graph._remedios_ignorados`, no `validar`, com `DICA_ASSUNTO`). Uma
das 11 era "E com varfarina?" (consulta `72e05926`): a resposta falava de clopidogrel, metotrexato e
vitamina B12, e só tinha reprovado por acaso, pela citação. A regra do assunto (seção 25) tira os nomes
de remédio da pergunta ("dose máxima de dipirona" confere "dose máxima"), e aqui o nome é o assunto.
Regra nova: remédio nomeado na pergunta tem de estar na resposta (nome, sal ou marca) ou ser o de um
trecho citado, **se algum trecho enviado fala dele**. Ficam de fora:
- o remédio da alergia ("sou alérgico a dipirona, posso tomar paracetamol?");
- o remédio que nenhum trecho enviado cita ("Exemplol com captopril", sem bula de captopril): a
  resposta nem poderia falar dele (regra 4 da fundamentação), e exigir isso só daria 3 rodadas e 502.
Medido nas 81 respostas aprovadas do histórico (61 com remédio na pergunta): nenhuma seria reprovada
(a única marcada pela conta é a resposta a "a alergia é a qual remédio?", que o grafo exclui).

**A medir com o LLM real** (F3e): quantas perguntas passam a terminar na rodada 1 e o tempo, com as
perguntas da quinta e o `benchmark.run --modo completo`.

## 32. Benchmark com vários números de candidatos e semente fixa no modo completo (2026-10-22)
Pedido do usuário: rodar o benchmark com 25, 50 e 100 candidatos "para ver se vai ser a mesma
resposta". O número de candidatos (`RERANK_CANDIDATOS`, seção 30) só muda a rodada 1 das perguntas com
remédio indexado: copiar a frase da bula, a busca geral e "para que serve" não passam pelo reranking, e
as rodadas 2 e 3 buscam pela seção, sem ele.

`python -m benchmark.run --candidatos 10 25 50 100` roda cada valor e compara com o primeiro (a base):
- **mesmos trechos na rodada 1, na mesma ordem** = mesmo prompt para o modelo (a ordem decide o `[n]`);
- quais perguntas **passaram a acertar** e quais **deixaram de acertar** (âncora nos trechos enviados);
- a âncora entre os N candidatos de cada valor (`semelhanca_ancora_nos_N_candidatos`);
- no modo completo, **mesma resposta** (status e texto) e a lista das que mudaram.
O embedding de cada pergunta é calculado uma vez só para todos os valores (~2 s na CPU por pergunta).
`--sem-rerank` continua valendo e é o mesmo que `--candidatos 0`.

**Modo completo sem horas de CPU:** a pergunta que recebe os mesmos trechos na mesma ordem recebe o mesmo
prompt. O LLM só roda de novo nas perguntas cujos trechos mudaram; as outras reaproveitam a resposta da
base (marcadas `reaproveitada`). Para isso valer, o modo completo usa **semente fixa**: `LLM_SEMENTE`
(nova, opcional, vazia no uso normal) vai como `seed` nas opções do Ollama, e o benchmark usa 42 quando o
`.env` não define. Com a temperatura 0,1 e sem semente, o mesmo prompt podia dar respostas diferentes, e
a comparação mediria o sorteio junto com a configuração.

**Resultado** (modo busca, 2026-10-22, base = 10, o valor atual):

| Candidatos por remédio | Acerto na rodada 1 | Âncora entre os N candidatos (51 pela semelhança) | Mesmos trechos que com 10 |
|---|---|---|---|
| 10 | 57/70 | 44/51 | base |
| 25 | 58/70 | 48/51 | 77/80 |
| 50 | 58/70 | 51/51 | 77/80 |
| 100 | 58/70 | 51/51 | 77/80 |

- 77 das 80 perguntas recebem exatamente os mesmos trechos, na mesma ordem, com qualquer valor: mesmo
  prompt, mesma resposta. 50 e 100 são idênticos em todas; 25 difere de 50 só no terceiro trecho da rea01.
- As 3 que mudam: rea01 ("Isotretinoína resseca os lábios e a pele?") passa a acertar, porque o trecho
  certo era o 16º candidato; pos06 (zolpidem) continua certa, com um trecho de posologia no lugar de um de
  advertências; int02 (losartana + ibuprofeno) continua errada.
- Com 50, o trecho certo está entre os candidatos nas 51 perguntas pela semelhança, mas o reranking
  escolhe certo em 41: os erros que sobram são de escolha (pesos e seções da pergunta), não de candidatos.
- Custo: nenhum que se meça (0,08 s de mediana por pergunta sem o embedding, igual nos quatro valores).
- Mesmo limite da seção 30: os pesos foram ajustados olhando estas 80 perguntas.
- `RERANK_CANDIDATOS` continua 10 até o usuário decidir.

## 33. Aviso de saúde em toda resposta (F2, 2026-10-24)
Pedido do usuário: toda resposta traz um aviso fixo de que a informação é da bula e não substitui o
farmacêutico. O rodapé da caixa de mensagem já dizia isso em geral; agora cada resposta leva o aviso.

**Texto** (`AVISO_SAUDE`, `schemas/api.py`): "Informação da bula, para conferência. Não substitui a
avaliação do farmacêutico nem a consulta médica."

**Decisões tomadas por mim, porque o usuário mandou fazer sem responder as três perguntas que o plano
(`.claude/FEATURES.md`) deixava em aberto. Mudar qualquer uma custa uma linha:**
- **Onde aparece:** só em `status = respondida` (também na resposta copiada da bula e na outra
  explicação do "não ajudou"). Em `nao_encontrada`, esclarecimento e erro não há informação de bula a
  avisar, e o aviso viraria ruído.
- **Quem monta:** a API, no campo `aviso` de `AnswerOut` e de `InquiryDetail` (`computed_field`: sai no JSON
  e na documentação em `/docs`). Um só texto, lido pela tela e pelo `benchmark.ask`; outro cliente da API
  também o recebe.
- **Nunca pelo LLM e nunca gravado:** não passa pela fundamentação, não entra em `resposta` nem em
  nenhuma coluna de `consultas`, e não custa tempo. O histórico o calcula ao abrir a consulta, então
  trocar o texto vale também para as respostas antigas.

**Tela:** `<p className="aviso-saude">` em `Message.tsx`, depois das fontes e antes da linha de
rodadas e tempo: texto pequeno, cor de apoio, só uma linha lateral (sem o fundo amarelo do
`.nota-aviso`, que é para avisos da resposta). `benchmark/ask.py` o imprime depois da resposta.

**Continua pendente** (F4/plano): tirar do prompt o conselho genérico "consulte um médico"; o aviso fixo
já cobre.

## 34. A seção da pergunta já na rodada 1 (F3b, 2026-10-24)
**Problema** (consulta `12f61084`, 98 s em 2 rodadas): a pergunta de indicação trouxe na rodada 1 dois
trechos de "resultados de eficácia" e um de indicações. O bônus de seção do reranking (seção 30) é de
0,08, e um trecho da seção pedida muito menos parecido que os outros ficava de fora.

**Duas regras em `services/rerank.py`**, ligadas por `RERANK_GARANTE_SECAO` (padrão `true`):
1. **Seções raras** (`SECOES_RARAS`: "resultados de eficácia" e "características farmacológicas"): só
   estudos clínicos e farmacologia, trechos longos que pesam no prompt (cada trecho a mais custa ~5 s de
   leitura) e quase nunca são a resposta no balcão. Só ficam na frente se a pergunta as pede; senão a
   nota cai `PENALIDADE_SECAO_RARA` (0,5), e elas vão para o fim da fila: entram só se faltar candidato
   (rebaixadas, não excluídas). Quem as pede: `intent._MAIS_GATILHOS` ganhou estudo, ensaio, eficácia,
   evidência, farmacocinética, farmacodinâmica, absorção, metabolização, excreção, "fazer efeito",
   "início de ação" e "duração do efeito"; "mecanismo", "como age" e "meia-vida" já existiam. Os
   gatilhos novos valem só no reranking (`secoes_para_rerank`), não na busca geral nem nas rodadas 2 e 3.
2. **Garantia:** se a pergunta indica seção e nenhum trecho escolhido de um remédio é de uma delas, o
   trecho da seção mais bem colocado desse remédio (pela nota) entra no lugar do escolhido de menor
   semelhança. Vale por remédio (pergunta de interação com 2 remédios leva um trecho de interações de
   cada). Não entra trecho quase igual (Jaccard >= 0,8) a um que fica, nem remédio sem trecho da seção
   entre os candidatos. O `score` devolvido continua sendo a semelhança (é o que a tela mostra).

`RERANK_GARANTE_SECAO=false` volta ao reranking da seção 30 (a linha de base para comparar).

**O que foi verificado e o que não foi:**
- Testes sem Ollama: `test_rerank.py` (reranking, garantia, remédios, duplicata, gatilhos) e um teste
  de grafo em `test_rag.py` (com a garantia a rodada 1 já leva a reação adversa e a pergunta acaba em 1
  chamada; sem ela, a rodada 1 erra a seção e paga a rodada 2 inteira).
- Gabarito de 80 perguntas: nenhuma âncora está em "resultados de eficácia" nem em "características
  farmacológicas" (conferido no arquivo), então a regra 1 não tira trecho certo do benchmark.
- **Não medido:** o acerto@k do modo busca com e sem a garantia precisa dos embeddings das 80 perguntas
  (Ollama), então fica para a quinta: `python -m benchmark.run` e
  `RERANK_GARANTE_SECAO=false python -m benchmark.run`. Pode mudar o acerto das perguntas em que a âncora
  era o escolhido de menor semelhança e a seção inferida estava errada.

## 35. Modelo aquecido e KEEP_ALIVE configurável (F3c, 2026-10-24)
**Problema:** a primeira pergunta depois de subir o Ollama paga a carga dos modelos (medido na etapa 0:
~4 s o bge-m3, ~7 s o llama3.2:3b; o qwen2.5:1.5b fica entre os dois), além da leitura do prompt.

- **`KEEP_ALIVE`** (`.env`, padrão `30m`): antes fixo no código. Aceita duração (`30m`, `2h`), segundos
  (`300`) e `-1` (para sempre). O Ollama recusa o texto "-1", então `ollama._keep_alive` o converte em
  número.
- **`ollama.aquecer(s)`**: carrega o modelo de embeddings (toda pergunta começa por ele), o de geração e,
  se for outro, o de verificação (`LLM_MODELO_VERIFICACAO`). Geração: `/api/generate` sem prompt e sem
  stream (o Ollama só carrega). Embeddings: `/api/embed` com um texto curto, porque o `/api/embed` não
  tem "só carregar". Devolve `{modelo: None | motivo}`; nunca levanta, e um modelo que falha (não
  baixado) não impede os outros.
- **Na subida da API** (`main.aquecer_em_segundo_plano`): uma thread daemon, em paralelo com a leitura
  do acervo no banco. Não atrasa a subida e não a derruba: Ollama fora do ar ou modelo não baixado viram
  um aviso no log (o `/saude` já mostra o que falta). Desliga com `AQUECER_MODELOS=false`.
- **Testes nunca aquecem de verdade:** o `conftest.py` põe `AQUECER_MODELOS=false` no ambiente na
  importação (as `Settings` dos testes são criadas antes de qualquer fixture); o teste do aquecimento liga
  a opção por argumento.

**Limites:** só a API aquece (`benchmark.ask` e `benchmark.run` carregam na primeira chamada, como antes).
Se o Ollama subir depois da API, a primeira pergunta ainda paga a carga. Com `uvicorn --reload` cada
reinício aquece de novo (inofensivo: o modelo já está carregado). **Não medido:** o ganho em segundos
na primeira pergunta (precisa do Ollama; quinta).

## 36. Progresso de verdade na tela (F3d, 2026-10-24)
**Problema:** uma pergunta leva de 6 s a ~2 min na CPU, e a tela só dizia "Pensando..." com frases
engraçadas (F1) que não diziam nada sobre o andamento. Quem espera não sabe se está buscando, escrevendo ou
refazendo uma rodada.

**O agente avisa cada etapa** (`services/graph.py`): `montar_grafo(acervo, s, avisar)` e
`responder(..., avisar)` recebem um `Progress = Callable[[etapa, rodada], None]` opcional, chamado quando
começa: `buscando` (nó `buscar`), `escrevendo` (nó `gerar`, só quando o LLM roda: a frase copiada da bula é
instantânea), `conferindo` (nó `validar`) e `verificando` (nó `verificar_llm`, só na rodada 3). O aviso só
informa: erro de quem ouve é engolido e nunca muda a resposta. No "não ajudou" a contagem recomeça em 1
(é uma pergunta nova para o agente).

**Transporte** (`api/progress.py`): NDJSON, uma linha de JSON por evento, via `StreamingResponse`.
- `{"tipo":"etapa","etapa":...,"rodada":n,"de":RODADAS_MAX}`, depois `{"tipo":"resposta","resposta":{AnswerOut}}`
  ou `{"tipo":"erro","erro":{codigo,mensagem},"consulta_id":...,"status":HTTP que o erro teria}`;
- `{"tipo":"vivo"}` a cada 15 s sem novidade (uma geração longa não manda nada por dezenas de segundos, e
  proxies fecham conexão parada);
- o trabalho roda numa thread e fala com o fluxo por uma fila. Se quem espera fechar a conexão, a thread
  termina a pergunta e grava a consulta no histórico: nada se perde;
- falha inesperada (bug): o fluxo acaba sem a linha final, e o cliente trata isso como "resposta
  interrompida" (o erro fica no log da API).

**Rotas novas** (as antigas continuam, com o mesmo resultado): `POST /api/v1/perguntas/com-progresso` e
`POST /api/v1/perguntas/{id}/outra-explicacao/com-progresso`. O que se checa antes do agente (entrada
inválida 422, acervo, banco fora 503, consulta inexistente 404, "não ajudou" de consulta não respondida
422) sai como erro HTTP comum; o que o agente descobre (filtro desconhecido, Ollama fora, rodadas
esgotadas) vai como a linha `erro` do fluxo. `_preparar_outra_explicacao` separou, em `routes.py`, o que
o "não ajudou" checa antes de rodar o agente, para as duas rotas usarem o mesmo código;
`errors.mensagem_banco_fora` guarda o texto do 503 num lugar só.

**Tela:** `api.ts` ganhou `transmitir` (lê o fluxo por `fetch` + `ReadableStream`, junta as linhas que
chegam partidas entre blocos); `perguntar` e `outraExplicacao` passam a usá-lo e recebem `aoEtapa`. `App.tsx`
guarda a etapa na mensagem em espera (`Message.etapa`); `format.rotuloEtapa` escreve o texto ("Buscando
nos trechos das bulas", "Escrevendo a resposta", "Conferindo se a resposta está na bula", "Rodada 2 de 3:
buscando outros trechos nas bulas", "verificando afirmação por afirmação"). A espera mostra a etapa, a frase
engraçada da F1 mais apagada ao lado e os segundos. O leitor de tela agora ouve a etapa (ela é real e muda
poucas vezes); a frase e o contador continuam fora dele.

**Verificado:** `test_progress.py` (ordem, erro, banco que cai, bug, chegada antes do fim, batimento,
cabeçalhos), 5 testes de aviso no grafo em `test_rag.py` e, fora da suíte, o `api.ts` real contra um
servidor FastAPI de mentira com o `progress.py` real, via HTTP e Node: a 1ª etapa chegou em 231 ms
enquanto a resposta só em 1221 ms (o fluxo é incremental de verdade), e erro no fluxo, fluxo cortado,
erro HTTP antes do fluxo e API fora do ar viram `ApiError` certo. **Não verificado:** os 4 testes novos
de `test_api.py` (precisam do PostgreSQL, ficaram pulados), o Vite repassando o fluxo (o proxy do Vite
não costuma guardar a resposta, mas não foi aberto no navegador) e o visual da espera.

## 37. Resposta menos robótica, sem perder a fonte (F4, 2026-10-24)
**Problema:** no teste pela tela (consulta `12f61084`, 17/10), "oque acha do ibuprofeno pra dor de cabeça?"
voltou certo, mas como a frase da bula copiada inteira: termo técnico ("cefaleia tensional"), sem dizer de
onde veio nem o que foi perguntado, e 98 s em 2 rodadas. Pedido do Sérgio: "que a resposta inteira seja menos
robotizada, da abertura até o final", e "o para que serve pode vir diretamente da bula, porém com a explicação
também do agente".

**Por que a resposta saía dura:** o prompt pedia "cada frase diz o que o trecho diz" e a fundamentação (§20,
§28) exige a maioria dos termos da frase no trecho, então o modelo de 1,5B copia. Soltar a checagem quebraria
as duas regras do projeto (só o que a bula diz, toda frase com fonte). **Decisão:** o que o modelo escreve, ou a
frase copiada da bula, continua igual e conferido; o que deixa a resposta com cara de conversa é montado **no
código, em volta dela**, sem LLM e sem passar pela fundamentação. O único texto novo do modelo é a explicação do
"para que serve", e ela é conferida com régua mais alta.

**O que entrou:**
- **Abertura** (`services/presentation.py`, `abertura`): uma frase antes da resposta, pelo caminho que a
  pergunta seguiu no `buscar`: "Pela bula de Ibuprofeno 400 mg:", "Pela posologia de cada bula:", "Sobre usar
  omeprazol com clopidogrel, pelas bulas de...:", "Para quem tem alergia a dipirona, pelas bulas do acervo:",
  "As bulas não dizem qual é o melhor, mas dizem isto:". Pergunta de opinião ("o que acha", "vale a pena",
  "você recomenda") ganha antes "Opinião eu não dou, mas posso dizer o que a bula diz." No "para que serve" a
  conclusão da §29 já abre a resposta, e a abertura fica só com a frase da opinião, se houver.
- **Pergunta de opinião vira "para que serve"** (`indication.py`: "acha" entre os verbos de uso): o caso
  `12f61084` passa a ser respondido pelas indicações, copiadas, em ~2 s, em vez de 2 rodadas do LLM.
- **Marcas na resposta**, como posições no texto (a frase não muda; a tela monta): `foco`, em negrito, as
  palavras que dizem o que foi perguntado, pelos sinônimos que `intent` já tem ("dor de cabeça" -> cefaleia,
  enxaqueca; no máximo 6 e nunca mais da metade do texto, senão o negrito não destaca nada); `termo`, o termo
  técnico com a palavra do dia a dia entre parênteses ("cefaleia (dor de cabeça)"), pelo glossário.
- **Glossário** (`services/glossary.py`, 97 entradas): termo técnico da bula -> palavra do dia a dia, com dois
  usos: a anotação na tela e a fundamentação, que passa a aceitar a expressão do dia a dia inteira ("dor de
  cabeça", "pressão alta", nunca "pressão" solta) contra um trecho que tem o termo técnico dela. Escrito por
  Claude, não tirado das bulas: **precisa da revisão do Sérgio** antes de confiar.
- **Prompt de balcão** (`SISTEMA_BALCAO` em `rag.py`, `LINGUAGEM_SIMPLES=true`): até 4 frases curtas, a
  primeira já respondendo, dizendo de qual remédio é cada informação, podendo trocar o termo técnico pela
  palavra do dia a dia (o prompt traz as do glossário que os trechos usam) e sem conselho que o trecho não traz.
  `LINGUAGEM_SIMPLES=false` volta ao `SISTEMA` anterior, para comparar no benchmark completo.
- **Explicação do "para que serve"** (`services/explanation.py`, nó `explicar`, `EXPLICAR_INDICACAO=true`):
  depois das frases copiadas das indicações, o modelo escreve de 1 a 3 frases curtas em palavras simples. Cada
  frase passa pela fundamentação com `EXPLICACAO_MIN_TERMOS=0.8` (a resposta usa 0,6), não pode repetir a frase
  da bula (Jaccard >= 0,8) e, quando a conclusão foi "não está nas indicações" ou "só dor em geral", não pode
  falar do uso perguntado ("serve para ansiedade e dor de cabeça" passaria pela fração de termos). Sem a
  pergunta no prompt, de propósito: com ela, o modelo pequeno responde à pergunta em vez de explicar. É extra:
  Ollama fora, tempo esgotado ou frase reprovada nunca viram erro, a resposta sai só com a cópia e o motivo fica
  na tentativa. Na ressalva ("não é recomendado... na insuficiência cardíaca") não há explicação: o "não" é o
  que importa, e a paráfrase pode perdê-lo. Etapa nova no progresso (§36): `explicando`.
- **Sugestões de pergunta seguinte** (`presentation.sugestoes`): até 3 botões sobre o mesmo remédio ("Para que
  serve...?", "Qual a dose de...?", "Quem não pode tomar...?", "Quais os efeitos colaterais de...?"), cada uma
  num caminho que o sistema responde bem; sai a da seção já perguntada. Só com um remédio citado e fora da alergia.

**API e banco:** `AnswerOut` e `InquiryDetail` ganharam `abertura`, `explicacao`, `marcas`
(`{inicio, fim, tipo, explicacao}`) e `sugestoes`; colunas novas em `consultas` (nulas nas consultas
anteriores a 24/10). **Tela** (`Message.tsx`, `Citations.tsx`): abertura antes da resposta, negrito e
parênteses montados pelas marcas (sem `dangerouslySetInnerHTML`), "Em outras palavras" com a explicação e as
sugestões como botões que continuam a conversa.

**Verificado em 2026-10-25:** 23 testes novos (`test_presentation.py`, `test_glossary.py`,
`test_explanation.py`) e os ajustes em `test_indication.py`, com LLM falso; build do frontend. **Não
verificado:** a explicação e o prompt de balcão com o LLM de verdade (quinta): medir se as respostas certas
gravadas continuam passando na fundamentação com o `SISTEMA_BALCAO`, e quantas explicações são aprovadas.

## 38. Tela no Docker Compose: nginx na frente do backend (2026-10-28)
**Problema:** o compose subia só o banco e a API. A tela chama `/api/v1/...` no próprio endereço
(`api.ts`), e quem repassava para a API era o proxy do Vite, que só existe no `npm run dev`. Para a entrega
(etapa 7 do `FarmaRAG.pdf`), a tela tem de subir com `docker compose up`, sem Node rodando.

**Decisão:** serviço `frontend` com duas etapas no `frontend/Dockerfile`: o Node (`node:24.22.1-alpine3.24`)
roda `npm ci` + `npm run build` e fica para trás; a imagem final é o nginx (`nginx:1.30.6-alpine3.24`, versão
estável) com a `dist/` e o `frontend/nginx.conf`. O nginx entrega a tela e repassa `/api/` para
`backend:8000`: mesma origem, como no Vite, então o CORS não entra em jogo (`CORS_ORIGENS` fica só para o
desenvolvimento). Alternativa descartada: o FastAPI servir a `dist/` (um container a menos, mas a imagem da
API passaria a construir a tela, e a especificação separa os blocos). Versões exatas das imagens, como no
`package.json`: atualizar é uma decisão.

**Detalhes do `nginx.conf`:**
- `proxy_buffering off` em `/api/`: as etapas do progresso (F3d, §36) saem na hora, não no fim. A API já
  manda `X-Accel-Buffering: no` no fluxo; o nginx respeita os dois.
- `proxy_read_timeout 300s`: as rotas sem progresso esperam a pergunta inteira (`ORCAMENTO_S=150` mais a
  última rodada); a com progresso manda um batimento a cada 15 s.
- `resolver 127.0.0.11` + `proxy_pass` com variável: o nome `backend` é resolvido de novo a cada 10 s, então o
  nginx sobe mesmo antes da API e acha o container novo depois de um restart.
- `assets/` (nome com hash) com cache longo; `index.html` com `no-cache`, para um build novo chegar ao navegador.
- O erro do nginx com a API fora (502, HTML) cai no caminho que a tela já tinha para o proxy do Vite:
  "A API não respondeu", agora com o comando do Docker e o do uvicorn.

**Dados sensíveis (pedido do Sérgio):**
- portas só em `127.0.0.1` (`FRONTEND_PORT`, padrão 8080), como as do banco e da API: ninguém da rede acessa;
- `.dockerignore` no `frontend/` e no `backend/`: `node_modules`, testes, caches e qualquer `.env` ficam fora do
  contexto do build (o `.env` da raiz nunca esteve no contexto: cada build só vê a própria pasta);
- a tela não lê variável de ambiente nenhuma (sem `import.meta.env`) e o build não gera `.map`: o navegador
  recebe só o JS minificado; conferido no `dist/` que não há senha, porta do banco ou nome de variável `DB_`;
- `server_tokens off` (sem versão do nginx), `X-Content-Type-Options`, `X-Frame-Options: DENY`,
  `Referrer-Policy: no-referrer`; arquivo com nome começando por ponto devolve 404; corpo acima de 16 KB é
  recusado (a pergunta tem no máximo 500 caracteres);
- o log de acesso do nginx guarda o caminho (`/api/v1/historico/<id>`), nunca o corpo: a pergunta vai no
  corpo do POST e não aparece no log;
- a senha do banco continua só no `.env`, passada à API em tempo de execução (`env_file`), nunca dentro de imagem.
- Para conferir o compose sem imprimir a senha: `docker compose config --quiet` (sem `--quiet` ele mostra o
  `.env` resolvido).

**Verificado em 2026-10-28:** `docker compose config --quiet` (sintaxe) e `npm run build` (tipos e build) com a
mensagem nova. **Não verificado (quinta, com permissão para baixar as imagens):** `docker compose build`, o
`nginx.conf` dentro do container (`nginx -t`), a tela em http://127.0.0.1:8080, as etapas chegando pelo nginx
durante a espera, e o Ollama do Windows visto de dentro do container (`host.docker.internal`).

## 39. Cache de embeddings no Git: um clone novo carrega o banco sem recalcular (2026-10-28)
**Problema:** `data/cache/` estava no `.gitignore`. Quem clonasse o projeto tinha as bulas em JSON e o
`chunks.jsonl`, mas não os embeddings: antes do `load_database`, precisava rodar o `generate_chunks` com o
Ollama, ~2,5 s por trecho na CPU, perto de 1 h para os 1.384 trechos.

**Decisão (do Sérgio, entre três opções):** versionar o próprio cache, `data/cache/embeddings.sqlite`
(6,2 MB). Descartadas: exportar para um arquivo novo em `data/processed/` (mais código para manter, e o
cache continuaria fora) e só avisar no README do tempo de espera. Com o cache no Git, a ordem num clone
novo é `docker compose up -d db` e `python -m ingestion.load_database`: a carga lê os três arquivos e não
chama o Ollama nem precisa dos PDFs.
- `.gitignore`: `data/cache/*` + `!data/cache/embeddings.sqlite` (com `data/cache/` a pasta inteira fica
  ignorada e a exceção não vale). O demo e as cópias continuam fora.
- `.gitattributes`: o arquivo é `binary` (sem diff nem conversão de fim de linha).
- Cada versão do cache que entra no Git pesa ~6 MB no histórico: commitar só quando o acervo mudar.

**Dados sensíveis:** cada linha guarda só o SHA-256 de (modelo, texto) e o vetor, nunca o texto. Só a
ingestão grava nele, e só trechos de bula (texto público); as perguntas da API e do benchmark vão direto ao
Ollama e não passam pelo cache. Mesmo assim havia **10 linhas que não eram de nenhum trecho atual**
(provavelmente versões antigas, de antes da tabela do Omenax e do lote 3) e, pelo hash, não dá para provar
o que eram. Por isso o cache vai **podado**: só os embeddings dos trechos atuais.
- `EmbeddingCache.sobras` (conta) e `EmbeddingCache.podar` (apaga e roda `VACUUM`, que reescreve o arquivo:
  nada apagado fica nas páginas livres).
- `python -m ingestion.generate_chunks --podar`: gera os trechos, completa os embeddings e poda. Sem a
  opção, o fim da ingestão avisa quantas linhas sobram ("Antes de commitar o cache: --podar"). A poda não é
  automática porque o cache serve também para experimentos (outro `CHUNK_TOKENS`, outro modelo de
  embeddings): voltar à configuração anterior custaria recalcular tudo.
- Podado em 2026-10-28: 1.394 → 1.384 linhas, 0 sobrando, 0 páginas livres; `chunks.jsonl` idêntico ao
  do commit. A cópia de antes ficou em `data/cache/embeddings.antes-da-poda-2026-10-28.sqlite` (fora do Git).

**Verificado:** 4 testes em `test_embedding_cache.py`: poda por texto e por modelo, `--podar` com tudo no
cache sem chamar o Ollama, `--podar --sem-embeddings` recusado, e o cache versionado cobrindo todos os
trechos do `chunks.jsonl` com o modelo padrão (se alguém mudar os trechos e esquecer o cache, a suíte
falha). Num clone simulado (só os arquivos que o Git levaria, sem `.env`), com o Ollama apontado para uma
porta fechada: 37 bulas, 1.384 trechos e a matriz (1.384, 1.024) carregados. Suíte: 392 passam, 40 de
banco pulados (PostgreSQL desligado). **Não verificado (quinta):** o `load_database` gravando no banco a
partir de um clone novo.

## 40. README do zero e a carga do acervo pelo Docker (2026-10-28)
**Problema:** a etapa 7 da especificação pergunta se alguém consegue rodar o projeto só seguindo o README.
Não conseguia: o "Como rodar" mandava rodar `process_batch`, que precisa dos PDFs (`data/raw/` fica fora do
Git), e só carregava o banco com Python instalado no Windows. Com o acervo inteiro no Git (§39), a carga
só copia arquivos e pode rodar em qualquer lugar.

**Decisão:**
- **Serviço `carga` no compose**, no perfil `carga` (o `docker compose up` não o sobe): a mesma imagem do
  backend, com `data/` e `ingestion/` montados só leitura, roda `python -m ingestion.load_database` contra o
  serviço `db`. Uso: `docker compose run --rm carga`. O caminho pelo Docker passa a pedir só Git, Docker
  Desktop e Ollama. A §7 continua valendo para a ingestão de verdade (PDF → trechos → embeddings), que roda
  no Windows: a carga é só a cópia para o banco.
- **O cache é aberto só para leitura na carga** (`EmbeddingCache(..., somente_leitura=True)` em
  `corpus.carregar_dataset`, via URI `mode=ro`): quem só lê nunca cria nem altera o arquivo versionado, e a
  montagem `:ro` do container não depende de o SQLite cair sozinho para leitura. Sem o arquivo, a carga
  explica ("Sem o cache de embeddings... ele vem no Git") em vez de criar um cache vazio e dizer que faltam
  1.384 embeddings.
- **README, "Como rodar" em três partes:** do zero pelo Docker (clone, modelos, `.env`, `db`, `carga`,
  `up --build`, tabela de comandos do dia a dia), desenvolvimento (venv, uvicorn, Vite, testes) e adicionar
  bulas (ingestão com `--podar` e o que commitar junto). Avisos que custaram tempo no desenvolvimento ficaram
  no texto: `127.0.0.1` em vez de `localhost`, `npm.cmd` no PowerShell, `--reload` que não recarrega, API
  subida da raiz.

**Dados sensíveis:** o `.env` é criado na máquina a partir do `.env.example` e está no `.gitignore`; o README
avisa que `docker compose config` sem `--quiet` imprime a senha e que `down -v` apaga o histórico de
consultas. A carga e a API recebem a senha só em tempo de execução (`env_file`).

**Verificado em 2026-10-28:** `docker compose config --quiet` e a lista de serviços com e sem o perfil
`carga`; 3 testes novos em `test_embedding_cache.py` (o modo só leitura recusa gravar; a carga lê um cache
protegido contra escrita sem mudar um byte; sem o cache, a mensagem certa e nenhum arquivo criado); suíte com
395 passando e 40 de banco pulados. **Não verificado (quinta, com permissão para baixar as imagens):** o
passo a passo inteiro num clone novo de verdade (`git clone` → `carga` → `up --build` → pergunta pela tela em
:8080), incluindo o Ollama do Windows visto de dentro do container.

## 41. Mais de uma apresentação do mesmo remédio: escolher pela pergunta ou perguntar qual (2026-10-28)
**Problema:** o acervo tem 5 remédios com mais de uma bula: paracetamol (Tylalgin gotas, comprimido 750 mg e
sachê), permetrina (Kaodine loção e loção cremosa), AAS (100 e 500 mg), isotretinoína (10 e 20 mg) e
colecalciferol (duas bulas do Delumi). A busca juntava trechos de todas, e para dose, modo de usar, superdose,
armazenamento ou criança o modelo pequeno podia juntar a dose de uma com a forma de outra. A §14 já previa
perguntar ("nome ambíguo... precisa_esclarecimento"), mas o nó que perguntava saiu na §20 e só ficou o contrato.

**Decisão** (`services/dosage_form.py`, sem LLM):
- **A pergunta escolhe a bula quando já diz qual** (`escolhidas`): as palavras e números que só aquela bula tem no
  nome e na forma farmacêutica ("gotas"/"gotinhas", "750", "sachê"/"sachet", "cremosa", "AAS 100", "Delumi
  50.000"). A busca, as frases copiadas (dose, interação, indicação) e as rodadas seguintes ficam nela. Número
  que mede outra coisa não conta ("criança de 20 kg", "a cada 6 horas"). Nomeou todas = não escolheu.
- **Pergunta qual quando não diz** (`_apresentacoes_a_perguntar` no `buscar`), só se tudo isto vale: um remédio
  só na pergunta; mais de uma bula dele ainda possível e de **apresentações diferentes** (forma ou números do
  nome; o mesmo produto de dois laboratórios não é outra apresentação); assunto que muda com a apresentação
  (`depende`: posologia, superdose, armazenamento, criança, aplicar/diluir/preparar); e a resposta iria para o
  LLM. A dose copiada da bula (§25), que já vem uma frase por apresentação, passa antes e não pergunta; para que
  serve, interação e contraindicação também não. Nunca no "não ajudou", na alergia, nem na resposta a um
  esclarecimento (no máximo um por pergunta, §14).
- **A pergunta de volta** (`status = precisa_esclarecimento`, motivo novo `apresentacao`): "Tenho 3 bulas de
  paracetamol, e a resposta muda conforme a apresentação. Qual delas?", um botão por bula ("Tylalgin 200 mg/mL
  gotas", "Tylalgin 750 mg (comprimido)", "Tylalgin Sachet 500 mg": a forma entre parênteses quando o nome não
  a diz). Sai antes da geração, sem gastar o LLM (`rodadas = 0`).
- **O botão reenvia a pergunta com `bula_id`** (campo novo em `QuestionIn` e em `OptionOut`, o mesmo nome da
  especificação): vale mais que `principios_ativos` (o remédio é o da bula); bula fora do acervo dá
  **404 `BULA_NAO_ENCONTRADA`**, checado antes do agente (na rota com progresso, como erro HTTP comum). Escrever
  a resposta no chat ("o comprimido") também escolhe, pelas mesmas palavras.
- **A conversa continua na apresentação escolhida**: `consultas` ganhou `bula_id_filtro` (o botão, para refazer a
  pergunta igual no "não ajudou" e ao reabrir pelo histórico) e `bulas` (as bulas a que a busca ficou presa);
  "e para criança?" depois de "Tylalgin gotas, qual a dose?" herda as gotas (`PreviousQuestion.bulas`).
- **Busca:** `VectorSearch.buscar(..., bulas)`, no `InMemoryIndex` e no `PgvectorSearch`: o princípio ativo que
  tem alguma das bulas só busca nelas; os outros remédios da pergunta continuam em todas as bulas deles.
- **Assunto da resposta:** a forma farmacêutica das bulas do remédio conta como nome, não como assunto (como o
  nome comercial e o fabricante). Sem isso, "o comprimido" escrito em resposta exigia a palavra "comprimido" na
  resposta e reprovava nas 3 rodadas (achado no teste).
- **Benchmark:** quando o agente pergunta a apresentação, `benchmark.run` responde como o balconista, com a bula
  que tem a âncora do gabarito, e grava `apresentacao_escolhida`; o tempo conta as duas chamadas. O
  `benchmark.ask` oferece as opções e reenvia com a bula.

**Pelo texto do gabarito (sem rodar, estimativa de 2026-10-28):** 6 perguntas passam a ficar numa bula só pela
própria pergunta (pos01 Tylalgin 750, pos11 Kaodine loção 10 mg/mL, pos12 AAS 500, sup01 gotas, comp02 Sachet,
fora05) e 3 passam a perguntar a apresentação (adv10 criança + AAS, cons03 Kaodine "loção", cons04 Delumi na
geladeira). Um falso positivo achado assim e corrigido: "cliente **bebe** álcool" casava com "bebê".

**Verificado:** 30 testes em `test_dosage_form.py` (escolha pela pergunta, assunto, mesmo produto de dois
laboratórios, botões, e o grafo com LLM falso: pergunta sem chamar o modelo, botão, resposta escrita, sem segundo
esclarecimento, conversa que herda, interação sem perguntar, dose copiada por apresentação) e 1 em
`test_vector_search.py`; suíte com 426 passando. **Não verificado (quinta, com o banco):** 2 testes de API
(404 e o `bula_id_filtro` no histórico) e 4 do `PgvectorSearch` com o filtro por bula; a tela com os botões novos.

## 42. O conselho que o trecho não traz sai da resposta, e "médico" não casa com "medicamento" (2026-10-28)
**Problema:** no caso "rocutan" (§28) a resposta aprovada terminava em "É recomendado consultar um médico",
conselho que o trecho citado (a bula do diclofenaco) não dava. A F4 (§37) pôs no prompt de balcão "não acrescente
conselho que o trecho não traz", mas o modelo de 1,5B escreve isso mesmo assim, e a fundamentação deixava passar:
"médico" e "medicamento" começam com as mesmas 5 letras, e todo trecho de bula tem "medicamento". O conselho
genérico também não faz falta: o aviso fixo da F2 (§33) já manda conferir com o farmacêutico e o médico.

**Decisão** (`grounding.sem_conselho`, no `validar`, antes de `corrigir_citacoes`, sem LLM):
- **Conselho** é a parte da frase com um verbo ou substantivo de conselho e, depois dele, quem procurar:
  "consulte um médico", "procure orientação médica", "fale com o farmacêutico", "informe seu médico", "em caso de
  dúvida, ..." (`_CONSELHO`). "O médico deve avaliar a função renal" não é conselho: a fundamentação decide.
- **Se o trecho citado fala de médico, farmacêutico, dentista ou profissional de saúde** ("em caso de superdose,
  procure atendimento médico"), o conselho é da bula e fica. O cabeçalho "Bula para o profissional de saúde", que
  vem no texto de 23 trechos, não conta. Nos trechos reais, 274 de 1.384 falam de algum desses.
- **Se não fala, o conselho sai em silêncio**: a parte da frase ("contraindicado na dengue, consulte um médico" ->
  "contraindicado na dengue."), a frase, ou a afirmação inteira. O que saiu vai para `Attempt.correcoes` ("conselho
  que o trecho [n] não traz, saiu/cortado"), como a citação corrigida da §31. Sem custo de rodada.
- **Resposta que era só conselho** reprova a rodada ("a resposta só traz conselho que os trechos não trazem"), com
  o aviso `DICA_ASSUNTO` (responda ao que foi perguntado): não há nada da bula nela.
- **"médico", "médica", "médicos" e "médicas" só valem inteiros** na fundamentação (`Reference.tem`): "consulte um
  médico" contra um trecho sem médico agora tem os dois termos fora. Vale também para a explicação do para que
  serve (§37), que já reprovava a maioria desses conselhos pela régua de 0,8.

**Verificado:** 10 testes novos em `test_grounding.py` (cortes, o trecho que traz o conselho, o cabeçalho,
"médico" contra "medicamento") e 2 em `test_rag.py` (o conselho sai sem custar rodada; a resposta só de conselho
reprova e a rodada 2 recebe o aviso); suíte com 438 passando. **Não verificado (quinta, com o banco):** quantas
respostas aprovadas no histórico mudariam (como na §28 e na §31: rodar `sem_conselho` e a fundamentação nova sobre
as `tentativas` gravadas) e o efeito no benchmark completo.

## 43. Lote 2 do gabarito, escrito sem olhar o que o sistema responde (2026-10-28)
**Problema:** a especificação pede "escreva as perguntas antes de olhar o que o sistema responde, para não viciar
o teste". As 80 perguntas do lote 1 (§30) foram escritas assim, mas depois os pesos do reranking (§30, §34) e várias
regras (sigla "AAS", alergia, uso com dois remédios) foram ajustados olhando os resultados delas: o 57/70 do modo
busca é, em parte, treino. Falta um conjunto que meça a generalização.

**Decisão:** `data/benchmark/perguntas_lote2.jsonl`, **41 perguntas (35 com resposta, 6 sem)**, ids `l2-...`,
escritas em 2026-10-28 lendo só o texto das bulas (`data/processed/bulas/`) e as perguntas do lote 1 (para não
repetir), sem rodar o sistema nem abrir resultado. Mesmo contrato do lote 1 (âncora = princípio ativo, seção e uma
frase copiada da bula).
- **Cobertura:** os 7 princípios ativos que o lote 1 não tinha (acetilcisteína, ácido fólico, albendazol, ambroxol,
  dexclorfeniramina, tiamina, nafazolina) e assuntos novos dos outros (omeprazol antes das refeições e cápsula
  aberta, zolpidem por no máximo 4 semanas, Pozato e vômito, digoxina com diclofenaco, pimozida com sertralina...).
- **Categorias:** posologia 13, contraindicação 5, advertência 4, interação 7, reação adversa 1, indicação 2,
  conservação 1, superdose 1, composição 1, fora do escopo 6. Inclui duas perguntas de remédio com mais de uma
  apresentação (Delumi 50.000, Kaodine loção cremosa), que exercitam a §41.
- **Fora do escopo** conferido pelo texto: nenhum trecho fala de boldo, calvície, horário da losartana ou preço; a
  do cachorro tem trechos de estudo em cães na bula da acetilcisteína (a pergunta é sobre amoxicilina), o que testa
  a recusa contra um falso parecido. A pergunta "a losartana genérica funciona igual à de marca?" saiu: o rótulo
  "medicamento genérico, Lei nº 9.787" de várias bulas ficava perto demais.
- **Rodar:** `python -m benchmark.run --perguntas data/benchmark/perguntas_lote2.jsonl --rotulo lote2` (modo busca)
  e `--modo completo` na quinta. Comparar com o lote 1: se o lote 2 acertar bem menos, os ajustes decoraram o lote 1.

**Teste de contrato novo** (`test_dataset_contract.py`): os dois gabaritos seguem o `BenchmarkQuestion`, **toda
âncora está em algum trecho do `chunks.jsonl`** e os ids não se repetem entre os lotes. O teste antigo procurava
`data/benchmark/perguntas.json` (o nome da especificação) e passava vazio: o arquivo real é `.jsonl`. As 106 âncoras
dos dois lotes passam. Assim, a resposta à pergunta da especificação "se a bula mudar, quais perguntas do benchmark
quebram?" sai na suíte, antes de rodar o benchmark.

## 44. Docker endurecido, análise de segurança e a F3b medida (2026-10-29)
Pedido do usuário: fechar o que desse, configurar o Docker, analisar a segurança e terminar. Esta seção
junta o que mudou e por quê; o modelo de ameaça, a lista de achados e os riscos aceitos estão em
[`SEGURANCA.md`](SEGURANCA.md).

### Os testes de banco voltaram a rodar (e 4 estavam defasados)
Desde a F3d os testes que precisam do PostgreSQL ficavam pulados quando o banco estava desligado, e
desde então foram escritas a F3b, a F4, o `bula_id` e a forma farmacêutica. Com o banco ligado, **22 falharam**;
as causas eram 4:
1. **`pytest.fail` dentro da API derruba o módulo todo.** Ele roda na thread do TestClient, mata o "portal"
   e o erro "This portal is not running" derrubava os testes seguintes em cascata (22 falhas por 1 causa).
   Troquei por um registro de chamadas com asserção depois.
2. **F4:** "para que serve" agora chama o LLM uma vez, para a explicação em palavras simples (opcional, nunca
   vira erro). O teste esperava "sem LLM". O LLM falso agora trata essa chamada à parte (a explicação não
   consome uma das saídas combinadas) e o teste confere que só ela chamou o modelo.
3. **`bula_id`** na opção do esclarecimento: o teste antigo esperava o formato sem o campo.
4. **F3b:** "características farmacológicas" deixou de ser o 2º trecho de "O que é o Exemplol?" (§34), e os dois
   testes do "não ajudou" (de 20/10) esperavam isso. Reescritos com a ordem de semelhança nova e a intenção
   original (evitar os citados; enviados e não citados depois dos nunca enviados). O comportamento estava
   certo; o gabarito do teste é que era de antes da F3b.

Resultado: **todos os testes passam com o banco ligado, 0 pulados.**

### A F3b, medida (modo busca, 2026-10-29)
`python -m benchmark.run` com e sem `RERANK_GARANTE_SECAO`:

| Lote | F3b ligada | F3b desligada | Perguntas com acerto diferente | Perguntas com trechos diferentes |
|---|---|---|---|---|
| 1 (80, escrito antes) | 58/70 | 58/70 | 0 | 8 de 80 |
| 2 (41, escrito sem olhar o sistema) | 30/35 | 30/35 | 0 | 4 de 41 |

A F3b muda **os trechos enviados em 12 de 121 perguntas**, trocando "resultados de eficácia" e "características
farmacológicas" por seções que a pergunta pede, e **não muda o acerto de nenhuma**. É o efeito pretendido
(prompt mais curto e mais relevante, sem perda). **Decisão: continua ligada.** O lote 2, escrito sem olhar
os resultados, acerta 30/35 (86%) contra 58/70 (83%) do lote 1: os pesos do reranking generalizam.
`RERANK_CANDIDATOS` continua 10 (§32: 25 ou mais só trocava uma pergunta).

### `/saude`: "ocupado" não é "fora do ar"
Na gravação da demonstração apareceu a faixa "Ollama não respondeu em 3 s" durante toda pergunta longa: a
sondagem do Ollama tem 3 s de limite, e na CPU o servidor passa 50-90 s gerando. `ollama._pedir` já separa
**estouro de tempo** (`TEMPO_ESGOTADO`: o servidor aceitou a conexão) de **conexão recusada**
(`OLLAMA_INDISPONIVEL`). O `/saude` passa a tratar o primeiro como "no ar, ocupado" e só o segundo como
problema.

### Docker
- **Imagens por digest** (verificadas no Docker Hub em 2026-10-29): `python:3.12-slim`, `node:24.22.1-alpine3.24`
  e `nginxinc/nginx-unprivileged:1.30.6-alpine3.24` (mantida pelo time do NGINX; mesma versão estável que o
  projeto já usava, mas roda como usuário `nginx`, sem root).
- **Dependências Python travadas:** `backend/requirements.lock` (54 pacotes), gerado por `backend/lock_deps.py`
  a partir do que está instalado e testado. A imagem instala dele; o `requirements.txt` continua sendo a
  lista legível.
- **Containers:** backend sem root (uid 10001), raiz somente leitura (só `/tmp` em memória), `cap_drop: ALL`,
  `no-new-privileges`, teto de memória e processos, reinício automático e healthchecks. O backend usa o
  `/docs`, não o `/saude`: o `/saude` sonda o banco e o Ollama e ficaria lento numa pergunta longa.
  `frontend` só sobe com o backend saudável. O banco ganhou `no-new-privileges` e reinício (o resto da lista
  exige ajuste fino na imagem do PostgreSQL).
- **nginx:** escuta a 8080 por dentro (sem root não abre a 80); a porta publicada no host continua
  `127.0.0.1:8080`. CSP restritiva, `Permissions-Policy`, COOP e CORP. A tela não usa estilo nem script inline,
  então a política não precisa de `'unsafe-inline'`.
- **Falhas das imagens (`docker scout`):** a base do nginx trazia `brotli` (CVE explorada, na lista da CISA),
  `curl` e `zlib` com correção já publicada. O `frontend/Dockerfile` roda `apk upgrade` e remove o `curl`.
  No backend, a "crítica" era o `certifi` antigo (corrigido no lock). Ficaram só `zlib` (Debian) e
  `libxml2` (Alpine), sem versão corrigida (risco aceito, em `SEGURANCA.md`).
- **LangSmith desligado** no Compose (`LANGSMITH_TRACING` e `LANGCHAIN_TRACING_V2` falsos): o LangGraph vem do
  ecossistema LangChain, que sabe mandar rastros para a nuvem.

**Verificado:** clone novo (pasta só com o que vai ao Git, `.env` do exemplo, banco novo, portas diferentes):
`up -d db`, `run --rm carga` (37 bulas e 1.384 trechos), `up -d --build`, os três containers saudáveis,
`/saude` ok. Pilha principal: pergunta que passa pelo LLM pela tela do Docker, com as etapas chegando em tempo
real pelo nginx (5,2 s "Buscando", 6,2 s "Escrevendo", resposta em 61 s) e **zero violações de CSP** no
console. **Não verificado:** `docker scout` em CI, e a pilha em Linux (todo o teste foi no Docker Desktop do
Windows; o `host.docker.internal:host-gateway` do Compose é o que o Linux também aceita).

### Correções de segurança no código
Ver os achados 4 a 7 de `SEGURANCA.md`: teto de perguntas em andamento (`PERGUNTAS_SIMULTANEAS`, 429
`SERVIDOR_OCUPADO`), `nosniff` e `no-store` nas respostas da API, limite de 200 caracteres por nome no filtro,
só `http(s)` em `OLLAMA_URL`, aviso no log para senha de exemplo, e na tela o aviso "Não escreva nome nem dado de
paciente". Todos têm teste (`test_security.py` e os de 429 em `test_api.py`).

### O que não foi feito
- **As 12 bulas que faltam** (naproxeno, cetirizina, loperamida, bisacodil, lactulose, escopolamina,
  dimenidrinato, guaifenesina, benzidamina, ácido ascórbico, sulfato ferroso, minoxidil): o Bulário usa
  Cloudflare e o download é manual (`data/FONTES.md`). O MVP fica em 28 de 40, mais 3 extras.
- **Meta de latência (mediana < 60 s):** não atingida com o modelo de 1,5B na CPU. A saída é `CHUNK_TOKENS`
  menor (exige refazer ~1.384 embeddings) ou outro hardware.
- **Limites conhecidos do agente:** a regra 6 (§45) pega a permissão explícita contra uma proibição clara, não
  toda inversão de sentido; subseções como metadado e cross-encoder só valem se o benchmark pedir.

## 45. Regra 6: a resposta que afirma o que a bula proíbe (2026-10-29)
Fechava o último limite conhecido da fundamentação. A regra 5 (§28) pega a **negação inventada** ("não é
indicado" citando um trecho que diz "é indicado"); faltava o caminho inverso, a **permissão inventada**
("pode ser usado na gravidez" citando "contraindicado na gravidez"). Numa resposta de saúde, é o erro mais
perigoso dos dois.

**A regra** (`grounding.afirma_o_que_o_trecho_proibe`, mensagem `AFIRMA`): uma frase da afirmação diz que o
remédio **pode ser usado**, e todas as frases do trecho que a sustentam proíbem. Estreita de propósito:
- **Permissão** na resposta só é "pode ser usado/utilizado/tomado/administrado", "pode usar/tomar",
  "permitido", "liberado", "seguro". "Pode" sozinho **não** conta: aparece em quase toda frase de bula
  ("pode causar sono", "pode reduzir o efeito") e reprovaria resposta certa. Se a frase também nega, evita ou
  proíbe ("pode, mas evite..."), não é permissão pura.
- **Proibição** na bula é "contraindicado", "proibido", "evite/evitar/evitado" e "não deve/use/tome/pode/
  recomendado". Fica de fora "evita" (3ª pessoa: "evita a formação de coágulos") e a frase que usa a palavra
  para liberar ("não é contraindicado", "não há contraindicação").
- **Apoio:** as frases do destaque (as mesmas da regra 5). Sem nenhuma (pouco em comum com o trecho, como em
  "É seguro na gravidez"), só decide se o trecho **inteiro** é feito de proibições; senão, a regra se abstém.
  Abster-se é o lado seguro: a regra só reprova quando tem certeza de qual frase da bula a afirmação contradiz.

Reprovada, a rodada seguinte recebe um aviso próprio (`DICA_AFIRMACAO`), no mesmo esquema das regras 3 e 5.

**Medido, para ter certeza de que não reprova resposta certa:**

| Amostra | Pares conferidos | Sinalizados |
|---|---|---|
| Respostas **aprovadas** do histórico real (92, 17 copiadas da bula) | 146 afirmação-trecho | **0** |
| Respostas **esperadas** do gabarito, nos trechos-âncora (121 perguntas, escritas à mão) | 123 | **0** |

**Sensibilidade:** a permissão inventada "O medicamento pode ser usado por qualquer paciente", citando cada um
dos 47 trechos de contraindicação do acervo, é reprovada em **24**. Nos outros 23 há alguma frase que não é
proibição pura e a regra se abstém. É uma rede de segurança parcial e sem falsos positivos medidos, não uma
garantia: a conferência por LLM da rodada 3 e a revisão humana do balcão continuam sendo necessárias.

Testes: `test_grounding.py` (permissões que devem ser pegas; possibilidade, apoio vizinho, ressalva, bula que
também permite, "não é contraindicado" e "evita" que não devem).


## 46. Benchmark completo com o LLM real: o que o "acerto" não mede (2026-10-29)
Primeira medição dos dois lotes (80 + 41 perguntas) com o `qwen2.5:1.5b` de verdade, semente 42, `top_k` 3,
reranking de 10 candidatos (arquivos `completo_2026-10-29_*` em `data/benchmark/resultados/`). O lote 1 rodou
antes da regra 6 (§45) e o lote 2 depois.

### O "acerto" mede a busca, não a resposta
`acerto_rodada1` diz se a frase-âncora estava entre os trechos que o modelo recebeu. Três respostas marcadas
como acerto estão **erradas** (`pos06`, `con06`, `adv03`): o trecho certo foi enviado e o modelo o contrariou.
Por isso as 88 respostas dadas foram lidas contra a resposta esperada do gabarito (leitura manual; o benchmark
não tem métrica automática de qualidade da resposta, e escrever uma é o próximo passo, abaixo).

| | Lote 1 | Lote 2 | Total |
|---|---|---|---|
| Perguntas com resposta na bula | 70 | 35 | 105 |
| Respondidas | 64 | 24 | 88 |
| Adequadas | 48 | 16 | **64** |
| Erradas | 6 | 3 | **9** |
| Fora do assunto ou incompletas | 10 | 5 | 15 |
| Terminaram em "não encontrei" ou erro do modelo | 6 | 11 | 17 |
| Perguntas sem resposta na bula: recusa certa | 5/10 | 2/6 | 7/16 |

No lote 1, as 5 sem recusa certa são 3 erros 502 (JSON inválido do modelo) e 2 pedidos de esclarecimento: nenhuma
resposta inventada. No lote 2, **3 foram respondidas indevidamente** (abaixo). Mediana 31,6 s (lote 1) e 24,8 s
(lote 2): a mediana é baixa porque 11 das 70 perguntas do lote 1 vão pelo caminho de cópia da bula, sem LLM; nas
respostas lidas que passaram pelo modelo, o tempo ficou entre 40 e 95 s (a meta de 60 s de mediana para o LLM
não é atingida na CPU).

### Os defeitos, com o trecho que os explica
- **`pos06`** "pode tomar outro comprimido se acordar?" → "Sim". O trecho enviado diz "não deve ser readministrado
  na mesma noite". A regra 6 não pega: a frase da bula de apoio é outra ("o comprimido deve ser tomado com
  líquido"), e "readministrar" não é sinônimo que a regra conheça.
- **`adv03`** "mãe que amamenta pode tomar sertralina?" → "a amamentação é um uso recomendado". O trecho diz "o uso
  em lactantes **não é recomendado** a menos que...". A regra 6 só reconhece "pode ser usado/tomado", "permitido",
  "liberado" e "seguro" como permissão.
- **`con08`** úlcera ativa e AAS → "Sim, pode ser usado". Aqui a busca falhou antes: o trecho de contraindicação
  não entrou entre os 3 enviados (só o de "usar com cautela em história de úlcera"), e o modelo concluiu "sim".
- **`con06`** metformina com TFG < 30 → "Sim, é recomendado verificar a função renal": o trecho de contraindicação
  estava entre os enviados, mas o modelo citou o de advertência e abriu com "Sim", sem sustentação.
- **Fora do escopo respondido** (lote 2): `fora01` (preço do Cozaar) e `fora03` (dose para cachorro) foram
  atendidas pelo caminho de resposta copiada da bula, que devolve a posologia do remédio citado sem checar se a
  pergunta é sobre dose; `fora04` ("omeprazol com chá de boldo") foi respondida com um trecho de interações que
  não contém "boldo", e a fundamentação aceitou porque "omeprazol" e "interação" estavam lá (regra 3 só barra
  quando a frase tem *só* o nome).
- **Fora do assunto**: a checagem "a resposta não fala do que foi perguntado" é frouxa: `adv09` (fígado) foi
  respondida com amamentação, `adv08` ("por quantos dias no máximo") com a dose pediátrica.
- **Variação entre execuções**, mesmo com semente fixa: `con09` ora termina em "não encontrei", ora responde
  "Sim, pode começar a sertralina" onde a bula manda esperar 14 dias (afirmação errada que passou na fundamentação).

### O que se conclui e o que não
- Vale dizer: citação verificada no código, recusa quando a bula não sustenta, 0 resposta inventada nas 10 perguntas
  sem resposta do lote 1.
- **Não vale dizer** "prefere recusar a errar": 9 de 88 respostas contradizem a bula e 3 de 6 perguntas fora do
  escopo do lote 2 foram respondidas. O README e os textos de apresentação foram corrigidos para isso.
- As regras de fundamentação (1 a 6) são conferência lexical. Elas não leem o sentido. Um modelo maior ou uma
  conferência semântica é o que falta, e as duas coisas custam latência na CPU.

### Próximos passos (não feitos hoje; mexem no comportamento e pedem novo benchmark completo, ~1 h)
1. **Guarda de escopo** antes da busca: preço, animal/veterinário e posologia pedida sem remédio do acervo
   respondem "fora do escopo" sem passar pelo modelo (corrige `fora01` e `fora03`).
2. **Sim/não com apoio:** abertura "Sim"/"Não" só se o trecho citado tem permissão ou proibição explícita; senão,
   cortar a abertura e deixar a frase da bula (pega `pos06`, `adv03`, `con06`); incluir "recomendado" e
   "readministr*" na regra 6 só depois de medir falso positivo nos 146 + 123 pares, como foi feito na §45.
3. **Termo da pergunta no trecho:** exigir que o substantivo novo da pergunta ("boldo", "cachorro") apareça no
   trecho citado, senão recusar (`fora04`).
4. **Métrica de qualidade da resposta** no benchmark (termos obrigatórios e proibidos por pergunta), para que o
   "acerto" deixe de ser lido como acerto da resposta.
5. Testar `qwen2.5:7b` só nas perguntas que o 1,5B erra, para saber quanto do problema é o tamanho do modelo.
