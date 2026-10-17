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
Código: `benchmark/metricas.py`; contrato: `PerguntaBenchmark` em `schemas/dataset.py`.

## 3. Pergunta com mais de um medicamento
Suportamos até 3 (`MAX_MEDICAMENTOS_POR_PERGUNTA`). A busca divide o `k` entre os medicamentos
(`IndiceEmMemoria.buscar`), então "ibuprofeno + losartana" recebe trechos das duas bulas.

## 4. Medicamento não indexado
`services/medicamentos.py` detecta, antes do LLM, nomes na pergunta (princípio ativo, nome
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
`ingestao/` e `benchmark/` importam `app` (cliente do Ollama, schemas, config). Instale com
`pip install -e backend` e rode da raiz: `python -m ingestao.extrair_pdf`.

## 9. Códigos de erro
Adicionado `SAIDA_INVALIDA` (502) para JSON quebrado; `CITACAO_INVALIDA` fica para citação
inexistente/incoerente. `BULA_NAO_ENCONTRADA` virou `MEDICAMENTO_NAO_ENCONTRADO`.

## 10. Limite conhecido da validação de citação
`services/citacoes.py` garante que as citações apontam para trechos enviados. Não garante que a
afirmação está sustentada: isso é a métrica de fidelidade (revisão manual) do benchmark.

## 11. Banco: PostgreSQL 17 + pgvector (2026-10-14)
Volta ao que o PDF previa. O MySQL ficou só no código (nunca foi instalado nem recebeu dados), então
não houve dado a migrar: o banco é preenchido a partir dos arquivos por `ingestao/carregar_banco.py`.
- Container `pgvector/pgvector:pg17` (serviço `db` do `docker-compose.yml`, volume `pgdata`). No
  Windows nativo o pgvector precisa ser compilado com Visual Studio; o container evita isso.
- Driver `psycopg[binary]` 3 (`postgresql+psycopg://`) e o pacote `pgvector` para o tipo `vector`.
- Modelos: listas em `ARRAY(Text)`, UUID nativo, `timestamptz` em UTC, `Text` sem tamanho,
  `chunks.embedding` em `vector(1024)` (`EMBED_DIM` em `db/models.py`, conferido contra o `.env`).
- `chunks.principio_ativo_norm` = `normalizar(principio_ativo)`: o filtro da busca no banco compara
  igual ao `IndiceEmMemoria`.
- `BuscaPgvector` (`db/busca_pgvector.py`) implementa `BuscaVetorial` com `<=>` (distância de
  cosseno) e a mesma divisão do k entre princípios ativos. O teste `test_busca_pgvector.py` confere
  que devolve os mesmos trechos e scores do numpy. O `IndiceEmMemoria` continua no fluxo por arquivos.
- **Sem índice HNSW**: com poucos milhares de trechos a varredura exata leva milissegundos e dá o
  top-k exato; o HNSW é aproximado e, com o filtro por princípio ativo, pode devolver menos de k.
- `carregar_banco.py` cria a extensão e as tabelas que faltam (`create_all`, sem Alembic por ora) e
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
ativos distintos) é versionado. Ao adaptar `Catalogo.montar` para usá-lo:
- Para os medicamentos **indexados**, usar princípio ativo + nomes comerciais do catálogo.
- Para os **não indexados** (`medicamentos_conhecidos`), usar só princípios ativos de produtos
  de um único princípio, não nomes comerciais: há marcas curtas ou ambíguas (ex.: "CORUS", "DIP")
  que gerariam falso positivo em perguntas normais.
- Sais: o catálogo traz "losartana potássica", a pergunta diz "losartana"; o alias da primeira
  palavra já cobre isso (ver `_aliases` em `services/medicamentos.py`).

**Implementado em 2026-10-13** (`Catalogo.montar(bulas, conhecidos, anvisa, ignorados)`):
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
  "Clavulin"): a pergunta segue sem filtro. Princípio ativo que só existe em associação
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

Mudanças de contrato decorrentes: `status` ganha `precisa_esclarecimento` e `CodigoErro` ganha
`RESPOSTA_NAO_VERIFICADA` (502), já feitos; env `RODADAS_MAX=3` e `ORCAMENTO_S=150`, já feitos.
Com a API (2026-10-16, seção 18): `consultas` ganhou `rodadas` (int), `tentativas` (JSON) e
`consulta_origem_id` (UUID, sem FK).

## 15. Fluxo de ponta a ponta por arquivos, para teste manual (2026-10-13)
Do PDF à resposta sem banco, API ou tela. Da raiz, com `pip install -e backend` feito (ou
`PYTHONPATH=backend` no terminal):

1. `python -m ingestao.processar_bula <pdf> --medicamento ... --principio-ativo ... --fabricante ...
   --data-versao AAAA-MM-DD` -> `data/processed/bulas/<id>.json` (lista as seções para conferir).
2. `python -m ingestao.gerar_chunks` -> `data/processed/chunks.jsonl` + embeddings no cache.
3. `python -m benchmark.perguntar ["pergunta"]` -> resposta, trechos usados e tempos.

Decisões tomadas ao implementar:
- **O dataset em arquivo é a fonte da verdade** e o `Acervo` (`services/acervo.py`) carrega dele. O
  PostgreSQL fica para o histórico de consultas e para a API; `rag.responder` não sabe de onde o acervo veio.
- **Cache de embeddings** em SQLite (`data/cache/embeddings.sqlite`, fora do Git), chave = hash de
  modelo + texto. Grava a cada lote de 8: dá para interromper e continuar.
- O texto que vira embedding leva medicamento e seção na frente (`texto_para_embedding`): mudar
  esse cabeçalho invalida o cache.
- **Seções novas** em `Secao`: `composicao` ("tem lactose?"), `resultados_eficacia` e
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

**Limites de hoje:** testado só com a bula inventada de `backend/tests/fixtures/bula_exemplo.txt`;
a leitura de PDF (`pdfplumber`) ainda não rodou nesta máquina, porque depende do Python 3.12
(seção 12). As rodadas e a checagem de fundamentação entraram em 2026-10-15 (seção 16).

## 16. Agente em grafo com LangGraph (2026-10-15)
Pedido do usuário: o agente decidir melhor o que fazer quando a resposta é fraca, em vez de uma
tentativa só. A seção 14 virou um grafo de **7 nós** (`app/services/grafo.py`): `entender`,
`esclarecer`, `buscar`, `gerar`, `validar`, `verificar_llm`, `finalizar`. `rag.responder`
continua sendo a porta de entrada, com o mesmo `Resultado` (mais `rodadas` e `tentativas`).

Decisões:
- **Só 2 nós chamam o LLM** (`gerar` e `verificar_llm`). As decisões ficam nas arestas
  condicionais (`depois_de_*`), em código. Motivo: 30-45 s por chamada nesta CPU, e o qwen2.5:1.5b
  não é confiável para escolher caminho; o `detectar()` determinístico já faz isso melhor. Nada
  de roteador, planejador ou reescritor de pergunta por LLM.
- **Seção da pergunta sem LLM** (`services/intencao.py`): palavras-gatilho por seção. A rodada 1
  não usa (busca só por semelhança, como antes); da rodada 2 em diante a busca fica nas seções
  indicadas, com os trechos ainda não usados primeiro. Os já usados continuam elegíveis: o trecho
  certo pode ter vindo na rodada 1 e o erro ter sido do modelo.
- **Recusa do modelo também pode gerar nova rodada**, uma vez: se a pergunta indica seção e nenhum
  trecho enviado era dela, a recusa é falha de busca, não falta de informação.
- **Checagem de fundamentação** (`services/fundamentacao.py`): todo número da afirmação tem de
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
Primeira ingestão das bulas reais (lote 1 de `FONTES.md`: 20 PDFs). `python -m ingestao.processar_lote`
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
- **`Bula` ganha `numero_registro`, `forma_farmaceutica` e `versao`** (opcionais só por causa das
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
Cinco rotas sob `/api/v1` (`app/main.py`, `app/api/rotas.py`): `saude`, `medicamentos`, `perguntas`,
`historico` e `historico/{id}`. O agente é o mesmo do terminal (`rag.responder`); a API acrescenta a
leitura do acervo no banco, o histórico e o formato de erro.

Decisões:
- **O acervo da API vem do banco** (`Acervo.do_banco`): bulas e trechos são lidos uma vez, ao subir,
  e a busca é a `BuscaPgvector`. Os arquivos continuam sendo a fonte da verdade (seção 15); o banco é
  o que a API enxerga. Depois de `carregar_banco`, a API precisa ser reiniciada.
- **`chunks` ganhou `medicamento` e `n_tokens`**, para o banco espelhar o `chunks.jsonl` inteiro e a
  API montar os trechos sem ler arquivo. Banco carregado antes delas: a API recusa subir o acervo e
  pede para rodar `carregar_banco` de novo (não adivinha valor).
- **A bula do banco não tem as seções** (o texto está nos trechos), então o `Acervo` passou a aceitar
  qualquer objeto com os dados da bula (`BulaDoAcervo`, um `Protocol`): a `Bula` do arquivo e a do
  banco servem.
- **`preparar_banco` saiu de `ingestao/` para `app/db/conexao.py`**: a imagem Docker da API só leva o
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
- **Todo erro sai em `ErroOut`** (`app/api/erros.py`), inclusive a validação de entrada do FastAPI
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
- **Busca geral** (`_busca_geral` em `grafo.py`) quando a pergunta não tem medicamento indexado
  (nenhum nome, ou só remédio sem bula no acervo): todas as bulas, só nas seções que a pergunta indica
  e **um trecho por princípio ativo**, até `TOP_K_GERAL=4`. Cada trecho é uma opção na resposta; sem a
  regra, os três melhores trechos podiam ser do mesmo remédio.
- **A seção vale já na rodada 1 da busca geral.** Sem isso, "melhor remédio pra dor de cabeça" traz
  reações adversas que citam cefaleia antes das indicações. `intencao.py` ganhou gatilhos de escolha e
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
  do assunto da pergunta (`intencao.termos_do_assunto`, com sinônimos de sintoma: "tonto" →
  tontura, vertigem, labirintite; "dor de cabeça" → cabeça, cefaleia, enxaqueca). Se a pergunta cita
  remédio sem bula, o trecho tem de mencioná-lo (em qualquer seção, os da seção da pergunta
  primeiro). Nenhum trecho assim = "não encontrei", sem chamar o LLM.
- **"Cefaleia" não é cefalexina**: a tolerância a erro de digitação de `detectar` lia "cefaleia" como o
  antibiótico (0,88 de semelhança). Palavra que aparece no texto das bulas (`Catalogo.vocabulario`)
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
- A conversa vive só na tela (`conversa.ts`): a API continua sem estado, cada pergunta é independente.
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
- `ingestao/extrair_pdf.py` lê **só esse formato** como tabela (duas colunas, rótulo "Impacto clínico"
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
a consulta no histórico (`rag.Anterior`: pergunta, princípios ativos usados, se pediu esclarecimento) e
continua sem estado. `consultas` ganhou `principios_ativos` (os usados na busca). No `entender`:
- sem remédio na pergunta nova ("pode na gravidez?"), herda o da anterior; se a anterior foi geral,
  só herda a pergunta que começa com "e"/"mas" ("e para criança?"), e o assunto vem da anterior;
- "e com varfarina?" junta o remédio novo com o da anterior (a pergunta é omeprazol + varfarina);
- pedido de remédio ("qual o melhor remédio para tosse?") é assunto novo e não herda nada;
- a busca usa as duas perguntas juntas, e o prompt ganha "PERGUNTA ANTERIOR: ...".

**Alergia.** "Alergia a / alérgico à / hipersensibilidade ao" + remédio (`intencao.texto_apos_alergia`,
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
registro. `Catalogo.classes` guarda, por princípio ativo (e pelos nomes sem o sal), a classe mais
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
`services/extrato.py` monta a resposta com as frases da bula, e o `gerar` não chama o modelo, em dois
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

- **Regra "fala do assunto"** (`intencao.responde_ao_assunto`, no `validar`): a resposta do LLM precisa
  usar uma palavra do assunto da pergunta (tirados os nomes de remédio); ou citar trecho da seção que
  a pergunta indica (menos posologia, onde a frase errada estava); ou, em pergunta de dose, trazer
  número com unidade ("½ a 1 comprimido até 4 vezes ao dia" não tem a palavra "dose"). Reprovou = nova
  rodada com o aviso `DICA_ASSUNTO`. Medida antes de entrar: das 12 respostas certas do LLM gravadas
  no dia, nenhuma seria reprovada. Sinônimos novos: rins/renal/rim, fígado/hepático.
- **Dose copiada da bula** (`extrato.frases_de_dose`): pergunta de dose sobre remédio do acervo traz,
  de cada bula dele, a frase da posologia com número e unidade que mais fala do pedido ("máxima",
  "inicial", "insuficiência cardíaca"); criança só se a pergunta é de criança. "Paracetamol, qual a
  dose máxima?" traz uma frase por apresentação (gotas, comprimido), que era o "depende da
  apresentação" esperado desde a seção 14. Sem frase assim, segue pelo LLM.
- **"E para criança?" sem pergunta anterior** pergunta de volta "sobre qual remédio?" em vez de buscar
  "criança" em todas as bulas (respondeu com uma frase da bula da permetrina).
- **O erro também traz o id da consulta** (`ErroOut.consulta_id`): a consulta já era gravada; agora a tela
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

**Destaque** (`fundamentacao.destaques`, sem LLM): de cada fonte citada, as frases inteiras do trecho
que sustentam as afirmações que a citam, como posições (início, fim) no texto do trecho. Frase copiada
da bula (resposta extrativa): as frases onde ela está. Paráfrase do modelo: as frases que mais cobrem
os termos e números da afirmação, uma de cada vez, até nenhuma acrescentar nada (no máximo 3); a
primeira precisa cobrir dois itens, porque uma palavra solta em comum não basta. O nome do remédio
não conta (está no trecho todo).

**Não ajudou** (`rag.OutraExplicacao`, grafo):
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

Feito em 2026-10-17: destaque e grafo (`Resultado.outra_explicacao`, `Fonte.destaques`), 16 testes.
Falta: a API (pedido de outra explicação a partir da consulta, destaques/fabricante/registro na
fonte, gravar em `consulta_fontes`) e a tela (botão, trecho com a parte destacada).
