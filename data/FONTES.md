# Fontes dos dados

## ANVISA: medicamentos registrados (dados abertos)

| | |
|---|---|
| Fonte | https://dados.anvisa.gov.br/dados/DADOS_ABERTOS_MEDICAMENTOS.csv |
| Dicionário | https://dados.anvisa.gov.br/dados/Documentacao_e_Dicionario_de_Dados_Registros_Validos_Medicamento_V1.pdf (2020, desatualizado: não cita `SITUACAO_REGISTRO`) |
| Baixado em | 2026-10-11 (arquivo com `Last-Modified` de 2026-10-07) |
| Tamanho | 45.901.211 bytes |
| SHA-256 | `b71283bece2e11db95ffc48069fee0b4020c117ead18007fbc95c4f2082965ea` |
| Local | `data/raw/anvisa/` (fora do Git; baixe de novo e confira o SHA-256 ou atualize esta tabela) |
| Formato | CSV, delimitador `;`, codificação cp1252, 11 colunas, 228.773 linhas de dados |
| Conteúdo | 85.091 `Ativo` e 143.682 `Inativo`. **Não tem bula, indicação nem link para bula.** |
| Licença | Não verificada. O dicionário não a informa. |
| Integridade | A ANVISA não publica checksum; a confiança vem do TLS do domínio `.gov.br`. |

Validação completa (`python -I ingestao/validar_csv_anvisa.py`): sem bytes nulos ou de controle,
sem assinatura de executável/zip/pdf/html, 11 campos em todas as linhas, sem tags, URLs, comandos
ou células de fórmula. Há 10 nomes com prefixo `"- "` (cadastros antigos), inofensivos.

Regras de uso: só ler com o módulo `csv` (nunca abrir no Excel); filtrar `SITUACAO_REGISTRO = Ativo`.

## Bulas (PDF)

Baixadas **manualmente** do Bulário Eletrônico (https://consultas.anvisa.gov.br/#/bulario/), versão
do profissional de saúde, para `data/raw/`. O portal usa Cloudflare Turnstile (anti-robô): não
automatizar. Registrar aqui, por bula: nome, número de registro, data da versão e SHA-256.

## Catálogo derivado (versionado no Git)

`data/processed/catalogo_anvisa.jsonl` + `catalogo_anvisa.meta.json`, gerados por
`python -I ingestao/catalogo_anvisa.py` a partir do CSV acima. Saída determinística (sem data/hora):
a mesma entrada gera os mesmos bytes, então o Git só mostra diferença quando a ANVISA mudar os dados.
`meta.json` guarda o SHA-256 da fonte e da saída. Uma linha por par (nome do produto, princípios
ativos) de registros ativos; `associacao` marca produtos com mais de um princípio ativo, e
`classe_terapeutica` só é preenchida nos de um princípio ativo.
