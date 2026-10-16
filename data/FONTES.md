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

Sempre a versão do profissional de saúde, em `data/raw/bulas/`. Duas origens aceitas:
- **Bulário Eletrônico** (https://consultas.anvisa.gov.br/#/bulario/): só download **manual**. O
  portal usa Cloudflare Turnstile (anti-robô): não automatizar nem contornar.
- **Site oficial do detentor do registro**, quando serve o PDF sem bloqueio. Sites que respondem 403
  a cliente automático (Bayer, Kenvue/Tylenol, buscopan.com.br, Eurofarma, Neo Química) ficam para o
  download manual. Não usar espelhos de terceiros (drogarias, agregadores de bula).

Registrar aqui, por bula: nome, número de registro, data da versão e SHA-256.

Depois de registrar aqui, acrescente o PDF em `data/bulas_manifesto.json` (mesmo SHA-256 e
registro; um teste confere) e rode `python -m ingestao.processar_lote`.

### Lote 1 (baixado em 2026-10-14)

Verificação feita **antes** de baixar:
- dono do domínio no registro.br (RDAP) = detentor do registro no CSV da ANVISA (mesmo CNPJ):
  geolab.com.br (03.485.572/0001-04, desde 2001), vitamedic.ind.br (30.222.814/0001-31, desde 2015);
  organon.com (registrado em 1997, analisado em 2026-10-13);
- link tirado da página de bulas do próprio fabricante, sem redirecionamento para outro host;
  cadeia TLS válida; Google Safe Browsing sem alerta para os domínios;
- resposta `application/pdf` começando com `%PDF`.

Verificação feita **depois** de baixar, numa pasta de quarentena fora do projeto, sem abrir os PDFs:
- Windows Defender (assinaturas 1.461.28.0, de 2026-10-14): nenhuma ameaça;
- `python -I ingestao/validar_pdf_bula.py`: nenhum conteúdo ativo (JavaScript, ações, anexos);
- texto lido com `pdfplumber`: confirma que é bula do profissional e que o número de registro
  impresso existe, está `Ativo` no CSV e é do princípio ativo esperado.

Recusados na conferência: o "Fluimucil" de fluimucil.com.br é a bula do **paciente**; o link
"maleato de dexclorfeniramina / Bula para Profissional" da Geolab aponta para o arquivo da bula de
**losartana + hidroclorotiazida** (registro 1.5423.0299). Este foi trocado pelo Hystin, da mesma Geolab.

"Versão" = código de versão do fabricante ou data de aprovação impressa; sem nenhum dos dois, a data
mais recente da tabela de histórico de alteração. Conferir ao rodar `processar_bula`.

| Arquivo | Produto (princípio ativo) | Registro | Origem | Versão | SHA-256 |
|---|---|---|---|---|---|
| `losartana_cozaar_profissional.pdf` | Cozaar (losartana potássica) | 1.0029.0007 | [Organon](https://www.organon.com/brazil/wp-content/uploads/sites/33/2023/06/cozaar_bula_profissional.pdf) | aprovada em 24/04/2025 | `1a3806e4b84063e665f4d8f8515dac84a40537a1c2b0734a28ce0bd2b20a6e84` |
| `desloratadina_desalex_profissional.pdf` | Desalex (desloratadina) | 1.0029.0191 | [Organon](https://www.organon.com/brazil/wp-content/uploads/sites/33/2023/06/desalex_bula_profissional.pdf) | histórico: 03/11/2025 | `c9e0ee072f8739016c43df23b704c74ae2fbbe71c80aa93f5978f57657b9285b` |
| `enalapril_renitec_profissional.pdf` | Renitec (maleato de enalapril) | 1.0029.0031 | [Organon](https://www.organon.com/brazil/wp-content/uploads/sites/33/2023/06/renitec_bula_profissional.pdf) | aprovada em 01/05/2025 | `5956e109861e27825c8279fcca2b39bba3aa4e589078c98b18e51d8adb875c0b` |
| `acetilcisteina_cisteil-geolab_profissional.pdf` | Cisteil xarope (acetilcisteína) | 1.5423.0140 | [Geolab](https://www.geolab.com.br/wp-content/uploads/2021/04/bula_1790109470632.pdf) | V.14_02/2026 | `c4211f5cd1c424366cc527db8f4063c980a75eaf4867d31ba1bcc5aa771e8aa9` |
| `albendazol_albel-geolab_profissional.pdf` | Albel (albendazol) | 1.5423.0323 | [Geolab](https://www.geolab.com.br/wp-content/uploads/2021/09/bula_1769000842190.pdf) | V.03_03/2025 | `f2dd8c4a7a2e5686ee4596d4905ef088ab0dd0e2908c2e04a1891bfc54b4ec71` |
| `ambroxol_generico-geolab_profissional.pdf` | cloridrato de ambroxol (genérico) | 1.5423.0067 | [Geolab](https://www.geolab.com.br/wp-content/uploads/2021/05/bula_1769017137320.pdf) | V.03_06/2018 | `3dceaac995836f0045de5b9dc685f84095f845ef0bc1f5eddde52540c3d148ea` |
| `metformina_generico-geolab_profissional.pdf` | cloridrato de metformina (genérico) | 1.5423.0270 | [Geolab](https://www.geolab.com.br/wp-content/uploads/2023/09/bula_1769084891745.pdf) | V.04 09/2025 | `80d8966c9e2165b190b6b2c73693a18018a2ad0cf84651f50da8efe5a52fbd39` |
| `colecalciferol_delumi-geolab_profissional.pdf` | Delumi (colecalciferol) | 1.5423.0302 | [Geolab](https://www.geolab.com.br/wp-content/uploads/2021/05/bula_1768324580160.pdf) | V.01_12/2025 | `861fe8aa5c03c7d9152929afbd2314332115976e5a723fece7c09226190486c1` |
| `diclofenaco-sodico_generico-geolab_profissional.pdf` | diclofenaco sódico (genérico) | 1.5423.0317 | [Geolab](https://www.geolab.com.br/wp-content/uploads/2021/09/bula_1790111050501.pdf) | V.04_02/2026 | `2cd9277351ea6d5158f7996e69c44c6585b649956b338a4018b43bd46d97f2c5` |
| `dexclorfeniramina_hystin-geolab_profissional.pdf` | Hystin (maleato de dexclorfeniramina) | 1.5423.0012 | [Geolab](https://www.geolab.com.br/wp-content/uploads/2021/05/HYSTIN-BULA-PROFISSIONAL.pdf) | V.02_01/2023 | `19fe6878d0d77aa8cd7435fa6f3d3ec67bc1bd12fe0c392fb46cfb3915f8b1ac` |
| `ibuprofeno_generico-geolab_profissional.pdf` | ibuprofeno (genérico) | 1.5423.0333 | [Geolab](https://www.geolab.com.br/wp-content/uploads/2021/05/bula_1784663951293.pdf) | V.05 10/2025 | `ffffa9683ecc7fe5edc7fdc6fb3eecb143cace76df7e097175739a9d3be6d1cc` |
| `loratadina_generico-geolab_profissional.pdf` | loratadina (genérico) | 1.5423.0336 | [Geolab](https://www.geolab.com.br/wp-content/uploads/2021/05/bula_1769174169341.pdf) | histórico: 27/12/2024 | `2d613f1b0b5f1b482ce0f52cbbf7d6920e4b908cf4f1f0efb2a220a2927ac5ac` |
| `acido-folico_folonin-geolab_profissional.pdf` | Folonin (ácido fólico) | 1.5423.0142 | [Geolab](https://www.geolab.com.br/wp-content/uploads/2021/05/bula_1768571065974.pdf) | V.06_09/2025 | `6a8badcfe84d5f67efba2572f9724a5b1a29582d8af031920ed086eaeaaeb803` |
| `nimesulida_nisoflan-geolab_profissional.pdf` | Nisoflan (nimesulida) | 1.5423.0013 | [Geolab](https://www.geolab.com.br/wp-content/uploads/2021/05/bula_1768574432875.pdf) | V.07_10/2025 | `12dc469d81b17bd8e21ca265d760f705e225f501cb9d22935e7b70a90427b665` |
| `omeprazol_omenax-geolab_profissional.pdf` | Omenax (omeprazol) | 1.5423.0328 | [Geolab](https://www.geolab.com.br/wp-content/uploads/2021/05/bula_1768575004911.pdf) | V.03_12/2025 | `5e89e82527ef00135bb97858f5ed3d3eaf51e1e6d76b87f693951c98eea1e6e7` |
| `dexpantenol_pantodex-geolab_profissional.pdf` | Pantodex (dexpantenol) | 1.5423.0148 | [Geolab](https://www.geolab.com.br/wp-content/uploads/2021/05/bula_1767819342805.pdf) | V.04_12/2025 | `38375b673a38c2d1da4c19eafa7c48277f15285e93d73585ff3c1ae7ea571b34` |
| `paracetamol_tylalgin-geolab_profissional.pdf` | Tylalgin (paracetamol) | 1.5423.0008 | [Geolab](https://www.geolab.com.br/wp-content/uploads/2021/05/bula_1767894323973.pdf) | V.05_05/2022 | `549ad68fce6d24efe081280d224c0be80617ecf71f3236a97a1c431b6bbbb843` |
| `permetrina_kaodine-geolab_profissional.pdf` | Kaodine (permetrina) | 1.5423.0342 | [Geolab](https://www.geolab.com.br/wp-content/uploads/2026/01/bula_1767798815437.pdf) | V.01_12/2025 | `e37ebcb43746fb2a699cddcbbfe4202ec0cf0a2c23484dd842b53b05a1d6ee0e` |
| `tiamina_vitaum-geolab_profissional.pdf` | Vitaum (cloridrato de tiamina) | 1.5423.0129 | [Geolab](https://www.geolab.com.br/wp-content/uploads/2021/05/bula_1768592730965.pdf) | V03_05/2025 | `755146b41d86872e933ea740db59405cdceecb8b8307fe145d86a9a08bb22dc6` |
| `fexofenadina_generico-vitamedic_profissional.pdf` | cloridrato de fexofenadina (genérico) | 1.0392.0230 | [Vitamedic](https://www.vitamedic.ind.br/wp-content/uploads/2023/08/cloridrato-de-fexofenadina-bula-profissional.pdf) | histórico: 13/03/2026 | `721c94b2ad1f99535e511918a33be933792319805d913629a1967af08d8c91a5` |

### Lote 2 (baixado em 2026-10-16)

Mesma verificação do lote 1, refeita para este lote:
- antes de baixar: geolab.com.br no registro.br em nome de 03.485.572/0001-04 (Geolab, desde 2001),
  o mesmo CNPJ do detentor no CSV; link da página do produto e da página de bulas da Geolab, sem
  redirecionamento; TLS válido; Google Safe Browsing sem alerta (status 1); `application/pdf`;
- download com limite de 20 MB numa pasta de quarentena vazia, fora do projeto; começa com `%PDF-1.7`;
- depois: Windows Defender (assinaturas 1.461.52.0, de 2026-10-15) sem ameaças;
  `validar_pdf_bula.py` sem conteúdo ativo (107 objetos, todos legíveis); texto de bula do
  profissional ("Resultados de eficácia"), uma bula só, registro impresso 1.5423.0350 `Ativo` no CSV
  (DORAGEO, dipirona monoidratada, similar).

Pesquisados e recusados neste lote: o link "Bula para Profissional" da dipirona genérica da Geolab
aponta para `Bula-do-paciente-dipirona-50mg_mL.pdf` (bula do paciente); escopolamina e nafazolina da
Geolab só existem em associação (Espaslit DUO, Colírio Geolab, Ocutil); Diprin e Vitacin Kids não
têm bula no site. A Cimed (domínio cimedremedios.com.br, mesmo CNPJ do detentor) só publica a bula
do paciente. O site de bulas da Prati-Donaduzzi é um aplicativo sem link direto para o PDF.

| Arquivo | Produto (princípio ativo) | Registro | Origem | Versão | SHA-256 |
|---|---|---|---|---|---|
| `dipirona_dorageo-geolab_profissional.pdf` | Dorageo 1 g comprimido (dipirona monoidratada) | 1.5423.0350 | [Geolab](https://www.geolab.com.br/wp-content/uploads/2024/08/bula_1787252112501.pdf) | V.03_03/2026 | `9962db94406f0835a09a4c0d539c64cbfe83ae446852f02e323229cd20940793` |

## Catálogo derivado (versionado no Git)

`data/processed/catalogo_anvisa.jsonl` + `catalogo_anvisa.meta.json`, gerados por
`python -I ingestao/catalogo_anvisa.py` a partir do CSV acima. Saída determinística (sem data/hora):
a mesma entrada gera os mesmos bytes, então o Git só mostra diferença quando a ANVISA mudar os dados.
`meta.json` guarda o SHA-256 da fonte e da saída. Uma linha por par (nome do produto, princípios
ativos) de registros ativos; `associacao` marca produtos com mais de um princípio ativo, e
`classe_terapeutica` só é preenchida nos de um princípio ativo.
