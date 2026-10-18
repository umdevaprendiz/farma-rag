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

Validação completa (`python -I ingestion/validate_anvisa_csv.py`): sem bytes nulos ou de controle,
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
registro; um teste confere) e rode `python -m ingestion.process_batch`.

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
- `python -I ingestion/validate_leaflet_pdf.py`: nenhum conteúdo ativo (JavaScript, ações, anexos);
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
  `validate_leaflet_pdf.py` sem conteúdo ativo (107 objetos, todos legíveis); texto de bula do
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

### Lote 3 (baixado em 2026-10-18)

Pedido do usuário: as que faltavam do MVP + zolpidem, sertralina e isotretinoína (perguntas reais
que voltaram "não encontrei"). Ele pediu a análise de procedência **antes** de baixar e aprovou só
os sites que passaram em tudo.

Verificação feita **antes** de baixar (RDAP do registro.br, certificado TLS validado, Google Safe
Browsing status 1 em todos, link tirado da página oficial do produto, registro `Ativo` no CSV):
- geolab.com.br: Geolab 03.485.572/0001-04, desde 2001 (mesmo CNPJ do detentor);
- **sara.com.br**, o portal de bulas do grupo EMS: registrado em nome de **EMS S/A 57.507.378/0003-65**
  (o CNPJ do detentor dos registros 1.0235.xxxx no CSV), desde 2018. A página do produto traz o
  caminho da bula do profissional (`professionalLeaflet`), servida por `cdn.sara.com.br` (Amazon
  CloudFront, certificado válido para o nome). O sara.com.br também hospeda bulas de outras
  empresas: só valem as de produto da própria EMS S/A. O site da Germed (domínio no CNPJ da
  Germed) aponta para ele; ems.com.br fica atrás de antirrobô (Incapsula) e não foi usado;
- libbs.com.br: Libbs 61.230.314/0001-75, desde 1997 (mesmo CNPJ). O PDF está no domínio da Libbs,
  mas a página do produto só mostra o link da bula do paciente (o do profissional fica atrás de
  "Sou Profissional de Saúde").

Verificação feita **depois** de baixar, numa pasta de quarentena vazia fora do projeto, sem abrir
os PDFs: limite de 20 MB; resposta `application/pdf` ou `binary/octet-stream` começando com `%PDF-`
(a 2ª versão da nitazoxanida veio em base64 e foi recusada); Windows Defender (assinaturas
1.461.84.0, de 2026-10-17) sem ameaças; `validate_leaflet_pdf.py` sem conteúdo ativo nos 15 PDFs;
texto de bula do profissional, registro impresso `Ativo` no CSV e do detentor certo. Para cada
produto do sara.com.br com duas versões, ficou a mais recente (as duas do clotrimazol eram o mesmo
arquivo).

Dois PDFs da EMS trazem duas bulas (sem o cabeçalho "MODELO DE BULA..."): AAS 500 mg + 100 mg
(indicações diferentes: a de 100 mg também como antiagregante) e isotretinoína 10 mg + 20 mg. O
separador passou a abrir a bula seguinte em "I - IDENTIFICAÇÃO DO MEDICAMENTO" depois do histórico;
as 25 bulas anteriores reprocessadas ficaram idênticas.

Ficaram de fora nesta análise (registrado para não repetir a busca):
- **com ressalva, aguardando o usuário**: Abbott (Duphalac; `abbottbrasil.com.br` está em nome de
  pessoa física desde 2000, mas o servidor usa certificado OV de "Abbott Laboratories", que cobre
  `dam.abbott.com`; o link "DUPHALAC (bula do profissional)" está na página de bulas); Germed via
  sara.com.br (naproxeno sódico 1.0583.0978, minoxidil 1.0583.0949: CNPJ diferente do dono do
  domínio, mesmo grupo); Sandoz (Diasec 1.0047.0318: a Sandoz lista como descontinuado;
  `sandoz.com.br` ainda em nome da Novartis Biociências);
- **reprovado**: natulab.com.br (Expectoflui, Masferol), registrado em nome de "A C PEIXOTO ME",
  CNPJ 03.047.773/0001-11 **baixado**, não da Natulab (02.456.955/0001-83); certificado só DV;
- **manual**: Eurofarma (Angino-Rub, benzidamina) responde 403 a cliente automático; Aché, Cimed e
  União Química (dimenidrinato) só publicam a bula do paciente; sem fonte oficial aberta:
  cetirizina, bisacodil, escopolamina, dimenidrinato, ácido ascórbico (Bulário, à mão).

| Arquivo | Produto (princípio ativo) | Registro | Origem | Versão | SHA-256 |
|---|---|---|---|---|---|
| `sertralina_generico-geolab_profissional.pdf` | cloridrato de sertralina 50 e 100 mg (genérico) | 1.5423.0225 | [Geolab](https://www.geolab.com.br/wp-content/uploads/2021/05/bula_1787254130249.pdf) | V.15_03/2026 | `587c9aecbf2aa9903a16a4e08a7b5bc068e1bb11e32e2c0b4ebb16a75481656d` |
| `zolpidem_generico-geolab_profissional.pdf` | hemitartarato de zolpidem 10 mg (genérico) | 1.5423.0303 | [Geolab](https://www.geolab.com.br/wp-content/uploads/2021/06/bula_1769171475573-1.pdf) | V.02_12/2024 | `e95ce1fd4f44863c429210a25789e64f66d1e6a83c4be3a39b065a4b3c5b8b56` |
| `isotretinoina_generico-ems_profissional.pdf` | isotretinoína 10 e 20 mg (genérico, 2 bulas) | 1.0235.1120 | [EMS / sara.com.br](https://cdn.sara.com.br/leafletDocuments/prof-102351120.pdf) | Bula Padrão de 01/08/2025 e 24/10/2022 | `0f77756c16cde34662ac3544fdd23ebbc91f3e6920f6b874f61cbb2bfb248797` |
| `acido-acetilsalicilico_generico-ems_profissional.pdf` | ácido acetilsalicílico 500 e 100 mg (genérico, 2 bulas) | 1.0235.0508 | [EMS / sara.com.br](https://cdn.sara.com.br/leafletDocuments/prof-102350508.pdf) | Bula Padrão de 12/12/2024; aprovada em 13/06/2025 | `772f5440dc8de432c3c30d74dc5b6109a9f7780e1378dde321ddcf7476f7e646` |
| `amoxicilina_generico-ems_profissional.pdf` | amoxicilina tri-hidratada 500 mg (genérico) | 1.0235.0455 | [EMS / sara.com.br](https://cdn.sara.com.br/leafletDocuments/21965855-29b8-4384-8d74-4c4dcf236b5b.pdf) | histórico: 28/07/2025 | `2f2e510f74f6f4abc2f740098583d8a407fa35995cdcb06b68b9c1bbedd5b8d2` |
| `clotrimazol_generico-ems_profissional.pdf` | clotrimazol 10 mg/g creme vaginal (genérico) | 1.0235.0712 | [EMS / sara.com.br](https://cdn.sara.com.br/leafletDocuments/0b4c0be2-43ff-4f70-8e20-b5e663e5a820.pdf) | Bula Padrão de 23/10/2024 | `df91c210c86f89964ebc90e070b21382f91a66a3901393e6f6e3fa6e910b3d9d` |
| `nitazoxanida_generico-ems_profissional.pdf` | nitazoxanida 500 mg e 20 mg/mL (genérico) | 1.0235.1352 | [EMS / sara.com.br](https://cdn.sara.com.br/leafletDocuments/prof_102351352.pdf) | Bula Padrão de 22/05/2024 | `17b93f647d2355905388b6a5a2de3264de7b30f16654f5ac607dcfa6445de4bd` |
| `oximetazolina_generico-ems_profissional.pdf` | cloridrato de oximetazolina 0,25 e 0,5 mg/mL (genérico) | 1.0235.0782 | [EMS / sara.com.br](https://cdn.sara.com.br/leafletDocuments/e1209022-3b93-43d9-bdd3-4713faf06ba6.pdf) | Bula Padrão de 23/04/2021 | `dc1c4e6d49119eb31636486371e699b2ae922b5ab613f3c928933c05e49ca94f` |
| `nafazolina_generico-ems_profissional.pdf` | cloridrato de nafazolina 0,5 mg/mL (genérico) | 1.0235.0718 | [EMS / sara.com.br](https://cdn.sara.com.br/leafletDocuments/0998348236_professional.pdf) | Bula Padrão de 12/07/2023 | `5aa31beb386ce63c74d72397289c42438e149cdca422a540a2fd659e173aaf26` |
| `levonorgestrel_pozato-uni-libbs_profissional.pdf` | Pozato Uni (levonorgestrel 1,5 mg) | 1.0033.0125 | [Libbs](https://www.libbs.com.br/wp-content/uploads/2015/12/Pozato-Uni_bula-do-profissional_v.09-24.pdf) | aprovada em 05/12/2024 | `75426f9c8addb3b38eda82c210cafd55f29acea50817027be2666a253ea1caaa` |

## Catálogo derivado (versionado no Git)

`data/processed/catalogo_anvisa.jsonl` + `catalogo_anvisa.meta.json`, gerados por
`python -I ingestion/anvisa_catalog.py` a partir do CSV acima. Saída determinística (sem data/hora):
a mesma entrada gera os mesmos bytes, então o Git só mostra diferença quando a ANVISA mudar os dados.
`meta.json` guarda o SHA-256 da fonte e da saída. Uma linha por par (nome do produto, princípios
ativos) de registros ativos; `associacao` marca produtos com mais de um princípio ativo, e
`classe_terapeutica` só é preenchida nos de um princípio ativo.
