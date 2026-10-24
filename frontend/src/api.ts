// Cliente da API /api/v1. Os tipos espelham backend/app/schemas/api.py: mudou lá, muda aqui.

export type ErrorCode =
  | "MEDICAMENTO_NAO_ENCONTRADO"
  | "CONSULTA_NAO_ENCONTRADA"
  | "ENTRADA_INVALIDA"
  | "SAIDA_INVALIDA"
  | "CITACAO_INVALIDA"
  | "RESPOSTA_NAO_VERIFICADA"
  | "OLLAMA_INDISPONIVEL"
  | "BANCO_INDISPONIVEL"
  | "TEMPO_ESGOTADO";

export interface Source {
  posicao: number; // o [n] da resposta
  chunk_id: string;
  medicamento: string;
  fabricante: string | null;
  numero_registro: string | null;
  secao: string;
  rotulo_secao: string;
  pagina: number;
  data_versao: string; // AAAA-MM-DD
  trecho: string;
  score: number;
  citado: boolean;
  // Frases do trecho que sustentam a resposta, como [início, fim] em `trecho`. As posições contam
  // caracteres como o Python (code points): fatiar com Array.from, não com slice direto.
  destaques: [number, number][];
}

export interface Option {
  rotulo: string;
  principios_ativos: string[];
}

export interface Clarification {
  // opcoes vazio (remedio_da_alergia): a pessoa responde no chat.
  motivo: "medicamento_nao_reconhecido" | "remedio_da_alergia";
  pergunta: string;
  opcoes: Option[];
}

export interface Answer {
  consulta_id: string;
  status: "respondida" | "nao_encontrada" | "precisa_esclarecimento";
  esclarecimento: Clarification | null;
  rodadas: number;
  // sem_outra_explicacao: o "não ajudou" não achou outro trecho além dos já citados.
  motivo: "sem_contexto" | "medicamento_nao_indexado" | "sem_outra_explicacao" | null;
  // Pergunta de para que serve: o veredito montado pelo sistema a partir das indicações da bula.
  conclusao: string | null;
  resposta: string | null;
  fontes: Source[];
  medicamentos_nao_indexados: string[];
  classes_terapeuticas: Record<string, string>; // remédio sem bula -> classe no registro da ANVISA
  principios_ativos: string[]; // os usados na busca
  continuacao: boolean; // usou a pergunta anterior da conversa
  alergia_a: string | null; // a resposta traz alternativas a este remédio
  extrativa: boolean; // frases copiadas da bula, sem o modelo de linguagem
  outra_explicacao: boolean; // resposta ao "não ajudou", sem os trechos já citados
  // Aviso de saúde, montado pela API (nunca pelo modelo): só na resposta "respondida".
  aviso: string | null;
  modelo: string;
  latencia_ms: number;
}

export interface LeafletSummary {
  id: string;
  medicamento: string;
  fabricante: string;
  forma_farmaceutica: string | null;
  data_versao: string;
}

export interface Medicine {
  principio_ativo: string;
  bulas: LeafletSummary[];
}

export type InquiryStatus = "respondida" | "nao_encontrada" | "precisa_esclarecimento" | "erro";

export interface InquirySummary {
  id: string;
  pergunta: string;
  status: InquiryStatus;
  outra_explicacao: boolean; // a mesma pergunta da consulta_origem_id, explicada de outro jeito
  criado_em: string;
}

export interface History {
  itens: InquirySummary[];
  total: number;
  pagina: number;
  limite: number;
}

export interface InquiryDetail extends InquirySummary {
  principios_ativos_filtro: string[];
  conclusao: string | null;
  resposta: string | null;
  motivo: string | null;
  codigo_erro: ErrorCode | null;
  mensagem_erro: string | null;
  rodadas: number;
  modelo: string;
  latencia_ms: number;
  consulta_origem_id: string | null;
  fontes: Source[];
  aviso: string | null; // o mesmo aviso da resposta, calculado ao abrir a consulta
}

export interface Health {
  status: "ok" | "degradado";
  banco: boolean;
  ollama: boolean;
  modelos_faltando: string[];
  bulas: number;
  trechos: number;
  problemas: string[];
}

/** Etapa de verdade do agente, avisada pela API quando começa (F3d). */
export type Stage = "buscando" | "escrevendo" | "conferindo" | "verificando";

export interface Progress {
  etapa: Stage;
  rodada: number; // começa em 1; cada rodada a mais é uma nova tentativa depois de uma conferência que falhou
  de: number; // máximo de rodadas (RODADAS_MAX)
}

/** Uma linha do fluxo NDJSON de POST /perguntas/com-progresso (backend/app/api/progress.py). */
type StreamEvent =
  | ({ tipo: "etapa" } & Progress)
  | { tipo: "vivo" } // batimento: nada de novo, a conexão segue aberta
  | { tipo: "resposta"; resposta: Answer }
  | { tipo: "erro"; erro: { codigo: ErrorCode; mensagem: string }; consulta_id: string | null; status: number };

export interface QuestionIn {
  pergunta: string;
  principios_ativos: string[];
  consulta_origem_id?: string;
}

/** Erro no formato ErrorOut da API, ou a API fora do ar (codigo "API_FORA_DO_AR", status 0). */
export class ApiError extends Error {
  readonly codigo: ErrorCode | "API_FORA_DO_AR";
  readonly status: number;
  readonly consultaId: string | null; // a consulta gravada, quando a pergunta chegou à API

  constructor(codigo: ErrorCode | "API_FORA_DO_AR", mensagem: string, status: number, consultaId: string | null = null) {
    super(mensagem);
    this.codigo = codigo;
    this.status = status;
    this.consultaId = consultaId;
  }
}

const FORA_DO_AR =
  "A API não respondeu. Ela está rodando? Na raiz: uvicorn app.main:app --app-dir backend";

const INTERROMPIDA = "A resposta foi interrompida antes do fim. Veja o log da API.";

/** O erro de uma resposta HTTP que não é 2xx. */
function erroHttp(status: number, corpo: { erro?: { codigo: ErrorCode; mensagem: string }; consulta_id?: string } | null) {
  // Sem corpo de ErrorOut: quem respondeu foi o proxy do Vite, com a API desligada.
  if (!corpo?.erro) return new ApiError("API_FORA_DO_AR", FORA_DO_AR, status);
  return new ApiError(corpo.erro.codigo, corpo.erro.mensagem, status, corpo.consulta_id ?? null);
}

async function pedir<T>(caminho: string, init?: RequestInit): Promise<T> {
  let r: Response;
  try {
    r = await fetch(`/api/v1${caminho}`, init);
  } catch {
    throw new ApiError("API_FORA_DO_AR", FORA_DO_AR, 0);
  }
  const corpo = await r.json().catch(() => null);
  if (!r.ok) throw erroHttp(r.status, corpo);
  return corpo as T;
}

/**
 * Pergunta em fluxo (F3d): chama `aoEtapa` a cada etapa que o agente começa e devolve a resposta.
 * Erro HTTP antes do fluxo (entrada inválida, banco fora) e a linha "erro" do fluxo viram ApiError,
 * como no `pedir`. Fluxo que acaba sem a resposta (conexão cortada, bug na API) também.
 */
async function transmitir(caminho: string, init: RequestInit, aoEtapa: (p: Progress) => void): Promise<Answer> {
  let r: Response;
  try {
    r = await fetch(`/api/v1${caminho}`, init);
  } catch {
    throw new ApiError("API_FORA_DO_AR", FORA_DO_AR, 0);
  }
  if (!r.ok) throw erroHttp(r.status, await r.json().catch(() => null));
  if (!r.body) throw new ApiError("API_FORA_DO_AR", INTERROMPIDA, 0);

  const final: { resposta: Answer | null } = { resposta: null };
  const tratar = (linha: string) => {
    if (!linha.trim()) return;
    const ev = JSON.parse(linha) as StreamEvent;
    if (ev.tipo === "etapa") aoEtapa({ etapa: ev.etapa, rodada: ev.rodada, de: ev.de });
    else if (ev.tipo === "resposta") final.resposta = ev.resposta;
    else if (ev.tipo === "erro") throw new ApiError(ev.erro.codigo, ev.erro.mensagem, ev.status, ev.consulta_id);
    // "vivo" e tipos que este cliente não conhece: ignorados.
  };

  const leitor = r.body.getReader();
  const decodificador = new TextDecoder();
  let pendente = ""; // o pedaço de linha que ainda não terminou
  try {
    for (;;) {
      const { done, value } = await leitor.read();
      if (done) break;
      pendente += decodificador.decode(value, { stream: true });
      const linhas = pendente.split("\n");
      pendente = linhas.pop() ?? "";
      linhas.forEach(tratar);
    }
    tratar(pendente + decodificador.decode());
  } catch (e) {
    if (e instanceof ApiError) throw e;
    throw new ApiError("API_FORA_DO_AR", INTERROMPIDA, 0); // conexão cortada no meio, ou linha ilegível
  }
  if (final.resposta === null) throw new ApiError("API_FORA_DO_AR", INTERROMPIDA, 0);
  return final.resposta;
}

export const api = {
  saude: () => pedir<Health>("/saude"),
  medicamentos: () => pedir<Medicine[]>("/medicamentos"),
  // As duas perguntas usam o fluxo com progresso (F3d); a API mantém as rotas sem fluxo, com o mesmo resultado.
  perguntar: (entrada: QuestionIn, aoEtapa: (p: Progress) => void) =>
    transmitir(
      "/perguntas/com-progresso",
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(entrada) },
      aoEtapa,
    ),
  // "Não ajudou": a mesma pergunta da consulta, sem os trechos que ela (e as explicações antes dela) citou.
  outraExplicacao: (consultaId: string, aoEtapa: (p: Progress) => void) =>
    transmitir(`/perguntas/${consultaId}/outra-explicacao/com-progresso`, { method: "POST" }, aoEtapa),
  historico: (pagina: number, limite: number) =>
    pedir<History>(`/historico?pagina=${pagina}&limite=${limite}`),
  consulta: (id: string) => pedir<InquiryDetail>(`/historico/${id}`),
};
