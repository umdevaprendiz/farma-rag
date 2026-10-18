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
  motivo: "sem_contexto" | "medicamento_nao_indexado" | null;
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

async function pedir<T>(caminho: string, init?: RequestInit): Promise<T> {
  let r: Response;
  try {
    r = await fetch(`/api/v1${caminho}`, init);
  } catch {
    throw new ApiError("API_FORA_DO_AR", FORA_DO_AR, 0);
  }
  const corpo = await r.json().catch(() => null);
  if (!r.ok) {
    const erro = corpo?.erro;
    // Sem corpo de ErrorOut: quem respondeu foi o proxy do Vite, com a API desligada.
    if (!erro) throw new ApiError("API_FORA_DO_AR", FORA_DO_AR, r.status);
    throw new ApiError(erro.codigo, erro.mensagem, r.status, corpo?.consulta_id ?? null);
  }
  return corpo as T;
}

export const api = {
  saude: () => pedir<Health>("/saude"),
  medicamentos: () => pedir<Medicine[]>("/medicamentos"),
  perguntar: (entrada: QuestionIn) =>
    pedir<Answer>("/perguntas", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(entrada),
    }),
  historico: (pagina: number, limite: number) =>
    pedir<History>(`/historico?pagina=${pagina}&limite=${limite}`),
  consulta: (id: string) => pedir<InquiryDetail>(`/historico/${id}`),
};
