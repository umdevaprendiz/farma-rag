// O modelo da conversa na tela. A API não guarda conversa (cada pergunta é independente); a lista
// de mensagens vive só aqui, e o histórico da barra lateral vem da API.

import type { Answer, ApiError, Clarification, ErrorCode, InquiryDetail, QuestionIn, Source } from "./api";

/** O que a tela mostra de uma resposta, venha ela da pergunta de agora, do histórico ou de um erro. */
export interface View {
  consultaId: string | null; // nulo quando a pergunta nem chegou à API
  status: "respondida" | "nao_encontrada" | "precisa_esclarecimento" | "erro";
  conclusao: string | null; // pergunta de para que serve: o veredito pelas indicações da bula
  texto: string | null;
  motivo: string | null;
  fontes: Source[];
  semBula: string[]; // remédios citados na pergunta sem bula no acervo
  classes: Record<string, string>; // e para que servem, pelo registro na ANVISA
  principios: string[];
  continuacao: boolean;
  alergiaA: string | null;
  extrativa: boolean;
  outraExplicacao: boolean; // resposta ao "não ajudou"
  esclarecimento: Clarification | null;
  erro: { codigo: ErrorCode | "API_FORA_DO_AR"; mensagem: string } | null;
  rodadas: number;
  latenciaMs: number;
}

export type Message =
  | { id: string; autor: "usuario"; texto: string }
  | { id: string; autor: "assistente"; pergunta: QuestionIn; carregando: true; inicio: number }
  | { id: string; autor: "assistente"; pergunta: QuestionIn; carregando: false; vista: View };

export function daResposta(r: Answer): View {
  return {
    consultaId: r.consulta_id,
    status: r.status,
    conclusao: r.conclusao,
    texto: r.resposta,
    motivo: r.motivo,
    fontes: r.fontes,
    semBula: r.medicamentos_nao_indexados,
    classes: r.classes_terapeuticas,
    principios: r.principios_ativos,
    continuacao: r.continuacao,
    alergiaA: r.alergia_a,
    extrativa: r.extrativa,
    outraExplicacao: r.outra_explicacao,
    esclarecimento: r.esclarecimento,
    erro: null,
    rodadas: r.rodadas,
    latenciaMs: r.latencia_ms,
  };
}

export function doHistorico(c: InquiryDetail): View {
  return {
    consultaId: c.id,
    status: c.status,
    conclusao: c.conclusao ?? null, // consulta antiga, de antes da coluna, vem sem
    texto: c.resposta,
    motivo: c.motivo,
    fontes: c.fontes,
    semBula: [],
    classes: {},
    principios: [],
    // Na outra explicação, a origem é a resposta que não ajudou, não a pergunta anterior.
    continuacao: c.consulta_origem_id !== null && !c.outra_explicacao,
    alergiaA: null,
    extrativa: c.modelo.startsWith("frases da bula"),
    outraExplicacao: c.outra_explicacao,
    esclarecimento: null,
    erro: c.codigo_erro ? { codigo: c.codigo_erro, mensagem: c.mensagem_erro ?? "" } : null,
    rodadas: c.rodadas,
    latenciaMs: c.latencia_ms,
  };
}

export function doErro(e: ApiError): View {
  return {
    consultaId: e.consultaId, // a pergunta seguinte continua desta, mesmo depois do erro
    status: "erro",
    conclusao: null,
    texto: null,
    motivo: null,
    fontes: [],
    semBula: [],
    classes: {},
    principios: [],
    continuacao: false,
    alergiaA: null,
    extrativa: false,
    outraExplicacao: false,
    esclarecimento: null,
    erro: { codigo: e.codigo, mensagem: e.message },
    rodadas: 0,
    latenciaMs: 0,
  };
}
