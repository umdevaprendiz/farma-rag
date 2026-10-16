// O modelo da conversa na tela. A API não guarda conversa (cada pergunta é independente); a lista
// de mensagens vive só aqui, e o histórico da barra lateral vem da API.

import type { CodigoErro, ConsultaDetalhe, ErroApi, Esclarecimento, Fonte, PerguntaIn, Resposta } from "./api";

/** O que a tela mostra de uma resposta, venha ela da pergunta de agora, do histórico ou de um erro. */
export interface Vista {
  consultaId: string | null; // nulo quando a pergunta nem chegou à API
  status: "respondida" | "nao_encontrada" | "precisa_esclarecimento" | "erro";
  texto: string | null;
  motivo: string | null;
  fontes: Fonte[];
  semBula: string[]; // remédios citados na pergunta sem bula no acervo
  classes: Record<string, string>; // e para que servem, pelo registro na ANVISA
  principios: string[];
  continuacao: boolean;
  alergiaA: string | null;
  extrativa: boolean;
  esclarecimento: Esclarecimento | null;
  erro: { codigo: CodigoErro | "API_FORA_DO_AR"; mensagem: string } | null;
  rodadas: number;
  latenciaMs: number;
}

export type Mensagem =
  | { id: string; autor: "usuario"; texto: string }
  | { id: string; autor: "assistente"; pergunta: PerguntaIn; carregando: true; inicio: number }
  | { id: string; autor: "assistente"; pergunta: PerguntaIn; carregando: false; vista: Vista };

export function daResposta(r: Resposta): Vista {
  return {
    consultaId: r.consulta_id,
    status: r.status,
    texto: r.resposta,
    motivo: r.motivo,
    fontes: r.fontes,
    semBula: r.medicamentos_nao_indexados,
    classes: r.classes_terapeuticas,
    principios: r.principios_ativos,
    continuacao: r.continuacao,
    alergiaA: r.alergia_a,
    extrativa: r.extrativa,
    esclarecimento: r.esclarecimento,
    erro: null,
    rodadas: r.rodadas,
    latenciaMs: r.latencia_ms,
  };
}

export function doHistorico(c: ConsultaDetalhe): Vista {
  return {
    consultaId: c.id,
    status: c.status,
    texto: c.resposta,
    motivo: c.motivo,
    fontes: c.fontes,
    semBula: [],
    classes: {},
    principios: [],
    continuacao: c.consulta_origem_id !== null,
    alergiaA: null,
    extrativa: c.modelo.startsWith("frases da bula"),
    esclarecimento: null,
    erro: c.codigo_erro ? { codigo: c.codigo_erro, mensagem: c.mensagem_erro ?? "" } : null,
    rodadas: c.rodadas,
    latenciaMs: c.latencia_ms,
  };
}

export function doErro(e: ErroApi): Vista {
  return {
    consultaId: e.consultaId, // a pergunta seguinte continua desta, mesmo depois do erro
    status: "erro",
    texto: null,
    motivo: null,
    fontes: [],
    semBula: [],
    classes: {},
    principios: [],
    continuacao: false,
    alergiaA: null,
    extrativa: false,
    esclarecimento: null,
    erro: { codigo: e.codigo, mensagem: e.message },
    rodadas: 0,
    latenciaMs: 0,
  };
}
