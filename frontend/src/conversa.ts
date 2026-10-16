// O modelo da conversa na tela. A API não guarda conversa (cada pergunta é independente); a lista
// de mensagens vive só aqui, e o histórico da barra lateral vem da API.

import type { CodigoErro, ConsultaDetalhe, ErroApi, Esclarecimento, Fonte, PerguntaIn, Resposta } from "./api";

/** O que a tela mostra de uma resposta, venha ela da pergunta de agora, do histórico ou de um erro. */
export interface Vista {
  consultaId: string | null; // nulo só em erro (a API não devolve o id no ErroOut)
  status: "respondida" | "nao_encontrada" | "precisa_esclarecimento" | "erro";
  texto: string | null;
  motivo: string | null;
  fontes: Fonte[];
  semBula: string[]; // remédios citados na pergunta sem bula no acervo
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
    esclarecimento: null,
    erro: c.codigo_erro ? { codigo: c.codigo_erro, mensagem: c.mensagem_erro ?? "" } : null,
    rodadas: c.rodadas,
    latenciaMs: c.latencia_ms,
  };
}

export function doErro(e: ErroApi): Vista {
  return {
    consultaId: null,
    status: "erro",
    texto: null,
    motivo: null,
    fontes: [],
    semBula: [],
    esclarecimento: null,
    erro: { codigo: e.codigo, mensagem: e.message },
    rodadas: 0,
    latenciaMs: 0,
  };
}
