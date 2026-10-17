import type { InquiryStatus } from "./api";

/** "2025-03-15" -> "15/03/2025", sem passar por Date (que mudaria o dia pelo fuso). */
export function dataBula(iso: string): string {
  const [ano, mes, dia] = iso.split("-");
  return `${dia}/${mes}/${ano}`;
}

/** ["a"] -> "a"; ["a", "b", "c"] -> "a, b e c". */
export function emLista(nomes: string[]): string {
  return nomes.length <= 1 ? nomes.join("") : `${nomes.slice(0, -1).join(", ")} e ${nomes[nomes.length - 1]}`;
}

/** 850 -> "850 ms"; 12345 -> "12,3 s". */
export function duracao(ms: number): string {
  if (ms < 1000) return `${ms} ms`;
  return `${(ms / 1000).toLocaleString("pt-BR", { maximumFractionDigits: 1 })} s`;
}

export function semelhanca(score: number): string {
  return score.toLocaleString("pt-BR", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

export const ROTULO_STATUS: Record<InquiryStatus, string> = {
  respondida: "Respondida",
  nao_encontrada: "Não encontrada",
  precisa_esclarecimento: "Pediu esclarecimento",
  erro: "Erro",
};

export const ROTULO_MOTIVO: Record<string, string> = {
  medicamento_nao_indexado: "A bula desse medicamento não está no acervo, e nenhuma outra bula fala disso.",
  sem_contexto: "Nenhuma das bulas consultadas fala disso.",
};

/** Grupo da barra lateral, como nos chats: "Hoje", "Ontem", "Últimos 7 dias", "Anteriores". */
export function grupoDaData(iso: string, agora = new Date()): string {
  const dia = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const dias = Math.round((dia(agora) - dia(new Date(iso))) / 86_400_000);
  if (dias <= 0) return "Hoje";
  if (dias === 1) return "Ontem";
  if (dias < 7) return "Últimos 7 dias";
  return "Anteriores";
}
