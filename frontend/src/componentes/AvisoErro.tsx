import type { ErroApi } from "../api";

const TITULO: Record<ErroApi["codigo"], string> = {
  RESPOSTA_NAO_VERIFICADA: "Nenhuma resposta passou na verificação",
  SAIDA_INVALIDA: "O modelo devolveu a resposta num formato inválido",
  CITACAO_INVALIDA: "O modelo citou os trechos de forma incoerente",
  OLLAMA_INDISPONIVEL: "O modelo de linguagem (Ollama) está fora do ar",
  BANCO_INDISPONIVEL: "O banco de dados não está disponível",
  TEMPO_ESGOTADO: "O modelo demorou demais para responder",
  MEDICAMENTO_NAO_ENCONTRADO: "Medicamento sem bula indexada",
  ENTRADA_INVALIDA: "Pergunta inválida",
  CONSULTA_NAO_ENCONTRADA: "Consulta não encontrada",
  API_FORA_DO_AR: "A API está fora do ar",
};

// Erros em que o sistema preferiu não responder a responder sem conferir com a bula.
const RECUSA_SEGURA = new Set(["RESPOSTA_NAO_VERIFICADA", "SAIDA_INVALIDA", "CITACAO_INVALIDA"]);

interface Props {
  codigo: ErroApi["codigo"];
  mensagem: string;
}

/** O erro com título claro. No 502, cada rodada que falhou vira um item da lista. */
export function AvisoErro({ codigo, mensagem }: Props) {
  // "Nenhuma resposta passou ... (limite de rodadas). rodada 1: ...; rodada 2: ..."
  const [resumo, ...rodadas] = mensagem.split(/(?:\.|;)\s+(?=rodada \d+:)/);
  return (
    <div className="aviso aviso-erro" role="alert">
      <strong>{TITULO[codigo] ?? codigo}</strong>
      {RECUSA_SEGURA.has(codigo) && (
        <p>Em vez de mostrar uma resposta que não confere com a bula, o sistema parou.</p>
      )}
      <p>{resumo}</p>
      {rodadas.length > 0 && (
        <ul>
          {rodadas.map((r) => (
            <li key={r}>{r}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
