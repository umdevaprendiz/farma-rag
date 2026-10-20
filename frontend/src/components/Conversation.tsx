import { useEffect, useRef } from "react";

import type { Option, QuestionIn } from "../api";
import type { Message, View } from "../conversation";
import { Logo } from "./Icons";
import { MessageView } from "./Message";
import { MessageInput } from "./MessageInput";

const SUGESTOES = [
  "Qual o melhor remédio pra dor de cabeça?",
  "Ibuprofeno pode ser usado com suspeita de dengue?",
  "Metformina pode ser usada com álcool?",
  "Como armazenar a loratadina?",
];

interface Props {
  mensagens: Message[];
  ocupado: boolean;
  bulas: number | null;
  aoPerguntar: (texto: string) => void;
  aoEscolherOpcao: (pergunta: QuestionIn, vista: View, opcao: Option) => void;
  aoNaoAjudou: (pergunta: QuestionIn, vista: View) => void;
}

export function Conversation({ mensagens, ocupado, bulas, aoPerguntar, aoEscolherOpcao, aoNaoAjudou }: Props) {
  const fim = useRef<HTMLDivElement>(null);
  const ultima = mensagens[mensagens.length - 1];
  // O sistema perguntou algo de volta sem opções ("a alergia é a qual remédio?"): a caixa pede a resposta.
  const respondendo =
    ultima?.autor === "assistente" && !ultima.carregando && ultima.vista.status === "precisa_esclarecimento";

  // Mensagem nova (ou resposta que chegou): rola até o fim, como num chat.
  useEffect(() => {
    fim.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [mensagens]);

  return (
    <div className="conversa">
      <div className="rolagem">
        {mensagens.length === 0 ? (
          <div className="boas-vindas">
            <span className="logo-grande">
              <Logo />
            </span>
            <h1>Pergunte sobre qualquer remédio</h1>
            <p>
              Respondo com o que as bulas dizem, com a fonte de cada frase.
              {bulas ? ` ${bulas} bula${bulas === 1 ? "" : "s"} no acervo.` : ""}
            </p>
            <div className="sugestoes">
              {SUGESTOES.map((s) => (
                <button key={s} type="button" onClick={() => aoPerguntar(s)} disabled={ocupado}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="mensagens" aria-live="polite">
            {mensagens.map((m) => (
              <MessageView
                key={m.id}
                mensagem={m}
                aoEscolherOpcao={aoEscolherOpcao}
                // "Não ajudou" só na última resposta: a outra explicação aparece logo embaixo dela.
                aoNaoAjudou={m === ultima ? aoNaoAjudou : undefined}
                ocupado={ocupado}
              />
            ))}
            <div ref={fim} />
          </div>
        )}
      </div>
      <MessageInput
        ocupado={ocupado}
        aoEnviar={aoPerguntar}
        dica={respondendo ? "Digite o nome do remédio..." : "Pergunte sobre qualquer remédio..."}
      />
    </div>
  );
}
