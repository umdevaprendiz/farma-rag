import { useEffect, useRef } from "react";

import type { Opcao, PerguntaIn } from "../api";
import type { Mensagem, Vista } from "../conversa";
import { Entrada } from "./Entrada";
import { Logo } from "./Icones";
import { MensagemVista } from "./Mensagem";

const SUGESTOES = [
  "Qual o melhor remédio pra dor de cabeça?",
  "Ibuprofeno pode ser usado com suspeita de dengue?",
  "Metformina pode ser usada com álcool?",
  "Como armazenar a loratadina?",
];

interface Props {
  mensagens: Mensagem[];
  ocupado: boolean;
  bulas: number | null;
  aoPerguntar: (texto: string) => void;
  aoEscolherOpcao: (pergunta: PerguntaIn, vista: Vista, opcao: Opcao) => void;
}

export function Conversa({ mensagens, ocupado, bulas, aoPerguntar, aoEscolherOpcao }: Props) {
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
              <MensagemVista key={m.id} mensagem={m} aoEscolherOpcao={aoEscolherOpcao} />
            ))}
            <div ref={fim} />
          </div>
        )}
      </div>
      <Entrada
        ocupado={ocupado}
        aoEnviar={aoPerguntar}
        dica={respondendo ? "Digite o nome do remédio..." : "Pergunte sobre qualquer remédio..."}
      />
    </div>
  );
}
