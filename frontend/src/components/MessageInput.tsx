import { useEffect, useRef, useState } from "react";

import { IconeEnviar } from "./Icons";

const MIN = 5;
const MAX = 500;
const ALTURA_MAX = 200;

interface Props {
  ocupado: boolean;
  aoEnviar: (texto: string) => void;
  dica: string; // o placeholder: muda quando o sistema perguntou algo de volta
}

/** A caixa de mensagem: cresce com o texto, Enter envia, Shift+Enter quebra a linha. */
export function MessageInput({ ocupado, aoEnviar, dica }: Props) {
  const [texto, setTexto] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);
  const tamanho = texto.trim().length;
  // A resposta a um "qual remédio?" pode ser curta ("AAS"): o mínimo de 5 letras é da API, então
  // a palavra curta é completada com o pedido.
  const valida = tamanho >= 2 && tamanho <= MAX;

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, ALTURA_MAX)}px`;
  }, [texto]);

  useEffect(() => {
    if (!ocupado) ref.current?.focus();
  }, [ocupado]);

  function enviar() {
    if (!valida || ocupado) return;
    const limpo = texto.trim();
    aoEnviar(limpo.length < MIN ? `Remédio: ${limpo}` : limpo);
    setTexto("");
  }

  return (
    <div className="entrada-area">
      <form
        className="entrada"
        onSubmit={(e) => {
          e.preventDefault();
          enviar();
        }}
      >
        <textarea
          ref={ref}
          rows={1}
          value={texto}
          maxLength={MAX}
          placeholder={dica}
          aria-label="Pergunta"
          onChange={(e) => setTexto(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              enviar();
            }
          }}
        />
        <button type="submit" className="enviar" disabled={!valida || ocupado} aria-label="Enviar pergunta">
          <IconeEnviar />
        </button>
      </form>
      <p className="rodape-entrada">
        {tamanho === 1
          ? "Escreva um pouco mais."
          : tamanho > MAX - 50
            ? `${tamanho}/${MAX} caracteres.`
            : "Responde com o que está nas bulas do acervo e pode errar: confira a fonte. Não substitui o farmacêutico. Não escreva nome nem dado de paciente."}
      </p>
    </div>
  );
}
