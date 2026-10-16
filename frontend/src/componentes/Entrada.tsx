import { useEffect, useRef, useState } from "react";

import { IconeEnviar } from "./Icones";

const MIN = 5;
const MAX = 500;
const ALTURA_MAX = 200;

interface Props {
  ocupado: boolean;
  aoEnviar: (texto: string) => void;
}

/** A caixa de mensagem: cresce com o texto, Enter envia, Shift+Enter quebra a linha. */
export function Entrada({ ocupado, aoEnviar }: Props) {
  const [texto, setTexto] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);
  const tamanho = texto.trim().length;
  const valida = tamanho >= MIN && tamanho <= MAX;

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
    aoEnviar(texto.trim());
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
          placeholder="Pergunte sobre qualquer remédio..."
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
        {tamanho > 0 && tamanho < MIN
          ? `Escreva pelo menos ${MIN} caracteres.`
          : tamanho > MAX - 50
            ? `${tamanho}/${MAX} caracteres.`
            : "Responde com o que está nas bulas do acervo e pode errar: confira a fonte. Não substitui o farmacêutico."}
      </p>
    </div>
  );
}
