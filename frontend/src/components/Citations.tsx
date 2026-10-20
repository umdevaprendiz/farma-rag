import { useEffect, useRef } from "react";
import type { ReactNode } from "react";

import type { Source } from "../api";
import { dataBula, semelhanca } from "../format";

/** Fonte aberta embaixo da resposta. `vez` muda a cada clique: clicar de novo no mesmo [n] rola até ela. */
export interface ActiveSource {
  posicao: number;
  vez: number;
}

interface TextProps {
  texto: string;
  ativa: ActiveSource | null;
  aoCitar: (posicao: number) => void;
}

/** A resposta com cada "[n]" virando um botão que abre a fonte n. */
export function TextoComCitacoes({ texto, ativa, aoCitar }: TextProps) {
  return (
    <p className="resposta-texto">
      {texto.split(/(\[\d+\])/).map((parte, i) => {
        const citacao = /^\[(\d+)\]$/.exec(parte);
        if (!citacao) return parte;
        const n = Number(citacao[1]);
        return (
          <button
            key={i}
            type="button"
            className={`citacao${ativa?.posicao === n ? " ativa" : ""}`}
            onClick={() => aoCitar(n)}
            aria-label={`Ver a fonte ${n}`}
          >
            {n}
          </button>
        );
      })}
    </p>
  );
}

interface SourcesProps {
  titulo: string;
  fontes: Source[];
  ativa: ActiveSource | null;
  aoEscolher: (posicao: number | null) => void;
}

/** As fontes como chips; a escolhida abre embaixo com o trecho inteiro da bula. */
export function Fontes({ titulo, fontes, ativa, aoEscolher }: SourcesProps) {
  const cartao = useRef<HTMLDivElement>(null);
  const escolhida = fontes.find((f) => f.posicao === ativa?.posicao);

  useEffect(() => {
    if (ativa) cartao.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [ativa]);

  return (
    <div className="fontes">
      <span className="fontes-titulo">{titulo}</span>
      <div className="chips-fontes">
        {fontes.map((f) => {
          const aberta = f.posicao === escolhida?.posicao;
          return (
            <button
              key={f.posicao}
              type="button"
              className={`chip-fonte${f.citado ? " citada" : ""}${aberta ? " ativa" : ""}`}
              onClick={() => aoEscolher(aberta ? null : f.posicao)}
              aria-expanded={aberta}
              title={`${f.medicamento} · ${f.rotulo_secao} · página ${f.pagina}`}
            >
              <span className="chip-numero">{f.posicao}</span>
              <span className="chip-nome">{f.medicamento}</span>
              <span className="chip-secao">{f.rotulo_secao}</span>
            </button>
          );
        })}
      </div>
      {escolhida && (
        <div ref={cartao}>
          <CartaoFonte fonte={escolhida} />
        </div>
      )}
    </div>
  );
}

/** O trecho com as frases usadas na resposta marcadas. As posições vêm do Python, que conta
 * caracteres (code points); o JavaScript conta unidades UTF-16, e Array.from conta como o Python. */
function TrechoDestacado({ texto, destaques }: { texto: string; destaques: [number, number][] }) {
  const letras = Array.from(texto);
  const partes: ReactNode[] = [];
  let pos = 0;
  for (const [inicio, fim] of [...destaques].sort((a, b) => a[0] - b[0])) {
    if (inicio < pos || fim <= inicio || fim > letras.length) continue; // sobreposta ou fora do trecho
    if (inicio > pos) partes.push(letras.slice(pos, inicio).join(""));
    partes.push(<mark key={inicio}>{letras.slice(inicio, fim).join("")}</mark>);
    pos = fim;
  }
  partes.push(letras.slice(pos).join(""));
  return <>{partes}</>;
}

function CartaoFonte({ fonte: f }: { fonte: Source }) {
  const bula = [
    f.fabricante,
    f.numero_registro && `Reg. ANVISA ${f.numero_registro}`,
    `bula de ${dataBula(f.data_versao)}`,
  ].filter(Boolean);
  return (
    <article className={`cartao-fonte${f.citado ? " citada" : ""}`}>
      <header>
        <strong>
          [{f.posicao}] {f.medicamento}
        </strong>
        <span>
          {f.rotulo_secao} · página {f.pagina}
        </span>
      </header>
      <p className="cartao-bula">{bula.join(" · ")}</p>
      <blockquote>
        <TrechoDestacado texto={f.trecho} destaques={f.destaques} />
      </blockquote>
      <footer>
        <span>
          {!f.citado
            ? "Consultado, não usado na resposta"
            : f.destaques.length > 0
              ? "Usado na resposta: em destaque, as frases que a sustentam"
              : "Usado na resposta"}
        </span>
        <span title="Similaridade de cosseno entre a pergunta e o trecho">semelhança {semelhanca(f.score)}</span>
      </footer>
    </article>
  );
}
