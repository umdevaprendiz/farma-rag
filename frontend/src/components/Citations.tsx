import { useEffect, useRef } from "react";
import type { ReactNode } from "react";

import type { Mark, Source } from "../api";
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
  marcas?: Mark[]; // F4: negrito do que foi perguntado e termos técnicos explicados
  // Com as fontes, a resposta que junta bulas diferentes vira lista, uma afirmação por item.
  fontes?: Source[];
}

interface Citation {
  inicio: number;
  fim: number;
  n: number;
}

/** Os "[n]" do texto, em posições de code point (as marcas vêm do Python, que conta assim). */
function citacoesDe(letras: string[]): Citation[] {
  const saida: Citation[] = [];
  for (let i = 0; i < letras.length; i++) {
    if (letras[i] !== "[") continue;
    let j = i + 1;
    while (j < letras.length && /\d/.test(letras[j])) j++;
    if (j > i + 1 && letras[j] === "]") {
      saida.push({ inicio: i, fim: j + 1, n: Number(letras.slice(i + 1, j).join("")) });
      i = j;
    }
  }
  return saida;
}

/** Onde cada afirmação começa e termina: o Python junta as frases como "frase [n]. frase [m]." */
function afirmacoesDe(letras: string[], citacoes: Citation[]): [number, number][] {
  const faixas: [number, number][] = [];
  let inicio = 0;
  for (const c of citacoes) {
    if (letras[c.fim] !== ".") continue;
    faixas.push([inicio, c.fim + 1]);
    inicio = c.fim + 1;
    while (inicio < letras.length && /\s/.test(letras[inicio])) inicio++;
  }
  if (letras.slice(inicio).join("").trim()) faixas.push([inicio, letras.length]);
  return faixas;
}

/** O pedaço [inicio, fim) do texto: cada "[n]" vira o botão da fonte n, o foco fica em negrito e o termo
 * técnico ganha a palavra do dia a dia entre parênteses (marcada como do sistema, não da bula). */
function pedacos(
  letras: string[],
  inicio: number,
  fim: number,
  citacoes: Citation[],
  marcas: Mark[],
  ativa: ActiveSource | null,
  aoCitar: (posicao: number) => void,
): ReactNode[] {
  const dentro = (p: number) => p > inicio && p < fim;
  const cortes = new Set([inicio, fim]);
  for (const c of citacoes) [c.inicio, c.fim].filter(dentro).forEach((p) => cortes.add(p));
  for (const m of marcas) (m.tipo === "foco" ? [m.inicio, m.fim] : [m.fim]).filter(dentro).forEach((p) => cortes.add(p));
  const pontos = [...cortes].sort((a, b) => a - b);
  const saida: ReactNode[] = [];
  for (let k = 0; k + 1 < pontos.length; k++) {
    const [a, b] = [pontos[k], pontos[k + 1]];
    const citacao = citacoes.find((c) => c.inicio === a);
    if (citacao) {
      const n = citacao.n;
      saida.push(
        <button
          key={`c${a}`}
          type="button"
          className={`citacao${ativa?.posicao === n ? " ativa" : ""}`}
          onClick={() => aoCitar(n)}
          aria-label={`Ver a fonte ${n}`}
        >
          {n}
        </button>,
      );
      continue;
    }
    const parte = letras.slice(a, b).join("");
    const emFoco = marcas.some((m) => m.tipo === "foco" && m.inicio <= a && b <= m.fim);
    saida.push(emFoco ? <strong key={`f${a}`} className="foco">{parte}</strong> : parte);
    for (const m of marcas) {
      if (m.tipo === "termo" && m.fim === b && m.explicacao) {
        saida.push(
          <span key={`t${b}`} className="glossario" title="Palavra do dia a dia, do glossário do sistema (não está na bula)">
            {` (${m.explicacao})`}
          </span>,
        );
      }
    }
  }
  return saida;
}

/** A resposta com cada "[n]" virando um botão que abre a fonte n. */
export function TextoComCitacoes({ texto, ativa, aoCitar, marcas = [], fontes }: TextProps) {
  const letras = Array.from(texto);
  const citacoes = citacoesDe(letras);
  const faixas = afirmacoesDe(letras, citacoes);
  // Uma frase por bula ("Tylalgin gotas: ... [1]. Tylalgin 750 mg: ... [2].") lê melhor como lista.
  const remedioDa = ([a, b]: [number, number]) =>
    fontes?.find((f) => f.posicao === citacoes.find((c) => c.inicio >= a && c.fim <= b)?.n)?.medicamento;
  const emLista = !!fontes && faixas.length >= 2 && new Set(faixas.map(remedioDa)).size >= 2;
  if (emLista) {
    return (
      <ul className="resposta-texto resposta-lista">
        {faixas.map(([a, b]) => (
          <li key={a}>{pedacos(letras, a, b, citacoes, marcas, ativa, aoCitar)}</li>
        ))}
      </ul>
    );
  }
  return <p className="resposta-texto">{pedacos(letras, 0, letras.length, citacoes, marcas, ativa, aoCitar)}</p>;
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
