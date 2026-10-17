import type { Health, InquirySummary } from "../api";
import { ROTULO_STATUS, grupoDaData } from "../format";
import { IconeFechar, IconeNovo, Logo } from "./Icons";

interface Props {
  historico: InquirySummary[] | null;
  erroHistorico: string | null;
  aberta: string | null; // consulta do histórico que está na tela
  saude: Health | null;
  apiFora: string | null;
  aoNova: () => void;
  aoAbrir: (id: string) => void;
  aoFechar: () => void; // só no celular, onde a barra é um menu
}

export function Sidebar({ historico, erroHistorico, aberta, saude, apiFora, aoNova, aoAbrir, aoFechar }: Props) {
  const grupos = new Map<string, InquirySummary[]>();
  for (const c of historico ?? []) {
    const grupo = grupoDaData(c.criado_em);
    grupos.set(grupo, [...(grupos.get(grupo) ?? []), c]);
  }
  const estado = apiFora ? "fora" : saude ? saude.status : "conferindo";

  return (
    <aside className="barra" aria-label="Conversas">
      <div className="barra-topo">
        <span className="marca">
          <span className="marca-logo">
            <Logo />
          </span>
          FarmaRAG
        </span>
        <button type="button" className="icone-botao so-celular" onClick={aoFechar} aria-label="Fechar menu">
          <IconeFechar />
        </button>
      </div>

      <button type="button" className="nova" onClick={aoNova}>
        <IconeNovo />
        Nova pergunta
      </button>

      <nav className="recentes" aria-label="Perguntas recentes">
        {erroHistorico && <p className="recentes-vazio">Histórico indisponível.</p>}
        {historico && historico.length === 0 && <p className="recentes-vazio">Suas perguntas aparecem aqui.</p>}
        {[...grupos].map(([grupo, consultas]) => (
          <section key={grupo}>
            <h2>{grupo}</h2>
            <ul>
              {consultas.map((c) => (
                <li key={c.id}>
                  <button
                    type="button"
                    className={c.id === aberta ? "ativa" : ""}
                    onClick={() => aoAbrir(c.id)}
                    title={`${c.pergunta} (${ROTULO_STATUS[c.status]})`}
                  >
                    <span className={`ponto ponto-${c.status}`} aria-label={ROTULO_STATUS[c.status]} />
                    <span className="recente-texto">{c.pergunta}</span>
                  </button>
                </li>
              ))}
            </ul>
          </section>
        ))}
      </nav>

      <div className="barra-rodape">
        <span className={`saude saude-${estado}`} title={(apiFora ? [apiFora] : (saude?.problemas ?? [])).join("\n") || undefined}>
          {estado === "ok" && `No ar · ${saude?.bulas} bula${saude?.bulas === 1 ? "" : "s"}`}
          {estado === "degradado" && "Com problemas"}
          {estado === "fora" && "API fora do ar"}
          {estado === "conferindo" && "Conferindo..."}
        </span>
      </div>
    </aside>
  );
}
