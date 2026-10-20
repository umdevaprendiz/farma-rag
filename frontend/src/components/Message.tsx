import { useEffect, useState } from "react";

import type { Option, QuestionIn } from "../api";
import type { Message, View } from "../conversation";
import { ROTULO_MOTIVO, duracao, emLista } from "../format";
import { Fontes, TextoComCitacoes } from "./Citations";
import type { ActiveSource } from "./Citations";
import { ErrorNotice } from "./ErrorNotice";
import { Logo } from "./Icons";

interface Props {
  mensagem: Message;
  aoEscolherOpcao: (pergunta: QuestionIn, vista: View, opcao: Option) => void;
}

export function MessageView({ mensagem: m, aoEscolherOpcao }: Props) {
  if (m.autor === "usuario") {
    return (
      <div className="msg msg-usuario">
        <div className="bolha">{m.texto}</div>
      </div>
    );
  }
  return (
    <div className="msg msg-assistente">
      <span className="avatar">
        <Logo />
      </span>
      <div className="conteudo">
        {m.carregando ? (
          <Pensando inicio={m.inicio} />
        ) : (
          <Resposta vista={m.vista} aoEscolherOpcao={(opcao) => aoEscolherOpcao(m.pergunta, m.vista, opcao)} />
        )}
      </div>
    </div>
  );
}

// Frases da espera, trocadas a cada SEGUNDOS_POR_FRASE em ordem sorteada. Nenhuma pode dar a
// entender que uma pessoa está respondendo.
const FRASES_ESPERA = [
  "Pensando...",
  "Memorizando...",
  "Caçando um urso...",
  "Folheando as bulas...",
  "Lendo a letra miúda...",
  "Procurando a página certa...",
  "Conferindo as citações...",
  "Tirando a poeira do bulário...",
  "Salvando o mundo...",
  "Procurando um tesouro...",
];
const SEGUNDOS_POR_FRASE = 3;

function embaralhar<T>(lista: readonly T[]): T[] {
  const copia = [...lista];
  for (let i = copia.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copia[i], copia[j]] = [copia[j], copia[i]];
  }
  return copia;
}

function Pensando({ inicio }: { inicio: number }) {
  const [agora, setAgora] = useState(Date.now());
  // Sorteada uma vez por pergunta: sortear a cada render trocaria a frase a cada segundo.
  const [frases] = useState(() => embaralhar(FRASES_ESPERA));
  useEffect(() => {
    const id = setInterval(() => setAgora(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);
  const s = Math.max(0, Math.floor((agora - inicio) / 1000));
  const frase = frases[Math.floor(s / SEGUNDOS_POR_FRASE) % frases.length];
  // O leitor de tela ouve só o texto fixo do status; a frase e o contador mudam e ficam de fora.
  return (
    <div className="pensando">
      <span className="pontos" aria-hidden="true">
        <i />
        <i />
        <i />
      </span>
      <span role="status" className="sr-only">
        Consultando as bulas
      </span>
      <span key={frase} className="frase-espera" aria-hidden="true">
        {frase}
      </span>
      <span className="segundos" aria-hidden="true">
        {s} s
      </span>
    </div>
  );
}

interface AnswerProps {
  vista: View;
  aoEscolherOpcao: (opcao: Option) => void;
}

function Resposta({ vista: v, aoEscolherOpcao }: AnswerProps) {
  const [ativa, setAtiva] = useState<ActiveSource | null>(null);
  const abrir = (posicao: number | null) =>
    setAtiva((a) => (posicao === null ? null : { posicao, vez: (a?.vez ?? 0) + 1 }));

  const classes = v.semBula.filter((n) => v.classes[n]).map((n) => `${n}: ${v.classes[n]}`);
  return (
    <>
      {v.continuacao && (
        <p className="meta">
          Continuando a pergunta anterior{v.principios.length > 0 ? ` (sobre ${emLista(v.principios)})` : ""}.
        </p>
      )}

      {v.alergiaA && v.status === "respondida" && (
        <p className="nota-aviso">
          Alternativas a <strong>{v.alergiaA}</strong> pelas bulas do acervo. Antes de trocar, confira na bula de cada
          uma as contraindicações por alergia (hipersensibilidade), inclusive a alergia cruzada.
        </p>
      )}

      {v.semBula.length > 0 && v.status === "respondida" && !v.alergiaA && (
        <p className="nota-aviso">
          {v.semBula.length === 1 ? "A bula de " : "As bulas de "}
          <strong>{emLista(v.semBula)}</strong>
          {v.semBula.length === 1 ? " não está" : " não estão"} no acervo: a resposta usa o que as outras bulas dizem.
          Confira em cada fonte de qual medicamento é a informação.
        </p>
      )}

      {v.status === "respondida" && v.texto && <TextoComCitacoes texto={v.texto} ativa={ativa} aoCitar={abrir} />}

      {v.status === "nao_encontrada" && (
        <p className="resposta-texto">
          Não encontrei isso nas bulas do acervo. {ROTULO_MOTIVO[v.motivo ?? ""] ?? ""}
        </p>
      )}

      {classes.length > 0 && (
        <p className="meta" title="Classe terapêutica no registro de medicamentos da ANVISA (dados abertos)">
          Para que serve, pelo registro na ANVISA: {classes.join("; ")}.
        </p>
      )}

      {v.status === "precisa_esclarecimento" && v.esclarecimento && (
        <div className="esclarecimento">
          <p className="resposta-texto">{v.esclarecimento.pergunta}</p>
          {v.esclarecimento.opcoes.length > 0 && (
            <div className="opcoes">
              {v.esclarecimento.opcoes.map((o) => (
                <button key={o.rotulo} type="button" onClick={() => aoEscolherOpcao(o)}>
                  {o.rotulo}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {v.status === "erro" && v.erro && <ErrorNotice codigo={v.erro.codigo} mensagem={v.erro.mensagem} />}

      {v.fontes.length > 0 && (
        <Fontes
          titulo={v.status === "respondida" ? "Fontes" : "Trechos consultados"}
          fontes={v.fontes}
          ativa={ativa}
          aoEscolher={abrir}
        />
      )}

      {(v.status === "respondida" || v.status === "nao_encontrada") && (
        <p className="meta">
          {v.extrativa
            ? `Frases copiadas da bula, sem resumo do modelo · ${duracao(v.latenciaMs)}`
            : `${v.rodadas} rodada${v.rodadas === 1 ? "" : "s"} · ${duracao(v.latenciaMs)}`}
        </p>
      )}
    </>
  );
}
