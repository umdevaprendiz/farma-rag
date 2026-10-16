import { useEffect, useState } from "react";

import type { Opcao, PerguntaIn } from "../api";
import type { Mensagem, Vista } from "../conversa";
import { ROTULO_MOTIVO, duracao, emLista } from "../formato";
import { AvisoErro } from "./AvisoErro";
import { Fontes, TextoComCitacoes } from "./Citacoes";
import type { FonteAtiva } from "./Citacoes";
import { Logo } from "./Icones";

interface Props {
  mensagem: Mensagem;
  aoEscolherOpcao: (pergunta: PerguntaIn, vista: Vista, opcao: Opcao) => void;
}

export function MensagemVista({ mensagem: m, aoEscolherOpcao }: Props) {
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

function Pensando({ inicio }: { inicio: number }) {
  const [agora, setAgora] = useState(Date.now());
  useEffect(() => {
    const id = setInterval(() => setAgora(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);
  const s = Math.floor((agora - inicio) / 1000);
  return (
    <div className="pensando" role="status">
      <span className="pontos" aria-hidden="true">
        <i />
        <i />
        <i />
      </span>
      <span>Consultando as bulas... {s} s</span>
      {s >= 15 && (
        <p>Nesta máquina (só CPU) a resposta leva de 10 a 45 s. Se ela não passar na verificação, tento de novo, até 3 rodadas.</p>
      )}
    </div>
  );
}

interface RespostaProps {
  vista: Vista;
  aoEscolherOpcao: (opcao: Opcao) => void;
}

function Resposta({ vista: v, aoEscolherOpcao }: RespostaProps) {
  const [ativa, setAtiva] = useState<FonteAtiva | null>(null);
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

      {v.status === "erro" && v.erro && <AvisoErro codigo={v.erro.codigo} mensagem={v.erro.mensagem} />}

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
