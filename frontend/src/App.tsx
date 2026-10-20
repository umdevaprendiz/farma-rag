import { useCallback, useEffect, useState } from "react";

import { ApiError, api } from "./api";
import type { Answer, Health, InquirySummary, Option, QuestionIn } from "./api";
import { Conversation } from "./components/Conversation";
import { IconeMenu, IconeNovo } from "./components/Icons";
import { Sidebar } from "./components/Sidebar";
import { daResposta, doErro, doHistorico } from "./conversation";
import type { Message, View } from "./conversation";

const INTERVALO_SAUDE_MS = 30_000;
const NO_HISTORICO = 50;

function comoErroApi(e: unknown): ApiError {
  return e instanceof ApiError ? e : new ApiError("API_FORA_DO_AR", String(e), 0);
}

export function App() {
  const [mensagens, setMensagens] = useState<Message[]>([]);
  const [ocupado, setOcupado] = useState(false);
  const [aberta, setAberta] = useState<string | null>(null);
  const [historico, setHistorico] = useState<InquirySummary[] | null>(null);
  const [erroHistorico, setErroHistorico] = useState<string | null>(null);
  const [saude, setSaude] = useState<Health | null>(null);
  const [apiFora, setApiFora] = useState<string | null>(null);
  const [menuAberto, setMenuAberto] = useState(false);

  const atualizarHistorico = useCallback(() => {
    api
      .historico(1, NO_HISTORICO)
      .then((h) => {
        setHistorico(h.itens);
        setErroHistorico(null);
      })
      .catch((e: ApiError) => setErroHistorico(e.message));
  }, []);

  // Saúde a cada 30 s: o banco ou o Ollama podem cair (ou subir) com a tela aberta.
  useEffect(() => {
    const conferir = () =>
      api
        .saude()
        .then((s) => {
          setSaude(s);
          setApiFora(null);
        })
        .catch((e: ApiError) => setApiFora(e.message));
    conferir();
    const id = setInterval(conferir, INTERVALO_SAUDE_MS);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    atualizarHistorico();
  }, [atualizarHistorico]);

  function perguntar(entrada: QuestionIn, textoDoUsuario: string) {
    if (ocupado) return;
    // A última resposta da conversa vai junto: a pergunta nova pode continuar dela ("e para
    // criança?") ou responder a um "a alergia é a qual remédio?". "Nova pergunta" zera a conversa.
    const ultima = [...mensagens].reverse().find((m) => m.autor === "assistente" && !m.carregando);
    if (!entrada.consulta_origem_id && ultima?.autor === "assistente" && !ultima.carregando && ultima.vista.consultaId) {
      entrada = { ...entrada, consulta_origem_id: ultima.vista.consultaId };
    }
    enviar(entrada, textoDoUsuario, () => api.perguntar(entrada));
  }

  function naoAjudou(pergunta: QuestionIn, vista: View) {
    // A API refaz a pergunta da consulta sem os trechos citados; a tela só diz qual consulta.
    if (ocupado || !vista.consultaId) return;
    const consultaId = vista.consultaId;
    enviar(pergunta, "Não ajudou, explique de outro jeito.", () => api.outraExplicacao(consultaId));
  }

  /** Mostra a mensagem do usuário e a espera, faz o pedido e troca a espera pela resposta. */
  async function enviar(entrada: QuestionIn, textoDoUsuario: string, pedido: () => Promise<Answer>) {
    const id = crypto.randomUUID();
    setAberta(null);
    setOcupado(true);
    setMensagens((m) => [
      ...m,
      { id: `${id}-pergunta`, autor: "usuario", texto: textoDoUsuario },
      { id, autor: "assistente", pergunta: entrada, carregando: true, inicio: Date.now() },
    ]);
    let vista: View;
    try {
      vista = daResposta(await pedido());
    } catch (e) {
      vista = doErro(comoErroApi(e));
    }
    // Se a conversa foi trocada no meio, a resposta não aparece aqui, mas fica no histórico.
    setMensagens((m) => m.map((x) => (x.id === id ? { id, autor: "assistente", pergunta: entrada, carregando: false, vista } : x)));
    setOcupado(false);
    atualizarHistorico();
  }

  function escolherOpcao(pergunta: QuestionIn, vista: View, opcao: Option) {
    // O reenvio leva o filtro escolhido: não há segundo esclarecimento para a mesma pergunta.
    perguntar(
      { pergunta: pergunta.pergunta, principios_ativos: opcao.principios_ativos, consulta_origem_id: vista.consultaId ?? undefined },
      opcao.rotulo,
    );
  }

  function novaPergunta() {
    setMensagens([]);
    setAberta(null);
    setMenuAberto(false);
  }

  async function abrirConsulta(id: string) {
    setMenuAberto(false);
    setAberta(id);
    let vista: View;
    let pergunta: QuestionIn = { pergunta: "", principios_ativos: [] };
    try {
      const c = await api.consulta(id);
      pergunta = { pergunta: c.pergunta, principios_ativos: c.principios_ativos_filtro };
      vista = doHistorico(c);
    } catch (e) {
      vista = doErro(comoErroApi(e));
    }
    setMensagens([
      ...(pergunta.pergunta ? [{ id: `${id}-pergunta`, autor: "usuario" as const, texto: pergunta.pergunta }] : []),
      { id, autor: "assistente", pergunta, carregando: false, vista },
    ]);
  }

  const problemas = apiFora ? [apiFora] : (saude?.problemas ?? []);

  return (
    <div className={`app${menuAberto ? " menu-aberto" : ""}`}>
      <Sidebar
        historico={historico}
        erroHistorico={erroHistorico}
        aberta={aberta}
        saude={saude}
        apiFora={apiFora}
        aoNova={novaPergunta}
        aoAbrir={abrirConsulta}
        aoFechar={() => setMenuAberto(false)}
      />
      {menuAberto && <div className="veu" onClick={() => setMenuAberto(false)} />}

      <main className="principal">
        <header className="topo-celular">
          <button type="button" className="icone-botao" onClick={() => setMenuAberto(true)} aria-label="Abrir menu">
            <IconeMenu />
          </button>
          <span>FarmaRAG</span>
          <button type="button" className="icone-botao" onClick={novaPergunta} aria-label="Nova pergunta">
            <IconeNovo />
          </button>
        </header>

        {problemas.length > 0 && (
          <div className="faixa-problemas" role="status">
            {problemas.map((p) => (
              <p key={p}>{p}</p>
            ))}
          </div>
        )}

        <Conversation
          mensagens={mensagens}
          ocupado={ocupado}
          bulas={saude?.bulas ?? null}
          aoPerguntar={(texto) => perguntar({ pergunta: texto, principios_ativos: [] }, texto)}
          aoEscolherOpcao={escolherOpcao}
          aoNaoAjudou={naoAjudou}
        />
      </main>
    </div>
  );
}
