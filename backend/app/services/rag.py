"""Pergunta -> detecção do medicamento -> busca -> geração -> validação das citações.

Uma tentativa só, por enquanto. As rodadas de verificação e o esclarecimento
(docs/DECISOES.md, seção 14) entram por cima deste fluxo.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from app.config import Settings, get_settings
from app.llm import ollama
from app.schemas.api import MAX_MEDICAMENTOS_POR_PERGUNTA, CodigoErro
from app.schemas.dataset import ROTULO_SECAO, Chunk
from app.services.acervo import Acervo
from app.services.citacoes import SaidaInvalidaError, validar_saida
from app.services.medicamentos import detectar, normalizar

SISTEMA = (
    "Você responde perguntas de profissionais de farmácia consultando bulas. "
    "Use SOMENTE os trechos numerados enviados; não use conhecimento próprio. "
    "Responda em português, em até 4 frases curtas. Não recomende dose nem conduta para um "
    "paciente específico. Devolva só JSON: "
    '{"status": "respondida" ou "nao_encontrada", "afirmacoes": [{"texto": uma frase da resposta, '
    '"trecho": número do trecho de onde ela saiu}]}. '
    'Se os trechos não respondem à pergunta, devolva "status": "nao_encontrada" e "afirmacoes": [].'
)
# Lições do teste de 2026-10-13 com o qwen2.5:1.5b:
# - ele não escreve "[1]" dentro do texto, mesmo pedindo; por isso cada frase vem num objeto com
#   o seu trecho, e o "[n]" é montado aqui (ver _para_saida);
# - exemplo de resposta pronta no prompt é copiado como se fosse a resposta. Não coloque.


def formato(n_trechos: int) -> dict:
    """JSON Schema da saída: o Ollama só deixa o modelo citar trechos que foram enviados."""
    return {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["respondida", "nao_encontrada"]},
            "afirmacoes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "texto": {"type": "string"},
                        "trecho": {"type": "integer", "enum": list(range(1, n_trechos + 1))},
                    },
                    "required": ["texto", "trecho"],
                },
            },
        },
        "required": ["status", "afirmacoes"],
    }


@dataclass(frozen=True)
class Fonte:
    posicao: int  # o [n] do prompt
    chunk: Chunk
    data_versao: date
    score: float
    citado: bool = False


@dataclass
class Resultado:
    status: Literal["respondida", "nao_encontrada", "erro"]
    motivo: Literal["sem_contexto", "medicamento_nao_indexado"] | None = None
    resposta: str | None = None
    fontes: list[Fonte] = field(default_factory=list)
    principios_ativos: list[str] = field(default_factory=list)  # filtro usado na busca
    medicamentos_nao_indexados: list[str] = field(default_factory=list)
    codigo_erro: CodigoErro | None = None
    detalhe_erro: str | None = None
    saida_bruta: str | None = None  # o que o modelo devolveu, para depurar
    tempos_s: dict[str, float] = field(default_factory=dict)
    prompt_tokens: int = 0


def montar_prompt(pergunta: str, fontes: list[Fonte]) -> str:
    trechos = "\n\n".join(
        f"[{f.posicao}] {f.chunk.medicamento} | {ROTULO_SECAO[f.chunk.secao]} | página {f.chunk.pagina}\n{f.chunk.texto}"
        for f in fontes
    )
    return f"TRECHOS:\n{trechos}\n\nPERGUNTA: {pergunta}"


def responder(
    pergunta: str, acervo: Acervo, principios_ativos: list[str] | None = None, s: Settings | None = None
) -> Resultado:
    """`principios_ativos` vazio = detectar o medicamento na própria pergunta."""
    s = s or get_settings()
    r = Resultado(status="nao_encontrada")

    if principios_ativos:
        # O filtro aceita qualquer nome que o catálogo conheça ("losartana", "Cozaar").
        achados = {p: acervo.catalogo.indexados.get(normalizar(p)) for p in principios_ativos}
        desconhecidos = [p for p, ativo in achados.items() if ativo is None]
        if desconhecidos:
            return _erro(r, CodigoErro.MEDICAMENTO_NAO_ENCONTRADO, f"sem bula indexada para: {', '.join(desconhecidos)}")
        r.principios_ativos = sorted(set(achados.values()))
    else:
        deteccao = detectar(pergunta, acervo.catalogo)
        r.principios_ativos = deteccao.indexados[:MAX_MEDICAMENTOS_POR_PERGUNTA]
        r.medicamentos_nao_indexados = deteccao.nao_indexados
        if not deteccao.indexados and deteccao.nao_indexados:
            r.motivo = "medicamento_nao_indexado"
            return r

    try:
        t = time.perf_counter()
        consulta = ollama.embed([pergunta], s)[0]
        achados = acervo.indice.buscar(consulta, s.top_k, r.principios_ativos or None)
        r.tempos_s["busca"] = time.perf_counter() - t
        if s.limiar_similaridade is not None:
            achados = [a for a in achados if a.score >= s.limiar_similaridade]
        r.fontes = [
            Fonte(i, acervo.chunks[a.chunk_id], acervo.bulas[acervo.chunks[a.chunk_id].bula_id].data_versao, a.score)
            for i, a in enumerate(achados, start=1)
        ]
        if not r.fontes:
            r.motivo = "sem_contexto"
            return r

        t = time.perf_counter()
        llm = ollama.chat(SISTEMA, montar_prompt(pergunta, r.fontes), formato(len(r.fontes)), s)
        r.tempos_s["geracao"] = time.perf_counter() - t
    except ollama.OllamaError as e:
        return _erro(r, e.codigo, e.detalhe)
    r.saida_bruta, r.prompt_tokens = llm.texto, llm.prompt_tokens

    try:
        saida = validar_saida(_para_saida(llm.texto), len(r.fontes))
    except SaidaInvalidaError as e:
        return _erro(r, e.codigo, e.detalhe)
    if saida.status == "nao_encontrada":
        r.motivo = "sem_contexto"
        return r
    r.status, r.resposta = "respondida", saida.resposta
    r.fontes = [Fonte(f.posicao, f.chunk, f.data_versao, f.score, f.posicao in saida.citacoes) for f in r.fontes]
    return r


def _erro(r: Resultado, codigo: CodigoErro, detalhe: str) -> Resultado:
    r.status, r.codigo_erro, r.detalhe_erro = "erro", codigo, detalhe
    return r


def _para_saida(bruto: str) -> str | dict:
    """Converte {"status", "afirmacoes"} no formato que validar_saida confere.

    Cada afirmação vira "frase [n]." Na recusa, o que o modelo escreveu a mais é descartado:
    modelo pequeno recusa e ainda explica, e a recusa é o que vale.
    """
    try:
        dados = json.loads(bruto)
        if dados["status"] == "nao_encontrada":
            return {"status": "nao_encontrada"}
        frases = [(a["texto"].strip().rstrip("."), a["trecho"]) for a in dados["afirmacoes"]]
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return bruto  # validar_saida devolve SAIDA_INVALIDA
    return {
        "status": dados["status"],
        "resposta": " ".join(f"{texto} [{trecho}]." for texto, trecho in frases if texto),
        "citacoes": sorted({trecho for texto, trecho in frases if texto}),
    }
