"""Explicação em palavras simples do "para que serve" copiado da bula (F4, docs/DECISOES.md, seção 37).

Pedido do usuário (2026-10-24): "o para que serve pode vir diretamente da bula, porém com a explicação
também do agente". A resposta continua sendo as frases da seção de indicações, copiadas (services/
indication.py); depois delas, o modelo escreve de 1 a 3 frases curtas dizendo o mesmo em palavras do dia a
dia. A explicação é um extra: se o modelo cair, demorar ou escrever algo que a bula não diz, a resposta
sai só com a cópia, nunca com erro.

O que a explicação precisa para entrar, frase a frase (as reprovadas saem, as outras ficam):
- as regras da fundamentação (services/grounding.py), mais exigente que na resposta: `EXPLICACAO_MIN_TERMOS`
  (padrão 0,8) dos termos no trecho, contando as expressões do glossário ("dor de cabeça" = cefaleia);
- se a conclusão foi "não está nas indicações" (ou "não cita, indica para dor em geral"), não falar do
  uso perguntado: "o paracetamol serve para ansiedade e dor de cabeça" passaria pela fração de termos;
- não repetir a frase da bula (aí não explica nada).
Na ressalva ("não é recomendado trocar... na insuficiência cardíaca") não há explicação: o "não" é o que
importa, e a paráfrase pode perdê-lo.
"""

import json
import re

from app.schemas.dataset import Chunk
from app.services import glossary, grounding
from app.services.medicines import Catalog, normalizar

MAX_FRASES = 3
_SEMELHANCA_COPIA = 0.8  # Jaccard das palavras: acima disso, a frase é a da bula repetida

SISTEMA = (
    "Você explica frases de bula para o balconista de farmácia, em português simples, como se explicasse "
    "no balcão. Use só o que as frases dizem: não acrescente uso, dose, efeito nem conselho que elas não "
    "trazem. Troque o termo técnico pela palavra do dia a dia. Escreva de 1 a 3 frases curtas, cada uma "
    'dizendo de qual frase numerada ela saiu. Devolva só JSON: {"afirmacoes": [{"texto": uma frase da '
    'explicação, "trecho": número da frase de onde ela saiu}]}'
)
# Sem a pergunta no prompt, de propósito: com "serve para ansiedade?", o modelo pequeno responde à
# pergunta ("não serve para ansiedade") em vez de explicar as frases, e a conclusão já respondeu.
# Sem exemplo de explicação pronta (o modelo copia, ver rag.py); as "palavras do dia a dia" do glossário
# são vocabulário, não resposta.


def montar_prompt(frases: list[tuple[str, int]], chunks: dict[int, Chunk]) -> str:
    """As frases copiadas, cada uma com o [n] do trecho de onde veio, e as palavras do glossário que elas
    usam."""
    linhas = []
    for texto, n in frases:
        prefixo = f"{chunks[n].medicamento}: "
        linhas.append(f"[{n}] {prefixo}{texto.removeprefix(prefixo)}")
    dicas = glossary.dicas(" ".join(t for t, _ in frases))
    extra = "\n\nPALAVRAS DO DIA A DIA: " + "; ".join(f"{t} = {e}" for t, e in dicas) if dicas else ""
    return "FRASES DA BULA:\n" + "\n".join(linhas) + extra


def formato(posicoes: list[int]) -> dict:
    """JSON Schema da saída: só os trechos das frases enviadas podem ser citados."""
    return {
        "type": "object",
        "properties": {
            "afirmacoes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"texto": {"type": "string"}, "trecho": {"type": "integer", "enum": posicoes}},
                    "required": ["texto", "trecho"],
                },
            },
        },
        "required": ["afirmacoes"],
    }


def _ler(bruto: str) -> list[tuple[str, int]]:
    try:
        dados = json.loads(bruto)
        pares = [(a["texto"].strip().rstrip("."), a["trecho"]) for a in dados["afirmacoes"]]
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return []
    return [(t, n) for t, n in pares if t and isinstance(n, int)]


def _palavras(texto: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", normalizar(texto)))


def _repete(texto: str, copiadas: list[str]) -> bool:
    a = _palavras(texto)
    return any(a and len(a & (b := _palavras(c))) / len(a | b) >= _SEMELHANCA_COPIA for c in copiadas)


def _fala_de(texto: str, grupos: list[tuple[str, ...]]) -> bool:
    termos = tuple(t for g in grupos for t in g)
    return bool(termos) and any(p.startswith(termos) for p in _palavras(texto))


def conferir(
    bruto: str,
    frases: list[tuple[str, int]],
    chunks: dict[int, Chunk],
    min_termos: float,
    catalogo: Catalog | None = None,
    proibidos: list[tuple[str, ...]] = (),
) -> tuple[list[tuple[str, int]], list[str]]:
    """As frases da explicação que passam, como (texto, trecho), e o motivo de cada uma que saiu.

    `frases`: as copiadas da bula (a resposta); `proibidos`: os termos do uso que a bula não indica
    (conclusão "ausente" ou "geral"), que a explicação não pode citar."""
    pares = [(t, n) for t, n in _ler(bruto) if n in chunks][:MAX_FRASES]
    if not pares:
        return [], ["o modelo não devolveu explicação"]
    corrigidas, _ = grounding.corrigir_citacoes(pares, chunks, min_termos, catalogo)
    copiadas = [t for t, _ in frases]
    aceitas: list[tuple[str, int]] = []
    descartadas: list[str] = []
    for i, (texto, n) in enumerate(corrigidas, start=1):
        problemas = [p.split(": ", 1)[-1] for p in grounding.checar([(texto, n)], chunks, min_termos, catalogo)]
        if not problemas and _fala_de(texto, proibidos):
            problemas = ["fala do uso que a bula não indica"]
        if not problemas and _repete(texto, copiadas):
            problemas = ["repete a frase da bula"]
        if problemas:
            descartadas.append(f"frase {i} [{n}]: {'; '.join(problemas)}")
        else:
            aceitas.append((texto, n))
    return aceitas, descartadas
