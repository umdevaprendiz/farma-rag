"""Teste manual do agente no terminal.

Uso (da raiz):
  python -m benchmark.perguntar "Losartana pode com ibuprofeno?"
  python -m benchmark.perguntar            (modo conversa: uma pergunta por linha, vazio sai)

Mostra a resposta, os trechos usados (com a nota de semelhança) e os tempos. Quando o modelo
devolve algo inválido, mostra o erro e a saída crua.
"""

import sys

from app.config import get_settings
from app.schemas.dataset import ROTULO_SECAO
from app.services.acervo import Acervo, AcervoIncompletoError
from app.services.rag import Resultado, responder

MAX_TRECHO = 300


def mostrar(r: Resultado) -> None:
    if r.principios_ativos:
        print(f"Medicamento(s) reconhecido(s): {', '.join(r.principios_ativos)}")
    if r.medicamentos_nao_indexados:
        print(f"Sem bula indexada: {', '.join(r.medicamentos_nao_indexados)}")

    if r.status == "respondida":
        print(f"\nRESPOSTA\n{r.resposta}")
    elif r.status == "nao_encontrada":
        motivo = {
            "medicamento_nao_indexado": "não tenho a bula desse medicamento",
            "sem_contexto": "os trechos encontrados não respondem à pergunta",
        }.get(r.motivo, r.motivo)
        print(f"\nNÃO ENCONTREI: {motivo}")
    else:
        print(f"\nERRO {r.codigo_erro.value}: {r.detalhe_erro}")
        if r.saida_bruta:
            print(f"Saída crua do modelo: {r.saida_bruta}")

    if r.fontes:
        print("\nTRECHOS ENVIADOS AO MODELO (* = citado na resposta)")
    for f in r.fontes:
        c = f.chunk
        texto = c.texto if len(c.texto) <= MAX_TRECHO else c.texto[:MAX_TRECHO] + "..."
        print(f" {'*' if f.citado else ' '}[{f.posicao}] {c.medicamento} | {ROTULO_SECAO[c.secao]} | p. {c.pagina} "
              f"| bula de {f.data_versao:%d/%m/%Y} | semelhança {f.score:.2f}\n     {texto}")
    if r.tempos_s:
        tempos = ", ".join(f"{nome} {valor:.1f} s" for nome, valor in r.tempos_s.items())
        print(f"\nTempo: {tempos}" + (f" | prompt de {r.prompt_tokens} tokens" if r.prompt_tokens else ""))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # o terminal do Windows nem sempre está em UTF-8
    s = get_settings()
    try:
        acervo = Acervo.carregar(s)
    except AcervoIncompletoError as e:
        sys.exit(str(e))
    print(f"{len(acervo.bulas)} bula(s), {len(acervo.chunks)} trechos | modelo {s.llm_modelo} | top_k {s.top_k}")

    if len(sys.argv) > 1:
        mostrar(responder(" ".join(sys.argv[1:]), acervo, s=s))
        return
    while True:
        try:
            pergunta = input("\nPergunta (vazio sai): ").strip()
        except EOFError:
            break
        if not pergunta:
            break
        mostrar(responder(pergunta, acervo, s=s))


if __name__ == "__main__":
    main()
