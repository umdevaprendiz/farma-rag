"""Teste manual do agente no terminal.

Uso (da raiz):
  python -m benchmark.ask "Losartana pode com ibuprofeno?"
  python -m benchmark.ask            (modo conversa: uma pergunta por linha, vazio sai)

Mostra a resposta, os trechos usados (com a nota de semelhança), as rodadas e os tempos. Quando
nenhuma rodada passa na verificação, mostra o erro e a saída crua da última. No modo conversa,
um pedido de esclarecimento deixa escolher o medicamento e refaz a pergunta com ele.
"""

import sys

from app.config import get_settings
from app.schemas.api import AVISO_SAUDE
from app.schemas.dataset import ROTULO_SECAO
from app.services.corpus import Corpus, IncompleteCorpusError
from app.services.presentation import Mark
from app.services.rag import PreviousQuestion, Result, responder

MAX_TRECHO = 300


def com_marcas(texto: str, marcas: list[Mark]) -> str:
    """A resposta como a tela mostra, em texto: *negrito* no que foi perguntado e a palavra do dia a dia
    entre parênteses depois do termo técnico (F4)."""
    insercoes: list[tuple[int, int, str]] = []  # (posição, ordem, texto): fecha antes de abrir
    for m in marcas:
        if m.tipo == "foco":
            insercoes += [(m.inicio, 1, "*"), (m.fim, 0, "*")]
        else:
            insercoes.append((m.fim, 0, f" ({m.explicacao})"))
    for pos, _, extra in sorted(insercoes, reverse=True):
        texto = texto[:pos] + extra + texto[pos:]
    return texto


def mostrar(r: Result) -> None:
    if r.principios_ativos:
        print(f"Medicamento(s) reconhecido(s): {', '.join(r.principios_ativos)}")
    if r.medicamentos_nao_indexados:
        print(f"Sem bula indexada: {', '.join(r.medicamentos_nao_indexados)}")

    if r.status == "respondida":
        print("\nRESPOSTA")
        for linha in (r.abertura, r.conclusao):
            if linha:
                print(linha)
        print(com_marcas(r.resposta, r.marcas))
        if r.explicacao:
            print(f"Em outras palavras: {r.explicacao}")
        print(f"\n{AVISO_SAUDE}")
        if r.sugestoes:
            print(f"Quer saber mais? {' | '.join(r.sugestoes)}")
    elif r.status == "precisa_esclarecimento":
        print(f"\nPRECISO SABER: {r.esclarecimento.pergunta}")
        for i, opcao in enumerate(r.esclarecimento.opcoes, start=1):
            print(f"  {i}. {opcao.rotulo}")
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
    if len(r.tentativas) > 1 or any(t.falha or t.correcoes or t.explicacao_descartada for t in r.tentativas):
        print(f"\nRODADAS ({r.rodadas})")
        for t in r.tentativas:
            print(f"  {t.rodada}. {t.resultado}" + (f": {t.falha}" if t.falha else ""))
            if t.correcoes:
                print(f"     citação corrigida no código: {'; '.join(t.correcoes)}")
            if t.explicacao_descartada:
                print(f"     explicação descartada: {'; '.join(t.explicacao_descartada)}")
    if r.tempos_s:
        tempos = ", ".join(f"{nome} {valor:.1f} s" for nome, valor in r.tempos_s.items())
        print(f"\nTempo: {tempos}" + (f" | prompt de {r.prompt_tokens} tokens" if r.prompt_tokens else ""))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # o terminal do Windows nem sempre está em UTF-8
    s = get_settings()
    try:
        acervo = Corpus.carregar(s)
    except IncompleteCorpusError as e:
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
        r = responder(pergunta, acervo, s=s)
        mostrar(r)
        if r.status == "precisa_esclarecimento":
            # Como o frontend vai fazer: reenvia a mesma pergunta com o filtro escolhido.
            escolha = input("\nNúmero da opção (vazio pula): ").strip()
            if escolha.isdigit() and 1 <= int(escolha) <= len(r.esclarecimento.opcoes):
                opcao = r.esclarecimento.opcoes[int(escolha) - 1]
                # Na pergunta de apresentação (DECISOES §41), a opção é uma bula.
                anterior = PreviousQuestion(pergunta, r.principios_ativos, pediu_esclarecimento=True)
                mostrar(responder(pergunta, acervo, opcao.principios_ativos, s, anterior, bula_id=opcao.bula_id))


if __name__ == "__main__":
    main()
