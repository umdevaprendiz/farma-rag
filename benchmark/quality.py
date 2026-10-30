"""Qualidade da resposta no benchmark completo (docs/DECISOES.md, seção 47).

O "acerto" de benchmark/metrics.py diz se a frase-âncora chegou ao modelo. Ele não diz se a resposta está certa: em
2026-10-29, `pos06`, `con06` e `adv03` eram "acerto" e estavam erradas. Aqui a resposta do sistema é conferida
contra a resposta esperada do gabarito, sem LLM, em três pontos:

- polaridade: a esperada abre com "Não" e a do sistema com "Sim" (ou o contrário) é erro;
- números: todo número da esperada ("12,5 mg", "3 dias") precisa estar na resposta;
- termos: `deve_conter` (todos, cada um com alternativas separadas por "|") e `nao_deve_conter` (nenhum) do gabarito.

É uma conferência de erros grosseiros, não uma nota: resposta que passa aqui ainda pode estar incompleta, e a
leitura humana continua valendo. Reavalia um resultado já gravado, sem rodar o modelo:
  python -m benchmark.quality data/benchmark/resultados/completo_<data>_<rotulo>.json [--perguntas outro.jsonl ...]
"""

import argparse
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path

from app.schemas.dataset import BenchmarkQuestion
from app.services.medicines import normalizar

_ABRE = re.compile(r"^\W*(sim|nao)\b")
_NUMERO = re.compile(r"\d+(?:[.,]\d+)?")


def _numero(texto: str) -> str:
    """"4.000" e "4000" são o mesmo número; "12,5" e "12.5" também."""
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", texto):
        return texto.replace(".", "")
    return texto.replace(",", ".")


def _numeros(texto: str) -> set[str]:
    return {_numero(n) for n in _NUMERO.findall(texto)}


def polaridade(texto: str) -> str | None:
    """"sim", "nao" ou None (a frase não abre com nenhum dos dois)."""
    achado = _ABRE.match(normalizar(texto))
    return achado.group(1) if achado else None


def confere(resposta: str, pergunta: BenchmarkQuestion) -> dict:
    """Os três pontos para uma resposta dada.

    `confere`: polaridade e termos estão certos (a resposta não contradiz o gabarito nem foge do assunto).
    `completa`: além disso, traz todos os números da esperada. Resposta certa e curta ("Não, não precisa de
    geladeira") confere sem ser completa, por isso os dois são contados à parte."""
    texto = normalizar(resposta)
    esperada = pergunta.resposta_esperada or ""
    erros: list[str] = []
    esperado, dado = polaridade(esperada), polaridade(resposta)
    if esperado and dado and esperado != dado:
        erros.append(f"polaridade: esperava “{esperado}”, veio “{dado}”")
    for grupo in pergunta.deve_conter:
        if not any(normalizar(alternativa) in texto for alternativa in grupo.split("|")):
            erros.append(f"não traz: {grupo}")
    erros += [f"traz o que não devia: {t}" for t in pergunta.nao_deve_conter if normalizar(t) in texto]
    faltam = sorted(n for n in _numeros(normalizar(esperada)) if n not in _numeros(texto))
    problemas = erros + ([f"números que faltam: {', '.join(faltam)}"] if faltam else [])
    return {"confere": not erros, "completa": not problemas, "problemas": problemas}


def avaliar(linhas: Iterable[dict], perguntas: dict[str, BenchmarkQuestion]) -> dict[str, dict]:
    """Por id, só das respondíveis que o sistema respondeu."""
    return {
        l["id"]: confere(l["resposta"], perguntas[l["id"]])
        for l in linhas
        if l.get("respondivel") and l.get("status") == "respondida" and l.get("resposta") and l["id"] in perguntas
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Reavalia a qualidade das respostas de um resultado do modo completo.")
    ap.add_argument("resultado", type=Path)
    ap.add_argument("--perguntas", type=Path, nargs="+", default=[Path("data/benchmark/perguntas.jsonl"), Path("data/benchmark/perguntas_lote2.jsonl")])
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    perguntas = {
        q.id: q for arq in a.perguntas for q in (BenchmarkQuestion.model_validate_json(l) for l in arq.read_text(encoding="utf-8").splitlines() if l.strip())
    }
    linhas = json.loads(a.resultado.read_text(encoding="utf-8"))["perguntas"]
    notas = avaliar(linhas, perguntas)
    for id_, nota in notas.items():
        if not nota["completa"]:
            print(f"{id_:10} {'ERRO      ' if not nota['confere'] else 'incompleta'} {'; '.join(nota['problemas'])}")
    print(f"\nrespostas que conferem: {sum(n['confere'] for n in notas.values())}/{len(notas)}")
    print(f"respostas completas:    {sum(n['completa'] for n in notas.values())}/{len(notas)}")


if __name__ == "__main__":
    main()
