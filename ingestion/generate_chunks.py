"""Divide as bulas processadas em trechos e calcula os embeddings.

Uso (da raiz):  python -m ingestion.generate_chunks [--sem-embeddings]
Entrada: data/processed/bulas/*.json
Saída:   data/processed/chunks.jsonl  e o cache de embeddings (data/cache/, fora do Git)

O tamanho do trecho vem de CHUNK_TOKENS e CHUNK_SOBREPOSICAO no .env. Embeddings já calculados
ficam no cache: rodar de novo só calcula o que mudou.
"""

import math
import re
import sys
import time
from pathlib import Path

from app.config import Settings, get_settings
from app.llm.embeddings import EmbeddingCache, embutir
from app.schemas.dataset import Chunk, Leaflet
from app.services.corpus import carregar_bulas, texto_para_embedding

# Medido em 2026-10-13 com qwen2.5:1.5b em texto de bula: 3,3 caracteres por token.
# É estimativa: serve para dimensionar o trecho, não para cobrar limite exato.
CHARS_POR_TOKEN = 3.3
_FRASE = re.compile(r"(?<=[.!?;])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ0-9•\-])")


def estimar_tokens(texto: str) -> int:
    return max(1, math.ceil(len(texto) / CHARS_POR_TOKEN))


def _unidades(texto: str, limite: int) -> list[tuple[int, str]]:
    """(páginas viradas até ali, frase). Frase maior que o trecho é cortada em palavras."""
    unidades: list[tuple[int, str]] = []
    for viradas, pagina in enumerate(texto.split("\f")):
        for paragrafo in pagina.split("\n"):
            for frase in _FRASE.split(paragrafo.strip()):
                if not frase:
                    continue
                if estimar_tokens(frase) <= limite:
                    unidades.append((viradas, frase))
                    continue
                pedaco: list[str] = []
                for palavra in frase.split():
                    if pedaco and estimar_tokens(" ".join([*pedaco, palavra])) > limite:
                        unidades.append((viradas, " ".join(pedaco)))
                        pedaco = []
                    pedaco.append(palavra)
                unidades.append((viradas, " ".join(pedaco)))
    return unidades


def _tokens(unidades: list[tuple[int, str]]) -> int:
    return estimar_tokens(" ".join(frase for _, frase in unidades))


def _fatiar(unidades: list[tuple[int, str]], limite: int, sobreposicao: int) -> list[list[tuple[int, str]]]:
    """Agrupa frases até `limite` tokens; cada trecho repete o fim do anterior (`sobreposicao`)."""
    trechos: list[list[tuple[int, str]]] = []
    atual: list[tuple[int, str]] = []
    novas = 0  # frases de `atual` que não vieram da sobreposição
    for unidade in unidades:
        if novas and _tokens([*atual, unidade]) > limite:
            trechos.append(atual)
            cauda: list[tuple[int, str]] = []
            for anterior in reversed(atual):
                if _tokens([anterior, *cauda]) > sobreposicao or _tokens([anterior, *cauda, unidade]) > limite:
                    break
                cauda.insert(0, anterior)
            atual, novas = cauda, 0
        atual.append(unidade)
        novas += 1
    if novas:
        trechos.append(atual)
    return trechos


def gerar_chunks(bula: Leaflet, chunk_tokens: int, sobreposicao: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    ordem_por_secao: dict[str, int] = {}  # duas seções podem ter a mesma chave ("outras")
    for secao in bula.secoes:
        for trecho in _fatiar(_unidades(secao.texto, chunk_tokens), chunk_tokens, sobreposicao):
            ordem = ordem_por_secao.get(secao.secao, 0)
            ordem_por_secao[secao.secao] = ordem + 1
            texto = " ".join(frase for _, frase in trecho)
            chunks.append(Chunk(
                chunk_id=f"{bula.id}__{secao.secao}__{ordem:03d}",
                bula_id=bula.id,
                medicamento=bula.medicamento,
                principio_ativo=bula.principio_ativo,
                secao=secao.secao,
                pagina=secao.pagina_inicio + trecho[0][0],
                ordem=ordem,
                texto=texto,
                n_tokens=estimar_tokens(texto),
            ))
    return chunks


def main(s: Settings | None = None, com_embeddings: bool = True) -> None:
    s = s or get_settings()
    bulas = carregar_bulas(s.bulas_dir)
    if not bulas:
        sys.exit(f"Nenhuma bula em {s.bulas_dir}. Rode antes: python -m ingestion.process_leaflet --help")
    chunks = [c for b in bulas for c in gerar_chunks(b, s.chunk_tokens, s.chunk_sobreposicao)]
    destino = Path(s.chunks_arquivo)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8", newline="\n") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")
    print(f"{len(bulas)} bula(s), {len(chunks)} trechos de até ~{s.chunk_tokens} tokens -> {destino}")
    for b in bulas:
        print(f"  {b.id}: {sum(c.bula_id == b.id for c in chunks)} trechos")
    if not com_embeddings:
        return

    inicio = time.perf_counter()

    def progresso(feitos: int, total: int) -> None:
        decorrido = time.perf_counter() - inicio
        print(f"  embeddings: {feitos}/{total} ({decorrido:.0f} s, faltam ~{decorrido / feitos * (total - feitos):.0f} s)", flush=True)

    cache = EmbeddingCache(s.cache_embeddings)
    embutir([texto_para_embedding(c) for c in chunks], cache, s, progresso)
    cache.fechar()
    print(f"Embeddings prontos ({s.embed_modelo}) em {s.cache_embeddings}")


if __name__ == "__main__":
    main(com_embeddings="--sem-embeddings" not in sys.argv)
