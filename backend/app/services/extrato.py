"""Respostas copiadas da bula, sem LLM (docs/DECISOES.md, seção 24).

Quando a resposta certa é uma frase que a bula já escreve, o modelo de 1,5B só atrapalha: ele junta
linhas vizinhas ("omeprazol deve ser evitado com clopidogrel e varfarina", quando para a varfarina a
bula manda monitorar o INR) ou recusa sem motivo. Dois casos usam a frase da bula como ela está:
- interação entre remédios citados: as frases da bula de um que falam do outro (na tabela de
  interações, a linha inteira: "Varfarina, intervenção: Monitorar o INR...");
- alternativas para quem tem alergia: a frase de hipersensibilidade da bula do remédio e a frase de
  indicação de cada alternativa.
A resposta passa pela mesma validação; como o texto é o do trecho, a fundamentação confere.
"""

import re

from app.schemas.dataset import Chunk
from app.services.medicamentos import formas_do_nome, normalizar

# Linha da tabela de interações, como ingestao/extrair_pdf.py escreve: "Varfarina, impacto clínico: ..."
_ROTULO = r"[A-ZÀ-Ú][^.:,]{1,70}, (?:impacto cl[ií]nico|interven[cç][aã]o)(?: \(continua[cç][aã]o\))?:"
_COMECO_DE_LINHA = re.compile(rf"(?={_ROTULO})")
_LINHA = re.compile(rf"^({_ROTULO[:-1]})")
_FIM_DE_FRASE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ú0-9])")
# Onde procurar o que a bula de um remédio diz de outro, na ordem de preferência.
_SECOES_DA_INTERACAO = ("interacoes", "advertencias", "contraindicacoes")
# A pergunta é sobre usar um com o outro.
INTERACAO = re.compile(r"\b(?:com|junto|juntos|interag\w*|interac\w*|associ\w*|mistur\w*|concomit\w*)\b")


def unidades(texto: str) -> list[str]:
    """As frases do trecho; a linha da tabela de interações fica inteira (pode ter várias frases)."""
    saida: list[str] = []
    for pedaco in _COMECO_DE_LINHA.split(texto):
        frases = [f.strip() for f in _FIM_DE_FRASE.split(pedaco.strip()) if f.strip()]
        if not frases:
            continue
        if not _LINHA.match(frases[0]):
            saida += frases
            continue
        # A linha vai até a última frase com ponto. O que sobra sem ponto é o cabeçalho de grupo
        # que vinha na linha seguinte da tabela ("Substratos de CYP2C19 (...)") e fica separado.
        fim = max((i for i, f in enumerate(frases) if f.endswith((".", "!", "?", ";"))), default=len(frases) - 1)
        saida.append(" ".join(frases[: fim + 1]))
        saida += frases[fim + 1 :]
    return saida


def _cita(texto_normalizado: str, formas: set[str]) -> bool:
    return any(re.search(rf"\b{re.escape(f)}\b", texto_normalizado) for f in formas)


def frases_sobre(chunks: list[Chunk], nome: str, limite: int = 3) -> list[tuple[Chunk, str]]:
    """O que a bula (os `chunks` de um remédio) diz do remédio `nome`: linhas da tabela de interações
    dele primeiro, depois frases de interações, advertências e contraindicações que o citam.
    Linha da tabela de outro remédio que só o cita de passagem fica fora; frase sem ponto final
    também (é o cabeçalho de grupo da tabela: "Substratos de CYP2C19 (por exemplo, clopidogrel...)")."""
    formas = formas_do_nome(nome)
    achadas: dict[str, tuple[tuple[int, int, str], Chunk, str]] = {}  # texto normalizado -> achado
    for chunk in sorted(chunks, key=lambda c: c.chunk_id):
        if chunk.secao not in _SECOES_DA_INTERACAO:
            continue
        for frase in unidades(chunk.texto):
            chave = normalizar(frase)
            if chave in achadas or not _cita(chave, formas):
                continue
            rotulo = _LINHA.match(frase)
            da_linha = bool(rotulo) and _cita(normalizar(rotulo.group(1).split(",")[0]), formas)
            if (rotulo and not da_linha) or (not rotulo and not frase.endswith((".", "!", "?", ";"))):
                continue
            ordem = (_SECOES_DA_INTERACAO.index(chunk.secao), 0 if da_linha else 1, chunk.chunk_id)
            achadas[chave] = (ordem, chunk, frase)
    # Os trechos se sobrepõem: a linha que termina cortada num trecho aparece inteira no seguinte.
    # Fica só a inteira.
    inteiras = [a for chave, a in achadas.items() if not any(o != chave and o.startswith(chave) for o in achadas)]
    return [(chunk, frase) for _, chunk, frase in sorted(inteiras, key=lambda a: a[0])[:limite]]


# Pergunta de dose ("qual a dose máxima de dipirona?"): a bula responde com número e unidade.
PERGUNTA_DE_DOSE = re.compile(r"\b(?:dose|doses|dosagem|posologia|quantos|quantas|quanto|maxima|maximo)\b")
_UNIDADE = re.compile(r"\d.*\b(?:comprimidos?|gotas?|mg|ml|capsulas?|saches?|vezes|ui|envelopes?)\b")
_PEDIATRICO = re.compile(r"\b(?:crianc\w*|pediatr\w*|lactentes?|bebes?|meses)\b")


def frases_de_dose(chunks: list[Chunk], termos: list[str], limite: int = 3) -> list[tuple[Chunk, str]]:
    """Frases da posologia com número e unidade de dose, das que mais falam do que a pergunta pede
    (`termos`: "maxim", "inicial", "crian"...), no máximo uma por bula: "paracetamol, dose máxima?"
    traz uma frase de cada apresentação (gotas, comprimido, sachê). Criança só entra se a pergunta é
    de criança. Nenhuma frase que fale do pedido = None no grafo, e segue pelo LLM."""
    pede_crianca = any(t.startswith(("crian", "pediat", "bebe", "lactent")) for t in termos)
    por_bula: dict[str, tuple[int, Chunk, str]] = {}
    for chunk in sorted(chunks, key=lambda c: c.chunk_id):
        if chunk.secao != "posologia":
            continue
        for frase in unidades(chunk.texto):
            texto = normalizar(frase)
            if not _UNIDADE.search(texto) or bool(_PEDIATRICO.search(texto)) != pede_crianca:
                continue
            pontos = sum(2 for t in termos if re.search(rf"\b{re.escape(t)}", texto)) + ("adult" in texto)
            if pontos and pontos > por_bula.get(chunk.bula_id, (0,))[0]:
                por_bula[chunk.bula_id] = (pontos, chunk, frase)
    melhores = sorted(por_bula.values(), key=lambda a: (-a[0], a[1].chunk_id))[:limite]
    return [(chunk, f"{chunk.medicamento}: {frase}") for _, chunk, frase in melhores]


def _primeira_com(chunk: Chunk, comecos: tuple[str, ...]) -> str:
    frases = unidades(chunk.texto)
    return next((f for f in frases if any(re.search(rf"\b{c}", normalizar(f)) for c in comecos)), frases[0])


def frase_da_alternativa(chunk: Chunk) -> str:
    """A afirmação de um trecho do modo alergia: a contraindicação por hipersensibilidade (bula do
    remédio da alergia) ou a indicação (bula da alternativa), com o nome do medicamento na frente."""
    if chunk.secao == "contraindicacoes":
        frase = _primeira_com(chunk, ("hipersensib", "alergi", "alergic"))
    else:
        frase = _primeira_com(chunk, ("indica", "destina", "tratamento", "alivio", "prevenc"))
    return f"{chunk.medicamento}: {frase}"
