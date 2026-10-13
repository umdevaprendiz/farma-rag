"""Contrato do dataset em JSON (fonte da verdade, versionado no Git)."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Secao = Literal[
    "composicao",
    "indicacoes",
    "resultados_eficacia",
    "caracteristicas_farmacologicas",
    "contraindicacoes",
    "advertencias",
    "interacoes",
    "armazenamento",
    "posologia",
    "reacoes_adversas",
    "superdose",
    "outras",
]

# Nome da seção como aparece para o modelo e para o usuário.
ROTULO_SECAO: dict[str, str] = {
    "composicao": "Composição",
    "indicacoes": "Indicações",
    "resultados_eficacia": "Resultados de eficácia",
    "caracteristicas_farmacologicas": "Características farmacológicas",
    "contraindicacoes": "Contraindicações",
    "advertencias": "Advertências e precauções",
    "interacoes": "Interações medicamentosas",
    "armazenamento": "Cuidados de armazenamento",
    "posologia": "Posologia e modo de usar",
    "reacoes_adversas": "Reações adversas",
    "superdose": "Superdose",
    "outras": "Outras informações",
}

Categoria = Literal[
    "interacao",
    "contraindicacao",
    "posologia",
    "conservacao",
    "reacao_adversa",
    "indicacao",
    "fora_do_escopo",
]


class SecaoBula(BaseModel):
    secao: Secao
    titulo_original: str
    pagina_inicio: int = Field(ge=1)
    pagina_fim: int = Field(ge=1)
    # Parágrafos separados por "\n"; "\f" marca a virada de página dentro da seção,
    # para que cada trecho saiba em que página começa.
    texto: str

    @model_validator(mode="after")
    def _paginas_coerentes(self):
        if self.pagina_fim < self.pagina_inicio:
            raise ValueError("pagina_fim menor que pagina_inicio")
        return self


class Bula(BaseModel):
    id: str
    medicamento: str
    principio_ativo: str
    # Nomes comerciais/sinônimos; usados na detecção do medicamento na pergunta.
    nomes_comerciais: list[str] = Field(default_factory=list)
    fabricante: str
    tipo_bula: Literal["profissional", "paciente"]
    data_versao: date
    fonte_pdf: str
    processado_em: datetime
    secoes: list[SecaoBula]


class Chunk(BaseModel):
    chunk_id: str
    bula_id: str
    medicamento: str
    # Denormalizado de propósito: permite casar âncoras do benchmark sem consultar a bula.
    principio_ativo: str
    secao: Secao
    pagina: int = Field(ge=1)
    ordem: int = Field(ge=0)
    texto: str
    n_tokens: int = Field(gt=0)

    @model_validator(mode="after")
    def _chunk_id_no_padrao(self):
        esperado = f"{self.bula_id}__{self.secao}__{self.ordem:03d}"
        if self.chunk_id != esperado:
            raise ValueError(f"chunk_id deveria ser {esperado!r}")
        return self


class Ancora(BaseModel):
    """Onde a resposta está, sem depender de como a bula foi dividida em trechos.

    `frase` é um pedaço curto (até ~12 palavras) copiado do texto da bula, de
    dentro de uma única frase, para não cair na fronteira entre dois trechos.
    """

    principio_ativo: str
    secao: Secao
    frase: str = Field(min_length=8)


class PerguntaBenchmark(BaseModel):
    id: str
    pergunta: str
    respondivel: bool
    # Uma âncora por medicamento necessário (perguntas de interação precisam de duas).
    ancoras: list[Ancora] = Field(default_factory=list)
    resposta_esperada: str | None = None
    categoria: Categoria

    @model_validator(mode="after")
    def _respondivel_coerente(self):
        if self.respondivel:
            if not self.ancoras or not self.resposta_esperada:
                raise ValueError("pergunta respondível precisa de ancoras e resposta_esperada")
        elif self.ancoras or self.resposta_esperada:
            raise ValueError("pergunta não respondível não pode ter ancoras nem resposta_esperada")
        return self
