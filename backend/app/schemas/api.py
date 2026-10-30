"""Modelos Pydantic dos endpoints /api/v1 e catálogo de erros."""

from datetime import date, datetime
from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from app.schemas.dataset import ROTULO_SECAO

MAX_MEDICAMENTOS_POR_PERGUNTA = 3
# Nome de remédio no filtro: o maior nome do catálogo tem bem menos que isto. Sem o limite, cada item podia
# ter o tamanho do corpo inteiro e ir para a busca de nomes.
MAX_NOME_NO_FILTRO = 200

# Aviso fixo de toda resposta `respondida` (docs/DECISOES.md, seção 33). Montado aqui, nunca pelo
# LLM: não passa pela fundamentação, não entra em `resposta` nem no histórico gravado e não custa
# tempo. Quem mostra a resposta (a tela, o benchmark.ask) lê este texto de `AnswerOut.aviso`.
AVISO_SAUDE = (
    "Informação da bula, para conferência. Não substitui a avaliação do farmacêutico nem a consulta médica."
)


class ErrorCode(str, Enum):
    MEDICAMENTO_NAO_ENCONTRADO = "MEDICAMENTO_NAO_ENCONTRADO"
    # bula_id que não está no acervo (o botão de uma pergunta de apresentação antiga, §41).
    BULA_NAO_ENCONTRADA = "BULA_NAO_ENCONTRADA"
    CONSULTA_NAO_ENCONTRADA = "CONSULTA_NAO_ENCONTRADA"
    ENTRADA_INVALIDA = "ENTRADA_INVALIDA"
    SAIDA_INVALIDA = "SAIDA_INVALIDA"
    CITACAO_INVALIDA = "CITACAO_INVALIDA"
    # Nenhuma rodada produziu resposta sustentada pelos trechos (fundamentação ou verificação).
    RESPOSTA_NAO_VERIFICADA = "RESPOSTA_NAO_VERIFICADA"
    OLLAMA_INDISPONIVEL = "OLLAMA_INDISPONIVEL"
    BANCO_INDISPONIVEL = "BANCO_INDISPONIVEL"
    TEMPO_ESGOTADO = "TEMPO_ESGOTADO"
    # Já há perguntas demais em andamento (PERGUNTAS_SIMULTANEAS): esperar e tentar de novo.
    SERVIDOR_OCUPADO = "SERVIDOR_OCUPADO"


HTTP_POR_CODIGO: dict[ErrorCode, int] = {
    ErrorCode.MEDICAMENTO_NAO_ENCONTRADO: 404,
    ErrorCode.BULA_NAO_ENCONTRADA: 404,
    ErrorCode.CONSULTA_NAO_ENCONTRADA: 404,
    ErrorCode.ENTRADA_INVALIDA: 422,
    ErrorCode.SAIDA_INVALIDA: 502,
    ErrorCode.CITACAO_INVALIDA: 502,
    ErrorCode.RESPOSTA_NAO_VERIFICADA: 502,
    ErrorCode.OLLAMA_INDISPONIVEL: 503,
    ErrorCode.BANCO_INDISPONIVEL: 503,
    ErrorCode.TEMPO_ESGOTADO: 504,
    ErrorCode.SERVIDOR_OCUPADO: 429,
}


class ErrorDetail(BaseModel):
    codigo: ErrorCode
    mensagem: str


class ErrorOut(BaseModel):
    erro: ErrorDetail
    # A consulta gravada no histórico (POST /perguntas que terminou em erro): a tela manda este id
    # como consulta_origem_id na pergunta seguinte, e a conversa continua depois do erro.
    consulta_id: UUID | None = None


class QuestionIn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    pergunta: str = Field(min_length=5, max_length=500)
    # Filtro por princípio ativo (não por bula): vários fabricantes têm bula do mesmo
    # princípio ativo. Vazio = a API tenta detectar o medicamento na própria pergunta.
    principios_ativos: list[str] = Field(
        default_factory=list, max_length=MAX_MEDICAMENTOS_POR_PERGUNTA
    )
    # A pergunta anterior da mesma conversa: a tela manda a última resposta. A pergunta nova pode
    # continuar dela ("e para criança?") ou responder a um esclarecimento ("dipirona", depois de "a
    # alergia é a qual remédio?"). A API continua sem estado: busca a anterior no histórico.
    consulta_origem_id: UUID | None = None
    # Uma bula só: a apresentação escolhida no botão da pergunta de volta (§41). Vale mais que
    # principios_ativos (o remédio é o da bula). Precisa existir no acervo: senão, 404 BULA_NAO_ENCONTRADA.
    bula_id: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("principios_ativos")
    @classmethod
    def _sem_vazios(cls, v: list[str]) -> list[str]:
        limpos = [p.strip() for p in v]
        if any(not p for p in limpos):
            raise ValueError("principio_ativo vazio")
        if any(len(p) > MAX_NOME_NO_FILTRO for p in limpos):
            raise ValueError(f"principio_ativo longo demais (no máximo {MAX_NOME_NO_FILTRO} caracteres)")
        return limpos


class SourceOut(BaseModel):
    posicao: int = Field(ge=1)  # o [n] da resposta
    chunk_id: str
    medicamento: str
    # Nulos só nas bulas de teste e em consultas gravadas antes destes campos.
    fabricante: str | None = None
    numero_registro: str | None = None
    secao: str
    pagina: int
    data_versao: date
    trecho: str
    score: float
    # Trecho enviado ao modelo e usado na resposta. Os não citados também aparecem: são o que
    # o sistema consultou.
    citado: bool
    # As frases do trecho que sustentam a resposta, como [início, fim] em `trecho` (contando
    # caracteres, como o Python). Vazio na fonte não citada e nas consultas gravadas antes do destaque.
    destaques: list[tuple[int, int]] = Field(default_factory=list)

    @field_validator("destaques", mode="before")
    @classmethod
    def _nulo_vira_vazio(cls, v: list | None) -> list:
        return v or []

    @computed_field
    @property
    def rotulo_secao(self) -> str:
        """A seção como o usuário lê ("Posologia e modo de usar"), sem o frontend repetir a tabela."""
        return ROTULO_SECAO.get(self.secao, self.secao)


class MarkOut(BaseModel):
    """F4 (docs/DECISOES.md, seção 37): uma marca em `resposta`, em posições como os destaques da fonte
    (caracteres, como o Python conta). A frase não muda: quem mostra monta o negrito e o parêntese."""

    inicio: int = Field(ge=0)
    fim: int = Field(ge=0)  # exclusivo
    # foco: o que foi perguntado (negrito); termo: termo técnico, com a palavra do dia a dia depois dele.
    tipo: Literal["foco", "termo"]
    explicacao: str | None = None  # só no termo: o que vai entre parênteses


class OptionOut(BaseModel):
    rotulo: str
    principios_ativos: list[str]
    # Na pergunta de apresentação (§41): a bula da opção, que a tela reenvia como bula_id.
    bula_id: str | None = None


class ClarificationOut(BaseModel):
    """O frontend mostra as opções e reenvia a pergunta com o filtro escolhido."""

    # opcoes vazio (remedio_da_alergia): a pessoa responde no chat, e a resposta vem com
    # consulta_origem_id desta consulta.
    # apresentacao (§41): o remédio tem mais de uma bula e a resposta muda de uma para outra.
    motivo: Literal["medicamento_nao_reconhecido", "remedio_da_alergia", "apresentacao"]
    pergunta: str
    opcoes: list[OptionOut]


class AnswerOut(BaseModel):
    consulta_id: UUID
    status: Literal["respondida", "nao_encontrada", "precisa_esclarecimento"]
    # Só preenchido quando status = precisa_esclarecimento.
    esclarecimento: ClarificationOut | None = None
    rodadas: int = 0
    # Só preenchido quando status = nao_encontrada. sem_outra_explicacao: o "não ajudou" não achou
    # outro trecho que fale disso além dos já citados. fora_do_escopo: preço ou animal (DECISOES §47); fora_da_bula:
    # chá, erva ou suco que as bulas do remédio não citam.
    motivo: Literal[
        "sem_contexto", "medicamento_nao_indexado", "sem_outra_explicacao", "fora_do_escopo", "fora_da_bula"
    ] | None = None
    # Pergunta de para que serve ("paracetamol serve para ansiedade?"): o veredito montado no código
    # pela seção de indicações, antes das frases da bula. Não é afirmação do modelo.
    conclusao: str | None = None
    # F4: a frase antes da resposta, montada no código pelo caminho da pergunta ("Pela bula de X:"). Nula
    # na de para que serve (a conclusão abre) e fora da respondida.
    abertura: str | None = None
    resposta: str | None
    # F4: no "para que serve", depois das frases copiadas da bula, a explicação em palavras simples escrita
    # pelo modelo, com [n] e conferida com os trechos. Nula se o modelo não explicou ou se reprovou.
    explicacao: str | None = None
    marcas: list[MarkOut] = Field(default_factory=list)
    # F4: perguntas seguintes sobre o mesmo remédio, para a tela oferecer como botões.
    sugestoes: list[str] = Field(default_factory=list)
    fontes: list[SourceOut]
    # Medicamentos citados na pergunta que não têm bula indexada (o frontend avisa).
    medicamentos_nao_indexados: list[str] = Field(default_factory=list)
    # Para que serve cada um deles, pela classe terapêutica no registro da ANVISA (dados abertos):
    # {"dipirona": "ANALGESICOS NAO NARCOTICOS"}. Sem acento, como no CSV.
    classes_terapeuticas: dict[str, str] = Field(default_factory=dict)
    principios_ativos: list[str] = Field(default_factory=list)  # os usados na busca
    continuacao: bool = False  # usou a pergunta anterior da conversa
    alergia_a: str | None = None  # a resposta traz alternativas a este remédio
    extrativa: bool = False  # a resposta são frases copiadas da bula, sem o modelo de linguagem
    # Resposta ao "não ajudou" (POST /perguntas/{id}/outra-explicacao), sem os trechos já citados.
    outra_explicacao: bool = False
    modelo: str
    latencia_ms: int

    @computed_field
    @property
    def aviso(self) -> str | None:
        """O aviso de saúde: só na resposta respondida (nas outras não há informação de bula a avisar)."""
        return AVISO_SAUDE if self.status == "respondida" else None


class LeafletSummary(BaseModel):
    id: str
    medicamento: str
    fabricante: str
    forma_farmaceutica: str | None = None
    data_versao: date


class MedicineOut(BaseModel):
    """Um item do seletor: um princípio ativo com as bulas indexadas dele."""

    principio_ativo: str
    bulas: list[LeafletSummary]


class InquirySummary(BaseModel):
    id: UUID
    pergunta: str
    status: Literal["respondida", "nao_encontrada", "precisa_esclarecimento", "erro"]
    # Outra explicação de uma resposta que não ajudou: a pergunta é a mesma da consulta_origem_id.
    outra_explicacao: bool = False
    criado_em: datetime


class HistoryOut(BaseModel):
    itens: list[InquirySummary]  # da mais recente para a mais antiga
    total: int
    pagina: int
    limite: int


class InquiryDetail(InquirySummary):
    """Uma consulta como foi respondida, com as fontes guardadas no momento (snapshot)."""

    principios_ativos_filtro: list[str]
    bula_id_filtro: str | None = None  # a bula escolhida no botão (§41), para refazer a pergunta igual
    conclusao: str | None = None
    # F4: como na resposta (AnswerOut). Nulos/vazios nas consultas gravadas antes de 2026-10-24.
    abertura: str | None = None
    resposta: str | None
    explicacao: str | None = None
    marcas: list[MarkOut] = Field(default_factory=list)
    sugestoes: list[str] = Field(default_factory=list)
    motivo: str | None
    codigo_erro: ErrorCode | None
    mensagem_erro: str | None
    rodadas: int
    modelo: str
    latencia_ms: int
    consulta_origem_id: UUID | None
    fontes: list[SourceOut]

    @field_validator("marcas", "sugestoes", mode="before")
    @classmethod
    def _nulo_vira_vazio(cls, v: list | None) -> list:
        return v or []

    @computed_field
    @property
    def aviso(self) -> str | None:
        """Calculado ao responder, não gravado: o texto do aviso pode mudar sem mexer no histórico."""
        return AVISO_SAUDE if self.status == "respondida" else None


class HealthOut(BaseModel):
    status: Literal["ok", "degradado"]
    banco: bool
    ollama: bool
    # Modelos do .env que o Ollama ainda não tem (ollama pull).
    modelos_faltando: list[str]
    bulas: int
    trechos: int
    # O que está fora do ar e por quê, para mostrar na tela.
    problemas: list[str]
