import pytest

from app.schemas.dataset import Chunk
from app.services.grounding import (
    Reference,
    checar,
    destaques,
    destaques_da_afirmacao,
    frases_com_posicao,
    problemas_da_afirmacao,
)
from app.services.intent import inferir_secoes, termos_do_assunto


def _chunk(texto: str) -> Chunk:
    return Chunk(
        chunk_id="b__posologia__000", bula_id="b", medicamento="Exemplol 50 mg", principio_ativo="maleato de exemplina",
        secao="posologia", pagina=1, ordem=0, texto=texto, n_tokens=20,
    )


REF = Reference.de_chunk(_chunk(
    "A dose inicial usual é de 50 mg uma vez ao dia. A dose máxima é de 100 mg. Usar 2,5 mL da "
    "suspensão. Conservar entre 15 °C e 30 °C. Reação comum (entre 1% e 10% dos pacientes): tontura. "
    "Este medicamento é contraindicado durante a gravidez."
))


@pytest.mark.parametrize("afirmacao", [
    "A dose máxima é de 100 mg.",
    "A dose inicial é de 50mg 1 vez ao dia",  # "uma" no trecho = 1; "50mg" junto
    "Use 2.5 ml da suspensão",  # vírgula x ponto, caixa da unidade
    "Deve ser conservado entre 15 °C e 30 ºC",
    "A tontura ocorre em 1% a 10% dos pacientes",
    "Exemplol é contraindicado na gravidez",  # nome do medicamento conta como termo do trecho
    "É contraindicada durante a gravidez",  # flexão
])
def test_afirmacao_sustentada(afirmacao):
    assert problemas_da_afirmacao(afirmacao, REF, 0.6) == []


@pytest.mark.parametrize("afirmacao, problema", [
    ("A dose máxima é de 200 mg.", "'200 mg' não está no trecho"),
    ("A dose máxima é de 100 mL.", "'100 ml' não está no trecho"),  # número certo, unidade errada
    ("Tomar a cada 8 horas.", "o número '8' não está no trecho"),
    ("Pode causar hepatite fulminante e insuficiência hepática", "termos fora do trecho"),
])
def test_afirmacao_sem_fonte(afirmacao, problema):
    assert any(problema in p for p in problemas_da_afirmacao(afirmacao, REF, 0.6))


def test_checar_aponta_afirmacao_e_trecho():
    problemas = checar([("A dose máxima é de 100 mg", 1), ("A dose máxima é de 300 mg", 2)],
                       {1: _chunk("A dose máxima é de 100 mg."), 2: _chunk("Tontura.")}, 0.6)
    assert problemas == ["afirmação 2, trecho [2]: '300 mg' não está no trecho",
                         "afirmação 2, trecho [2]: termos fora do trecho: dose, maxima",
                         "afirmação 2, trecho [2]: não diz nada do texto do trecho, só o nome do medicamento"]


@pytest.mark.parametrize("afirmacao", [
    "Exemplol 50 mg",  # resposta que é só o nome: o nome está no cabeçalho, não no texto
    "Maleato de exemplina é indicado",  # "indicado" não está no texto deste trecho
])
def test_afirmacao_so_com_o_nome_do_medicamento(afirmacao):
    assert "não diz nada do texto do trecho, só o nome do medicamento" in problemas_da_afirmacao(afirmacao, REF, 0.6)


def test_remedio_de_outra_bula():
    from app.services.medicines import Catalog

    catalogo = Catalog.montar([], ["dipirona", "ibuprofeno"])
    trecho = {1: _chunk("Não usar em crianças com menos de 20 kg. Não usar com ibuprofeno.")}
    # Fala de dipirona citando a bula do Exemplol, que não menciona dipirona: reprova.
    assert checar([("A dipirona não deve ser usada em crianças com menos de 20 kg", 1)], trecho, 0.6, catalogo) == [
        "afirmação 1, trecho [1]: fala de dipirona, mas o trecho é da bula de Exemplol 50 mg"
    ]
    # Ibuprofeno aparece no texto do trecho: pode.
    assert checar([("O Exemplol não deve ser usado com ibuprofeno", 1)], trecho, 0.6, catalogo) == []


@pytest.mark.parametrize("pergunta, secoes", [
    ("Losartana pode ser tomada junto com ibuprofeno?", ["interacoes"]),
    ("Qual a dose máxima de dipirona?", ["posologia"]),
    ("Pode usar na gravidez?", ["contraindicacoes", "advertencias"]),
    ("Precisa guardar na geladeira?", ["armazenamento"]),
    ("Tem lactose?", ["composicao"]),
    ("Paracetamol: superdose", ["superdose"]),  # "superdose" não dispara "dose"
    ("Exemplol tem registro?", []),
    ("Qual o melhor remédio pra dor de cabeça?", ["indicacoes"]),
    ("Estou tonto, qual o melhor medicamento?", ["indicacoes"]),
    ("Tenho alergia a um remédio, tem outro que eu possa tomar?", ["contraindicacoes", "indicacoes"]),
    ("Losartana causa dor de cabeça?", ["reacoes_adversas", "indicacoes"]),
])
def test_inferir_secoes(pergunta, secoes):
    assert inferir_secoes(pergunta) == secoes


@pytest.mark.parametrize("pergunta, termos", [
    ("Qual o melhor remédio pra dor de cabeça?", ["cabeca", "cefale", "enxaquec"]),  # "dor" sai
    ("Qual remédio para dor?", ["dor", "dores", "analges"]),
    ("Estou tonto, qual o melhor medicamento?", ["tontura", "vertig", "labirint"]),
    ("Qual o melhor remédio para pressão alta?", ["pressao", "hipertens", "hipotens"]),
    ("Dipirona pode ser dada para criança?", ["dipir", "crian"]),
    ("Qual o melhor remédio?", []),
])
def test_termos_do_assunto(pergunta, termos):
    assert termos_do_assunto(pergunta) == termos


def test_alergia_vale_como_hipersensibilidade():
    ref = Reference.de_chunk(_chunk("É contraindicado para pacientes com hipersensibilidade à exemplina."))
    assert problemas_da_afirmacao("É contraindicado em caso de alergia à exemplina", ref, 0.6) == []
    ref = Reference.de_chunk(_chunk("Não usar em caso de alergia à exemplina."))
    assert problemas_da_afirmacao("Não usar em caso de hipersensibilidade", ref, 0.6) == []


# ---- Destaque das frases que sustentam a resposta (docs/DECISOES.md, seção 26) -----------------

POSOLOGIA = _chunk(
    "Posologia A dose inicial usual é de 50 mg uma vez ao dia. A dose máxima é de 100 mg uma vez ao "
    "dia. Os comprimidos podem ser tomados com ou sem alimentos."
)
ARMAZENAMENTO = _chunk(
    "Conservar em temperatura ambiente (entre 15 °C e 30 °C), protegido da luz e da umidade. O prazo "
    "de validade é de 24 meses a partir da data de fabricação."
)
ADVERTENCIAS = _chunk(
    "Em pacientes com insuficiência renal, recomenda-se monitorar a creatinina e o potássio no sangue. "
    "Posologia em idosos: não é necessário ajuste."
)


def _destacado(afirmacao: str, chunk: Chunk) -> list[str]:
    return [chunk.texto[i:f] for i, f in destaques_da_afirmacao(afirmacao, chunk)]


def test_frases_com_posicao_no_texto_original():
    texto = "  Primeira frase. Segunda, com 2,5 mL.\nTerceira!  "
    assert [texto[i:f] for i, f in frases_com_posicao(texto)] == ["Primeira frase.", "Segunda, com 2,5 mL.", "Terceira!"]
    assert frases_com_posicao("   ") == []


def test_destaca_a_frase_da_parafrase():
    assert _destacado("A dose máxima do Exemplol é de 100 mg por dia", POSOLOGIA) == [
        "A dose máxima é de 100 mg uma vez ao dia."
    ]


def test_frase_copiada_da_bula_destaca_onde_ela_esta():
    # Resposta extrativa (services/extract.py): "<medicamento>: <frase da bula>".
    assert _destacado("Exemplol 50 mg: A dose inicial usual é de 50 mg uma vez ao dia", POSOLOGIA) == [
        "Posologia A dose inicial usual é de 50 mg uma vez ao dia."
    ]


def test_afirmacao_que_junta_duas_frases_destaca_as_duas():
    assert _destacado("Conservar entre 15 °C e 30 °C; a validade é de 24 meses", ARMAZENAMENTO) == [
        "Conservar em temperatura ambiente (entre 15 °C e 30 °C), protegido da luz e da umidade.",
        "O prazo de validade é de 24 meses a partir da data de fabricação.",
    ]


def test_uma_palavra_em_comum_nao_basta_para_destacar():
    assert _destacado("A tontura é comum em idosos", ADVERTENCIAS) == []


def test_destaques_de_varias_afirmacoes_sem_repetir_frase():
    afirmacoes = ["A dose máxima é de 100 mg por dia", "A dose máxima do Exemplol é 100 mg"]
    assert destaques(afirmacoes, POSOLOGIA) == destaques_da_afirmacao(afirmacoes[0], POSOLOGIA)
    assert len(destaques(afirmacoes, POSOLOGIA)) == 1


def test_trecho_vazio_nao_destaca_nada():
    assert destaques_da_afirmacao("A dose máxima é de 100 mg por dia", _chunk("")) == []
