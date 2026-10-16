import pytest

from app.schemas.dataset import Chunk
from app.services.fundamentacao import Referencia, checar, problemas_da_afirmacao
from app.services.intencao import inferir_secoes, termos_do_assunto


def _chunk(texto: str) -> Chunk:
    return Chunk(
        chunk_id="b__posologia__000", bula_id="b", medicamento="Exemplol 50 mg", principio_ativo="maleato de exemplina",
        secao="posologia", pagina=1, ordem=0, texto=texto, n_tokens=20,
    )


REF = Referencia.de_chunk(_chunk(
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
    from app.services.medicamentos import Catalogo

    catalogo = Catalogo.montar([], ["dipirona", "ibuprofeno"])
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
    ref = Referencia.de_chunk(_chunk("É contraindicado para pacientes com hipersensibilidade à exemplina."))
    assert problemas_da_afirmacao("É contraindicado em caso de alergia à exemplina", ref, 0.6) == []
    ref = Referencia.de_chunk(_chunk("Não usar em caso de alergia à exemplina."))
    assert problemas_da_afirmacao("Não usar em caso de hipersensibilidade", ref, 0.6) == []
