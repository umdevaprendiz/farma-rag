import pytest

from app.schemas.dataset import Chunk
from app.services.grounding import (
    AFIRMA,
    NEGA,
    afirma_o_que_o_trecho_proibe,
    Reference,
    checar,
    corrigir_citacoes,
    destaques,
    destaques_da_afirmacao,
    frases_com_posicao,
    nega_sem_o_trecho,
    problemas_da_afirmacao,
    sim_sem_permissao,
)
from app.services.intent import inferir_secoes, termos_do_assunto
from app.services.medicines import Catalog


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


# ---- Regra 5: negação (docs/DECISOES.md, seção 28) ---------------------------------------------

# Caso real (consulta b14b2d1a, 18/10): "tomei rocutan para dor no ouvido, fiz certo?" foi respondida
# com "Não, RoctoN não é indicado para dor no ouvido [...]" citando este trecho do diclofenaco, que diz
# o contrário. As regras 1 a 4 aprovavam: as palavras são as mesmas, e "RoctoN" não está no catálogo.
DICLOFENACO = _chunk(
    "Este medicamento está indicado para o tratamento de: Formas degenerativas e inflamatórias de reumatismo: "
    "artrite reumatoide; Crises agudas de gota; Como auxiliar no tratamento de processos infecciosos acompanhados "
    "de dor e inflamação de ouvido, nariz ou garganta, como por exemplo, faringoamigdalites, otites. De acordo "
    "com os princípios terapêuticos gerais, a doença de fundo deve ser tratada com a terapia básica "
    "adequadamente. Febre isolada não é uma indicação."
)


def test_negacao_que_o_trecho_nao_tem_reprova():
    resposta = (
        "Não, RoctoN não é indicado para dor no ouvido. É recomendado consultar um médico para a avaliação "
        "e tratamento adequado"
    )
    assert problemas_da_afirmacao(resposta, Reference.de_chunk(DICLOFENACO), 0.6) == []  # as regras 1-3 passam
    assert checar([(resposta, 1)], {1: DICLOFENACO}, 0.6) == [f"afirmação 1, trecho [1]: {NEGA}"]


def test_negacao_em_outra_frase_do_trecho_nao_vale():
    # "Febre isolada não é uma indicação" nega outra coisa: quem sustenta a afirmação é a 1ª frase.
    assert nega_sem_o_trecho("O medicamento não é indicado para inflamação de ouvido", DICLOFENACO)


@pytest.mark.parametrize("afirmacao, trecho", [
    ("Não use durante a gravidez", "Este medicamento é contraindicado durante a gravidez."),
    ("Não deve ser usado na gravidez", "Este medicamento é contra-indicado durante a gravidez."),  # grafia antiga
    ("Nunca use com álcool", "Evite o consumo de bebidas alcoólicas durante o tratamento."),
    ("Exemplol 50 mg: Não usar em crianças com menos de 20 kg.", "Não usar em crianças com menos de 20 kg."),  # copiada
    ("Febre isolada não é indicação para o uso", DICLOFENACO.texto),
    ("Não há interação com alimentos", "Interações com alimentos: não foram observadas."),
])
def test_negacao_que_o_trecho_tambem_tem_passa(afirmacao, trecho):
    assert not nega_sem_o_trecho(afirmacao, _chunk(trecho))


def test_afirmacao_sem_negacao_nao_passa_pela_regra_5():
    assert not nega_sem_o_trecho("É indicado como auxiliar no tratamento de otites", DICLOFENACO)


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


# ---- Citação corrigida no código (docs/DECISOES.md, seção 31) ----------------------------------

EFICACIA = _chunk(
    "Em um estudo com 400 pacientes, a exemplina reduziu a pressão arterial sistólica em 12 mmHg após oito semanas."
)
DOSE_MAXIMA = "A dose máxima é de 100 mg uma vez ao dia"


def _dipirona(texto: str) -> Chunk:
    return Chunk(
        chunk_id="d__indicacoes__000", bula_id="d", medicamento="Dorageo 500 mg", principio_ativo="dipirona monoidratada",
        secao="indicacoes", pagina=1, ordem=0, texto=texto, n_tokens=20,
    )


def test_citacao_trocada_vai_para_o_trecho_que_sustenta_a_frase():
    # Caso real (consulta 12f61084, 17/10): a frase certa do [3] citando o [1] custou uma rodada (~45 s).
    chunks = {1: EFICACIA, 2: ARMAZENAMENTO, 3: POSOLOGIA}
    corrigidas, mudancas = corrigir_citacoes([(DOSE_MAXIMA, 1)], chunks, 0.6)
    assert (corrigidas, mudancas) == ([(DOSE_MAXIMA, 3)], ["afirmação 1: trecho [1] -> [3]"])
    assert checar(corrigidas, chunks, 0.6) == []


def test_citacao_que_ja_sustenta_a_frase_nao_muda():
    # A frase também está no [1], mas o [2] que ela citou já a sustenta.
    chunks = {1: POSOLOGIA, 2: _chunk(POSOLOGIA.texto)}
    assert corrigir_citacoes([(DOSE_MAXIMA, 2)], chunks, 0.6) == ([(DOSE_MAXIMA, 2)], [])


def test_frase_repetida_sai():
    afirmacoes = [(DOSE_MAXIMA, 1), (f"{DOSE_MAXIMA.lower()}.", 3)]  # caixa e ponto final não contam
    assert corrigir_citacoes(afirmacoes, {1: EFICACIA, 3: POSOLOGIA}, 0.6) == (
        [(DOSE_MAXIMA, 3)], ["afirmação 1: trecho [1] -> [3]", "afirmação 2 repetida, saiu"]
    )


def test_frase_que_nenhum_trecho_sustenta_fica_como_veio():
    # A checagem reprova depois, e a rodada seguinte corrige: o código não inventa a citação.
    afirmacoes = [("A dose máxima é de 300 mg", 1)]
    assert corrigir_citacoes(afirmacoes, {1: EFICACIA, 2: POSOLOGIA}, 0.6) == (afirmacoes, [])


def test_nao_troca_para_a_bula_de_outro_remedio_que_a_frase_nao_nomeia():
    # Sem nome na frase, ela é sobre o remédio do trecho citado: não pode virar a bula da dipirona.
    catalogo = Catalog.montar([], ["dipirona"])
    afirmacoes = [("É indicado como analgésico e antitérmico", 1)]
    chunks = {1: EFICACIA, 2: _dipirona("Indicado como analgésico e antitérmico.")}
    assert corrigir_citacoes(afirmacoes, chunks, 0.6, catalogo) == (afirmacoes, [])


def test_troca_para_a_bula_do_remedio_que_a_frase_nomeia():
    catalogo = Catalog.montar([], ["dipirona"])
    frase = "A dipirona é indicada como analgésico e antitérmico"
    chunks = {1: EFICACIA, 2: _dipirona("A dipirona é indicada como analgésico e antitérmico.")}
    assert corrigir_citacoes([(frase, 1)], chunks, 0.6, catalogo) == ([(frase, 2)], ["afirmação 1: trecho [1] -> [2]"])


# ---- Conselho que o trecho não traz (DECISOES §42) --------------------------------------------

from app.services.grounding import sem_conselho, tirar_conselho, traz_conselho  # noqa: E402


@pytest.mark.parametrize(("texto", "limpo"), [
    ("Consulte um médico.", None),
    ("Em caso de dúvida, procure o farmacêutico.", None),
    ("Procure orientação médica antes de usar.", None),
    ("O ibuprofeno é contraindicado na dengue, consulte um médico.", "O ibuprofeno é contraindicado na dengue."),
    (
        "Não, o Roacutan não é indicado para dor no ouvido. É recomendado consultar um médico.",
        "Não, o Roacutan não é indicado para dor no ouvido.",
    ),
    ("Informe seu médico se estiver grávida; a bula contraindica na gravidez.", "A bula contraindica na gravidez."),
    # Sem verbo de conselho, o médico é só assunto: a fundamentação decide.
    ("O médico deve avaliar a função renal.", "O médico deve avaliar a função renal."),
    ("A dose inicial usual é de 50 mg uma vez ao dia.", "A dose inicial usual é de 50 mg uma vez ao dia."),
])
def test_tirar_conselho(texto, limpo):
    assert tirar_conselho(texto) == limpo


def test_conselho_que_o_trecho_traz_fica():
    sem_medico = _chunk("A dose máxima é de 100 mg.")
    com_medico = _chunk("Em caso de superdose, procure imediatamente atendimento médico.")
    cabecalho = _chunk("Bula para o profissional de saúde. A dose máxima é de 100 mg.")
    assert (traz_conselho(sem_medico), traz_conselho(com_medico), traz_conselho(cabecalho)) == (False, True, False)
    chunks = {1: sem_medico, 2: com_medico}
    afirmacoes = [("A dose máxima é de 100 mg.", 1), ("Procure um médico.", 1), ("Procure atendimento médico.", 2)]
    assert sem_conselho(afirmacoes, chunks) == (
        [("A dose máxima é de 100 mg.", 1), ("Procure atendimento médico.", 2)],
        ["afirmação 2: conselho que o trecho [1] não traz, saiu"],
    )


def test_medico_nao_casa_com_medicamento():
    # Caso "rocutan" (§28): "médico" passava pelo começo da palavra em qualquer trecho com "medicamento".
    ref = Reference.de_chunk(_chunk("Este medicamento é contraindicado durante a gravidez."))
    assert problemas_da_afirmacao("Consulte um médico", ref, 0.6)
    ref = Reference.de_chunk(_chunk("Em caso de superdose, procure atendimento médico."))
    assert not problemas_da_afirmacao("Procure atendimento médico", ref, 0.6)


# --- Regra 6: afirma o que o trecho proíbe (o caminho inverso da regra 5; docs/DECISOES.md, seção 45) -----

GRAVIDEZ = _chunk("Este medicamento é contraindicado durante a gravidez e a amamentação.")


@pytest.mark.parametrize("afirmacao", [
    "O medicamento pode ser usado na gravidez",
    "Pode ser utilizado durante a amamentação",
    "O uso na gravidez é permitido",
    "É seguro na gravidez",
    "Gestantes podem tomar o medicamento na gravidez",
])
def test_permissao_que_a_bula_proibe_e_pega(afirmacao):
    assert afirma_o_que_o_trecho_proibe(afirmacao, GRAVIDEZ)
    assert checar([(afirmacao, 1)], {1: GRAVIDEZ}, 0.6) == [f"afirmação 1, trecho [1]: {AFIRMA}"]


@pytest.mark.parametrize("afirmacao, trecho", [
    # A resposta também proíbe: é o que a bula diz.
    ("O medicamento é contraindicado na gravidez", "Este medicamento é contraindicado durante a gravidez."),
    ("Não pode ser usado na gravidez", "Este medicamento é contraindicado durante a gravidez."),
    # "Pode" como possibilidade ("pode causar") não é permissão.
    ("O medicamento pode causar sono", "Evite dirigir veículos. O medicamento pode causar sono."),
    ("Pode reduzir o efeito anti-hipertensivo", "A combinação deve ser evitada: pode reduzir o efeito anti-hipertensivo."),
    # A frase que sustenta a permissão é outra que não proíbe: o apoio é ela, não a vizinha.
    ("O comprimido pode ser usado com alimentos", "Evite usar com álcool. O comprimido pode ser usado com alimentos."),
    # Permissão com ressalva não é permissão pura.
    ("Pode ser usado na gravidez, mas evite no primeiro trimestre", "Evite o uso no primeiro trimestre da gravidez."),
    # A bula também permite.
    ("O medicamento pode ser usado na gravidez", "Este medicamento pode ser usado na gravidez, sob orientação médica."),
    # Sem nada em comum com o trecho e o trecho não é só proibição: a regra não decide.
    ("O gelo pode ser usado no tornozelo", "Dose inicial de 50 mg. Este medicamento é contraindicado na gravidez."),
    # A frase da bula usa a palavra, mas libera ("não é contraindicado").
    ("Os idosos podem usar o medicamento", "O medicamento não é contraindicado para idosos."),
    ("O medicamento pode ser usado em idosos", "Não há contraindicação para o uso em idosos."),
    # "evita" descreve o remédio; só "evite", "evitar" proíbem.
    ("O remédio pode ser usado para evitar coágulos", "Este medicamento evita a formação de coágulos no sangue."),
])
def test_nao_reprova_o_que_nao_e_permissao_contra_a_bula(afirmacao, trecho):
    assert not afirma_o_que_o_trecho_proibe(afirmacao, _chunk(trecho))


def test_a_regra_6_reprova_junto_das_outras_e_tem_aviso_proprio_no_grafo():
    from app.services import graph

    problemas = checar([("O medicamento pode ser usado na gravidez", 1)], {1: GRAVIDEZ}, 0.6)
    assert problemas and all(p.endswith(AFIRMA) for p in problemas)
    assert "proíbe" in graph.DICA_AFIRMACAO and graph.DICA_AFIRMACAO != graph.DICA_NEGACAO


# --- Seção 47: cautela, "é recomendado" e o "sim" sem permissão ---------------------------------------------

CAUTELA = _chunk(
    "O ácido acetilsalicílico deve ser utilizado com cautela em pacientes com história de úlceras gastrintestinais."
)
LACTACAO = _chunk(
    "O uso em lactantes não é recomendado a menos que, na avaliação do médico, os benefícios superarem os riscos."
)


@pytest.mark.parametrize("afirmacao, trecho", [
    ("O ácido acetilsalicílico pode ser usado na úlcera", CAUTELA),
    ("A amamentação é um uso recomendado durante o tratamento", LACTACAO),
])
def test_permissao_onde_a_bula_manda_ter_cautela_ou_nao_recomenda_e_pega(afirmacao, trecho):
    assert afirma_o_que_o_trecho_proibe(afirmacao, trecho)


@pytest.mark.parametrize("afirmacao, trecho", [
    # A afirmação repete a cautela da bula.
    ("O ácido acetilsalicílico pode ser usado com cautela na úlcera", CAUTELA),
    ("O ácido acetilsalicílico pode ser usado na úlcera, sob orientação médica", CAUTELA),
    # "É recomendado" que não é permissão contra proibição: a bula recomenda.
    ("É recomendado tomar o comprimido com água", _chunk("Recomenda-se tomar o comprimido com um copo de água.")),
])
def test_nao_reprova_a_permissao_que_repete_a_cautela_da_bula(afirmacao, trecho):
    assert not afirma_o_que_o_trecho_proibe(afirmacao, trecho)


@pytest.mark.parametrize("pergunta, afirmacao", [
    ("Pode usar metformina com insuficiência renal grave?", "Sim, é recomendado verificar a função renal antes do uso"),
    ("Mãe que amamenta pode tomar sertralina?", "Sim, a amamentação é um uso recomendado"),
    ("Posso dar o remédio para criança?", "Sim. O medicamento é eficaz em crianças"),
])
def test_sim_que_nao_diz_que_pode_e_pego(pergunta, afirmacao):
    assert sim_sem_permissao(pergunta, afirmacao)


@pytest.mark.parametrize("pergunta, afirmacao", [
    ("Pode usar com álcool?", "Sim, pode ser usado com álcool"),
    ("Pode tomar com álcool?", "Sim, o uso é permitido"),
    ("Pode tomar com álcool?", "Não é recomendado o uso com álcool"),  # não abre com "sim"
    ("Ela precisa parar de amamentar?", "Sim, ela precisa parar de amamentar temporariamente"),  # não é "pode?"
    ("Tylalgin sachet tem açúcar?", "Sim, contém açúcar"),
])
def test_sim_com_permissao_ou_fora_de_pergunta_de_permissao_passa(pergunta, afirmacao):
    assert not sim_sem_permissao(pergunta, afirmacao)
