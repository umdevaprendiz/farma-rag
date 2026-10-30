import pytest

from app.services import scope


@pytest.mark.parametrize("pergunta, esperado", [
    ("Quanto custa o Cozaar 50 mg com 30 comprimidos?", scope.PRECO),
    ("Qual o preço da losartana?", scope.PRECO),
    ("Onde comprar dipirona mais barata?", scope.PRECO),
    ("A losartana tem desconto na Farmácia Popular?", scope.PRECO),
    ("Qual a dose de amoxicilina para um cachorro de 10 kg?", scope.ANIMAL),
    ("Posso dar dipirona para o meu gato?", scope.ANIMAL),
    ("Isso serve para uso veterinário?", scope.ANIMAL),
])
def test_pergunta_de_preco_ou_animal_e_fora_do_escopo(pergunta, esperado):
    assert scope.fora_do_escopo(pergunta) == esperado


@pytest.mark.parametrize("pergunta", [
    "Qual a dose máxima de paracetamol?",
    "Dipirona pode causar agranulocitose?",
    "Os estudos em animais mostraram risco na gravidez?",  # "animais" sozinho é assunto de bula
    "Qual o valor desejado de 25(OH)D para parar o tratamento?",
    "Delumi 50.000 UI: de quanto em quanto tempo toma?",
    "A metformina pode dar gosto metálico?",
    "Posso tomar o remédio com o suco?",
])
def test_pergunta_de_bula_nao_e_barrada(pergunta):
    assert scope.fora_do_escopo(pergunta) is None


@pytest.mark.parametrize("pergunta, esperado", [
    ("Omeprazol pode ser tomado junto com chá de boldo?", "boldo"),
    ("Posso tomar a sertralina com erva-doce?", "doce"),
    ("Losartana com suco de toranja pode?", "toranja"),
    ("Pode tomar com chá verde?", "verde"),
])
def test_substancia_fora_do_remedio(pergunta, esperado):
    assert scope.substancia_fora_do_remedio(pergunta) == esperado


@pytest.mark.parametrize("pergunta", [
    "Qual a dose de omeprazol?",
    "Posso tomar com alimentos?",
    "Tem algum chá para isso?",  # "algum" não é o nome
    "Que erva medicinal ajuda?",
])
def test_pergunta_sem_substancia_nao_e_olhada(pergunta):
    assert scope.substancia_fora_do_remedio(pergunta) is None


def test_citada_em_usa_o_comeco_da_palavra():
    assert scope.citada_em("toranja", ["nao tome com suco de toranjas"])
    assert not scope.citada_em("boldo", ["pode reduzir o efeito anti-hipertensivo"])
