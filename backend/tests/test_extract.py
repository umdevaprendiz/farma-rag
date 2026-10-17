"""Respostas copiadas da bula (services/extract.py), com texto como o da tabela do Omenax depois da
ingestão (ingestion/extract_pdf.py): cada linha começa com "Remédio, impacto clínico:"."""

from app.schemas.dataset import Chunk
from app.services.extract import frase_da_alternativa, frases_sobre, unidades


def _chunk(secao: str, texto: str, ordem: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"omx__{secao}__{ordem:03d}", bula_id="omx", medicamento="Omenax 20 mg", principio_ativo="omeprazol",
        secao=secao, pagina=1, ordem=ordem, texto=texto, n_tokens=50,
    )


TABELA = (
    "Varfarina, impacto clínico: Aumento do INR em pacientes recebendo omeprazol e varfarina. Pode levar a "
    "sangramento. Varfarina, intervenção: Monitorar o INR. Substratos de CYP2C19 (por exemplo, clopidogrel, "
    "citalopram) Clopidogrel, impacto clínico: Redução do metabólito ativo do clopidogrel. Clopidogrel, "
    "intervenção: Evitar o uso concomitante de omeprazol. Citalopram, impacto clínico: Aumento da exposição; "
    "não usar com varfarina sem orientação."
)


def test_linha_da_tabela_fica_inteira_e_o_cabecalho_de_grupo_separado():
    assert unidades(TABELA) == [
        "Varfarina, impacto clínico: Aumento do INR em pacientes recebendo omeprazol e varfarina. Pode levar a sangramento.",
        "Varfarina, intervenção: Monitorar o INR.",
        "Substratos de CYP2C19 (por exemplo, clopidogrel, citalopram)",
        "Clopidogrel, impacto clínico: Redução do metabólito ativo do clopidogrel.",
        "Clopidogrel, intervenção: Evitar o uso concomitante de omeprazol.",
        "Citalopram, impacto clínico: Aumento da exposição; não usar com varfarina sem orientação.",
    ]


def test_frases_sobre_o_outro_remedio():
    chunks = [
        _chunk("interacoes", TABELA),
        # Sobreposição: o trecho anterior termina com a linha cortada; fica só a inteira.
        _chunk("interacoes", "Varfarina, impacto clínico: Aumento do INR em pacientes recebendo omeprazol e varfarina.", 1),
        _chunk("advertencias", "Interação com o clopidogrel: evitar o uso concomitante de omeprazol com clopidogrel."),
        _chunk("posologia", "Tomar com clopidogrel não muda a dose."),  # seção que não é de interação
    ]
    assert [f for _, f in frases_sobre(chunks, "varfarina sodica")] == [
        "Varfarina, impacto clínico: Aumento do INR em pacientes recebendo omeprazol e varfarina. Pode levar a sangramento.",
        "Varfarina, intervenção: Monitorar o INR.",
        # A linha do citalopram cita a varfarina de passagem: é de outro remédio, fica fora.
    ]
    assert [f for _, f in frases_sobre(chunks, "clopidogrel")] == [
        "Clopidogrel, impacto clínico: Redução do metabólito ativo do clopidogrel.",
        "Clopidogrel, intervenção: Evitar o uso concomitante de omeprazol.",
        "Interação com o clopidogrel: evitar o uso concomitante de omeprazol com clopidogrel.",
    ]  # o cabeçalho de grupo, sem ponto final, não entra
    assert frases_sobre(chunks, "captopril") == []


def test_frase_da_alternativa():
    contra = _chunk("contraindicacoes", "Este medicamento é contraindicado na gravidez. Não use em caso de hipersensibilidade ao omeprazol.")
    indicacao = _chunk("indicacoes", "Cápsulas de 20 mg. Omenax é indicado para o tratamento da úlcera duodenal.")
    assert frase_da_alternativa(contra) == "Omenax 20 mg: Não use em caso de hipersensibilidade ao omeprazol."
    assert frase_da_alternativa(indicacao) == "Omenax 20 mg: Omenax é indicado para o tratamento da úlcera duodenal."
