from datetime import date, datetime

from app.schemas.dataset import Leaflet, LeafletSection
from app.services.corpus import texto_para_embedding
from ingestion.generate_chunks import estimar_tokens, gerar_chunks


def _bula(*secoes):
    return Leaflet(
        id="exemplol", medicamento="Exemplol 50 mg", principio_ativo="exemplina", fabricante="F",
        tipo_bula="profissional", data_versao=date(2025, 3, 15), fonte_pdf="x.pdf",
        processado_em=datetime(2026, 10, 13, 12, 0), secoes=list(secoes),
    )


def _secao(chave, texto, pagina=1, fim=None):
    return LeafletSection(secao=chave, titulo_original=chave.upper(), pagina_inicio=pagina, pagina_fim=fim or pagina, texto=texto)


FRASES = [f"Esta é a frase número {i} do texto, com tamanho parecido com as outras." for i in range(12)]


def test_secao_curta_vira_um_trecho_com_id_no_padrao():
    (c,) = gerar_chunks(_bula(_secao("interacoes", "Pode reduzir o efeito.", pagina=4)), 300, 50)
    assert c.chunk_id == "exemplol__interacoes__000"
    assert (c.pagina, c.ordem, c.texto) == (4, 0, "Pode reduzir o efeito.")
    assert c.principio_ativo == "exemplina"


def test_secao_longa_respeita_o_limite_e_repete_o_fim_do_trecho_anterior():
    chunks = gerar_chunks(_bula(_secao("posologia", " ".join(FRASES))), 80, 25)
    assert len(chunks) > 2
    assert all(c.n_tokens <= 80 for c in chunks)
    assert [c.ordem for c in chunks] == list(range(len(chunks)))
    for anterior, seguinte in zip(chunks, chunks[1:]):
        repetida = next(f for f in FRASES if seguinte.texto.startswith(f))
        assert anterior.texto.endswith(repetida)
    # nada se perde: toda frase aparece em algum trecho
    assert all(any(f in c.texto for c in chunks) for f in FRASES)


def test_sem_sobreposicao_nenhuma_frase_se_repete():
    chunks = gerar_chunks(_bula(_secao("posologia", " ".join(FRASES))), 80, 0)
    assert sum(c.texto.count("Esta é a frase") for c in chunks) == len(FRASES)


def test_pagina_do_trecho_acompanha_a_virada_de_pagina():
    texto = f"{FRASES[0]} {FRASES[1]}\f{FRASES[2]} {FRASES[3]}\f{FRASES[4]}"
    chunks = gerar_chunks(_bula(_secao("advertencias", texto, pagina=7, fim=9)), 30, 0)
    assert [c.pagina for c in chunks] == [7, 7, 8, 8, 9]
    assert all("\f" not in c.texto for c in chunks)


def test_frase_gigante_sem_pontuacao_e_cortada():
    chunks = gerar_chunks(_bula(_secao("composicao", " ".join(["excipiente"] * 200))), 50, 0)
    assert len(chunks) > 5 and all(c.n_tokens <= 50 for c in chunks)


def test_duas_secoes_com_a_mesma_chave_nao_repetem_chunk_id():
    chunks = gerar_chunks(_bula(_secao("outras", "Apresentações."), _secao("outras", "Dizeres legais.")), 300, 50)
    assert [c.chunk_id for c in chunks] == ["exemplol__outras__000", "exemplol__outras__001"]


def test_texto_para_embedding_leva_medicamento_e_secao():
    (c,) = gerar_chunks(_bula(_secao("interacoes", "Pode reduzir o efeito.")), 300, 50)
    assert texto_para_embedding(c) == "Exemplol 50 mg (exemplina) | Interações medicamentosas\nPode reduzir o efeito."


def test_estimar_tokens():
    assert estimar_tokens("") == 1
    assert estimar_tokens("a" * 330) == 100
