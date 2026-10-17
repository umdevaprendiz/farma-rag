from ingestion.validate_leaflet_pdf import validar


def _pdf(catalogo: str = "", extras: tuple[str, ...] = (), solto: str = "") -> bytes:
    """PDF mínimo de uma página. `extras` são os objetos 4, 5...; `solto` fica fora da tabela xref."""
    objetos = [
        f"<< /Type /Catalog /Pages 2 0 R {catalogo} >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>",
        *extras,
    ]
    corpo = b"%PDF-1.4\n"
    posicoes = []
    for n, objeto in enumerate(objetos, start=1):
        posicoes.append(len(corpo))
        corpo += f"{n} 0 obj\n{objeto}\nendobj\n".encode()
    if solto:
        corpo += f"99 0 obj\n{solto}\nendobj\n".encode()
    xref = len(corpo)
    corpo += f"xref\n0 {len(objetos) + 1}\n0000000000 65535 f \n".encode()
    corpo += b"".join(f"{p:010d} 00000 n \n".encode() for p in posicoes)
    corpo += f"trailer\n<< /Size {len(objetos) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return corpo


def _validar(tmp_path, conteudo: bytes, bytes_esperados: int | None = None):
    arquivo = tmp_path / "bula.pdf"
    arquivo.write_bytes(conteudo)
    return validar(arquivo, bytes_esperados)


def test_pdf_limpo_passa(tmp_path):
    rel, problemas = _validar(tmp_path, _pdf())
    assert problemas == []
    assert rel["paginas"] == 1 and rel["conteudo_ativo"] == []


def test_link_e_salto_de_pagina_nao_sao_problema(tmp_path):
    rel, problemas = _validar(tmp_path, _pdf(
        catalogo="/OpenAction [3 0 R /Fit]",
        extras=("<< /Type /Action /S /URI /URI (https://www.gov.br/anvisa) >>",),
    ))
    assert problemas == []
    assert rel["links (informativo)"] == ["www.gov.br"]


def test_javascript_ao_abrir_e_barrado(tmp_path):
    _, problemas = _validar(tmp_path, _pdf(
        catalogo="/OpenAction 4 0 R", extras=("<< /S /JavaScript /JS (app.alert(1)) >>",)
    ))
    assert len(problemas) == 1 and "/JS" in problemas[0] and "ação /JavaScript" in problemas[0]


def test_nome_escapado_em_hexadecimal_e_barrado(tmp_path):
    _, problemas = _validar(tmp_path, _pdf(extras=("<< /S /J#61vaScript /J#53 (x) >>",)))
    assert "ação /JavaScript" in problemas[0]


def test_arquivo_embutido_e_comando_sao_barrados(tmp_path):
    _, problemas = _validar(tmp_path, _pdf(
        catalogo="/Names << /EmbeddedFiles 4 0 R >>",
        extras=("<< /Names [] >>", "<< /S /Launch /F (cmd.exe) >>"),
    ))
    assert "/EmbeddedFiles" in problemas[0] and "ação /Launch" in problemas[0]


def test_objeto_fora_da_tabela_e_visto_nos_bytes(tmp_path):
    rel, problemas = _validar(tmp_path, _pdf(solto="<< /S /JavaScript /JS (x) >>"))
    assert rel["conteudo_ativo"] == [] and rel["nomes_ativos_nos_bytes"] == ["/JS", "/JavaScript"]
    assert "conteúdo ativo" in problemas[0]


def test_tamanho_diferente_do_anunciado(tmp_path):
    conteudo = _pdf()
    assert _validar(tmp_path, conteudo, len(conteudo))[1] == []
    assert "tamanho diferente" in _validar(tmp_path, conteudo, len(conteudo) + 1)[1][0]


def test_executavel_renomeado_e_pdf_cifrado_ou_truncado(tmp_path):
    assert "não começa com %PDF-" in _validar(tmp_path, b"MZ\x90\x00" + b"\x00" * 64)[1][-1]
    assert any("cifrado" in p for p in _validar(tmp_path, _pdf(extras=("<< /Encrypt 5 0 R >>",)))[1])
    assert any("%%EOF" in p for p in _validar(tmp_path, _pdf() + b"\x00" * 2000)[1])
