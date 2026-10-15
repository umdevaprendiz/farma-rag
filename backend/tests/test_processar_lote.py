import json
import re
from datetime import date
from pathlib import Path

import pytest

from ingestao.processar_lote import MANIFESTO, Manifesto, data_da_versao, processar, sha256

RAIZ = Path(__file__).resolve().parents[2]

# Um "PDF" (.txt, páginas separadas por \f) com duas bulas, como o do Tylalgin.
DUAS_BULAS = (
    "V.05_05/2022MODELO DE BULA PARA O PROFISSIONAL DE SAÚDE\n1. INDICAÇÕES\nDor e febre.\n"
    "Registro M.S. nº 1.5423.0008\fHistórico de Alteração para a Bula\ntabela\n"
    "V.04_05/2022MODELO DE BULA PARA O PROFISSIONAL DE SAÚDE\n1. INDICAÇÕES\nDor.\nRegistro 1.5423.0008"
)


@pytest.fixture
def lote(tmp_path):
    """(manifesto como dict, pasta de destino); o arquivo da bula fica em tmp_path/raw."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "tylalgin.txt").write_text(DUAS_BULAS, encoding="utf-8")
    item = {
        "arquivo": "tylalgin.txt", "sha256": sha256(raw / "tylalgin.txt"), "fabricante": "Geolab",
        "numero_registro": "1.5423.0008", "principio_ativo": "paracetamol", "nomes_comerciais": ["Tylalgin"],
        "partes": [
            {"medicamento": "Tylalgin 200 mg/mL gotas", "forma_farmaceutica": "solução oral gotas"},
            {"medicamento": "Tylalgin 750 mg", "forma_farmaceutica": "comprimido", "data_versao": "2022-05-20"},
        ],
    }
    return {"pasta": str(raw), "bulas": [item]}, tmp_path / "bulas"


def _rodar(manifesto: dict, destino: Path):
    return processar(Manifesto.model_validate(manifesto), destino)


def test_pdf_com_duas_bulas_vira_dois_json(lote):
    manifesto, destino = lote
    r = _rodar(manifesto, destino)
    assert r.erros == {}
    gotas, comprimido = r.gravadas
    assert (gotas.id, comprimido.id) == ("tylalgin-200-mg-ml-gotas-geolab", "tylalgin-750-mg-geolab")
    assert (gotas.forma_farmaceutica, comprimido.forma_farmaceutica) == ("solução oral gotas", "comprimido")
    assert gotas.numero_registro == comprimido.numero_registro == "1.5423.0008"
    assert (gotas.versao, gotas.data_versao) == ("V.05_05/2022", date(2022, 5, 1))  # data tirada do código
    assert comprimido.data_versao == date(2022, 5, 20)  # a do manifesto vale mais
    assert sorted(p.name for p in destino.glob("*.json")) == ["tylalgin-200-mg-ml-gotas-geolab.json", "tylalgin-750-mg-geolab.json"]


@pytest.mark.parametrize("mudanca, erro", [
    (lambda i: i.update(sha256="0" * 64), "SHA-256 diferente"),
    (lambda i: i["partes"].pop(), "o PDF tem 2 bula(s), o manifesto descreve 1"),
    (lambda i: i.update(numero_registro="1.5423.0099"), "registro 1.5423.0099 não está no texto"),
    (lambda i: i["partes"][0].update(versao="V.09_09/2099"), "versão V.09_09/2099 no manifesto, V.05_05/2022 no PDF"),
    (lambda i: i.update(arquivo="sumiu.txt"), "arquivo não encontrado"),
])
def test_pdf_que_nao_bate_com_o_manifesto_nao_grava_nada(lote, mudanca, erro):
    manifesto, destino = lote
    mudanca(manifesto["bulas"][0])
    r = _rodar(manifesto, destino)
    assert r.gravadas == [] and erro in next(iter(r.erros.values()))
    assert not destino.exists() or not list(destino.glob("*.json"))


def test_json_antigo_que_nao_veio_do_manifesto_gera_aviso(lote):
    manifesto, destino = lote
    destino.mkdir()
    (destino / "bula-velha.json").write_text("{}", encoding="utf-8")
    assert any("bula-velha" in a for a in _rodar(manifesto, destino).avisos)


def test_data_da_versao():
    assert data_da_versao("V.01_12/2025") == date(2025, 12, 1)
    assert data_da_versao("aprovada em 24/04/2025") is None and data_da_versao(None) is None


def test_manifesto_do_repositorio_bate_com_fontes_md():
    """Todo PDF do manifesto está em FONTES.md com o mesmo SHA-256 e registro (sem precisar dos PDFs)."""
    manifesto = Manifesto.model_validate(json.loads((RAIZ / MANIFESTO).read_text(encoding="utf-8")))
    fontes = (RAIZ / "data" / "FONTES.md").read_text(encoding="utf-8")
    linhas = {m[0]: m for m in re.findall(r"\| `([^`]+\.pdf)` \|[^|]+\| ([\d.]+) \|.*\| `([0-9a-f]{64})` \|", fontes)}
    for item in manifesto.bulas:
        arquivo, registro, hash_fontes = linhas[item.arquivo]
        assert (item.numero_registro, item.sha256) == (registro, hash_fontes), item.arquivo
    assert len({i.arquivo for i in manifesto.bulas}) == len(manifesto.bulas)
