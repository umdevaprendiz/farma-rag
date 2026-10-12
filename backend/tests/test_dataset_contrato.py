import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas.dataset import Bula, Chunk, PerguntaBenchmark

RAIZ = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures"


def _chunk(**kw):
    base = dict(
        chunk_id="b1__interacoes__003", bula_id="b1", medicamento="M", principio_ativo="m",
        secao="interacoes", pagina=4, ordem=3, texto="t", n_tokens=10,
    )
    return Chunk(**{**base, **kw})


def test_fixture_de_bula_segue_o_contrato():
    Bula.model_validate_json((FIXTURES / "bula_exemplo.json").read_text(encoding="utf-8"))


def test_chunk_id_precisa_seguir_o_padrao():
    _chunk()
    with pytest.raises(ValidationError):
        _chunk(chunk_id="b1__interacoes__3")
    with pytest.raises(ValidationError):
        _chunk(secao="posologia")  # id diz interacoes


def test_secao_fora_das_chaves_fixas_e_rejeitada():
    with pytest.raises(ValidationError):
        _chunk(secao="dosagem", chunk_id="b1__dosagem__003")


def _pergunta(**kw):
    base = dict(
        id="q001", pergunta="A losartana interage com anti-inflamatórios?", respondivel=True,
        ancoras=[{"principio_ativo": "losartana potássica", "secao": "interacoes",
                  "frase": "anti-inflamatórios não esteroidais"}],
        resposta_esperada="Sim, reduz o efeito.", categoria="interacao",
    )
    return PerguntaBenchmark(**{**base, **kw})


def test_pergunta_respondivel_exige_ancora_e_resposta():
    _pergunta()
    with pytest.raises(ValidationError):
        _pergunta(ancoras=[])
    with pytest.raises(ValidationError):
        _pergunta(resposta_esperada=None)


def test_pergunta_sem_resposta_nao_pode_ter_gabarito():
    _pergunta(id="q041", respondivel=False, ancoras=[], resposta_esperada=None, categoria="fora_do_escopo")
    with pytest.raises(ValidationError):
        _pergunta(respondivel=False, categoria="fora_do_escopo")


def test_dataset_real_segue_o_contrato():
    """Valida o que estiver em data/ (passa vazio enquanto não há dados)."""
    for caminho in (RAIZ / "data/processed/bulas").glob("*.json"):
        Bula.model_validate_json(caminho.read_text(encoding="utf-8"))

    chunks = RAIZ / "data/processed/chunks.jsonl"
    if chunks.exists():
        ids = set()
        for linha in chunks.read_text(encoding="utf-8").splitlines():
            c = Chunk.model_validate_json(linha)
            assert c.chunk_id not in ids, f"chunk_id duplicado: {c.chunk_id}"
            ids.add(c.chunk_id)

    perguntas = RAIZ / "data/benchmark/perguntas.json"
    if perguntas.exists():
        for item in json.loads(perguntas.read_text(encoding="utf-8")):
            PerguntaBenchmark.model_validate(item)
