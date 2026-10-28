"""Cache de embeddings: poda antes do Git e o arquivo versionado cobrindo o acervo (DECISOES §39)."""

import hashlib
import os
import shutil
import sqlite3
import stat
from contextlib import closing
from pathlib import Path

import pytest

from app.config import Settings
from app.llm import ollama
from app.llm.embeddings import EmbeddingCache, _chave
from app.services.corpus import IncompleteCorpusError, carregar_chunks, carregar_dataset, texto_para_embedding
from ingestion import generate_chunks

RAIZ = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures"


def _linhas(caminho: Path) -> set[str]:
    with closing(sqlite3.connect(f"file:{caminho}?mode=ro", uri=True)) as db:
        return {k for (k,) in db.execute("SELECT chave FROM embeddings")}


def test_podar_deixa_so_os_textos_atuais_do_modelo(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.sqlite")
    cache.gravar("bge-m3", ["atual 1", "atual 2", "antigo"], [[0.1, 0.2]] * 3)
    cache.gravar("outro-modelo", ["atual 1"], [[0.3, 0.4]])

    assert cache.sobras("bge-m3", ["atual 1", "atual 2"]) == 2  # só conta, não apaga
    assert cache.podar("bge-m3", ["atual 1", "atual 2"]) == 2
    assert cache.sobras("bge-m3", ["atual 1", "atual 2"]) == 0
    achados = cache.buscar("bge-m3", ["atual 1", "atual 2", "antigo"])
    assert [v is not None for v in achados] == [True, True, False]
    assert cache.buscar("outro-modelo", ["atual 1"]) == [None]
    cache.fechar()
    with closing(sqlite3.connect(tmp_path / "e.sqlite")) as db:
        assert db.execute("PRAGMA freelist_count").fetchone()[0] == 0  # VACUUM: nada apagado sobra no arquivo


def _settings(tmp_path: Path) -> Settings:
    bulas = tmp_path / "bulas"
    bulas.mkdir()
    shutil.copy(FIXTURES / "sample_leaflet.json", bulas / "exemplol.json")
    return Settings(
        bulas_dir=str(bulas), chunks_arquivo=str(tmp_path / "chunks.jsonl"),
        cache_embeddings=str(tmp_path / "cache" / "e.sqlite"), embed_modelo="bge-m3",
    )


def test_generate_chunks_podar_com_tudo_no_cache_nao_chama_o_ollama(tmp_path, monkeypatch, capsys):
    s = _settings(tmp_path)
    generate_chunks.main(s, com_embeddings=False)  # só os trechos, para saber os textos
    textos = [texto_para_embedding(c) for c in carregar_chunks(s.chunks_arquivo)]
    cache = EmbeddingCache(s.cache_embeddings)
    cache.gravar(s.embed_modelo, textos + ["trecho de uma versão antiga"], [[0.5, 0.5]] * (len(textos) + 1))
    cache.fechar()

    def sem_ollama(*_a, **_k):
        raise AssertionError("com tudo no cache, o Ollama não pode ser chamado")

    monkeypatch.setattr(ollama, "embed", sem_ollama)
    generate_chunks.main(s)
    assert "Aviso: 1 linha(s)" in capsys.readouterr().out
    assert len(_linhas(Path(s.cache_embeddings))) == len(textos) + 1  # sem --podar, nada sai

    generate_chunks.main(s, podar=True)
    assert "Poda: 1 linha(s)" in capsys.readouterr().out
    assert _linhas(Path(s.cache_embeddings)) == {_chave(s.embed_modelo, t) for t in textos}
    _bulas, chunks, matriz = carregar_dataset(s)  # o que o load_database faz num clone novo
    assert matriz.shape == (len(chunks), 2)


def test_podar_sem_embeddings_nao_roda(tmp_path):
    with pytest.raises(SystemExit):
        generate_chunks.main(_settings(tmp_path), com_embeddings=False, podar=True)


def test_cache_versionado_cobre_todos_os_trechos_do_acervo():
    """Um clone novo carrega o banco só com o que está no Git: todo trecho do chunks.jsonl precisa do
    embedding no data/cache/embeddings.sqlite, com o modelo padrão (o do .env.example)."""
    cache = RAIZ / "data/cache/embeddings.sqlite"
    chunks = carregar_chunks(RAIZ / "data/processed/chunks.jsonl")
    if not cache.exists() or not chunks:
        pytest.skip("sem acervo em data/")
    modelo = Settings.model_fields["embed_modelo"].default
    faltam = {_chave(modelo, texto_para_embedding(c)) for c in chunks} - _linhas(cache)
    assert not faltam, f"{len(faltam)} trecho(s) sem embedding no cache: python -m ingestion.generate_chunks"


def _acervo_pronto(tmp_path: Path) -> Settings:
    """Bula da fixture com trechos e todos os embeddings no cache, como num clone novo."""
    s = _settings(tmp_path)
    generate_chunks.main(s, com_embeddings=False)
    textos = [texto_para_embedding(c) for c in carregar_chunks(s.chunks_arquivo)]
    cache = EmbeddingCache(s.cache_embeddings)
    cache.gravar(s.embed_modelo, textos, [[0.5, 0.5]] * len(textos))
    cache.fechar()
    return s


def test_carga_le_o_cache_sem_alterar_nem_mesmo_so_leitura(tmp_path):
    """No container da carga, data/ é montado só leitura: a carga não pode tentar escrever no cache."""
    s = _acervo_pronto(tmp_path)
    arquivo = Path(s.cache_embeddings)
    antes = hashlib.sha256(arquivo.read_bytes()).hexdigest()
    os.chmod(arquivo, stat.S_IREAD)
    try:
        _bulas, chunks, matriz = carregar_dataset(s)
    finally:
        os.chmod(arquivo, stat.S_IREAD | stat.S_IWRITE)
    assert matriz.shape == (len(chunks), 2)
    assert hashlib.sha256(arquivo.read_bytes()).hexdigest() == antes


def test_carga_sem_o_cache_explica_e_nao_cria_arquivo(tmp_path):
    s = _acervo_pronto(tmp_path)
    Path(s.cache_embeddings).unlink()
    with pytest.raises(IncompleteCorpusError, match="Sem o cache de embeddings"):
        carregar_dataset(s)
    assert not Path(s.cache_embeddings).exists()


def test_somente_leitura_recusa_gravar(tmp_path):
    """O modo da carga é só leitura de verdade (no Windows, o SQLite abriria o arquivo protegido sozinho)."""
    cache = EmbeddingCache(tmp_path / "e.sqlite")
    cache.gravar("bge-m3", ["a"], [[0.1, 0.2]])
    cache.fechar()
    leitura = EmbeddingCache(tmp_path / "e.sqlite", somente_leitura=True)
    assert leitura.buscar("bge-m3", ["a"])[0] is not None
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        leitura.gravar("bge-m3", ["b"], [[0.3, 0.4]])
    leitura.fechar()
