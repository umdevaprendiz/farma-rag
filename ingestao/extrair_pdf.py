"""Extrai o texto de uma bula, uma string por página.

Aceita .pdf (pdfplumber, só no host: ver backend/requirements-dev.txt) e .txt, em que as páginas
são separadas por form feed ("\\f"); o .txt serve para teste e para bula colada à mão.
"""

from pathlib import Path


def extrair_paginas(caminho: str | Path) -> list[str]:
    caminho = Path(caminho)
    if caminho.suffix.lower() == ".txt":
        return caminho.read_text(encoding="utf-8").split("\f")
    try:
        import pdfplumber
    except ImportError as e:
        raise RuntimeError("Falta o pdfplumber: pip install -r backend/requirements-dev.txt") from e
    with pdfplumber.open(caminho) as pdf:
        return [pagina.extract_text() or "" for pagina in pdf.pages]
