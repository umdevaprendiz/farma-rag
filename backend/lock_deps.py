"""Gera backend/requirements.lock: as versões EXATAS do que o requirements.txt instala neste ambiente.

Uso (da raiz, com a .venv do projeto): python backend/lock_deps.py

Por quê: requirements.txt usa `>=`, então cada build da imagem pode trazer versões novas, nunca testadas
aqui (e, na cadeia de suprimentos, uma versão maliciosa publicada ontem entra sozinha). A imagem Docker
instala deste arquivo (backend/Dockerfile). Atualizar é uma decisão: rode os testes (`pytest`) e o
`python -m pip_audit -r backend/requirements.lock` antes de commitar a mudança.

Só biblioteca padrão e `packaging` (que o pip já traz): percorre as dependências de cada pacote de
requirements.txt (sem extras opcionais que ninguém pediu) e fixa o que está instalado.
"""

import re
import sys
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

AQUI = Path(__file__).resolve().parent


def pedidos() -> list[Requirement]:
    linhas = (l.split("#")[0].strip() for l in (AQUI / "requirements.txt").read_text(encoding="utf-8").splitlines())
    return [Requirement(l) for l in linhas if l]


def fechamento(raizes: list[Requirement]) -> dict[str, str]:
    """{nome: versão instalada} do pacote e de tudo de que ele depende."""
    achados: dict[str, str] = {}
    fila = [(r, set(r.extras)) for r in raizes]
    while fila:
        req, extras = fila.pop()
        nome = canonicalize_name(req.name)
        try:
            dist = metadata.distribution(req.name)
        except metadata.PackageNotFoundError:
            sys.exit(f"{req.name} não está instalado: rode `pip install -r backend/requirements.txt` primeiro.")
        primeira_vez = nome not in achados
        achados[nome] = dist.version
        for dep in dist.requires or []:
            r = Requirement(dep)
            if r.marker is None or any(r.marker.evaluate({"extra": e}) for e in (extras or {""})):
                if r.marker is None or r.marker.evaluate({"extra": ""}) or extras:
                    if primeira_vez or r.extras:
                        fila.append((r, set(r.extras)))
    return dict(sorted(achados.items()))


if __name__ == "__main__":
    pacotes = fechamento(pedidos())
    cabecalho = (
        "# Gerado por backend/lock_deps.py: versões exatas testadas. Não edite à mão.\n"
        f"# Python {sys.version_info.major}.{sys.version_info.minor}; {len(pacotes)} pacotes.\n"
    )
    (AQUI / "requirements.lock").write_text(
        cabecalho + "".join(f"{n}=={v}\n" for n, v in pacotes.items()), encoding="utf-8", newline="\n"
    )
    print(f"requirements.lock: {len(pacotes)} pacotes")
