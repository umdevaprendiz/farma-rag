"""Resposta em fluxo com o progresso do agente (F3d, docs/DECISOES.md, seção 36).

Uma pergunta leva 30-45 s na CPU, e até ~2 min com rodadas. Em vez de a tela esperar calada, a API
manda uma linha de JSON (NDJSON) a cada etapa que começa e, no fim, uma linha com a resposta (ou o erro):

    {"tipo": "etapa", "etapa": "buscando", "rodada": 1, "de": 3}
    {"tipo": "etapa", "etapa": "escrevendo", "rodada": 1, "de": 3}
    {"tipo": "etapa", "etapa": "conferindo", "rodada": 1, "de": 3}
    {"tipo": "resposta", "resposta": {...o mesmo AnswerOut do POST /perguntas...}}
ou  {"tipo": "erro", "erro": {"codigo": ..., "mensagem": ...}, "consulta_id": ..., "status": 502}

`{"tipo": "vivo"}` sai a cada HEARTBEAT_S sem novidade, para nenhum proxy fechar a conexão parada
durante uma geração longa. O erro vai no fluxo (o status 200 já foi enviado com o primeiro byte);
`status` leva o código HTTP que o mesmo erro teria no endpoint comum.

O trabalho roda numa thread própria e conversa com o fluxo por uma fila. Se quem espera fechar a
conexão, a thread termina a pergunta e grava a consulta no histórico: nada se perde, só não há mais
quem receba. Falha inesperada (bug): o fluxo termina sem a linha final, e quem lê deve tratar isso
como resposta interrompida.
"""

import json
import logging
import queue
import threading
from collections.abc import Callable, Iterator

from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.exc import OperationalError

from app.api.errors import ApiError
from app.schemas.api import HTTP_POR_CODIGO, ErrorCode
from app.services.rag import Progress

log = logging.getLogger("farmarag")

HEARTBEAT_S = 15.0
MEDIA_TYPE = "application/x-ndjson"


def _linha(evento: dict) -> str:
    return json.dumps(evento, ensure_ascii=False) + "\n"


def eventos(
    trabalho: Callable[[Progress], BaseModel], rodadas_max: int, mensagem_banco: str, heartbeat_s: float = HEARTBEAT_S
) -> Iterator[str]:
    """As linhas do fluxo, à medida que o `trabalho` avisa as etapas. `trabalho(avisar)` devolve a
    resposta (um modelo Pydantic) ou levanta ApiError / OperationalError."""
    fila: queue.Queue[dict | None] = queue.Queue()

    def avisar(etapa: str, rodada: int) -> None:
        fila.put({"tipo": "etapa", "etapa": etapa, "rodada": rodada, "de": rodadas_max})

    def erro(codigo: ErrorCode, mensagem: str, consulta_id=None) -> dict:
        return {
            "tipo": "erro", "erro": {"codigo": codigo.value, "mensagem": mensagem},
            "consulta_id": str(consulta_id) if consulta_id else None, "status": HTTP_POR_CODIGO[codigo],
        }

    def correr() -> None:
        try:
            fila.put({"tipo": "resposta", "resposta": trabalho(avisar).model_dump(mode="json")})
        except ApiError as e:
            fila.put(erro(e.codigo, e.mensagem, e.consulta_id))
        except OperationalError:
            fila.put(erro(ErrorCode.BANCO_INDISPONIVEL, mensagem_banco))
        except Exception:
            log.exception("Falha inesperada ao responder a pergunta (fluxo com progresso)")
        finally:
            fila.put(None)  # fim do fluxo

    threading.Thread(target=correr, name="pergunta-com-progresso", daemon=True).start()
    while True:
        try:
            evento = fila.get(timeout=heartbeat_s)
        except queue.Empty:
            yield _linha({"tipo": "vivo"})
            continue
        if evento is None:
            return
        yield _linha(evento)


def transmitir(trabalho: Callable[[Progress], BaseModel], rodadas_max: int, mensagem_banco: str) -> StreamingResponse:
    return StreamingResponse(
        eventos(trabalho, rodadas_max, mensagem_banco),
        media_type=MEDIA_TYPE,
        # no-cache e X-Accel-Buffering: nenhum intermediário (nginx, proxy) guarda o fluxo até o fim.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
