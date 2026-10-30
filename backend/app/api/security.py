"""Cabeçalhos de segurança das respostas da API (docs/SEGURANCA.md, achado 5).

Middleware ASGI puro, não `BaseHTTPMiddleware`: este último guarda a resposta em buffer e atrapalha o fluxo
NDJSON do progresso (api/progress.py), que precisa sair linha a linha.

- `X-Content-Type-Options: nosniff`: o navegador não "adivinha" que um JSON é HTML.
- `Cache-Control: no-store`: a pergunta e a resposta não ficam em cache de navegador nem de proxy (a pergunta
  pode, mesmo sem querer, conter dado de paciente).
"""

CABECALHOS = [(b"x-content-type-options", b"nosniff"), (b"cache-control", b"no-store")]


class SecurityHeadersMiddleware:
    def __init__(self, app, prefixo: str = "/api/"):
        self.app = app
        self.prefixo = prefixo

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith(self.prefixo):
            await self.app(scope, receive, send)
            return

        async def enviar(mensagem):
            if mensagem["type"] == "http.response.start":
                existentes = {nome.lower() for nome, _ in mensagem.get("headers", [])}
                extra = [(n, v) for n, v in CABECALHOS if n not in existentes]
                mensagem = {**mensagem, "headers": [*mensagem.get("headers", []), *extra]}
            await send(mensagem)

        await self.app(scope, receive, enviar)
