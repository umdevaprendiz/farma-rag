# Tela em produção (docs/DECISOES.md §38 e §44): o Node só constrói (tsc + vite build) e fica para trás;
# a imagem final é o nginx SEM privilégios, com a pasta dist/ e o nginx.conf, que também repassa /api para o
# backend. Versões exatas e imagens fixadas por digest: atualizar é sempre uma decisão.
#
# Imagens verificadas no Docker Hub em 2026-10-29:
#   library/node:24.22.1-alpine3.24                     (imagem oficial do Node)
#   nginxinc/nginx-unprivileged:1.30.6-alpine3.24       (mantida pelo time do NGINX; roda como usuário
#                                                        "nginx" (uid 101) e escuta na 8080, sem root)

FROM node:24.22.1-alpine3.24@sha256:06d7adf3e0ef61a13f1d28682726366f97a15aa1888c934d3ecec86c33918b53 AS build
WORKDIR /app
# Primeiro só o que define as dependências: muda pouco, e o Docker reaproveita a camada do npm ci.
# O .npmrc vem junto: ignore-scripts=true vale aqui também.
COPY package.json package-lock.json .npmrc ./
RUN npm ci
COPY index.html tsconfig.json vite.config.ts ./
COPY src ./src
RUN npm run build

FROM nginxinc/nginx-unprivileged:1.30.6-alpine3.24@sha256:f5116ed4606f3eb015dd4a8e28cb9b351476dc0ad4bc6e9f93b19cec130cdf03
# Como root só durante a construção; o processo roda como "nginx" (uid 101).
USER root
# O docker scout achou, na imagem base de 2026-10-19, pacotes do Alpine com falha JÁ corrigida: brotli
# (CVE-2026-111497, crítica e na lista de explorados da CISA), curl e zlib. `apk upgrade` traz as versões
# corrigidas; o curl sai (a tela não o usa: o healthcheck é wget). Fica só o que o Alpine ainda não corrigiu
# (libxml2), registrado em docs/SEGURANCA.md. Refazer o scan ao trocar o digest da base.
RUN apk upgrade --no-cache  && apk del --no-cache curl  && rm -rf /usr/share/nginx/html/*
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
USER 101
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["wget", "-q", "-O", "/dev/null", "http://127.0.0.1:8080/"]
