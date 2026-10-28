# Tela em produção (docs/DECISOES.md §38): o Node só constrói (tsc + vite build) e fica para trás;
# a imagem final é o nginx com a pasta dist/ e o nginx.conf, que também repassa /api para o backend.
# Versões exatas, como no package.json: atualizar é sempre uma decisão.

FROM node:24.22.1-alpine3.24 AS build
WORKDIR /app
# Primeiro só o que define as dependências: muda pouco, e o Docker reaproveita a camada do npm ci.
# O .npmrc vem junto: ignore-scripts=true vale aqui também.
COPY package.json package-lock.json .npmrc ./
RUN npm ci
COPY index.html tsconfig.json vite.config.ts ./
COPY src ./src
RUN npm run build

FROM nginx:1.30.6-alpine3.24
# Sai a página de boas-vindas do nginx; fica só a tela.
RUN rm -rf /usr/share/nginx/html/*
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
