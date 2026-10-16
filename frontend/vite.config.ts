import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Em desenvolvimento a tela chama /api no próprio Vite, que repassa para a API: mesma origem,
// sem depender de CORS. 127.0.0.1 e não "localhost": no Windows o localhost tenta o IPv6 primeiro.
// FARMARAG_API troca o destino (ex.: uma API de teste em outra porta).
const api = process.env.FARMARAG_API ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    // Sem host fixo, o Vite escuta em "localhost", que nesta máquina vira só o IPv6 (::1), e o
    // navegador que tenta 127.0.0.1 recebe conexão recusada (erro -102). Endereço: 127.0.0.1:5173.
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: { "/api": api },
  },
});
