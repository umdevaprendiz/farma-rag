// Ícones em SVG embutido (traço na cor do texto): sem biblioteca de ícones.

const BASE = {
  width: 20,
  height: 20,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
};

export function Logo() {
  return (
    <svg width="100%" height="100%" viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="8" fill="var(--acento)" />
      <path d="M13 7h6v6h6v6h-6v6h-6v-6H7v-6h6z" fill="var(--sobre-acento)" />
    </svg>
  );
}

export function IconeEnviar() {
  return (
    <svg {...BASE}>
      <path d="M12 19V5M5 12l7-7 7 7" />
    </svg>
  );
}

export function IconeNovo() {
  return (
    <svg {...BASE}>
      <path d="M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z" />
    </svg>
  );
}

export function IconeMenu() {
  return (
    <svg {...BASE}>
      <path d="M4 6h16M4 12h16M4 18h16" />
    </svg>
  );
}

export function IconeFechar() {
  return (
    <svg {...BASE}>
      <path d="M18 6 6 18M6 6l12 12" />
    </svg>
  );
}
