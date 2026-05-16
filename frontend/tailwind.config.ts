import type { Config } from "tailwindcss";

const ink = Object.fromEntries(
  [50, 100, 200, 300, 400, 500, 600, 700, 800, 900].map((n) => [
    n,
    `rgb(var(--ink-${n}) / <alpha-value>)`,
  ])
);

const config: Config = {
  darkMode: "class",
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        ink,
        surface: "rgb(var(--surface) / <alpha-value>)",
        brand: {
          DEFAULT: "#4f46e5",
          light: "#a5b4fc",
          dark: "rgb(var(--brand-strong) / <alpha-value>)",
        },
      },
    },
  },
  plugins: [],
};
export default config;
