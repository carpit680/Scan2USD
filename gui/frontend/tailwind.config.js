/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#0c1014",
          900: "#12181f",
          800: "#1a2330",
          700: "#243042",
          600: "#334155",
          400: "#94a3b8",
          300: "#cbd5e1",
          100: "#e8eef5",
        },
        accent: {
          DEFAULT: "#2dd4bf",
          dim: "#14b8a6",
          muted: "#0f766e",
        },
        warn: "#f59e0b",
        danger: "#f87171",
        ok: "#4ade80",
      },
      fontFamily: {
        sans: ['"DM Sans"', "system-ui", "sans-serif"],
        display: ['"Space Grotesk"', "system-ui", "sans-serif"],
        mono: ['"IBM Plex Mono"', "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
