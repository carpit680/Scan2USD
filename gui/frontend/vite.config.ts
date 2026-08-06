import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // Bind to the ZeroTier interface only, so the GUI is reachable from
    // ZeroTier peers but not from the regular LAN.
    host: "192.168.194.3",
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8765",
        changeOrigin: true,
      },
    },
  },
});
