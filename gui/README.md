# Scan2USD GUI

Separate FastAPI + React/Vite UI for driving every Scan2USD command, config parameter, pipeline stage, and review gate.

The Gradio `scan2usd review` app is unchanged; this GUI also implements review via `ReviewSession` and is the intended long-term replacement.

## Requirements

- Scan2USD installed in the active venv (`pip install -e ".[dev,geometry,review]"`)
- Node.js 18+ for the frontend
- Python 3.10+

## Install

From the repo root:

```bash
pip install -e ".[dev,geometry,review]"
pip install -e gui/backend
cd gui/frontend && npm install && cd ../..
```

## Run (dev)

```bash
make gui
```

Or manually:

```bash
# terminal 1 — API on :8765
cd gui/backend && python -m scan2usd_gui --reload

# terminal 2 — Vite on :5173 (proxies /api → :8765)
cd gui/frontend && npm run dev
```

Open http://127.0.0.1:5173

## Upload video from phone (LAN)

Phone and desktop must be on the same Wi‑Fi. The QR always points at the API port (**8765**), not Vite.

```bash
make gui-lan
```

1. Open a project in the desktop GUI (http://127.0.0.1:5173).
2. **Config → Essentials → Source video → Browse → From phone…**
3. Scan the QR (or open the shown `http://<LAN-IP>:8765/m?t=…` URL).
4. Choose a video on the phone; the desktop picker fills the path when upload completes.

Allow inbound TCP **8765** on the host firewall if the phone cannot connect.

## Production-ish (single process)

```bash
cd gui/frontend && npm run build
cd ../backend && python -m scan2usd_gui
```

FastAPI serves `frontend/dist` when present.

## Layout

```text
gui/
  backend/scan2usd_gui/   # FastAPI app, schema, jobs, routes
  frontend/               # React + Tailwind + Radix tooltips
```

## Tests

```bash
cd gui/backend && python -m pytest -q
```

`schema.py` is the GUI contract for config fields and CLI/tool commands. Completeness tests assert Typer CLI commands and SceneConfig fields stay covered.
