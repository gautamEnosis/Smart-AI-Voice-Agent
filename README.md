# Smart AI Voice Agent — OmniVox

Autonomous AI voice agent platform with sub-second latency, local-first LLMs, dynamic RAG, and multimodal Generative UI.

The platform ships three applications plus a local LLM daemon:

| Service | What it is | Port |
|---|---|---|
| **backend** | FastAPI gateway + Pipecat voice pipeline (WebRTC, VAD → STT → LLM → TTS, MCP tools, RAG) | `8000` |
| **consumer_ui** | Guest concierge portal (WebRTC voice, live captions, generative booking UI) | `3000` |
| **admin_panel** | Operations console (tenants, personas, voice studio, knowledge base) | `5173` |
| **ollama** | Local LLM daemon (default `llama3.2:3b`) | `11434` |

---

## Prerequisites

- **Python 3.11+** (tested on 3.14)
- **Node.js 18+** (tested on 24) and npm
- **Ollama** — <https://ollama.com> (local LLM; the default config needs no API key)
- **Docker** (optional — only if you use the Compose path)
- A browser with **microphone permission** for the voice portal (Chrome/Firefox/Edge)

---

## Quick Start

### Option A — Docker Compose (everything in containers)

```bash
cp .env.example .env          # required: compose reads .env
docker compose up --build     # first run (builds images, pulls the LLM model)
docker compose up             # subsequent runs
```

This starts all four services with health checks and correct wiring
(`OLLAMA_BASE_URL=http://ollama:11434/v1` inside the stack):

- Consumer portal → <http://localhost:3000>
- Admin console → <http://localhost:5173>
- API gateway + Swagger docs → <http://localhost:8000/docs>
- Ollama → <http://localhost:11434>

Stop everything with `docker compose down` (add `-v` to also drop volumes).

### Option B — Manual (local development)

**1. Backend (FastAPI + Pipecat)**

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows — use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt

cp .env.example .env            # local-first defaults; edit only if you need cloud providers

uvicorn backend.api.app:app --host 0.0.0.0 --port 8000
```

Verify: <http://localhost:8000/health> → `{"status":"ok","service":"omnivox-gateway"}`
API reference: <http://localhost:8000/docs>

> If you already have a `.venv` without dependencies, re-run
> `pip install -r requirements.txt` inside it before starting the server.

**2. Ollama (local LLM)**

```bash
ollama pull llama3.2:3b         # matches OLLAMA_MODEL in .env
ollama serve                    # usually already running as a service
```

Without Ollama the server still boots, but agent replies (and therefore live
captions) will fail while `LLM_PROVIDER=ollama`.

**3. Consumer portal (guest voice UI)**

```bash
cd consumer_ui
npm install
npm run dev                     # http://localhost:3000
```

The portal talks to `http://127.0.0.1:8000` by default; override with
`VITE_API_BASE_URL` if the gateway lives elsewhere.

**4. Admin console (operations UI)**

```bash
cd admin_panel
npm install
npm run dev                     # http://localhost:5173
```

API base defaults to `http://localhost:8000`; override with `VITE_API_BASE_URL`.
The console asks for `ADMIN_API_KEY` from the backend `.env` and keeps it only
for the current browser tab. Generate a unique key before starting the backend:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Set the generated value as `ADMIN_API_KEY` in `.env`; admin endpoints fail
closed when the key is unset.
For HTTPS deployments, update the Nginx `connect-src` policy in
`docker/nginx/admin.conf` and `docker/nginx/consumer.conf` to the deployed API
and signaling origins.

**5. Open the portal**

Go to <http://localhost:3000>, click **Connect**, allow microphone access, and
start talking (or type). Localhost WebRTC needs no STUN/TURN server.

---

## Configuration (`.env`)

Copy `.env.example` → `.env`. All defaults are local-first ($0 compute):

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` \| `gemini` \| `groq` \| `openai` |
| `OLLAMA_MODEL` / `OLLAMA_BASE_URL` | `llama3.2:3b` / `http://localhost:11434/v1` | Local LLM endpoint |
| `STT_PROVIDER` | `faster_whisper` | Speech-to-text backend |
| `TTS_PROVIDER` | `kokoro` | Text-to-speech backend |
| `ADMIN_API_KEY` | *(unset; required for admin access)* | Secret bearer key used by the Admin Console |
| `FASTER_WHISPER_MODEL` | `base.en` | Whisper model size |
| `KOKORO_VOICE_ID` / `KOKORO_SPEED` | `af_sarah` / `1.00` | Voice profile |
| `DATABASE_URL` | `sqlite:///./data/voice_agent.db` | SQLite persistence |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | Embedded ChromaDB (RAG) |
| `EMBEDDING_PROVIDER` | `local_hash` | `local_hash` (no download) \| `nomic_ollama` \| `fastembed` |
| `WEATHER_PROVIDER` | `stub` | `stub` \| `open_meteo` (no key needed) |
| `*_API_KEY` | *(empty)* | Only required when switching off local providers |

Cloud providers are hot-swappable: set the provider variable and its API key in
`.env` — no code changes, no rebuild in Compose.

---

## Local speech (STT/TTS)

The default voice path requires the Pipecat `whisper` and `kokoro` extras.
Install all backend dependencies from `requirements.txt`; without the speech
dependencies, voice-session startup reports an error instead of silently
running pass-through processors that cannot transcribe or speak.

- **faster-whisper** downloads its configured model (`base.en` by default) on
  first use. The initial session may take longer while the model is downloaded
  and loaded; later turns reuse the local model.
- **kokoro-onnx** downloads the Kokoro model and voice pack on first use.

---

## Tests & checks

```bash
# Backend — run ruff first, then the suite
ruff check backend/
python -m pytest backend/tests -q

# Frontend (run inside consumer_ui/ and admin_panel/)
npm run lint        # tsc type-check
npm run build       # tsc + production bundle
```

The backend suite includes a live WebRTC/pipeline integration group
(`test_pipeline_integration.py`, `test_webrtc_*`) that exercises a real
`SmallWebRTCConnection`, DataChannel emitters, and — when Ollama is running —
a real local-LLM roundtrip.

---

## Project layout

```
backend/
  api/            FastAPI app, voice/admin routes, WebRTC session manager
  engine/         Pipecat pipeline, provider factories (LLM/STT/TTS), audio
  mcp/            In-process MCP tool hub + out-loud confirmation gate
  rag/            ChromaDB retrieval pipeline
  config/         Pydantic settings + tenant/persona YAML
  tests/          pytest suite
consumer_ui/      React + Vite guest portal (WebRTC, live captions, booking UI)
admin_panel/      React + Vite operations console
data/             SQLite DB, ChromaDB store, uploads (created at runtime)
planning/         Architecture & security master documents
.agents/          Agent memory, active state, session context
```

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Portal shows "Conversation ended" on load | Harmless React StrictMode dev-only effect; the Connect button still works |
| `Connect` fails with "Microphone access is required" | Browser denied mic access — allow it for `localhost` and retry |
| Connect succeeds but no reply/caption | Check backend logs for a pipeline startup error; verify the Pipecat provider extras in `requirements.txt` and that Ollama is running with the configured model |
| First voice response is slow | Faster-Whisper or Kokoro may be downloading/loading local model assets for the first time |
| `404 tenant not available` | `tenant_id` in the URL/config doesn't match `backend/config/tenants/` |
| CORS errors in the browser console | UI origin missing from `admin_cors_origins` / `consumer_cors_origins` in `.env` |
| Port already in use | Stop the other instance or change `--port` / the Vite `server.port` |
