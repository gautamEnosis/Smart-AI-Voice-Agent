# Smart AI Voice Agent

> **Carrier-grade, autonomous AI voice agent platform** — sub-second latency, 100% local-first, multimodal Generative UI, and mandatory spoken confirmation gates for all booking writes.

Built on [Pipecat](https://pipecat.ai) 1.12.0 · FastAPI · React 19 · TypeScript · Tailwind CSS · SQLite · ChromaDB

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start (Local-First, \$0 Compute)](#quick-start-local-first-0-compute)
- [Running Each Service](#running-each-service)
  - [1 — Backend API Gateway](#1--backend-api-gateway)
  - [2 — Consumer Concierge Portal](#2--consumer-concierge-portal)
  - [3 — Admin Operations Console](#3--admin-operations-console)
- [Running the Test Suite](#running-the-test-suite)
- [Environment Variables Reference](#environment-variables-reference)
- [Provider Hot-Swap Guide](#provider-hot-swap-guide)
- [Multi-Tenant Configuration](#multi-tenant-configuration)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Security Notes](#security-notes)

---

## Overview

**Smart AI Voice Agent** is a modular, multi-tenant enterprise voice agent platform. It powers role-based autonomous voice agents — Hotel Booking Concierge, Front Desk Receptionist, Room Service, Travel Desk — as browser-embedded WebRTC widgets with real-time multimodal cards (room recommendations, booking progress tracker, live captions).

**Core Design Principles:**

| Principle | Detail |
|---|---|
| **Local-First Independence** | All core features run fully offline using Ollama + Faster-Whisper + Kokoro-82M + ChromaDB. No external API required. |
| **Cloud Hot-Swap** | Swap to Gemini / OpenAI / Groq (LLM), Deepgram (STT), Cartesia / ElevenLabs (TTS) via `.env` — zero code changes. |
| **Sub-Second Latency** | Silero VAD v5 + streaming TTS on sentence boundaries → TTFB < 500 ms, full turn < 800 ms. |
| **Mandatory Confirmation Gate** | All booking, cancellation, and payment writes require explicit **verbal** caller confirmation. Screen taps are context-only. |

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    Browser / Mobile Client                         │
│  Consumer Portal (React 19, Port 3000)                            │
│  ┌─────────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ AudioVisualizer │  │ LiveCaptions │  │ GenerativeCards      │  │
│  │ DualInputBar    │  │ (WebRTC DC)  │  │ RoomCard             │  │
│  │ PTT / Text      │  │              │  │ BookingStepper       │  │
│  └─────────────────┘  └──────────────┘  └──────────────────────┘  │
└────────────────────────────┬───────────────────────────────────────┘
                             │ WebRTC (SmallWebRTC)
                             │ SDP/ICE + Audio + DataChannel
┌────────────────────────────▼───────────────────────────────────────┐
│           FastAPI Gateway (Python 3.11+, Port 8000)                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Pipecat 1.12.0 Voice Pipeline                  │   │
│  │  Transport → VAD → STT → LLM+MCP Tools → Normalizer → TTS  │   │
│  │  DataChannelBridge emits: caption, room_card, booking_step  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌───────────────┐  ┌─────────────┐  ┌──────────────────────────┐  │
│  │ MCP Tool Hub  │  │ RAG Engine  │  │ Admin REST API           │  │
│  │ 6 Tool Servers│  │ ChromaDB    │  │ (Personas, Providers,    │  │
│  │ SQLite Store  │  │ TF-IDF      │  │  Knowledge, Audit)       │  │
│  └───────────────┘  └─────────────┘  └──────────────────────────┘  │
└────────────────────────────┬───────────────────────────────────────┘
                             │ REST API
┌────────────────────────────▼───────────────────────────────────────┐
│          Admin Console (React 19, Port 5173)                       │
│  Dashboard · VoiceStudio · ProviderSettings · KnowledgeHub        │
└────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Requirement | Minimum Version | Notes |
|---|---|---|
| **Python** | 3.11 or 3.14 | 3.14 tested; 3.12 recommended for production |
| **Node.js** | 18 LTS or higher | For frontend builds |
| **npm** | 9+ | Comes with Node.js |
| **Ollama** | Latest | Required for local LLM (default mode) |
| **Git** | Any | For cloning |

> **Windows users:** All commands below use PowerShell. Replace `python` with `py` if needed.

---

## Quick Start (Local-First, \$0 Compute)

### Step 1 — Clone and configure environment

```powershell
git clone <repository-url>
cd Smart-AI-Voice-Agent

# Copy environment template
Copy-Item .env.example .env
```

The default `.env` is configured for 100% local operation. No API keys are needed.

---

### Step 2 — Set up the Python virtual environment

```powershell
# Create virtual environment
python -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1   # PowerShell
# or: .venv\Scripts\activate.bat   # CMD

# Install dependencies
pip install -r requirements.txt
```

> **Note on Pipecat extras:** For local STT/TTS (Whisper + Kokoro), install optional extras:
> ```powershell
> pip install "pipecat-ai[whisper,silero,kokoro,webrtc]==1.12.0"
> ```
> The platform gracefully stubs missing extras and runs tests regardless.

---

### Step 3 — Pull the local LLM with Ollama

```powershell
# Install Ollama: https://ollama.com/download
ollama pull llama3.2:3b
```

> **Tip:** Ollama must be running before you start the backend. It starts automatically as a service on Windows after install, or run `ollama serve` manually.

---

### Step 4 — Install frontend dependencies

```powershell
# Consumer Portal
cd consumer_ui
npm install
cd ..

# Admin Console
cd admin_panel
npm install
cd ..
```

---

### Step 5 — Start all three services

Open **three separate terminal windows**:

**Terminal 1 — Backend API:**
```powershell
cd Smart-AI-Voice-Agent
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.api.app:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 — Consumer Portal:**
```powershell
cd Smart-AI-Voice-Agent\consumer_ui
npm run dev
```

**Terminal 3 — Admin Console:**
```powershell
cd Smart-AI-Voice-Agent\admin_panel
npm run dev
```

Then open:
- **Consumer Portal (Elena):** http://localhost:3000
- **Admin Console:** http://localhost:5173
- **API Docs (Swagger):** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

---

## Running Each Service

### 1 — Backend API Gateway

The FastAPI backend hosts:
- `/api/v1/voice/*` — WebRTC SDP/ICE signalling + Pipecat pipeline
- `/api/v1/admin/*` — Persona, provider, knowledge, and audit management
- `/api/v1/voice/audition` — Voice provider audition endpoint

**Development (with auto-reload):**
```powershell
python -m uvicorn backend.api.app:app --host 127.0.0.1 --port 8000 --reload
```

**Production:**
```powershell
python -m uvicorn backend.api.app:app --host 0.0.0.0 --port 8000 --workers 1
```

> **Important:** Use `--workers 1` — SmallWebRTC transport maintains in-process session state and must not be sharded across multiple processes.

**CORS origins** (configured in `.env`):
```env
ADMIN_CORS_ORIGINS=http://localhost:5173
CONSUMER_CORS_ORIGINS=http://localhost:3000
```

---

### 2 — Consumer Concierge Portal

The guest-facing multimodal voice portal. Guests click **Connect**, grant microphone permission, and speak naturally with Elena (or the active persona). Features:

- **Live Captions** — Real-time speech-to-text for both guest and agent
- **Room Cards** — Generative room recommendation cards pushed over DataChannel
- **Booking Stepper** — Real-time visual progress tracker (Dates → Room → Guest → Confirm)
- **Dual Input Bar** — Type messages when speaking is inconvenient
- **Push-to-Talk** — Optional PTT mode for noisy environments

```powershell
cd consumer_ui
npm run dev       # Dev server at http://localhost:3000
npm run build     # Production build to dist/
```

**URL parameters:**
- `?tenant=grand-meridian` — Select tenant (default: `grand-meridian`)
- `?embed=1` — Embedded mode (hides header, compact layout)

**Embedded widget** — After building, add this to any hotel website:
```html
<script
  src="https://your-deployment.com/voice-widget.js"
  data-tenant="grand-meridian"
  data-app-url="https://your-deployment.com/"
  defer>
</script>
```

**Vite dev proxy** is pre-configured to forward `/api/*` to `http://localhost:8000` — no CORS issues during development.

---

### 3 — Admin Operations Console

The enterprise admin dashboard for hotel operations staff. Features:

- **Dashboard** — Live call metrics, active sessions, system health
- **Voice Studio** — Persona management, voice settings, prompt editing
- **Provider Settings** — Hot-swap LLM/STT/TTS providers at runtime
- **Knowledge Hub** — Upload and manage RAG knowledge documents

```powershell
cd admin_panel
npm run dev       # Dev server at http://localhost:5173
npm run build     # Production build to dist/
```

---

## Running the Test Suite

The test suite uses the standard library `unittest` — no pytest required.

```powershell
cd Smart-AI-Voice-Agent

# Run all tests (103 tests)
python -m unittest discover -s backend/tests -p "test_*.py" -v

# Run a specific test module
python -m unittest backend.tests.test_voice_sessions -v
python -m unittest backend.tests.test_mcp -v
python -m unittest backend.tests.test_rag -v
python -m unittest backend.tests.test_factories -v
python -m unittest backend.tests.test_voice_engine -v
```

**Expected output:**
```
Ran 103 tests in ~40s
OK
```

> **Note:** Deprecation warnings about Kokoro/Whisper missing extras are expected and safe — the factories gracefully stub both services. All 103 tests pass regardless of which extras are installed.

**Test coverage by module:**

| Module | Tests | What is Tested |
|---|---|---|
| `test_factories.py` | 10 | LLM/STT/TTS provider factory, settings, tenant YAML loading |
| `test_mcp.py` | 26 | Hotel booking tools, confirmation gate, dining, ops tickets, audit log |
| `test_rag.py` | 18 | Chunker, vector store, retriever, embeddings, TF-IDF fallback |
| `test_voice_engine.py` | 24 | Normalizer, voice catalog, voice controller, verbal commands |
| `test_admin_api.py` + `test_admin_runtime.py` | 8 | Admin REST API, provider runtime store |
| `test_voice_sessions.py` | 14 | WebRTC sessions, DataChannel emitters, pipeline context injection |
| `test_pipeline_integration.py` | 5 | AGENTS.md §7 runtime verification: live pipeline startup, typed text roundtrip, spoken utterance confirmation gate, local LLM roundtrip |

---

## Environment Variables Reference

Copy `.env.example` to `.env` and edit as needed.

### AI Provider Selection

```env
# LLM backend: ollama (local, default) | gemini | groq | openai
LLM_PROVIDER=ollama

# Speech-to-Text: faster_whisper (local, default) | deepgram | groq
STT_PROVIDER=faster_whisper

# Text-to-Speech: kokoro (local, default) | piper | cartesia | elevenlabs
TTS_PROVIDER=kokoro
```

### Local Providers (Default — No API Keys)

```env
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434/v1

FASTER_WHISPER_MODEL=base.en      # Options: tiny.en, base.en, small.en, medium.en
KOKORO_VOICE_ID=af_sarah          # Elena's voice
KOKORO_SPEED=1.00                  # Range: 0.70 – 1.40
```

### Cloud Providers (Optional)

```env
# Google Gemini
GEMINI_API_KEY=your-key-here

# OpenAI / Groq (LLM or STT)
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-4o-mini
GROQ_API_KEY=your-key-here
GROQ_MODEL=llama-3.3-70b-versatile

# Cloud STT
DEEPGRAM_API_KEY=your-key-here

# Cloud TTS
CARTESIA_API_KEY=your-key-here
CARTESIA_VOICE_ID=86e30c1d-714b-4074-a1f2-1cb6b552fb49
```

### Server & Database

```env
HOST=0.0.0.0
PORT=8000
DEBUG=true
DATABASE_URL=sqlite:///./data/voice_agent.db
CHROMA_PERSIST_DIR=./data/chroma_db
LOG_LEVEL=INFO

ADMIN_CORS_ORIGINS=http://localhost:5173
CONSUMER_CORS_ORIGINS=http://localhost:3000
```

### RAG & Weather

```env
# Embeddings: local_hash (default, no download) | nomic_ollama | fastembed
EMBEDDING_PROVIDER=local_hash
EMBEDDING_MODEL=nomic-embed-text

# Weather: stub (default) | open_meteo (free, no key)
WEATHER_PROVIDER=stub
```

---

## Provider Hot-Swap Guide

You can switch AI providers **without touching any code** — just edit `.env` and restart the backend.

### Switch to Groq (fast cloud LLM + STT)

```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your-key-here
GROQ_MODEL=llama-3.3-70b-versatile

STT_PROVIDER=groq
# GROQ_API_KEY is shared for both LLM and STT
```

### Switch to OpenAI

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
```

### Switch to Cartesia TTS (hyper-realistic voice)

```env
TTS_PROVIDER=cartesia
CARTESIA_API_KEY=your-key-here
CARTESIA_VOICE_ID=86e30c1d-714b-4074-a1f2-1cb6b552fb49
```

### Switch to Deepgram STT (real-time streaming)

```env
STT_PROVIDER=deepgram
DEEPGRAM_API_KEY=your-key-here
```

> **Note:** Providers can also be changed at runtime via the Admin Console **Provider Settings** page without a server restart.

---

## Multi-Tenant Configuration

The active tenant is defined in `backend/config/tenants/default_hotel.yaml`. The default tenant is **Grand Meridian Hotel & Resort** with five specialist personas:

| Persona Key | Agent Name | Tools | Capability |
|---|---|---|---|
| `hotel_booking_specialist` | **Elena** | check_availability, book_room, cancel_booking, get_amenities | Full booking lifecycle |
| `receptionist` | **Elena** | check_availability, get_amenities, weather, ops_ticket | Front desk info |
| `restaurant_dining` | **Marco** | list_menu, place_order, check_order, cancel_order | Room service |
| `travel_desk` | **Priya** | weather, calendar_list, calendar_create, messaging_send | Travel & scheduling |
| `customer_care` | **Jordan** | ops_ticket, list_tickets, messaging_send, get_amenities | Guest care |

**To customize a persona**, edit `backend/config/tenants/default_hotel.yaml`:

```yaml
personas:
  hotel_booking_specialist:
    name: "Sofia"                        # Agent display name
    system_prompt: |
      You are Sofia, concierge at The Ritz...
    voice:
      provider: "cartesia"
      voice_id: "your-voice-id"
    llm:
      provider: "openai"
      model: "gpt-4o-mini"
```

---

## Project Structure

```
Smart-AI-Voice-Agent/
│
├── backend/                        # Python backend
│   ├── api/
│   │   ├── app.py                  # FastAPI application factory
│   │   ├── voice_sessions.py       # WebRTC session manager + pipeline lifecycle
│   │   ├── schemas.py              # Pydantic v2 request/response schemas
│   │   ├── routes/
│   │   │   ├── voice.py            # WebRTC SDP/ICE endpoints
│   │   │   └── admin.py            # Admin REST API routes
│   │   ├── audition.py             # Voice provider audition endpoint
│   │   ├── knowledge.py            # Knowledge document management
│   │   └── runtime.py              # Runtime provider hot-swap store
│   │
│   ├── config/
│   │   ├── settings.py             # Pydantic v2 AppSettings (reads .env)
│   │   └── tenants/
│   │       └── default_hotel.yaml  # Grand Meridian tenant & persona config
│   │
│   ├── engine/
│   │   ├── pipeline.py             # 🔴 CORE: Pipecat voice pipeline + DataChannel bridge
│   │   ├── audio/
│   │   │   ├── normalizer.py       # Speech text normalizer (currency, IDs, dates)
│   │   │   ├── voice_catalog.py    # Voice profile catalog (Kokoro, Cartesia, etc.)
│   │   │   └── voice_controller.py # Dynamic in-call acoustic controller
│   │   └── providers/
│   │       ├── llm_factory.py      # LLM service factory (Ollama/Gemini/Groq/OpenAI)
│   │       ├── stt_factory.py      # STT service factory (Whisper/Deepgram/Groq)
│   │       └── tts_factory.py      # TTS service factory (Kokoro/Cartesia/ElevenLabs)
│   │
│   ├── mcp/
│   │   ├── client.py               # MCPHub: tool dispatch with persona isolation
│   │   ├── tools.py                # 6 tool servers (hotel, dining, calendar, etc.)
│   │   ├── store.py                # SQLite persistence (rooms, bookings, orders)
│   │   ├── session.py              # Per-call agent session state
│   │   ├── confirmation.py         # Out-loud confirmation gate logic
│   │   ├── registry.py             # Persona → tool allow-lists (max 4 per persona)
│   │   └── schemas.py              # Pydantic tool argument schemas
│   │
│   ├── rag/
│   │   ├── vector_store.py         # ChromaDB + Memory TF-IDF fallback store
│   │   ├── retriever.py            # Top-K retriever (K≤2, similarity≥0.72)
│   │   ├── chunker.py              # Voice-optimised conversation chunker
│   │   ├── ingestion.py            # Document ingestion pipeline
│   │   └── embeddings.py           # Local hash / Ollama / FastEmbed providers
│   │
│   └── tests/
│       ├── test_factories.py
│       ├── test_mcp.py
│       ├── test_rag.py
│       ├── test_voice_engine.py
│       ├── test_admin_api.py
│       ├── test_admin_runtime.py
│       └── test_voice_sessions.py
│
├── consumer_ui/                    # React 19 guest-facing portal (Port 3000)
│   └── src/
│       ├── App.tsx                 # Root app: session + event routing
│       ├── hooks/
│       │   └── useVoiceSession.ts  # WebRTC + DataChannel session hook
│       ├── components/
│       │   ├── AudioVisualizer.tsx # Microphone audio bar visualizer
│       │   ├── LiveCaptions.tsx    # Real-time speech captions
│       │   ├── DualInputBar.tsx    # Voice PTT + text input bar
│       │   └── generative/
│       │       ├── RoomCard.tsx    # Generative room recommendation card
│       │       ├── BookingStepper.tsx # Booking progress tracker
│       │       └── CheckoutModal.tsx  # Secure checkout modal
│       ├── types.ts                # Shared TypeScript types (server events, models)
│       └── api.ts                  # Fetch helpers for configuration endpoint
│
├── admin_panel/                    # React 19 admin console (Port 5173)
│   └── src/
│       └── pages/
│           ├── Dashboard.tsx       # Call metrics, session monitor
│           ├── VoiceStudio.tsx     # Persona editor, voice settings
│           ├── ProviderSettings.tsx # Runtime provider hot-swap
│           └── KnowledgeHub.tsx    # RAG document management
│
├── data/                           # Runtime data (gitignored)
│   ├── voice_agent.db              # SQLite — bookings, rooms, tickets, orders
│   └── chroma_db/                  # ChromaDB vector embeddings
│
├── .env                            # Local environment config (gitignored)
├── .env.example                    # Environment template
├── requirements.txt                # Python dependencies
├── AGENTS.md                       # Agent governance and approval gates
└── GEMINI.md                       # Project architecture rules
```

---

## Development Workflow

### Committing changes

This project follows the **Anti-Stub Rule**: never commit a milestone that declares generative UI events "done" without an active backend pipeline producer wired end-to-end. See `AGENTS.md` for full governance rules.

Commit message format (imperative mood):
```
feat(pipeline): attach Pipecat worker to SmallWebRTC connection on offer
fix(mcp): correctly gate hotel_book_room behind needs_confirmation status
test(voice_sessions): add DataChannel bridge caption emission assertions
```

### Frontend type checking

```powershell
cd consumer_ui && npm run lint    # TypeScript strict check
cd admin_panel && npm run lint
```

### Backend imports run cleanly without extras

```powershell
python -c "from backend.engine.pipeline import create_consumer_voice_pipeline; print('OK')"
```

---

## Security Notes

- **No card data over voice** — The platform never asks callers to speak credit card numbers. Generative UI uses `checkout_unavailable` unless a secure provider is configured.
- **All booking writes are gated** — `hotel_book_room`, `hotel_cancel_booking`, `dining_place_order`, and `calendar_create_event` all require an explicit verbal "yes" / "confirm" from the caller before persisting to SQLite.
- **Screen taps are context only** — Clicking a room card or booking step sends a `client_touch_override` event that injects conversational context into the LLM — it never triggers a database write directly.
- **API keys never leave the server** — The consumer bootstrap endpoint (`/api/v1/voice/config`) returns only safe metadata. No credentials are exposed to the browser bundle.
- **Parameterized SQL** — All SQLite writes use parameterized queries via the `HotelStore` connection manager.

---

## Supported Ollama Models

Any model served by Ollama works. Recommended options:

| Model | Size | Best For |
|---|---|---|
| `llama3.2:3b` | 2 GB | Development, low-RAM machines (default) |
| `llama3.2:8b` | 5 GB | Better accuracy, mid-range machines |
| `llama3.1:8b` | 5 GB | Strong tool-calling performance |
| `mistral:7b` | 4 GB | Fast responses, good instruction following |
| `qwen2.5:7b` | 5 GB | Excellent for structured JSON / tool calls |

Change the model in `.env`:
```env
OLLAMA_MODEL=llama3.1:8b
```

---

*Grand Meridian Hotel & Resort — Powered by Smart AI Voice Agent Platform*
