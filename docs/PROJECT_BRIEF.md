# Yapper — Project Brief

## Mission

Yapper is a local-first voice assistant spike, delivered as a **React web application** backed by a **Go relay server**. The user speaks; the browser transcribes with Whisper (ONNX Web + WebGPU), sends text to an LLM via the Go relay, and speaks the reply back with Kokoro (ONNX Web + WebGPU). All speech processing runs locally in the browser via hardware-accelerated model inference. The LLM defaults to a local Ollama instance for a fully offline loop, but is configurable to any cloud provider.

**Module path:** `github.com/eddiecarpenter/yapper`

**Spike goal:** Validate the feasibility of full generative STT + TTS for voice-interactive applications — specifically whether latency, quality, and offline capability are viable alternatives to pre-recorded phrase sets (e.g. ElevenLabs' constrained vocabulary approach).

---

## Scope

### Phase 1 — MVP (build first)

A working turn-based voice loop in the browser. The React SPA handles the full voice pipeline (mic capture → VAD → STT → LLM relay → TTS → playback). The Go server relays LLM calls and serves the SPA. Maintain multi-turn conversation history so context carries across turns.

### Phase 2 — Latency reduction (only after Phase 1 is approved)

Stream LLM output tokens from the Go relay to the browser, chunk on sentence boundaries, and feed each chunk to Kokoro so playback starts before generation finishes. Optionally add barge-in. The Phase 1 interfaces must already allow for streaming so this is additive, not a rewrite.

---

## Architecture Split

| Concern | Where it lives | Technology |
|---|---|---|
| Microphone capture | Browser | Web Audio API (`getUserMedia`) |
| VAD endpointing | Browser | Silero ONNX Web |
| STT | Browser | Whisper ONNX Web + WebGPU (Transformers.js) |
| TTS + playback | Browser | Kokoro ONNX Web + WebGPU |
| Conversation loop + history | Browser | React state / custom hook |
| LLM relay | Go server | Ported from ocs-testbench (OpenAI-compat + Anthropic adapters) |
| SPA serving | Go server | Embedded `net/http` static file server |
| Config + credentials | Go server | YAML + env vars |
| Observability | Both | Browser Performance API; server `log/slog` |

---

## Technology Choices

| Component | Choice | Rationale |
|---|---|---|
| STT model | Whisper ONNX (base.en) | Same model as native path; runs in-browser via ONNX Web |
| TTS model | Kokoro v1.0 ONNX | Same model as native path; runs in-browser via ONNX Web |
| Browser ML runtime | ONNX Web + WebGPU | Hardware-accelerated inference; Metal via WebGPU on Apple Silicon |
| Browser ML library | Transformers.js (Hugging Face) | Production-ready Whisper + Kokoro support; ONNX Web backed |
| VAD | Silero (ONNX Web) | Same model family as native; gates transcription on silence |
| Frontend | React + TypeScript | Portable to AegrosAI's environment; Tauri-compatible |
| LLM relay | Go + ocs-testbench adapters | Proven implementation; handles Ollama + cloud providers |
| LLM default | Ollama (`llama3.2:3b`) | Fully offline; OpenAI-compatible endpoint |
| Logging | `log/slog` (server) + Browser Performance API | Structured; per-stage timing |

---

## Deployment Profiles

| Profile | How launched | LLM | STT/TTS |
|---|---|---|---|
| **Local dev** | `yapper serve` → open `http://localhost:PORT` | Ollama (offline) | ONNX Web + WebGPU |
| **Hosted SPA** | `https://yapper.example.com` | Cloud (OpenAI / Anthropic) | ONNX Web + WebGPU |
| **Tauri desktop** | Native app wrapping the React SPA | Ollama (offline) or cloud | ONNX Web + WebGPU |

---

## Package Layout

```
yapper/
├── cmd/yapper/         Go server entrypoint (serve + future subcommands)
├── internal/
│   ├── api/            HTTP server — serves SPA, WebSocket/SSE LLM relay
│   ├── llm/            LLM adapters (ported from ocs-testbench)
│   └── config/         Config loading and validation
├── web/                React SPA (TypeScript)
│   ├── src/
│   │   ├── audio/      Web Audio API capture + playback
│   │   ├── vad/        Silero VAD (ONNX Web)
│   │   ├── stt/        Whisper ONNX Web + WebGPU
│   │   ├── tts/        Kokoro ONNX Web + WebGPU
│   │   ├── llm/        WebSocket/SSE client → Go relay
│   │   └── dialogue/   Conversation loop + history hook
│   └── public/
├── docs/
│   ├── PROJECT_BRIEF.md
│   └── ARCHITECTURE.md
├── go.mod
├── go.sum
└── README.md
```

---

## Confirmed Decisions

| # | Decision | Resolution |
|---|---|---|
| 1 | Voice pipeline location | ✅ Browser — ONNX Web + WebGPU |
| 2 | STT model | ✅ Whisper ONNX base.en (Transformers.js) |
| 3 | TTS model | ✅ Kokoro v1.0 ONNX (Transformers.js); voice `bf_emma` |
| 4 | LLM relay | ✅ Go server; ocs-testbench adapters (OpenAI-compat + Anthropic) |
| 5 | Default LLM | ✅ Ollama `llama3.2:3b` — offline, no API key |
| 6 | Turn-taking | ✅ Automatic VAD endpointing (Silero ONNX Web) |
| 7 | Frontend | ✅ React + TypeScript |
| 8 | OS TTS MCP service | ✅ Parked — good idea, out of scope for this spike |
| 9 | Phase 2 (streaming + barge-in) | ✅ Out of scope until Phase 1 signed off |

---

## Deliverables

1. React SPA with working voice loop (mic → VAD → STT → LLM → TTS → playback)
2. Go relay server serving the SPA and streaming LLM responses
3. README (UK English) — prerequisites, build/run, config, model licences
4. **Latency report** — measured per-stage and end-to-end on Apple Silicon M2; comparison baseline against constrained phrase-set approach
5. TypeScript tests for STT (known audio clip), TTS (sample rate check), LLM relay (stub server)

---

## Acceptance Criteria

- Opening `http://localhost:PORT` in a browser starts the voice loop
- User speaks → transcribed → LLM replies → reply is spoken back, context held across ≥ 3 turns
- Fully offline after initial model download (Ollama running, no internet required)
- Switching LLM provider is a config/env-var change only; default requires no API keys
- Per-stage and end-to-end turn latency is measured, logged, and reported

---

## Non-Goals (this spike)

- OS TTS MCP server (parked for future standalone utility)
- Native Go audio pipeline (malgo, sherpa-onnx Go bindings)
- Voice cloning, speaker diarisation, translation
- Tool-calling / function execution
- Wake-word detection
- Production hardening, containerisation, web UI polish

---

## Licences

| Component | Licence |
|---|---|
| Whisper | MIT |
| Kokoro | Apache 2.0 |
| Transformers.js | Apache 2.0 |
| ONNX Web Runtime | MIT |
| React | MIT |
