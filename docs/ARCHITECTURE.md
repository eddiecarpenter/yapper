# Yapper — Foundation Solution Architecture

**Status:** Draft — spike phase
**Stack:** React + TypeScript (SPA) · Go (relay server)
**Platform:** macOS arm64 (Apple Silicon M2) — primary spike target
**Last updated:** 2026-05-21

---

## 1. Vision

Yapper is a local-first voice assistant spike delivered as a browser-based React application. All speech processing (STT, TTS, VAD) runs locally in the browser via ONNX Web and WebGPU — hardware-accelerated model inference with no server round-trips for audio. The Go relay server handles LLM calls and serves the SPA. The default configuration runs fully offline with Ollama.

The spike validates the feasibility of full generative STT + TTS as a replacement for pre-recorded phrase sets — measuring latency, quality, and offline viability on real hardware.

---

## 2. System Context

```
┌─────────────────────────────────────────────────────────┐
│                      Browser                             │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │  audio/  │  │  vad/    │  │  stt/    │  │  tts/  │  │
│  │getUserMe-│─▶│ Silero   │─▶│ Whisper  │  │ Kokoro │  │
│  │  dia()   │  │ ONNX Web │  │ ONNX Web │  │ONNX Web│  │
│  └──────────┘  └──────────┘  └──────────┘  └───▲────┘  │
│                                      │           │       │
│                               ┌──────▼───────────┴────┐ │
│                               │      dialogue/         │ │
│                               │  conversation loop     │ │
│                               │  history · React hook  │ │
│                               └──────────┬─────────────┘ │
│                                          │ WebSocket/SSE  │
└──────────────────────────────────────────┼───────────────┘
                                           │
┌──────────────────────────────────────────▼───────────────┐
│                    Go relay server                         │
│                                                           │
│  ┌───────────┐  ┌────────────────────────────────────┐   │
│  │  api/     │  │  llm/                              │   │
│  │ HTTP +    │  │  OpenAI-compat adapter (Ollama,    │   │
│  │ WebSocket │  │  OpenAI, Groq, …)                  │   │
│  │ SPA serve │  │  Anthropic native adapter          │   │
│  └───────────┘  │  (ported from ocs-testbench)       │   │
│                 └───────────────────┬──────────────────┘  │
│  ┌───────────┐                      │                     │
│  │  config/  │                      │                     │
│  └───────────┘                      │                     │
└─────────────────────────────────────┼─────────────────────┘
                                      │
               ┌──────────────────────┴──────────────────────┐
               │                                              │
    ┌──────────▼──────────┐                    ┌─────────────▼───────┐
    │  Ollama (local)     │                    │  Cloud LLM          │
    │  localhost:11434    │                    │  OpenAI / Anthropic │
    │  llama3.2:3b        │                    │  (config-driven)    │
    └─────────────────────┘                    └─────────────────────┘
```

---

## 3. Dialogue Loop

```
getUserMedia() — microphone
       │ float32 PCM (48 kHz)
       ▼
Silero VAD (ONNX Web)
       │ speech segment detected
       ▼
Whisper STT (ONNX Web + WebGPU)
       │ text
       ▼
WebSocket → Go relay → LLM
       │ reply text (streaming tokens)
       ▼
Kokoro TTS (ONNX Web + WebGPU)
       │ float32 PCM (24 kHz)
       ▼
Web Audio API playback
       │
       ▼
back to listening
```

Phase 1: sequential (full utterance → full transcription → full LLM reply → full synthesis → playback).
Phase 2 (deferred): stream LLM tokens → chunk on sentence boundaries → feed Kokoro incrementally.

---

## 4. Package Layout

```
yapper/
├── cmd/
│   └── yapper/             Go entrypoint — `serve` subcommand
├── internal/
│   ├── api/                HTTP server: static SPA serving + WebSocket/SSE LLM relay
│   ├── llm/                LLM adapters (ported from ocs-testbench)
│   │   ├── client.go       LLMClient interface + OpenAI-compat adapter
│   │   ├── anthropic.go    Anthropic native adapter
│   │   └── types.go        Message, CompletionRequest, CompletionResponse, Usage
│   └── config/             Config loading and validation
├── web/                    React SPA (TypeScript + Vite)
│   ├── src/
│   │   ├── audio/          getUserMedia wrapper, Web Audio API playback
│   │   ├── vad/            Silero VAD (ONNX Web) — gates STT on speech
│   │   ├── stt/            Whisper ONNX Web + WebGPU (Transformers.js)
│   │   ├── tts/            Kokoro ONNX Web + WebGPU (Transformers.js)
│   │   ├── llm/            WebSocket / SSE client → Go relay
│   │   ├── dialogue/       useDialogue hook — conversation loop + history
│   │   └── App.tsx         Root component
│   ├── public/
│   └── vite.config.ts
├── docs/
├── go.mod
├── go.sum
└── README.md
```

---

## 5. Core Interfaces

### 5.1 Go — LLM relay (ported from ocs-testbench)

```go
// internal/llm/client.go
type LLMClient interface {
    Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error)
    CompleteStream(ctx context.Context, req CompletionRequest,
        onToken func(string), onUsage func(Usage)) (*CompletionResponse, error)
}

type Message struct {
    Role    string `json:"role"`    // "system" | "user" | "assistant"
    Content string `json:"content"`
}

type CompletionRequest struct {
    Model    string    `json:"model"`
    Messages []Message `json:"messages"`
    Stream   bool      `json:"stream"`
}
```

### 5.2 Go — WebSocket relay API

The relay exposes a single WebSocket endpoint. Messages are JSON:

```
// Browser → server
{ "type": "turn", "text": "What is the capital of France?" }

// Server → browser (streaming)
{ "type": "token", "text": "Paris" }
{ "type": "token", "text": " is" }
{ "type": "done",  "usage": { "input": 12, "output": 8 } }

// Server → browser (error)
{ "type": "error", "message": "Ollama unreachable at localhost:11434" }
```

### 5.3 TypeScript — voice pipeline interfaces

```typescript
// web/src/stt/types.ts
interface Transcriber {
  transcribe(audio: Float32Array, sampleRate: number): Promise<string>;
  dispose(): void;
}

// web/src/tts/types.ts
interface Speaker {
  speak(text: string): Promise<void>;   // resolves when playback complete
  cancel(): void;
}

// web/src/vad/types.ts
interface VAD {
  process(frame: Float32Array): boolean;  // true = speech detected
}
```

---

## 6. Component Detail

### 6.1 Browser — Audio capture (`web/src/audio/`)

- `navigator.mediaDevices.getUserMedia({ audio: true })` — microphone access
- Web Audio API `AudioWorkletProcessor` — real-time float32 PCM at device sample rate (typically 48 kHz)
- Resamples to 16 kHz before passing to Silero VAD and Whisper
- Playback: `AudioContext.decodeAudioData()` + `AudioBufferSourceNode` for Kokoro 24 kHz output

### 6.2 Browser — VAD (`web/src/vad/`)

- Silero VAD ONNX model via ONNX Web
- Processes 512-sample frames at 16 kHz
- Configurable speech/silence thresholds and minimum silence duration
- Emits speech segments to the STT stage; discards silence

### 6.3 Browser — STT (`web/src/stt/`)

- Whisper ONNX Web via Transformers.js (`@huggingface/transformers`)
- Default model: `Xenova/whisper-base.en` (English-only; ~145 MB; cached after first load)
- WebGPU execution provider — Metal-accelerated on Apple Silicon via browser WebGPU
- CPU fallback via WASM for non-WebGPU browsers
- Input: 16 kHz mono Float32Array

### 6.4 Browser — TTS (`web/src/tts/`)

- Kokoro v1.0 ONNX via Transformers.js (`@huggingface/transformers`)
- Default voice: `bf_emma` (British English female)
- WebGPU execution provider — Metal-accelerated on Apple Silicon
- Output: 24 kHz mono Float32Array → Web Audio API playback
- `Speaker.speak()` resolves when the `AudioBufferSourceNode` ends — clean turn sequencing
- `Speaker.cancel()` for Phase 2 barge-in

### 6.5 Browser — Dialogue loop (`web/src/dialogue/`)

- `useDialogue` React hook owns the turn-based loop
- Orchestrates: VAD trigger → STT → WebSocket send → token stream → TTS → back to VAD
- Conversation history: `Message[]` — maintained in React state; passed to Go relay with each turn
- Configurable context budget with sliding-window truncation
- Per-stage timing via `performance.now()`; emits total turn latency to console

### 6.6 Go — LLM relay (`internal/api/`, `internal/llm/`)

- `net/http` server with WebSocket upgrade (using `golang.org/x/net/websocket` or `nhooyr.io/websocket`)
- Receives turn messages from browser; passes history + new message to `LLMClient`
- Streams tokens back via WebSocket as they arrive (`onToken` callback from ocs-testbench adapter)
- Serves embedded React SPA via `//go:embed web/dist`
- Conversation history maintained server-side per session (keyed by session ID cookie)
- Credentials: env vars / `.env` only; masked in logs (`GetMaskedAPIKey()` pattern)

### 6.7 Go — Config (`internal/config/`)

```yaml
server:
  port: 8080
  session_ttl_minutes: 30

llm:
  provider: openai_compat        # openai_compat | anthropic
  base_url: "http://localhost:11434/v1"
  model: "llama3.2:3b"
  stream: true
  context_budget: 4096
  system_prompt: "You are Yapper, a helpful voice assistant. Be concise."

# Optional overrides via environment variables:
# YAPPER_LLM_BASE_URL, YAPPER_LLM_MODEL, YAPPER_LLM_API_KEY
```

---

## 7. Architectural Decisions

| # | Decision | Choice | Rationale |
|---|---|---|---|
| AD-1 | Voice pipeline location | Browser (ONNX Web + WebGPU) | Portable to any webview/Tauri/hosted deployment; no cgo in Go binary |
| AD-2 | Browser ML runtime | Transformers.js + ONNX Web | Production-ready; ships Whisper + Kokoro + Silero; WebGPU backend |
| AD-3 | LLM relay | Go server; ocs-testbench adapters ported | Proven implementation; handles streaming, credentials, Anthropic quirks |
| AD-4 | LLM default | Ollama `llama3.2:3b` via OpenAI-compat adapter | Fully offline; no API key; same adapter covers cloud providers |
| AD-5 | Frontend framework | React + TypeScript | Portable to AegrosAI environment; Tauri-compatible |
| AD-6 | SPA serving | Go `//go:embed` | Single binary deployment; no separate static server |
| AD-7 | WebSocket vs REST | WebSocket | Required for streaming token delivery without SSE complexity |
| AD-8 | Credentials | Env vars / `.env` only | Never committed; offline path requires no keys |
| AD-9 | OS TTS MCP server | Parked | Adds process management complexity; ONNX Web Kokoro is sufficient for the spike |
| AD-10 | Default TTS voice | `bf_emma` (British English female) | Consistent with UK English documentation style |
| AD-11 | STT model | `whisper-base.en` | ~145 MB; fast on WebGPU; English-only scope confirmed |

---

## 8. Deployment Profiles

| Profile | Launch | LLM | Notes |
|---|---|---|---|
| Local dev | `go run ./cmd/yapper serve` → `localhost:8080` | Ollama offline | Primary spike target |
| Hosted SPA | Deploy Go binary to server | Cloud (OpenAI/Anthropic) | `base_url` + API key in env |
| Tauri desktop | React SPA in Tauri webview | Ollama offline | Go binary as Tauri sidecar |

---

## 9. Non-Functional Requirements

| NFR | Requirement |
|---|---|
| Offline operation | Full loop works after initial model download (browser caches ONNX models); Ollama running locally |
| Apple Silicon | WebGPU → Metal for ONNX Web inference; WASM CPU fallback for non-WebGPU |
| Latency | Per-stage and end-to-end turn latency measured and reported — key spike deliverable |
| Single binary | `go build ./cmd/yapper` produces one binary; SPA embedded via `go:embed` |
| Credentials | Never logged, committed, or hard-coded |
| Error quality | Actionable browser console errors + server log errors for: Ollama unreachable, model not loaded, mic permission denied, WebGPU unavailable |
| Dependencies | Go: `go.mod` pinned + `go.sum` committed. Web: `package.json` locked via `package-lock.json` |

---

## 10. Evolution Seams

| Seam | What's left clean |
|---|---|
| Streaming TTS (Phase 2) | `Speaker.cancel()` defined; `dialogue/` loop can be extended to chunk LLM tokens |
| Barge-in (Phase 2) | VAD runs continuously; `Speaker.cancel()` provides the interrupt hook |
| OS TTS MCP server | `Speaker` interface in TypeScript allows a second implementation; parked |
| Tool-calling | `Message` type and relay protocol can carry tool call/result roles |
| Additional LLM providers | Implement `LLMClient` in Go; register in config |
| Tauri integration | React SPA is Tauri-compatible today; Go relay becomes a Tauri sidecar |

---

## 11. Open Decisions

All architectural decisions confirmed. No open decisions outstanding.
