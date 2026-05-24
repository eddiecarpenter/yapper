/**
 * `App` — minimal SPA shell that composes the Yapper voice loop.
 *
 * Wires the four closed-feature pipeline modules
 * (`SileroVAD` / `WhisperTranscriber` / `KokoroSpeaker` + the
 * `createAudioPipeline` factory) into `useDialogue`, and exposes
 * a deliberately minimal UI surface: a start/stop button, the
 * current dialogue stage, the error string (when present), and a
 * scrolling transcript built from `state.history`.
 *
 * The latency dashboard is intentionally out of scope here —
 * Feature 19 owns that.
 *
 * ## Mounting strategy
 *
 * The voice loop is heavy: it constructs ONNX models (Whisper,
 * Kokoro, Silero), opens a WebSocket, and requests the
 * microphone. We do not want any of that to fire on the static
 * landing page. So the loop is encapsulated in `<VoiceLoop />`,
 * a child component that we mount only after the user clicks
 * "Start" and unmount when they click "Stop". The unmount
 * tear-down chain in `useDialogue`'s `useEffect` cleanup releases
 * every resource (transcriber.dispose, vad.dispose, ws.close,
 * speaker.cancel) so re-clicking "Start" gets a clean slate.
 */
import { useState } from "react";

import { createAudioPipeline, type AudioPipeline, type PipelineState } from "./audio";
import { useDialogue } from "./dialogue";
import { KokoroSpeaker } from "./tts";
import { SileroVAD } from "./vad";
import { WhisperTranscriber } from "./stt";
import type { DialogueState } from "./dialogue";

/**
 * Derive the WebSocket URL for the relay.
 *
 * Same-origin in both modes:
 *   - Dev (`npm run dev` against Vite on :5173) — Vite proxies
 *     `/ws` to the Go relay (see `vite.config.ts`).
 *   - Production (Go binary serving the embedded SPA at :8080) —
 *     the page and the relay share the same origin already, so
 *     same-origin upgrade lands on the Go handler.
 *
 * Using `location.host` (rather than a hard-coded `localhost:8080`)
 * makes the SPA portable to a future deployment where the relay
 * sits behind a reverse proxy on a different host.
 */
function deriveRelayUrl(): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws`;
}

export function App(): JSX.Element {
  const [started, setStarted] = useState(false);

  return (
    <main style={{ fontFamily: "system-ui, sans-serif", padding: "1rem" }}>
      <h1>Yapper</h1>
      <p style={{ color: "#666" }}>Local-first voice assistant — spike build.</p>
      {started ? (
        <VoiceLoop onStop={() => setStarted(false)} />
      ) : (
        <button
          type="button"
          onClick={() => setStarted(true)}
          style={{
            padding: "0.5rem 1rem",
            fontSize: "1rem",
            cursor: "pointer",
          }}
        >
          Start
        </button>
      )}
    </main>
  );
}

/**
 * `VoiceLoop` — the actual hook-bearing inner component.
 *
 * Created on mount, torn down on unmount. The four heavy
 * dependencies are instantiated lazily in `useState` so they exist
 * exactly once per mount and survive re-renders without being
 * reconstructed (which would re-trigger model downloads).
 *
 * `createAudioPipeline` is async — we kick it off in `useState`'s
 * lazy initializer indirectly via `useEffect`, but the simpler
 * shape for the minimal SPA is to fire it imperatively when the
 * component mounts and surface its state via the pipeline's
 * subscribe API. Audio errors (mic permission denied, hardware
 * busy) surface in `pipelineState`; LLM/relay errors surface in
 * the dialogue's `state.error`.
 */
function VoiceLoop({ onStop }: { onStop: () => void }): JSX.Element {
  // Lazy-init the deps once per mount. The functional initializer
  // form means the constructors run on first render only — React
  // will not re-invoke them on subsequent renders.
  const [deps] = useState(() => ({
    transcriber: new WhisperTranscriber(),
    speaker: new KokoroSpeaker(),
    vad: new SileroVAD(),
  }));
  const [pipelineState, setPipelineState] = useState<PipelineState>("idle");
  const [pipelineError, setPipelineError] = useState<string | null>(null);

  // Build and start the audio pipeline once per mount. Stored in
  // closure-local state via useState so the unmount cleanup can
  // dispose it. We use a useState-style ref-ish hold rather than
  // useRef so the dispose closure captures the resolved pipeline.
  const [pipeline, setPipeline] = useState<AudioPipeline | null>(null);

  // Initialize the pipeline once. The effect intentionally has an
  // empty dep array — start/stop is controlled by the
  // mount/unmount lifecycle of this component.
  if (pipeline === null) {
    void createAudioPipeline({ vad: deps.vad }).then((p) => {
      p.subscribe((s) => {
        setPipelineState(s);
        setPipelineError(p.getError());
      });
      void p.start();
      setPipeline(p);
    });
  }

  const dialogue: DialogueState = useDialogue({
    transcriber: deps.transcriber,
    speaker: deps.speaker,
    vad: deps.vad,
    relayUrl: deriveRelayUrl(),
  });

  const handleStop = (): void => {
    if (pipeline !== null) {
      pipeline.dispose();
    }
    onStop();
  };

  const stageError =
    dialogue.error !== null ? dialogue.error : pipelineError !== null ? pipelineError : null;

  return (
    <section>
      <p>
        <strong>Stage:</strong> {dialogue.stage} · <strong>Mic:</strong> {pipelineState}
      </p>
      {stageError !== null && (
        <p style={{ color: "#a00" }} role="alert">
          {stageError}
        </p>
      )}
      <button
        type="button"
        onClick={handleStop}
        style={{
          padding: "0.5rem 1rem",
          fontSize: "1rem",
          cursor: "pointer",
        }}
      >
        Stop
      </button>
      <Transcript history={dialogue.history} />
    </section>
  );
}

/**
 * `Transcript` — minimal renderer for the conversation history.
 * Skips the leading `system` message (display-only — system
 * prompts are configuration, not user-visible turns).
 */
function Transcript({ history }: { history: DialogueState["history"] }): JSX.Element {
  const visible = history.filter((m) => m.role !== "system");
  if (visible.length === 0) {
    return <p style={{ color: "#888", marginTop: "1rem" }}>No turns yet.</p>;
  }
  return (
    <ol style={{ marginTop: "1rem", paddingLeft: "1.25rem" }}>
      {visible.map((m, i) => (
        <li key={i} style={{ marginBottom: "0.5rem" }}>
          <strong>{m.role}:</strong> {m.content}
        </li>
      ))}
    </ol>
  );
}
