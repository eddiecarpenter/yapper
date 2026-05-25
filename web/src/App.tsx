/**
 * `App` — minimal SPA shell with two modes:
 *
 * 1. **VAD mode** (original) — Silero VAD gates the mic continuously.
 * 2. **Push-to-talk mode** — hold the button to record, release to
 *    transcribe. Bypasses VAD entirely so we can test the audio →
 *    STT → LLM → TTS pipeline independently of VAD.
 *
 * PTT mode is a debug aid: it uses MicrophoneCapture directly,
 * accumulates frames while the button is held, then hands the
 * concatenated segment to WhisperTranscriber and the LLM WebSocket.
 */
import { useCallback, useEffect, useRef, useState } from "react";

import { MicrophoneCapture, TARGET_SAMPLE_RATE_HZ, FRAME_SAMPLE_COUNT } from "./audio";
import { KokoroSpeaker, type SupertonicVoiceId } from "./tts";
import { TextSplitterStream } from "./tts/splitter";
import { WhisperTranscriber, type LoadingState } from "./stt";
import type { DialogueState } from "./dialogue";
import { useDialogue } from "./dialogue";
import { createAudioPipeline, type AudioPipeline, type PipelineState } from "./audio";
import { SileroVAD } from "./vad";

function deriveRelayUrl(): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws`;
}

type Mode = "vad" | "ptt";

/** Supertonic-TTS voices — binary embeddings (1×101×128 Float32Array). */
const TTS_VOICES: Array<{ id: SupertonicVoiceId; label: string; note: string }> = [
  { id: "F1", label: "Female 1", note: "Supertonic F1" },
  { id: "F2", label: "Female 2", note: "Supertonic F2" },
  { id: "M1", label: "Male 1", note: "Supertonic M1" },
  { id: "M2", label: "Male 2", note: "Supertonic M2" },
];

/** Available Whisper models, ordered fastest → most accurate. */
const WHISPER_MODELS = [
  { id: "Xenova/whisper-tiny.en", label: "Tiny", note: "~150 MB · fastest, lower accuracy" },
  { id: "Xenova/whisper-base.en", label: "Base", note: "~290 MB · fast" },
  { id: "Xenova/whisper-small.en", label: "Small", note: "~490 MB · good balance (recommended)" },
  { id: "Xenova/whisper-medium.en", label: "Medium", note: "~1.5 GB · accurate, slow to download" },
  {
    id: "Xenova/whisper-large-v3-turbo",
    label: "Large v3 Turbo",
    note: "~1.6 GB · most accurate, slow to download",
  },
] as const;

export function App(): JSX.Element {
  const [started, setStarted] = useState(false);
  const [mode, setMode] = useState<Mode>("ptt");
  const [selectedWhisperModel, setSelectedWhisperModel] =
    useState<string>("Xenova/whisper-small.en");

  return (
    <main style={{ fontFamily: "system-ui, sans-serif", padding: "1rem" }}>
      <h1>Yapper</h1>
      <p style={{ color: "#666" }}>Local-first voice assistant — spike build.</p>
      {!started && (
        <>
          {/* Mode selector */}
          <div style={{ marginBottom: "1rem" }}>
            <label style={{ marginRight: "1rem" }}>
              <input
                type="radio"
                name="mode"
                value="ptt"
                checked={mode === "ptt"}
                onChange={() => setMode("ptt")}
              />{" "}
              Push-to-talk <span style={{ color: "#888", fontSize: "0.85em" }}>(recommended)</span>
            </label>
            <label>
              <input
                type="radio"
                name="mode"
                value="vad"
                checked={mode === "vad"}
                onChange={() => setMode("vad")}
              />{" "}
              VAD (auto)
            </label>
          </div>

          {/* Whisper model selector */}
          <div style={{ marginBottom: "1.25rem" }}>
            <p style={{ margin: "0 0 0.5rem", fontWeight: 600, fontSize: "0.9rem" }}>
              Speech recognition model
            </p>
            <div style={{ display: "flex", flexDirection: "column", gap: "0.35rem" }}>
              {WHISPER_MODELS.map((m) => (
                <label
                  key={m.id}
                  style={{
                    display: "flex",
                    alignItems: "baseline",
                    gap: "0.5rem",
                    cursor: "pointer",
                    fontSize: "0.9rem",
                  }}
                >
                  <input
                    type="radio"
                    name="whisper-model"
                    value={m.id}
                    checked={selectedWhisperModel === m.id}
                    onChange={() => setSelectedWhisperModel(m.id)}
                  />
                  <span style={{ fontWeight: selectedWhisperModel === m.id ? 600 : 400 }}>
                    {m.label}
                  </span>
                  <span style={{ color: "#888", fontSize: "0.8rem" }}>{m.note}</span>
                </label>
              ))}
            </div>
          </div>
        </>
      )}
      {started ? (
        mode === "ptt" ? (
          <PushToTalkLoop onStop={() => setStarted(false)} whisperModelId={selectedWhisperModel} />
        ) : (
          <VoiceLoop onStop={() => setStarted(false)} />
        )
      ) : (
        <button
          type="button"
          onClick={() => setStarted(true)}
          style={{ padding: "0.5rem 1rem", fontSize: "1rem", cursor: "pointer" }}
        >
          Start
        </button>
      )}
    </main>
  );
}

/* ── Model preload hook + loading bar ────────────────────────── */

/** Minimal interface shared by WhisperTranscriber and KokoroSpeaker. */
interface Loadable {
  subscribe(listener: (state: LoadingState) => void): () => void;
  getLoadingState(): LoadingState;
  getProvider(): string;
  preload(): void;
  /** Only WhisperTranscriber has download-progress — optional. */
  subscribeProgress?: (listener: (pct: number) => void) => () => void;
  getProgress?: () => number;
  /** Error message from the last failed load attempt — optional. */
  getError?: () => string | null;
}

function useModelLoader(model: Loadable): { loadingState: LoadingState; progress: number } {
  const [loadingState, setLoadingState] = useState<LoadingState>(() => model.getLoadingState());
  const [progress, setProgress] = useState<number>(() => model.getProgress?.() ?? 0);

  useEffect(() => {
    const unsubState = model.subscribe(setLoadingState);
    const unsubProgress = model.subscribeProgress?.(setProgress);
    model.preload();
    return () => {
      unsubState();
      unsubProgress?.();
    };
  }, [model]);

  return { loadingState, progress };
}

/**
 * Visual loading bar for any Loadable model. Shows a progress bar while
 * loading and a backend badge (WebGPU / WASM) once ready.
 */
function ModelLoader({ model, label }: { model: Loadable; label: string }): JSX.Element {
  const { loadingState, progress } = useModelLoader(model);
  const provider = model.getProvider();

  if (loadingState === "ready") {
    const badge =
      provider === "webgpu"
        ? { icon: "⚡", text: "WebGPU", bg: "#d4edda", fg: "#155724" }
        : provider === "browser"
          ? { icon: "🌐", text: "Browser TTS", bg: "#fff3cd", fg: "#856404" }
          : { icon: "🖥", text: "WASM", bg: "#e2e3e5", fg: "#383d41" };
    return (
      <p style={{ margin: "0.25rem 0", fontSize: "0.85rem", color: "#555" }}>
        ✅ {label} ready ·{" "}
        <span
          style={{
            display: "inline-block",
            padding: "1px 7px",
            borderRadius: "10px",
            fontSize: "0.8rem",
            fontWeight: 600,
            background: badge.bg,
            color: badge.fg,
          }}
        >
          {badge.icon} {badge.text}
        </span>
      </p>
    );
  }

  const isError = loadingState === "error";
  const errorDetail = model.getError?.() ?? null;
  const statusLabel = isError
    ? `Failed to load ${label}${errorDetail ? `: ${errorDetail}` : ""}`
    : progress > 0
      ? `Loading ${label}… ${progress}%`
      : `Loading ${label}…`;

  return (
    <div style={{ margin: "0.4rem 0" }}>
      <p style={{ margin: "0 0 0.3rem", fontSize: "0.85rem", color: isError ? "#a00" : "#555" }}>
        {statusLabel}
      </p>
      {!isError && (
        <div
          role="progressbar"
          aria-valuenow={progress}
          aria-valuemin={0}
          aria-valuemax={100}
          style={{
            height: "6px",
            borderRadius: "3px",
            background: "#e0e0e0",
            overflow: "hidden",
            width: "260px",
          }}
        >
          <div
            style={{
              height: "100%",
              borderRadius: "3px",
              background: "#007aff",
              width: progress > 0 ? `${progress}%` : "100%",
              animation: progress === 0 ? "indeterminate 1.4s infinite ease-in-out" : undefined,
              transition: progress > 0 ? "width 0.2s ease" : undefined,
            }}
          />
        </div>
      )}
      <style>{`
        @keyframes indeterminate {
          0%   { transform: translateX(-100%) scaleX(0.4); }
          50%  { transform: translateX(0%)    scaleX(0.7); }
          100% { transform: translateX(100%)  scaleX(0.4); }
        }
      `}</style>
    </div>
  );
}

/* ── Push-to-talk ─────────────────────────────────────────────── */

/** Per-turn timing breakdown, all values in milliseconds. */
type TurnTiming = {
  /** Time spent transcribing the audio segment. */
  sttMs: number;
  /** Time from sending the relay request to receiving the first token. */
  ttftMs: number;
  /** Total time from relay request sent to done frame received. */
  llmMs: number;
  /** Total time for TTS synthesis + playback. */
  ttsMs: number;
};

type HistoryMessage = {
  role: string;
  content: string;
  /** Present on assistant messages once the turn is complete. */
  timing?: TurnTiming;
};

function PushToTalkLoop({
  onStop,
  whisperModelId,
}: {
  onStop: () => void;
  whisperModelId: string;
}): JSX.Element {
  const [transcriber] = useState(() => new WhisperTranscriber(whisperModelId));
  const [speaker] = useState(() => new KokoroSpeaker());
  const [mic] = useState(() => new MicrophoneCapture());

  const [status, setStatus] = useState<string>("Initialising mic…");
  const [recording, setRecording] = useState(false);
  const [history, setHistory] = useState<HistoryMessage[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [noThinking, setNoThinking] = useState(true); // thinking off by default
  const [browserVoices, setBrowserVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [selectedVoice, setSelectedVoice] = useState<SpeechSynthesisVoice | null>(null);
  // Supertonic voice — only visible when ONNX pipeline is loaded.
  const [selectedTTSVoice, setSelectedTTSVoice] = useState<SupertonicVoiceId>(TTS_VOICES[0].id);
  const [ttsQuality, setTtsQuality] = useState(5); // num_inference_steps 1–50
  const [ttsSpeed, setTtsSpeed] = useState(1.0); // 0.8–1.2x

  // Both models must be ready before the button is enabled.
  const { loadingState: sttState } = useModelLoader(transcriber);
  const { loadingState: ttsState } = useModelLoader(speaker);
  const modelReady = sttState === "ready" && ttsState === "ready";

  // Populate voice list once the speaker is ready (voices may not be
  // available until after a user gesture or the voiceschanged event).
  useEffect(() => {
    if (ttsState !== "ready") return;
    const load = () => {
      const voices = KokoroSpeaker.getBrowserVoices();
      if (voices.length > 0) {
        setBrowserVoices(voices);
        setSelectedVoice((prev) => prev ?? voices[0] ?? null);
      }
    };
    load();
    window.speechSynthesis?.addEventListener("voiceschanged", load);
    return () => window.speechSynthesis?.removeEventListener("voiceschanged", load);
  }, [ttsState]);

  const framesRef = useRef<Float32Array[]>([]);
  const recordingRef = useRef(false);

  // Start mic on mount
  useEffect(() => {
    mic
      .start()
      .then(() => {
        setStatus("Ready — hold button to record");
        mic.onFrame = (frame) => {
          if (recordingRef.current) {
            framesRef.current.push(frame.slice()); // copy — worklet reuses buffer
          }
        };
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : String(err));
        setStatus("Mic error");
      });

    return () => {
      mic.dispose();
      transcriber.dispose();
      speaker.dispose();
    };
  }, [mic, transcriber, speaker]);

  // Stable ref to history so stopRecording always reads the latest value
  // without needing it in the useCallback dependency array.
  const historyRef = useRef<HistoryMessage[]>([]);

  // Keep the speaker's voice in sync with UI selection.
  useEffect(() => {
    speaker.setBrowserVoice(selectedVoice);
  }, [speaker, selectedVoice]);

  // Keep Supertonic voice in sync with the voice selector.
  useEffect(() => {
    speaker.setVoice(selectedTTSVoice);
  }, [speaker, selectedTTSVoice]);

  useEffect(() => {
    speaker.setNumInferenceSteps(ttsQuality);
  }, [speaker, ttsQuality]);
  useEffect(() => {
    speaker.setSpeed(ttsSpeed);
  }, [speaker, ttsSpeed]);

  /** Phrases that trigger the voice-clear command (case-insensitive, full utterance). */
  const CLEAR_COMMAND =
    /^(clear( (session|history|chat|conversation|everything))?|reset( (session|history|chat|conversation))?|start (over|a new (session|chat|conversation))|new (chat|session|conversation)|forget (everything|that|it all))\.?[!?]?$/i;

  const clearHistory = useCallback(() => {
    speaker.cancel();
    setHistory([]);
    historyRef.current = [];
    setStatus("Ready — hold button to record");
    setError(null);
  }, [speaker]);

  const startRecording = useCallback(() => {
    // Cancel any TTS that's currently playing so the user can interrupt.
    speaker.cancel();
    // Warm the AudioContext while we're inside a direct user-gesture
    // handler — browsers require this to transition from suspended→running.
    // By the time speak() fires (after STT + LLM), the gesture is gone.
    speaker.warmAudio();
    framesRef.current = [];
    recordingRef.current = true;
    setRecording(true);
    setStatus("Recording… release to transcribe");
    setError(null);
  }, [speaker]);

  const stopRecording = useCallback(async () => {
    recordingRef.current = false;
    setRecording(false);

    const frames = framesRef.current;
    framesRef.current = [];

    if (frames.length === 0) {
      setStatus("No audio captured — try holding longer");
      return;
    }

    // Concatenate frames into one Float32Array
    const totalSamples = frames.length * FRAME_SAMPLE_COUNT;
    const audio = new Float32Array(totalSamples);
    frames.forEach((f, i) => audio.set(f, i * FRAME_SAMPLE_COUNT));

    try {
      const sttStart = performance.now();
      setStatus(`Transcribing ${(totalSamples / TARGET_SAMPLE_RATE_HZ).toFixed(1)}s of audio…`);
      const text = await transcriber.transcribe(audio, TARGET_SAMPLE_RATE_HZ);
      const sttMs = Math.round(performance.now() - sttStart);

      if (!text.trim()) {
        setStatus("No speech detected — try speaking more clearly");
        return;
      }

      // ── Clear-session voice command ────────────────────────────
      if (CLEAR_COMMAND.test(text.trim())) {
        setStatus("Speaking…");
        speaker.warmAudio();
        clearHistory();
        void speaker
          .speak("Session cleared.")
          .then(() => {
            setStatus("✅ Done — hold to talk again");
          })
          .catch((err: unknown) => {
            setError(`TTS error: ${err instanceof Error ? err.message : String(err)}`);
            setStatus("✅ Done — hold to talk again");
          });
        return;
      }

      // ── Step 1 done: add user turn and show it immediately ─────
      const userMsg: HistoryMessage = { role: "user", content: text };
      setHistory((h) => {
        const next = [...h, userMsg];
        historyRef.current = next;
        return next;
      });

      // ── Step 2: send to relay and stream the LLM response ──────
      setStatus("Thinking…");
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      const ws = new WebSocket(`${proto}//${window.location.host}/ws`);

      await new Promise<void>((resolve, reject) => {
        let assistantText = "";
        let llmStart = 0;
        let llmMs = 0;
        let ttftMs = 0;
        let firstToken = true;

        // ── Sentence-streaming TTS ─────────────────────────────────
        // Tokens from the LLM are fed into a TextSplitterStream that
        // detects real sentence boundaries (handling abbreviations,
        // URLs, quotes, etc.). Complete sentences are merged until we
        // have ≥ 100 chars — matching the Supertonic demo's chunk
        // sizing — before dispatching to the TTS pipeline. This keeps
        // chunks large enough for good prosody while still starting
        // playback well before the LLM finishes.
        const splitter = new TextSplitterStream();
        let mergeBuffer = ""; // complete sentences waiting to merge
        const MIN_CHUNK_CHARS = 100;
        let ttsChain = Promise.resolve() as Promise<void>;
        let ttsStartMs = 0;
        let ttsStarted = false;

        const enqueueChunk = (chunk: string) => {
          const s = chunk.trim();
          if (!s) return;
          if (!ttsStarted) {
            ttsStarted = true;
            ttsStartMs = performance.now();
            setStatus("Speaking…");
          }
          ttsChain = ttsChain
            .then(() => {
              if (recordingRef.current) return;
              return speaker.speak(s);
            })
            .catch((ttsErr: unknown) => {
              const msg = ttsErr instanceof Error ? ttsErr.message : String(ttsErr);
              console.error("[TTS] speak() failed:", ttsErr);
              setError(`TTS error: ${msg}`);
            });
        };

        // Drain completed sentences from the splitter into mergeBuffer,
        // then dispatch once we reach the minimum chunk size.
        const flushBuffer = (forceAll = false) => {
          // Pull any newly completed sentences out of the splitter.
          while (splitter.sentences.length > 0) {
            const s = splitter.sentences.shift()!;
            mergeBuffer += (mergeBuffer ? " " : "") + s;
          }
          // Dispatch accumulated text when large enough (or forced at end).
          while (mergeBuffer.length >= MIN_CHUNK_CHARS) {
            enqueueChunk(mergeBuffer);
            mergeBuffer = "";
          }
          if (forceAll && mergeBuffer.trim()) {
            enqueueChunk(mergeBuffer);
            mergeBuffer = "";
          }
        };
        // ──────────────────────────────────────────────────────────

        ws.onopen = () => {
          llmStart = performance.now();
          const req = {
            type: "turn",
            text,
            no_thinking: noThinking,
            history: historyRef.current.map((m) => ({
              role: m.role as "user" | "assistant" | "system",
              content: m.content,
            })),
          };
          ws.send(JSON.stringify(req));
        };

        ws.onmessage = (ev: MessageEvent<string>) => {
          let frame: { type: string; text?: string; message?: string };
          try {
            frame = JSON.parse(ev.data) as typeof frame;
          } catch {
            return;
          }

          if (frame.type === "token" && typeof frame.text === "string") {
            if (firstToken) {
              ttftMs = Math.round(performance.now() - llmStart);
              firstToken = false;
            }
            assistantText += frame.text;
            splitter.push(frame.text);
            flushBuffer();
            // Update the assistant placeholder in-place while streaming.
            setHistory((h) => {
              const rest = h.filter((m) => m.role !== "assistant" || m.content !== "");
              const prev = rest.at(-1);
              if (prev?.role === "assistant") {
                return [...rest.slice(0, -1), { role: "assistant", content: assistantText }];
              }
              return [...rest, { role: "assistant", content: assistantText }];
            });
            if (!ttsStarted) setStatus("Streaming…");
          } else if (frame.type === "done") {
            llmMs = Math.round(performance.now() - llmStart);
            // Flush the splitter (handles any text without a terminator),
            // then force-dispatch the remaining merge buffer.
            splitter.flush();
            flushBuffer(true);
            // Finalise history without timing — TTS chain is still running.
            setHistory((h) => {
              const updated = h.filter((m) => !(m.role === "assistant" && m.content === ""));
              historyRef.current = updated;
              return updated;
            });
            ws.close();
            // Wait for the full TTS chain to drain, then stamp timing.
            ttsChain
              .then(() => {
                const ttsMs = ttsStarted ? Math.round(performance.now() - ttsStartMs) : 0;
                const timing: TurnTiming = { sttMs, ttftMs, llmMs, ttsMs };
                setHistory((h) => {
                  const updated = h.map((m, i, arr) =>
                    i === arr.length - 1 && m.role === "assistant" ? { ...m, timing } : m,
                  );
                  historyRef.current = updated;
                  return updated;
                });
                setStatus("✅ Done — hold to talk again");
              })
              .catch(() => {
                setStatus("✅ Done — hold to talk again");
              });
            resolve();
          } else if (frame.type === "error" && typeof frame.message === "string") {
            ws.close();
            reject(new Error(frame.message));
          }
        };

        ws.onerror = () => reject(new Error("WebSocket error connecting to relay"));
        ws.onclose = (ev) => {
          if (!ev.wasClean && assistantText === "") {
            reject(new Error("Relay connection closed unexpectedly"));
          }
        };
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      setStatus("Error — see above");
    }
  }, [transcriber, speaker, noThinking]);

  return (
    <section>
      <p>
        <strong>Status:</strong> {status}
      </p>
      {error && (
        <p style={{ color: "#a00" }} role="alert">
          ⚠ {error}
        </p>
      )}
      {/* Loading bars — one per model */}
      <ModelLoader model={transcriber} label="Speech recognition" />
      <ModelLoader model={speaker} label="Voice synthesis" />
      {/* Supertonic controls — shown when ONNX pipeline is loaded. */}
      {ttsState === "ready" && speaker.getProvider() !== "browser" && (
        <div
          style={{
            margin: "0.25rem 0 0.5rem",
            fontSize: "0.9rem",
            display: "flex",
            flexDirection: "column",
            gap: "0.4rem",
          }}
        >
          {/* Voice */}
          <label style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span style={{ minWidth: "4.5rem", fontWeight: 600 }}>Voice:</span>
            <select
              value={selectedTTSVoice}
              onChange={(e) => setSelectedTTSVoice(e.target.value as SupertonicVoiceId)}
              style={{ fontSize: "0.9rem", padding: "2px 4px" }}
            >
              {TTS_VOICES.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.label}
                </option>
              ))}
            </select>
          </label>

          {/* Quality / inference steps */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span style={{ minWidth: "4.5rem", fontWeight: 600 }}>Quality:</span>
            <input
              type="range"
              min={1}
              max={50}
              step={1}
              value={ttsQuality}
              onChange={(e) => setTtsQuality(parseInt(e.target.value))}
              style={{ width: "120px", accentColor: "#007aff" }}
            />
            <span style={{ color: "#555", minWidth: "3rem" }}>{ttsQuality} steps</span>
            <span style={{ color: "#aaa", fontSize: "0.78rem" }}>higher = slower, better</span>
          </div>

          {/* Speed */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span style={{ minWidth: "4.5rem", fontWeight: 600 }}>Speed:</span>
            <input
              type="range"
              min={0.8}
              max={1.2}
              step={0.01}
              value={ttsSpeed}
              onChange={(e) => setTtsSpeed(parseFloat(e.target.value))}
              style={{ width: "120px", accentColor: "#007aff" }}
            />
            <span style={{ color: "#555", minWidth: "3rem" }}>{ttsSpeed.toFixed(2)}×</span>
          </div>
        </div>
      )}
      <label
        style={{
          display: "flex",
          alignItems: "center",
          gap: "0.5rem",
          margin: "0.5rem 0",
          fontSize: "0.9rem",
          cursor: "pointer",
          userSelect: "none",
        }}
      >
        <input
          type="checkbox"
          checked={noThinking}
          onChange={(e) => setNoThinking(e.target.checked)}
        />
        <span>No thinking</span>
        <span style={{ color: "#888", fontSize: "0.8rem" }}>
          {noThinking
            ? "(faster — skips chain-of-thought)"
            : "(slower — model reasons before answering)"}
        </span>
      </label>
      {ttsState === "ready" && speaker.getProvider() === "browser" && browserVoices.length > 0 && (
        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.5rem",
            margin: "0.25rem 0",
            fontSize: "0.9rem",
          }}
        >
          <span>Voice:</span>
          <select
            value={selectedVoice?.name ?? ""}
            onChange={(e) => {
              const v = browserVoices.find((v) => v.name === e.target.value) ?? null;
              setSelectedVoice(v);
            }}
            style={{ fontSize: "0.9rem", padding: "2px 4px" }}
          >
            {browserVoices.map((v) => (
              <option key={v.name} value={v.name}>
                {v.name} ({v.lang})
                {v.name.includes("Premium") ? " ★" : v.name.includes("Enhanced") ? " ✓" : ""}
              </option>
            ))}
          </select>
        </label>
      )}
      <div style={{ display: "flex", gap: "0.75rem", margin: "1rem 0" }}>
        <button
          type="button"
          disabled={!modelReady}
          onMouseDown={modelReady ? startRecording : undefined}
          onMouseUp={modelReady ? () => void stopRecording() : undefined}
          onTouchStart={
            modelReady
              ? (e) => {
                  e.preventDefault();
                  startRecording();
                }
              : undefined
          }
          onTouchEnd={modelReady ? () => void stopRecording() : undefined}
          style={{
            padding: "1rem 2rem",
            fontSize: "1.1rem",
            cursor: modelReady ? "pointer" : "not-allowed",
            background: !modelReady ? "#aaa" : recording ? "#c00" : "#007aff",
            color: "#fff",
            border: "none",
            borderRadius: "8px",
            userSelect: "none",
            opacity: modelReady ? 1 : 0.6,
          }}
        >
          {!modelReady ? "⏳ Model loading…" : recording ? "🔴 Recording…" : "🎙 Hold to talk"}
        </button>
        <button
          type="button"
          disabled={history.length === 0}
          onClick={clearHistory}
          title="Clear session history"
          style={{
            padding: "0.5rem 1rem",
            fontSize: "1rem",
            cursor: history.length === 0 ? "not-allowed" : "pointer",
            opacity: history.length === 0 ? 0.4 : 1,
          }}
        >
          🗑 Clear
        </button>
        <button
          type="button"
          onClick={onStop}
          style={{ padding: "0.5rem 1rem", fontSize: "1rem", cursor: "pointer" }}
        >
          Stop
        </button>
      </div>
      <Transcript
        history={history}
        onReplay={(text) => {
          speaker.cancel();
          speaker.warmAudio();
          void speaker.speak(text).catch((err: unknown) => {
            setError(`TTS error: ${err instanceof Error ? err.message : String(err)}`);
          });
        }}
      />
    </section>
  );
}

/* ── VAD mode (original) ──────────────────────────────────────── */

function VoiceLoop({ onStop }: { onStop: () => void }): JSX.Element {
  const [deps] = useState(() => ({
    transcriber: new WhisperTranscriber(),
    speaker: new KokoroSpeaker(),
    vad: new SileroVAD(),
  }));
  const [pipelineState, setPipelineState] = useState<PipelineState>("idle");
  const [pipelineError, setPipelineError] = useState<string | null>(null);
  const [pipeline, setPipeline] = useState<AudioPipeline | null>(null);

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
    if (pipeline !== null) pipeline.dispose();
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
        style={{ padding: "0.5rem 1rem", fontSize: "1rem", cursor: "pointer" }}
      >
        Stop
      </button>
      <Transcript history={dialogue.history} />
    </section>
  );
}

/* ── Shared ───────────────────────────────────────────────────── */

function Transcript({
  history,
  onReplay,
}: {
  history: readonly HistoryMessage[];
  onReplay?: (text: string) => void;
}): JSX.Element {
  const [replayingIdx, setReplayingIdx] = useState<number | null>(null);
  const visible = history.filter((m) => m.role !== "system");
  if (visible.length === 0) {
    return <p style={{ color: "#888", marginTop: "1rem" }}>No turns yet.</p>;
  }
  return (
    <ol style={{ marginTop: "1rem", paddingLeft: "1.25rem" }}>
      {visible.map((m, i) => (
        <li key={i} style={{ marginBottom: "0.75rem" }}>
          <strong>{m.role}:</strong> {m.content}
          {m.role === "assistant" && onReplay && (
            <button
              type="button"
              title="Replay speech"
              disabled={replayingIdx === i}
              onClick={() => {
                setReplayingIdx(i);
                onReplay(m.content);
                // Reset the indicator after a generous window — the
                // actual duration isn't tracked here so we use a timeout
                // long enough for typical responses.
                setTimeout(() => setReplayingIdx(null), 500);
              }}
              style={{
                marginLeft: "0.5rem",
                background: "none",
                border: "none",
                cursor: replayingIdx === i ? "default" : "pointer",
                fontSize: "0.9rem",
                opacity: replayingIdx === i ? 0.4 : 0.65,
                padding: "0 2px",
                verticalAlign: "middle",
              }}
            >
              🔊
            </button>
          )}
          {m.timing && (
            <span
              style={{
                display: "inline-flex",
                gap: "0.6rem",
                marginLeft: "0.5rem",
                fontSize: "0.75rem",
                color: "#888",
                fontVariantNumeric: "tabular-nums",
              }}
            >
              <span title="Speech-to-text">🎙 {m.timing.sttMs} ms</span>
              <span title="Time to first token">⚡ {m.timing.ttftMs} ms</span>
              <span title="Total LLM time">🤖 {m.timing.llmMs} ms</span>
              <span title="TTS synthesis + playback">🔊 {m.timing.ttsMs} ms</span>
            </span>
          )}
        </li>
      ))}
    </ol>
  );
}
