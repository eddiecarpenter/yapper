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
import {
  SupertonicSpeaker,
  type SupertonicVoiceId,
} from "./tts";
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

/** Minimal interface shared by WhisperTranscriber and SupertonicSpeaker. */
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
  /** Time from first sentence dispatched to first audio playing (pure synthesis latency). */
  ttfsMs: number;
  /** Total time for TTS synthesis + playback. */
  ttsMs: number;
  /** Output tokens generated this turn (from LLM usage). */
  outputTokens: number;
  /** Output tokens per second — outputTokens / (llmMs / 1000). */
  tokensPerSec: number;
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
  const [speaker] = useState(() => new SupertonicSpeaker());
  const activeSpeakerRef = useRef<SupertonicSpeaker>(speaker);
  const [mic] = useState(() => new MicrophoneCapture());

  const [status, setStatus] = useState<string>("Initialising mic…");
  const [recording, setRecording] = useState(false);
  const [history, setHistory] = useState<HistoryMessage[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [noThinking, setNoThinking] = useState(true); // thinking off by default
  const [browserVoices, setBrowserVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [selectedVoice, setSelectedVoice] = useState<SpeechSynthesisVoice | null>(null);
  const [selectedTTSVoice, setSelectedTTSVoice] = useState<SupertonicVoiceId>(TTS_VOICES[0]!.id);
  const [ttsQuality, setTtsQuality] = useState(5); // num_inference_steps 1–50
  const [ttsSpeed, setTtsSpeed] = useState(1.2); // 0.5–2.0x
  const [firstChunkChars, setFirstChunkChars] = useState(35); // min chars for first sentence
  const firstChunkCharsRef = useRef(35);
  const [firstChunkEnabled, setFirstChunkEnabled] = useState(false); // false = fire on any sentence
  const firstChunkEnabledRef = useRef(false);

  // Both models must be ready before the button is enabled.
  const { loadingState: sttState } = useModelLoader(transcriber);
  const { loadingState: ttsState } = useModelLoader(speaker);
  const modelReady = sttState === "ready" && ttsState === "ready";

  // Populate browser voice list once TTS is ready.
  useEffect(() => {
    if (ttsState !== "ready") return;
    const load = () => {
      const voices = SupertonicSpeaker.getBrowserVoices();
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
  // Incremented by clearHistory() so any in-flight turn's done-frame can
  // detect that the session was cleared mid-turn and skip restoring stale
  // context into historyRef.
  const sessionVersionRef = useRef(0);

  // Keep browser-TTS voice in sync with UI selection.
  useEffect(() => {
    speaker.setBrowserVoice(selectedVoice);
  }, [speaker, selectedVoice]);

  // Keep ONNX voice in sync.
  useEffect(() => {
    speaker.setVoice(selectedTTSVoice);
  }, [speaker, selectedTTSVoice]);

  // Quality (inference steps).
  useEffect(() => {
    speaker.setNumInferenceSteps(ttsQuality);
  }, [speaker, ttsQuality]);

  // Speed.
  useEffect(() => {
    speaker.setSpeed(ttsSpeed);
  }, [speaker, ttsSpeed]);
  /** Phrases that trigger the voice-clear command (case-insensitive, full utterance). */
  const CLEAR_COMMAND =
    /^(clear( (session|history|chat|conversation|everything))?|reset( (session|history|chat|conversation))?|start (over|a new (session|chat|conversation))|new (chat|session|conversation)|forget (everything|that|it all))\.?[!?]?$/i;

  const clearHistory = useCallback(() => {
    activeSpeakerRef.current.cancel();
    sessionVersionRef.current++; // invalidate any in-flight turn's done-frame
    setHistory([]);
    historyRef.current = [];
    setStatus("Ready — hold button to record");
    setError(null);
    // Clear the server-side session history so the next turn's LLM
    // request doesn't inherit the old conversation context. The cookie
    // is HttpOnly so the frontend cannot manage it directly — the
    // relay owns the session store and exposes this endpoint for it.
    void fetch("/clear", { method: "POST" }).catch((err: unknown) => {
      console.warn("[session] /clear failed:", err);
    });
  }, []);

  const startRecording = useCallback(() => {
    // Cancel any TTS that's currently playing so the user can interrupt.
    activeSpeakerRef.current.cancel();
    // Warm the AudioContext while we're inside a direct user-gesture
    // handler — browsers require this to transition from suspended→running.
    // By the time speak() fires (after STT + LLM), the gesture is gone.
    activeSpeakerRef.current.warmAudio();
    framesRef.current = [];
    recordingRef.current = true;
    setRecording(true);
    setStatus("Recording… release to transcribe");
    setError(null);
  }, []);

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
        const activeSpeaker = activeSpeakerRef.current;
        setStatus("Speaking…");
        activeSpeaker.warmAudio();
        clearHistory();
        void activeSpeaker
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
      // Update historyRef synchronously so ws.onopen reads the correct
      // history immediately — never inside a setHistory updater where it
      // can race with a concurrent clearHistory() call.
      historyRef.current = [...historyRef.current, userMsg];
      setHistory(historyRef.current);

      // ── Step 2: send to relay and stream the LLM response ──────
      setStatus("Thinking…");
      // Snapshot the generation so the done-frame can detect a mid-turn clear.
      const myVersion = sessionVersionRef.current;
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      const ws = new WebSocket(`${proto}//${window.location.host}/ws`);

      await new Promise<void>((resolve, reject) => {
        // Capture the active engine once per turn — ref may change between turns.
        const activeSpeaker = activeSpeakerRef.current;
        let assistantText = "";
        let llmStart = 0;
        let llmMs = 0;
        let ttftMs = 0;
        let firstToken = true;
        let outputTokens = 0;

        // ── Sentence-streaming TTS ─────────────────────────────────
        // Tokens from the LLM are fed into a TextSplitterStream that
        // detects real sentence boundaries (handling abbreviations,
        // URLs, quotes, etc.).
        //
        // Dispatch strategy:
        //   • First chunk: accumulate complete sentences until their
        //     combined length >= FIRST_CHUNK_CHARS, then dispatch.
        //     Waiting for a full sentence (rather than cutting at a
        //     word boundary) ensures the first audio chunk is long
        //     enough that ONNX synthesis of chunk 2 completes before
        //     chunk 1 finishes playing — eliminating the audible gap.
        //   • Subsequent chunks: each complete sentence dispatches
        //     immediately. Synthesis starts in parallel so by the time
        //     the previous chunk ends, the next is ready to play.
        let splitter = new TextSplitterStream();
        let mergeBuffer = ""; // complete sentences accumulating toward first dispatch
        // FIRST_CHUNK_CHARS: minimum combined sentence length before first dispatch.
        // When disabled, any complete sentence fires immediately.
        const FIRST_CHUNK_CHARS = firstChunkEnabledRef.current ? firstChunkCharsRef.current : 0;
        let firstChunkDispatched = false;
        let ttsChain = Promise.resolve() as Promise<void>;
        let ttsStartMs = 0;
        let ttsStarted = false;
        let ttfsMs = 0; // time-to-first-sound: ms from first dispatch to first audio playing
        let firstSoundPlayed = false;

        const enqueueChunk = (chunk: string) => {
          const s = chunk.trim();
          if (!s) return;
          if (!ttsStarted) {
            ttsStarted = true;
            ttsStartMs = performance.now();
            // Clear the abort flag that cancel() set in startRecording().
            // Without this, synthesize() discards its result (returns null),
            // falls back to speak() which re-queues synthesis behind all the
            // other enqueueChunk calls, so audio only starts after everything
            // is synthesised — which looks like "message appears, then audio".
            activeSpeaker.beginTurn();
            setStatus("Preparing voice…");
          }
          // Kick off synthesis immediately — it runs in parallel with the
          // current chunk's playback so there's no synthesis gap between chunks.
          const synthPromise = activeSpeaker.synthesize(s);
          ttsChain = ttsChain
            .then(async () => {
              if (recordingRef.current) return;
              const data = await synthPromise;
              if (!firstSoundPlayed) {
                firstSoundPlayed = true;
                ttfsMs = Math.round(performance.now() - ttsStartMs);
                setStatus("Speaking…");
              }
              if (data === null) {
                // ONNX synthesis skipped (browser TTS fallback or abort) —
                // fall back to the blocking speak() path.
                return activeSpeaker.speak(s);
              }
              return activeSpeaker.playAudioData(data);
            })
            .catch((ttsErr: unknown) => {
              const msg = ttsErr instanceof Error ? ttsErr.message : String(ttsErr);
              console.error("[TTS] speak() failed:", ttsErr);
              setError(`TTS error: ${msg}`);
            });
        };

        // Drain completed sentences from the splitter and dispatch them.
        //
        // First-chunk rule: accumulate complete sentences until the buffer
        // reaches FIRST_CHUNK_CHARS. This ensures the first audio chunk is
        // long enough that ONNX synthesis of the second chunk completes
        // before the first chunk finishes playing — eliminating the audible
        // gap that occurs when a short first chunk (word-boundary cut) ends
        // before the next synthesis is ready.
        //
        //   • No sentence yet → wait.
        //   • Sentence(s) accumulated but < FIRST_CHUNK_CHARS → wait (merge).
        //   • Sentence(s) >= FIRST_CHUNK_CHARS → dispatch immediately.
        //   • forceAll (stream ended) → dispatch whatever is left.
        //
        // No word-boundary cuts during streaming — they produce chunks that
        // are too short, causing audible pauses between the first and second
        // chunks while ONNX finishes the second synthesis.
        //
        // Subsequent chunks: each complete sentence dispatches immediately.
        // splitter.flush() (called externally before forceAll) moves any
        // partial remainder into splitter.sentences, so the same loop
        // handles both mid-stream and end-of-stream cases.
        const flushBuffer = (forceAll = false) => {
          if (!firstChunkDispatched) {
            // Accumulate complete sentences into mergeBuffer.
            while (splitter.sentences.length > 0) {
              const s = splitter.sentences.shift()!;
              mergeBuffer += (mergeBuffer ? " " : "") + s;
            }

            if (!mergeBuffer.trim()) return; // no sentence yet
            // Wait until merged sentences are long enough for good overlap,
            // unless the stream has ended (forceAll).
            if (!forceAll && mergeBuffer.length < FIRST_CHUNK_CHARS) return;

            firstChunkDispatched = true;
            enqueueChunk(mergeBuffer);
            mergeBuffer = "";
            return;
          }

          // Subsequent chunks: dispatch each complete sentence immediately.
          while (splitter.sentences.length > 0) {
            const s = splitter.sentences.shift()!;
            if (s.trim()) enqueueChunk(s.trim());
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
          let frame: { type: string; text?: string; message?: string; usage?: { input: number; output: number } };
          try {
            frame = JSON.parse(ev.data) as typeof frame;
          } catch {
            return;
          }

          if (frame.type === "token" && typeof frame.text === "string") {
            // If the user cleared the session mid-turn, discard tokens so
            // stale assistant text doesn't re-appear in the cleared history.
            if (sessionVersionRef.current !== myVersion) return;
            if (firstToken) {
              ttftMs = Math.round(performance.now() - llmStart);
              firstToken = false;
            }
            assistantText += frame.text;
            splitter.push(frame.text);
            flushBuffer();
            // Update the assistant placeholder in-place while streaming.
            // NOTE: historyRef is NOT updated here — it only tracks the
            // turn-complete state. The streaming display is pure React state.
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
            outputTokens = frame.usage?.output ?? 0;
            // Flush the splitter (handles any text without a terminator),
            // then force-dispatch the remaining merge buffer.
            splitter.flush();
            flushBuffer(true);
            // Commit the completed assistant turn to historyRef so the NEXT
            // turn's ws.onopen sends the full conversation context to the LLM.
            // Guard with the generation counter: if clearHistory() fired
            // mid-turn (sessionVersionRef was incremented), skip this — the
            // session has already been wiped and we must not restore it.
            if (sessionVersionRef.current === myVersion) {
              historyRef.current = [
                ...historyRef.current,
                { role: "assistant", content: assistantText },
              ];
              setHistory(historyRef.current);
            }
            ws.close();
            // Wait for the full TTS chain to drain, then stamp timing.
            // NOTE: do NOT write historyRef.current here. This callback fires
            // after cancel() resolves the chain (e.g. when the user says
            // "clear"), and at that point the old React state may not yet
            // have been replaced by setHistory([]) — so writing historyRef
            // here would silently restore the cleared context. Timing is
            // display-only; the LLM never sees it.
            ttsChain
              .then(() => {
                const ttsMs = ttsStarted ? Math.round(performance.now() - ttsStartMs) : 0;
                const tokensPerSec = outputTokens > 0 && llmMs > 0
                  ? Math.round(outputTokens / (llmMs / 1000))
                  : 0;
                const timing: TurnTiming = { sttMs, ttftMs, llmMs, ttfsMs, ttsMs, outputTokens, tokensPerSec };
                setHistory((h) => {
                  return h.map((m, i, arr) =>
                    i === arr.length - 1 && m.role === "assistant" ? { ...m, timing } : m,
                  );
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
      {/* TTS controls */}
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
            <span style={{ color: "#555", minWidth: "3rem" }}>{ttsQuality} step{ttsQuality !== 1 ? "s" : ""}</span>
            <span style={{ color: ttsQuality === 1 ? "#2a9d2a" : ttsQuality <= 5 ? "#aaa" : "#c00", fontSize: "0.78rem" }}>
              {ttsQuality === 1 ? "⚡ fastest — best for conversation" : ttsQuality <= 5 ? `~${ttsQuality}× slower` : `~${ttsQuality}× slower — noticeable delay`}
            </span>
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

          {/* First-chunk threshold */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span style={{ minWidth: "4.5rem", fontWeight: 600 }}>Trigger:</span>
            <label style={{ display: "flex", alignItems: "center", gap: "0.3rem", cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={firstChunkEnabled}
                onChange={(e) => {
                  const v = e.target.checked;
                  setFirstChunkEnabled(v);
                  firstChunkEnabledRef.current = v;
                }}
              />
              <span style={{ fontSize: "0.85rem", color: "#555" }}>enabled</span>
            </label>
            <input
              type="range"
              min={10}
              max={200}
              step={5}
              disabled={!firstChunkEnabled}
              value={firstChunkChars}
              onChange={(e) => {
                const v = parseInt(e.target.value);
                setFirstChunkChars(v);
                firstChunkCharsRef.current = v;
              }}
              style={{ width: "120px", accentColor: "#007aff", opacity: firstChunkEnabled ? 1 : 0.35 }}
            />
            <span style={{ color: firstChunkEnabled ? "#555" : "#aaa", minWidth: "3rem" }}>
              {firstChunkEnabled ? `${firstChunkChars} ch` : "off"}
            </span>
            <span style={{ color: "#aaa", fontSize: "0.78rem" }}>min chars for first sentence</span>
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
          activeSpeakerRef.current.cancel();
          activeSpeakerRef.current.warmAudio();
          void activeSpeakerRef.current.speak(text).catch((err: unknown) => {
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
    speaker: new SupertonicSpeaker(),
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
              <span title="Speech-to-text (Whisper)">🎙 {m.timing.sttMs} ms</span>
              <span title="Time to first token (LLM)">⚡ {m.timing.ttftMs} ms</span>
              <span title="Time to first sound (TTS synthesis)">🔉 {m.timing.ttfsMs} ms</span>
            </span>
          )}
        </li>
      ))}
    </ol>
  );
}
