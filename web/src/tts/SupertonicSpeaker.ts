/**
 * `SupertonicSpeaker` — browser-side `Speaker` backed by Supertonic-TTS
 * running via Transformers.js + ONNX Web on WebGPU (with WASM fallback).
 *
 * Model: onnx-community/Supertonic-TTS-ONNX
 * Reference demo: https://huggingface.co/spaces/webml-community/Supertonic-TTS-WebGPU
 *
 * Key differences from SpeechT5:
 *   - Speaker embeddings are 1×101×128 Float32Array `.bin` files (not 512-dim x-vectors).
 *   - Runs on WebGPU without dtype override — library picks optimal defaults.
 *   - Supports `num_inference_steps` (quality, 1–10) and `speed` (0.5–2.0).
 *   - Sampling rate: 44 100 Hz.
 *   - A warm-up call (1 inference step, dummy embeddings) is issued right
 *     after the pipeline loads to pre-compile WebGPU shaders.
 */
import { pipeline } from "@huggingface/transformers";

import type { Speaker } from "./types";

export type Provider = "webgpu" | "wasm" | "browser";

export type LoadingState = "idle" | "loading" | "ready" | "error";

type TtsPipeline = Awaited<ReturnType<typeof pipeline<"text-to-speech">>>;

/** Model identifier on HuggingFace. */
export const SUPERTONIC_MODEL_ID = "onnx-community/Supertonic-TTS-ONNX";

/** Base URL for pre-built speaker embedding binaries. */
const VOICES_BASE_URL = `https://huggingface.co/${SUPERTONIC_MODEL_ID}/resolve/main/voices/`;

/** Available Supertonic voices — Float32Array binaries, shape 1×101×128. */
export const SUPERTONIC_VOICES = {
  F1: `${VOICES_BASE_URL}F1.bin`,
  F2: `${VOICES_BASE_URL}F2.bin`,
  M1: `${VOICES_BASE_URL}M1.bin`,
  M2: `${VOICES_BASE_URL}M2.bin`,
} as const;

export type SupertonicVoiceId = keyof typeof SUPERTONIC_VOICES;

export const DEFAULT_VOICE_ID: SupertonicVoiceId = "F1";

/** Default voice — kept for interface compat. */
export const DEFAULT_VOICE = "F1";

/**
 * Shape of a single Supertonic synthesis output.
 * `{ audio: Float32Array, sampling_rate: number }` — 44 100 Hz.
 */
type SynthesisOutput = {
  audio: Float32Array;
  sampling_rate: number;
};

export function extractSynthesisOutput(result: unknown): SynthesisOutput {
  if (result === null || typeof result !== "object") {
    throw new Error(`TTS pipeline returned ${String(result)} — expected an object`);
  }

  const r = result as Record<string, unknown>;

  let audioCandidate: unknown = r["audio"];
  let rateCandidate: unknown = r["sampling_rate"];

  // Tensor-like: { data: Float32Array, ... }
  if (
    audioCandidate !== undefined &&
    typeof audioCandidate === "object" &&
    audioCandidate !== null
  ) {
    const maybeData = (audioCandidate as Record<string, unknown>)["data"];
    if (maybeData instanceof Float32Array) {
      audioCandidate = maybeData;
    }
  }

  // Array wrapper: [{ audio, sampling_rate }]
  if (Array.isArray(result) && result.length > 0) {
    return extractSynthesisOutput(result[0]);
  }

  if (!(audioCandidate instanceof Float32Array)) {
    console.error("[TTS] extractSynthesisOutput: unexpected result shape", result);
    throw new Error(
      `TTS pipeline returned unexpected audio shape — got ${Object.keys(r).join(", ")}`,
    );
  }
  if (typeof rateCandidate !== "number" || rateCandidate <= 0) {
    console.error("[TTS] extractSynthesisOutput: bad sampling_rate", rateCandidate);
    throw new Error(`TTS pipeline returned invalid sampling_rate: ${String(rateCandidate)}`);
  }

  return { audio: audioCandidate, sampling_rate: rateCandidate };
}

export type LoadErrorCause = "quota" | "network" | "unknown";

function classifyLoadError(err: unknown): LoadErrorCause {
  if (err instanceof Error) {
    const name = err.name;
    const message = err.message ?? "";
    if (name === "QuotaExceededError" || /quota/i.test(message)) return "quota";
    if (name === "TypeError" && /(failed to fetch|network|networkerror)/i.test(message))
      return "network";
  }
  return "unknown";
}

function loadErrorMessage(cause: LoadErrorCause, err: unknown): string {
  const original = err instanceof Error ? err.message : String(err);
  switch (cause) {
    case "quota":
      return `Failed to load TTS model: browser storage quota exceeded (${original})`;
    case "network":
      return `Failed to load TTS model: network unreachable (${original})`;
    case "unknown":
      return `Failed to load TTS model: ${original}`;
  }
}

export interface SupertonicSpeakerOptions {
  voice?: string;
}

export class SupertonicSpeaker implements Speaker {
  private provider: Provider;
  private readonly voice: string;

  private pipelinePromise: Promise<TtsPipeline> | null = null;
  private loadingState: LoadingState = "idle";
  private lastError: string | null = null;
  private readonly listeners: Set<(state: LoadingState) => void> = new Set();

  /** Currently selected voice ID. */
  private voiceId: SupertonicVoiceId = DEFAULT_VOICE_ID;
  /** Cache of fetched embeddings — keyed by voice ID so switching voices
   *  doesn't re-download a previously loaded embedding. */
  private readonly embeddingsCache: Map<string, Float32Array> = new Map();

  /** Number of diffusion inference steps (quality). Higher = slower + better.
   *  Default 1 for minimum synthesis latency — fast enough that audio starts
   *  before the LLM finishes streaming even short responses. */
  private numInferenceSteps = 1;
  /** Playback speed multiplier. */
  private speed = 1.0;

  private usingBrowserTTS = false;
  private activeUtterance: SpeechSynthesisUtterance | null = null;
  private browserVoice: SpeechSynthesisVoice | null = null;

  private audioContext: AudioContext | null = null;
  private activeSource: AudioBufferSourceNode | null = null;
  private activeResolver: (() => void) | null = null;
  private activeAborted = false;

  constructor(options: SupertonicSpeakerOptions = {}) {
    const hasWebGPU =
      typeof navigator !== "undefined" &&
      ((navigator as unknown as { gpu?: unknown }).gpu ?? null) !== null;

    // Supertonic-TTS runs on WebGPU — that's the whole point. WASM is the
    // fallback for browsers without WebGPU support.
    this.provider = hasWebGPU ? "webgpu" : "wasm";
    this.voice = options.voice ?? DEFAULT_VOICE;
    console.log(`provider: ${this.provider}`);
  }

  getProvider(): Provider {
    return this.provider;
  }
  getVoice(): string {
    return this.voice;
  }
  getLoadingState(): LoadingState {
    return this.loadingState;
  }
  getError(): string | null {
    return this.lastError;
  }

  /**
   * Switch the active Supertonic voice. The embedding for the new voice is
   * fetched (and cached) on the next `speak()` call.
   */
  setVoice(voiceId: SupertonicVoiceId): void {
    this.voiceId = voiceId;
  }

  /** Set the number of diffusion inference steps (1 = fast/rough, 50 = slow/best). */
  setNumInferenceSteps(steps: number): void {
    this.numInferenceSteps = Math.max(1, Math.min(50, Math.round(steps)));
  }

  /** Set the TTS speed multiplier (0.8 = slower, 1.0 = normal, 1.2 = faster). */
  setSpeed(speed: number): void {
    this.speed = Math.max(0.8, Math.min(1.2, speed));
  }

  preload(): void {
    void this.loadPipeline().catch(() => undefined);
  }

  subscribe(listener: (state: LoadingState) => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  private setState(next: LoadingState): void {
    if (this.loadingState === next) return;
    this.loadingState = next;
    for (const listener of this.listeners) {
      try {
        listener(next);
      } catch {
        /* best-effort */
      }
    }
  }

  private loadPipeline(): Promise<TtsPipeline> {
    if (this.pipelinePromise !== null) return this.pipelinePromise;
    this.setState("loading");
    this.lastError = null;
    this.pipelinePromise = this._buildPipeline();
    return this.pipelinePromise;
  }

  private async _buildPipeline(): Promise<TtsPipeline> {
    // Try WebGPU first (fast — GPU shader execution); fall back to WASM.
    const attempts: Array<"webgpu" | "wasm"> =
      this.provider === "webgpu" ? ["webgpu", "wasm"] : ["wasm"];

    let lastErr: unknown;
    for (const device of attempts) {
      try {
        // No dtype override needed — Transformers.js picks optimal defaults
        // for the device (matches the official demo's pipeline() call).
        const p = await pipeline("text-to-speech", SUPERTONIC_MODEL_ID, { device });
        this.provider = device;
        console.log(`tts: loaded on ${device}`);

        // Warm-up: pre-compile WebGPU shaders with a dummy 1-step inference
        // so the first real speak() isn't penalised by shader compilation.
        if (device === "webgpu") {
          console.log("[TTS] warming up WebGPU shaders…");
          try {
            await (
              p as unknown as (
                text: string,
                opts: {
                  speaker_embeddings: Float32Array;
                  num_inference_steps: number;
                  speed: number;
                },
              ) => Promise<unknown>
            )("Hello", {
              speaker_embeddings: new Float32Array(1 * 101 * 128),
              num_inference_steps: 1,
              speed: 1.0,
            });
            console.log("[TTS] warm-up done");
          } catch (warmErr) {
            // Warm-up failure is non-fatal — real synthesis may still work.
            console.warn("[TTS] warm-up failed (non-fatal):", warmErr);
          }
        }

        // Preload the default voice embedding so the first speak() call
        // doesn't stall on a HuggingFace network fetch. The ~51 KB .bin
        // file is fetched and cached here; subsequent calls return instantly
        // from the in-memory embeddingsCache.
        console.log("[TTS] preloading default voice embedding…");
        this.loadEmbeddings(this.voiceId).then(
          () => console.log("[TTS] voice embedding ready"),
          (e: unknown) => console.warn("[TTS] voice embedding preload failed (non-fatal):", e),
        );

        this.setState("ready");
        return p;
      } catch (err: unknown) {
        console.warn(
          `[SupertonicSpeaker] ${device} pipeline failed: ${err instanceof Error ? err.message : String(err)}`,
        );
        lastErr = err;
        if (device === "webgpu") {
          // Downgrade and retry on WASM.
          this.provider = "wasm";
        }
      }
    }

    // Both ONNX paths failed — fall back to Web Speech API.
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      console.warn("[SupertonicSpeaker] ONNX unavailable; falling back to Web Speech API");
      this.provider = "browser";
      this.usingBrowserTTS = true;
      this.setState("ready");
      return null as unknown as TtsPipeline;
    }

    this.pipelinePromise = null;
    this.setState("error");
    const cause = classifyLoadError(lastErr);
    const msg = loadErrorMessage(cause, lastErr);
    this.lastError = msg;
    throw new Error(msg);
  }

  /**
   * Fetch and cache the speaker embedding binary for `voiceId`.
   * The Float32Array (1×101×128 = 12 928 floats) is cached after first fetch.
   */
  private async loadEmbeddings(voiceId: SupertonicVoiceId): Promise<Float32Array> {
    const cached = this.embeddingsCache.get(voiceId);
    if (cached) return cached;

    const url = SUPERTONIC_VOICES[voiceId];
    const buf = await fetch(url).then((r) => {
      if (!r.ok) throw new Error(`Failed to fetch voice embedding ${voiceId}: HTTP ${r.status}`);
      return r.arrayBuffer();
    });
    const arr = new Float32Array(buf);
    this.embeddingsCache.set(voiceId, arr);
    return arr;
  }

  private ensureAudioContext(): AudioContext {
    if (this.audioContext === null) {
      this.audioContext = new AudioContext();
    }
    return this.audioContext;
  }

  /**
   * Run ONNX inference for `text` and return the raw audio data.
   * Returns `null` if the speaker is using browser TTS (no pre-synthesis
   * available) or if aborted before synthesis completed.
   *
   * Callers can start synthesis early (while the previous chunk is still
   * playing) and hand the result to `playAudioData()` when ready.
   */
  async synthesize(text: string): Promise<{ audio: Float32Array; samplingRate: number } | null> {
    if (text.trim() === "") return null;
    if (this.usingBrowserTTS) return null;

    const tts = await this.loadPipeline();
    if (this.usingBrowserTTS) return null;

    const speaker_embeddings = await this.loadEmbeddings(this.voiceId);

    const rawResult = await (
      tts as unknown as (
        text: string,
        opts: {
          speaker_embeddings: Float32Array;
          num_inference_steps: number;
          speed: number;
        },
      ) => Promise<unknown>
    )(text, {
      speaker_embeddings,
      num_inference_steps: this.numInferenceSteps,
      speed: this.speed,
    });

    if (this.activeAborted) return null;

    const { audio: rawAudio, sampling_rate } = extractSynthesisOutput(rawResult);

    // Pad with 0.5 s of silence at the end for natural inter-sentence pacing.
    const silenceSamples = Math.floor(0.5 * sampling_rate);
    const audio = new Float32Array(rawAudio.length + silenceSamples);
    audio.set(rawAudio);

    return { audio, samplingRate: sampling_rate };
  }

  /**
   * Play a pre-synthesised audio buffer through the Web Audio API.
   * Returns a promise that resolves when playback (including the silence
   * padding already baked into `data.audio`) completes.
   */
  async playAudioData(data: { audio: Float32Array; samplingRate: number }): Promise<void> {
    if (this.activeAborted) {
      this.activeAborted = false;
      return;
    }

    const ctx = this.ensureAudioContext();
    if (ctx.state === "suspended") {
      await ctx.resume();
    }

    const buffer = ctx.createBuffer(1, data.audio.length, data.samplingRate);
    buffer.copyToChannel(data.audio as Float32Array<ArrayBuffer>, 0);

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    this.activeSource = source;

    await new Promise<void>((resolve) => {
      this.activeResolver = resolve;
      source.onended = () => {
        if (this.activeSource === source) this.activeSource = null;
        if (this.activeResolver === resolve) this.activeResolver = null;
        resolve();
      };
      source.start();
    });
  }

  async speak(text: string): Promise<void> {
    if (text.trim() === "") return;
    this.activeAborted = false;

    if (this.usingBrowserTTS) {
      return this.speakWithBrowserTTS(text);
    }

    const data = await this.synthesize(text);
    if (data === null) return;
    return this.playAudioData(data);
  }

  static getBrowserVoices(): SpeechSynthesisVoice[] {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return [];
    return window.speechSynthesis
      .getVoices()
      .filter((v) => v.lang.startsWith("en"))
      .sort((a, b) => {
        const rank = (v: SpeechSynthesisVoice) =>
          v.name.includes("Premium") ? 0 : v.name.includes("Enhanced") ? 1 : 2;
        return rank(a) - rank(b);
      });
  }

  setBrowserVoice(voice: SpeechSynthesisVoice | null): void {
    this.browserVoice = voice;
  }

  private speakWithBrowserTTS(text: string): Promise<void> {
    return new Promise<void>((resolve) => {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      const voice = this.browserVoice ?? SupertonicSpeaker.getBrowserVoices()[0] ?? null;
      if (voice) utterance.voice = voice;
      utterance.rate = 1.0;
      this.activeUtterance = utterance;
      utterance.onend = () => {
        if (this.activeUtterance === utterance) this.activeUtterance = null;
        resolve();
      };
      utterance.onerror = () => {
        if (this.activeUtterance === utterance) this.activeUtterance = null;
        resolve();
      };
      window.speechSynthesis.speak(utterance);
    });
  }

  warmAudio(): void {
    if (this.usingBrowserTTS) return;
    const ctx = this.ensureAudioContext();
    if (ctx.state === "suspended") {
      void ctx.resume();
    }
  }

  /**
   * Call at the start of each new TTS turn (before the first `synthesize()`
   * call) to clear the abort flag that `cancel()` sets. Without this,
   * synthesis results from the new turn are silently discarded if `cancel()`
   * was called just before (e.g. by `startRecording()` to stop previous
   * speech).
   */
  beginTurn(): void {
    this.activeAborted = false;
  }

  cancel(): void {
    if (this.usingBrowserTTS) {
      window.speechSynthesis.cancel();
      this.activeUtterance = null;
      return;
    }

    if (this.audioContext !== null && this.audioContext.state === "suspended") {
      void this.audioContext.resume();
    }

    this.activeAborted = true;

    const resolver = this.activeResolver;
    if (resolver !== null) {
      this.activeResolver = null;
      resolver();
    }

    const source = this.activeSource;
    if (source !== null) {
      this.activeSource = null;
      try {
        source.stop();
      } catch {
        /* defensive */
      }
    }
  }

  dispose(): void {
    this.cancel();
    this.activeAborted = false;

    if (this.pipelinePromise !== null) {
      const promise = this.pipelinePromise;
      promise
        .then((p) => {
          p.dispose().catch(() => undefined);
        })
        .catch(() => undefined);
      this.pipelinePromise = null;
    }

    if (this.audioContext !== null) {
      const ctx = this.audioContext;
      this.audioContext = null;
      try {
        const r = ctx.close();
        if (r && typeof r.catch === "function") r.catch(() => undefined);
      } catch {
        /* defensive */
      }
    }

    this.usingBrowserTTS = false;
    this.activeUtterance = null;
    this.embeddingsCache.clear();
    this.listeners.clear();
    this.loadingState = "idle";
  }
}
