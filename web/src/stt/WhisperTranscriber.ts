/**
 * `WhisperTranscriber` — browser-side `Transcriber` backed by Whisper running
 * via Transformers.js + ONNX Web.
 *
 * Canonical specification: `docs/ARCHITECTURE.md` §6.3 + AD-2 / AD-11.
 *
 * Layered across the three tasks of Feature #13:
 *   - Task 1 (this commit) — backend selection + provider logging only.
 *     The constructor probes `navigator.gpu` and chooses `"webgpu"` or
 *     `"wasm"`, logging the chosen provider deterministically so the
 *     acceptance criteria (AC-2 / AC-3) can be observed without coupling
 *     to internal state. The ASR pipeline is NOT instantiated yet;
 *     `transcribe()` is a stub returning the empty string and `dispose()`
 *     is a no-op. Later tasks layer model loading + transcription on top.
 *   - Task 2 — lazy model loading + loading-state observable (AC-4).
 *   - Task 3 — transcription wiring (AC-1).
 *
 * The class is exported through `./index.ts` alongside the existing
 * `Transcriber` type so consumers import from a single barrel.
 */
import type { Transcriber } from "./types";

/**
 * The ONNX Web execution provider this Transcriber selected at
 * construction. Surfaced via `getProvider()` so tests can assert the
 * selection branch without grepping console output, and so a future UI
 * shell can display the active backend.
 */
export type Provider = "webgpu" | "wasm";

export class WhisperTranscriber implements Transcriber {
  private readonly provider: Provider;

  /**
   * Probes `navigator.gpu` synchronously and pins the provider for the
   * lifetime of the instance. A truthy `navigator.gpu` selects WebGPU;
   * an undefined or null `navigator.gpu` falls back to WASM without
   * throwing — WASM is a first-class supported path per §6.3 / AD-2.
   *
   * The chosen provider is logged exactly once via `console.log` in the
   * canonical `provider: <name>` format required by AC-2 / AC-3.
   */
  constructor() {
    // `navigator.gpu` is the WebGPU entry point per the W3C WebGPU spec.
    // It is undefined in browsers without WebGPU and in jsdom (the test
    // environment), so the wasm-fallback branch is the natural default
    // for unit tests and the webgpu branch must be exercised with a
    // truthy stub.
    const hasWebGPU =
      typeof navigator !== "undefined" &&
      // The presence check uses `?? null` to coerce `undefined` to a
      // falsy value the linter can reason about; the actual API surface
      // is not invoked in Task 1.
      ((navigator as unknown as { gpu?: unknown }).gpu ?? null) !== null;

    this.provider = hasWebGPU ? "webgpu" : "wasm";
    // Deterministic single-line log so AC-2 / AC-3 can be verified from
    // the browser console without inspecting internal fields. The exact
    // wording (`provider: webgpu` / `provider: wasm`) is what the
    // acceptance criteria call out and is asserted in unit tests.
    console.log(`provider: ${this.provider}`);
  }

  /**
   * Return the execution provider chosen at construction. Used by tests
   * to assert the selection branch and by any future UI surface that
   * wants to display the active backend.
   */
  getProvider(): Provider {
    return this.provider;
  }

  /**
   * Task 1 stub — the real implementation lands in Task 3 once the
   * ASR pipeline is loaded by Task 2. The stub honours the
   * `Transcriber` interface contract (returns the empty string for
   * "no recognised speech") so this class can already be instantiated
   * by code paths that wire the interface but do not yet exercise
   * transcription.
   */
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async transcribe(_audio: Float32Array, _sampleRate: number): Promise<string> {
    return "";
  }

  /**
   * Task 1 stub — there is no pipeline to release yet. Task 2 will
   * extend this to clear the cached pipeline promise and detach
   * loading-state subscribers; Task 3 will not need to change it
   * further.
   */
  dispose(): void {
    // no-op until Task 2 attaches a pipeline + subscribers to release.
  }
}
