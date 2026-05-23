/**
 * `KokoroSpeaker` — browser-side `Speaker` backed by Kokoro running
 * via Transformers.js + ONNX Web.
 *
 * Canonical specification: `docs/ARCHITECTURE.md` §6.4 + AD-2 / AD-10.
 *
 * Layered across the four tasks of Feature #14:
 *   - Task 1 (this commit) — backend selection + provider logging (AC-4
 *     logging clause). Smallest testable surface; every subsequent task
 *     instantiates the class, so the provider field has to settle
 *     correctly first. `speak()`, `cancel()`, `dispose()` are stubbed
 *     here so the file compiles against the `Speaker` interface; the
 *     real bodies land in Tasks 2–4.
 *   - Task 2 — lazy Kokoro pipeline loading + loading-state observable
 *     (AC-4 fallback clause).
 *   - Task 3 — `speak()` synthesis + Web Audio playback + voice option
 *     (AC-1, AC-3).
 *   - Task 4 — `cancel()` coordination + `dispose()` lifecycle (AC-2).
 *
 * The class is exported through `./index.ts` alongside the existing
 * `Speaker` type so consumers import from a single barrel — same shape
 * as the `web/src/stt/` module.
 */
import type { Speaker } from "./types";

/**
 * The ONNX Web execution provider this Speaker selected at
 * construction. Surfaced via `getProvider()` so tests can assert the
 * selection branch without grepping console output, and so a future UI
 * shell can display the active backend. Mirrors the `Provider` type
 * exported by `web/src/stt/WhisperTranscriber` so a future
 * cross-module surface can use one shared shape.
 */
export type Provider = "webgpu" | "wasm";

export class KokoroSpeaker implements Speaker {
  private readonly provider: Provider;

  /**
   * Probes `navigator.gpu` synchronously and pins the provider for the
   * lifetime of the instance. A truthy `navigator.gpu` selects WebGPU;
   * an undefined or null `navigator.gpu` falls back to WASM without
   * throwing — WASM is a first-class supported path per §6.4 / AD-2.
   *
   * The chosen provider is logged exactly once via `console.log` in the
   * canonical `provider: <name>` format required by AC-4. The log line
   * is identical to `WhisperTranscriber`'s so dialogue-loop log
   * scraping stays uniform across STT and TTS.
   *
   * The Kokoro pipeline and the `AudioContext` are deliberately NOT
   * instantiated here — see Tasks 2 and 3 for the lazy-load gates.
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
      // falsy value the linter can reason about.
      ((navigator as unknown as { gpu?: unknown }).gpu ?? null) !== null;

    this.provider = hasWebGPU ? "webgpu" : "wasm";
    // Deterministic single-line log so AC-4 can be verified from the
    // browser console without inspecting internal fields.
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
   * Synthesise `text` to speech. Task-1 stub: resolves immediately
   * without touching the (not-yet-wired) Kokoro pipeline. Task 2
   * layers the lazy pipeline load on top; Task 3 wires the real
   * synthesis + Web Audio playback path.
   */
  async speak(_text: string): Promise<void> {
    // Intentionally empty in Task 1. The parameter is prefixed with `_`
    // so the linter does not complain about the unused argument while
    // still exposing the canonical `Speaker` signature.
    return;
  }

  /**
   * Cancel any in-flight playback. Task-1 stub: no-op. Task 4 wires
   * the real coordination with the active source node + in-flight
   * `speak()` resolver.
   */
  cancel(): void {
    // Intentionally empty in Task 1.
  }

  /**
   * Release the underlying model + audio resources. Task-1 stub: no-op.
   * Task 4 wires the full teardown (pipeline release, AudioContext
   * close, subscriber drop, state reset).
   */
  dispose(): void {
    // Intentionally empty in Task 1.
  }
}
