/**
 * Unit tests for `KokoroSpeaker`.
 *
 * Coverage layers across the feature's four tasks:
 *   - Task 1 (this layer): backend-selection branches and the canonical
 *     provider log line. Does not touch the Kokoro pipeline or the
 *     Web Audio API.
 *   - Task 2 will add lazy pipeline loading + loading-state observable
 *     coverage.
 *   - Task 3 will add the AC-1 / AC-3 synthesis + playback fixture.
 *   - Task 4 will add cancel + dispose lifecycle coverage.
 *
 * jsdom does not expose `navigator.gpu`, so the WASM branch is the
 * natural default and the WebGPU branch must be exercised by stubbing
 * the field on the global `navigator` object and restoring afterwards.
 *
 * Mirrors the structure of `web/src/stt/WhisperTranscriber.test.ts` so
 * the two test files read as siblings — a future reader who has read
 * one can navigate the other.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { KokoroSpeaker } from "./KokoroSpeaker";

/**
 * `navigator.gpu` is not declared on the default DOM lib's `Navigator`
 * type, so the stub goes through a typed cast. Using a sentinel object
 * (rather than `true`) matches the real shape closely enough — the
 * production code only checks for non-null truthiness — while keeping
 * the value distinct in case future tests need to round-trip it.
 */
type GpuLike = Record<string, never>;
const GPU_SENTINEL: GpuLike = Object.freeze({});

describe("KokoroSpeaker — backend selection", () => {
  let logSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    // Spy on console.log so we can assert the provider line was emitted
    // exactly once, then restore in afterEach. mockImplementation is a
    // noop so the actual log does not noise up the test runner output.
    logSpy = vi.spyOn(console, "log").mockImplementation(() => undefined);
  });

  afterEach(() => {
    logSpy.mockRestore();
    // Always wipe any navigator.gpu stub a test installed, so a leaked
    // value cannot leak into the next test. Using `delete` rather than
    // assigning `undefined` makes the property genuinely absent — which
    // is the precise state jsdom presents and the WASM path expects.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    delete (navigator as any).gpu;
  });

  it("selects wasm and logs 'provider: wasm' when navigator.gpu is undefined", () => {
    // jsdom's navigator has no `gpu`; this is the default state — the
    // delete in afterEach guarantees no prior test polluted it.
    const s = new KokoroSpeaker();

    expect(s.getProvider()).toBe("wasm");
    expect(logSpy).toHaveBeenCalledTimes(1);
    expect(logSpy).toHaveBeenCalledWith("provider: wasm");
  });

  it("does not throw when navigator.gpu is undefined (WASM fallback is supported)", () => {
    // AC-4 explicitly says "no error thrown" on the WASM path.
    expect(() => new KokoroSpeaker()).not.toThrow();
  });

  it("selects webgpu and logs 'provider: webgpu' when navigator.gpu is present", () => {
    // Install a truthy `gpu` field for the duration of this test only.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (navigator as any).gpu = GPU_SENTINEL;

    const s = new KokoroSpeaker();

    expect(s.getProvider()).toBe("webgpu");
    expect(logSpy).toHaveBeenCalledTimes(1);
    expect(logSpy).toHaveBeenCalledWith("provider: webgpu");
  });

  it("logs the provider exactly once per construction", () => {
    // Two instances → two log lines, never more. Catches a future
    // refactor that accidentally calls the logger from getProvider()
    // or some other accessor.
    new KokoroSpeaker();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (navigator as any).gpu = GPU_SENTINEL;
    new KokoroSpeaker();

    expect(logSpy).toHaveBeenCalledTimes(2);
    expect(logSpy).toHaveBeenNthCalledWith(1, "provider: wasm");
    expect(logSpy).toHaveBeenNthCalledWith(2, "provider: webgpu");
  });

  it("Task-1 stubs do not throw and have the expected shapes", async () => {
    // The Speaker interface contract requires speak() to return a
    // Promise<void>, cancel() and dispose() to return void. Pin those
    // shapes against the Task-1 stubs so a Task-3/4 implementation
    // can be diffed against the same surface.
    const s = new KokoroSpeaker();
    await expect(s.speak("anything")).resolves.toBeUndefined();
    expect(() => s.cancel()).not.toThrow();
    expect(() => s.dispose()).not.toThrow();
  });
});
