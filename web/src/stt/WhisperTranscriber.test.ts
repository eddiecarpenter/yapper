/**
 * Unit tests for `WhisperTranscriber`.
 *
 * Task 1 scope — exercise the backend-selection branches and the
 * canonical provider log line. The ASR pipeline is not constructed
 * in Task 1, so these tests do not need to mock `@huggingface/
 * transformers`; that arrives in Task 2 along with the lazy-load
 * tests.
 *
 * jsdom does not expose `navigator.gpu`, so the WASM branch is the
 * natural default and the WebGPU branch must be exercised by stubbing
 * the field on the global `navigator` object and restoring afterwards.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { WhisperTranscriber } from "./WhisperTranscriber";

/**
 * `navigator.gpu` is not declared on the default DOM lib's `Navigator`
 * type, so the stub goes through a typed cast. Using a sentinel object
 * (rather than `true`) matches the real shape closely enough — the
 * production code only checks for non-null truthiness — while keeping
 * the value distinct in case future tests need to round-trip it.
 */
type GpuLike = Record<string, never>;
const GPU_SENTINEL: GpuLike = Object.freeze({});

describe("WhisperTranscriber — backend selection", () => {
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
    const t = new WhisperTranscriber();

    expect(t.getProvider()).toBe("wasm");
    expect(logSpy).toHaveBeenCalledTimes(1);
    expect(logSpy).toHaveBeenCalledWith("provider: wasm");
  });

  it("does not throw when navigator.gpu is undefined (WASM fallback is supported)", () => {
    // AC-3 explicitly says "no error thrown" on the WASM path.
    expect(() => new WhisperTranscriber()).not.toThrow();
  });

  it("selects webgpu and logs 'provider: webgpu' when navigator.gpu is present", () => {
    // Install a truthy `gpu` field for the duration of this test only.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (navigator as any).gpu = GPU_SENTINEL;

    const t = new WhisperTranscriber();

    expect(t.getProvider()).toBe("webgpu");
    expect(logSpy).toHaveBeenCalledTimes(1);
    expect(logSpy).toHaveBeenCalledWith("provider: webgpu");
  });

  it("logs the provider exactly once per construction", () => {
    // Two instances → two log lines, never more. Catches a future
    // refactor that accidentally calls the logger from getProvider()
    // or some other accessor.
    new WhisperTranscriber();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (navigator as any).gpu = GPU_SENTINEL;
    new WhisperTranscriber();

    expect(logSpy).toHaveBeenCalledTimes(2);
    expect(logSpy).toHaveBeenNthCalledWith(1, "provider: wasm");
    expect(logSpy).toHaveBeenNthCalledWith(2, "provider: webgpu");
  });
});

describe("WhisperTranscriber — Task 1 stubs", () => {
  // Quiet the constructor log; this block does not assert on it.
  beforeEach(() => {
    vi.spyOn(console, "log").mockImplementation(() => undefined);
  });

  it("transcribe() resolves with the empty string (Task 1 stub)", async () => {
    const t = new WhisperTranscriber();

    // The body of transcribe is filled in by Task 3; until then the
    // stub honours the Transcriber contract by resolving with "".
    await expect(t.transcribe(new Float32Array(0), 16000)).resolves.toBe("");
  });

  it("dispose() is a no-op that does not throw (Task 1 stub)", () => {
    const t = new WhisperTranscriber();

    expect(() => t.dispose()).not.toThrow();
  });
});
