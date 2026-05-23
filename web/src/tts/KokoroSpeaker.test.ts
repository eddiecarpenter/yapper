/**
 * Unit tests for `KokoroSpeaker`.
 *
 * Coverage layers across the feature's four tasks:
 *   - Task 1: backend-selection branches and the canonical
 *     provider log line. Does not touch the Kokoro pipeline.
 *   - Task 2 (this layer): lazy model loading + loading-state
 *     observable + load-error classification. The Transformers.js
 *     `pipeline` factory is mocked at the module boundary via
 *     `vi.mock` so tests do not download ~80 MB of model weights and
 *     do not depend on a browser environment that supports ONNX Web
 *     inference.
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

// `vi.mock` must execute before the module under test is imported, so
// we declare the mock first and use `vi.hoisted` to share a handle the
// per-test `beforeEach` can reconfigure. The hoisted handle holds a
// vitest-style mock function that stands in for `pipeline()` — each
// test rewires its implementation to return a resolved pipeline-like
// value or a rejected promise to drive the success/failure branches.
const mocks = vi.hoisted(() => {
  return {
    pipeline: vi.fn(),
  };
});

vi.mock("@huggingface/transformers", () => ({
  pipeline: mocks.pipeline,
}));

import { KOKORO_MODEL_ID, KokoroSpeaker } from "./KokoroSpeaker";
import type { LoadingState } from "./KokoroSpeaker";

/**
 * `navigator.gpu` is not declared on the default DOM lib's `Navigator`
 * type, so the stub goes through a typed cast. Using a sentinel object
 * (rather than `true`) matches the real shape closely enough — the
 * production code only checks for non-null truthiness — while keeping
 * the value distinct in case future tests need to round-trip it.
 */
type GpuLike = Record<string, never>;
const GPU_SENTINEL: GpuLike = Object.freeze({});

/**
 * Minimal stand-in for the real TTS pipeline object: callable and
 * disposable. Task 1/2 tests don't invoke the call path (the
 * loading-state machinery resolves without inspecting the output);
 * Task 3 tests do invoke it, so we accept an optional `output`
 * override that the production reducer will see verbatim. The
 * `calls` array captures every (text, options) pair passed in, so
 * Task 3 tests can assert the voice plumbing.
 */
function makeFakePipeline(
  output: unknown = {
    audio: new Float32Array(0),
    sampling_rate: 24000,
  },
): {
  pipeline: (text: string, options?: unknown) => Promise<unknown>;
  dispose: ReturnType<typeof vi.fn>;
  calls: Array<{ text: string; options: unknown }>;
} {
  const dispose = vi.fn().mockResolvedValue(undefined);
  const calls: Array<{ text: string; options: unknown }> = [];
  // The callable + dispose-bearing pipeline shape Transformers.js
  // returns. We attach `dispose` to the callable so the production
  // code's `p.dispose()` works against the stub the same way.
  const fn = (async (text: string, options?: unknown) => {
    calls.push({ text, options });
    return output;
  }) as ((text: string, options?: unknown) => Promise<unknown>) & {
    dispose: ReturnType<typeof vi.fn>;
  };
  fn.dispose = dispose;
  return { pipeline: fn, dispose, calls };
}

describe("KokoroSpeaker — backend selection", () => {
  let logSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    // Spy on console.log so we can assert the provider line was emitted
    // exactly once, then restore in afterEach. mockImplementation is a
    // noop so the actual log does not noise up the test runner output.
    logSpy = vi.spyOn(console, "log").mockImplementation(() => undefined);
    mocks.pipeline.mockReset();
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

  it("cancel() and dispose() Task-1/2 stubs do not throw", () => {
    const s = new KokoroSpeaker();
    expect(() => s.cancel()).not.toThrow();
    expect(() => s.dispose()).not.toThrow();
  });
});

describe("KokoroSpeaker — lazy model loading + state observable", () => {
  beforeEach(() => {
    // Silence the provider log; this block does not assert on it.
    vi.spyOn(console, "log").mockImplementation(() => undefined);
    mocks.pipeline.mockReset();
  });

  it("starts in the 'idle' state with no pipeline construction attempted", () => {
    const s = new KokoroSpeaker();

    expect(s.getLoadingState()).toBe("idle");
    // The constructor must NOT touch the pipeline factory — that is the
    // whole point of lazy loading (AD-2 / §6.4 economy).
    expect(mocks.pipeline).not.toHaveBeenCalled();
  });

  it("transitions idle → loading → ready on first speak(), notifying listeners in order", async () => {
    const fake = makeFakePipeline();
    mocks.pipeline.mockResolvedValue(fake.pipeline);

    const s = new KokoroSpeaker();
    const seen: LoadingState[] = [];
    s.subscribe((st) => seen.push(st));

    // Task-2 speak() throws "not yet implemented" AFTER the load
    // succeeds — swallow the throw so we can assert the state
    // transitions, which is the actual Task-2 behaviour under test.
    await expect(s.speak("hello")).rejects.toThrow(/not yet implemented/);

    expect(seen).toEqual(["loading", "ready"]);
    expect(s.getLoadingState()).toBe("ready");
  });

  it("calls pipeline(...) with the canonical Kokoro model id and the selected device", async () => {
    const fake = makeFakePipeline();
    mocks.pipeline.mockResolvedValue(fake.pipeline);

    const s = new KokoroSpeaker();
    await expect(s.speak("hi")).rejects.toThrow(/not yet implemented/);

    // Argument tuple matches the Transformers.js pipeline signature:
    // (task, modelId, options) — verifying it pins both AD-10 (the
    // model) and the wiring of provider → device option.
    expect(mocks.pipeline).toHaveBeenCalledTimes(1);
    expect(mocks.pipeline).toHaveBeenCalledWith("text-to-speech", KOKORO_MODEL_ID, {
      device: "wasm",
    });
  });

  it("does not re-load the model on subsequent speak() calls", async () => {
    const fake = makeFakePipeline();
    mocks.pipeline.mockResolvedValue(fake.pipeline);

    const s = new KokoroSpeaker();
    await expect(s.speak("a")).rejects.toThrow(/not yet implemented/);
    const stateAfterFirst = s.getLoadingState();
    const seen: LoadingState[] = [];
    s.subscribe((st) => seen.push(st));

    await expect(s.speak("b")).rejects.toThrow(/not yet implemented/);
    await expect(s.speak("c")).rejects.toThrow(/not yet implemented/);

    // Three calls in total; pipeline factory invoked only once.
    expect(mocks.pipeline).toHaveBeenCalledTimes(1);
    expect(stateAfterFirst).toBe("ready");
    // No state transitions on subsequent calls → no listener notifications.
    expect(seen).toEqual([]);
    expect(s.getLoadingState()).toBe("ready");
  });

  it("coalesces concurrent speak() calls onto the same in-flight load", async () => {
    // The whole point of caching the *promise* (not the resolved value)
    // is that a second call while the first is still loading does not
    // kick off a second ~80 MB download. Drive that explicitly.
    const fake = makeFakePipeline();
    let resolve!: (p: typeof fake.pipeline) => void;
    mocks.pipeline.mockReturnValue(
      new Promise((r) => {
        resolve = r;
      }),
    );

    const s = new KokoroSpeaker();
    const a = s.speak("first");
    const b = s.speak("second");

    resolve(fake.pipeline);
    // Both calls share the in-flight load, then each independently
    // hits the Task-2 stub throw — collect both rejections to keep
    // the promise hygiene clean for the assertion at the end.
    await expect(a).rejects.toThrow(/not yet implemented/);
    await expect(b).rejects.toThrow(/not yet implemented/);

    expect(mocks.pipeline).toHaveBeenCalledTimes(1);
  });

  it("subscribe() returns an unsubscribe handle that stops further notifications", async () => {
    const fake = makeFakePipeline();
    mocks.pipeline.mockResolvedValue(fake.pipeline);

    const s = new KokoroSpeaker();
    const seen: LoadingState[] = [];
    const unsubscribe = s.subscribe((st) => seen.push(st));

    // Detach BEFORE the load triggers — must observe no states.
    unsubscribe();
    await expect(s.speak("hi")).rejects.toThrow(/not yet implemented/);

    expect(seen).toEqual([]);
    // And the loading state still settled correctly even with no listeners.
    expect(s.getLoadingState()).toBe("ready");
  });
});

describe("KokoroSpeaker — load error classification", () => {
  beforeEach(() => {
    vi.spyOn(console, "log").mockImplementation(() => undefined);
    mocks.pipeline.mockReset();
  });

  it("transitions to 'error' and rejects when the pipeline factory rejects", async () => {
    mocks.pipeline.mockRejectedValue(new Error("boom"));

    const s = new KokoroSpeaker();
    const seen: LoadingState[] = [];
    s.subscribe((st) => seen.push(st));

    await expect(s.speak("hi")).rejects.toThrow(/Failed to load Kokoro model/);
    expect(seen).toEqual(["loading", "error"]);
    expect(s.getLoadingState()).toBe("error");
  });

  it("classifies QuotaExceededError as a storage-quota failure", async () => {
    // jsdom does not provide the real DOMException constructor in all
    // versions, so we synthesise an Error whose .name matches what
    // browsers actually throw on IndexedDB quota exhaustion.
    const quotaErr = new Error("The quota has been exceeded.");
    quotaErr.name = "QuotaExceededError";
    mocks.pipeline.mockRejectedValue(quotaErr);

    const s = new KokoroSpeaker();

    await expect(s.speak("hi")).rejects.toThrow(/storage quota exceeded/);
  });

  it("classifies a TypeError with 'Failed to fetch' as a network failure", async () => {
    const netErr = new TypeError("Failed to fetch");
    mocks.pipeline.mockRejectedValue(netErr);

    const s = new KokoroSpeaker();

    await expect(s.speak("hi")).rejects.toThrow(/network unreachable/);
  });

  it("falls back to 'unknown' classification and surfaces the original message", async () => {
    mocks.pipeline.mockRejectedValue(new Error("some weird onnxruntime error"));

    const s = new KokoroSpeaker();

    await expect(s.speak("hi")).rejects.toThrow(/some weird onnxruntime error/);
  });

  it("allows a retry after a failure — the cached rejected promise is dropped", async () => {
    // First load fails; second load succeeds. Without dropping the
    // cached promise on failure the second call would short-circuit
    // to the same rejection forever.
    mocks.pipeline.mockRejectedValueOnce(new Error("transient"));
    const fake = makeFakePipeline();
    mocks.pipeline.mockResolvedValueOnce(fake.pipeline);

    const s = new KokoroSpeaker();
    await expect(s.speak("hi")).rejects.toThrow(/Failed to load Kokoro model/);
    // Second call now hits the success branch — and lands on the
    // Task-2 "not yet implemented" stub AFTER load succeeds.
    await expect(s.speak("hi")).rejects.toThrow(/not yet implemented/);

    expect(mocks.pipeline).toHaveBeenCalledTimes(2);
    expect(s.getLoadingState()).toBe("ready");
  });

  it("a listener that throws does not block notifications to other listeners", async () => {
    const fake = makeFakePipeline();
    mocks.pipeline.mockResolvedValue(fake.pipeline);

    const s = new KokoroSpeaker();
    const seen: LoadingState[] = [];
    s.subscribe(() => {
      throw new Error("buggy listener");
    });
    s.subscribe((st) => seen.push(st));

    await expect(s.speak("hi")).rejects.toThrow(/not yet implemented/);

    // Despite the first listener throwing, the second listener saw the
    // full transition sequence — the production code's try/catch
    // around each callback is what enforces this.
    expect(seen).toEqual(["loading", "ready"]);
  });
});
