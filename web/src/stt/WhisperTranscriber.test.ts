/**
 * Unit tests for `WhisperTranscriber`.
 *
 * Coverage layers across the feature's three tasks:
 *   - Task 1: backend-selection branches and the canonical
 *     provider log line. Does not touch the ASR pipeline.
 *   - Task 2 (this layer): lazy model loading + loading-state
 *     observable + load-error classification + dispose lifecycle.
 *     The Transformers.js `pipeline` factory is mocked at the
 *     module boundary via `vi.mock` so tests do not download
 *     145 MB of model weights and do not depend on a browser
 *     environment that supports ONNX Web inference.
 *   - Task 3 will add the AC-1 fixture test asserting the post-load
 *     audio-to-text path.
 *
 * jsdom does not expose `navigator.gpu`, so the WASM branch is the
 * natural default and the WebGPU branch must be exercised by stubbing
 * the field on the global `navigator` object and restoring afterwards.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// `vi.mock` must execute before the module under test is imported, so
// we declare the mock first and use `vi.hoisted` to share a handle the
// per-test `beforeEach` can reconfigure. The hoisted handle holds a
// jest-style mock function that stands in for `pipeline()` — each test
// rewires its implementation to return a resolved pipeline-like value
// or a rejected promise to drive the success/failure branches.
const mocks = vi.hoisted(() => {
  return {
    pipeline: vi.fn(),
  };
});

vi.mock("@huggingface/transformers", () => ({
  pipeline: mocks.pipeline,
}));

import { WhisperTranscriber, WHISPER_MODEL_ID } from "./WhisperTranscriber";
import type { LoadingState } from "./WhisperTranscriber";

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
 * Minimal stand-in for the real ASR pipeline object: callable and
 * disposable. Tests do not invoke the call path (Task 3 covers that);
 * they only check that the load → ready transition completes and that
 * dispose() forwards through.
 */
function makeFakePipeline(): {
  pipeline: (audio: Float32Array) => Promise<{ text: string }>;
  dispose: ReturnType<typeof vi.fn>;
} {
  const dispose = vi.fn().mockResolvedValue(undefined);
  // The callable + dispose-bearing pipeline shape Transformers.js
  // returns. We attach `dispose` to the callable so the production
  // code's `p.dispose()` works against the stub the same way.
  const fn = (async (_audio: Float32Array) => ({ text: "" })) as ((
    audio: Float32Array,
  ) => Promise<{ text: string }>) & { dispose: ReturnType<typeof vi.fn> };
  fn.dispose = dispose;
  return { pipeline: fn, dispose };
}

describe("WhisperTranscriber — backend selection", () => {
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

describe("WhisperTranscriber — lazy model loading + state observable", () => {
  beforeEach(() => {
    // Silence the provider log; this block does not assert on it.
    vi.spyOn(console, "log").mockImplementation(() => undefined);
    mocks.pipeline.mockReset();
  });

  it("starts in the 'idle' state with no pipeline construction attempted", () => {
    const t = new WhisperTranscriber();

    expect(t.getLoadingState()).toBe("idle");
    // The constructor must NOT touch the pipeline factory — that is the
    // whole point of lazy loading (AD-2 / §6.3 economy).
    expect(mocks.pipeline).not.toHaveBeenCalled();
  });

  it("transitions idle → loading → ready on first transcribe(), notifying listeners in order", async () => {
    const fake = makeFakePipeline();
    mocks.pipeline.mockResolvedValue(fake.pipeline);

    const t = new WhisperTranscriber();
    const seen: LoadingState[] = [];
    t.subscribe((s) => seen.push(s));

    const result = await t.transcribe(new Float32Array(0), 16000);

    expect(seen).toEqual(["loading", "ready"]);
    expect(t.getLoadingState()).toBe("ready");
    // Task 2's transcribe body is still the empty-string stub — Task 3
    // wires the real return value. Asserting "" here pins the contract
    // explicitly so a Task-3 regression that returns the raw pipeline
    // output object is caught.
    expect(result).toBe("");
  });

  it("calls pipeline(...) with the canonical model id and the selected device", async () => {
    const fake = makeFakePipeline();
    mocks.pipeline.mockResolvedValue(fake.pipeline);

    const t = new WhisperTranscriber();
    await t.transcribe(new Float32Array(0), 16000);

    // Argument tuple matches the Transformers.js pipeline signature:
    // (task, modelId, options) — verifying it pins both AD-11 (the
    // model) and the wiring of provider → device option.
    expect(mocks.pipeline).toHaveBeenCalledTimes(1);
    expect(mocks.pipeline).toHaveBeenCalledWith(
      "automatic-speech-recognition",
      WHISPER_MODEL_ID,
      { device: "wasm" },
    );
  });

  it("does not re-load the model on subsequent transcribe() calls", async () => {
    const fake = makeFakePipeline();
    mocks.pipeline.mockResolvedValue(fake.pipeline);

    const t = new WhisperTranscriber();
    await t.transcribe(new Float32Array(0), 16000);
    const stateAfterFirst = t.getLoadingState();
    const seen: LoadingState[] = [];
    t.subscribe((s) => seen.push(s));

    await t.transcribe(new Float32Array(0), 16000);
    await t.transcribe(new Float32Array(0), 16000);

    // Three calls in total; pipeline factory invoked only once.
    expect(mocks.pipeline).toHaveBeenCalledTimes(1);
    expect(stateAfterFirst).toBe("ready");
    // No state transitions on subsequent calls → no listener notifications.
    expect(seen).toEqual([]);
    expect(t.getLoadingState()).toBe("ready");
  });

  it("coalesces concurrent transcribe() calls onto the same in-flight load", async () => {
    // The whole point of caching the *promise* (not the resolved value)
    // is that a second call while the first is still loading does not
    // kick off a second 145 MB download. Drive that explicitly.
    const fake = makeFakePipeline();
    let resolve!: (p: typeof fake.pipeline) => void;
    mocks.pipeline.mockReturnValue(
      new Promise((r) => {
        resolve = r;
      }),
    );

    const t = new WhisperTranscriber();
    const a = t.transcribe(new Float32Array(0), 16000);
    const b = t.transcribe(new Float32Array(0), 16000);

    resolve(fake.pipeline);
    await Promise.all([a, b]);

    expect(mocks.pipeline).toHaveBeenCalledTimes(1);
  });

  it("subscribe() returns an unsubscribe handle that stops further notifications", async () => {
    const fake = makeFakePipeline();
    mocks.pipeline.mockResolvedValue(fake.pipeline);

    const t = new WhisperTranscriber();
    const seen: LoadingState[] = [];
    const unsubscribe = t.subscribe((s) => seen.push(s));

    // Detach BEFORE the load triggers — must observe no states.
    unsubscribe();
    await t.transcribe(new Float32Array(0), 16000);

    expect(seen).toEqual([]);
    // And the loading state still settled correctly even with no listeners.
    expect(t.getLoadingState()).toBe("ready");
  });
});

describe("WhisperTranscriber — load error classification", () => {
  beforeEach(() => {
    vi.spyOn(console, "log").mockImplementation(() => undefined);
    mocks.pipeline.mockReset();
  });

  it("transitions to 'error' and rejects when the pipeline factory rejects", async () => {
    mocks.pipeline.mockRejectedValue(new Error("boom"));

    const t = new WhisperTranscriber();
    const seen: LoadingState[] = [];
    t.subscribe((s) => seen.push(s));

    await expect(t.transcribe(new Float32Array(0), 16000)).rejects.toThrow(
      /Failed to load Whisper model/,
    );
    expect(seen).toEqual(["loading", "error"]);
    expect(t.getLoadingState()).toBe("error");
  });

  it("classifies QuotaExceededError as a storage-quota failure", async () => {
    // jsdom does not provide the real DOMException constructor in all
    // versions, so we synthesise an Error whose .name matches what
    // browsers actually throw on IndexedDB quota exhaustion.
    const quotaErr = new Error("The quota has been exceeded.");
    quotaErr.name = "QuotaExceededError";
    mocks.pipeline.mockRejectedValue(quotaErr);

    const t = new WhisperTranscriber();

    await expect(t.transcribe(new Float32Array(0), 16000)).rejects.toThrow(
      /storage quota exceeded/,
    );
  });

  it("classifies a TypeError with 'Failed to fetch' as a network failure", async () => {
    const netErr = new TypeError("Failed to fetch");
    mocks.pipeline.mockRejectedValue(netErr);

    const t = new WhisperTranscriber();

    await expect(t.transcribe(new Float32Array(0), 16000)).rejects.toThrow(
      /network unreachable/,
    );
  });

  it("falls back to 'unknown' classification and surfaces the original message", async () => {
    mocks.pipeline.mockRejectedValue(new Error("some weird onnxruntime error"));

    const t = new WhisperTranscriber();

    await expect(t.transcribe(new Float32Array(0), 16000)).rejects.toThrow(
      /some weird onnxruntime error/,
    );
  });

  it("allows a retry after a failure — the cached rejected promise is dropped", async () => {
    // First load fails; second load succeeds. Without dropping the
    // cached promise on failure the second call would short-circuit
    // to the same rejection forever.
    mocks.pipeline.mockRejectedValueOnce(new Error("transient"));
    const fake = makeFakePipeline();
    mocks.pipeline.mockResolvedValueOnce(fake.pipeline);

    const t = new WhisperTranscriber();
    await expect(t.transcribe(new Float32Array(0), 16000)).rejects.toThrow();
    await expect(t.transcribe(new Float32Array(0), 16000)).resolves.toBe("");

    expect(mocks.pipeline).toHaveBeenCalledTimes(2);
    expect(t.getLoadingState()).toBe("ready");
  });
});

describe("WhisperTranscriber — dispose lifecycle", () => {
  beforeEach(() => {
    vi.spyOn(console, "log").mockImplementation(() => undefined);
    mocks.pipeline.mockReset();
  });

  it("dispose() before any transcribe() is a safe no-op (state stays 'idle')", () => {
    const t = new WhisperTranscriber();

    expect(() => t.dispose()).not.toThrow();
    expect(t.getLoadingState()).toBe("idle");
  });

  it("dispose() after a successful load releases the pipeline and resets to 'idle'", async () => {
    const fake = makeFakePipeline();
    mocks.pipeline.mockResolvedValue(fake.pipeline);

    const t = new WhisperTranscriber();
    await t.transcribe(new Float32Array(0), 16000);
    expect(t.getLoadingState()).toBe("ready");

    t.dispose();
    expect(t.getLoadingState()).toBe("idle");

    // .dispose on the underlying pipeline is fire-and-forget; we yield
    // a microtask so the .then in the production code can run.
    await Promise.resolve();
    await Promise.resolve();
    expect(fake.dispose).toHaveBeenCalledTimes(1);
  });

  it("dispose() drops listeners — a post-dispose load triggers no notifications on prior subscribers", async () => {
    const fake = makeFakePipeline();
    mocks.pipeline.mockResolvedValue(fake.pipeline);

    const t = new WhisperTranscriber();
    const seen: LoadingState[] = [];
    t.subscribe((s) => seen.push(s));

    // Load once so the listener has something to observe, then dispose
    // and load again.
    await t.transcribe(new Float32Array(0), 16000);
    expect(seen).toEqual(["loading", "ready"]);

    t.dispose();
    // Subsequent load should NOT notify the dropped listener.
    await t.transcribe(new Float32Array(0), 16000);
    expect(seen).toEqual(["loading", "ready"]);
    expect(t.getLoadingState()).toBe("ready");
  });

  it("a listener that throws does not block notifications to other listeners", async () => {
    const fake = makeFakePipeline();
    mocks.pipeline.mockResolvedValue(fake.pipeline);

    const t = new WhisperTranscriber();
    const seen: LoadingState[] = [];
    t.subscribe(() => {
      throw new Error("buggy listener");
    });
    t.subscribe((s) => seen.push(s));

    await t.transcribe(new Float32Array(0), 16000);

    // Despite the first listener throwing, the second listener saw the
    // full transition sequence — the production code's try/catch
    // around each callback is what enforces this.
    expect(seen).toEqual(["loading", "ready"]);
  });
});
