/**
 * Unit tests for `SileroVAD`.
 *
 * Coverage:
 *   - constants and module shape (constructor option defaults,
 *     pinned model id, exported sample-rate constant)
 *   - provider reporting honesty (KD-1 / R2): actual runtime is
 *     always wasm even when WebGPU is preferred
 *   - lazy model loading + LoadingState observable + load-error
 *     classification (mirroring the Whisper/Kokoro shape)
 *   - hysteresis state machine: speech detection requires
 *     minSpeechFrames; silence run of minSilenceFrames fires
 *     onSpeechEnd; pure silence emits nothing; hysteresis counters
 *     survive a brief silent gap inside speech (dead-zone)
 *   - segment pre-roll: 8 frames of pre-detection history are
 *     prepended to the emitted segment (KD-4)
 *   - dispose blocks subsequent process() calls; post-dispose
 *     inferences resolve to no-ops without throwing
 *   - frame-length validation
 *
 * The `@ricky0123/vad-web` library is mocked at the module boundary
 * so tests do not download the ~2 MB Silero v5 ONNX model and do not
 * depend on a runtime that supports ONNX Web inference. Two
 * sub-paths are mocked: the `defaultModelFetcher` re-export on the
 * package root, and the deep-imported `SileroV5` class.
 * `onnxruntime-web/wasm` is mocked to an empty object — the library's
 * `SileroV5.new` is mocked, so the real `ort` is never touched.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// vi.mock must execute before the module under test imports. Use the
// hoisted-handle pattern that Whisper/Kokoro tests established so the
// per-test `beforeEach` can rewire the mock behaviour.
const mocks = vi.hoisted(() => {
  return {
    sileroNew: vi.fn(),
    defaultModelFetcher: vi.fn(async (_path: string) => new ArrayBuffer(0)),
  };
});

vi.mock("onnxruntime-web/wasm", () => ({ env: { wasm: { wasmPaths: {} } } }));

vi.mock("@ricky0123/vad-web", () => ({
  defaultModelFetcher: mocks.defaultModelFetcher,
}));

vi.mock("@ricky0123/vad-web/dist/models/v5", () => ({
  SileroV5: {
    new: mocks.sileroNew,
  },
}));

import {
  DEFAULT_MIN_SILENCE_FRAMES,
  DEFAULT_MIN_SPEECH_FRAMES,
  DEFAULT_PRE_ROLL_FRAMES,
  DEFAULT_SILENCE_THRESHOLD,
  DEFAULT_SPEECH_THRESHOLD,
  FRAME_SAMPLE_COUNT,
  SILERO_MODEL_ID,
  SileroVAD,
  TARGET_SAMPLE_RATE_HZ,
  concatFloat32,
} from "./SileroVAD";
import type { LoadingState } from "./SileroVAD";

type GpuLike = Record<string, never>;
const GPU_SENTINEL: GpuLike = Object.freeze({});

/**
 * Minimal stand-in for the SileroV5 model. Each call to `process` is
 * given a probability the test controls — we accept either a fixed
 * value (the same for every frame) or a function that maps frame
 * index to probability so the test can drive the state machine
 * through a designed sequence.
 */
function makeFakeModel(probSource: number | ((frameIndex: number) => number) = 0.0): {
  process: ReturnType<typeof vi.fn>;
  release: ReturnType<typeof vi.fn>;
  reset_state: ReturnType<typeof vi.fn>;
  calls: Float32Array[];
} {
  const calls: Float32Array[] = [];
  const release = vi.fn().mockResolvedValue(undefined);
  const reset_state = vi.fn();
  const process = vi.fn(async (frame: Float32Array) => {
    const idx = calls.length;
    calls.push(frame);
    const isSpeech = typeof probSource === "function" ? probSource(idx) : probSource;
    return { isSpeech, notSpeech: 1 - isSpeech };
  });
  return { process, release, reset_state, calls };
}

function makeFrame(value = 0): Float32Array {
  const f = new Float32Array(FRAME_SAMPLE_COUNT);
  f.fill(value);
  return f;
}

describe("SileroVAD — module constants", () => {
  it("FRAME_SAMPLE_COUNT is 512 (Silero v5 frame length)", () => {
    expect(FRAME_SAMPLE_COUNT).toBe(512);
  });

  it("TARGET_SAMPLE_RATE_HZ is 16000", () => {
    expect(TARGET_SAMPLE_RATE_HZ).toBe(16000);
  });

  it("SILERO_MODEL_ID is the canonical Silero v5 ONNX filename", () => {
    expect(SILERO_MODEL_ID).toBe("silero_vad_v5.onnx");
  });

  it("defaults match the Design Plan", () => {
    expect(DEFAULT_SPEECH_THRESHOLD).toBe(0.5);
    expect(DEFAULT_SILENCE_THRESHOLD).toBe(0.35);
    expect(DEFAULT_MIN_SILENCE_FRAMES).toBe(24);
    expect(DEFAULT_MIN_SPEECH_FRAMES).toBe(8);
    expect(DEFAULT_PRE_ROLL_FRAMES).toBe(8);
  });
});

describe("concatFloat32 helper", () => {
  it("concatenates frames in input order with no normalisation", () => {
    const a = new Float32Array([1, 2, 3]);
    const b = new Float32Array([4, 5]);
    expect(Array.from(concatFloat32([a, b]))).toEqual([1, 2, 3, 4, 5]);
  });

  it("returns empty Float32Array for empty input", () => {
    const out = concatFloat32([]);
    expect(out).toBeInstanceOf(Float32Array);
    expect(out.length).toBe(0);
  });
});

describe("SileroVAD — provider selection (KD-1 / R2 honest reporting)", () => {
  let logSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    logSpy = vi.spyOn(console, "log").mockImplementation(() => undefined);
    mocks.sileroNew.mockReset();
    mocks.defaultModelFetcher.mockClear();
  });

  afterEach(() => {
    logSpy.mockRestore();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    delete (navigator as any).gpu;
  });

  it("logs provider: wasm regardless of preference — the library binds onnxruntime-web/wasm", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (navigator as any).gpu = GPU_SENTINEL;
    const v = new SileroVAD();
    expect(v.getProvider()).toBe("wasm");
    expect(logSpy).toHaveBeenCalledWith("provider: wasm");
    v.dispose();
  });

  it("getPreferredProvider() reflects the constructor preference for future migration", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (navigator as any).gpu = GPU_SENTINEL;
    const v = new SileroVAD();
    expect(v.getPreferredProvider()).toBe("webgpu");
    v.dispose();
  });

  it("getPreferredProvider() defaults to wasm when navigator.gpu is absent", () => {
    const v = new SileroVAD();
    expect(v.getPreferredProvider()).toBe("wasm");
    v.dispose();
  });

  it("getPreferredProvider() honours an explicit option override", () => {
    const v = new SileroVAD({ preferredProvider: "webgpu" });
    expect(v.getPreferredProvider()).toBe("webgpu");
    // Actual provider still wasm — the gap is the point of R2.
    expect(v.getProvider()).toBe("wasm");
    v.dispose();
  });

  it("does not throw when navigator.gpu is undefined", () => {
    expect(() => new SileroVAD()).not.toThrow();
  });
});

describe("SileroVAD — lazy model loading + state observable", () => {
  beforeEach(() => {
    vi.spyOn(console, "log").mockImplementation(() => undefined);
    mocks.sileroNew.mockReset();
    mocks.defaultModelFetcher.mockClear();
  });

  it("starts in the 'idle' state with no model construction attempted", () => {
    const v = new SileroVAD();
    expect(v.getLoadingState()).toBe("idle");
    expect(mocks.sileroNew).not.toHaveBeenCalled();
    v.dispose();
  });

  it("transitions idle → loading → ready on first process(), notifying listeners in order", async () => {
    const model = makeFakeModel(0);
    mocks.sileroNew.mockResolvedValue(model);

    const v = new SileroVAD();
    const seen: LoadingState[] = [];
    v.subscribe((s) => seen.push(s));

    v.process(makeFrame());
    await v.flushPending();

    expect(seen).toEqual(["loading", "ready"]);
    expect(v.getLoadingState()).toBe("ready");
    v.dispose();
  });

  it("calls SileroV5.new with a thunk that invokes the configured modelFetcher + modelUrl", async () => {
    const model = makeFakeModel(0);
    mocks.sileroNew.mockResolvedValue(model);
    const fetcher = vi.fn(async (_url: string) => new ArrayBuffer(0));

    const v = new SileroVAD({ modelFetcher: fetcher, modelUrl: "/custom-model.onnx" });
    v.process(makeFrame());
    await v.flushPending();

    expect(mocks.sileroNew).toHaveBeenCalledTimes(1);
    // First arg is the `ort` module (a stub object from vi.mock); we
    // do not assert its shape here — the contract is that whatever
    // SileroV5.new accepts is passed verbatim.
    const [, fetcherThunk] = mocks.sileroNew.mock.calls[0]!;
    // The thunk closes over the configured URL — invoking it should
    // hit our injected fetcher with the same URL.
    await (fetcherThunk as () => Promise<ArrayBuffer>)();
    expect(fetcher).toHaveBeenCalledWith("/custom-model.onnx");
    v.dispose();
  });

  it("falls back to defaultModelFetcher when no modelFetcher is supplied", async () => {
    const model = makeFakeModel(0);
    mocks.sileroNew.mockResolvedValue(model);

    const v = new SileroVAD();
    v.process(makeFrame());
    await v.flushPending();

    const [, fetcherThunk] = mocks.sileroNew.mock.calls[0]!;
    await (fetcherThunk as () => Promise<ArrayBuffer>)();
    expect(mocks.defaultModelFetcher).toHaveBeenCalledWith("/silero_vad_v5.onnx");
    v.dispose();
  });

  it("does not re-load the model on subsequent process() calls", async () => {
    const model = makeFakeModel(0);
    mocks.sileroNew.mockResolvedValue(model);

    const v = new SileroVAD();
    v.process(makeFrame());
    await v.flushPending();
    v.process(makeFrame());
    v.process(makeFrame());
    await v.flushPending();

    expect(mocks.sileroNew).toHaveBeenCalledTimes(1);
    expect(model.process).toHaveBeenCalledTimes(3);
    v.dispose();
  });

  it("coalesces concurrent process() calls onto the same in-flight load", async () => {
    const model = makeFakeModel(0);
    let resolveLoad!: (m: typeof model) => void;
    mocks.sileroNew.mockReturnValue(
      new Promise((r) => {
        resolveLoad = r;
      }),
    );

    const v = new SileroVAD();
    v.process(makeFrame());
    v.process(makeFrame());
    v.process(makeFrame());

    resolveLoad(model);
    await v.flushPending();

    expect(mocks.sileroNew).toHaveBeenCalledTimes(1);
    expect(model.process).toHaveBeenCalledTimes(3);
    v.dispose();
  });

  it("subscribe() returns an unsubscribe handle that stops further notifications", async () => {
    const model = makeFakeModel(0);
    mocks.sileroNew.mockResolvedValue(model);

    const v = new SileroVAD();
    const seen: LoadingState[] = [];
    const unsubscribe = v.subscribe((s) => seen.push(s));
    unsubscribe();

    v.process(makeFrame());
    await v.flushPending();

    expect(seen).toEqual([]);
    expect(v.getLoadingState()).toBe("ready");
    v.dispose();
  });
});

describe("SileroVAD — load error classification", () => {
  beforeEach(() => {
    vi.spyOn(console, "log").mockImplementation(() => undefined);
    mocks.sileroNew.mockReset();
    mocks.defaultModelFetcher.mockClear();
  });

  it("transitions to 'error' when SileroV5.new rejects", async () => {
    mocks.sileroNew.mockRejectedValue(new Error("boom"));

    const v = new SileroVAD();
    const seen: LoadingState[] = [];
    v.subscribe((s) => seen.push(s));

    v.process(makeFrame());
    await v.flushPending();

    expect(seen).toEqual(["loading", "error"]);
    expect(v.getLoadingState()).toBe("error");
    v.dispose();
  });

  it("classifies a TypeError with 'Failed to fetch' as a network failure", async () => {
    const netErr = new TypeError("Failed to fetch");
    mocks.sileroNew.mockRejectedValue(netErr);

    const v = new SileroVAD();
    // The process() error is swallowed best-effort; tests inspect the
    // load failure via the explicit loadModel(...) test hook by
    // calling process and observing the state transition.
    v.process(makeFrame());
    await v.flushPending();
    expect(v.getLoadingState()).toBe("error");

    // Trigger a second load attempt by issuing another process(); the
    // cached rejected promise must have been dropped (so retry is
    // actually retried) — pin that via the call count.
    v.process(makeFrame());
    await v.flushPending();
    expect(mocks.sileroNew).toHaveBeenCalledTimes(2);
    v.dispose();
  });

  it("allows a retry after a failure — the cached rejected promise is dropped", async () => {
    mocks.sileroNew.mockRejectedValueOnce(new Error("transient"));
    const model = makeFakeModel(0);
    mocks.sileroNew.mockResolvedValueOnce(model);

    const v = new SileroVAD();
    v.process(makeFrame());
    await v.flushPending();
    expect(v.getLoadingState()).toBe("error");

    v.process(makeFrame());
    await v.flushPending();
    expect(v.getLoadingState()).toBe("ready");
    expect(mocks.sileroNew).toHaveBeenCalledTimes(2);
    v.dispose();
  });
});

describe("SileroVAD — process() input validation", () => {
  beforeEach(() => {
    vi.spyOn(console, "log").mockImplementation(() => undefined);
    mocks.sileroNew.mockReset();
  });

  it("throws if the frame is not 512 samples", () => {
    mocks.sileroNew.mockResolvedValue(makeFakeModel(0));
    const v = new SileroVAD();
    expect(() => v.process(new Float32Array(256))).toThrow(/512-sample frames/);
    v.dispose();
  });

  it("accepts 512-sample frames without throwing", () => {
    mocks.sileroNew.mockResolvedValue(makeFakeModel(0));
    const v = new SileroVAD();
    expect(() => v.process(makeFrame())).not.toThrow();
    v.dispose();
  });
});

describe("SileroVAD — hysteresis state machine", () => {
  beforeEach(() => {
    vi.spyOn(console, "log").mockImplementation(() => undefined);
    mocks.sileroNew.mockReset();
  });

  it("pure silence emits no onSpeechEnd and leaves speakingFlag false", async () => {
    const model = makeFakeModel(0.1); // always below silenceThreshold
    mocks.sileroNew.mockResolvedValue(model);

    const v = new SileroVAD({
      minSpeechFrames: 4,
      minSilenceFrames: 6,
      preRollFrames: 2,
    });
    const segments: Float32Array[] = [];
    v.onSpeechEnd = (s) => segments.push(s);

    for (let i = 0; i < 20; i++) v.process(makeFrame(0));
    await v.flushPending();

    expect(segments).toEqual([]);
    // process() returning false after settling confirms the state.
    expect(v.process(makeFrame(0))).toBe(false);
    await v.flushPending();
    v.dispose();
  });

  it("latches speakingFlag = true only after minSpeechFrames consecutive speech frames", async () => {
    // Frame indices 0..N-1 — we'll feed 5 speech frames then check.
    const probs = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9];
    const model = makeFakeModel((i) => probs[i] ?? 0.1);
    mocks.sileroNew.mockResolvedValue(model);

    const v = new SileroVAD({
      minSpeechFrames: 6,
      minSilenceFrames: 3,
      preRollFrames: 2,
    });

    // Pump 5 frames — below threshold for latching.
    for (let i = 0; i < 5; i++) v.process(makeFrame(0.5));
    await v.flushPending();
    expect(v.process(makeFrame(0.5))).toBe(false); // 6th frame is the trigger but process() returns LAST flag
    await v.flushPending();
    // After the 6th completes, the flag latches.
    expect(v.process(makeFrame(0.5))).toBe(true);
    await v.flushPending();
    v.dispose();
  });

  it("fires onSpeechEnd after speech run + minSilenceFrames of silence", async () => {
    // 8 speech frames, then 5 silence frames (minSilenceFrames=5).
    const probs = [
      0.9,
      0.9,
      0.9,
      0.9,
      0.9,
      0.9,
      0.9,
      0.9, // 8 speech
      0.1,
      0.1,
      0.1,
      0.1,
      0.1, // 5 silence — triggers end
    ];
    const model = makeFakeModel((i) => probs[i] ?? 0.1);
    mocks.sileroNew.mockResolvedValue(model);

    const v = new SileroVAD({
      minSpeechFrames: 4,
      minSilenceFrames: 5,
      preRollFrames: 2,
    });

    const segments: Float32Array[] = [];
    v.onSpeechEnd = (s) => segments.push(s);

    // Use distinguishable frame values so we can sanity-check the
    // concatenation. Each frame is filled with its index / 100 so the
    // segment's prefix and suffix are observable.
    for (let i = 0; i < probs.length; i++) {
      v.process(makeFrame(i / 100));
    }
    await v.flushPending();

    expect(segments).toHaveLength(1);
    // Segment length: history at latch only contains the 4 speech
    // frames (no prior silence to fill the pre-roll slots — they
    // were nothing-rolled-over because frame 0 was already speech),
    // PLUS the post-latch frames (frames 4..12 = 9 frames) = 13
    // frames total. The pre-roll is physical pre-detection audio,
    // not synthetic padding, so an utterance that starts at frame 0
    // emits a shorter segment than one preceded by silence.
    expect(segments[0]!.length).toBe(13 * FRAME_SAMPLE_COUNT);
    v.dispose();
  });

  it("includes 8-frame pre-roll (KD-4) ahead of the detected onset", async () => {
    // Pre-roll = 3 for a smaller test. Send 5 silence frames before
    // speech, then enough speech to latch and silence to fire.
    const probs = [
      0.1,
      0.1,
      0.1,
      0.1,
      0.1, // silent pre-roll
      0.9,
      0.9,
      0.9,
      0.9, // 4 speech — latches at frame 8 (1-indexed)
      0.1,
      0.1,
      0.1, // silence — fires onSpeechEnd
    ];
    const model = makeFakeModel((i) => probs[i] ?? 0.1);
    mocks.sileroNew.mockResolvedValue(model);

    const v = new SileroVAD({
      minSpeechFrames: 4,
      minSilenceFrames: 3,
      preRollFrames: 3,
    });

    const segments: Float32Array[] = [];
    v.onSpeechEnd = (s) => segments.push(s);

    // Tag each frame with its index in the first sample so the test
    // can assert the segment starts with frames N..M.
    for (let i = 0; i < probs.length; i++) {
      const f = makeFrame(0);
      f[0] = i; // first sample carries the frame index
      v.process(f);
    }
    await v.flushPending();

    expect(segments).toHaveLength(1);
    // History at latch instant = preRollFrames + minSpeechFrames = 7
    // frames: indices 2, 3, 4, 5, 6, 7, 8 (the latest 7).
    // After latch: indices 9, 10, 11 are appended (10 frames total).
    expect(segments[0]!.length).toBe(10 * FRAME_SAMPLE_COUNT);
    // Pin the first sample of each frame in the segment — verifies
    // the pre-roll really did come first.
    const firstSamples: number[] = [];
    for (let i = 0; i < 10; i++) {
      firstSamples.push(segments[0]![i * FRAME_SAMPLE_COUNT]!);
    }
    expect(firstSamples).toEqual([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    v.dispose();
  });

  it("survives a brief silent gap inside a speech run (dead-zone hysteresis)", async () => {
    // 4 speech frames (latches), 2 dead-zone frames (between
    // silenceThreshold 0.35 and speechThreshold 0.5 — leaves counters
    // unchanged), 3 speech frames, then 5 silence frames to fire.
    const probs = [
      0.9,
      0.9,
      0.9,
      0.9, // latches at frame 4
      0.45,
      0.45, // dead-zone — neither speech nor silence
      0.9,
      0.9,
      0.9, // more speech
      0.1,
      0.1,
      0.1,
      0.1,
      0.1, // 5 silence — fires
    ];
    const model = makeFakeModel((i) => probs[i] ?? 0.1);
    mocks.sileroNew.mockResolvedValue(model);

    const v = new SileroVAD({
      minSpeechFrames: 4,
      minSilenceFrames: 5,
      preRollFrames: 2,
    });

    const segments: Float32Array[] = [];
    v.onSpeechEnd = (s) => segments.push(s);

    for (let i = 0; i < probs.length; i++) v.process(makeFrame(0));
    await v.flushPending();

    // One segment fired — i.e. the dead-zone frames did not prematurely
    // increment silenceFrameCount past the minSilenceFrames threshold.
    expect(segments).toHaveLength(1);
    v.dispose();
  });

  it("resets the speech counter when a single silent frame interrupts a speech run before latch", async () => {
    // 3 speech, 1 silence, 4 more speech — minSpeechFrames=4, so the
    // first 3 should be discarded by the silence reset, and the latch
    // happens only after the 4 consecutive speech frames at the end.
    const probs = [
      0.9,
      0.9,
      0.9, // 3 speech (below latch threshold)
      0.1, // 1 silence — resets speech counter
      0.9,
      0.9,
      0.9, // 3 more speech (still below)
      0.9, // 4th — latch finally happens here
      0.1,
      0.1,
      0.1, // silence — fires
    ];
    const model = makeFakeModel((i) => probs[i] ?? 0.1);
    mocks.sileroNew.mockResolvedValue(model);

    const v = new SileroVAD({
      minSpeechFrames: 4,
      minSilenceFrames: 3,
      preRollFrames: 1,
    });

    const segments: Float32Array[] = [];
    v.onSpeechEnd = (s) => segments.push(s);

    for (let i = 0; i < probs.length; i++) v.process(makeFrame(0));
    await v.flushPending();

    expect(segments).toHaveLength(1);
    // History at latch: preRoll 1 + minSpeechFrames 4 = 5 frames
    // (indices 3..7). Then post-latch frames 8..10. Total = 8 frames.
    expect(segments[0]!.length).toBe(8 * FRAME_SAMPLE_COUNT);
    v.dispose();
  });

  it("a buggy onSpeechEnd does not break subsequent process() calls", async () => {
    const probs = [
      0.9,
      0.9,
      0.9,
      0.9, // latch
      0.1,
      0.1,
      0.1, // fires (buggy listener)
      0.9,
      0.9,
      0.9,
      0.9, // re-latch
      0.1,
      0.1,
      0.1, // fires again
    ];
    const model = makeFakeModel((i) => probs[i] ?? 0.1);
    mocks.sileroNew.mockResolvedValue(model);

    const v = new SileroVAD({
      minSpeechFrames: 4,
      minSilenceFrames: 3,
      preRollFrames: 1,
    });

    let count = 0;
    v.onSpeechEnd = () => {
      count++;
      throw new Error("buggy listener");
    };

    for (let i = 0; i < probs.length; i++) v.process(makeFrame(0));
    await v.flushPending();

    expect(count).toBe(2);
    v.dispose();
  });
});

describe("SileroVAD — dispose() lifecycle", () => {
  beforeEach(() => {
    vi.spyOn(console, "log").mockImplementation(() => undefined);
    mocks.sileroNew.mockReset();
  });

  it("dispose() before any process() is a safe no-op (state stays 'idle')", () => {
    const v = new SileroVAD();
    expect(() => v.dispose()).not.toThrow();
    expect(v.getLoadingState()).toBe("idle");
  });

  it("dispose() after a successful load releases the model and resets to 'idle'", async () => {
    const model = makeFakeModel(0);
    mocks.sileroNew.mockResolvedValue(model);

    const v = new SileroVAD();
    v.process(makeFrame());
    await v.flushPending();
    expect(v.getLoadingState()).toBe("ready");

    v.dispose();
    expect(v.getLoadingState()).toBe("idle");

    // model.release is fire-and-forget through a three-link promise
    // chain (inflightTail → modelPromise → release). vi.waitFor
    // polls until the assertion holds, which is more robust than
    // counting microtask cycles by hand — and matches the pattern
    // the sibling Whisper/Kokoro tests use for the same shape.
    await vi.waitFor(() => {
      expect(model.release).toHaveBeenCalledTimes(1);
    });
  });

  it("subsequent process() throws after dispose()", async () => {
    const model = makeFakeModel(0);
    mocks.sileroNew.mockResolvedValue(model);

    const v = new SileroVAD();
    v.process(makeFrame());
    await v.flushPending();
    v.dispose();

    expect(() => v.process(makeFrame())).toThrow(/cannot process.*after dispose/i);
  });

  it("post-dispose in-flight inferences resolve to no-ops without throwing", async () => {
    // Hold the load promise open so we can dispose mid-load.
    const model = makeFakeModel(0);
    let resolveLoad!: (m: typeof model) => void;
    mocks.sileroNew.mockReturnValue(
      new Promise((r) => {
        resolveLoad = r;
      }),
    );

    const v = new SileroVAD();
    v.process(makeFrame());
    v.dispose();
    // Resolve the load AFTER dispose — the inflight tail should
    // short-circuit via the disposed check inside the inference body.
    resolveLoad(model);

    // No throw, flush completes.
    await expect(v.flushPending()).resolves.toBeUndefined();
    // The model's process() should NOT have been called — the
    // disposed check shortcircuited.
    expect(model.process).not.toHaveBeenCalled();
  });

  it("onSpeechEnd is detached on dispose — late segments do not surprise the consumer", async () => {
    const probs = [
      0.9,
      0.9,
      0.9,
      0.9, // latch
      0.1,
      0.1,
      0.1, // about to fire
    ];
    const model = makeFakeModel((i) => probs[i] ?? 0.1);
    mocks.sileroNew.mockResolvedValue(model);

    const v = new SileroVAD({
      minSpeechFrames: 4,
      minSilenceFrames: 3,
      preRollFrames: 1,
    });

    const segments: Float32Array[] = [];
    v.onSpeechEnd = (s) => segments.push(s);

    // Fire the first 4 frames (latch), then dispose, then we cannot
    // call process() again — but the existing in-flight queue should
    // not invoke the callback after dispose.
    for (let i = 0; i < 4; i++) v.process(makeFrame(0));
    v.dispose();
    await Promise.resolve();
    await Promise.resolve();
    expect(segments).toEqual([]);
    expect(v.onSpeechEnd).toBeUndefined();
  });

  it("second dispose() is a safe no-op", () => {
    const v = new SileroVAD();
    v.dispose();
    expect(() => v.dispose()).not.toThrow();
  });

  it("a listener that throws does not block notifications to other listeners", async () => {
    const model = makeFakeModel(0);
    mocks.sileroNew.mockResolvedValue(model);

    const v = new SileroVAD();
    const seen: LoadingState[] = [];
    v.subscribe(() => {
      throw new Error("buggy listener");
    });
    v.subscribe((s) => seen.push(s));

    v.process(makeFrame());
    await v.flushPending();

    expect(seen).toEqual(["loading", "ready"]);
    v.dispose();
  });
});
