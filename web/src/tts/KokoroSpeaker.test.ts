/**
 * Unit tests for `KokoroSpeaker`.
 *
 * Coverage layers across the feature's four tasks:
 *   - Task 1: backend-selection branches and the canonical
 *     provider log line. Does not touch the Kokoro pipeline.
 *   - Task 2: lazy model loading + loading-state observable +
 *     load-error classification. The Transformers.js `pipeline`
 *     factory is mocked at the module boundary via `vi.mock` so
 *     tests do not download ~80 MB of model weights and do not depend
 *     on a browser environment that supports ONNX Web inference.
 *   - Task 3 (this layer): synthesis + Web Audio playback +
 *     voice option (AC-1, AC-3). jsdom does not implement the Web
 *     Audio API, so an `AudioContext` stub is installed on
 *     `globalThis` before each test and the buffer-source's
 *     `onended` event is driven manually where timing matters.
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

import {
  DEFAULT_VOICE,
  KOKORO_MODEL_ID,
  KokoroSpeaker,
  extractSynthesisOutput,
} from "./KokoroSpeaker";
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
 * disposable. The callable accepts `(text, options)` and returns a
 * shape that matches Transformers.js v3's `TextToAudioOutput`
 * (`{ audio: Float32Array, sampling_rate: number }`). The `calls`
 * array captures every (text, options) tuple so Task 3 tests can
 * assert voice plumbing.
 */
function makeFakePipeline(
  output: unknown = {
    audio: new Float32Array(240), // 10 ms at 24 kHz — non-empty, valid buffer
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

/**
 * AudioContext / AudioBufferSourceNode / AudioBuffer stubs.
 *
 * jsdom does not implement the Web Audio API, so these mocks stand in
 * for the real implementation. The default behaviour is "auto-fire
 * `onended` on the next microtask after `start()`" — most tests do
 * not care about timing and want playback to settle naturally. For
 * the AC-1 timing test we override that behaviour via
 * `autoEnd: false` so the test can drive `onended` manually.
 */
type FakeSource = {
  buffer: FakeBuffer | null;
  onended: (() => void) | null;
  started: boolean;
  stopped: boolean;
  connect: ReturnType<typeof vi.fn>;
  start: ReturnType<typeof vi.fn>;
  stop: ReturnType<typeof vi.fn>;
};

type FakeBuffer = {
  numberOfChannels: number;
  length: number;
  sampleRate: number;
  duration: number;
  copyToChannel: ReturnType<typeof vi.fn>;
  channels: Float32Array[];
};

type FakeContextState = {
  contexts: FakeContext[];
  sources: FakeSource[];
  buffers: FakeBuffer[];
  /** When false, source.start() does NOT auto-fire onended. */
  autoEnd: boolean;
};

type FakeContext = {
  destination: object;
  close: ReturnType<typeof vi.fn>;
  createBuffer: ReturnType<typeof vi.fn>;
  createBufferSource: ReturnType<typeof vi.fn>;
};

function installAudioContext(): FakeContextState {
  const state: FakeContextState = {
    contexts: [],
    sources: [],
    buffers: [],
    autoEnd: true,
  };

  function makeBuffer(numberOfChannels: number, length: number, sampleRate: number): FakeBuffer {
    const channels: Float32Array[] = [];
    for (let i = 0; i < numberOfChannels; i++) {
      channels.push(new Float32Array(length));
    }
    const buffer: FakeBuffer = {
      numberOfChannels,
      length,
      sampleRate,
      duration: length / sampleRate,
      channels,
      copyToChannel: vi.fn((data: Float32Array, channel: number) => {
        channels[channel]?.set(data);
      }),
    };
    state.buffers.push(buffer);
    return buffer;
  }

  function makeSource(): FakeSource {
    const source: FakeSource = {
      buffer: null,
      onended: null,
      started: false,
      stopped: false,
      connect: vi.fn(),
      start: vi.fn(() => {
        source.started = true;
        if (state.autoEnd) {
          // Fire onended on the next microtask so the awaiting code in
          // speak() has a chance to attach the handler before it
          // resolves — the real Web Audio path is also asynchronous.
          queueMicrotask(() => {
            if (!source.stopped && source.onended) {
              source.onended();
            }
          });
        }
      }),
      stop: vi.fn(() => {
        source.stopped = true;
        if (source.onended) {
          source.onended();
        }
      }),
    };
    state.sources.push(source);
    return source;
  }

  class MockAudioContext {
    destination = {};
    close = vi.fn().mockResolvedValue(undefined);
    createBuffer = vi.fn((numberOfChannels: number, length: number, sampleRate: number) =>
      makeBuffer(numberOfChannels, length, sampleRate),
    );
    createBufferSource = vi.fn(() => makeSource());

    constructor() {
      state.contexts.push(this as unknown as FakeContext);
    }
  }

  vi.stubGlobal("AudioContext", MockAudioContext);
  return state;
}

function stubFetch(): void {
  const embedBuffer = new Float32Array(1 * 101 * 128).buffer;
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      arrayBuffer: () => Promise.resolve(embedBuffer),
    }),
  );
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

  it("cancel() and dispose() Task-1/2/3 stubs do not throw", () => {
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
    installAudioContext();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
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

    await s.speak("hello");

    expect(seen).toEqual(["loading", "ready"]);
    expect(s.getLoadingState()).toBe("ready");
  });

  it("calls pipeline(...) with the canonical Kokoro model id and the selected device", async () => {
    const fake = makeFakePipeline();
    mocks.pipeline.mockResolvedValue(fake.pipeline);

    const s = new KokoroSpeaker();
    await s.speak("hi");

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
    await s.speak("a");
    const stateAfterFirst = s.getLoadingState();
    const seen: LoadingState[] = [];
    s.subscribe((st) => seen.push(st));

    await s.speak("b");
    await s.speak("c");

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
    await Promise.all([a, b]);

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
    await s.speak("hi");

    expect(seen).toEqual([]);
    // And the loading state still settled correctly even with no listeners.
    expect(s.getLoadingState()).toBe("ready");
  });
});

describe("KokoroSpeaker — load error classification", () => {
  beforeEach(() => {
    vi.spyOn(console, "log").mockImplementation(() => undefined);
    mocks.pipeline.mockReset();
    installAudioContext();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("transitions to 'error' and rejects when the pipeline factory rejects", async () => {
    mocks.pipeline.mockRejectedValue(new Error("boom"));

    const s = new KokoroSpeaker();
    const seen: LoadingState[] = [];
    s.subscribe((st) => seen.push(st));

    await expect(s.speak("hi")).rejects.toThrow(/Failed to load TTS model/);
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
    // First load fails; second load succeeds.
    mocks.pipeline.mockRejectedValueOnce(new Error("transient"));
    const fake = makeFakePipeline();
    mocks.pipeline.mockResolvedValueOnce(fake.pipeline);

    const s = new KokoroSpeaker();
    await expect(s.speak("hi")).rejects.toThrow(/Failed to load TTS model/);
    await s.speak("hi");

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

    await s.speak("hi");

    // Despite the first listener throwing, the second listener saw the
    // full transition sequence — the production code's try/catch
    // around each callback is what enforces this.
    expect(seen).toEqual(["loading", "ready"]);
  });
});

// ────────────────────────────────────────────────────────────────────
// Task 3 — speak() synthesis + Web Audio playback + voice option
// ────────────────────────────────────────────────────────────────────

describe("KokoroSpeaker — speak() synthesis + Web Audio playback (AC-1, AC-3)", () => {
  let audioState: FakeContextState;

  beforeEach(() => {
    vi.spyOn(console, "log").mockImplementation(() => undefined);
    mocks.pipeline.mockReset();
    audioState = installAudioContext();
    stubFetch();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("resolves immediately on empty input, does not invoke pipeline or construct AudioContext", async () => {
    mocks.pipeline.mockResolvedValue(makeFakePipeline().pipeline);

    const s = new KokoroSpeaker();
    await s.speak("");

    expect(mocks.pipeline).not.toHaveBeenCalled();
    expect(audioState.contexts).toHaveLength(0);
  });

  it("resolves immediately on whitespace-only input — same no-op contract", async () => {
    mocks.pipeline.mockResolvedValue(makeFakePipeline().pipeline);

    const s = new KokoroSpeaker();
    await s.speak("   \n\t  ");

    expect(mocks.pipeline).not.toHaveBeenCalled();
    expect(audioState.contexts).toHaveLength(0);
  });

  it("invokes the Kokoro pipeline with the default voice when no override is supplied", async () => {
    const fake = makeFakePipeline();
    mocks.pipeline.mockResolvedValue(fake.pipeline);

    const s = new KokoroSpeaker();
    await s.speak("hello world");

    expect(fake.calls).toHaveLength(1);
    expect(fake.calls[0]).toEqual({
      text: "hello world",
      options: expect.objectContaining({
        speaker_embeddings: expect.any(Float32Array),
        num_inference_steps: expect.any(Number),
        speed: expect.any(Number),
      }),
    });
    expect(s.getVoice()).toBe(DEFAULT_VOICE);
  });

  it("invokes the Kokoro pipeline with a constructor-supplied voice override", async () => {
    const fake = makeFakePipeline();
    mocks.pipeline.mockResolvedValue(fake.pipeline);

    const s = new KokoroSpeaker({ voice: "af_heart" });
    await s.speak("hello");

    // The constructor voice is stored and returned by getVoice();
    // the pipeline call uses speaker_embeddings (not the voice string).
    expect(s.getVoice()).toBe("af_heart");
    expect(fake.calls[0]?.options).toEqual(
      expect.objectContaining({ speaker_embeddings: expect.any(Float32Array) }),
    );
  });

  it("DEFAULT_VOICE is 'F1'", () => {
    // Pin the constant so a refactor that accidentally changes the
    // default fails this test rather than silently switching voices.
    expect(DEFAULT_VOICE).toBe("F1");
  });

  it("wraps the pipeline output in an AudioBuffer at the reported sample rate", async () => {
    const fixtureAudio = new Float32Array(2400); // 100 ms at 24 kHz
    for (let i = 0; i < fixtureAudio.length; i++) {
      fixtureAudio[i] = Math.sin((2 * Math.PI * i) / 24);
    }
    const fake = makeFakePipeline({ audio: fixtureAudio, sampling_rate: 24000 });
    mocks.pipeline.mockResolvedValue(fake.pipeline);

    const s = new KokoroSpeaker();
    await s.speak("non-empty");

    expect(audioState.buffers).toHaveLength(1);
    const buffer = audioState.buffers[0]!;
    // Mono buffer at the pipeline's reported rate. Length includes the
    // 0.5 s of silence padding added by the production code.
    expect(buffer.numberOfChannels).toBe(1);
    expect(buffer.sampleRate).toBe(24000);
    expect(buffer.length).toBe(fixtureAudio.length + Math.floor(0.5 * 24000));
    // The audio data was actually copied into the buffer channel.
    expect(buffer.copyToChannel).toHaveBeenCalledTimes(1);
  });

  it("schedules the buffer on a source node and connects it to the destination", async () => {
    const fake = makeFakePipeline();
    mocks.pipeline.mockResolvedValue(fake.pipeline);

    const s = new KokoroSpeaker();
    await s.speak("hi");

    expect(audioState.sources).toHaveLength(1);
    const source = audioState.sources[0]!;
    expect(source.buffer).toBe(audioState.buffers[0]);
    expect(source.connect).toHaveBeenCalledTimes(1);
    expect(source.connect).toHaveBeenCalledWith(audioState.contexts[0]!.destination);
    expect(source.start).toHaveBeenCalledTimes(1);
  });

  it("AC-1: the Promise resolves only when source.onended fires, not before", async () => {
    // Drive `onended` manually so the test can assert that the
    // Promise is pending until the event fires.
    audioState.autoEnd = false;

    const fake = makeFakePipeline();
    mocks.pipeline.mockResolvedValue(fake.pipeline);

    const s = new KokoroSpeaker();
    const speakPromise = s.speak("hi");

    let resolved = false;
    void speakPromise.then(() => {
      resolved = true;
    });

    // Wait until the source has been constructed (pipeline + fetch +
    // synthesis all settled). The Promise must STILL be pending —
    // synthesis having completed does not equal playback having
    // completed (the entire point of AC-1).
    await vi.waitFor(() => {
      expect(audioState.sources.length).toBeGreaterThan(0);
    });
    expect(resolved).toBe(false);

    // Drive playback end manually.
    audioState.sources[0]!.onended!();
    await speakPromise;
    expect(resolved).toBe(true);
  });

  it("AC-3 fixture: non-silent samples reach the buffer, duration proportional to input text length", async () => {
    // Two inputs of different lengths: each gets a fake-pipeline
    // output sized proportionally (mirroring how a real TTS model's
    // audio length scales with input characters). Asserts that the
    // wrapped AudioBuffer's duration tracks the input proportionally
    // and that the samples themselves are non-silent.
    const shortText = "hi";
    const longText = "hello world, this is a longer utterance";

    const samplesPerChar = 1200; // arbitrary but deterministic
    const sampleRate = 24000;

    function makeNonSilent(length: number): Float32Array {
      const a = new Float32Array(length);
      for (let i = 0; i < length; i++) {
        a[i] = Math.sin((2 * Math.PI * i) / 32);
      }
      return a;
    }

    const silenceSamples = Math.floor(0.5 * sampleRate);

    // First synthesis — short text.
    const shortAudio = makeNonSilent(shortText.length * samplesPerChar);
    const fake1 = makeFakePipeline({ audio: shortAudio, sampling_rate: sampleRate });
    mocks.pipeline.mockResolvedValue(fake1.pipeline);
    const s1 = new KokoroSpeaker();
    await s1.speak(shortText);
    const shortBuffer = audioState.buffers.at(-1)!;
    expect(shortBuffer.length).toBe(shortAudio.length + silenceSamples);
    // Samples are non-silent — the buffer's channel data contains
    // non-zero entries after copyToChannel ran.
    const shortChannel = shortBuffer.channels[0]!;
    const shortNonZero = Array.from(shortChannel).some((v) => v !== 0);
    expect(shortNonZero).toBe(true);

    // Reset between Speakers so we can install a different mock output.
    mocks.pipeline.mockReset();
    const longAudio = makeNonSilent(longText.length * samplesPerChar);
    const fake2 = makeFakePipeline({ audio: longAudio, sampling_rate: sampleRate });
    mocks.pipeline.mockResolvedValue(fake2.pipeline);
    const s2 = new KokoroSpeaker();
    await s2.speak(longText);
    const longBuffer = audioState.buffers.at(-1)!;
    expect(longBuffer.length).toBe(longAudio.length + silenceSamples);
    const longChannel = longBuffer.channels[0]!;
    const longNonZero = Array.from(longChannel).some((v) => v !== 0);
    expect(longNonZero).toBe(true);

    // Duration ratio matches input-text-length ratio (within float
    // tolerance). Subtract silence padding before comparing so we
    // measure only the speech portion.
    const durationRatio =
      (longBuffer.length - silenceSamples) / (shortBuffer.length - silenceSamples);
    const textRatio = longText.length / shortText.length;
    expect(durationRatio).toBeCloseTo(textRatio, 5);
  });

  it("subsequent speak() calls reuse the cached pipeline (audio still flows)", async () => {
    const fake = makeFakePipeline();
    mocks.pipeline.mockResolvedValue(fake.pipeline);

    const s = new KokoroSpeaker();
    await s.speak("first");
    await s.speak("second");
    expect(mocks.pipeline).toHaveBeenCalledTimes(1);
    expect(fake.calls).toHaveLength(2);
    expect(fake.calls[0]?.text).toBe("first");
    expect(fake.calls[1]?.text).toBe("second");
  });

  it("AudioContext is constructed lazily on first non-empty speak()", async () => {
    const fake = makeFakePipeline();
    mocks.pipeline.mockResolvedValue(fake.pipeline);

    const s = new KokoroSpeaker();
    expect(audioState.contexts).toHaveLength(0);
    await s.speak("hi");
    expect(audioState.contexts).toHaveLength(1);
    // A second speak() reuses the same context — no second instance.
    await s.speak("again");
    expect(audioState.contexts).toHaveLength(1);
  });

  it("throws an actionable error if the pipeline returns an unexpected output shape", async () => {
    const malformed = makeFakePipeline({ unexpected: "shape" } as unknown);
    mocks.pipeline.mockResolvedValue(malformed.pipeline);

    const s = new KokoroSpeaker();
    await expect(s.speak("hi")).rejects.toThrow(/unexpected.*audio.*shape/);
  });
});

// ────────────────────────────────────────────────────────────────────
// Task 3 — extractSynthesisOutput parser (unit-level)
// ────────────────────────────────────────────────────────────────────

describe("extractSynthesisOutput — output parser", () => {
  it("returns the audio + sampling_rate fields verbatim from a well-formed output", () => {
    const audio = new Float32Array([0.1, -0.2]);
    expect(extractSynthesisOutput({ audio, sampling_rate: 24000 })).toEqual({
      audio,
      sampling_rate: 24000,
    });
  });

  it("throws on null input", () => {
    expect(() => extractSynthesisOutput(null)).toThrow(/null.*expected an object|expected an object/);
  });

  it("throws when audio is not a Float32Array", () => {
    expect(() => extractSynthesisOutput({ audio: [0.1], sampling_rate: 24000 })).toThrow(
      /unexpected.*audio.*shape/,
    );
  });

  it("throws when sampling_rate is not a number", () => {
    expect(() =>
      extractSynthesisOutput({ audio: new Float32Array(0), sampling_rate: "24000" }),
    ).toThrow(/invalid sampling_rate|unexpected.*shape/);
  });

  it("throws when fields are missing", () => {
    expect(() => extractSynthesisOutput({})).toThrow(/unexpected.*audio.*shape/);
  });
});

// ────────────────────────────────────────────────────────────────────
// Task 4 — cancel() coordination + dispose() lifecycle (AC-2)
// ────────────────────────────────────────────────────────────────────

describe("KokoroSpeaker — cancel() coordination (AC-2)", () => {
  let audioState: FakeContextState;

  beforeEach(() => {
    vi.spyOn(console, "log").mockImplementation(() => undefined);
    mocks.pipeline.mockReset();
    audioState = installAudioContext();
    stubFetch();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("cancel() while playback is active: source.stop is called and the in-flight speak() Promise resolves", async () => {
    // Hold playback open so cancel() can act on a live source.
    audioState.autoEnd = false;

    const fake = makeFakePipeline();
    mocks.pipeline.mockResolvedValue(fake.pipeline);

    const s = new KokoroSpeaker();
    const speakPromise = s.speak("hi");

    // Wait until the source has been constructed (load + fetch + synthesis settled).
    await vi.waitFor(() => {
      expect(audioState.sources).toHaveLength(1);
    });
    const source = audioState.sources[0]!;
    expect(source.started).toBe(true);
    expect(source.stopped).toBe(false);

    // Now cancel — speak() must resolve without error and source.stop
    // must have been called.
    s.cancel();
    await expect(speakPromise).resolves.toBeUndefined();
    expect(source.stop).toHaveBeenCalledTimes(1);
  });

  it("cancel() while playback is active: Speaker is reusable — a follow-up speak() works", async () => {
    const fake = makeFakePipeline();
    mocks.pipeline.mockResolvedValue(fake.pipeline);

    const s = new KokoroSpeaker();
    // First call: cancel during playback.
    audioState.autoEnd = false;
    const first = s.speak("a");
    await vi.waitFor(() => {
      expect(audioState.sources).toHaveLength(1);
    });
    s.cancel();
    await first;

    // Second call: allow natural completion.
    audioState.autoEnd = true;
    await s.speak("b");

    // Pipeline factory invoked once (cached); synthesis callable
    // invoked twice (one per speak); two sources created.
    expect(mocks.pipeline).toHaveBeenCalledTimes(1);
    expect(fake.calls).toHaveLength(2);
    expect(audioState.sources).toHaveLength(2);
  });

  it("cancel() while synthesis is in flight: speak() resolves, no source constructed", async () => {
    // Make the pipeline call (synthesis) pending — load itself
    // resolves normally.
    let resolveSynth!: (out: unknown) => void;
    const pendingSynth = new Promise<unknown>((r) => {
      resolveSynth = r;
    });
    const synthCallable = Object.assign(
      vi.fn(() => pendingSynth),
      { dispose: vi.fn().mockResolvedValue(undefined) },
    );
    mocks.pipeline.mockResolvedValue(synthCallable as unknown);

    const s = new KokoroSpeaker();
    const speakPromise = s.speak("hi");

    // Wait until the synthesis call has been issued (load + fetch settled).
    await vi.waitFor(() => {
      expect(synthCallable).toHaveBeenCalledTimes(1);
    });
    // No source has been constructed yet — synthesis is still pending.
    expect(audioState.sources).toHaveLength(0);

    // Cancel while synthesis is pending.
    s.cancel();
    // Now late-resolve synthesis with a valid output.
    resolveSynth({ audio: new Float32Array(240), sampling_rate: 24000 });

    // speak() must resolve cleanly without scheduling playback.
    await expect(speakPromise).resolves.toBeUndefined();
    expect(audioState.sources).toHaveLength(0);
  });

  it("cancel() while synthesis is in flight: a follow-up speak() still works", async () => {
    // First synthesis pending; cancel; second synthesis succeeds.
    let resolveFirst!: (out: unknown) => void;
    const synthCallable = vi.fn();
    synthCallable.mockReturnValueOnce(
      new Promise((r) => {
        resolveFirst = r;
      }),
    );
    synthCallable.mockResolvedValueOnce({
      audio: new Float32Array(240),
      sampling_rate: 24000,
    });
    const callable = Object.assign(synthCallable, {
      dispose: vi.fn().mockResolvedValue(undefined),
    });
    mocks.pipeline.mockResolvedValue(callable as unknown);

    const s = new KokoroSpeaker();
    const first = s.speak("a");
    await vi.waitFor(() => {
      expect(callable).toHaveBeenCalledTimes(1);
    });
    s.cancel();
    resolveFirst({ audio: new Float32Array(240), sampling_rate: 24000 });
    await first;

    // Second call uses the same cached pipeline; synthesis completes
    // immediately; source is constructed and plays naturally.
    await s.speak("b");
    expect(audioState.sources).toHaveLength(1);
    expect(callable).toHaveBeenCalledTimes(2);
  });

  it("cancel() with no active speak() is a safe no-op (no throw)", () => {
    const s = new KokoroSpeaker();
    expect(() => s.cancel()).not.toThrow();
    expect(() => s.cancel()).not.toThrow(); // double-cancel also safe
  });

  it("cancel() before any speak() does not latently abort the next speak()", async () => {
    const fake = makeFakePipeline();
    mocks.pipeline.mockResolvedValue(fake.pipeline);

    const s = new KokoroSpeaker();
    s.cancel(); // pre-speak cancel
    await s.speak("hi");

    // The speak() call ran to completion — a source was constructed
    // and natural playback ended (autoEnd default true).
    expect(audioState.sources).toHaveLength(1);
    expect(fake.calls).toHaveLength(1);
  });

  it("cancel() does not reject the in-flight speak() — Promise resolves without error", async () => {
    audioState.autoEnd = false;
    const fake = makeFakePipeline();
    mocks.pipeline.mockResolvedValue(fake.pipeline);

    const s = new KokoroSpeaker();
    const speakPromise = s.speak("hi");
    await vi.waitFor(() => {
      expect(audioState.sources).toHaveLength(1);
    });

    s.cancel();
    // AC-2 explicit: "in-flight speak() Promise resolves without error".
    let resolved = false;
    let rejected: unknown = null;
    await speakPromise.then(
      () => {
        resolved = true;
      },
      (err) => {
        rejected = err;
      },
    );
    expect(resolved).toBe(true);
    expect(rejected).toBe(null);
  });
});

describe("KokoroSpeaker — dispose() lifecycle (AC-2)", () => {
  let audioState: FakeContextState;

  beforeEach(() => {
    vi.spyOn(console, "log").mockImplementation(() => undefined);
    mocks.pipeline.mockReset();
    audioState = installAudioContext();
    stubFetch();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("dispose() before any speak() is a safe no-op (state stays 'idle')", () => {
    const s = new KokoroSpeaker();

    expect(() => s.dispose()).not.toThrow();
    expect(s.getLoadingState()).toBe("idle");
    expect(audioState.contexts).toHaveLength(0);
  });

  it("dispose() after a successful load releases the pipeline and closes the AudioContext", async () => {
    const fake = makeFakePipeline();
    mocks.pipeline.mockResolvedValue(fake.pipeline);

    const s = new KokoroSpeaker();
    await s.speak("hi");
    expect(s.getLoadingState()).toBe("ready");
    expect(audioState.contexts).toHaveLength(1);

    s.dispose();
    expect(s.getLoadingState()).toBe("idle");
    // AudioContext.close was called.
    expect(audioState.contexts[0]!.close).toHaveBeenCalledTimes(1);
    // The pipeline.dispose() is fire-and-forget — yield microtasks so
    // the .then chain in the production code can run.
    await Promise.resolve();
    await Promise.resolve();
    expect(fake.dispose).toHaveBeenCalledTimes(1);
  });

  it("dispose() while playback is active: cancels the playback and releases resources", async () => {
    audioState.autoEnd = false;
    const fake = makeFakePipeline();
    mocks.pipeline.mockResolvedValue(fake.pipeline);

    const s = new KokoroSpeaker();
    const speakPromise = s.speak("hi");
    await vi.waitFor(() => {
      expect(audioState.sources).toHaveLength(1);
    });

    s.dispose();
    await expect(speakPromise).resolves.toBeUndefined();

    // Source was stopped via the cancel() code path that dispose()
    // routes through.
    expect(audioState.sources[0]!.stop).toHaveBeenCalledTimes(1);
    // AudioContext closed.
    expect(audioState.contexts[0]!.close).toHaveBeenCalledTimes(1);
    // State reset.
    expect(s.getLoadingState()).toBe("idle");
  });

  it("dispose() detaches subscribers — no notifications fire on subsequent state changes", async () => {
    const fake = makeFakePipeline();
    mocks.pipeline.mockResolvedValue(fake.pipeline);

    const s = new KokoroSpeaker();
    const seen: LoadingState[] = [];
    s.subscribe((st) => seen.push(st));

    // Load once so the listener has something to observe, then dispose
    // and load again.
    await s.speak("hi");
    expect(seen).toEqual(["loading", "ready"]);

    s.dispose();
    // Subsequent load should NOT notify the dropped listener.
    await s.speak("again");
    expect(seen).toEqual(["loading", "ready"]);
    // After re-load, the Speaker is again in 'ready' (lazy re-load
    // succeeded).
    expect(s.getLoadingState()).toBe("ready");
  });

  it("double-dispose is a safe no-op", async () => {
    const fake = makeFakePipeline();
    mocks.pipeline.mockResolvedValue(fake.pipeline);

    const s = new KokoroSpeaker();
    await s.speak("hi");
    s.dispose();
    expect(() => s.dispose()).not.toThrow();
    // The second dispose did not call close again — pipeline + ctx
    // were already cleared on the first call.
    expect(audioState.contexts[0]!.close).toHaveBeenCalledTimes(1);
    await Promise.resolve();
    await Promise.resolve();
    expect(fake.dispose).toHaveBeenCalledTimes(1);
  });

  it("dispose() with nothing active (no load, no playback): safe no-op", () => {
    const s = new KokoroSpeaker();
    // No speak ever called.
    expect(() => s.dispose()).not.toThrow();
    // No context was constructed (lazy AudioContext was never created).
    expect(audioState.contexts).toHaveLength(0);
  });

  it("speak() after dispose() lazily re-loads the pipeline and re-creates the AudioContext", async () => {
    // dispose() does not permanently disable speak() — the Speaker
    // returns to its initial idle state and a follow-up speak() acts
    // exactly like the first call on a fresh instance. Verifies that
    // the disposed flag (if any) does not lock out the reusable
    // pathway.
    const fake1 = makeFakePipeline();
    mocks.pipeline.mockResolvedValue(fake1.pipeline);

    const s = new KokoroSpeaker();
    await s.speak("first");
    s.dispose();
    expect(audioState.contexts).toHaveLength(1);

    // Re-arm the mock for the second load (the prior cache was
    // cleared by dispose).
    mocks.pipeline.mockReset();
    const fake2 = makeFakePipeline();
    mocks.pipeline.mockResolvedValue(fake2.pipeline);

    await s.speak("second");
    expect(audioState.contexts).toHaveLength(2);
    expect(mocks.pipeline).toHaveBeenCalledTimes(1);
  });
});
