/**
 * Unit tests for `AudioPlayer`.
 *
 * jsdom does not implement the Web Audio API, so an `AudioContext`
 * stub is installed on `globalThis` before each test. The
 * source-node's `onended` event is auto-fired on the next microtask
 * after `start()` so happy-path tests do not need to drive timing
 * by hand; the test that exercises mid-playback `cancel()` overrides
 * `autoEnd` so the test controls when `onended` runs.
 *
 * Pattern is copied from `KokoroSpeaker.test.ts`'s
 * `installAudioContext` so the two stubs read as siblings — a
 * future reader who has read one can navigate the other.
 *
 * Coverage:
 *   - Module-shape pins (constants + AudioPlayerBusyError)
 *   - Successful play() resolves on the `ended` event
 *   - Concurrent play() rejects synchronously with
 *     `AudioPlayerBusyError`
 *   - Empty-samples input resolves without constructing an
 *     `AudioContext`
 *   - cancel() resolves the outstanding promise without throwing
 *   - cancel() with no active play() is a safe no-op
 *   - 24 kHz input feeds an AudioBuffer at exactly 24 kHz —
 *     verifying that the Web Audio resampler is left to handle the
 *     output rate gap (AC-3 spirit)
 *   - dispose() closes the context, cancels in-flight playback,
 *     and is idempotent
 *   - post-dispose play() rejects
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AudioPlayer, AudioPlayerBusyError } from "./AudioPlayer";

// ────────────────────────────────────────────────────────────────────
// AudioContext / AudioBufferSourceNode / AudioBuffer stubs.
// Adapted from KokoroSpeaker.test.ts so the two share a layout.
// ────────────────────────────────────────────────────────────────────

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

type FakeContext = {
  destination: object;
  close: ReturnType<typeof vi.fn>;
  createBuffer: ReturnType<typeof vi.fn>;
  createBufferSource: ReturnType<typeof vi.fn>;
};

type FakeContextState = {
  contexts: FakeContext[];
  sources: FakeSource[];
  buffers: FakeBuffer[];
  /** When false, source.start() does NOT auto-fire onended. */
  autoEnd: boolean;
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
          queueMicrotask(() => {
            if (!source.stopped && source.onended) {
              source.onended();
            }
          });
        }
      }),
      stop: vi.fn(() => {
        source.stopped = true;
        if (source.onended) source.onended();
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

describe("AudioPlayer — module shape", () => {
  it("AudioPlayerBusyError is a typed Error subclass with an actionable message", () => {
    const err = new AudioPlayerBusyError();
    expect(err).toBeInstanceOf(AudioPlayerBusyError);
    expect(err).toBeInstanceOf(Error);
    expect(err.name).toBe("AudioPlayerBusyError");
    expect(err.message).toMatch(/already in flight/i);
    expect(err.message).toMatch(/await.*cancel/i);
  });

  it("AudioPlayerBusyError accepts a custom message override", () => {
    const err = new AudioPlayerBusyError("custom");
    expect(err.message).toBe("custom");
  });
});

describe("AudioPlayer — happy path", () => {
  let audio: FakeContextState;

  beforeEach(() => {
    audio = installAudioContext();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("constructor does not allocate an AudioContext (lazy)", () => {
    new AudioPlayer();
    expect(audio.contexts).toHaveLength(0);
  });

  it("play(samples, rate) resolves on the source's onended event", async () => {
    const p = new AudioPlayer();
    const samples = new Float32Array(240);
    samples[0] = 0.5;
    await expect(p.play(samples, 24000)).resolves.toBeUndefined();
    expect(audio.sources).toHaveLength(1);
    expect(audio.sources[0]!.started).toBe(true);
    p.dispose();
  });

  it("AC-3: 24 kHz Float32 input is fed into the AudioBuffer at exactly 24 kHz", async () => {
    const p = new AudioPlayer();
    const samples = new Float32Array(240);
    for (let i = 0; i < samples.length; i++) samples[i] = i / 240;
    await p.play(samples, 24000);

    expect(audio.buffers).toHaveLength(1);
    const buf = audio.buffers[0]!;
    expect(buf.numberOfChannels).toBe(1);
    expect(buf.length).toBe(240);
    expect(buf.sampleRate).toBe(24000);
    // The samples should be copied into channel 0 verbatim — the Web
    // Audio resampler runs on the way to the destination, not in the
    // buffer itself.
    expect(Array.from(buf.channels[0]!)).toEqual(Array.from(samples));
    p.dispose();
  });

  it("connects the source into the context's destination", async () => {
    const p = new AudioPlayer();
    await p.play(new Float32Array(240), 24000);

    const source = audio.sources[0]!;
    expect(source.connect).toHaveBeenCalledTimes(1);
    expect(source.connect).toHaveBeenCalledWith(audio.contexts[0]!.destination);
    p.dispose();
  });

  it("empty samples input resolves immediately without constructing an AudioContext", async () => {
    const p = new AudioPlayer();
    await expect(p.play(new Float32Array(0), 24000)).resolves.toBeUndefined();
    expect(audio.contexts).toHaveLength(0);
    expect(audio.sources).toHaveLength(0);
    p.dispose();
  });

  it("rejects with a clear error when sampleRate is non-finite or non-positive", async () => {
    const p = new AudioPlayer();
    await expect(p.play(new Float32Array(240), 0)).rejects.toThrow(/positive finite/);
    await expect(p.play(new Float32Array(240), -1)).rejects.toThrow(/positive finite/);
    await expect(p.play(new Float32Array(240), NaN)).rejects.toThrow(/positive finite/);
    p.dispose();
  });

  it("sequential play() calls work — context is reused", async () => {
    const p = new AudioPlayer();
    await p.play(new Float32Array(240), 24000);
    await p.play(new Float32Array(240), 24000);
    await p.play(new Float32Array(240), 24000);

    // One context, three sources.
    expect(audio.contexts).toHaveLength(1);
    expect(audio.sources).toHaveLength(3);
    p.dispose();
  });
});

describe("AudioPlayer — concurrent play() rejects (primitive does not queue)", () => {
  let audio: FakeContextState;

  beforeEach(() => {
    audio = installAudioContext();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("a second play() while the first is in flight rejects with AudioPlayerBusyError", async () => {
    audio.autoEnd = false; // keep first play() open

    const p = new AudioPlayer();
    const first = p.play(new Float32Array(240), 24000);
    // Yield one microtask so the first play() reaches the active state
    // (sets activeSource / activeResolver).
    await Promise.resolve();
    await Promise.resolve();

    await expect(p.play(new Float32Array(240), 24000)).rejects.toBeInstanceOf(AudioPlayerBusyError);

    // Tear down — fire onended on the first source so the first
    // play() can resolve cleanly (otherwise the test runner times
    // out waiting on the unresolved promise).
    audio.sources[0]!.onended?.();
    await expect(first).resolves.toBeUndefined();
    p.dispose();
  });
});

describe("AudioPlayer — cancel()", () => {
  let audio: FakeContextState;

  beforeEach(() => {
    audio = installAudioContext();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("cancel() with no active play() is a safe no-op", () => {
    const p = new AudioPlayer();
    expect(() => p.cancel()).not.toThrow();
    p.dispose();
  });

  it("cancel() during playback resolves the outstanding play() promise without throwing", async () => {
    audio.autoEnd = false;
    const p = new AudioPlayer();
    const inflight = p.play(new Float32Array(240), 24000);
    // Yield so play() reaches the active state.
    await Promise.resolve();
    await Promise.resolve();

    p.cancel();
    await expect(inflight).resolves.toBeUndefined();
    expect(audio.sources[0]!.stop).toHaveBeenCalledTimes(1);
    expect(audio.sources[0]!.stopped).toBe(true);
    p.dispose();
  });

  it("cancel() survives source.stop() throwing", async () => {
    audio.autoEnd = false;
    const p = new AudioPlayer();
    const inflight = p.play(new Float32Array(240), 24000);
    await Promise.resolve();
    await Promise.resolve();

    // Force stop() to throw — the resolver-before-stop ordering must
    // unblock the caller anyway.
    audio.sources[0]!.stop = vi.fn(() => {
      throw new Error("InvalidStateError");
    });

    expect(() => p.cancel()).not.toThrow();
    await expect(inflight).resolves.toBeUndefined();
    p.dispose();
  });

  it("after cancel(), a follow-up play() works", async () => {
    audio.autoEnd = false;
    const p = new AudioPlayer();
    const first = p.play(new Float32Array(240), 24000);
    await Promise.resolve();
    await Promise.resolve();
    p.cancel();
    await first;

    // Re-enable auto-end for the follow-up.
    audio.autoEnd = true;
    await expect(p.play(new Float32Array(240), 24000)).resolves.toBeUndefined();
    expect(audio.sources).toHaveLength(2);
    expect(audio.contexts).toHaveLength(1); // context reused
    p.dispose();
  });
});

describe("AudioPlayer — dispose() lifecycle", () => {
  let audio: FakeContextState;

  beforeEach(() => {
    audio = installAudioContext();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("dispose() before any play() is a safe no-op", () => {
    const p = new AudioPlayer();
    expect(() => p.dispose()).not.toThrow();
    expect(audio.contexts).toHaveLength(0);
  });

  it("dispose() after a successful play() closes the context", async () => {
    const p = new AudioPlayer();
    await p.play(new Float32Array(240), 24000);
    p.dispose();
    expect(audio.contexts[0]!.close).toHaveBeenCalledTimes(1);
  });

  it("dispose() cancels in-flight playback", async () => {
    audio.autoEnd = false;
    const p = new AudioPlayer();
    const inflight = p.play(new Float32Array(240), 24000);
    await Promise.resolve();
    await Promise.resolve();

    p.dispose();
    await expect(inflight).resolves.toBeUndefined();
    expect(audio.sources[0]!.stop).toHaveBeenCalledTimes(1);
    expect(audio.contexts[0]!.close).toHaveBeenCalledTimes(1);
  });

  it("second dispose() is a safe no-op (close not called twice)", async () => {
    const p = new AudioPlayer();
    await p.play(new Float32Array(240), 24000);
    p.dispose();
    expect(() => p.dispose()).not.toThrow();
    expect(audio.contexts[0]!.close).toHaveBeenCalledTimes(1);
  });

  it("post-dispose play() rejects with a clear error", async () => {
    const p = new AudioPlayer();
    await p.play(new Float32Array(240), 24000);
    p.dispose();
    await expect(p.play(new Float32Array(240), 24000)).rejects.toThrow(
      /cannot play.*after dispose/i,
    );
  });

  it("dispose() survives the AudioContext close() returning a non-thenable", async () => {
    const p = new AudioPlayer();
    await p.play(new Float32Array(240), 24000);

    audio.contexts[0]!.close = vi.fn(() => undefined as unknown as Promise<void>);
    expect(() => p.dispose()).not.toThrow();
  });
});
