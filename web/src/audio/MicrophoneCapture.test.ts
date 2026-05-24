/**
 * Unit tests for `MicrophoneCapture`.
 *
 * jsdom does not implement the Web Audio API, `getUserMedia`, or
 * `AudioWorkletNode` — so the test installs lightweight stubs on
 * `globalThis` / `navigator` before each test, captures the
 * worklet-node's `port.onmessage` handler the production code
 * registers, and fires synthetic `"frame"` messages to drive the
 * `onFrame` consumer callback without ever running the real worklet
 * JS. The pattern mirrors `KokoroSpeaker.test.ts`'s `AudioContext`
 * stub so the two read as siblings.
 *
 * Coverage:
 *   - successful `start()` wires the audio graph and delivers frames
 *   - permission denial (`getUserMedia` throws `NotAllowedError`)
 *     surfaces as `MicPermissionDeniedError`
 *   - other `getUserMedia` errors propagate unchanged
 *   - `dispose()` stops every `MediaStreamTrack` and closes the
 *     `AudioContext`
 *   - second `dispose()` is a no-op
 *   - `dispose()` before `start()` is a no-op
 *   - `start()` after `dispose()` rejects with an actionable error
 *   - `addModule()` is called with the Blob URL the test stub records
 *   - `processorOptions` carry the live `audioContext.sampleRate`
 *   - the worklet source string registers the canonical processor name
 *   - `onFrame` rebinding works (a frame after rebinding hits the new
 *     callback, not the old one)
 *   - a throwing `onFrame` does not tear down the capture pipeline
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  FRAME_SAMPLE_COUNT,
  MicPermissionDeniedError,
  MicrophoneCapture,
  TARGET_SAMPLE_RATE_HZ,
} from "./MicrophoneCapture";
import { DECIMATOR_WORKLET_NAME, DECIMATOR_WORKLET_SOURCE } from "./decimator-worklet";

// ────────────────────────────────────────────────────────────────────
// Web Audio + mic stubs. Mirrors the shape of KokoroSpeaker.test.ts's
// installAudioContext() so a reader familiar with that file lands
// immediately.
// ────────────────────────────────────────────────────────────────────

type FakeTrack = {
  stop: ReturnType<typeof vi.fn>;
  stopped: boolean;
};

type FakeMediaStream = {
  getTracks: () => FakeTrack[];
  tracks: FakeTrack[];
};

type FakeAudioWorklet = {
  addModule: ReturnType<typeof vi.fn>;
};

type FakeSourceNode = {
  connect: ReturnType<typeof vi.fn>;
  disconnect: ReturnType<typeof vi.fn>;
};

type FakeWorkletNode = {
  port: { onmessage: ((ev: MessageEvent) => void) | null };
  disconnect: ReturnType<typeof vi.fn>;
  options: AudioWorkletNodeOptions | undefined;
  name: string;
};

type FakeAudioContext = {
  sampleRate: number;
  audioWorklet: FakeAudioWorklet;
  createMediaStreamSource: ReturnType<typeof vi.fn>;
  close: ReturnType<typeof vi.fn>;
  state: "running" | "closed";
};

interface AudioState {
  contexts: FakeAudioContext[];
  workletNodes: FakeWorkletNode[];
  sourceNodes: FakeSourceNode[];
  blobUrls: string[];
  revokedUrls: string[];
  blobBodies: BlobPart[][];
  /** The next sample rate the constructed AudioContext should report. */
  nextSampleRate: number;
}

function installAudio(): AudioState {
  const state: AudioState = {
    contexts: [],
    workletNodes: [],
    sourceNodes: [],
    blobUrls: [],
    revokedUrls: [],
    blobBodies: [],
    nextSampleRate: 48000,
  };

  // Fake Blob — capture the parts so a test can assert what was packed
  // into the worklet source URL.
  class FakeBlob {
    parts: BlobPart[];
    type: string;
    constructor(parts: BlobPart[], opts: { type?: string } = {}) {
      this.parts = parts;
      this.type = opts.type ?? "";
      state.blobBodies.push(parts);
    }
  }
  vi.stubGlobal("Blob", FakeBlob);

  // URL is provided by jsdom but the methods are sometimes either
  // missing or noop-stubs; replace with deterministic counterparts so
  // we can pin the create/revoke handshake.
  vi.stubGlobal("URL", {
    createObjectURL: vi.fn(() => {
      const url = `blob:fake/${state.blobUrls.length}`;
      state.blobUrls.push(url);
      return url;
    }),
    revokeObjectURL: vi.fn((url: string) => {
      state.revokedUrls.push(url);
    }),
  });

  class MockAudioContext {
    sampleRate: number;
    audioWorklet: FakeAudioWorklet;
    state: "running" | "closed" = "running";
    close = vi.fn(() => {
      this.state = "closed";
      return Promise.resolve();
    });
    createMediaStreamSource: ReturnType<typeof vi.fn>;

    constructor() {
      this.sampleRate = state.nextSampleRate;
      this.audioWorklet = {
        addModule: vi.fn(async (_url: string) => undefined),
      };
      this.createMediaStreamSource = vi.fn(() => {
        const node: FakeSourceNode = {
          connect: vi.fn(),
          disconnect: vi.fn(),
        };
        state.sourceNodes.push(node);
        return node;
      });
      // Register self after wiring so tests can read .sampleRate / etc.
      state.contexts.push(this as unknown as FakeAudioContext);
    }
  }
  vi.stubGlobal("AudioContext", MockAudioContext);

  class MockAudioWorkletNode {
    port: { onmessage: ((ev: MessageEvent) => void) | null };
    disconnect = vi.fn();
    options: AudioWorkletNodeOptions | undefined;
    name: string;
    // The Web Audio spec gives AudioWorkletNode a `(context, name, options)`
    // signature; matching it keeps the production code paths type-clean.
    constructor(_context: AudioContext, name: string, options?: AudioWorkletNodeOptions) {
      this.name = name;
      this.options = options;
      this.port = { onmessage: null };
      state.workletNodes.push(this as unknown as FakeWorkletNode);
    }
  }
  vi.stubGlobal("AudioWorkletNode", MockAudioWorkletNode);

  return state;
}

function makeFakeStream(): FakeMediaStream {
  const tracks: FakeTrack[] = [
    {
      stop: vi.fn(function (this: FakeTrack) {
        this.stopped = true;
      }),
      stopped: false,
    },
  ];
  return {
    tracks,
    getTracks: () => tracks,
  };
}

function installGetUserMedia(impl: () => Promise<FakeMediaStream | unknown>): {
  getUserMedia: ReturnType<typeof vi.fn>;
} {
  const getUserMedia = vi.fn(impl);
  // `navigator.mediaDevices` is read-only on jsdom's Navigator. Replace
  // it via a defineProperty assignment so the production code's
  // `navigator.mediaDevices.getUserMedia(...)` call hits our stub.
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: { getUserMedia },
  });
  return { getUserMedia };
}

/**
 * Fire a synthetic "frame" message on the most-recently-constructed
 * worklet node. The production code's `port.onmessage` handler is the
 * production frame-dispatch logic, so invoking it through the same
 * `MessageEvent` envelope exercises the same code path the real
 * worklet would.
 */
function deliverFrame(state: AudioState, frame: Float32Array): void {
  const node = state.workletNodes[state.workletNodes.length - 1];
  if (!node) throw new Error("test bug: no worklet node constructed yet");
  const handler = node.port.onmessage;
  if (handler === null) throw new Error("test bug: no onmessage handler installed");
  // MessageEvent constructor in jsdom accepts `{ data }`.
  handler(new MessageEvent("message", { data: { type: "frame", frame } }));
}

// ────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────

describe("MicrophoneCapture — module constants", () => {
  it("FRAME_SAMPLE_COUNT is 512 (Silero v5 frame length)", () => {
    expect(FRAME_SAMPLE_COUNT).toBe(512);
  });

  it("TARGET_SAMPLE_RATE_HZ is 16000 (matches Whisper / wire constants)", () => {
    expect(TARGET_SAMPLE_RATE_HZ).toBe(16000);
  });

  it("DECIMATOR_WORKLET_SOURCE registers the canonical processor name", () => {
    // Pinning the registration call inside the worklet string prevents
    // a future refactor that renames or moves it from silently breaking
    // the main-thread AudioWorkletNode construction.
    expect(DECIMATOR_WORKLET_SOURCE).toContain(`registerProcessor("${DECIMATOR_WORKLET_NAME}"`);
  });

  it("MicPermissionDeniedError carries an actionable default message", () => {
    const err = new MicPermissionDeniedError();
    expect(err.name).toBe("MicPermissionDeniedError");
    expect(err.message).toMatch(/microphone access/i);
    expect(err.message).toMatch(/click the lock icon/i);
    // instanceof must still hold after the Object.setPrototypeOf in the
    // constructor — this is the regression hedge for transpilation.
    expect(err instanceof MicPermissionDeniedError).toBe(true);
    expect(err instanceof Error).toBe(true);
  });

  it("MicPermissionDeniedError accepts a custom message override", () => {
    const err = new MicPermissionDeniedError("custom");
    expect(err.message).toBe("custom");
  });
});

describe("MicrophoneCapture — start() happy path", () => {
  let audio: AudioState;

  beforeEach(() => {
    audio = installAudio();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("opens the mic, constructs the audio graph, and resolves", async () => {
    const stream = makeFakeStream();
    installGetUserMedia(async () => stream);

    const mic = new MicrophoneCapture();
    await mic.start();

    expect(audio.contexts).toHaveLength(1);
    expect(audio.workletNodes).toHaveLength(1);
    expect(audio.sourceNodes).toHaveLength(1);
    // The worklet must be loaded via a Blob URL — pin both that
    // createObjectURL fired and that the URL the addModule call got is
    // exactly the one we minted.
    expect(audio.blobUrls).toHaveLength(1);
    expect(audio.contexts[0]!.audioWorklet.addModule).toHaveBeenCalledTimes(1);
    expect(audio.contexts[0]!.audioWorklet.addModule).toHaveBeenCalledWith(audio.blobUrls[0]);
    // Source node is connected into the worklet — verifies the graph
    // is actually wired, not just constructed.
    expect(audio.sourceNodes[0]!.connect).toHaveBeenCalledTimes(1);
    expect(audio.sourceNodes[0]!.connect).toHaveBeenCalledWith(audio.workletNodes[0]);
  });

  it("packs the decimator worklet source into the Blob it loads", async () => {
    installGetUserMedia(async () => makeFakeStream());

    const mic = new MicrophoneCapture();
    await mic.start();

    expect(audio.blobBodies).toHaveLength(1);
    // The Blob constructor is called with `[DECIMATOR_WORKLET_SOURCE]`;
    // matching the array contents pins both the wrapping and the
    // source content end-to-end.
    expect(audio.blobBodies[0]).toEqual([DECIMATOR_WORKLET_SOURCE]);
  });

  it("constructs the AudioWorkletNode with the canonical name and live processorOptions", async () => {
    audio.nextSampleRate = 44100;
    installGetUserMedia(async () => makeFakeStream());

    const mic = new MicrophoneCapture();
    await mic.start();

    expect(audio.workletNodes[0]!.name).toBe(DECIMATOR_WORKLET_NAME);
    const opts = audio.workletNodes[0]!.options;
    expect(opts?.processorOptions).toEqual({
      deviceSampleRate: 44100,
      targetSampleRate: 16000,
      frameSampleCount: 512,
    });
  });

  it("delivers frames from the worklet to onFrame in order", async () => {
    installGetUserMedia(async () => makeFakeStream());

    const seen: Float32Array[] = [];
    const mic = new MicrophoneCapture();
    mic.onFrame = (f) => seen.push(f);
    await mic.start();

    const a = new Float32Array(512);
    const b = new Float32Array(512);
    a[0] = 0.5;
    b[0] = -0.5;
    deliverFrame(audio, a);
    deliverFrame(audio, b);

    expect(seen).toHaveLength(2);
    expect(seen[0]).toBe(a);
    expect(seen[1]).toBe(b);
  });

  it("does not invoke onFrame when it is null (frame is dropped silently)", async () => {
    installGetUserMedia(async () => makeFakeStream());

    const mic = new MicrophoneCapture();
    await mic.start();
    mic.onFrame = null;

    expect(() => deliverFrame(audio, new Float32Array(512))).not.toThrow();
  });

  it("supports rebinding onFrame between frames", async () => {
    installGetUserMedia(async () => makeFakeStream());

    const first: Float32Array[] = [];
    const second: Float32Array[] = [];
    const mic = new MicrophoneCapture();
    mic.onFrame = (f) => first.push(f);
    await mic.start();

    deliverFrame(audio, new Float32Array(512));
    mic.onFrame = (f) => second.push(f);
    deliverFrame(audio, new Float32Array(512));

    expect(first).toHaveLength(1);
    expect(second).toHaveLength(1);
  });

  it("a throwing onFrame does not break subsequent frame delivery", async () => {
    installGetUserMedia(async () => makeFakeStream());

    const mic = new MicrophoneCapture();
    let count = 0;
    mic.onFrame = () => {
      count++;
      throw new Error("buggy frame handler");
    };
    await mic.start();

    deliverFrame(audio, new Float32Array(512));
    deliverFrame(audio, new Float32Array(512));

    expect(count).toBe(2);
  });

  it("ignores non-frame messages on the worklet port", async () => {
    installGetUserMedia(async () => makeFakeStream());

    const mic = new MicrophoneCapture();
    const seen: Float32Array[] = [];
    mic.onFrame = (f) => seen.push(f);
    await mic.start();

    // Synthesise a non-frame message — the production handler must
    // ignore it without crashing or invoking onFrame.
    const node = audio.workletNodes[0]!;
    node.port.onmessage?.(
      new MessageEvent("message", { data: { type: "telemetry", kind: "underrun" } }),
    );

    expect(seen).toHaveLength(0);
  });
});

describe("MicrophoneCapture — start() failure paths", () => {
  let audio: AudioState;

  beforeEach(() => {
    audio = installAudio();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("translates NotAllowedError into MicPermissionDeniedError", async () => {
    // Browsers emit this exact `name` for permission denial.
    const denied = new Error("Permission denied");
    denied.name = "NotAllowedError";
    installGetUserMedia(async () => {
      throw denied;
    });

    const mic = new MicrophoneCapture();
    await expect(mic.start()).rejects.toBeInstanceOf(MicPermissionDeniedError);
    // No audio context was constructed — the failure is at the mic
    // open step, before any Web Audio resources are allocated.
    expect(audio.contexts).toHaveLength(0);
  });

  it("propagates non-permission errors unchanged", async () => {
    const hardware = new Error("hardware busy");
    hardware.name = "NotReadableError";
    installGetUserMedia(async () => {
      throw hardware;
    });

    const mic = new MicrophoneCapture();
    await expect(mic.start()).rejects.toThrow(/hardware busy/);
    // Should NOT be re-wrapped in MicPermissionDeniedError — the
    // pipeline layer distinguishes the two.
    await expect(mic.start().catch((e) => e)).resolves.not.toBeInstanceOf(
      MicPermissionDeniedError,
    );
  });
});

describe("MicrophoneCapture — dispose() lifecycle", () => {
  let audio: AudioState;

  beforeEach(() => {
    audio = installAudio();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("dispose() before start() is a safe no-op", () => {
    const mic = new MicrophoneCapture();
    expect(() => mic.dispose()).not.toThrow();
    // Nothing was allocated.
    expect(audio.contexts).toHaveLength(0);
  });

  it("dispose() stops every MediaStreamTrack and closes the AudioContext", async () => {
    const stream = makeFakeStream();
    installGetUserMedia(async () => stream);

    const mic = new MicrophoneCapture();
    await mic.start();
    mic.dispose();

    for (const track of stream.tracks) {
      expect(track.stop).toHaveBeenCalledTimes(1);
      expect(track.stopped).toBe(true);
    }
    expect(audio.contexts[0]!.close).toHaveBeenCalledTimes(1);
    expect(audio.contexts[0]!.state).toBe("closed");
  });

  it("dispose() disconnects the worklet node and the source node", async () => {
    installGetUserMedia(async () => makeFakeStream());

    const mic = new MicrophoneCapture();
    await mic.start();
    mic.dispose();

    expect(audio.workletNodes[0]!.disconnect).toHaveBeenCalledTimes(1);
    expect(audio.sourceNodes[0]!.disconnect).toHaveBeenCalledTimes(1);
  });

  it("dispose() revokes the Blob URL the worklet was loaded from", async () => {
    installGetUserMedia(async () => makeFakeStream());

    const mic = new MicrophoneCapture();
    await mic.start();
    expect(audio.revokedUrls).toEqual([]);

    mic.dispose();
    expect(audio.revokedUrls).toEqual(audio.blobUrls);
  });

  it("dispose() clears onFrame and stops delivering frames", async () => {
    installGetUserMedia(async () => makeFakeStream());

    const mic = new MicrophoneCapture();
    const seen: Float32Array[] = [];
    mic.onFrame = (f) => seen.push(f);
    await mic.start();

    // Capture the handler BEFORE dispose so we can attempt a post-dispose
    // delivery — the production handler reads `this.onFrame` live, and
    // dispose() sets it to null.
    const handler = audio.workletNodes[0]!.port.onmessage;
    mic.dispose();
    expect(mic.onFrame).toBeNull();

    // The handler reference is still callable — but the production
    // close-out cleared `this.onFrame`, so no delivery happens.
    handler?.(new MessageEvent("message", { data: { type: "frame", frame: new Float32Array(512) } }));
    expect(seen).toHaveLength(0);
  });

  it("second dispose() is a safe no-op (track.stop() not called twice)", async () => {
    const stream = makeFakeStream();
    installGetUserMedia(async () => stream);

    const mic = new MicrophoneCapture();
    await mic.start();
    mic.dispose();
    expect(() => mic.dispose()).not.toThrow();

    // The track was stopped exactly once across two dispose calls.
    expect(stream.tracks[0]!.stop).toHaveBeenCalledTimes(1);
    expect(audio.contexts[0]!.close).toHaveBeenCalledTimes(1);
  });

  it("start() after dispose() rejects with an actionable error", async () => {
    installGetUserMedia(async () => makeFakeStream());

    const mic = new MicrophoneCapture();
    await mic.start();
    mic.dispose();

    await expect(mic.start()).rejects.toThrow(/cannot start.*after dispose/i);
  });

  it("dispose() survives the AudioContext close() returning a non-thenable", async () => {
    // Some jsdom versions / mocks return undefined from close() — make
    // sure the production code does not throw trying to .catch() that.
    installGetUserMedia(async () => makeFakeStream());

    const mic = new MicrophoneCapture();
    await mic.start();
    // Override the close mock to return undefined.
    audio.contexts[0]!.close = vi.fn(() => undefined as unknown as Promise<void>);

    expect(() => mic.dispose()).not.toThrow();
  });
});
