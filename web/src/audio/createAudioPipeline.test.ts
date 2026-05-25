/**
 * Unit tests for `createAudioPipeline`.
 *
 * Tests inject stub `MicrophoneCapture` and stub `VAD` instances
 * matching the production shapes — no real Web Audio API or Silero
 * model is involved. The stub style follows
 * `web/src/dialogue/__test-helpers__/stub-deps.ts`'s pattern for
 * the `VAD` stub so the two read as siblings.
 *
 * Coverage:
 *   - Happy-path start() transitions idle → starting → running and
 *     wires microphone.onFrame to vad.process
 *   - MicPermissionDeniedError thrown by the mic stub transitions
 *     to "permission-denied" instead of rejecting (KD-5 / AC-4)
 *   - Generic mic errors (NotReadableError, etc.) transition to
 *     "error" — the underlying message is exposed via getError()
 *   - dispose() runs cleanly from any state, including
 *     "permission-denied" / "error"
 *   - dispose() does NOT dispose the VAD (lifecycle separation)
 *   - dispose() clears the mic's onFrame and disposes the mic
 *   - Subscribers receive every transition; unsubscribe stops them
 *   - A buggy listener does not block others
 *   - A buggy vad.process inside the onFrame wiring does not tear
 *     down the audio graph
 *   - start() while already running is a safe no-op
 *   - start() after a denial constructs a fresh MicrophoneCapture
 *     (when the default mic is used) so the post-dispose sticky
 *     flag does not lock the pipeline
 *   - start() after dispose() rejects
 *   - getError() returns null in idle / starting / running and the
 *     message string in error / permission-denied
 */
import { describe, expect, it, vi } from "vitest";

import { type AudioPipeline, type PipelineState, createAudioPipeline } from "./createAudioPipeline";
import { MicPermissionDeniedError, MicrophoneCapture } from "./MicrophoneCapture";

import type { VAD } from "../vad/types";

// ────────────────────────────────────────────────────────────────────
// Stubs
// ────────────────────────────────────────────────────────────────────

/**
 * Stub mic + VAD shapes. Note the intersection pattern (rather than
 * a named interface `extends MicrophoneCapture` / `extends VAD`) —
 * vitest 4's Mock<T> default call signature does not structurally
 * satisfy the production interfaces under a NAMED-extends
 * declaration, but the intersection form `Production & { mocks }`
 * lets the inline literal's inferred call types unify with both
 * sides. This is the same pattern
 * `web/src/dialogue/__test-helpers__/stub-deps.ts` uses.
 */
type StubMic = {
  start: ReturnType<typeof vi.fn>;
  dispose: ReturnType<typeof vi.fn>;
  onFrame: ((frame: Float32Array) => void) | null;
};

function makeStubMic(startImpl: () => Promise<void> = async () => undefined): StubMic {
  const mic: StubMic = {
    onFrame: null,
    start: vi.fn(startImpl),
    dispose: vi.fn((): void => undefined),
  };
  return mic;
}

type StubVad = VAD & {
  process: ReturnType<typeof vi.fn>;
  dispose: ReturnType<typeof vi.fn>;
};

function makeStubVad(processImpl: (frame: Float32Array) => boolean = () => false): StubVad {
  const vad: StubVad = {
    process: vi.fn(processImpl),
    dispose: vi.fn((): void => undefined),
  };
  return vad;
}

/**
 * Cast the stub mic into the production `MicrophoneCapture` type so
 * the factory accepts it. Tests use this in the `microphone` option
 * — the production code never inspects fields outside the documented
 * public surface (onFrame, start, dispose), so the cast is safe.
 */
function asMic(stub: StubMic): MicrophoneCapture {
  return stub as unknown as MicrophoneCapture;
}

// ────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────

describe("createAudioPipeline — initial state", () => {
  it("returns a pipeline whose initial state is 'idle'", async () => {
    const vad = makeStubVad();
    const mic = makeStubMic();
    const pipe = await createAudioPipeline({ vad, microphone: asMic(mic) });

    expect(pipe.pipelineState).toBe("idle");
    expect(pipe.getError()).toBeNull();
    expect(mic.start).not.toHaveBeenCalled();
  });

  it("the returned pipelineState property is read-only (setting throws)", async () => {
    const pipe = await createAudioPipeline({
      vad: makeStubVad(),
      microphone: asMic(makeStubMic()),
    });
    // Cast away the type so we can probe the runtime guard.
    expect(() => {
      (pipe as unknown as { pipelineState: PipelineState }).pipelineState = "running";
    }).toThrow();
  });
});

describe("createAudioPipeline — start() happy path", () => {
  it("transitions idle → starting → running on successful start", async () => {
    const vad = makeStubVad();
    const mic = makeStubMic();
    const pipe = await createAudioPipeline({ vad, microphone: asMic(mic) });

    const seen: PipelineState[] = [];
    pipe.subscribe((s) => seen.push(s));

    await pipe.start();

    expect(seen).toEqual(["starting", "running"]);
    expect(pipe.pipelineState).toBe("running");
    expect(pipe.getError()).toBeNull();
  });

  it("wires microphone.onFrame to vad.process after start succeeds", async () => {
    const vad = makeStubVad();
    const mic = makeStubMic();
    const pipe = await createAudioPipeline({ vad, microphone: asMic(mic) });

    await pipe.start();
    expect(mic.onFrame).toBeTypeOf("function");

    const frame = new Float32Array(512);
    frame[0] = 0.5;
    mic.onFrame?.(frame);

    expect(vad.process).toHaveBeenCalledTimes(1);
    expect(vad.process).toHaveBeenCalledWith(frame);
  });

  it("a throwing vad.process inside the onFrame wiring does not propagate", async () => {
    const vad = makeStubVad((_f) => {
      throw new Error("vad blew up");
    });
    const mic = makeStubMic();
    const pipe = await createAudioPipeline({ vad, microphone: asMic(mic) });
    await pipe.start();

    // The wiring catches — this must NOT throw.
    expect(() => mic.onFrame?.(new Float32Array(512))).not.toThrow();
    expect(vad.process).toHaveBeenCalledTimes(1);
  });

  it("start() does not wire onFrame until after microphone.start() resolves", async () => {
    let resolveStart!: () => void;
    const mic = makeStubMic(
      () =>
        new Promise<void>((r) => {
          resolveStart = r;
        }),
    );
    const vad = makeStubVad();
    const pipe = await createAudioPipeline({ vad, microphone: asMic(mic) });

    const startPromise = pipe.start();
    // Yield so the state transition to "starting" happens.
    await Promise.resolve();
    expect(pipe.pipelineState).toBe("starting");
    // onFrame must NOT be set yet — wiring runs after start resolves.
    expect(mic.onFrame).toBeNull();

    resolveStart();
    await startPromise;
    expect(mic.onFrame).toBeTypeOf("function");
    expect(pipe.pipelineState).toBe("running");
  });

  it("start() while already running is a safe no-op", async () => {
    const vad = makeStubVad();
    const mic = makeStubMic();
    const pipe = await createAudioPipeline({ vad, microphone: asMic(mic) });

    await pipe.start();
    expect(mic.start).toHaveBeenCalledTimes(1);

    const seen: PipelineState[] = [];
    pipe.subscribe((s) => seen.push(s));
    await pipe.start();

    expect(mic.start).toHaveBeenCalledTimes(1);
    expect(seen).toEqual([]); // no transitions
  });
});

describe("createAudioPipeline — permission denial (KD-5 / AC-4)", () => {
  it("transitions to 'permission-denied' instead of rejecting when mic.start throws MicPermissionDeniedError", async () => {
    const denied = new MicPermissionDeniedError();
    const mic = makeStubMic(async () => {
      throw denied;
    });
    const vad = makeStubVad();
    const pipe = await createAudioPipeline({ vad, microphone: asMic(mic) });

    const seen: PipelineState[] = [];
    pipe.subscribe((s) => seen.push(s));

    // The whole point of KD-5 — start() resolves, does not reject.
    await expect(pipe.start()).resolves.toBeUndefined();
    expect(pipe.pipelineState).toBe("permission-denied");
    expect(seen).toEqual(["starting", "permission-denied"]);
    expect(pipe.getError()).toBe(denied.message);
  });

  it("after a denial, start() on a default-mic pipeline retries against a fresh MicrophoneCapture", async () => {
    // No `microphone` injection → factory uses `new MicrophoneCapture()`
    // each (re)start. We cannot exercise the production
    // MicrophoneCapture without the Web Audio stubs, so instead we
    // verify the contract from the outside: the pipeline must accept
    // a second start() invocation and re-attempt without throwing the
    // sticky post-dispose error.
    const denied = new MicPermissionDeniedError();
    let callCount = 0;
    // Spy on the MicrophoneCapture constructor by stubbing its
    // prototype methods. Each instance gets its own start() — the
    // first throws, the second resolves.
    const startSpy = vi
      .spyOn(MicrophoneCapture.prototype, "start")
      .mockImplementation(async function (this: MicrophoneCapture) {
        callCount++;
        if (callCount === 1) throw denied;
        return undefined;
      });
    const disposeSpy = vi
      .spyOn(MicrophoneCapture.prototype, "dispose")
      .mockImplementation(function (this: MicrophoneCapture) {
        // no-op for the default mic — we are not testing teardown
        // here, only the retry plumbing.
      });

    try {
      const vad = makeStubVad();
      const pipe = await createAudioPipeline({ vad });

      await pipe.start();
      expect(pipe.pipelineState).toBe("permission-denied");

      await pipe.start();
      expect(pipe.pipelineState).toBe("running");
      expect(callCount).toBe(2);
    } finally {
      startSpy.mockRestore();
      disposeSpy.mockRestore();
    }
  });
});

describe("createAudioPipeline — generic errors", () => {
  it("transitions to 'error' on a non-permission error and exposes the underlying message", async () => {
    const hardware = new Error("hardware busy");
    hardware.name = "NotReadableError";
    const mic = makeStubMic(async () => {
      throw hardware;
    });
    const vad = makeStubVad();
    const pipe = await createAudioPipeline({ vad, microphone: asMic(mic) });

    const seen: PipelineState[] = [];
    pipe.subscribe((s) => seen.push(s));

    await expect(pipe.start()).resolves.toBeUndefined();
    expect(pipe.pipelineState).toBe("error");
    expect(seen).toEqual(["starting", "error"]);
    expect(pipe.getError()).toMatch(/hardware busy/);
  });

  it("does not mistake a generic Error for a permission denial", async () => {
    const mic = makeStubMic(async () => {
      throw new Error("worklet load failed");
    });
    const vad = makeStubVad();
    const pipe = await createAudioPipeline({ vad, microphone: asMic(mic) });

    await pipe.start();
    expect(pipe.pipelineState).toBe("error");
    // Not "permission-denied" — the typed-class check distinguishes
    // the two without parsing the message.
    expect(pipe.pipelineState).not.toBe("permission-denied");
  });
});

describe("createAudioPipeline — dispose()", () => {
  it("dispose() before any start() is a safe no-op", async () => {
    const vad = makeStubVad();
    const mic = makeStubMic();
    const pipe = await createAudioPipeline({ vad, microphone: asMic(mic) });

    expect(() => pipe.dispose()).not.toThrow();
    // Even on a never-started pipeline, dispose disposes the injected mic.
    expect(mic.dispose).toHaveBeenCalledTimes(1);
  });

  it("dispose() clears mic.onFrame and disposes the mic", async () => {
    const vad = makeStubVad();
    const mic = makeStubMic();
    const pipe = await createAudioPipeline({ vad, microphone: asMic(mic) });
    await pipe.start();
    expect(mic.onFrame).toBeTypeOf("function");

    pipe.dispose();
    expect(mic.onFrame).toBeNull();
    expect(mic.dispose).toHaveBeenCalledTimes(1);
  });

  it("dispose() does NOT dispose the VAD (lifecycle separation)", async () => {
    const vad = makeStubVad();
    const mic = makeStubMic();
    const pipe = await createAudioPipeline({ vad, microphone: asMic(mic) });
    await pipe.start();

    pipe.dispose();
    expect(vad.dispose).not.toHaveBeenCalled();
  });

  it("dispose() runs cleanly from 'permission-denied'", async () => {
    const mic = makeStubMic(async () => {
      throw new MicPermissionDeniedError();
    });
    const vad = makeStubVad();
    const pipe = await createAudioPipeline({ vad, microphone: asMic(mic) });
    await pipe.start();
    expect(pipe.pipelineState).toBe("permission-denied");

    expect(() => pipe.dispose()).not.toThrow();
    expect(mic.dispose).toHaveBeenCalledTimes(1);
    expect(vad.dispose).not.toHaveBeenCalled();
  });

  it("dispose() runs cleanly from 'error'", async () => {
    const mic = makeStubMic(async () => {
      throw new Error("hardware busy");
    });
    const vad = makeStubVad();
    const pipe = await createAudioPipeline({ vad, microphone: asMic(mic) });
    await pipe.start();
    expect(pipe.pipelineState).toBe("error");

    expect(() => pipe.dispose()).not.toThrow();
  });

  it("second dispose() is a safe no-op (mic.dispose not called twice)", async () => {
    const vad = makeStubVad();
    const mic = makeStubMic();
    const pipe = await createAudioPipeline({ vad, microphone: asMic(mic) });

    pipe.dispose();
    expect(() => pipe.dispose()).not.toThrow();
    expect(mic.dispose).toHaveBeenCalledTimes(1);
  });

  it("start() after dispose() rejects", async () => {
    const vad = makeStubVad();
    const mic = makeStubMic();
    const pipe = await createAudioPipeline({ vad, microphone: asMic(mic) });

    pipe.dispose();
    await expect(pipe.start()).rejects.toThrow(/cannot start.*after dispose/i);
  });

  it("dispose() survives a stub mic whose .dispose() throws", async () => {
    const mic = makeStubMic();
    mic.dispose = vi.fn(() => {
      throw new Error("dispose boom");
    });
    const vad = makeStubVad();
    const pipe = await createAudioPipeline({ vad, microphone: asMic(mic) });

    expect(() => pipe.dispose()).not.toThrow();
  });
});

describe("createAudioPipeline — subscription", () => {
  it("unsubscribe stops further notifications", async () => {
    const vad = makeStubVad();
    const mic = makeStubMic();
    const pipe = await createAudioPipeline({ vad, microphone: asMic(mic) });

    const seen: PipelineState[] = [];
    const unsubscribe = pipe.subscribe((s) => seen.push(s));
    unsubscribe();

    await pipe.start();
    expect(seen).toEqual([]);
    // But the state still transitioned — observable from the getter.
    expect(pipe.pipelineState).toBe("running");
  });

  it("a listener that throws does not block notifications to other listeners", async () => {
    const vad = makeStubVad();
    const mic = makeStubMic();
    const pipe = await createAudioPipeline({ vad, microphone: asMic(mic) });

    const seen: PipelineState[] = [];
    pipe.subscribe(() => {
      throw new Error("buggy listener");
    });
    pipe.subscribe((s) => seen.push(s));

    await pipe.start();
    expect(seen).toEqual(["starting", "running"]);
  });
});

describe("createAudioPipeline — getError()", () => {
  it("returns null in idle / starting / running", async () => {
    const vad = makeStubVad();
    const mic = makeStubMic();
    const pipe = await createAudioPipeline({ vad, microphone: asMic(mic) });

    expect(pipe.getError()).toBeNull();
    await pipe.start();
    expect(pipe.pipelineState).toBe("running");
    expect(pipe.getError()).toBeNull();
  });

  it("clears lastError on a successful start that follows a previous failure", async () => {
    // Stub mic that fails first, succeeds on second call.
    let calls = 0;
    const mic = makeStubMic(async () => {
      calls++;
      if (calls === 1) throw new MicPermissionDeniedError();
    });
    const vad = makeStubVad();
    const pipe = await createAudioPipeline({ vad, microphone: asMic(mic) });

    await pipe.start();
    expect(pipe.getError()).not.toBeNull();

    await pipe.start();
    expect(pipe.pipelineState).toBe("running");
    expect(pipe.getError()).toBeNull();
  });
});

describe("createAudioPipeline — type re-exports", () => {
  it("AudioPipeline type is a structural interface (compile-time check)", () => {
    // This test exists to anchor the type at runtime — if a future
    // refactor accidentally drops a field from the interface, this
    // assignment fails to compile and the test never runs.
    const _typeCheck: AudioPipeline = {
      pipelineState: "idle",
      start: async () => undefined,
      subscribe: () => () => undefined,
      getError: () => null,
      dispose: () => undefined,
    };
    expect(_typeCheck.pipelineState).toBe("idle");
  });
});
