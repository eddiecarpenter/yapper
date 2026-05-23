import { vi } from "vitest";

import type { DialogueOptions } from "../types";
import type { Transcriber } from "../../stt/types";
import type { Speaker } from "../../tts/types";
import type { VAD } from "../../vad/types";

/**
 * Test fixture: in-memory stubs for the three browser-side module
 * contracts plus a `DialogueOptions` builder. Each stub exposes the
 * mock functions so tests can assert call counts and arguments.
 */
export interface StubDeps {
  transcriber: Transcriber & {
    transcribe: ReturnType<typeof vi.fn>;
    dispose: ReturnType<typeof vi.fn>;
  };
  speaker: Speaker & {
    speak: ReturnType<typeof vi.fn>;
    cancel: ReturnType<typeof vi.fn>;
    dispose: ReturnType<typeof vi.fn>;
  };
  vad: VAD & {
    process: ReturnType<typeof vi.fn>;
    dispose: ReturnType<typeof vi.fn>;
  };
  /**
   * Convenience accessor: returns the currently-installed `onSpeechEnd`
   * callback, or throws if the hook has not registered one yet.
   */
  fireSpeechEnd: (segment?: Float32Array) => Promise<void>;
}

export function makeStubDeps(
  overrides: {
    transcribe?: (audio: Float32Array, rate: number) => Promise<string>;
    speak?: (text: string) => Promise<void>;
  } = {},
): StubDeps {
  const transcribe = vi.fn(
    overrides.transcribe ?? (async (_audio: Float32Array, _rate: number) => "transcribed"),
  );
  const speak = vi.fn(overrides.speak ?? (async (_text: string) => undefined));

  const transcriber = {
    transcribe,
    dispose: vi.fn(),
  };
  const speaker = {
    speak,
    cancel: vi.fn(),
    dispose: vi.fn(),
  };
  const vad: VAD & { process: ReturnType<typeof vi.fn>; dispose: ReturnType<typeof vi.fn> } = {
    process: vi.fn((_frame: Float32Array) => false),
    dispose: vi.fn(),
  };

  const fireSpeechEnd = async (segment?: Float32Array): Promise<void> => {
    const cb = vad.onSpeechEnd;
    if (!cb) throw new Error("vad.onSpeechEnd has not been registered by the hook yet");
    await cb(segment ?? new Float32Array([0.1, 0.2, 0.3]));
  };

  return { transcriber, speaker, vad, fireSpeechEnd };
}

export function makeOptions(
  deps: StubDeps,
  overrides: Partial<DialogueOptions> = {},
): DialogueOptions {
  return {
    transcriber: deps.transcriber,
    speaker: deps.speaker,
    vad: deps.vad,
    relayUrl: "ws://localhost:8080/ws",
    ...overrides,
  };
}
