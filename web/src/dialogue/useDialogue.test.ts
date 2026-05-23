import { describe, expect, it, vi } from "vitest";
import { renderHook } from "@testing-library/react";

import { DEFAULT_CONTEXT_BUDGET, INITIAL_DIALOGUE_STATE, useDialogue } from "./index";
import type { DialogueOptions, DialogueStage, DialogueState, TimingRecord } from "./index";
import type { Message } from "../llm/types";
import type { Transcriber } from "../stt/types";
import type { Speaker } from "../tts/types";
import type { VAD } from "../vad/types";

/** Builds a stub set of dependencies that satisfies `DialogueOptions`. */
function buildStubOptions(overrides: Partial<DialogueOptions> = {}): DialogueOptions {
  const transcriber: Transcriber = {
    transcribe: vi.fn(async (_audio: Float32Array, _rate: number) => ""),
    dispose: vi.fn(),
  };
  const speaker: Speaker = {
    speak: vi.fn(async (_text: string) => undefined),
    cancel: vi.fn(),
    dispose: vi.fn(),
  };
  const vad: VAD = {
    process: vi.fn((_frame: Float32Array) => false),
    dispose: vi.fn(),
  };
  return {
    transcriber,
    speaker,
    vad,
    relayUrl: "ws://localhost:8080/ws",
    ...overrides,
  };
}

describe("useDialogue (Task 1 scaffold)", () => {
  it("returns the initial idle state on first render", () => {
    const opts = buildStubOptions();

    const { result } = renderHook(() => useDialogue(opts));

    expect(result.current).toEqual(INITIAL_DIALOGUE_STATE);
    expect(result.current.stage).toBe<DialogueStage>("idle");
    expect(result.current.history).toEqual([]);
    expect(result.current.lastTiming).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it("accepts a custom contextBudget without crashing", () => {
    const opts = buildStubOptions({ contextBudget: 1024 });

    const { result } = renderHook(() => useDialogue(opts));

    // The stub does nothing with contextBudget yet — it just must not throw.
    expect(result.current.stage).toBe("idle");
  });

  it("does not invoke any dependency in the scaffold (no side effects yet)", () => {
    const opts = buildStubOptions();

    renderHook(() => useDialogue(opts));

    expect(opts.transcriber.transcribe).not.toHaveBeenCalled();
    expect(opts.speaker.speak).not.toHaveBeenCalled();
    expect(opts.vad.process).not.toHaveBeenCalled();
  });

  it("does not throw when relayUrl is empty (validation deferred to Task 4)", () => {
    const opts = buildStubOptions({ relayUrl: "" });
    expect(() => renderHook(() => useDialogue(opts))).not.toThrow();
  });
});

describe("dialogue module surface", () => {
  it("exports the documented public surface", () => {
    expect(useDialogue).toBeTypeOf("function");
    expect(DEFAULT_CONTEXT_BUDGET).toBe(4096);
    expect(INITIAL_DIALOGUE_STATE.stage).toBe("idle");
  });

  it("DialogueStage covers every documented stage", () => {
    // Compile-time check: every stage must be assignable to DialogueStage.
    const stages: DialogueStage[] = [
      "idle",
      "listening",
      "transcribing",
      "relaying",
      "speaking",
      "error",
    ];
    expect(stages).toHaveLength(6);
  });

  it("TimingRecord shape contains all six latency fields", () => {
    const sample: TimingRecord = {
      vad_ms: 0,
      stt_ms: 1,
      llm_first_token_ms: 2,
      llm_total_ms: 3,
      tts_ms: 4,
      total_ms: 5,
    };
    expect(Object.keys(sample)).toEqual([
      "vad_ms",
      "stt_ms",
      "llm_first_token_ms",
      "llm_total_ms",
      "tts_ms",
      "total_ms",
    ]);
  });

  it("DialogueState.history accepts a typed Message array", () => {
    const history: Message[] = [
      { role: "system", content: "You are Yapper." },
      { role: "user", content: "Hi." },
      { role: "assistant", content: "Hello." },
    ];
    const sample: DialogueState = {
      stage: "listening",
      history,
      lastTiming: null,
      error: null,
    };
    expect(sample.history).toHaveLength(3);
  });
});
