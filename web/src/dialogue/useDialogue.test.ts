import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { DEFAULT_CONTEXT_BUDGET, INITIAL_DIALOGUE_STATE, useDialogue } from "./index";
import type { DialogueStage, DialogueState, TimingRecord } from "./index";
import type { Message } from "../llm/types";

import { MockWebSocket, makeOptions, makeStubDeps } from "./__test-helpers__";

beforeEach(() => {
  MockWebSocket.reset();
  vi.stubGlobal("WebSocket", MockWebSocket);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

/** Resolve the WebSocket instance the hook just constructed. */
function getWs(): MockWebSocket {
  const ws = MockWebSocket.instances[0];
  if (!ws) throw new Error("hook did not construct a WebSocket");
  return ws;
}

/**
 * End-to-end drive helper: opens the WS, lets the hook reach `listening`,
 * fires VAD speech-end, streams the supplied tokens + `{done}`, and
 * resolves when `state.stage` returns to `listening`.
 */
async function driveSingleTurn(
  result: { current: DialogueState },
  fire: () => Promise<void>,
  ws: MockWebSocket,
  tokens: string[],
): Promise<void> {
  await act(async () => {
    ws.triggerOpen();
  });
  await waitFor(() => expect(result.current.stage).toBe<DialogueStage>("listening"));

  await act(async () => {
    // fire returns once the per-stage `await`s have all yielded enough
    // for the WS send to land — but the listener handler resolves
    // synchronously below, so we don't await here.
    void fire();
    await Promise.resolve();
  });

  await act(async () => {
    for (const t of tokens) {
      ws.triggerMessage({ type: "token", text: t });
    }
    ws.triggerMessage({ type: "done", usage: { input: 1, output: tokens.length } });
  });

  await waitFor(() => expect(result.current.stage).toBe<DialogueStage>("listening"));
}

describe("useDialogue — Task 2: voice turn loop and timing", () => {
  it("opens a WebSocket to relayUrl on mount", () => {
    const deps = makeStubDeps();
    renderHook(() => useDialogue(makeOptions(deps, { relayUrl: "ws://localhost:9999/ws" })));

    expect(MockWebSocket.instances).toHaveLength(1);
    expect(MockWebSocket.instances[0]!.url).toBe("ws://localhost:9999/ws");
  });

  it("registers vad.onSpeechEnd on mount", () => {
    const deps = makeStubDeps();
    renderHook(() => useDialogue(makeOptions(deps)));

    expect(deps.vad.onSpeechEnd).toBeTypeOf("function");
  });

  it("transitions idle → listening when the WS opens", async () => {
    const deps = makeStubDeps();
    const { result } = renderHook(() => useDialogue(makeOptions(deps)));

    expect(result.current.stage).toBe<DialogueStage>("idle");

    await act(async () => {
      getWs().triggerOpen();
    });

    await waitFor(() => expect(result.current.stage).toBe<DialogueStage>("listening"));
  });

  it("drives transcribing → relaying → speaking → listening for a complete turn", async () => {
    const deps = makeStubDeps({
      transcribe: async () => "what's the time?",
      speak: async () => undefined,
    });
    const { result } = renderHook(() => useDialogue(makeOptions(deps)));

    await driveSingleTurn(result, () => deps.fireSpeechEnd(), getWs(), [
      "The",
      " time",
      " is",
      " noon",
    ]);

    expect(deps.transcriber.transcribe).toHaveBeenCalledTimes(1);
    expect(deps.transcriber.transcribe.mock.calls[0]).toEqual([expect.any(Float32Array), 16000]);
    expect(deps.speaker.speak).toHaveBeenCalledTimes(1);
    expect(deps.speaker.speak).toHaveBeenCalledWith("The time is noon");
    expect(result.current.stage).toBe<DialogueStage>("listening");
  });

  it("sends a {type:'turn'} payload with the transcribed text and current history", async () => {
    const deps = makeStubDeps({ transcribe: async () => "hello" });
    const { result } = renderHook(() => useDialogue(makeOptions(deps)));

    await driveSingleTurn(result, () => deps.fireSpeechEnd(), getWs(), ["world"]);

    const sent = getWs().lastSent<{ type: string; text: string; history: Message[] }>();
    expect(sent?.type).toBe("turn");
    expect(sent?.text).toBe("hello");
    // Task 2: history is still empty until Task 3 wires accumulation in.
    expect(sent?.history).toEqual([]);
  });

  it("emits a TimingRecord with all six numeric fields via console.table", async () => {
    const tableSpy = vi.spyOn(console, "table").mockImplementation(() => undefined);

    const deps = makeStubDeps({ transcribe: async () => "x", speak: async () => undefined });
    const { result } = renderHook(() => useDialogue(makeOptions(deps)));

    await driveSingleTurn(result, () => deps.fireSpeechEnd(), getWs(), ["tok"]);

    expect(tableSpy).toHaveBeenCalledTimes(1);
    const arg = tableSpy.mock.calls[0]?.[0] as TimingRecord;
    expect(arg).toBeDefined();
    expect(arg.vad_ms).toBe(0);
    for (const field of [
      "stt_ms",
      "llm_first_token_ms",
      "llm_total_ms",
      "tts_ms",
      "total_ms",
    ] as const) {
      expect(typeof arg[field]).toBe("number");
      expect(arg[field]).toBeGreaterThanOrEqual(0);
    }
    // Total is the sum of stages — must be ≥ each individual leg.
    expect(arg.total_ms).toBeGreaterThanOrEqual(arg.stt_ms);
    expect(arg.total_ms).toBeGreaterThanOrEqual(arg.llm_total_ms);

    expect(result.current.lastTiming).toEqual(arg);
  });

  it("closes the WebSocket and detaches the VAD callback on unmount", async () => {
    const deps = makeStubDeps();
    const { unmount } = renderHook(() => useDialogue(makeOptions(deps)));

    const ws = getWs();
    expect(deps.vad.onSpeechEnd).toBeTypeOf("function");

    unmount();

    expect(ws.readyState).toBe(MockWebSocket.CLOSED);
    expect(deps.vad.onSpeechEnd).toBeUndefined();
  });
});

describe("useDialogue — Task 1 scaffold contracts (still hold)", () => {
  it("returns the initial idle state on first render (before WS opens)", () => {
    const deps = makeStubDeps();
    const { result } = renderHook(() => useDialogue(makeOptions(deps)));

    expect(result.current).toEqual(INITIAL_DIALOGUE_STATE);
  });

  it("does not throw when relayUrl is empty (validation deferred to Task 4)", () => {
    const deps = makeStubDeps();
    expect(() => renderHook(() => useDialogue(makeOptions(deps, { relayUrl: "" })))).not.toThrow();
  });
});

describe("dialogue module surface", () => {
  it("exports the documented public surface", () => {
    expect(useDialogue).toBeTypeOf("function");
    expect(DEFAULT_CONTEXT_BUDGET).toBe(4096);
    expect(INITIAL_DIALOGUE_STATE.stage).toBe("idle");
  });

  it("DialogueStage covers every documented stage", () => {
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
