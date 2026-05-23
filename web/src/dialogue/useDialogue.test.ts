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

describe("useDialogue — Task 3: conversation history and truncation", () => {
  it("accumulates user + assistant messages after each turn", async () => {
    let replyCounter = 0;
    const transcripts = ["hello", "again", "third"];
    const replies = ["hi back", "and again", "third reply"];

    const deps = makeStubDeps({
      transcribe: async () => transcripts[replyCounter] ?? "",
    });
    const { result } = renderHook(() => useDialogue(makeOptions(deps)));

    for (let turn = 0; turn < 3; turn++) {
      const reply = replies[turn] ?? "";
      await driveSingleTurn(result, () => deps.fireSpeechEnd(), getWs(), [reply]);
      replyCounter++;
    }

    expect(result.current.history).toHaveLength(6);
    expect(result.current.history.map((m) => m.role)).toEqual([
      "user",
      "assistant",
      "user",
      "assistant",
      "user",
      "assistant",
    ]);
    expect(result.current.history.map((m) => m.content)).toEqual([
      "hello",
      "hi back",
      "again",
      "and again",
      "third",
      "third reply",
    ]);
  });

  it("sends prior history (before this turn's append) to the relay", async () => {
    let counter = 0;
    const deps = makeStubDeps({
      transcribe: async () => `utterance-${counter}`,
    });
    const { result } = renderHook(() => useDialogue(makeOptions(deps)));

    // Turn 1 — no prior history sent
    await driveSingleTurn(result, () => deps.fireSpeechEnd(), getWs(), ["reply-0"]);
    counter++;
    let sent = getWs().lastSent<{ history: Message[] }>();
    expect(sent?.history).toEqual([]);

    // Turn 2 — prior history = [user:utterance-0, assistant:reply-0]
    await driveSingleTurn(result, () => deps.fireSpeechEnd(), getWs(), ["reply-1"]);
    counter++;
    sent = getWs().lastSent<{ history: Message[] }>();
    expect(sent?.history).toEqual([
      { role: "user", content: "utterance-0" },
      { role: "assistant", content: "reply-0" },
    ]);

    // Turn 3 — prior history = 4 messages
    await driveSingleTurn(result, () => deps.fireSpeechEnd(), getWs(), ["reply-2"]);
    sent = getWs().lastSent<{ history: Message[] }>();
    expect(sent?.history).toHaveLength(4);
    expect(sent?.history[3]).toEqual({ role: "assistant", content: "reply-1" });
  });

  it("drops the oldest non-system message when contextBudget is exceeded", async () => {
    // Budget of 4 estimated tokens. Each turn adds 2 single-word messages
    // → 2 words → ceil(2 * 1.3) = 3 tokens. After two turns we'd have 4
    // messages → 4 words → ceil(4 * 1.3) = 6 tokens > 4 → truncate.
    const deps = makeStubDeps({ transcribe: async () => "user" });
    const { result } = renderHook(() => useDialogue(makeOptions(deps, { contextBudget: 4 })));

    await driveSingleTurn(result, () => deps.fireSpeechEnd(), getWs(), ["assist"]);
    await driveSingleTurn(result, () => deps.fireSpeechEnd(), getWs(), ["assist"]);

    // After two turns + truncation: history must be shorter than 4.
    expect(result.current.history.length).toBeLessThan(4);
    expect(result.current.history.length).toBeGreaterThan(0);
  });

  it("retains the leading system message under aggressive truncation", async () => {
    const deps = makeStubDeps({ transcribe: async () => "userword" });
    const seedSystem: Message = { role: "system", content: "You are Yapper." };

    const { result } = renderHook(() =>
      useDialogue(
        makeOptions(deps, {
          contextBudget: 3,
          initialHistory: [seedSystem],
        }),
      ),
    );

    for (let i = 0; i < 3; i++) {
      await driveSingleTurn(result, () => deps.fireSpeechEnd(), getWs(), ["replyword"]);
    }

    // System prompt is still index 0 …
    expect(result.current.history[0]).toEqual(seedSystem);
    // … and some non-system messages have been dropped.
    expect(result.current.history.length).toBeLessThan(1 + 6);
  });

  it("seeds history from opts.initialHistory on mount", () => {
    const deps = makeStubDeps();
    const seed: Message[] = [
      { role: "system", content: "primer" },
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
    ];
    const { result } = renderHook(() => useDialogue(makeOptions(deps, { initialHistory: seed })));

    expect(result.current.history).toEqual(seed);
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

describe("useDialogue — Task 4: error handling (AC-3) and lifecycle cleanup (AC-4)", () => {
  /**
   * Opens the WS, waits for the hook to reach `"listening"`, then drives
   * one turn:
   *   1. Calls `fire()` (returns the in-flight handleSpeechEnd promise)
   *   2. Yields one microtask so the handler reaches its first await
   *   3. Runs `triggerFrames()` to deliver any relay frames
   *   4. Awaits the handleSpeechEnd promise so the catch block (if any)
   *      has completed before the test continues
   * All four steps live inside a single `act()` so RTL batches the
   * resulting dispatches deterministically.
   */
  async function driveOneTurn(
    result: { current: DialogueState },
    deps: ReturnType<typeof makeStubDeps>,
    triggerFrames: () => void,
  ): Promise<void> {
    await act(async () => {
      getWs().triggerOpen();
    });
    await waitFor(() => expect(result.current.stage).toBe<DialogueStage>("listening"));

    await act(async () => {
      const p = deps.fireSpeechEnd();
      // Two microtask ticks: one for fireSpeechEnd → handleSpeechEnd's
      // first await, one for the async function transcribe() to start.
      await Promise.resolve();
      await Promise.resolve();
      triggerFrames();
      await p;
    });
  }

  it("transitions to error with the relay's message on {type:'error'} frame", async () => {
    const deps = makeStubDeps();
    const { result } = renderHook(() => useDialogue(makeOptions(deps)));

    await driveOneTurn(result, deps, () => {
      getWs().triggerMessage({ type: "error", message: "Ollama unreachable at localhost:11434" });
    });

    expect(result.current.stage).toBe<DialogueStage>("error");
    expect(result.current.error).toBe("Ollama unreachable at localhost:11434");
    expect(deps.speaker.speak).not.toHaveBeenCalled();
  });

  it("auto-recovers from error back to listening after the recovery delay", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      const deps = makeStubDeps();
      const { result } = renderHook(() => useDialogue(makeOptions(deps)));

      await driveOneTurn(result, deps, () => {
        getWs().triggerMessage({ type: "error", message: "boom" });
      });
      expect(result.current.stage).toBe<DialogueStage>("error");

      await act(async () => {
        await vi.advanceTimersByTimeAsync(ERROR_RECOVERY_DELAY_MS_EXPECTED);
      });

      expect(result.current.stage).toBe<DialogueStage>("listening");
      expect(result.current.error).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it("surfaces 'Speech recognition failed' when transcribe() throws", async () => {
    const deps = makeStubDeps({
      transcribe: async () => {
        throw new Error("WebGPU lost device");
      },
    });
    const { result } = renderHook(() => useDialogue(makeOptions(deps)));

    await driveOneTurn(result, deps, () => {
      // No relay frames — failure happens before send.
    });

    expect(result.current.stage).toBe<DialogueStage>("error");
    expect(result.current.error).toBe("Speech recognition failed — check WebGPU/WASM availability");
    expect(getWs().sent).toHaveLength(0);
  });

  it("surfaces 'Speech synthesis failed' when speak() throws", async () => {
    const deps = makeStubDeps({
      speak: async () => {
        throw new Error("AudioContext suspended");
      },
    });
    const { result } = renderHook(() => useDialogue(makeOptions(deps)));

    await driveOneTurn(result, deps, () => {
      const ws = getWs();
      ws.triggerMessage({ type: "token", text: "hi" });
      ws.triggerMessage({ type: "done", usage: { input: 1, output: 1 } });
    });

    expect(result.current.stage).toBe<DialogueStage>("error");
    expect(result.current.error).toBe("Speech synthesis failed — check WebGPU/WASM availability");
  });

  it("surfaces 'Relay unreachable' when ws.send() throws", async () => {
    const deps = makeStubDeps();
    const { result } = renderHook(() =>
      useDialogue(makeOptions(deps, { relayUrl: "ws://broken:1234/ws" })),
    );

    await act(async () => {
      getWs().triggerOpen();
    });
    await waitFor(() => expect(result.current.stage).toBe<DialogueStage>("listening"));

    // Close the WS so the next ws.send() throws inside the turn.
    await act(async () => {
      getWs().close();
    });

    await act(async () => {
      const p = deps.fireSpeechEnd();
      await Promise.resolve();
      await Promise.resolve();
      await p;
    });

    expect(result.current.stage).toBe<DialogueStage>("error");
    expect(result.current.error).toBe(
      "Relay unreachable — check that the Go server is running at ws://broken:1234/ws",
    );
  });

  it("surfaces 'Relay unreachable' when the WS closes mid-turn", async () => {
    const deps = makeStubDeps({ transcribe: async () => "hi" });
    const { result } = renderHook(() =>
      useDialogue(makeOptions(deps, { relayUrl: "ws://localhost:8080/ws" })),
    );

    await driveOneTurn(result, deps, () => {
      // Close the socket between send and {done} → triggers the close
      // listener inside collectRelayTurn → rejects with connection error.
      getWs().close();
    });

    expect(result.current.stage).toBe<DialogueStage>("error");
    expect(result.current.error).toBe(
      "Relay unreachable — check that the Go server is running at ws://localhost:8080/ws",
    );
  });

  it("calls speaker.cancel() on unmount if mid-utterance", async () => {
    // A `speak()` that never resolves — keeps the hook in the "speaking"
    // stage until the test unmounts.
    let resolveSpeak: () => void = () => undefined;
    const speakPromise = new Promise<void>((res) => {
      resolveSpeak = res;
    });
    const deps = makeStubDeps({
      transcribe: async () => "hi",
      speak: () => speakPromise,
    });
    const { result, unmount } = renderHook(() => useDialogue(makeOptions(deps)));

    await act(async () => {
      getWs().triggerOpen();
    });
    await waitFor(() => expect(result.current.stage).toBe<DialogueStage>("listening"));

    // Start the turn and let it advance to "speaking" — but never await
    // the handler promise, because speak() will never resolve.
    await act(async () => {
      void deps.fireSpeechEnd();
      await Promise.resolve();
      await Promise.resolve();
      getWs().triggerMessage({ type: "token", text: "hi" });
      getWs().triggerMessage({ type: "done" });
      // Yield enough ticks for the dispatch into "speaking" to land.
      await Promise.resolve();
      await Promise.resolve();
    });
    await waitFor(() => expect(result.current.stage).toBe<DialogueStage>("speaking"));

    unmount();

    expect(deps.speaker.cancel).toHaveBeenCalledTimes(1);

    // Release the dangling promise so vitest cleans up.
    resolveSpeak();
  });

  it("calls transcriber.dispose() and vad.dispose() exactly once on unmount", () => {
    const deps = makeStubDeps();
    const { unmount } = renderHook(() => useDialogue(makeOptions(deps)));

    expect(deps.transcriber.dispose).not.toHaveBeenCalled();
    expect(deps.vad.dispose).not.toHaveBeenCalled();

    unmount();

    expect(deps.transcriber.dispose).toHaveBeenCalledTimes(1);
    expect(deps.vad.dispose).toHaveBeenCalledTimes(1);
  });

  it("does not call speaker.cancel() on unmount when not speaking", () => {
    const deps = makeStubDeps();
    const { unmount } = renderHook(() => useDialogue(makeOptions(deps)));

    unmount();

    expect(deps.speaker.cancel).not.toHaveBeenCalled();
  });

  it("cancels a pending auto-recovery timer on unmount", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      const deps = makeStubDeps();
      const { result, unmount } = renderHook(() => useDialogue(makeOptions(deps)));

      await driveOneTurn(result, deps, () => {
        getWs().triggerMessage({ type: "error", message: "boom" });
      });
      expect(result.current.stage).toBe<DialogueStage>("error");

      // Unmount BEFORE the recovery timer fires.
      unmount();

      // Advance past the recovery delay — the cleared timer must not
      // fire, and no dispatch-after-unmount warning may be raised.
      await vi.advanceTimersByTimeAsync(1000);

      expect(deps.vad.dispose).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it("survives a subsequent successful turn after auto-recovery", async () => {
    let firstAttempt = true;
    const deps = makeStubDeps({
      transcribe: async () => {
        if (firstAttempt) {
          firstAttempt = false;
          throw new Error("first attempt fails");
        }
        return "second-utterance";
      },
    });
    const { result } = renderHook(() => useDialogue(makeOptions(deps)));

    // First turn fails → error.
    await driveOneTurn(result, deps, () => {
      // No frames — transcribe rejects before send.
    });
    expect(result.current.stage).toBe<DialogueStage>("error");

    // Real timers — wait for the 500ms recovery to elapse naturally.
    await waitFor(() => expect(result.current.stage).toBe<DialogueStage>("listening"), {
      timeout: 2000,
    });

    // Second turn succeeds.
    await act(async () => {
      const p = deps.fireSpeechEnd();
      await Promise.resolve();
      await Promise.resolve();
      getWs().triggerMessage({ type: "token", text: "ok" });
      getWs().triggerMessage({ type: "done" });
      await p;
    });

    expect(result.current.stage).toBe<DialogueStage>("listening");
    expect(result.current.history).toHaveLength(2);
    expect(result.current.history[0]?.content).toBe("second-utterance");
    expect(result.current.history[1]?.content).toBe("ok");
  });
});

/** Mirror of the production constant — kept here so the test imports stay scoped. */
const ERROR_RECOVERY_DELAY_MS_EXPECTED = 500;
