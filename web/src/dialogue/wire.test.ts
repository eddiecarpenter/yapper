import { describe, expect, it } from "vitest";

import { STT_SAMPLE_RATE_HZ, parseRelayFrame } from "./wire";

describe("parseRelayFrame", () => {
  it("parses a token frame", () => {
    const out = parseRelayFrame(JSON.stringify({ type: "token", text: "hi" }));
    expect(out).toEqual({ type: "token", text: "hi" });
  });

  it("parses a done frame without usage", () => {
    const out = parseRelayFrame(JSON.stringify({ type: "done" }));
    expect(out).toEqual({ type: "done" });
  });

  it("parses a done frame with usage", () => {
    const out = parseRelayFrame(JSON.stringify({ type: "done", usage: { input: 12, output: 8 } }));
    expect(out).toEqual({ type: "done", usage: { input: 12, output: 8 } });
  });

  it("parses an error frame", () => {
    const out = parseRelayFrame(JSON.stringify({ type: "error", message: "boom" }));
    expect(out).toEqual({ type: "error", message: "boom" });
  });

  it("returns null for non-string input", () => {
    expect(parseRelayFrame(42)).toBeNull();
    expect(parseRelayFrame(null)).toBeNull();
    expect(parseRelayFrame(undefined)).toBeNull();
  });

  it("returns null for malformed JSON", () => {
    expect(parseRelayFrame("not json")).toBeNull();
  });

  it("returns null for unknown type", () => {
    expect(parseRelayFrame(JSON.stringify({ type: "fnord" }))).toBeNull();
  });

  it("returns null for token frame missing text", () => {
    expect(parseRelayFrame(JSON.stringify({ type: "token" }))).toBeNull();
  });

  it("returns null for error frame missing message", () => {
    expect(parseRelayFrame(JSON.stringify({ type: "error" }))).toBeNull();
  });

  it("returns null for a JSON null payload", () => {
    expect(parseRelayFrame("null")).toBeNull();
  });

  it("exports the STT sample rate as 16000 Hz", () => {
    expect(STT_SAMPLE_RATE_HZ).toBe(16000);
  });
});
