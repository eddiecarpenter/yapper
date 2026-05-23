import { describe, expect, it } from "vitest";

import { appendTurn, estimateTokenCount, truncateHistory } from "./history";
import type { Message } from "../llm/types";

describe("estimateTokenCount", () => {
  it("returns 0 for empty history", () => {
    expect(estimateTokenCount([])).toBe(0);
  });

  it("applies Math.ceil(words * 1.3) per the design plan heuristic", () => {
    const msgs: Message[] = [{ role: "user", content: "hello world" }]; // 2 words
    // 2 * 1.3 = 2.6 → ceil = 3
    expect(estimateTokenCount(msgs)).toBe(3);
  });

  it("aggregates across messages", () => {
    const msgs: Message[] = [
      { role: "system", content: "You are Yapper." }, // 3 words
      { role: "user", content: "Hi there friend" }, // 3 words
      { role: "assistant", content: "Hello!" }, // 1 word
    ];
    // 7 * 1.3 = 9.1 → ceil = 10
    expect(estimateTokenCount(msgs)).toBe(10);
  });
});

describe("truncateHistory", () => {
  it("returns history unchanged when under budget", () => {
    const h: Message[] = [{ role: "user", content: "small" }];
    expect(truncateHistory(h, 4096)).toEqual(h);
  });

  it("drops the oldest non-system message when over budget", () => {
    const h: Message[] = [
      { role: "user", content: "first" },
      { role: "assistant", content: "reply" },
      { role: "user", content: "second" },
      { role: "assistant", content: "reply2" },
    ];
    // 4 * 1.3 = 5.2 → ceil 6. Budget 4 forces removal until <= 4.
    const out = truncateHistory(h, 4);
    expect(out.length).toBeLessThan(h.length);
    // The oldest non-system message must be the dropped one — and the
    // most recent assistant reply must still be present.
    expect(out[out.length - 1]).toEqual({ role: "assistant", content: "reply2" });
  });

  it("retains a leading system message even under aggressive truncation", () => {
    const h: Message[] = [
      { role: "system", content: "You are Yapper. Be concise." },
      { role: "user", content: "old1" },
      { role: "assistant", content: "old1-reply" },
      { role: "user", content: "old2" },
      { role: "assistant", content: "old2-reply" },
      { role: "user", content: "recent" },
      { role: "assistant", content: "recent-reply" },
    ];
    const out = truncateHistory(h, 2);
    // The system prompt must still be at index 0.
    expect(out[0]?.role).toBe("system");
    // At least one non-system message was dropped.
    expect(out.length).toBeLessThan(h.length);
  });

  it("does not drop the system prompt even when only it remains over budget", () => {
    const h: Message[] = [{ role: "system", content: "long system prompt content" }];
    const out = truncateHistory(h, 1);
    // estimate is ceil(4 * 1.3) = 6, > 1. But truncate must NOT remove the system prompt.
    expect(out).toEqual(h);
  });

  it("treats contextBudget <= 0 as a defensive 'keep only system' floor", () => {
    const h: Message[] = [
      { role: "system", content: "sys" },
      { role: "user", content: "drop me" },
    ];
    expect(truncateHistory(h, 0)).toEqual([{ role: "system", content: "sys" }]);
    expect(truncateHistory([{ role: "user", content: "x" }], 0)).toEqual([]);
  });
});

describe("appendTurn", () => {
  it("appends user + assistant messages in order", () => {
    const out = appendTurn([], "hello", "world", 4096);
    expect(out).toEqual([
      { role: "user", content: "hello" },
      { role: "assistant", content: "world" },
    ]);
  });

  it("preserves existing history", () => {
    const prior: Message[] = [{ role: "system", content: "sys" }];
    const out = appendTurn(prior, "hi", "hello", 4096);
    expect(out).toEqual([
      { role: "system", content: "sys" },
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
    ]);
  });

  it("applies truncation when appending pushes over budget", () => {
    const prior: Message[] = [
      { role: "system", content: "sys" },
      { role: "user", content: "old" },
      { role: "assistant", content: "old-r" },
    ];
    const out = appendTurn(prior, "new", "new-r", 3);
    // System retained, and at least one non-system message dropped.
    expect(out[0]?.role).toBe("system");
    expect(out.find((m) => m.content === "new-r")).toBeDefined();
    expect(out.length).toBeLessThan(prior.length + 2);
  });
});
