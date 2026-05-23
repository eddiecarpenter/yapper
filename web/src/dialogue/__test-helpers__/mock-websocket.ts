/**
 * Manually-driven WebSocket double for hook tests.
 *
 * `vi.stubGlobal("WebSocket", MockWebSocket)` swaps this in for the global
 * constructor; tests drive the lifecycle through `triggerOpen()`,
 * `triggerMessage()`, `triggerError()`, and `triggerClose()`. Every
 * constructed instance is appended to `MockWebSocket.instances` so the
 * test can grab the latest one without exposing a global registry from
 * the production code.
 */
type Listener = (ev: unknown) => void;

export class MockWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;

  static instances: MockWebSocket[] = [];

  public readyState: number = MockWebSocket.CONNECTING;
  public readonly url: string;
  public readonly sent: string[] = [];

  private readonly listeners: Map<string, Listener[]> = new Map();

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  addEventListener(type: string, fn: Listener): void {
    const arr = this.listeners.get(type) ?? [];
    arr.push(fn);
    this.listeners.set(type, arr);
  }

  removeEventListener(type: string, fn: Listener): void {
    const arr = this.listeners.get(type);
    if (!arr) return;
    this.listeners.set(
      type,
      arr.filter((l) => l !== fn),
    );
  }

  send(data: string): void {
    if (this.readyState !== MockWebSocket.OPEN) {
      throw new Error(`MockWebSocket.send called while readyState=${this.readyState}`);
    }
    this.sent.push(data);
  }

  close(): void {
    if (this.readyState === MockWebSocket.CLOSED) return;
    this.readyState = MockWebSocket.CLOSED;
    this.dispatch("close", { code: 1000, reason: "" });
  }

  // ── Test helpers ─────────────────────────────────────────────────────

  triggerOpen(): void {
    this.readyState = MockWebSocket.OPEN;
    this.dispatch("open", {});
  }

  triggerMessage(payload: unknown): void {
    const data = typeof payload === "string" ? payload : JSON.stringify(payload);
    this.dispatch("message", { data });
  }

  triggerError(message = "mock-websocket-error"): void {
    this.dispatch("error", { message });
  }

  /** Inspect the most recently parsed payload sent through `.send()`. */
  lastSent<T = unknown>(): T | undefined {
    const raw = this.sent[this.sent.length - 1];
    if (raw === undefined) return undefined;
    return JSON.parse(raw) as T;
  }

  private dispatch(type: string, ev: unknown): void {
    const arr = this.listeners.get(type);
    if (!arr) return;
    // Clone in case a listener removes itself during dispatch.
    [...arr].forEach((l) => l(ev));
  }

  static reset(): void {
    MockWebSocket.instances = [];
  }
}
