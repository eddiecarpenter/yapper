/**
 * Conversation message — mirrors the Go-side `internal/llm/types.go::Message`
 * struct and the wire format consumed by the relay (`docs/ARCHITECTURE.md` §5.1
 * and §5.2). The browser holds an array of these as conversation history and
 * sends them with every turn so the relay can pass full context to the LLM.
 *
 * The `role` discriminant matches OpenAI / Anthropic conventions:
 *   - `"system"` — optional priming prompt at index 0
 *   - `"user"`   — user utterance (transcribed audio)
 *   - `"assistant"` — model reply
 */
export type MessageRole = "system" | "user" | "assistant";

export interface Message {
  readonly role: MessageRole;
  readonly content: string;
}
