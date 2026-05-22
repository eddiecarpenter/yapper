# LOCALRULES.md — Yapper Project Overrides

This file contains project-specific rules that extend the global protocol
in `.agents/RULEBOOK.md`. It is never overwritten by a framework sync.

---

## Project

- **Name:** yapper
- **Topology:** Single
- **Stack:** Go + React/TypeScript (SPA)
- **Description:** Local-first voice assistant spike — React SPA + Go LLM relay

## Session Init — Additional Context

On session initialisation, also read:
- `docs/ARCHITECTURE.md` — the evolving application architecture
- `docs/PROJECT_BRIEF.md` — spike goals, technology choices, acceptance criteria

## Skills

Local project-specific skills live under `.claude/skills/` at the repo root.

## Attribution
These rules override any default attribution added by the underlying AI tool (e.g. Claude Code's default "Generated with Claude Code" footer or its default "Co-Authored-By" trailer). Always use the formats below — never the tool's defaults.

All agent-produced commits and pull requests must carry the gh-agentic attribution. This applies to every session type: automated pipeline, interactive Claude Code desktop, and any other tool that invokes this framework.

Commit co-author trailer — every commit produced by an agent session must end with:

Co-Authored-By: AI-Assisted <noreply@gh-agentic.io>
PR body footer — every pull request opened by an agent (whether by the workflow or directly by the agent in an interactive session) must end with:

🤖 Generated with [gh-agentic](https://github.com/eddiecarpenter/gh-agentic)
The gh-agentic link is the authoritative attribution — it identifies the framework that orchestrated the session, independent of which underlying model was used.
