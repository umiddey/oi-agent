# AGENTS.md

Instructions for AI agents working on this repo. Humans: see CONTRIBUTING.md.

## What this is

A standalone Discord bot that read-only audits a local repo clone and answers
questions autonomously — no human-approval tier. Safety is structural, not a
review queue.

## Commands

```bash
uv sync --dev        # install
uv run pytest        # all tests must pass before you finish
```

**After ANY code change, reinstall or `oi run` runs the OLD build:**

```bash
uv pip install -e . --force-reinstall --no-deps
```

## Safety invariants — never break these

- Git remote creds are read-scoped only. Never add code that writes/pushes.
- All repo access goes through `tools.py`, which is path-jailed to the clone and
  denies `.git`/dotfiles/`.env*`/key files. Never bypass `_resolve`.
- Never log repository evidence bodies. File contents / grep matches are logged
  as char-counts only; hidden model reasoning is never logged.
- Posting is gated by the channel allowlist + per-channel hourly cap + kill
  switch (`oi pause`). Don't weaken these.

## Confidentiality

- Never include personal details, real names, internal URLs, deployment
  specifics, credentials, or the substance of private discussions.
- Never describe what this project is *for*, who runs it, or the business/IP
  behind it. Keep commit messages to the technical change only.
- Don't paste repository contents, config values, or evidence into external
  services. Assume anything sent out is cached and permanent.

## Where things live

- `agent/llm.py` — multi-provider LLM client + agentic tool loop.
- `agent/tools.py` — jailed read-only repo tools (the only repo access).
- `agent/responder.py` — response pipeline (seed snapshot → tool loop → post).
- `config.py` — TOML config with strict validation + atomic writes.

## Config

Never hand-edit `~/.config/oi/config.toml`. Use `oi config set <key> <value>`
(validated, atomic). Ranges are enforced (e.g. `max_tool_iterations` ∈ [1, 50]).

## Plan + memory (agent-only convention)

For non-tiny work, before coding write a plan in `docs/plans/immediate/` and a
1:1 mapped memory file in `docs/memory/`; update phase status as you go. This is
agent scaffolding for cross-session continuity — humans are not bound by it.
