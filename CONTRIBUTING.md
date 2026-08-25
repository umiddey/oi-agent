# Contributing

Work however you like — we gate on outcomes, not process. A change is good when
the tests pass, the safety invariants hold, and it's been reviewed.

## Getting started

```bash
uv sync --dev
uv run pytest
```

After a code change, reinstall so `oi run` picks it up:

```bash
uv pip install -e . --force-reinstall --no-deps
```

## Before you open a PR

- `uv run pytest` is green.
- New behavior has a test.
- The safety invariants below still hold.
- The PR description says what changed and why (a couple of lines is fine).

## Safety invariants (non-negotiable)

This bot answers autonomously with no approval tier, so these are the guardrails:

- **Read-only creds.** Git remote is read-scoped; never add write/push paths.
- **Jailed repo access.** All repo reads go through `agent/tools.py`, path-jailed
  to the clone, denying `.git`/dotfiles/`.env*`/key files. Don't bypass it.
- **No evidence leakage.** Never log file contents, grep matches, or model
  reasoning. Metadata (char counts) only.
- **Posting rails.** Channel allowlist + per-channel hourly cap + kill switch
  (`oi pause`) must stay intact.

## Config

Use `oi config set <key> <value>` rather than hand-editing the TOML — it
validates and writes atomically.

## Where things live

- `agent/llm.py` — LLM client + agentic tool loop
- `agent/tools.py` — jailed read-only repo tools
- `agent/responder.py` — response pipeline
- `config.py` — config model + validation

> Note: `AGENTS.md` covers extra conventions for AI coding agents (e.g. the
> plan/memory files under `docs/`). Human contributors don't need to follow
> those.
