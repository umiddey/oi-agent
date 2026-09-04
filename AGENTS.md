# AGENTS.md

Instructions for AI agents working on this repo. Humans: see CONTRIBUTING.md.

## What this is

A standalone Discord bot that read-only audits a local repo clone and answers
questions autonomously. Safety is structural, not a review queue.

## Commands

```bash
uv sync --dev
uv run pytest
```

**After ANY code change, reinstall or `oi run` runs the OLD build:**

```bash
uv pip install -e . --force-reinstall --no-deps
```

## Safety invariants — never break these

- OpenCode is the only model/tool/session harness. Do not add provider HTTP,
  fallback engines, custom tool loops, manifests, or skills.
- Git remote creds are read-scoped only. Never add code that writes/pushes.
- OpenCode runs with `OPENCODE_DISABLE_PROJECT_CONFIG=1`, isolated HOME/XDG
  roots, an inline native `agent.oi`, and no `--auto`.
- Native permissions allow only repository-scoped `read`, `glob`, `grep`, `list`,
  and `lsp`; edit/bash/web/MCP/skill/task/question/external access is denied.
- Never log prompts, repository evidence, replies, model reasoning, credentials,
  or raw JSONL events. Bounded metadata only.
- Posting is gated by the channel allowlist + per-channel hourly cap + kill
  switch (`oi pause`). Poster is the only Discord sender.
- SQLite stores only pause/cap state, durable queue/outbox metadata, scope
  cursors, conversation/repository OpenCode session IDs, and bounded, structured,
  versioned behavioral memory signals (scoped member profiles, expiring transient
  state, team pulse, and agent calibration). Never store prompts, repository evidence,
  reasoning, source message bodies, thread excerpts, or free-form generated text.
- Bounded outbound reply chunks are stored in the SQLite outbox ONLY until
  Discord delivery confirmation, after which body text is scrubbed (`body = NULL`).

## Confidentiality

- Never include personal details, real names, internal URLs, deployment
  specifics, credentials, or the substance of private discussions.
- Never paste repository contents, config values, or evidence into external
  services. Assume anything sent out is cached and permanent.

## Team memory and reflection contract

- **Bounded structured signals only**: SQLite may store versioned, bounded,
  structured behavioral signals. Every profile is strictly scoped by
  `(platform, scope_id, member_id)` and team pulse/calibration by
  `(platform, scope_id)`. One team can never access another team's memory.
- **Strictly forbidden storage**: Raw source messages, excerpts, agent replies,
  prompts, reasoning, tool traces, or free-form prose critiques are NEVER stored.
- **Prohibited inference categories**: Protected traits, medical or psychological
  diagnoses, performance evaluations, employee scoring, and cross-scope identity
  linking are strictly prohibited.
- **Transient expiry & operator controls**: Transient state (energy, pulse, calibration)
  must expire automatically. Operators have local inspection, reset, prune, and deletion
  controls (`oi memory`).
- **Best-effort reflection**: Reflection runs asynchronously after confirmed delivery
  without delaying posting. Raw inputs are never persisted for retries; crashes may
  safely lose an observation.

## Where things live

- `opencode/bootstrap.py` — dependency, auth, and model bootstrap.
- `opencode/policy.py` — native fail-closed permission map.
- `opencode/prompt.py` — embedded OI safety/evidence/response contract.
- `opencode/runner.py` — bounded async OpenCode subprocess runner.
- `watch/discord_client.py` — admission, debounce, sessions, and Poster seam.
- `store.py` — pause, cap, and OpenCode session state.
- `daemon.py` — dedicated user and hardened systemd lifecycle.
- `config.py` — strict OpenCode-only TOML validation and atomic writes.

## Durable delivery & queue contract

- **At-least-once delivery**: Mention jobs are admitted into SQLite before OpenCode
  execution and leased atomically. If the daemon restarts mid-execution or mid-post,
  unconfirmed work is recovered and retried.
- **Startup reconciliation**: On connection/reconnect, OI queries channel and thread
  history beyond stored `discord_scopes` cursors to backfill any missed mentions.
- **Outbox lifecycle & privacy**: Generated response chunks are recorded in
  `outbound_chunks` with unique nonces and SHA-256 digests. Once Discord delivery is
  confirmed, the batch is marked done and chunk bodies are scrubbed (`body = NULL`).
- **Queue operations**: Operators can inspect the queue with `oi queue status`,
  retry dead-lettered jobs with `oi queue retry`, and prune completed records with
  `oi queue purge`.
## Config

Never hand-edit `~/.config/oi/config.toml`. Use `oi config set <key> <value>`
(validated, atomic). Legacy provider/base-URL/API-key/tool-loop/search keys are
rejected; provider credentials belong to OpenCode's isolated auth store.

## Plan + memory

For non-tiny work, before coding write a plan in `docs/plans/immediate/` and a
1:1 mapped memory file in `docs/memory/`; update phase status as you go.
