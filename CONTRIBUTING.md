# Contributing

A change is good when the tests pass, the safety invariants hold, and it has
been reviewed.

## Getting started

```bash
uv sync --dev
uv run pytest
```

After a code change, reinstall so the installed `oi` command uses current code:

```bash
uv pip install -e . --force-reinstall --no-deps
```

## Before opening a PR

- `uv run pytest` is green.
- New behavior has a behavioral test.
- OpenCode-only safety invariants still hold.
- The PR description says what changed and why.

## Safety invariants

- **Read-only credentials.** Git remotes are read-scoped; never add write/push
  paths.
- **Final-only output.** Only the final completed OpenCode assistant message is
  deliverable. Intermediate narration, reasoning, and partial assistant steps
  never reach the outbox or Discord, and reply truncation preserves the
  provenance footer.
- **Read-only merge-request audits.** MR/PR references resolve only against
  configured watched repositories. Controller-side MR fetching is read-only and
  limited to configured repositories; temporary base/head snapshots are never
  persisted. Model permissions remain read-only file inspection with no Git,
  network, or shell access. Review provenance carries exact base/head SHAs,
  and unsupported, ambiguous, or unresolved references fail honestly instead of
  auditing the current checkout.
- **Native fail-closed agent.** OpenCode receives an inline `agent.oi` config;
  only repository-scoped `read`, `glob`, `grep`, `list`, and `lsp` are allowed.
  Editing, shell, web, MCP, skills, tasks, questions, external directories, and
  unknown permissions are denied.
- **Project-config isolation.** Runtime sets
  `OPENCODE_DISABLE_PROJECT_CONFIG=1` and isolated HOME/XDG roots. Never import
  personal OpenCode auth, plugins, agents, MCP servers, or config.
- **No evidence leakage.** Never log prompts, file contents, grep matches,
  replies, model reasoning, credentials, or raw OpenCode events.
- **Posting rails.** Channel allowlist, per-channel hourly cap, pause switch,
  burst coalescing, and Poster-only Discord delivery stay intact.
- **Session privacy & bounded memory.** SQLite stores only an OpenCode session
  ID, conversation ID, repository path, update timestamp, and bounded, structured,
  versioned behavioral memory signals scoped strictly by platform/team/member.
  Raw source messages, excerpts, prompts, reasoning, and replies are strictly
  forbidden. Transient states require automatic expiry, and operators maintain
  mandatory inspection and deletion controls (`oi memory`). Prohibited inference
  categories (protected traits, diagnoses, personnel judgments, and cross-scope
  identity linking) are strictly barred. Reflection is asynchronous, failure-isolated,
  and best-effort without storing raw retry inputs.

## Config and runtime

Use `oi config set <key> <value>` instead of hand-editing TOML. The supported
runtime fields are `opencode_binary`, exact `opencode_model` (`provider/model`),
`opencode_steps`, `opencode_timeout_seconds`, `environment_mode`
(`workstation | server`; legacy `auto` values are rejected with a repair hint
and are accepted only as the explicit repair target), Discord settings, reply
limits, personality, and watch targets. Provider API keys and base URLs belong
only to OpenCode's isolated credential store.

## Where things live

- `opencode/bootstrap.py` — dependency, auth, and model bootstrap
- `opencode/policy.py` — fail-closed native-agent permissions
- `opencode/prompt.py` — inline OI evidence/safety/response contract
- `opencode/runner.py` — bounded async JSONL subprocess execution
- `watch/discord_client.py` — Gateway admission, scheduling, and Poster seam
- `store.py` — pause, cap, and session state
- `daemon.py` — systemd unit and service-user lifecycle
- `config.py` — validated OpenCode-only config
