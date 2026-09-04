# OI Agent

Standalone Discord repository auditor. OI watches configured channels and their
threads, admits explicit requests, runs a native OpenCode `oi` agent against a
local clone, and posts through Discord safety rails. There is no approval tier;
safety is structural.

## Runtime boundary

```text
gateway -> admission/debounce/single-flight -> OpenCode runner -> Poster
 OI                  OI                       harness       OI
```

OI owns Discord transport, allowlists, burst coalescing, concurrency, hourly
caps, pause/resume, delivery, process isolation, and audit provenance. OpenCode
owns provider connectors, authentication, model execution, repository
exploration, context management, tools, and session compaction.

OpenCode is required. OI has no custom provider HTTP client, fallback engine,
tool loop, repository manifest, or skill integration.

## Safety model

- Only the final completed OpenCode assistant message is deliverable.
  Intermediate narration, reasoning, and partial assistant steps never reach
  the outbox or Discord, and reply truncation preserves the provenance footer.
- Git remotes remain read-scoped. OI may run
  `git pull --ff-only --autostash` to refresh a watched clone; OpenCode cannot
  edit source files or execute shell commands.
- Each run supplies an inline native `agent.oi` configuration with explicit
  read-only repository permissions. `read`, `glob`, `grep`, `list`, and `lsp`
  are allowed only inside the watched root. `edit`, `bash`, web access, MCP,
  skills, tasks, questions, and external directories are denied.
- `.git`, `.env*`, private keys, certificates, credentials, and known secret
  files are denied even inside the repository.
- Watched-repository `opencode.json` and `.opencode/` configuration are disabled
  with `OPENCODE_DISABLE_PROJECT_CONFIG=1`.
- OI uses isolated OpenCode HOME/XDG roots. It never imports the operator's
  personal OpenCode config, plugins, MCP servers, agents, or credentials.
- Poster delivery is restricted to configured channels and their threads, has
  an atomic per-channel hourly cap, and honors the global `oi pause` switch.
- Logs contain bounded metadata only: no prompts, repository evidence, replies,
  reasoning, credentials, or raw OpenCode events.

## Merge request reviews

An explicit `MR 188` / `!188` (GitLab) or `PR 188` / `pull request 188`
(GitHub) reference, or a canonical merge-request URL, in the latest request
triggers a read-only merge-request audit instead of the watched-checkout
audit. Bare numbers (`188`) and `#188`-style references are deliberately
treated as ordinary text: only the explicit forms above trigger a review.

- References resolve only against configured watched repositories. A bare
  number must match exactly one of them; ambiguous, unavailable, unsupported,
  or malformed references fail with an actionable message and never fall back
  to auditing the current branch.
- Controller-side MR fetching is read-only (`ls-remote` and `fetch` on
  provider-owned refs, then bounded `ls-tree`, `cat-file`, and `diff`) and
  limited to the configured repository's `origin`. It verifies the base/head
  relationship and materializes temporary read-only base/head snapshots.
- The model receives only those snapshot directories through the existing
  read-only file tools. Permissions remain read-only file inspection with no
  Git, shell, web, MCP, or credential access; `.git` and secret files stay
  denied.
- Review provenance carries the exact full base and head SHAs, e.g.
  `MR !188 base <sha> head <sha>`, and survives reply truncation. Review runs
  never resume or store conversation sessions.
- Temporary snapshots and object repositories are removed after every run —
  success, failure, timeout, or cancellation — and are never persisted.

## Quick start

```bash
uv sync --dev
opencode --version
opencode auth login

oi init --model provider/model
# Interactive init asks for Discord token, tone, and watched repositories.
oi run
oi pause
oi resume
```

If OpenCode is missing, interactive `oi init` asks for installation consent.
Non-interactive setup must use `oi init --install-opencode --model provider/model`
or install OpenCode first. Installation uses a fixed available package-manager
command and verifies `opencode --version` afterward. Declining installation
aborts setup; it never creates a degraded configuration.

OpenCode credentials stay in its isolated credential store. OI's `.env` stores
only the configured Discord token environment variable and is mode `0600`.

## Config sketch

```toml
discord_token_env = "OI_DISCORD_TOKEN"
db_path = "~/.local/state/oi/state.db"
personality = "A sharp, witty senior engineer. Dry humor, opinionated, direct."
max_reply_chars = 2000
reply_delivery = "chunked"
opencode_binary = "opencode"
opencode_model = "provider/model"
opencode_steps = 30
opencode_timeout_seconds = 900
environment_mode = "workstation"

# 1. Channel-specific watch auditing multiple repositories:
[[watch]]
channel_id = 123456789012345678
repos = [
  "/srv/repos/backend-erp",
  "/srv/repos/contractor-portal"
]
bot_user_id = 345678901234567890
post_hourly_cap = 5

# 2. Server-wide (guild) watch with noise channel exclusions:
[[watch]]
guild_id = 987654321098765432
ignored_channels = [111111111111111111, 222222222222222222]
repos = ["/srv/repos/backend-erp"]
bot_user_id = 345678901234567890
post_hourly_cap = 3
```

Use `oi config set <key> <value>`, `oi config watch-add --channel-id ... --repos ...`,
`oi config watch-add --guild-id ... --repos ... --ignored-channels ...`, and
`oi config watch-remove <channel_or_guild_id>`; writes are validated before an atomic replacement.
Legacy provider sections, API keys, base URLs, fallback settings, tool-loop limits, and search
vocabulary fail with a migration error. `oi config show` never prints credentials.

### Environment mode

`environment_mode` selects the execution perspective and accepts exactly
`workstation` or `server`:

- `workstation` (default): OI runs interactively from a developer checkout.
- `server`: OI runs as the hardened non-root daemon service. `oi init
  --install-daemon` defaults to `server`; interactive `oi init` without daemon
  installation defaults to `workstation`. Either value can be chosen explicitly
  with `oi init --environment-mode workstation|server`.

Configs written by older releases may still contain `environment_mode = "auto"`.
Loading such a config fails with a migration error naming the exact repair
command, and the repair itself is tolerated on the legacy file:

```bash
oi config set environment_mode server   # or: workstation
oi config show
```

`oi config set environment_mode` validates against `workstation | server` and
writes atomically; any other value is rejected.

Messages must mention the bot, start with `oi:`, or contain `#oi`, `#audit`, or
`#bugreport`. Ordinary chatter stays silent. A burst is coalesced into one
latest-context request, and a conversation resumes its stored OpenCode session.
Threads have separate sessions from their parent and siblings. A session is
stored only after a successful OpenCode response and successful Discord post.
## Durable delivery & queue management

OI provides at-least-once message processing with crash and restart recovery:

- **Durable job admission**: Admitted mention jobs are persisted to SQLite with
  per-conversation FIFO leasing and atomic state transitions (`pending`, `leased`,
  `done`, `dead`).
- **Startup & reconnect reconciliation**: Scope cursors (`discord_scopes`) track the
  highest seen message ID per channel and thread. Upon daemon startup or gateway
  reconnect, OI queries backlog history to backfill and process missed mentions.
- **Outbox deduplication & privacy**: Generated response chunks are recorded in
  `outbound_chunks` with deterministic nonces and SHA-256 digests. Once Discord
  confirms delivery, the batch is marked `done` and the chunk payload text is
  immediately scrubbed (`body = NULL`). Source message bodies, thread evidence,
  and prompts are never stored.
- **Dead-letter visibility**: Permanently failing jobs or jobs exceeding retry
  limits are moved to `dead` status.

```bash
oi queue status
oi queue retry --all
oi queue retry --id 123456789012345678
oi queue purge --older-than-days 7
```

## Team dynamics and long-term memory (V2)

OI maintains bounded, structured behavioral memory to adapt communication to individual
teammates and team workflow across daemon restarts and OpenCode sessions:

- **Privacy first**: OI never stores raw messages, chat transcripts, prompts, replies,
  reasoning, or excerpts. Stored data consists exclusively of versioned, bounded,
  validated structured signals.
- **Scope isolation**: Every member profile is keyed by `(platform, scope_id, member_id)`.
  Memory is never shared across different servers or scopes, and cross-team identity linking
  is strictly prohibited.
- **Retained fields**:
  - **Member profile**: `directness` (bounded score), `detail_preference` (`brief | balanced | detailed`),
    `challenge_preference` (`gentle | direct | adversarial`), `humor_preference` (`low | medium | high`),
    `decision_style` (`options | recommendation | execution_first`), confidence, evidence count,
    sanitized recurring topics, and timestamps.
  - **Transient member state**: `energy` band (`low | steady | high | frustrated | unknown`),
    current focus labels, `observed_at`, and automatic `expires_at` (hours/days).
  - **Team pulse**: Focus labels, friction categories, momentum (`blocked | planning | building | debugging | shipping | celebrating | unknown`),
    confidence, `observed_at`, and automatic `expires_at`.
  - **Agent calibration**: Preferred verbosity band, directness band, formatting avoidances,
    confidence, `updated_at`, and automatic `expires_at`.
- **Prohibited inference categories**: Protected demographic traits, psychological or medical
  diagnoses, employee evaluations, performance scoring, and cross-scope identity tracking
  are strictly forbidden.
- **Best-effort reflection**: Reflection runs asynchronously on a bounded background queue
  only after confirmed delivery. Posting is never blocked. In a crash, reflection observations
  may safely be lost; raw inputs are never retained for retries.
- **Operator controls**:

```bash
oi memory status --scope 123456789012345678
oi memory show --scope 123456789012345678 --member 234567890123456789
oi memory reset-member --scope 123456789012345678 --member 234567890123456789
oi memory reset-scope --scope 123456789012345678
oi memory prune
oi memory migrate-legacy --scope 123456789012345678
oi memory purge-legacy
```

## Daemon installation

```bash
sudo oi daemon install
sudo oi daemon status
sudo oi daemon restart
sudo oi daemon logs
sudo -u oi -H oi doctor --config /home/oi/.config/oi/config.toml --user oi
sudo oi daemon install --dry-run
```

Installation provisions a dedicated unprivileged `oi` user, copies only OI's
config and Discord-token file, creates private OpenCode XDG roots, and renders a
hardened systemd unit. OpenCode authentication must be performed for the
service user; root's HOME and credential store are never used.

The unit uses `ProtectSystem=strict`, `ProtectHome=read-only`,
`NoNewPrivileges=true`, `PrivateTmp=true`, kernel/control-group restrictions,
and narrow `ReadWritePaths` for OI state, isolated OpenCode state, and watched
clones required by the existing read-scoped pull.

## Operator cleanup after upgrade

Older releases may have left `<repo>/.oi/index.md` and a `.oi/` entry in
`.git/info/exclude`. They are obsolete. Inspect and remove them explicitly
when convenient; ordinary daemon startup never deletes watched-repository files.

## Development

```bash
uv sync --dev
uv run pytest
uv pip install -e . --force-reinstall --no-deps
```
