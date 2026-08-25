# OI Agent

Standalone 24/7 Discord feedback-channel respondent. Watches any number of
Discord channels (and their threads), answers questions by auditing a local
clone of a repository, and posts replies autonomously. There is no
human-approval tier; safety is structural (scoped credentials, channel
allowlist, hourly cap, kill switch), not a review queue.

Fully decoupled from watched repos and from other agent systems (the
manifest-generator idea/code was adapted from an internal orchestrator project;
there is no runtime dependency).

## Safety model

**Credential boundary (the hard guarantee):** git remote creds on the
deployment box are **read-scoped only** (`git pull` works, `git push` 403s).
No DB credentials, no SSH keys, no deploy tokens.

**The watched clone is NOT filesystem-immutable.** OI deliberately performs
these local mutations to keep audits fresh:

- `git pull --ff-only --autostash` into the clone (stashes any local changes,
  pulls, restores them).
- writes `.git/info/exclude` (adds a `.oi/` entry once) — not the tracked
  `.gitignore`.
- writes the manifest to `<clone>/.oi/index.md`.

Everything else is read-only: evidence comes from static reads (`read_file` /
`grep` / manifest), path-jailed to the clone, with dotfiles/`.git`/`.env*`/
key files denied. If you need strict immutability, point OI at a mirror that
only your sync process may write.

**Posting rails:** poster allowlist = configured watch targets + their threads;
per-channel hourly cap with atomic reservation (one reply = one unit); global
kill switch (`oi pause`).

**Egress:** deploy behind an allowlist — git host, Discord API, LLM endpoint.

## Quick start

```bash
uv sync                      # or pip install -e .
export OI_DISCORD_TOKEN=...  # bot token (view+send in target channels)
export OI_AGENT_API_KEY=...
export OI_FALLBACK_API_KEY=...  # optional: only used if the main model fails

oi init      # guided config -> ~/.config/oi/config.toml + first indexes
oi index     # refresh manifests manually
oi run       # daemon loop
oi pause     # kill switch
```

## Config sketch

```toml
discord_token_env = "OI_DISCORD_TOKEN"
db_path = "~/.local/state/oi/state.db"
personality = "A sharp, witty senior engineer. Dry humor, opinionated, direct."
max_reply_chars = 2000
max_response_tokens = 4000   # LLM output budget; raise for reasoning models

[fallback]
base_url = "https://llm.example/v1"   # small model, only after main failure
model = "qwen3-27b"

[agent]
base_url = "https://llm.example/v1"   # main model
model = "qwen3-27b"

[[watch]]
channel_id = 123456789012345678
repo_path = "/srv/repos/your-repo"
founder_ids = [234567890123456789]
bot_user_id = 345678901234567890
post_hourly_cap = 3

# Optional domain vocabulary — keeps the pipeline generic. Keys are lowercase.
search_aliases = { kpi = ["metric", "dashboard"] }
preferred_tokens = ["billing", "invoice"]
search_stopwords = ["scratch"]
```

Add as many `[[watch]]` blocks as you like — one gateway connection serves
them all; each keeps its own repo binding, founders, cap, vocabulary and
state. `search_aliases` expands question terms before grepping the repo,
`preferred_tokens` ranks which evidence files get quoted, and
`search_stopwords` filters deployment-specific noise words out of search.

## Message pipeline

```
gateway event -> deterministic gate -> main responder -> poster
                         |                 |
                      IGNORE        fallback model on failure
```

Every posted answer is stamped `audited at <sha>` with the manifest SHA it
was built from.

## Development

```bash
uv sync --dev
uv run pytest
```
