# Setup Guide — AniSage

Step-by-step instructions to get the bot running with [`uv`](https://docs.astral.sh/uv/).

---

## 1. Prerequisites

- **Python 3.14+** (the project pins `requires-python = ">=3.14"`).
- **uv** installed. Get it from <https://docs.astral.sh/uv/getting-started/installation/>
  or:
  ```bash
  # macOS / Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Windows (PowerShell)
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- A **Discord account** and permission to add a bot to a server (or just use it in
  your DMs — it is a *personal* bot).

---

## 2. Create the Discord bot application

1. Go to <https://discord.com/developers/applications> → **New Application**.
2. Name it (e.g. `AniSage`), then open the **Bot** tab → **Add Bot**.
3. Under **Token** → **Reset Token** and copy it. This is your `DISCORD_TOKEN`.
4. **Privileged Gateway Intents**: the default intents are enough (the bot only
   needs to read DM interactions and send messages). No message-content intent
   required.
5. **OAuth2 → URL Generator** → tick `bot`, and the `applications.commands` scope.
   Under **Bot Permissions** select at least `Send Messages` + `Attach Files` (for `/learned-db`).
6. Copy the generated URL, open it, and invite the bot to your server (or just to
   your own account for DM-only use).

> The bot uses **slash (application) commands**, which Discord syncs on startup.

---

## 3. Configure environment

Copy the example env file and fill it in:

```bash
cp .env.example .env
```

`.env` fields:

| Variable | Required | Description |
|---|---|---|
| `DISCORD_TOKEN` | ✅ | Bot token from step 2. |
| `OWNER_ID` | ✅ | Your Discord user ID. Right-click your name → *Copy User ID* (enable Developer Mode in Settings → Advanced). DM delivery, `/learn`, `/input-learned*` and admin commands are restricted to this ID. |
| `NEWS_CHANNEL_ID` | ⬜ | A server text-channel ID. Followed-title news is also posted here (in addition to your DMs). Leave `0` to disable. |
| `LEARN_INTERVAL` | ⬜ | Seconds between full learn cycles. Default `900` (15 min). |
| `WATCH_INTERVAL` | ⬜ | Seconds between follow-alert checks. Default `1800` (30 min). |
| `MATCH_THRESHOLD` | ⬜ | Min match confidence (0–100) to auto-accept a title. Default `78` — **auto-tuned hourly** via 7-day feedback (58-84). |
| `EMOE_REFRESH` | ⬜ | Seconds between EverythingMoe index re-syncs. Default `3600` (1 h). |

Example `.env`:

```env
DISCORD_TOKEN=MTk4N...your.token...here
OWNER_ID=123456789012345678
NEWS_CHANNEL_ID=0
LEARN_INTERVAL=900
WATCH_INTERVAL=1800
MATCH_THRESHOLD=78
EMOE_REFRESH=3600
```

`.env` is gitignored — never commit it.

---

## 4. Install dependencies (uv)

From the project root:

```bash
uv sync            # creates/uses .venv and installs pyproject dependencies
```

Or, if you add a new package later:

```bash
uv add <package>   # e.g. uv add aiohttp
uv remove <package>
```

Dependencies currently managed by `uv`:
`discord-py`, `aiohttp`, `beautifulsoup4`, `feedparser`, `rapidfuzz`, `python-dotenv`, `ddgs` (DuckDuckGo).

---

## 5. Run the bot

```bash
uv run main.py
```

On first launch you should see:

```
AniSage online as <botname> (id <botid>)
[boot] sources seeded: 14 news sources
[emoe] learned N free hosts from EverythingMoe
[emoe] index refreshed: ... alive hosts
[learn] ... proficiency 49.2%
```

The bot will:

- Sync slash commands (may take a few seconds; global commands can take up to an hour to propagate on Discord — see Troubleshooting).
- Run an initial learn cycle (exhaustive news + trending + EverythingMoe index) — now **whole-web** via `fetch_news_exhaustive` (RSS ranked by reliability + 4 web queries).
- Start the **five** background loops.

For continuous operation use a process manager / keep the terminal open. Example
with `nohup` / a systemd service / a `screen` session.

---

## 6. First-run checklist

- [ ] `.env` filled with a valid `DISCORD_TOKEN` and your `OWNER_ID`.
- [ ] Bot invited to your server (or you're DMing it directly).
- [ ] `uv run main.py` prints `AniSage online as ...`.
- [ ] Run `/learn` (owner only) to force an immediate learning cycle.
- [ ] Try `/search JJK` → you should get a DM/ephemeral with the resolved title + 4 related news via whole-web crawl + ✅/❌.
- [ ] Try `/search Spirit Chronicales` (typo) → should resolve via `jaro/ngram` hybrid to `Seirei Gensouki` (not hard-coded).
- [ ] Try `/search BTTH` (acronym) → `Battle Through the Heavens` via initials.
- [ ] Run `/stats` → proficiency 7-factor breakdown, `/learning` → hybrid weights & threshold.
- [ ] Run `/news` → `15 stories: X from whole web + Y cached` via exhaustive.

---

## 7. The database

`anisage.db` is **auto-created** on first import (WAL mode, `check_same_thread=False` + `RLock`, 30s timeout). It stores:

| Table | Purpose | Self-learning |
|---|---|---|
| `items` | Every scraped news item (source, title, url, summary, media type, timestamps, fetched_at). | `bulk_add_items` ON CONFLICT DO NOTHING, web news persisted for future `/news` |
| `titles` | Learned canonical titles + `confidence` (Bayesian `Beta` + `times_seen` + recency + alias + source) + `times_seen` + `feedback_correct/wrong` + `ngram_sig`. | `learn_title` Bayesian, `decay_stale_titles` |
| `aliases` | Learned alias → title (`weight`, `hits_correct/wrong`, `last_used`, `created_at`). | `_set_alias` blended `1.0 +0.55*c-0.68*w`, `prune_stale_aliases` |
| `feedback` | Human ✅/❌ (`query`, `matched_key`, `correct`, `match_method`, `score`, `ts`). | `record_feedback` Bayesian, `auto_tune_threshold` |
| `news_sources` | **Self-learning RSS** (`url` PK, `name/kind/status/reliability 0-1/score/avg_items/latency/last_checked/discovered_via`). | `sync_news_sources_from_config`, `discover_rss_sources_via_web` (DDG `anime news RSS` → `<rss>` test), `update_news_source_stats` EMA, `prune_dead_news_sources` |
| `title_search_sources` | **Self-learning title APIs** (`anilist/kitsu/jikan/ddg/mal` → `reliability/weight/latency`). | `update_title_source_stats` → `fetch_exhaustive_search` weighted `* (0.78+0.44*weight)` |
| `learning_metrics` | 7-day windows (`total/correct/wrong`, `avg_confidence`, `threshold`, `weights_json`). | `log_learning_window` hourly |
| `scrape_log` | Per-source scrape success/duration. | `log_scrape` |
| `follows` | `user_id, title_key, guild_id` unique (per-server isolated). | `follow`/`unfollow` |
| `broadcasts` | `user_id, guild_id` unique, `channel_id`, `last_sent`. | `set_broadcast` |
| `resources` | Free hosts: `slug/name/kind/page_url/domain/search_url/search_param/status`. | `upsert`, `resolve_host_domain`, `learn_host_search`, self-heal |

To **backup**: `/learned` → `anisage_learned.json` (titles+items) or `/learned-db` → `anisage_<guild>.db` (raw SQLite, WAL checkpoint).

To **restore** (merge, skip existing): `/input-learned file: anisage_learned.json` or `/input-learned-db file: anisage_123.db` — both `owner_only`, `INSERT OR IGNORE` style, report `Imported X new / Skipped Y already existed`.

To wipe learning: stop the bot and delete `anisage.db` (it will be recreated empty, then seeded with 14 RSS + 5 title sources, then self-discovers).

---

## 8. Troubleshooting

**Commands don't appear in Discord.**
Global slash commands can take up to ~1 hour to register. For instant testing,
invite the bot to a server and wait, or restart the bot (it calls `tree.sync()`
on every `on_ready`).

**`SystemExit: Set DISCORD_TOKEN in .env`.**
The token is missing or `.env` is not in the project root. Verify the file and
variable name.

**Network / scraping errors in the log.**
These are non-fatal — the bot logs and continues. Sources that are temporarily
down are auto-penalized (`reliability` ↓, `weight` ↓, `consecutive_fails` ↑, `paused` after 3, `dead` after 7) and retried. `TitleSearchSource` with `anilist 0.05` when down → `kitsu 0.97` preferred (seen in logs).

**`charmap` / encoding errors when printing to the Windows console.**
This is only a console display issue (some sites return non-ASCII). The bot itself works fine; avoid `print`-ing raw titles to a legacy Windows terminal, or run in a UTF-8 capable terminal (`chcp 65001`).

**High CPU / network usage.**
Lower the frequency via `LEARN_INTERVAL` / `WATCH_INTERVAL` / `EMOE_REFRESH`, or the bot will auto-prune to ≤40 active news sources and ≤12 follows per `watch_loop` cycle.

**Backup restore says `Skipped X already existed` for everything.**
That's correct — it merges, never overwrites. If you want a clean restore, delete `anisage.db` first, then `/input-learned`.

---

## 9. Updating

```bash
uv sync            # pull dependency changes
uv run main.py
```

No migrations needed for the DB — `KnowledgeDB._migrate_*` adds any missing
columns/tables (`image`, `feedback_correct`, `news_sources`, etc.) automatically on startup, preserving data.
