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
   Under **Bot Permissions** select at least `Send Messages`.
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
| `OWNER_ID` | ✅ | Your Discord user ID. Right-click your name → *Copy User ID* (enable Developer Mode in Settings → Advanced). DM delivery, `/learn`, and admin commands are restricted to this ID. |
| `NEWS_CHANNEL_ID` | ⬜ | A server text-channel ID. Followed-title news is also posted here (in addition to your DMs). Leave `0` to disable. |
| `LEARN_INTERVAL` | ⬜ | Seconds between full learn cycles. Default `900` (15 min). |
| `WATCH_INTERVAL` | ⬜ | Seconds between follow-alert checks. Default `1800` (30 min). |
| `MATCH_THRESHOLD` | ⬜ | Min match confidence (0–100) to auto-accept a title. Default `78`. |
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
`discord-py`, `aiohttp`, `beautifulsoup4`, `feedparser`, `rapidfuzz`,
`python-dotenv`.

---

## 5. Run the bot

```bash
uv run main.py
```

On first launch you should see:

```
AniSage online as <botname> (id <botid>)
[emoe] learned N free hosts from EverythingMoe
[emoe] index refreshed: ... alive hosts
```

The bot will:

- Sync slash commands (may take a few seconds; global commands can take up to an
  hour to propagate on Discord — see Troubleshooting).
- Run an initial learn cycle (news + trending + EverythingMoe index).
- Start the three background loops.

For continuous operation use a process manager / keep the terminal open. Example
with `nohup` / a systemd service / a `screen` session.

---

## 6. First-run checklist

- [ ] `.env` filled with a valid `DISCORD_TOKEN` and your `OWNER_ID`.
- [ ] Bot invited to your server (or you're DMing it directly).
- [ ] `uv run main.py` prints `AniSage online as ...`.
- [ ] Run `/learn` (owner only) to force an immediate learning cycle.
- [ ] Try `/search JJK` → you should get a DM with the resolved title + links.
- [ ] Run `/stats` → proficiency should be > 0 and climbing as it learns.

---

## 7. The database

`anisage.db` is **auto-created** on first import (the parent directory is created
if missing). It stores:

| Table | Purpose |
|---|---|
| `items` | Every scraped news item (source, title, url, summary, media type, timestamps). |
| `titles` | Learned canonical titles + confidence + times seen. |
| `aliases` | Learned alias → title mappings (e.g. `JJK` → `Jujutsu Kaisen`), with weights. |
| `feedback` | Human ✅/❌ corrections used to boost/lower confidence. |
| `scrape_log` | Per-source scrape success/duration (reliability tracking). |
| `follows` | `user_id → title_key` subscriptions for alerts. |
| `resources` | Free hosts: domain, search URL, learned search param, status (alive/dead), notes. |

To wipe learning: stop the bot and delete `anisage.db` (it will be recreated
empty on next start).

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
down are skipped and retried on the next cycle. The `scrape_log` table records
per-source health.

**`charmap` / encoding errors when printing to the Windows console.**
This is only a console display issue (some sites return non-ASCII characters).
The bot itself works fine; avoid `print`-ing raw titles to a legacy Windows
terminal, or run in a UTF-8 capable terminal (`chcp 65001`).

**High CPU / network usage.**
Lower the frequency via `LEARN_INTERVAL` / `WATCH_INTERVAL` / `EMOE_REFRESH`, or
reduce `RSS_SOURCES` / `SCRAPE_TARGETS` in `config.py`.

---

## 9. Updating

```bash
uv sync            # pull dependency changes
uv run main.py
```

No migrations needed for the DB — `KnowledgeDB._migrate()` adds any missing
columns automatically on startup.
