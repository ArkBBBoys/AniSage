# AniSage — Self-Learning Anime / Manga / Manhwa Discord Bot

> A personal Discord bot that continuously scrapes the web for anime, manga and
> manhwa news, **learns** from everything it finds, and delivers matched-based
> search + DM alerts directly to you.

---

## What it is

AniSage is a personal, always-on research bot. It does **not** guess — it
**scrapes, stores, learns, and improves**. Every run makes it better at:

- Finding news across many sites (RSS + direct HTML scraping + AniList/Jikan APIs).
- Resolving a title you type (e.g. `JJK`) to the correct canonical work
  (`Jujutsu Kaisen`) using **matched-based**, not keyword, search.
- Remembering which free streaming / reading hosts exist, what their **real,
  current domains** are (these change *constantly*), and exactly what search
  parameter each one expects (e.g. `?q=` vs `?keyword=`).
- Alerting you in DM when news breaks for titles you follow.

## Core features

| Feature | Description |
|---|---|
| 🌐 Continuous crawler | Three background loops (`learn`, `watch`, `explore`) keep the bot scraping 24/7 while it is online. |
| 🧠 Self-learning knowledge base | A persistent SQLite DB (`anisage.db`) stores items, titles, aliases, feedback, follows and host resources. It only grows smarter. |
| 🎯 Matched-based search | Token-set fuzzy matching (`rapidfuzz`) + a learned alias table. `JJK` → `Jujutsu Kaisen`. Not substring/keyword search. |
| 🆓 Free-host index (EverythingMoe) | Learns the currently-live free streaming/reading sites and **self-heals** when a host dies or moves. |
| 🔗 Direct host crawling | Resolves each indexed site's real domain and learns its exact search endpoint, so deep-links use the *site-specific* query param. |
| 📬 DM delivery | News, digests, search results, and a full "what I've learned" export are sent straight to your DMs. |
| 🔔 Follow & alerts | Follow a title; get DM pinged when related news appears. Optional server news channel too. |
| 📊 Proficiency metric | A live `0–100` score that rises as the bot accumulates experience and verified human feedback. |

## How the "self-learning" actually works

1. **Scrape widely.** The `explore_loop` (every ~2 min) and `learn_loop`
   (every `LEARN_INTERVAL`) pull RSS feeds, scrape HTML release pages, query
   AniList/Jikan, and re-sync the EverythingMoe index.
2. **Store everything.** Every item, title, alias, host, and feedback is
   persisted — nothing is thrown away between runs.
3. **Distill.** Titles are normalized into canonical keys. Confidence grows each
   time a title is seen (`times_seen` + `confidence`).
4. **Feedback reinforces.** When you react ✅/❌ to a search result, the bot
   records it: a correct match boosts that title's confidence and saves the query
   as a permanent alias; a wrong match lowers it.
5. **Self-heal.** If a free host vanishes from EverythingMoe (its URL moved) the
   bot marks it `dead` and tries to find its replacement by name similarity. It
   also re-resolves each host's real domain and re-learns its search param on a
   rotating basis.
6. **Improve.** `proficiency` blends average confidence with verified-match
   accuracy, so the number you see in `/stats` literally goes up the more the bot
   is used and corrected.

## Scope / intent note

AniSage is an **information aggregator and linker**. It scrapes public indexes
(RSS, EverythingMoe, AniList, Jikan) and surfaces **links** — including legal
streaming/reading sources (Crunchyroll, MangaPlus, Webtoon, etc.) and the
currently-live free hosts indexed by EverythingMoe. It does **not** download,
host, or redistribute any copyrighted video/image/file content. The deep-links it
builds point to a host's *search page* for a title; the bot never extracts
episode/stream/video URLs or media files.

## Project layout

```
anisage/
├── main.py          # Bot entry, slash commands, background loops, DM delivery
├── config.py        # Env vars, sources, tunables
├── database.py      # Self-learning SQLite store (items, titles, aliases, feedback, follows, resources)
├── fetchers.py      # RSS, AniList, Jikan, EverythingMoe index, direct-host discovery
├── matcher.py       # normalize / tokenize / matched-based fuzzy search
├── embeds.py        # Pretty Discord embeds
├── anisage.db       # Auto-created persistent knowledge base (gitignored)
├── pyproject.toml   # uv-managed project
└── .env             # Your secrets (gitignored) — see setup.md
```

## Quick start

```bash
cp .env.example .env      # fill in DISCORD_TOKEN and OWNER_ID
uv run main.py
```

Then in Discord: `/help`.

Full setup → **setup.md** · All commands → **commands.md**
