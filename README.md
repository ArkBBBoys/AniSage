# AniSage — Self-Learning Anime / Manga / Manhwa Discord Bot

> A personal Discord bot that continuously scrapes the **whole web** for anime, manga and manhwa news, **learns** from everything it finds, and delivers exhaustive matched-based search + DM alerts. Not keyword search — hybrid fuzzy + self-learning.

---

## What it is

AniSage is a personal, always-on research bot. It does **not** guess — it **scrapes, stores, learns, and self-improves**. Every run makes it smarter:

- **Exhaustive title search** — fires **AniList + Jikan + Kitsu + DDG web + MAL scrape in parallel** (`fetch_exhaustive_search`), weights candidates by **self-learned source reliability** (`TitleSearchSource` weight 0.4-1.8) and **hybrid scorer** (`token_set 35% + token_sort 22% + jaro-winkler 18% + ngram 15% + partial 10%` auto-tuned hourly). `JJK` → `Jujutsu Kaisen`, `BTTH` → `Battle Through the Heavens` via acronym initials, typos like `Chronicales` handled via `ngram 90 + jaro 98`.
- **Self-learning knowledge base** — persistent SQLite (`anisage.db` WAL) stores items, titles, aliases (with `hits_correct/wrong` + `weight`), feedback, follows/broadcasts per-guild, resources, plus `news_sources`/`title_search_sources`/`learning_metrics`. `learn_title` uses **Bayesian `Beta(2,1)` + frequency + recency + alias quality + source quality**; `record_feedback` updates Bayesian `confidence` monotonically (+1.4 correct, -1.3 wrong) and alias `weight ±0.32/0.42`.
- **Proficiency 0-100** — 7-factor: `mastery 30%` (avg conf) + `accuracy 25%` (smoothed `(good+2)/(total+4)`) + `breadth 15%` (log10 titles) + `depth 10%` (items/title + avg times_seen) + `vitality 10%` (alive hosts + recent items) + `velocity 7%` (recent titles/items/feedback) + `alias_quality 3%` + `momentum`
- **Sources self-learn & self-update** — `NewsSource` (`url/name/kind/status/reliability 0-1/score/avg_items/latency`) and `TitleSearchSource` are **seeded from `config.RSS_SOURCES`**, then **discovered via DDG** (`discover_rss_sources_via_web` searches `anime news RSS` → extracts `.xml/rss/feed` URLs → tests `<rss>` → `discover_new_sources_from_web`), scored by `success_rate*0.60 + (1-fail/12)*0.25 + prod_boost`, pruned hourly (paused after 3 fails, dead after 7, keep ≤40 active). Title sources weighted similarly (`anilist 0.05` when down → `kitsu 0.97` preferred).
- **Whole-web news** — `run_news_cycle` = `fetch_news_exhaustive` (all RSS ranked by `score` + `scrape` + 4 random web queries `anime/manga/manhwa news` + DDG titles), `fetch_news_for_title(title)` parallel `Web:anime news` + `Web:manga news` + DDG titles + RSS filtered, merged de-dup, ranked by `score_pair`. Even unpopular `Hyouge Mono`/`Kaiji` get 3-6 web hits.
- **Per-server & per-person isolated** — `follows (user_id,title_key,guild_id)` + `broadcasts (user_id,guild_id)` unique, `_guild_id` (`0`=DM), `deliver()` verifies DM and falls back to followup so `sent to DM` never lies. `/search` in server stays in server, DM stays in DM.
- **Backup / restore** — `/learned` JSON + `/learned-db` raw SQLite, `/input-learned` + `/input-learned-db` **merge skip-existing** (`INSERT OR IGNORE` style, `if db.get_title(key) exists → skipped++`).

## Core features

| Feature | Description |
|---|---|
| 🌐 Exhaustive crawler | `learn_loop` 15m + `explore_loop` 120s + `watch_loop` 30m + `broadcast_loop` 30m + `self_improvement_loop` 60m keep scraping 24/7 |
| 🧠 Hybrid scorer | `token_set`/`token_sort`/`jaro`/`ngram`/`partial` weighted, learned hourly from 7-day feedback gap |
| 🔍 Exhaustive search | `AniList+Kitsu+Jikan+DDG+MAL` parallel, acronym `BTTH` initials, typo `Chronicales` via ngram, source-weighted `* (0.78+0.44*src_w)` |
| 📰 Whole-web news | `RSS (self-learned rank) + DDG news + DDG titles` parallel per title, not fixed 8 feeds |
| 🔗 Free-host self-heal | `EverythingMoe` index + `resolve_host_domain` + `learn_host_search` (`?q` vs `?keyword`) per-host, `resources` table |
| 📬 Isolated delivery | `send_dm` verified, `deliver()` guild vs DM, `follow`/`broadcast` per `guild_id` |
| 💾 Backup/restore | `/learned` JSON + `/learned-db` DB → `/input-learned`/`/input-learned-db` merge skip-existing |
| 📊 Proficiency | 7-factor 0-100, `avg_conf*0.3+accuracy*0.25+breadth*0.15+...` + momentum |
| 🔔 Follow & alerts | `watch_loop` live `fetch_news_for_title` per follow (parallel 12) + `NEWS_CHANNEL_ID` |
| 🧬 Self-improving | Hourly: prune aliases, decay stale titles, recompute alias weights, auto-tune `MATCH_THRESHOLD`, tune hybrid weights, discover/prune sources, log windows |

## How the “self-learning” actually works

1. **Scrape widely — whole web.** `explore_loop` (120s) and `learn_loop` (15m) pull `get_active_news_sources(limit 10)` ordered by `score`, `fetch_web_news` (DDG), `fetch_news_exhaustive` (10 web queries), `fetch_ddg_titles`, `fetch_mal_scrape`, plus `AniList/Jikan/Kitsu` (weighted). No longer limited to 8 hard-coded RSS.
2. **Store everything.** Every item, title, alias (`weight`, `hits_correct/wrong`, `last_used`), feedback (`match_method`, `score`), `news_sources`/`title_search_sources` reliability.
3. **Distill with Bayesian.** `confidence = base + posterior*48 + log1p(times_seen)*4.5 + recency (±3.5) + alias_bonus + source_bonus` where `posterior=(correct+alpha)/(total+alpha+beta)` with **adaptive prior** `alpha=1.8+acc*1.8` from global feedback. `times_seen` and `feedback_correct` both matter.
4. **Feedback reinforces.** `✅` → `feedback_correct++`, `confidence +1.6+(75-conf)*0.035`, alias `weight +0.32`; `❌` → `feedback_wrong++`, `weight -0.42`, also `title confidence -1.3`. New alias created with `weight 3.0`.
5. **Self-heal & self-update sources.** `NewsSource` `reliability` EMA + `score` ranking; `TitleSearchSource` weight 0.4-1.8 from success/latency. Hourly `discover_rss_sources_via_web` (DDG `anime news RSS` → parse `<rss>` → test) adds new, `prune_dead_news_sources` keeps ≤40 active. `Resource` `domain` re-resolved, `search_param` re-learned.
6. **Self-improve hourly.** `self_improvement_loop` (3600s): `prune_stale_aliases(90d,0.32)`, `decay_stale_titles(60d,0.12)`, `recompute_alias_weights_from_feedback` (Bayesian `1+0.52*c-0.68*w`), `auto_tune_threshold` (7-day `0.62*avg_c+0.38*avg_w` → 22% toward ideal), hybrid weight tuner (gap<18 → boost `jaro/ngram`).

## Scope / intent note

Information aggregator and linker. Scrapes public indexes (RSS, EverythingMoe, AniList, Jikan, Kitsu, DDG, MAL) and surfaces **links** — legal (Crunchyroll, MangaPlus) + currently-live free hosts. Does **not** download/host copyrighted media. Deep-links point to host's *search page*.

## Project layout

```
anisage/
├── main.py          # Bot entry, slash commands, 5 loops, DM delivery, backup/restore
├── config.py        # Env, RSS_SOURCES (seed), SCRAPE_TARGETS, thresholds
├── database.py      # KnowledgeDB (items, titles, aliases, feedback, follows/broadcasts, resources, news_sources, title_search_sources, learning_metrics) + Bayesian + proficiency
├── fetchers.py      # RSS, AniList, Jikan, Kitsu, DDG titles, MAL scrape, web news, exhaustive parallel (source-weighted), EverythingMoe, host discovery
├── matcher.py       # normalize, hybrid scorer (5 signals, learned weights), match_title/news with confidence nudge
├── embeds.py        # stats_embed with 7-factor breakdown, search_result, news, digest
├── anisage.db       # Auto-created WAL DB (gitignored)
├── pyproject.toml   # uv
└── .env             # DISCORD_TOKEN, OWNER_ID, etc. (gitignored)
```

## Quick start

```bash
cp .env.example .env      # fill DISCORD_TOKEN and OWNER_ID
uv sync
uv run main.py
```

Then in Discord: `/help` (shows hybrid weights, threshold, isolation, exhaustive sources).

Full setup → **setup.md** · All commands → **commands.md**

## Commands overview

- `/search <name> [type]` — exhaustive (`Anilist+Kitsu+Jikan+DDG+MAL` parallel, source-weighted) → learn + alias + live news (whole web) + ✅/❌
- `/where <name>` — where to watch/read + free hosts + direct deep-links
- `/news [type]` — exhaustive: `cached 20` + live `fetch_web_news` (3 web queries) merged, persisted
- `/follow <name>` / `/unfollow <name>` (autocomplete your follows) / `/unfollow-all` / `/following` — per `guild_id`
- `/digest` — exhaustive `fetch_web_news` + cached merge
- `/trending [type]` — AniList + fallback `fetch_exhaustive_search` for 3 titles
- `/learned` → `anisage_learned.json` (titles+items, per-context) + paginated embeds
- `/learned-db` → raw `anisage.db` (per-context, WAL checkpoint)
- `/input-learned <file: anisage_learned.json>` — **merge, skip existing** (`if get_title(key) exists → skipped`) — restores after DB loss
- `/input-learned-db <file: anisage_*.db>` — **merge .db**, skips existing titles/items/aliases/resources by `key/url/alias/slug`, handles old DB without new columns
- `/stats` — 7-factor proficiency `mastery/accuracy/breadth/depth/vitality/velocity/alias_quality` + momentum
- `/learning` — hybrid weights, 7-day accuracy, history
- `/start`/`/stop` — per-guild auto-news digest
- `/learn` (owner) — force cycle
```

