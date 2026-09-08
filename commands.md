# Commands — AniSage

All commands are **Discord slash commands** (`/command`). Most results are delivered to your **DM** or **ephemerally in the server** (per-guild isolated); `/stats`, `/help`, `/learning` and follow-management replies are ephemeral (only you see them). Owner-only commands require your `OWNER_ID`.

Legend:
- 📬 = sends to your DM (or ephemeral in server if used in guild — per `guild_id` isolated)
- 👁 = ephemeral (only you see it)
- 🔒 = owner only

---

## `/help` 👁
Shows a quick list of every command, hybrid scorer weights, Bayesian self-learning, and exhaustive web sources. Explains matched-based search and isolation model.

---

## `/search <name> [media_type]` 📬
The flagship **exhaustive, hybrid, self-learning** lookup. Resolves a title and DMs/ephemerals you:
- The canonical title, **confidence** (Bayesian `Beta(2,1)` + `times_seen` + recency + alias weight + source quality), AniList/MAL IDs, known aliases, `score` + `match_method` (`alias`/`alias_token`/`fuzzy`/`live-exhaustive`)
- Legal **watch/read** links + 🆓 **Free hosts** (EverythingMoe live index) + 🔗 **direct search deep-links** on any host the bot has learned a search endpoint for
- **Exhaustive related news** — whole-web crawl (`fetch_news_for_title` parallel `Web:anime news` + DDG titles + RSS filtered, merged, `match_news` scored, persisted)
- A **✅/❌ feedback** row — `✅` → `feedback_correct++`, `confidence` Bayesian boost + alias `weight +0.32` + new alias `weight 3.0`; `❌` → `feedback_wrong++`, `-1.1` + alias decay `-0.42`
- **`name`**: any spelling, typo, acronym, partial (`JJK` → `Jujutsu Kaisen` initials 96, `Spirit Chronicales` typo via `jaro 18% + ngram 15%` → `96.5`, `BTTH` → `Battle Through the Heavens` acronym, `Solo Leveling ch 150`)
- **`media_type`** (optional): `anime` / `manga` / `manhwa` (default `anime`) — preferred type gets `+1.5` bonus
- **Resolution order**: `1` learned alias (instant `90+weight*2`) → `2` `alias_token` containment → `3` hybrid fuzzy over `canonical+aliases` (`threshold=config.MATCH_THRESHOLD` auto-tuned hourly 58-84) → `4` **exhaustive live search** (`AniList×2 + Jikan×2 + Kitsu×2 + DDG×2 + MAL×2` parallel, weighted by `TitleSearchSource` reliability 0.4-1.8, `score * (0.78+0.44*weight)`) → `5` recent news title fallback
- **Exhaustive**: not DB-only, not hard-coded — fires **10 sources in parallel**, acronym expansion via DDG (`BTTH` → DDG `Battle` → re-search), `normalize` + `score_pair` hybrid.

---

## `/where <name> [media_type]` 📬
"Where can I watch/read this?" — same exhaustive `fetch_exhaustive_search` as `/search` (tries `anime` then `manga` opposite), then shows `search_result_embed` + free hosts + direct deep-links. Per `guild_id` isolated.

---

## `/news [media_type]` 📬 — **Exhaustive web crawl**
Sends latest news (up to 15, newest first) — **whole web, not just fixed RSS**.
- **`media_type`**: `anime` / `manga` / `manhwa` / `all` (default `all`)
- **Exhaustive**: `cached recent_items (20)` + live `fetch_web_news` parallel (`anime news` + `manga news` + `manhwa news` for `all`, else `"<type> news"` + `"<type> release news"`) + `run_news_cycle` fallback, de-dup by `url`, persisted via `db.add_item` so future `/news` is richer. If `Web:` items exist, header shows `X from whole web + Y cached`.
- **Self-learning sources**: uses `get_active_news_sources(limit 12)` ordered by `score` (reliability 0-1), not just `config.RSS_SOURCES` (14 seeded, plus web-discovered via `discover_rss_sources_via_web`).
- **Isolation**: `server` → ephemeral in that server, `DM` → DM with `send_dm` verification + fallback to followup (never `sent to DM` with nothing).

---

## `/follow <name>` 👁
Follows a title so you get **DM + channel alerts** whenever related news breaks (via `watch_loop` live `fetch_news_for_title` per follow, threshold 58/55).
- **Per-guild isolated**: `follows (user_id, title_key, guild_id)` unique, `follow(user, key, guild_id=gid)` where `gid = guild.id or 0`. `/following` shows only this guild/DM.
- Resolves via `match_title 60` → fallback `fetch_exhaustive_search` (`anime` then `manga`) → `learn_title`.

## `/unfollow <name>` 👁 — **Enhanced with autocomplete**
Stops following a title. **Autocomplete** shows only titles you actually follow in this guild/DM (up to 25, filtered by `current` input). Handles direct canonical match and fuzzy fallback, per `guild_id` + cleans global.
- `name` description: `Title to unfollow — autocomplete shows your followed titles`
- If not found, shows helpful `You follow: ONE PIECE, ...` sample.

## `/unfollow-all` 👁 — **New**
Unfollows **all** titles you follow **in this server/DM only** (per `guild_id` isolated, does not affect other guilds). Counts and reports `Unfollowed all 3 titles in this server (guild 123)`.

## `/following` 👁
Lists everything you currently follow **in this context** (`guild_id`). If empty in this guild but you follow elsewhere, hints `You follow 2 title(s) in other contexts`. Shows `in DM/inbox` vs `in ServerName`.

---

## `/digest` 📬 — **Exhaustive**
Sends a fresh news digest (recent items) — **cached + live web** merged (`fetch_web_news anime news 10` + `manga news 6` fallback, persisted, `run_news_cycle` if still empty), per `guild_id` isolated.

---

## `/learned` 📬
**Exports EVERYTHING the bot knows to your DM/server (per-context):**
- A `/stats` embed (7-factor proficiency + breakdown)
- The full title knowledge base, paginated into embeds of 20 lines each (`canonical (type) — conf X, seen N` + aliases)
- A downloadable **`anisage_learned.json`** file containing `stats`, all learned titles (with aliases, watch/read links, exact `times_seen`/`confidence`/`feedback`/`ngram_sig`), and `recent_items` (200) — the complete machine-readable brain.

---

## `/learned-db` 📬 — **Per-context**
Sends the raw **`anisage.db` SQLite file** (`anisage_<guild_id>.db`, `WAL checkpoint TRUNCATE` first, `<24 MB` check, `tempfile` copy to avoid locks) — per `guild_id` isolated, not leaked to DMs.

---

## `/input-learned <file: anisage_learned.json>` 👁 🔒 — **New, merge skip-existing**
Restores backup JSON from `/learned` (and also accepts `/learned-db` JSON). **Merges, skips existing** (no overwrite, no duplicate) — ideal if DB is lost.
- Validates `filename endswith .json`, `<25 MB`, `JSON.parse`, `{"titles": [...], "recent_items": [...]}` with `key/canonical` required
- **Titles**: `if db.get_title(key) exists → skipped++` else `learn_title(TitleRecord)` then patch `times_seen/confidence/first_seen/last_seen` to backup values if higher; registers aliases via `_set_alias`
- **Items**: `if Item.url exists → skipped` else `add_item(NewsItem)` (`ON CONFLICT DO NOTHING`)
- Reports `Imported X new / Skipped Y already existed / Failed Z` + `Aliases/Resources` for DB, plus `Now: N titles · M items · proficiency P%`

---

## `/input-learned-db <file: anisage_*.db>` 👁 🔒 — **New, merge skip-existing**
Restores backup **raw `.db` file** from `/learned-db` (and also accepts any SQLite). Same merge logic but via direct `sqlite3` read-only attach:
- Validates `SQLite format 3` header, `<24 MB`, writes to temp
- **Titles**: `SELECT key,canonical,... FROM titles` → `if get_title(key) exists → skipped` else `INSERT` preserving `times_seen/confidence/feedback_correct/wrong/ngram_sig` + re-register aliases
- **Items**: `SELECT url ... FROM items` → `if Item.url exists → skipped` else `INSERT` preserving `fetched_at`
- **Aliases/Resources**: `SELECT alias/title_key/weight ... FROM aliases/resources` → `if exists → skipped` else `INSERT`
- Handles both new DB (with `feedback_correct` columns) and old DB (fallback `SELECT key,canonical...` without new cols)
- Reports `Titles: +X/skip Y, Items: +X/skip Y, Aliases: +X/skip Y, Resources: +X/skip Y`

---

## `/stats` 👁
Shows the live **7-factor proficiency** dashboard:
- `📚 Knowledge base` — `titles`/`items` + `Depth` (`avg_times_seen`) + `Breadth`
- `🎯 Performance (self-learning)` — `Accuracy` + `smooth` (`Beta(2,2)` prior) + `avg confidence` + `7-day fb` + `alias_quality`
- `🆓 Free hosts` — `alive/total`
- `💬 Feedback` — total
- `⚡ Vitality / Velocity` — `Vitality` + `Velocity`
- `🧩 Proficiency Breakdown` — `mastery 30% + accuracy 25% + breadth 15% + depth 10% + vitality 10% + velocity 7% + alias_quality 3% + momentum`
- `🚀 Proficiency — self-improving` `bar 49.2%`

Formula: `proficiency = mastery*0.30 + accuracy_smooth*0.25 + breadth*0.15 + depth*0.10 + vitality*0.10 + velocity*0.07 + alias_quality*0.03 + momentum` (momentum `+4.5` if `acc>78` & `recent_titles≥2`), clamped 0-100, min 8 if any data.

---

## `/learning` 👁 — **New**
Self-learning & self-improvement report: hybrid weights (`set/sort/jaro/ngram/partial`), `MATCH_THRESHOLD` (auto-tuned 58-84), 7-day `Queries Acc (correct/wrong)`, `Titles Avg conf`, `Aliases` counts, `Recent Windows` (`<t:R> acc thr`), and `Self-Improvement` hourly tasks.

---

## `/trending [media_type]` 📬
What's hot right now (top 10 from AniList, also learned). Fallback via `fetch_exhaustive_search` for 3 titles if AniList 504. Per `guild_id` isolated, DM verification.

---

## `/learn` 🔒
Forces an immediate learning/scrape cycle: `run_news_cycle` (exhaustive) + `fetch_anilist_trending` 30+30 + `EverythingMoe` refresh + `log_scrape`. Owner only.

---

## `/start` / `/stop` 👁
Auto-news: per `guild_id` isolated digest every `BROADCAST_INTERVAL` (30 min). `start` → `set_broadcast(user, channel_id, guild_id)`, `stop` → `stop_broadcast(user, guild_id)` (also cleans global). `broadcast_loop` uses `run_news_cycle(db=db)` + `news_since(last_sent)`.

---

# How search & learning work (the important part)

### Hybrid matched-based, not keyword — self-learning weighted
`matcher.score_pair()` is **hybrid 5-signal** (`token_set`, `token_sort`, `partial`, `jaro-winkler`, `ngram cosine` for typos like `Chronicales` vs `Chronicles`) with **learned weights** (`set 0.35 sort 0.22 jaro 0.18 ngram 0.15 partial 0.10`) auto-tuned hourly from 7-day `correct vs wrong` gap. Acronyms (`BTTH` initials `B-T-T-H`) handled generically. `normalize` strips `the/an` etc., handles `NFKD` + `Seirei Gensouki` aliases. `score_pair` conservative for short queries (overlap guard, length penalty).

`matcher.match_title()` 3-stage: `1` exact `alias_lookup` (`90+weight*2`) → `1b` `alias_token` containment (`JJK`) → `2` hybrid fuzzy over `canonical+aliases` + `confidence*0.08` nudge, **strict `threshold=config.MATCH_THRESHOLD` (58-84 auto-tuned)** — returns `None` if below, forcing live exhaustive (was bug returning low-confidence random isekai).

### Exhaustive live search — whole internet, not DB-only (no hard-coding)
`fetch_exhaustive_search(session, query, preferred_type, db)` fires **10 sources in parallel**: `AniList×2 + Jikan×2 + Kitsu×2 + DDG×2 + MAL×2` (+ normalized variant), each wrapped with `_track` latency/success → `update_title_source_stats` (weight `0.4-1.8`). Candidates scored via `score_pair(query, canonical+aliases+key)` + `preferred_type +1.5` + `internal score*0.95` → weighted `* (0.78+0.44*src_w)`. Acronym expansion via `fetch_ddg_titles` → `expanded` → re-search, `the`-strip fallback. Filters web noise (`reddit/youtube` <82 → prefer non-noise). Not hard-coded: discovers `BTTH`→`Battle` via DDG title `Battle Through the Heavens - Wikipedia` + initials.

### Continuous self-learning & self-improvement
**Loops:**
| Loop | Cadence | What it does |
|---|---|---|
| `learn_loop` | `LEARN_INTERVAL` 15m | `run_news_cycle(db=db)` exhaustive → `bulk_add_items` → `fetch_anilist_trending 30+30` → `learn_title` (Bayesian) → `EverythingMoe` refresh + self-heal |
| `watch_loop` | `WATCH_INTERVAL` 30m | `all_follows_detailed()` per `guild_id` → `fetch_news_for_title` (parallel 12, `cached 80` + live `Web:anime news` 8, merged, `match_news` 58/55, persist live) → `send_dm` + `NEWS_CHANNEL_ID` + per-guild `broadcast` channel, `_sent_cache` 8000 |
| `explore_loop` | 120s | `EverythingMoe` re-sync every 2, health check 12 hosts, `resolve_host_domain` + `learn_host_search` 6 hosts, **exhaustive news** `2 RSS (DB-ranked) + 2 web news + full exhaustive` → `db` |
| `broadcast_loop` | `BROADCAST_INTERVAL` 30m | `run_news_cycle(db=db)` → `news_since(last_sent)` per `broadcasts (user_id,guild_id)` |
| `self_improvement_loop` | 3600s | `prune_stale_aliases(90d,0.32)`, `decay_stale_titles(60d,0.12)`, `recompute_alias_weights_from_feedback` (Bayesian `1+0.52*c-0.68*w`), `auto_tune_threshold` (7-day `0.62*avg_c+0.38*avg_w` → 22% toward ideal), hybrid weight tuner (gap<18 → boost `jaro/ngram`), `sync_news_sources_from_config`, `discover_rss_sources_via_web(limit 4)`, `prune_dead_news_sources(40)`, `log_learning_window` |

**Bayesian confidence** (`database.py:542`): `alpha=1.8+acc*1.8`, `beta=1.2+(1-acc)*1.4`, `posterior=(correct+alpha)/(total+alpha+beta)` shrink `1-1/(n+4)`, `freq=4.5*log1p(times_seen)` (+1.2 log1p >30, cap 22), `recency +3.5 if ≤7d else -10 if >45d`, `alias_bonus=(avg_weight-1)*6.5`, `source_bonus=(q-0.5)*8`, `base=old*0.15+7.8`, `re-seen +0.35`, `feedback correct +1.6+(75-old)*0.035` (streak +0.9) vs `wrong -1.3`.

**Sources self-learn**: `NewsSource` (`reliability` EMA `0.88*new+0.12*old`, `score 0.68*rel+0.18*avg_items/8+0.14*recency`, `active→paused 3 fails →dead 7`), `TitleSearchSource` (`weight` from `succ_rate*latency_factor`), `discover_rss_sources_via_web` (DDG `anime news RSS` → extract `.xml/rss/feed` → test `<rss>` → `discover_new_sources_from_web`).

### Free-host self-healing (URLs churn constantly)
`fetch_everythingmoe_index` → `upsert_resource` → `mark_resource_dead` if missing (`not in current index`) → `_find_replacement` (`score_pair` ≥70). `resolve_host_domain` extracts real domain from `EverythingMoe /s/` page, `learn_host_search` reads `<form>` for exact `search_param` (`?q` vs `?keyword`).

### Feedback reinforces learning
`MatchFeedback` view `✅/❌` → `record_feedback(query, key, correct, method, score)` → `feedback_correct/wrong++`, `confidence` Bayesian recomputed, `Alias hits_correct/wrong` → `weight ±0.32/0.42`, `_set_alias(query,key,3.0)` on correct (blended `max` with learned). `get_learning_report` tracks 7-day accuracy.

---

# Examples

```
/search JJK
→ DMs you Jujutsu Kaisen with legal + free-host links, 4 related news via whole-web crawl, and ✅/❌.

/where Solo Leveling
→ DMs you where to watch/read it (legal + free index + direct searches).

/follow Frieren
→ You'll be DM'd when Frieren news appears (per guild).

/unfollow Bleach  (autocomplete shows your follows)
→ 🚫 Unfollowed Bleach (guild 123)

/unfollow-all
→ 🚫 Unfollowed all 3 titles in this server

/learned
→ DMs the entire knowledge base + anisage_learned.json download.

/input-learned file: anisage_learned.json
→ ✅ Imported 1 new / Skipped 1 already existed

/learned-db → anisage_123.db (0.34 MB)
/input-learned-db file: anisage_123.db
→ ✅ Imported 1 new / Skipped 1 already existed

/where One Piece
→ DMs sources for One Piece.

/news (all)
→ 15 stories: 4 from whole web + 11 cached, exhaustive (web news for anime/manga/manhwa parallel)

Proficiency — /stats shows 7-factor breakdown + momentum, /learning shows hybrid weights
```
