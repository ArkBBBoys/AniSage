# Commands — AniSage

All commands are **Discord slash commands** (`/command`). Most results are delivered
to your **DM**; `/stats`, `/help`, and follow-management replies are ephemeral
(only you see them). Owner-only commands require your `OWNER_ID`.

Legend:
- 📬 = sends to your DM
- 👁 = ephemeral (only you see it)
- 🔒 = owner only

---

## `/help` 👁
Shows a quick list of every command and explains that search is matched-based and
that EverythingMoe is used as a valuable resource.

---

## `/news [media_type]` 📬
Sends the latest scraped news to your DM (up to 15 items), newest first.
- **`media_type`** (optional): `anime` / `manga` / `manhwa` / `all` (default `all`).
- If nothing is cached yet, run `/learn` first.

---

## `/search <name> [media_type]` 📬
The flagship matched-based lookup. Resolves a title and DMs you:
- The canonical title, confidence, AniList/MAL IDs, known aliases.
- Legal **watch/read** links (Crunchyroll, MangaPlus, Webtoon, etc.).
- 🆓 Free hosts (EverythingMoe index) **and** 🔗 direct search deep-links on any
  host the bot has learned a search endpoint for.
- Related news matching that title.
- A **✅/❌ feedback** row — reacting teaches the bot (see Self-Learning below).
- **`name`**: any spelling, abbreviation, or partial (`JJK`, `jujutsu kaisen season 2`).
- **`media_type`** (optional): `anime` / `manga` / `manhwa` (default `anime`).
- Resolution order: learned knowledge base → live AniList/Jikan (then learned).

---

## `/where <name> [media_type]` 📬
"Where can I watch/read this?" Sends the resolved title with both **legal** links
and the **free EverythingMoe index + direct host search links**. Useful when you
just want sources, not the full search dossier.

---

## `/follow <name>` 👁
Follows a title so you get **DM alerts** whenever related news breaks. The bot
resolves the name (matched-based) and stores the subscription. Optional
`NEWS_CHANNEL_ID` also receives the alert in your server.

## `/unfollow <name>` 👁
Stops following a title (matched-based resolution).

## `/following` 👁
Lists everything you currently follow.

---

## `/digest` 📬
Sends a fresh news digest (recent items) to your DM.

---

## `/learned` 📬
**Exports EVERYTHING the bot knows to your DM:**
- A `/stats` embed (knowledge + proficiency).
- The full title knowledge base, paginated into DM messages.
- A downloadable **`anisage_learned.json`** file containing stats, all learned
  titles (with aliases, watch/read links), and recent items — the complete,
  machine-readable brain of the bot.

---

## `/stats` 👁
Shows the live knowledge & proficiency dashboard:
- News items stored, titles learned, **free hosts** (EverythingMoe), human feedback.
- **Match accuracy** (from your ✅/❌ feedback).
- **Average confidence** and the composite **🚀 Proficiency (0–100)**.

---

## `/trending [media_type]` 📬
What's hot right now (top 10 from AniList, also learned into the knowledge base).
Each result includes legal links, free-host links, and direct search deep-links.
- **`media_type`** (optional): `anime` (default) / `manga`.

---

## `/learn` 🔒
Forces an immediate learning/scrape cycle: news + trending + EverythingMoe index
refresh + self-heal. Owner only.

---

# How search & learning work (the important part)

### Matched-based, not keyword
`matcher.match_title()` does **token-set fuzzy matching** (`rapidfuzz`) over
normalized title tokens, not substring matching. Stage 1 checks a learned alias
table (instant, high confidence); stage 2 fuzzy-scores every canonical title and
returns the best above `MATCH_THRESHOLD`. This is why `JJK`, `jujutsu`, and
`Jujutsu Kaisen season 2` all collapse to the same canonical work.

### Continuous self-learning
Three background loops run while the bot is online:

| Loop | Cadence | What it does |
|---|---|---|
| `learn_loop` | `LEARN_INTERVAL` (15 min) | News cycle + AniList trending + EverythingMoe refresh + self-heal. |
| `watch_loop` | `WATCH_INTERVAL` (30 min) | For each followed title, DMs you new related news (and posts to `NEWS_CHANNEL_ID`). |
| `explore_loop` | ~120 s | **Always crawling**: re-syncs EverythingMoe, round-robins health checks on hosts, resolves each host's real domain, learns its exact search param, rotates through news sources for freshness. |

### Free-host self-healing (URLs churn constantly)
EverythingMoe annotates dead hosts (`"… moved to Graveyard"`). When a host
vanishes from the live index, the bot marks it `dead` and tries to discover its
replacement by name similarity. It also continuously **re-resolves each host's
real domain** (these pirate/mirror domains move constantly) and **re-learns the
exact search parameter** that host expects (one site uses `?q=`, another
`?keyword=`, another `?s=`). The learned param is stored per-host
(`resources.search_param`) and used verbatim in deep-links — never assumed.

### Feedback reinforces learning
Reacting ✅ to a `/search` result records a correct match: it boosts that title's
confidence and saves your query as a permanent alias. Reacting ❌ lowers it. This
is what drives the `accuracy` and `proficiency` numbers upward over time.

---

# Examples

```
/search JJK
→ DMs you Jujutsu Kaisen with legal + free-host links, related news, and ✅/❌.

/where Solo Leveling
→ DMs you where to watch/read it (legal + free index + direct searches).

/follow Frieren
→ You'll be DM'd when Frieren news appears.

/learned
→ DMs the entire knowledge base + anisage_learned.json download.

/where One Piece
→ DMs sources for One Piece.
```
