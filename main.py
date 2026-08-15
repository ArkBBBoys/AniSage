"""AniSage — a personal, self-learning anime / manga / manhwa news & lookup bot.

Features
--------
* Scrapes a wide net of RSS feeds + direct HTML + AniList/Jikan APIs.
* Distills everything into a persistent SQLite knowledge base that *improves*
  with use (feedback-reinforced aliases + confidence scoring).
* Matched-based (NOT keyword) title search via token-set fuzzy matching.
* DM delivery of news, digests, search results and a full "what I've learned"
  export. Plus optional server news channel.
* Follow titles to get DM alerts when related news breaks.
* A live "proficiency" metric that rises as the bot learns.

Note on content: the bot links to *legal* streaming/reading sources rather than
redistributing copyrighted episode/video/manga files.
"""
from __future__ import annotations

import asyncio
import io
import json
import time

import aiohttp
import discord
from discord import app_commands
from discord.ext import tasks

import config
import embeds
from database import KnowledgeDB, NewsItem, TitleRecord
from fetchers import (
    fetch_anilist_search,
    fetch_anilist_trending,
    fetch_everythingmoe_index,
    fetch_jikan_search,
    fetch_rss,
    learn_host_search,
    resolve_host_domain,
    run_news_cycle,
)
from matcher import match_news, match_title, normalize, parse_query, score_pair

intents = discord.Intents.default()


class AniSageClient(discord.Client):
    """Client subclass so cleanup actually runs: discord.py has no on_close
    event (it is never dispatched), but close() is always called on shutdown."""

    async def close(self):
        if _session and not _session.closed:
            await _session.close()
        db.close()
        await super().close()


bot = AniSageClient(intents=intents)
tree = app_commands.CommandTree(bot)

db = KnowledgeDB()
_session: aiohttp.ClientSession | None = None
_last_watch_run = time.time()
_sent_cache: set[tuple[str, int]] = set()  # (follow_key, item_id) already alerted


# ----------------------------------------------------------------- helpers
async def session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession()
    return _session


async def safe_defer(inter: discord.Interaction, ephemeral: bool = True) -> bool:
    """Defer an interaction, tolerating an already-expired interaction.

    Returns False (do not continue) if the interaction is already gone, so a slow
    startup / saturated event loop can't crash the command with a 404.
    """
    try:
        await inter.response.defer(ephemeral=ephemeral)
        return True
    except (discord.NotFound, discord.InteractionResponded, discord.HTTPException):
        return False


async def safe_respond(inter: discord.Interaction, **kwargs) -> bool:
    """Send the initial response, tolerating an already-expired interaction."""
    try:
        await inter.response.send_message(**kwargs)
        return True
    except (discord.NotFound, discord.InteractionResponded, discord.HTTPException):
        # Already responded or expired — nothing to do.
        try:
            await inter.followup.send(**kwargs)
            return True
        except Exception:
            return False


async def send_dm(user_id: int, *args, **kwargs) -> bool:
    try:
        user = bot.get_user(user_id) or await bot.fetch_user(user_id)
        if user is None:
            return False
        ch = user.dm_channel or await user.create_dm()
        # Discord caps message content at 2000 chars; split long text automatically.
        MAX = 1950
        content = args[0] if args and isinstance(args[0], str) else kwargs.get("content")
        if content and isinstance(content, str) and len(content) > MAX and not kwargs.get("file"):
            rest = args[1:]
            parts = [content[i:i + MAX] for i in range(0, len(content), MAX)]
            for i, part in enumerate(parts):
                if i == 0:
                    await ch.send(part, *rest, **{k: v for k, v in kwargs.items() if k != "content"})
                else:
                    await ch.send(part)
            return True
        await ch.send(*args, **kwargs)
        return True
    except Exception as ex:
        print(f"[dm] failed for {user_id}: {ex}")
        return False


def owner_only():
    async def pred(inter: discord.Interaction) -> bool:
        return inter.user.id == config.OWNER_ID
    return app_commands.check(pred)


def free_links(kind: str, limit: int = 8) -> str:
    """Currently-live free hosts for a media type (EverythingMoe index + direct)."""
    emoe_kind = "anime" if kind == "anime" else "manga"
    res = db.get_resources(emoe_kind)[:limit]
    if not res:
        return "— (run /learn to refresh EverythingMoe index)"
    lines = []
    for r in res:
        direct = r.get("domain") or ""
        if direct:
            lines.append(f"[{r['name']}](https://{direct})")
        else:
            lines.append(f"[{r['name']}]({r['page_url']})")
    return "\n".join(lines)


def direct_search_links(rec: dict) -> str:
    """Build per-title direct search deep-links on the hosts that have one."""
    emoe_kind = "anime" if rec.get("media_type") == "anime" else "manga"
    canonical = rec.get("canonical", "")
    q = canonical.replace(" ", "+")
    links = []
    for r in db.get_resources(emoe_kind):
        search = r.get("search_url") or ""
        if search and "{q}" in search:
            links.append(f"[{r['name']}]({search.replace('{q}', q)})")
        if len(links) >= 10:  # keep the embed field under Discord's 1024-char cap
            break
    return "\n".join(links) if links else "—"


# --------------------------------------------------------------- feedback UI
class MatchFeedback(discord.ui.View):
    def __init__(self, query: str, key: str):
        super().__init__(timeout=300)
        self.query = query
        self.key = key

    async def _respond(self, inter: discord.Interaction, content: str):
        # Run the (synchronous) DB write off the event loop, then edit the
        # message. Tolerate an already-expired interaction so a saturated loop
        # can't crash the view callback.
        try:
            if inter.response.is_done():
                await inter.edit_original_response(content=content, view=None)
            else:
                await inter.response.edit_message(content=content, view=None)
        except (discord.NotFound, discord.InteractionResponded, discord.HTTPException):
            pass

    @discord.ui.button(label="✅ Correct", style=discord.ButtonStyle.success)
    async def yes(self, inter: discord.Interaction, _b: discord.ui.Button):
        await asyncio.get_event_loop().run_in_executor(
            None, db.record_feedback, self.query, self.key, True
        )
        await self._respond(
            inter,
            f"✅ Learned: “{self.query}” → that title. Confidence boosted.",
        )

    @discord.ui.button(label="❌ Wrong", style=discord.ButtonStyle.danger)
    async def no(self, inter: discord.Interaction, _b: discord.ui.Button):
        await asyncio.get_event_loop().run_in_executor(
            None, db.record_feedback, self.query, self.key, False
        )
        await self._respond(
            inter, "❌ Noted — that match was wrong, I'll rank it lower."
        )


# ----------------------------------------------------------------- commands
@tree.command(name="help", description="Show what AniSage can do")
async def cmd_help(inter: discord.Interaction):
    e = discord.Embed(
        title="🧠 AniSage — self-learning anime/manga/manhwa bot",
        color=config.THEME_COLOR,
        description="Search is **matched**, not keyword: `JJK` resolves to "
                    "*Jujutsu Kaisen*. EverythingMoe.com teaches the bot which "
                    "free anime/manga/manhwa hosts are currently live.",
    )
    e.add_field(
        name="🔍 Search & discover",
        value="`/search <name> [type]` — matched lookup (DM + feedback), supports "
              "`JJK Season 2`, `Solo Leveling ch 150`, `S02E05`\n"
              "`/news [type]` — latest scraped news (DM)\n"
              "`/trending [anime|manga]` — what's hot right now\n"
              "`/where <name>` — where to watch/read (legal + free hosts)",
        inline=False,
    )
    e.add_field(
        name="🔔 Follow & alerts",
        value="`/follow <name>` — DM alerts when news breaks\n"
              "`/unfollow <name>` / `/following`\n"
              "`/start` / `/stop` — auto-news digest every 30 min (DM + channel)",
        inline=False,
    )
    e.add_field(
        name="🧠 Learn & inspect",
        value="`/digest` — fresh digest to DM\n"
              "`/learned` — export EVERYTHING the bot knows to DM\n"
              "`/stats` — proficiency & knowledge stats\n"
              "`/learn` — force a learning cycle now (owner)",
        inline=False,
    )
    embeds._stamp(e)
    if not await safe_respond(inter, embed=e, ephemeral=True):
        return


@tree.command(name="news", description="Get the latest news (sent to your DM)")
@app_commands.describe(media_type="Filter: anime / manga / manhwa / all")
async def cmd_news(inter: discord.Interaction, media_type: str = "all"):
    if not await safe_defer(inter, ephemeral=True):
        return
    items = db.recent_items(limit=20, media_type="" if media_type == "all" else media_type)
    if not items:
        await inter.followup.send("No news cached yet — run `/learn` first.", ephemeral=True)
        return
    await send_dm(inter.user.id, embed=embeds.header_embed(
        "📰 Latest news",
        f"{min(15, len(items))} stories from the crawler · newest first"))
    for it in items[:15]:
        await send_dm(inter.user.id, embed=embeds.news_embed(it))
    await inter.followup.send(f"📬 Sent {min(15,len(items))} stories to your DM.", ephemeral=True)


@tree.command(name="search", description="Matched-based title search (not keyword)")
@app_commands.describe(name="Title to look up (e.g. 'JJK Season 2', 'Solo Leveling ch 150')", media_type="anime / manga / manhwa")
async def cmd_search(inter: discord.Interaction, name: str, media_type: str = "anime"):
    if not await safe_defer(inter, ephemeral=True):
        return
    mt = media_type.upper()
    anilist_type = "ANIME" if mt == "ANIME" else "MANGA"
    clean, season, unit = parse_query(name)

    # 1) try learned knowledge base first
    rec = match_title(name, db, threshold=config.MATCH_THRESHOLD)
    # 2) if unsure, ask AniList + Jikan live and learn it
    if not rec or rec["score"] < config.MATCH_THRESHOLD:
        s = await session()
        live = await fetch_anilist_search(s, clean, anilist_type)
        if not live and mt == "ANIME":
            live = await fetch_jikan_search(s, clean, "anime")
        if live:
            db.learn_title(TitleRecord(
                key=live["key"], canonical=live["canonical"],
                media_type=live["media_type"], external_id=live.get("anilist_id", ""),
                anilist_id=live.get("anilist_id", ""), mal_id=live.get("mal_id", ""),
                image=live.get("image", ""),
                aliases=[], watch_links=live["watch_links"], read_links=live["read_links"],
            ))
            rec = db.get_title(live["key"])
            if rec:
                rec["score"] = 95.0
                rec["match_method"] = "live"

    if not rec:
        await inter.followup.send(f"🤔 No solid match for “{name}”. Try a different spelling.", ephemeral=True)
        return

    rec["season"], rec["unit"] = season, unit

    embed = embeds.search_result_embed(rec, name)
    embed.add_field(
        name="🆓 Free hosts (EverythingMoe)",
        value=free_links(rec.get("media_type", "anime")),
        inline=False,
    )
    dsl = direct_search_links(rec)
    if dsl != "—":
        embed.add_field(name="🔗 Direct search on free hosts", value=dsl, inline=False)
    await send_dm(inter.user.id, embed=embed,
                  view=MatchFeedback(name, rec["key"]))
    # also related news
    news = match_news(rec["canonical"], db.recent_items(50), threshold=55)
    if news:
        await send_dm(inter.user.id, embed=embeds.item_search_embed(news, rec["canonical"]))
    await inter.followup.send("📬 Match + related news sent to your DM. React to teach me!", ephemeral=True)


@tree.command(name="follow", description="Follow a title for DM alerts")
async def cmd_follow(inter: discord.Interaction, name: str):
    if not await safe_defer(inter, ephemeral=True):
        return
    rec = match_title(name, db, threshold=60)
    if not rec:
        s = await session()
        clean, _, _ = parse_query(name)
        live = await fetch_anilist_search(s, clean, "ANIME") or await fetch_anilist_search(s, clean, "MANGA")
        if live:
            db.learn_title(TitleRecord(
                key=live["key"], canonical=live["canonical"],
                media_type=live["media_type"], anilist_id=live["anilist_id"],
                image=live.get("image", ""),
                watch_links=live["watch_links"], read_links=live["read_links"]))
            rec = db.get_title(live["key"])
    if not rec:
        await inter.followup.send(f"Couldn't resolve “{name}”. Search first.", ephemeral=True)
        return
    db.follow(inter.user.id, rec["key"])
    await inter.followup.send(f"🔔 Following **{rec['canonical']}**. You'll get DM alerts.", ephemeral=True)


@tree.command(name="unfollow", description="Stop following a title")
async def cmd_unfollow(inter: discord.Interaction, name: str):
    rec = match_title(name, db, threshold=60)
    if rec:
        db.unfollow(inter.user.id, rec["key"])
        await safe_respond(inter, content=f"🚫 Unfollowed **{rec['canonical']}**.", ephemeral=True)
    else:
        await safe_respond(inter, content="No match to unfollow.", ephemeral=True)


@tree.command(name="following", description="List titles you follow")
async def cmd_following(inter: discord.Interaction):
    keys = db.followed(inter.user.id)
    if not keys:
        await safe_respond(inter, content="You're not following anything yet.", ephemeral=True)
        return
    lines = []
    for k in keys:
        t = db.get_title(k)
        lines.append(f"• {t['canonical'] if t else k}")
    await safe_respond(inter, content="🔔 **Following:**\n" + "\n".join(lines), ephemeral=True)


@tree.command(name="digest", description="Send a fresh news digest to your DM")
async def cmd_digest(inter: discord.Interaction):
    if not await safe_defer(inter, ephemeral=True):
        return
    items = db.recent_items(15)
    if not items:
        await inter.followup.send("Nothing digested yet — run `/learn`.", ephemeral=True)
        return
    await send_dm(inter.user.id, embed=embeds.digest_embed(items, "Your feed"))
    await inter.followup.send("📬 Digest sent to DM.", ephemeral=True)


@tree.command(name="learned", description="Export EVERYTHING the bot has learned to your DM")
async def cmd_learned(inter: discord.Interaction):
    if not await safe_defer(inter, ephemeral=True):
        return
    s = db.stats()
    await send_dm(inter.user.id, embed=embeds.stats_embed(s))
    titles = db.all_titles()
    # paginate titles into embeds of 20 lines each
    chunk = []
    page = 0
    total_pages = max(1, -(-len(titles) // 20))
    for i, t in enumerate(titles, 1):
        aliases = t["aliases"]
        if isinstance(aliases, str):
            try:
                aliases = json.loads(aliases)
            except Exception:
                aliases = []
        line = (f"**{t['canonical']}** ({t['media_type']}) "
                f"— conf {t['confidence']:.0f}, seen {t['times_seen']}")
        if aliases:
            line += f"\n   aliases: {', '.join(aliases[:6])}"
        chunk.append(line)
        if len(chunk) == 20:
            page += 1
            await send_dm(inter.user.id, embed=embeds.learned_chunk_embed(chunk, page, total_pages))
            chunk = []
    if chunk:
        page += 1
        await send_dm(inter.user.id, embed=embeds.learned_chunk_embed(chunk, page, total_pages))
    # full machine-readable export as a file
    export = {
        "stats": s,
        "titles": [
            {**t, "aliases": (json.loads(t["aliases"]) if isinstance(t["aliases"], str) else t["aliases"]),
             "watch_links": (json.loads(t["watch_links"]) if isinstance(t["watch_links"], str) else t["watch_links"]),
             "read_links": (json.loads(t["read_links"]) if isinstance(t["read_links"], str) else t["read_links"])}
            for t in titles
        ],
        "recent_items": db.recent_items(200),
    }
    data = json.dumps(export, indent=2, default=str).encode()
    await send_dm(inter.user.id, embed=embeds.header_embed("🗂️ Full export (JSON)"),
                  file=discord.File(io.BytesIO(data), filename="anisage_learned.json"))
    await inter.followup.send(f"📬 Exported {len(titles)} titles + {s['items']} items to your DM.", ephemeral=True)


@tree.command(name="stats", description="Show knowledge & proficiency")
async def cmd_stats(inter: discord.Interaction):
    if not await safe_respond(inter, embed=embeds.stats_embed(db.stats()), ephemeral=True):
        return


@tree.command(name="where", description="Where can I watch/read <name>? (legal + free indexes)")
@app_commands.describe(name="Title to locate", media_type="anime / manga / manhwa")
async def cmd_where(inter: discord.Interaction, name: str, media_type: str = "anime"):
    if not await safe_defer(inter, ephemeral=True):
        return
    rec = match_title(name, db, threshold=60)
    if not rec:
        s = await session()
        clean, _, _ = parse_query(name)
        live = await fetch_anilist_search(s, clean, "ANIME") or await fetch_anilist_search(s, clean, "MANGA")
        if live:
            db.learn_title(TitleRecord(
                key=live["key"], canonical=live["canonical"],
                media_type=live["media_type"], anilist_id=live["anilist_id"],
                image=live.get("image", ""),
                watch_links=live["watch_links"], read_links=live["read_links"]))
            rec = db.get_title(live["key"])
    if not rec:
        await inter.followup.send(f"Couldn't resolve “{name}”. Try /search first.", ephemeral=True)
        return
    embed = embeds.search_result_embed(rec, name)
    embed.add_field(
        name="🆓 Free hosts (EverythingMoe index)",
        value=free_links(rec.get("media_type", "anime")),
        inline=False,
    )
    dsl = direct_search_links(rec)
    if dsl != "—":
        embed.add_field(name="🔗 Direct search on free hosts", value=dsl, inline=False)
    await send_dm(inter.user.id, embed=embed)
    await inter.followup.send("📬 Where-to-watch/read sent to your DM.", ephemeral=True)


@tree.command(name="trending", description="What's trending right now")
@app_commands.describe(media_type="anime / manga")
async def cmd_trending(inter: discord.Interaction, media_type: str = "anime"):
    if not await safe_defer(inter, ephemeral=True):
        return
    s = await session()
    mt = "ANIME" if media_type.lower() == "anime" else "MANGA"
    recs = await fetch_anilist_trending(s, mt, per=10)
    for r in recs:
        db.learn_title(r)
    if not recs:
        await inter.followup.send("Trending fetch failed.", ephemeral=True)
        return
    await send_dm(inter.user.id, embed=embeds.header_embed(
        f"🔥 Trending {media_type}",
        "Top 10 from AniList · learned into the knowledge base",
        media_type=media_type))
    for r in recs:
        em = embeds.search_result_embed({
            **r.__dict__, "score": 99, "match_method": "trending",
            "aliases": r.aliases, "watch_links": r.watch_links, "read_links": r.read_links,
        }, r.canonical)
        em.add_field(
            name="🆓 Free hosts (EverythingMoe)",
            value=free_links(r.media_type), inline=False,
        )
        dsl = direct_search_links({
            "canonical": r.canonical, "media_type": r.media_type,
        })
        if dsl != "—":
            em.add_field(name="🔗 Direct search on free hosts", value=dsl, inline=False)
        await send_dm(inter.user.id, embed=em)
    await inter.followup.send("📬 Trending list sent to DM.", ephemeral=True)


@tree.command(name="learn", description="Force a learning/scrape cycle now")
@owner_only()
async def cmd_learn(inter: discord.Interaction):
    if not await safe_defer(inter, ephemeral=True):
        return
    n, items = await do_learn_cycle()
    await inter.followup.send(f"🧠 Cycle done: {n} new items ingested, knowledge updated.", ephemeral=True)


@tree.command(name="start", description="Auto-news: a digest every 30 min to your DM and this channel")
async def cmd_start(inter: discord.Interaction):
    if not await safe_defer(inter, ephemeral=True):
        return
    ch_id = inter.channel_id if inter.guild else 0
    db.set_broadcast(inter.user.id, ch_id)
    mins = config.BROADCAST_INTERVAL // 60
    where = "this channel **and** your DM" if ch_id else "your DM"
    await inter.followup.send(
        f"📡 **Auto-news ON** — every {mins} min to {where}. Use `/stop` to turn it off.",
        ephemeral=True,
    )


@tree.command(name="stop", description="Stop auto-news digests")
async def cmd_stop(inter: discord.Interaction):
    if not await safe_defer(inter, ephemeral=True):
        return
    if db.stop_broadcast(inter.user.id):
        await inter.followup.send("📡 **Auto-news OFF.**", ephemeral=True)
    else:
        await inter.followup.send("You don't have auto-news running — use `/start`.", ephemeral=True)


# ------------------------------------------------------------- learning loop
async def do_learn_cycle() -> tuple[int, int]:
    """Scrape widely, store items, learn trending titles. Returns (new_items, total)."""
    s = await session()
    loop = asyncio.get_event_loop()
    t0 = time.time()
    n, items = await run_news_cycle(s)
    # Bulk insert in a worker thread so the event loop (and Discord heartbeat) is
    # never blocked by synchronous SQLite commits.
    new_count = await loop.run_in_executor(None, db.bulk_add_items, items)
    # learn trending anime + manga/manhwa to grow the knowledge graph
    trending_recs: list = []
    for mt in ("ANIME", "MANGA"):
        try:
            recs = await fetch_anilist_trending(s, mt, per=30)
            trending_recs.extend(recs)
        except Exception as ex:
            print(f"[learn] trending {mt} failed: {ex}")

    def _learn_all(recs):
        for r in recs:
            db.learn_title(r)

    if trending_recs:
        await loop.run_in_executor(None, _learn_all, trending_recs)
    await refresh_free_index(s)
    db.log_scrape("news-cycle", n, (time.time() - t0) * 1000, ok=True)
    return new_count, db.stats()["items"]


async def refresh_free_index(s: aiohttp.ClientSession):
    """Learn the currently-live free hosts from EverythingMoe AND self-heal.

    EverythingMoe annotates dead hosts ("… moved to Graveyard"). When a host
    disappears from the live index (its URL changed/moved), we mark it dead and
    try to discover its replacement by name similarity — so the bot keeps working
    even as free-stream URLs churn constantly. All DB writes run in a worker
    thread to avoid blocking the event loop.
    """
    try:
        res = await fetch_everythingmoe_index(s)
        fresh_keys = {(d["slug"], d["kind"]) for d in res}
        loop = asyncio.get_event_loop()

        def _persist(res, fresh_keys):
            for d in res:
                db.upsert_resource(d["slug"], d["name"], d["kind"],
                                   d["page_url"], d["status"], d["note"])
                if d["status"] == "graveyard":
                    db.mark_resource_dead(d["slug"], note=d["note"] or "graveyard")
            for r in db.get_resources():
                key = (r["slug"], r["kind"])
                if key in fresh_keys:
                    if r["status"] != "alive":
                        db.mark_resource_alive(r["slug"])
                    continue
                if r["status"] == "alive":
                    db.mark_resource_dead(r["slug"], note="not in current EverythingMoe index")
                    repl = _find_replacement(r["name"], res)
                    if repl:
                        print(f"[emoe] self-heal: '{r['name']}' dead -> "
                              f"replacement '{repl['name']}' ({repl['page_url']})")

        await loop.run_in_executor(None, _persist, res, fresh_keys)
        alive = db.alive_resource_count()
        print(f"[emoe] index refreshed: {len(res)} listed, {alive} alive hosts")
    except Exception as ex:
        print(f"[emoe] refresh failed: {ex}")


def _find_replacement(name: str, fresh: list[dict]) -> dict | None:
    best, best_score = None, 0.0
    for d in fresh:
        if d["status"] == "graveyard":
            continue
        if d["name"].lower() == name.lower():
            continue
        sc = score_pair(name, d["name"])
        if sc > best_score:
            best_score, best = sc, d
    return best if best and best_score >= 70 else None


# ------------------------------------------------------------------ explore
_explore_cycle = 0
_health_offset = 0
_discover_offset = 0


@tasks.loop(seconds=120)
async def explore_loop():
    """Always-on crawler: continuously rediscover + health-check the web.

    Runs constantly while the bot is up: re-syncs the EverythingMoe index (URLs
    move fast), round-robins liveness checks on known hosts, and rotates through
    news sources so fresh content keeps flowing and dead sources get retried.
    """
    global _explore_cycle, _health_offset, _discover_offset
    try:
        s = await session()
        _explore_cycle += 1

        # 1) continuous EverythingMoe re-sync (catch URL changes quickly)
        if _explore_cycle % 2 == 0:
            await refresh_free_index(s)

        # 2) round-robin health check on a slice of alive hosts (EverythingMoe pages)
        alive = db.get_resources(status="alive")
        batch = 12
        slice_ = alive[_health_offset:_health_offset + batch]
        _health_offset = (_health_offset + batch) % max(1, len(alive))
        for r in slice_:
            try:
                async with s.get(r["page_url"], headers={"User-Agent": config.USER_AGENT},
                                 timeout=10) as resp:
                    if resp.status >= 400:
                        db.mark_resource_dead(r["slug"], note=f"HTTP {resp.status}")
            except Exception:
                db.mark_resource_dead(r["slug"], note="unreachable")

        # 2b) continuously crawl the DIRECT host sites: resolve their real,
        # ever-changing domains and learn each one's search endpoint. This is the
        # "scrape the actual indexed websites" part -- always learning, never
        # just trusting the middleman index.
        need = [r for r in alive if not r.get("domain")]
        pool = need if need else alive
        d_batch = pool[_discover_offset:_discover_offset + 6]
        _discover_offset = (_discover_offset + 6) % max(1, len(pool))
        for r in d_batch:
            dom = await resolve_host_domain(s, r["page_url"])
            if not dom:
                continue
            search, param = await learn_host_search(s, dom)
            search = search or ""
            param = param or ""
            db.update_resource_host(r["slug"], dom, search, param)
            # the real host itself can die -> mark dead so we rediscover later
            try:
                async with s.get(f"https://{dom}", headers={"User-Agent": config.USER_AGENT},
                                 timeout=10) as hr:
                    if hr.status >= 400:
                        db.mark_resource_dead(r["slug"], note=f"host HTTP {hr.status}")
            except Exception:
                db.mark_resource_dead(r["slug"], note="host unreachable")

        # 3) rotate through news sources to keep the feed fresh + retry broken ones
        src = config.RSS_SOURCES[_explore_cycle % len(config.RSS_SOURCES)]
        items = await fetch_rss(s, src)
        new = 0
        for it in items:
            if db.add_item(it):
                new += 1

        st = db.stats()
        await bot.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name=f"crawling web · {st['resources']} hosts · {st['proficiency']}/100",
            )
        )
        if _explore_cycle % 10 == 0:
            print(f"[explore] cycle {_explore_cycle}: +{new} news from {src['name']}, "
                  f"{len(alive)} hosts watched")
    except Exception as ex:
        print(f"[explore_loop] error: {ex}")


@tasks.loop(seconds=config.LEARN_INTERVAL)
async def learn_loop():
    try:
        new_count, total = await do_learn_cycle()
        st = db.stats()
        await bot.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name=f"anime/manga · {st['proficiency']}/100 proficiency",
            )
        )
        if config.OWNER_ID:
            await send_dm(config.OWNER_ID, embed=embeds.header_embed(
                "🧠 Auto-learn complete",
                f"+**{new_count}** new items · **{total}** stored · "
                f"proficiency **{st['proficiency']}/100**"))
    except Exception as ex:
        print(f"[learn_loop] error: {ex}")


@tasks.loop(seconds=config.WATCH_INTERVAL)
async def watch_loop():
    global _last_watch_run
    try:
        recent = db.recent_items(100)
        for uid, key in db.all_follows():
            rec = db.get_title(key)
            if not rec:
                continue
            canon = rec["canonical"]
            matches = match_news(canon, recent, threshold=62)
            for m in matches:
                iid = m.get("id")
                if (key, iid) in _sent_cache:
                    continue
                _sent_cache.add((key, iid))
                await send_dm(int(uid), embed=embeds.news_embed(m, alert=f"{canon} — new related news"))
                if config.NEWS_CHANNEL_ID:
                    ch = bot.get_channel(config.NEWS_CHANNEL_ID)
                    if ch:
                        await ch.send(embed=embeds.news_embed(m, alert=f"{canon} — new related news"))
        _last_watch_run = time.time()
    except Exception as ex:
        print(f"[watch_loop] error: {ex}")


@tasks.loop(seconds=config.BROADCAST_INTERVAL)
async def broadcast_loop():
    """Auto-news (/start): every BROADCAST_INTERVAL, scrape fresh and send the
    newly-ingested stories as a digest to each active target's DM + channel."""
    try:
        targets = db.all_broadcasts()
        if not targets:
            return
        s = await session()
        n, items = await run_news_cycle(s)
        if items:
            db.bulk_add_items(items)
        for t in targets:
            fresh = db.news_since(t["last_sent"], limit=12)
            if not fresh:
                continue
            try:
                await send_dm(int(t["user_id"]),
                              embed=embeds.digest_embed(fresh, "Auto-news digest"))
                ch_id = t["channel_id"]
                if ch_id:
                    ch = bot.get_channel(ch_id)
                    if ch:
                        await ch.send(embed=embeds.digest_embed(fresh, "Auto-news digest"))
                    else:
                        print(f"[broadcast] channel {ch_id} gone for user {t['user_id']}")
            except Exception as ex:
                print(f"[broadcast] failed for {t['user_id']}: {ex}")
            db.touch_broadcast(t["user_id"], time.time())
    except Exception as ex:
        print(f"[broadcast_loop] error: {ex}")


# ------------------------------------------------------------------- events
@bot.event
async def on_ready():
    await tree.sync()
    print(f"AniSage online as {bot.user} (id {bot.user.id})")
    if not learn_loop.is_running():
        learn_loop.start()
    if not watch_loop.is_running():
        watch_loop.start()
    if not explore_loop.is_running():
        explore_loop.start()
    if not broadcast_loop.is_running():
        broadcast_loop.start()
    # Prime the knowledge base on boot WITHOUT blocking on_ready — run it as a
    # background task so the bot stays responsive to commands immediately.
    asyncio.create_task(_boot_learn())


async def _boot_learn():
    try:
        await do_learn_cycle()
    except Exception as ex:
        print(f"[boot] learn failed: {ex}")


@bot.event
async def on_disconnect():
    print("[client] disconnected")


if __name__ == "__main__":
    if not config.DISCORD_TOKEN:
        raise SystemExit("Set DISCORD_TOKEN in .env (see .env.example).")
    bot.run(config.DISCORD_TOKEN)
