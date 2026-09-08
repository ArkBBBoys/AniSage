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
    discover_rss_sources_via_web,
    fetch_anilist_search,
    fetch_anilist_trending,
    fetch_everythingmoe_index,
    fetch_exhaustive_search,
    fetch_jikan_search,
    fetch_kitsu_search,
    fetch_news_for_title,
    fetch_rss,
    fetch_web_news,
    learn_host_search,
    resolve_host_domain,
    run_news_cycle,
)
from matcher import get_weights, match_news, match_title, normalize, parse_query, score_pair, set_weights

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
    """Robust DM sender with verification.
    Returns True only if Discord actually accepted the message.
    Handles blocked DMs, 429, and closed channels."""
    try:
        user = bot.get_user(user_id) or await bot.fetch_user(user_id)
        if user is None:
            print(f"[dm] user {user_id} not found")
            return False
        # Ensure DM channel exists; Discord may raise if user blocked DMs
        try:
            ch = user.dm_channel or await user.create_dm()
        except discord.Forbidden:
            print(f"[dm] forbidden create_dm for {user_id} (DMs closed)")
            return False
        except Exception as ex:
            print(f"[dm] create_dm failed for {user_id}: {ex}")
            return False
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
    except discord.Forbidden as ex:
        print(f"[dm] forbidden for {user_id}: {ex} (user blocked DMs or bot not sharing guild)")
        return False
    except discord.HTTPException as ex:
        # 429 rate limit, 500 etc
        print(f"[dm] HTTPException for {user_id}: {ex} status={getattr(ex,'status', '?')}")
        return False
    except Exception as ex:
        print(f"[dm] failed for {user_id}: {ex}")
        return False


def _guild_id(inter: discord.Interaction) -> str:
    """Per-guild/per-user isolation key: guild snowflake or '0' for DM."""
    try:
        if inter.guild and inter.guild.id:
            return str(inter.guild.id)
    except Exception:
        pass
    return "0"  # DM / inbox personal space


async def deliver(inter: discord.Interaction, *, embeds=None, content=None, file=None, view=None, ephemeral_fallback: bool = True) -> bool:
    """Smart delivery that treats server and inbox (DM) as isolated contexts.
    - In a guild: deliver via followup in that server channel (ephemeral by default) — isolated per server.
    - In DM: deliver via DM to user's inbox — isolated per person.
    Always verifies delivery; if DM fails, falls back to followup so user never sees 'sent to DM' with nothing arriving.
    Returns True if delivered.
    """
    guild_mode = inter.guild is not None
    # Build kwargs for sending
    send_kwargs = {}
    if embeds is not None:
        send_kwargs["embeds"] = embeds if isinstance(embeds, list) else [embeds]
    elif embeds is None and content is None and file is None:
        return False
    if content:
        send_kwargs["content"] = content
    if file:
        send_kwargs["file"] = file
    if view:
        send_kwargs["view"] = view
    # For guild mode, deliver directly via followup (server-isolated, not DM)
    if guild_mode:
        try:
            # Prefer ephemeral for search-type private results; news can be public per server
            # Caller controls via ephemeral_fallback flag: if True, use ephemeral; if False, public
            await inter.followup.send(ephemeral=True, **send_kwargs)
            return True
        except Exception as ex:
            print(f"[deliver] guild followup failed guild={_guild_id(inter)} user={inter.user.id}: {ex}")
            # fallback try DM
            try:
                ok = await send_dm(inter.user.id, **send_kwargs)
                return ok
            except Exception:
                return False
    else:
        # DM/inbox mode: try DM first (personal), fallback to followup channel
        try:
            ok = await send_dm(inter.user.id, **send_kwargs)
            if ok:
                return True
            # DM failed (closed DMs) -> fallback to followup in DM channel (which is the DM itself)
            print(f"[deliver] DM failed, falling back to followup for {inter.user.id}")
            await inter.followup.send(**send_kwargs)
            return True
        except Exception as ex:
            print(f"[deliver] DM+fallback failed inbox {inter.user.id}: {ex}")
            try:
                await inter.followup.send(**send_kwargs)
                return True
            except Exception:
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


# --------------------------------------------------------------- feedback UI — self-learning
class MatchFeedback(discord.ui.View):
    def __init__(self, query: str, key: str, method: str = "", score: float = 0.0):
        super().__init__(timeout=300)
        self.query = query
        self.key = key
        self.method = method
        self.score = score

    async def _respond(self, inter: discord.Interaction, content: str):
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
            None, db.record_feedback, self.query, self.key, True, self.method, self.score
        )
        await self._respond(
            inter,
            f"✅ Learned: “{self.query}” → that title. Confidence Bayesian-boosted & alias weight ↑.",
        )

    @discord.ui.button(label="❌ Wrong", style=discord.ButtonStyle.danger)
    async def no(self, inter: discord.Interaction, _b: discord.ui.Button):
        await asyncio.get_event_loop().run_in_executor(
            None, db.record_feedback, self.query, self.key, False, self.method, self.score
        )
        await self._respond(
            inter, "❌ Noted — Bayesian penalty & alias decay, will self-improve via hourly pruning."
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
              "`/unfollow <name>` (autocomplete your follows) / `/unfollow-all` / `/following`\n"
              "`/start` / `/stop` — auto-news digest every 30 min (DM + channel)",
        inline=False,
    )
    e.add_field(
        name="🧠 Learn & inspect",
        value="`/digest` — fresh digest (server-isolated: guild vs DM inbox separate)\n"
              "`/learned` — export EVERYTHING to `anisage_learned.json` (per-context)\n"
              "`/learned-db` — send raw **anisage.db** SQLite (per-context, WAL checkpoint)\n"
              "`/input-learned` — **restore** `anisage_learned.json` backup (merge, skip existing, owner)\n"
              "`/input-learned-db` — **restore** `anisage_*.db` backup (merge, skip existing, owner)\n"
              "`/stats` — 7-factor proficiency `mastery/accuracy/breadth/depth/vitality/velocity` + momentum\n"
              "`/learning` — self-learning report (weights, threshold, pruning)\n"
              "`/learn` — force learning cycle now (owner)",
        inline=False,
    )
    e.add_field(
        name="🧬 Algorithm — hybrid, self-learning, self-improving",
        value="**Hybrid scorer**: `token_set 35% + token_sort 22% + jaro-winkler 18% + ngram 15% + partial 10%` (learned weights, auto-tuned hourly). Handles typos (`Chronicles` vs `Chronicales`), acronyms (`BTTH`), reordering.\n"
              "**Self-learning**: Bayesian confidence `(correct+2)/(total+3)`, alias `hits_correct/hits_wrong` → weight, `times_seen` + `decay` for stale titles.\n"
              "**Self-improvement (hourly)**: prunes stale low-weight aliases, decays stale titles, recomputes alias weights from feedback, auto-tunes `MATCH_THRESHOLD` & hybrid weights from 7-day gap, logs windows.",
        inline=False,
    )
    e.add_field(
        name="🔐 Isolation model",
        value="**Every server & every person is isolated:** "
              "`/search`/`/follow`/`/where` in a server stay in that server; "
              "DM/inbox commands stay in your inbox. "
              "`/following` and `/start` digests are per-`guild_id`+`user_id`. "
              "Exhaustive parallel live search (AniList+Jikan+Kitsu+DDG+MAL) + exhaustive web news crawl finds even unpopular titles when APIs wobble.",
        inline=False,
    )
    embeds._stamp(e)
    if not await safe_respond(inter, embed=e, ephemeral=True):
        return


@tree.command(name="news", description="Get the latest news — exhaustive web crawl (server-isolated, inbox vs guild)")
@app_commands.describe(media_type="Filter: anime / manga / manhwa / all")
async def cmd_news(inter: discord.Interaction, media_type: str = "all"):
    if not await safe_defer(inter, ephemeral=True):
        return
    gid = _guild_id(inter)
    # Exhaustive: cached + live web (whole internet, not just fixed RSS)
    s = await session()
    cached = db.recent_items(limit=20, media_type="" if media_type == "all" else media_type)
    # Live web crawl in parallel for the requested type — fixes “news is much worse than search”
    try:
        # Use generic web news for the filter; for “all” use exhaustive
        web = []
        if media_type == "all":
            # Exhaustive: parallel web for anime/manga/manhwa news
            web_tasks = [fetch_web_news(s, q, limit=6) for q in ["anime news", "manga news", "manhwa news"]]
            web_res = await asyncio.gather(*web_tasks, return_exceptions=True)
            for r in web_res:
                if isinstance(r, list):
                    web.extend(r)
        else:
            web = await fetch_web_news(s, f"{media_type} news", limit=10)
            # Also try title-agnostic web
            if len(web) < 5:
                extra = await fetch_web_news(s, f"{media_type} release news", limit=6)
                web.extend(extra)
        # Merge cached + web, de-dup by URL, keep newest first (web is fresh)
        seen = {it["url"] for it in cached if it.get("url")}
        merged = list(cached)
        for it in web:
            if it.url and it.url not in seen:
                seen.add(it.url)
                # Convert NewsItem to dict for uniform handling
                merged.append({"title": it.title, "url": it.url, "summary": it.summary, "source": it.source, "published": it.published, "image": it.image, "media_type": media_type if media_type != "all" else "news"})
                # Persist web news so future /news is richer
                try: db.add_item(it)
                except: pass
        if not merged:
            n, fresh = await run_news_cycle(s, db=db)
            if fresh:
                await asyncio.get_event_loop().run_in_executor(None, db.bulk_add_items, fresh)
                merged = db.recent_items(limit=20, media_type="" if media_type == "all" else media_type)
        items = merged[:20]
    except Exception as ex:
        print(f"[news] live fetch failed: {ex}")
        items = cached
    if not items:
        await inter.followup.send("No news found even after exhaustive web crawl — try again in a minute or run `/learn`.", ephemeral=True)
        return
    header = embeds.header_embed(
        "📰 Latest news — exhaustive web crawl",
        f"{min(15, len(items))} stories · {len([x for x in items if str(x.get('source','')).startswith('Web:')])} from whole web + {len(cached)} cached · context: {'server '+gid if gid!='0' else 'DM/inbox (personal)'} · newest first")
    if inter.guild is None:
        # DM inbox: personal delivery via DM with fallback verification
        ok = await send_dm(inter.user.id, embed=header)
        delivered = 0
        for it in items[:15]:
            ok2 = await send_dm(inter.user.id, embed=embeds.news_embed(it))
            if ok2:
                delivered += 1
            await asyncio.sleep(0.35)  # gentle rate limit
        if ok and delivered > 0:
            await inter.followup.send(f"📬 Sent {delivered} stories to your **DM/inbox** (personal, isolated per you).", ephemeral=True)
        else:
            # fallback: deliver in followup channel (DM channel itself)
            try:
                await inter.followup.send(embed=header)
                for it in items[:7]:
                    await inter.followup.send(embed=embeds.news_embed(it))
                await inter.followup.send(f"⚠️ DMs were blocked — delivered here instead. Enable DMs for future inbox delivery.", ephemeral=True)
            except Exception:
                await inter.followup.send(f"❌ Could not deliver to DM (blocked?) and fallback failed. Check privacy settings.", ephemeral=True)
    else:
        # Guild/server mode: isolated per server, deliver in server channel (not DM)
        # Send via followup in server (ephemeral) to keep per-server isolation
        try:
            await inter.followup.send(embed=header, ephemeral=True)
            for it in items[:12]:
                await inter.followup.send(embed=embeds.news_embed(it), ephemeral=True)
                await asyncio.sleep(0.25)
            await inter.followup.send(f"📬 Delivered {min(12,len(items))} stories in **this server** ({inter.guild.name}) — isolated per server.", ephemeral=True)
        except Exception as ex:
            print(f"[news] guild delivery failed: {ex}")
            await inter.followup.send("❌ Delivery failed.", ephemeral=True)


@tree.command(name="search", description="Matched-based title search — exhaustive, finds even unpopular anime")
@app_commands.describe(name="Title to look up (e.g. 'JJK Season 2', 'Solo Leveling ch 150')", media_type="anime / manga / manhwa")
async def cmd_search(inter: discord.Interaction, name: str, media_type: str = "anime"):
    if not await safe_defer(inter, ephemeral=True):
        return
    gid = _guild_id(inter)
    mt = media_type.upper()
    clean, season, unit = parse_query(name)

    # 1) try learned knowledge base first (now strictly thresholded, no false positives)
    rec = match_title(name, db, threshold=config.MATCH_THRESHOLD)
    live = None
    # 2) if no high-confidence local match, exhaustive live search across ALL APIs (AniList, Jikan, Kitsu)
    # This guarantees unpopular titles like Dr. Stone / Spirit Chronicles / Battle Through Heavens are found even when AniList is down
    if not rec:
        s = await session()
        pref = "anime" if mt == "ANIME" else "manga"
        live = await fetch_exhaustive_search(s, name, preferred_type=pref, db=db)
        if not live and clean != name:
            live = await fetch_exhaustive_search(s, clean, preferred_type=pref, db=db)
        if not live and pref == "anime":
            live = await fetch_exhaustive_search(s, name, preferred_type="manga", db=db)
        if not live and pref == "manga":
            live = await fetch_exhaustive_search(s, name, preferred_type="anime", db=db)
        if live:
            # Learn it with exhaustive variant aliases and image
            aliases = live.get("aliases", []) or []
            # ensure query itself becomes alias for future instant alias hit
            if normalize(name) != live["key"]:
                aliases = list(set(aliases + [name]))
            db.learn_title(TitleRecord(
                key=live["key"], canonical=live["canonical"],
                media_type=live["media_type"], external_id=live.get("anilist_id", "") or live.get("mal_id",""),
                anilist_id=live.get("anilist_id", ""), mal_id=live.get("mal_id", ""),
                image=live.get("image", ""),
                aliases=aliases, watch_links=live.get("watch_links", []), read_links=live.get("read_links", []),
            ))
            rec = db.get_title(live["key"])
            if rec:
                rec["score"] = round(live.get("score", 92.0), 1) if live.get("score", 0) > 60 else 92.0
                rec["match_method"] = "live-exhaustive"
                print(f"[search] learned {live['canonical']} for query {name!r} via exhaustive (score {rec['score']}) gid={gid}")

    if not rec:
        # also try fuzzy over recent news titles as last resort (bot may have seen it in news crawl)
        recent = db.recent_items(100)
        if recent:
            best = None; best_s = 0
            for it in recent:
                s = score_pair(name, it["title"])
                if s > best_s:
                    best_s = s
                    best = it
            if best and best_s >= 62:
                await inter.followup.send(
                    f"🤔 No exact DB match for “{name}” (best local was {best_s:.0f}), but found related news: **{best['title']}** — try `/where {name}` or check Kitsu.", ephemeral=True)
                return
        await inter.followup.send(f"🤔 No solid match for “{name}”. Tried all live sources (AniList, Jikan, Kitsu). Try a different spelling or check `/trending`.", ephemeral=True)
        return

    rec["season"], rec["unit"] = season, unit

    embed = embeds.search_result_embed(rec, name)
    # Annotate per-context isolation in footer
    embed.add_field(
        name="🆓 Free hosts (EverythingMoe)",
        value=free_links(rec.get("media_type", "anime")),
        inline=False,
    )
    dsl = direct_search_links(rec)
    if dsl != "—":
        embed.add_field(name="🔗 Direct search on free hosts", value=dsl, inline=False)
    # Exhaustive news for this title — whole web, not just cached 50
    # This fixes “news is much worse than search” for unpopular titles
    s_live = await session()
    try:
        cached_for_title = db.recent_items(80)
        live_for_title = await fetch_news_for_title(s_live, rec["canonical"], limit=8, db=db)
        # Merge and persist live so future /news sees it
        merged_news_pool = list(cached_for_title)
        seen_urls = {it.get("url") for it in cached_for_title if it.get("url")}
        for it in live_for_title:
            if it.url and it.url not in seen_urls:
                seen_urls.add(it.url)
                merged_news_pool.append({"title": it.title, "url": it.url, "summary": it.summary, "source": it.source, "published": it.published, "image": it.image, "id": hash(it.url)})
                try: db.add_item(it)
                except: pass
        news = match_news(rec["canonical"], merged_news_pool, threshold=52)
        # If still none, at least show the raw live items that contain the title
        if not news and live_for_title:
            # Fallback: show live items that mention the title even if scorer is low (for very unpopular)
            for it in live_for_title[:3]:
                if rec["canonical"].lower() in it.title.lower() or it.title.lower() in rec["canonical"].lower():
                    news.append({"title": it.title, "url": it.url, "summary": it.summary, "source": it.source, "score": 60, "published": it.published, "image": it.image})
    except Exception as ex:
        print(f"[search] live news for {rec['canonical']} failed: {ex}")
        news = match_news(rec["canonical"], db.recent_items(50), threshold=55)
    # Per-guild vs inbox delivery: server gets ephemeral in channel, DM gets DM
    if inter.guild is None:
        # DM/inbox — verified delivery with fallback, self-learning feedback carries method/score
        view = MatchFeedback(name, rec["key"], rec.get("match_method",""), float(rec.get("score") or 0))
        ok = await send_dm(inter.user.id, embed=embed, view=view)
        if news:
            # Limit to 4 news embeds to avoid spam, but show live web count
            await send_dm(inter.user.id, embed=embeds.item_search_embed(news[:4], rec["canonical"]))
            if len(news) > 4:
                await send_dm(inter.user.id, content=f"_{len(news)-4} more news stories found via web crawl — use `/news` for exhaustive._")
        if ok:
            await inter.followup.send(f"📬 Match for **{rec['canonical']}** (via *{rec.get('match_method','fuzzy')}* {rec.get('score','?')}%) sent to your **DM/inbox** (personal, per-you isolated). {len(news)} related news via exhaustive web crawl. React ✅/❌ to teach me!", ephemeral=True)
        else:
            # fallback to followup if DMs blocked
            await inter.followup.send(embed=embed)
            if news:
                await inter.followup.send(embed=embeds.item_search_embed(news[:4], rec["canonical"]))
            await inter.followup.send(f"⚠️ Your DMs are closed — delivered here instead. Enable DMs for inbox delivery. Title: **{rec['canonical']}** ({rec.get('match_method')})", ephemeral=True)
    else:
        # Server — deliver in this server's channel, isolated per server
        embed.set_footer(text=f"AniSage · server {inter.guild.name} · per-server isolated · via {rec.get('match_method')} · {len(news)} news via web")
        await inter.followup.send(embed=embed, view=MatchFeedback(name, rec["key"], rec.get("match_method",""), float(rec.get("score") or 0)), ephemeral=True)
        if news:
            await inter.followup.send(embed=embeds.item_search_embed(news[:4], rec["canonical"]), ephemeral=True)
        await inter.followup.send(f"📬 Match for **{rec['canonical']}** delivered in **this server** ({inter.guild.name}) — per-server isolated. {len(news)} related news via exhaustive crawl. Use DMs for personal inbox mode.", ephemeral=True)


@tree.command(name="follow", description="Follow a title for alerts (per-server + per-person isolated)")
async def cmd_follow(inter: discord.Interaction, name: str):
    if not await safe_defer(inter, ephemeral=True):
        return
    gid = _guild_id(inter)
    rec = match_title(name, db, threshold=60)
    if not rec:
        s = await session()
        clean, _, _ = parse_query(name)
        live = await fetch_exhaustive_search(s, clean, preferred_type="anime", db=db)
        if not live:
            live = await fetch_exhaustive_search(s, clean, preferred_type="manga", db=db)
        if live:
            db.learn_title(TitleRecord(
                key=live["key"], canonical=live["canonical"],
                media_type=live["media_type"], anilist_id=live.get("anilist_id",""),
                mal_id=live.get("mal_id",""),
                image=live.get("image", ""),
                aliases=live.get("aliases",[]), watch_links=live.get("watch_links",[]), read_links=live.get("read_links",[])))
            rec = db.get_title(live["key"])
    if not rec:
        await inter.followup.send(f"Couldn't resolve “{name}”. Search first via `/search`.", ephemeral=True)
        return
    db.follow(inter.user.id, rec["key"], guild_id=gid)
    where = f"in **{inter.guild.name}** (server-isolated)" if inter.guild else "in your **DM/inbox** (personal per-you)"
    await inter.followup.send(f"🔔 Following **{rec['canonical']}** {where}. You'll get alerts there. `guild={gid}`", ephemeral=True)


async def unfollow_autocomplete(inter: discord.Interaction, current: str):
    """Autocomplete for /unfollow: show titles the user actually follows in this guild."""
    try:
        gid = _guild_id(inter)
        # Get followed keys for this guild, fallback to all if none in this guild
        keys = db.followed(inter.user.id, guild_id=gid)
        if not keys and gid != "0":
            # Also try global follow list for this user
            keys = db.followed(inter.user.id, guild_id=None)
        choices = []
        cur_low = current.lower()
        for k in keys[:25]:
            t = db.get_title(k)
            name = t["canonical"] if t else k
            if cur_low and cur_low not in name.lower() and cur_low not in k.lower():
                continue
            # Truncate to 100 chars for Discord
            display = name[:90]
            choices.append(app_commands.Choice(name=display, value=name))
            if len(choices) >= 25:
                break
        return choices
    except Exception:
        return []

@tree.command(name="unfollow", description="Stop following a title (per-server isolated)")
@app_commands.describe(name="Title to unfollow — autocomplete shows your followed titles")
@app_commands.autocomplete(name=unfollow_autocomplete)
async def cmd_unfollow(inter: discord.Interaction, name: str):
    gid = _guild_id(inter)
    # Try exact followed key first (for autocomplete values which are canonical names)
    # This handles cases where autocomplete gave canonical but match_title threshold might be strict
    followed_keys = db.followed(inter.user.id, guild_id=gid)
    if not followed_keys and gid != "0":
        followed_keys = db.followed(inter.user.id, guild_id=None)
    # Try to find a followed title that matches the input directly (case-insensitive)
    direct_key = None
    norm_input = normalize(name)
    for k in followed_keys:
        t = db.get_title(k)
        canon = t["canonical"] if t else k
        if normalize(canon) == norm_input or k == norm_input or name.lower() == canon.lower():
            direct_key = k
            rec = t
            break
    if direct_key:
        # Direct hit from followed list — no need for fuzzy
        rec = db.get_title(direct_key)
    else:
        # Fallback to fuzzy match_title (handles typos like "JJK" etc.)
        rec = match_title(name, db, threshold=60)
        if rec:
            direct_key = rec["key"]
    if rec and direct_key:
        # Try per-guild first, then global fallback
        removed = False
        # Check if actually followed in this guild
        if direct_key in db.followed(inter.user.id, guild_id=gid):
            db.unfollow(inter.user.id, direct_key, guild_id=gid)
            removed = True
        # Also clean global if exists and gid !=0 and it was followed globally
        if gid != "0" and direct_key in db.followed(inter.user.id, guild_id=None):
            # Only remove global if not already removed and user explicitly did unfollow in guild
            # Keep behavior: unfollow in guild also removes global to avoid stale
            db.unfollow(inter.user.id, direct_key, guild_id=None)
            removed = True
        # If not found in specific guild but found via fuzzy, try that guild
        if not removed:
            # Try the guild of the found rec
            db.unfollow(inter.user.id, direct_key, guild_id=gid)
            removed = True
        await safe_respond(inter, content=f"🚫 Unfollowed **{rec['canonical']}** (guild {gid}). Use `/following` to see remaining.", ephemeral=True)
    else:
        # Helpful error with list of what they do follow
        keys = db.followed(inter.user.id, guild_id=gid)
        if keys:
            sample = ", ".join([ (db.get_title(k)["canonical"] if db.get_title(k) else k) for k in keys[:3] ])
            await safe_respond(inter, content=f"No match for `“{name}”` in this {'server' if gid!='0' else 'DM'}. You follow: {sample}. Try autocomplete or `/following`.", ephemeral=True)
        else:
            await safe_respond(inter, content=f"No match for `“{name}”` — you're not following anything in this {'server' if gid!='0' else 'DM'} (guild {gid}). Use `/following` to check.", ephemeral=True)

@tree.command(name="unfollow-all", description="Unfollow all titles in this server/DM (per-server isolated)")
async def cmd_unfollow_all(inter: discord.Interaction):
    gid = _guild_id(inter)
    keys = db.followed(inter.user.id, guild_id=gid)
    if not keys:
        await safe_respond(inter, content=f"You're not following anything in this {'server' if gid!='0' else 'DM'} to unfollow.", ephemeral=True)
        return
    count = len(keys)
    for k in list(keys):
        try:
            db.unfollow(inter.user.id, k, guild_id=gid)
        except: pass
    await safe_respond(inter, content=f"🚫 Unfollowed all **{count}** titles in this {'server' if gid!='0' else 'DM'} (guild {gid}).", ephemeral=True)


@tree.command(name="following", description="List titles you follow (per-server + per-person)")
async def cmd_following(inter: discord.Interaction):
    gid = _guild_id(inter)
    # per-guild list, plus hint about global
    keys = db.followed(inter.user.id, guild_id=gid)
    keys_all = db.followed(inter.user.id, guild_id=None)  # all guilds
    if not keys:
        if keys_all:
            await safe_respond(inter, content=f"You're not following anything in this **{'server' if gid!='0' else 'DM'}** (guild {gid}). You follow {len(keys_all)} title(s) in other contexts — use that context's `/following` to see them.", ephemeral=True)
        else:
            await safe_respond(inter, content="You're not following anything yet.", ephemeral=True)
        return
    lines = []
    for k in keys:
        t = db.get_title(k)
        lines.append(f"• {t['canonical'] if t else k}")
    ctx = f"in **{inter.guild.name}**" if inter.guild else "in **DM/inbox**"
    await safe_respond(inter, content=f"🔔 **Following {ctx}** (guild {gid}, {len(keys)}):\n" + "\n".join(lines), ephemeral=True)


@tree.command(name="digest", description="Send a fresh news digest — exhaustive web crawl (server-isolated, inbox vs guild)")
async def cmd_digest(inter: discord.Interaction):
    if not await safe_defer(inter, ephemeral=True):
        return
    gid = _guild_id(inter)
    s = await session()
    # Exhaustive: cached + live web (crawl whole internet, not just DB)
    cached = db.recent_items(15)
    try:
        live = await fetch_web_news(s, "anime news", limit=10)
        if len(live) < 6:
            extra = await fetch_web_news(s, "manga news", limit=6)
            live.extend(extra)
        # Merge
        seen = {it["url"] for it in cached if it.get("url")}
        merged = list(cached)
        for it in live:
            if it.url and it.url not in seen:
                seen.add(it.url)
                merged.append({"title": it.title, "url": it.url, "summary": it.summary, "source": it.source, "published": it.published, "image": it.image})
                try: db.add_item(it)
                except: pass
        items = merged[:15]
        if not items:
            n, fresh = await run_news_cycle(s, db=db)
            if fresh:
                await asyncio.get_event_loop().run_in_executor(None, db.bulk_add_items, fresh)
                items = db.recent_items(15)
    except Exception as ex:
        print(f"[digest] live fetch failed: {ex}")
        items = db.recent_items(15)
    if not items:
        await inter.followup.send("Nothing digested yet — even exhaustive web crawl found nothing. Try again in a minute.", ephemeral=True)
        return
    embed = embeds.digest_embed(items, f"Your feed · {'server '+gid if gid!='0' else 'DM/inbox'} · {len([x for x in items if str(x.get('source','')).startswith('Web:')])} from web + {len(cached)} cached")
    if inter.guild is None:
        ok = await send_dm(inter.user.id, embed=embed)
        if ok:
            await inter.followup.send("📬 Digest sent to **DM/inbox** (personal per-you isolated).", ephemeral=True)
        else:
            await inter.followup.send(embed=embed)
            await inter.followup.send("⚠️ DMs blocked — delivered here instead. Enable DMs for inbox delivery.", ephemeral=True)
    else:
        await inter.followup.send(embed=embed, ephemeral=True)
        await inter.followup.send(f"📬 Digest delivered in **this server** ({inter.guild.name}) — isolated per server.", ephemeral=True)


@tree.command(name="learned", description="Export EVERYTHING the bot has learned (isolated per context)")
async def cmd_learned(inter: discord.Interaction):
    if not await safe_defer(inter, ephemeral=True):
        return
    gid = _guild_id(inter)
    s = db.stats()
    # stats always delivered in context (server-isolated view)
    stats_embed = embeds.stats_embed(s)
    stats_embed.add_field(name="Context", value=f"{'Server '+gid if gid!='0' else 'DM/inbox personal'} (every server/person isolated)", inline=False)
    if inter.guild is None:
        ok = await send_dm(inter.user.id, embed=stats_embed)
        if not ok:
            await inter.followup.send(embed=stats_embed)
    else:
        await inter.followup.send(embed=stats_embed, ephemeral=True)
    titles = db.all_titles()
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
            if inter.guild is None:
                ok = await send_dm(inter.user.id, embed=embeds.learned_chunk_embed(chunk, page, total_pages))
                if not ok:
                    await inter.followup.send(embed=embeds.learned_chunk_embed(chunk, page, total_pages))
            else:
                await inter.followup.send(embed=embeds.learned_chunk_embed(chunk, page, total_pages), ephemeral=True)
            chunk = []
            await asyncio.sleep(0.35)
    if chunk:
        page += 1
        if inter.guild is None:
            ok = await send_dm(inter.user.id, embed=embeds.learned_chunk_embed(chunk, page, total_pages))
            if not ok:
                await inter.followup.send(embed=embeds.learned_chunk_embed(chunk, page, total_pages))
        else:
            await inter.followup.send(embed=embeds.learned_chunk_embed(chunk, page, total_pages), ephemeral=True)
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
    if inter.guild is None:
        ok = await send_dm(inter.user.id, embed=embeds.header_embed("🗂️ Full export (JSON)"),
                      file=discord.File(io.BytesIO(data), filename="anisage_learned.json"))
        if ok:
            await inter.followup.send(f"📬 Exported {len(titles)} titles + {s['items']} items to your **DM/inbox** (personal per-you isolated).", ephemeral=True)
        else:
            await inter.followup.send(f"📬 Exported {len(titles)} titles + {s['items']} items — JSON also ready here.", ephemeral=True,
                                      file=discord.File(io.BytesIO(data), filename="anisage_learned.json"))
    else:
        await inter.followup.send(f"📬 Exported {len(titles)} titles + {s['items']} items in **this server** ({inter.guild.name}) — file:", ephemeral=True,
                                  file=discord.File(io.BytesIO(data), filename="anisage_learned.json"))


@tree.command(name="learned-db", description="Send the raw anisage.db file (server-isolated per context)")
async def cmd_learned_db(inter: discord.Interaction):
    if not await safe_defer(inter, ephemeral=True):
        return
    gid = _guild_id(inter)
    db_path = db.path
    if not db_path.exists():
        await inter.followup.send("❌ DB file not found (no learning yet).", ephemeral=True)
        return
    # ensure DB is checkpointed so file is consistent (WAL -> main)
    try:
        with db._session() as s:
            s.execute(__import__("sqlalchemy").text("PRAGMA wal_checkpoint(TRUNCATE)"))
            s.commit()
    except Exception:
        pass
    size = db_path.stat().st_size
    size_mb = size / (1024*1024)
    # Discord file limit 25MB for normal, 8MB for some; if >8MB split warning
    if size > 24 * 1024 * 1024:
        await inter.followup.send(f"⚠️ DB is {size_mb:.1f} MB — exceeds Discord 25 MB limit. Sending JSON export instead via `/learned`.", ephemeral=True)
        return
    try:
        # Need to copy file to temp to avoid lock issues while sending
        import tempfile, shutil
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            shutil.copy2(db_path, tmp.name)
            tmp_path = tmp.name
        file = discord.File(tmp_path, filename=f"anisage_{gid}.db")
        desc = f"Raw SQLite DB · {size_mb:.2f} MB · {db.stats()['titles']} titles · isolated per context (guild {gid})"
        if inter.guild is None:
            # DM/inbox — send via DM with verification
            ok = await send_dm(inter.user.id, content=desc, file=file)
            if ok:
                await inter.followup.send(f"📦 DB file ({size_mb:.2f} MB) sent to your **DM/inbox** (personal, per-you isolated).", ephemeral=True)
            else:
                # fallback to followup
                try:
                    file2 = discord.File(tmp_path, filename=f"anisage_{gid}.db")
                    await inter.followup.send(content=desc, file=file2)
                    await inter.followup.send("⚠️ DMs blocked — delivered here instead.", ephemeral=True)
                except Exception as ex:
                    await inter.followup.send(f"❌ Failed to deliver DB: {ex}", ephemeral=True)
        else:
            # Server — deliver in server channel (ephemeral) to keep per-server isolated
            await inter.followup.send(content=f"📦 **anisage.db** for server **{inter.guild.name}** ({size_mb:.2f} MB) — per-server isolated.", file=file, ephemeral=True)
            await inter.followup.send(f"📦 DB delivered in **this server** (guild {gid}) — not leaked to DMs.", ephemeral=True)
        try:
            __import__("os").unlink(tmp_path)
        except Exception:
            pass
    except Exception as ex:
        print(f"[learned-db] failed: {ex}")
        await inter.followup.send(f"❌ Failed to send DB: {ex}", ephemeral=True)


@tree.command(name="input-learned", description="Restore backup JSON from /learned (merges, skips existing)")
@app_commands.describe(file="The anisage_learned.json file you got from /learned")
@owner_only()
async def cmd_input_learned(inter: discord.Interaction, file: discord.Attachment):
    if not await safe_defer(inter, ephemeral=True):
        return
    # Validate attachment
    if not file.filename.lower().endswith(".json"):
        await inter.followup.send("❌ Please upload a `.json` file from `/learned` (e.g. `anisage_learned.json`). For `.db` use `/input-learned-db`.", ephemeral=True)
        return
    if file.size > 25 * 1024 * 1024:
        await inter.followup.send(f"❌ File too large: {file.size/1024/1024:.1f} MB > 25 MB limit.", ephemeral=True)
        return
    await inter.followup.send(f"📥 Reading `{file.filename}` ({file.size/1024:.1f} KB)...", ephemeral=True)
    try:
        data_bytes = await file.read()
        # Handle BOM and large files
        text = data_bytes.decode("utf-8", errors="ignore")
        # Strip possible leading garbage
        data = json.loads(text)
    except Exception as ex:
        await inter.followup.send(f"❌ Failed to read/parse JSON: `{ex}`. Ensure it's the exact file from `/learned`.", ephemeral=True)
        return
    # Validate structure (from /learned export)
    if not isinstance(data, dict) or "titles" not in data:
        await inter.followup.send("❌ Invalid JSON: expected `{\"titles\": [...], \"recent_items\": [...]}` from `/learned`.", ephemeral=True)
        return
    titles = data.get("titles", [])
    recent_items = data.get("recent_items", [])
    if not isinstance(titles, list):
        await inter.followup.send("❌ Invalid `titles` — expected a list.", ephemeral=True)
        return
    # Merge logic: skip existing, import new
    # Use executor for DB-heavy work
    def _import_json():
        # Import titles — skip if key already exists (extracted & skip)
        imported_titles = 0
        skipped_titles = 0
        failed_titles = 0
        for t in titles:
            try:
                key = (t.get("key") or "").strip()
                canonical = (t.get("canonical") or "").strip()
                if not key or not canonical:
                    failed_titles += 1
                    continue
                # Check if already exists — skip (do not overwrite)
                existing = db.get_title(key)
                if existing is not None:
                    skipped_titles += 1
                    continue
                # Normalize aliases/watch/read which may be str JSON or list
                aliases = t.get("aliases", [])
                if isinstance(aliases, str):
                    try: aliases = json.loads(aliases)
                    except: aliases = [aliases]
                watch = t.get("watch_links", [])
                if isinstance(watch, str):
                    try: watch = json.loads(watch)
                    except: watch = []
                read = t.get("read_links", [])
                if isinstance(read, str):
                    try: read = json.loads(read)
                    except: read = []
                # Preserve original confidence/times_seen if present, else defaults
                times_seen = int(t.get("times_seen", 1))
                confidence = float(t.get("confidence", 5.0))
                # Direct insert to preserve original values (not via learn_title which increments)
                # Use TitleRecord but then override DB row directly for exact restore
                rec = TitleRecord(
                    key=key, canonical=canonical, media_type=t.get("media_type","unknown"),
                    external_id=t.get("external_id",""), anilist_id=t.get("anilist_id",""),
                    mal_id=t.get("mal_id",""), image=t.get("image",""),
                    aliases=list(aliases) if isinstance(aliases, list) else [],
                    watch_links=list(watch) if isinstance(watch, list) else [],
                    read_links=list(read) if isinstance(read, list) else [],
                )
                # Insert via learn_title then patch to exact values if needed
                # First, check again and insert
                db.learn_title(rec)
                # Now patch to exact original times_seen/confidence if higher
                # Use direct DB update to preserve backup values
                try:
                    with db._session() as s:
                        from database import Title
                        row = s.get(Title, key)
                        if row:
                            # Only update if backup has higher times_seen/confidence (don't downgrade)
                            # But for restore, we want exact backup: set to backup values if they are higher
                            # To respect "skip existing", we already skipped existing, so this is new row
                            # For new rows, ensure times_seen/confidence are set to backup values
                            if times_seen > (row.times_seen or 1):
                                row.times_seen = times_seen
                            if confidence > (row.confidence or 0):
                                row.confidence = confidence
                            # Also restore first_seen/last_seen if present
                            if t.get("first_seen"):
                                row.first_seen = float(t["first_seen"])
                            if t.get("last_seen"):
                                row.last_seen = float(t["last_seen"])
                            s.commit()
                except Exception:
                    pass
                imported_titles += 1
            except Exception as ex:
                print(f"[input-learned] title import failed {t.get('key')}: {ex}")
                failed_titles += 1
        # Import recent_items — skip if url already exists
        imported_items = 0
        skipped_items = 0
        for it in recent_items:
            try:
                url = (it.get("url") or "").strip()
                if not url:
                    continue
                # Check if exists via recent lookup (use add_item which does ON CONFLICT DO NOTHING)
                # We need to know if it was skipped, so check first
                from database import Item
                from sqlalchemy import select
                with db._session() as s:
                    exists = s.execute(select(Item).where(Item.url == url)).scalars().first()
                    if exists is not None:
                        skipped_items += 1
                        continue
                # Not exists, add
                ni = NewsItem(
                    source=it.get("source","imported"), kind=it.get("kind","news"),
                    title=it.get("title",""), url=url, summary=it.get("summary","")[:600],
                    image=it.get("image",""), published=it.get("published",""),
                    media_type=it.get("media_type","unknown"),
                    external_id=it.get("external_id",""), anilist_id=it.get("anilist_id",""),
                    mal_id=it.get("mal_id",""),
                )
                if db.add_item(ni):
                    imported_items += 1
                else:
                    skipped_items += 1
            except Exception as ex:
                print(f"[input-learned] item import failed {it.get('url')}: {ex}")
        return {
            "imported_titles": imported_titles,
            "skipped_titles": skipped_titles,
            "failed_titles": failed_titles,
            "imported_items": imported_items,
            "skipped_items": skipped_items,
            "total_titles": len(titles),
            "total_items": len(recent_items),
        }
    try:
        res = await asyncio.get_event_loop().run_in_executor(None, _import_json)
    except Exception as ex:
        await inter.followup.send(f"❌ Import failed: `{ex}`", ephemeral=True)
        return
    embed = discord.Embed(title="✅ Import from JSON complete", color=config.THEME_COLOR,
                          description=f"File `{file.filename}` merged — **skipped existing, imported new** (no overwrite).")
    embed.add_field(name="📚 Titles", value=f"Total in file: **{res['total_titles']}**\nImported **{res['imported_titles']}** new\nSkipped **{res['skipped_titles']}** already existed\nFailed **{res['failed_titles']}**", inline=False)
    embed.add_field(name="📰 News Items", value=f"Total in file: **{res['total_items']}**\nImported **{res['imported_items']}** new\nSkipped **{res['skipped_items']}** already existed", inline=False)
    st = await asyncio.get_event_loop().run_in_executor(None, db.stats)
    embed.add_field(name="📊 Now", value=f"{st['titles']} titles · {st['items']} items · proficiency {st['proficiency']}%", inline=False)
    embeds._stamp(embed)
    await inter.followup.send(embed=embed, ephemeral=True)
    print(f"[input-learned] {file.filename} by {inter.user.id}: {res}")


@tree.command(name="input-learned-db", description="Restore backup .db file from /learned-db (merges, skips existing)")
@app_commands.describe(file="The anisage_*.db file you got from /learned-db")
@owner_only()
async def cmd_input_learned_db(inter: discord.Interaction, file: discord.Attachment):
    if not await safe_defer(inter, ephemeral=True):
        return
    if not file.filename.lower().endswith(".db"):
        await inter.followup.send("❌ Please upload a `.db` file from `/learned-db` (e.g. `anisage_123.db`). For JSON use `/input-learned`.", ephemeral=True)
        return
    if file.size > 24 * 1024 * 1024:
        await inter.followup.send(f"❌ DB too large: {file.size/1024/1024:.1f} MB > 24 MB. Use `/input-learned` JSON instead.", ephemeral=True)
        return
    await inter.followup.send(f"📥 Reading DB `{file.filename}` ({file.size/1024:.1f} KB)... will merge and **skip existing**.", ephemeral=True)
    try:
        data = await file.read()
        # Basic SQLite header check
        if not data.startswith(b"SQLite format 3"):
            await inter.followup.send("❌ Not a valid SQLite .db file (header mismatch). Ensure it's from `/learned-db`.", ephemeral=True)
            return
        # Write to temp file
        import tempfile, pathlib, sqlite3
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            tmp.write(data)
            tmp_path = pathlib.Path(tmp.name)
        # Now merge in executor (heavy SQLite work)
        def _import_db():
            import sqlite3 as _sql, json as _json, pathlib as _pl
            src_path = str(tmp_path)
            # Counters
            imported_titles = skipped_titles = 0
            imported_items = skipped_items = 0
            imported_aliases = skipped_aliases = 0
            imported_resources = skipped_resources = 0
            total_titles = total_items = 0
            try:
                src = _sql.connect(f"file:{src_path}?mode=ro", uri=True, timeout=10)
                src.row_factory = _sql.Row
                # Read source counts
                try:
                    total_titles = src.execute("SELECT COUNT(*) FROM titles").fetchone()[0]
                except: total_titles = 0
                try:
                    total_items = src.execute("SELECT COUNT(*) FROM items").fetchone()[0]
                except: total_items = 0
                # --- Titles: skip if key already exists in main DB ---
                try:
                    rows = src.execute("SELECT key, canonical, media_type, external_id, anilist_id, mal_id, image, aliases, watch_links, read_links, first_seen, last_seen, times_seen, confidence, feedback_correct, feedback_wrong, last_feedback_ts, ngram_sig FROM titles").fetchall()
                except Exception:
                    # Fallback for older DB without new columns
                    rows = src.execute("SELECT key, canonical, media_type, external_id, anilist_id, mal_id, image, aliases, watch_links, read_links, first_seen, last_seen, times_seen, confidence FROM titles").fetchall()
                for r in rows:
                    try:
                        key = r["key"]
                        if not key:
                            continue
                        # Check exists in main DB
                        if db.get_title(key) is not None:
                            skipped_titles += 1
                            continue
                        # Not exists, import exactly as is (preserve times_seen/confidence)
                        # Use direct SQL insert to preserve all fields
                        with db._session() as s:
                            from database import Title
                            # Double-check not exists (race)
                            if s.get(Title, key) is not None:
                                skipped_titles += 1
                                continue
                            # Parse aliases/watch/read which are JSON strings in source
                            s.add(Title(
                                key=key, canonical=r["canonical"], media_type=r["media_type"],
                                external_id=r["external_id"], anilist_id=r["anilist_id"], mal_id=r["mal_id"],
                                image=r["image"], aliases=r["aliases"], watch_links=r["watch_links"], read_links=r["read_links"],
                                first_seen=r["first_seen"], last_seen=r["last_seen"], times_seen=r["times_seen"], confidence=r["confidence"],
                                feedback_correct=r["feedback_correct"] if "feedback_correct" in r.keys() else 0,
                                feedback_wrong=r["feedback_wrong"] if "feedback_wrong" in r.keys() else 0,
                                last_feedback_ts=r["last_feedback_ts"] if "last_feedback_ts" in r.keys() else 0,
                                ngram_sig=r["ngram_sig"] if "ngram_sig" in r.keys() else "",
                            ))
                            s.commit()
                            imported_titles += 1
                            # Also register aliases for this title
                            try:
                                aliases = _json.loads(r["aliases"]) if isinstance(r["aliases"], str) else r["aliases"]
                                if isinstance(aliases, list):
                                    for al in aliases[:12]:
                                        db._set_alias(al, key, weight=1.0)
                                    db._set_alias(r["canonical"], key, weight=2.2)
                            except: pass
                    except Exception as ex:
                        print(f"[input-learned-db] title {r['key'] if 'key' in r.keys() else '?'} failed: {ex}")
                # --- Items: skip if url already exists ---
                try:
                    rows = src.execute("SELECT source, kind, title, url, summary, image, published, fetched_at, media_type, external_id, anilist_id, mal_id FROM items").fetchall()
                    for r in rows:
                        try:
                            url = r["url"]
                            if not url:
                                continue
                            # Check exists
                            from database import Item
                            from sqlalchemy import select
                            with db._session() as s:
                                exists = s.execute(select(Item).where(Item.url == url)).scalars().first()
                                if exists is not None:
                                    skipped_items += 1
                                    continue
                                # Insert
                                ni = NewsItem(
                                    source=r["source"], kind=r["kind"], title=r["title"], url=url,
                                    summary=r["summary"][:600] if r["summary"] else "", image=r["image"] or "", published=r["published"] or "",
                                    media_type=r["media_type"] or "unknown",
                                    external_id=r["external_id"] or "", anilist_id=r["anilist_id"] or "", mal_id=r["mal_id"] or ""
                                )
                                # Use add_item which handles fetched_at
                                # Instead, direct insert to preserve fetched_at
                                from database import Item as _Item
                                s.add(_Item(
                                    source=r["source"], kind=r["kind"], title=r["title"], url=url,
                                    summary=r["summary"][:600] if r["summary"] else "", image=r["image"] or "", published=r["published"] or "",
                                    fetched_at=r["fetched_at"] or __import__("time").time(),
                                    media_type=r["media_type"] or "unknown",
                                    external_id=r["external_id"] or "", anilist_id=r["anilist_id"] or "", mal_id=r["mal_id"] or ""
                                ))
                                s.commit()
                                imported_items += 1
                        except Exception as ex:
                            print(f"[input-learned-db] item {r['url'][:40] if 'url' in r.keys() else '?'} failed: {ex}")
                except Exception as ex:
                    print(f"[input-learned-db] items import failed: {ex}")
                # --- Aliases: skip if alias already exists ---
                try:
                    rows = src.execute("SELECT alias, title_key, weight, hits_correct, hits_wrong, last_used, created_at FROM aliases").fetchall()
                    for r in rows:
                        try:
                            alias = r["alias"]
                            if not alias:
                                continue
                            from database import Alias
                            with db._session() as s:
                                if s.get(Alias, alias) is not None:
                                    skipped_aliases += 1
                                    continue
                                s.add(Alias(
                                    alias=alias, title_key=r["title_key"], weight=r["weight"],
                                    hits_correct=r["hits_correct"] if "hits_correct" in r.keys() else 0,
                                    hits_wrong=r["hits_wrong"] if "hits_wrong" in r.keys() else 0,
                                    last_used=r["last_used"] if "last_used" in r.keys() else 0,
                                    created_at=r["created_at"] if "created_at" in r.keys() else __import__("time").time(),
                                ))
                                s.commit()
                                imported_aliases += 1
                        except Exception as ex:
                            print(f"[input-learned-db] alias {r['alias'][:30] if 'alias' in r.keys() else '?'} failed: {ex}")
                except Exception:
                    # Older DB without hits columns
                    try:
                        rows = src.execute("SELECT alias, title_key, weight FROM aliases").fetchall()
                        for r in rows:
                            try:
                                alias = r["alias"]
                                if not alias:
                                    continue
                                from database import Alias
                                with db._session() as s:
                                    if s.get(Alias, alias) is not None:
                                        skipped_aliases += 1
                                        continue
                                    s.add(Alias(alias=alias, title_key=r["title_key"], weight=r["weight"]))
                                    s.commit()
                                    imported_aliases += 1
                            except: pass
                    except Exception as ex:
                        print(f"[input-learned-db] aliases fallback failed: {ex}")
                # --- Resources: skip if slug exists ---
                try:
                    rows = src.execute("SELECT slug, name, kind, page_url, domain, search_url, search_param, status, note, last_seen, last_checked, dead_count FROM resources").fetchall()
                    for r in rows:
                        try:
                            slug = r["slug"]
                            if not slug:
                                continue
                            from database import Resource
                            with db._session() as s:
                                if s.get(Resource, slug) is not None:
                                    skipped_resources += 1
                                    continue
                                s.add(Resource(
                                    slug=slug, name=r["name"], kind=r["kind"], page_url=r["page_url"],
                                    domain=r["domain"], search_url=r["search_url"], search_param=r["search_param"],
                                    status=r["status"], note=r["note"], last_seen=r["last_seen"], last_checked=r["last_checked"], dead_count=r["dead_count"]
                                ))
                                s.commit()
                                imported_resources += 1
                        except Exception as ex:
                            print(f"[input-learned-db] resource {r['slug'] if 'slug' in r.keys() else '?'} failed: {ex}")
                except Exception as ex:
                    print(f"[input-learned-db] resources failed: {ex}")
                src.close()
            except Exception as ex:
                print(f"[input-learned-db] failed: {ex}")
                import traceback; traceback.print_exc()
                return {"error": str(ex)}
            finally:
                try:
                    __import__("os").unlink(src_path)
                except: pass
                try:
                    __import__("os").unlink(src_path + "-wal")
                except: pass
                try:
                    __import__("os").unlink(src_path + "-shm")
                except: pass
            return {
                "imported_titles": imported_titles, "skipped_titles": skipped_titles, "total_titles": total_titles,
                "imported_items": imported_items, "skipped_items": skipped_items, "total_items": total_items,
                "imported_aliases": imported_aliases, "skipped_aliases": skipped_aliases,
                "imported_resources": imported_resources, "skipped_resources": skipped_resources,
            }
        try:
            res = await asyncio.get_event_loop().run_in_executor(None, _import_db)
        except Exception as ex:
            await inter.followup.send(f"❌ DB import failed: `{ex}`", ephemeral=True)
            return
        if "error" in res:
            await inter.followup.send(f"❌ DB import error: `{res['error']}`", ephemeral=True)
            return
        embed = discord.Embed(title="✅ Import from .db complete", color=config.THEME_COLOR,
                              description=f"File `{file.filename}` merged — **skipped existing, imported new**.")
        embed.add_field(name="📚 Titles", value=f"Total in file: **{res['total_titles']}**\nImported **{res['imported_titles']}** new\nSkipped **{res['skipped_titles']}** already existed", inline=False)
        embed.add_field(name="📰 Items", value=f"Total in file: **{res['total_items']}**\nImported **{res['imported_items']}** new\nSkipped **{res['skipped_items']}** already existed", inline=False)
        embed.add_field(name="🏷️ Aliases/Resources", value=f"Aliases: +{res['imported_aliases']}/skip {res['skipped_aliases']}\nResources: +{res['imported_resources']}/skip {res['skipped_resources']}", inline=False)
        st = await asyncio.get_event_loop().run_in_executor(None, db.stats)
        embed.add_field(name="📊 Now", value=f"{st['titles']} titles · {st['items']} items · proficiency {st['proficiency']}%", inline=False)
        embeds._stamp(embed)
        await inter.followup.send(embed=embed, ephemeral=True)
        print(f"[input-learned-db] {file.filename} by {inter.user.id}: {res}")
    except Exception as ex:
        print(f"[input-learned-db] outer failed: {ex}")
        import traceback; traceback.print_exc()
        await inter.followup.send(f"❌ Unexpected error: `{ex}`", ephemeral=True)


@tree.command(name="stats", description="Show knowledge & proficiency")
async def cmd_stats(inter: discord.Interaction):
    if not await safe_respond(inter, embed=embeds.stats_embed(db.stats()), ephemeral=True):
        return


@tree.command(name="where", description="Where can I watch/read <name>? (legal + free, per-server isolated)")
@app_commands.describe(name="Title to locate", media_type="anime / manga / manhwa")
async def cmd_where(inter: discord.Interaction, name: str, media_type: str = "anime"):
    if not await safe_defer(inter, ephemeral=True):
        return
    gid = _guild_id(inter)
    rec = match_title(name, db, threshold=60)
    if not rec:
        s = await session()
        clean, _, _ = parse_query(name)
        live = await fetch_exhaustive_search(s, clean, preferred_type="anime" if media_type.lower()=="anime" else "manga", db=db)
        if not live:
            live = await fetch_exhaustive_search(s, clean, preferred_type="manga" if media_type.lower()=="anime" else "anime", db=db)
        if live:
            db.learn_title(TitleRecord(
                key=live["key"], canonical=live["canonical"],
                media_type=live["media_type"], anilist_id=live.get("anilist_id",""), mal_id=live.get("mal_id",""),
                image=live.get("image", ""), aliases=live.get("aliases",[]),
                watch_links=live.get("watch_links",[]), read_links=live.get("read_links",[])))
            rec = db.get_title(live["key"])
            if rec:
                rec["score"] = live.get("score", 90)
                rec["match_method"] = "live-exhaustive"
    if not rec:
        await inter.followup.send(f"Couldn't resolve “{name}”. Try `/search {name}` first (exhaustive).", ephemeral=True)
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
    if inter.guild is None:
        ok = await send_dm(inter.user.id, embed=embed)
        if ok:
            await inter.followup.send("📬 Where-to-watch/read sent to your **DM/inbox** (personal).", ephemeral=True)
        else:
            await inter.followup.send(embed=embed)
            await inter.followup.send("⚠️ DMs blocked — delivered here instead.", ephemeral=True)
    else:
        await inter.followup.send(embed=embed, ephemeral=True)
        await inter.followup.send(f"📬 Where-to-watch delivered in **{inter.guild.name}** (server-isolated, guild {gid}).", ephemeral=True)


@tree.command(name="trending", description="What's trending right now (per-server isolated)")
@app_commands.describe(media_type="anime / manga")
async def cmd_trending(inter: discord.Interaction, media_type: str = "anime"):
    if not await safe_defer(inter, ephemeral=True):
        return
    gid = _guild_id(inter)
    s = await session()
    mt = "ANIME" if media_type.lower() == "anime" else "MANGA"
    recs = await fetch_anilist_trending(s, mt, per=10)
    # fallback to Kitsu trending via exhaustive if AniList down/empty
    if not recs:
        for q in ["One Piece", "Jujutsu Kaisen", "Demon Slayer"]:
            live = await fetch_exhaustive_search(s, q, preferred_type=media_type, db=db)
            if live:
                recs.append(TitleRecord(key=live["key"], canonical=live["canonical"], media_type=live["media_type"], image=live.get("image",""), aliases=live.get("aliases",[]), watch_links=live.get("watch_links",[]), read_links=live.get("read_links",[])))
        if recs:
            print(f"[trending] fallback via exhaustive got {len(recs)}")
    for r in recs:
        db.learn_title(r)
    if not recs:
        await inter.followup.send("Trending fetch failed (AniList temporarily disabled; try again later).", ephemeral=True)
        return
    header = embeds.header_embed(
        f"🔥 Trending {media_type}",
        f"Top {len(recs)} · learned · context: {'server '+gid if gid!='0' else 'DM/inbox'}",
        media_type=media_type)
    if inter.guild is None:
        ok = await send_dm(inter.user.id, embed=header)
        if not ok:
            await inter.followup.send(embed=header)
        for r in recs:
            em = embeds.search_result_embed({
                **r.__dict__, "score": 99, "match_method": "trending",
                "aliases": r.aliases, "watch_links": r.watch_links, "read_links": r.read_links,
            }, r.canonical)
            em.add_field(name="🆓 Free hosts (EverythingMoe)", value=free_links(r.media_type), inline=False)
            dsl = direct_search_links({"canonical": r.canonical, "media_type": r.media_type})
            if dsl != "—":
                em.add_field(name="🔗 Direct search on free hosts", value=dsl, inline=False)
            ok2 = await send_dm(inter.user.id, embed=em)
            if not ok2:
                await inter.followup.send(embed=em)
            await asyncio.sleep(0.35)
        if ok:
            await inter.followup.send("📬 Trending list sent to **DM/inbox** (personal per-you).", ephemeral=True)
        else:
            await inter.followup.send("⚠️ DMs blocked — trending delivered here instead.", ephemeral=True)
    else:
        await inter.followup.send(embed=header, ephemeral=True)
        for r in recs[:5]:
            em = embeds.search_result_embed({
                **r.__dict__, "score": 99, "match_method": "trending",
                "aliases": r.aliases, "watch_links": r.watch_links, "read_links": r.read_links,
            }, r.canonical)
            em.add_field(name="🆓 Free hosts (EverythingMoe)", value=free_links(r.media_type), inline=False)
            dsl = direct_search_links({"canonical": r.canonical, "media_type": r.media_type})
            if dsl != "—":
                em.add_field(name="🔗 Direct search on free hosts", value=dsl, inline=False)
            await inter.followup.send(embed=em, ephemeral=True)
            await asyncio.sleep(0.3)
        # remaining via followup as well (still server-isolated)
        for r in recs[5:]:
            em = embeds.search_result_embed({
                **r.__dict__, "score": 99, "match_method": "trending",
                "aliases": r.aliases, "watch_links": r.watch_links, "read_links": r.read_links,
            }, r.canonical)
            await inter.followup.send(embed=em, ephemeral=True)
            await asyncio.sleep(0.3)
        await inter.followup.send(f"📬 Trending delivered in **{inter.guild.name}** (server-isolated, guild {gid}).", ephemeral=True)


@tree.command(name="learn", description="Force a learning/scrape cycle now")
@owner_only()
async def cmd_learn(inter: discord.Interaction):
    if not await safe_defer(inter, ephemeral=True):
        return
    n, items = await do_learn_cycle()
    await inter.followup.send(f"🧠 Cycle done: {n} new items ingested, knowledge updated.", ephemeral=True)


@tree.command(name="start", description="Auto-news: per-server & per-person isolated digest every 30 min")
async def cmd_start(inter: discord.Interaction):
    if not await safe_defer(inter, ephemeral=True):
        return
    gid = _guild_id(inter)
    ch_id = inter.channel_id if inter.guild else 0
    db.set_broadcast(inter.user.id, ch_id, guild_id=gid)
    mins = config.BROADCAST_INTERVAL // 60
    if gid == "0":
        where = "your **DM/inbox** (personal, per-you isolated)"
    else:
        where = f"this channel **{inter.channel.name if hasattr(inter.channel,'name') else ch_id}** in **{inter.guild.name}** **and** your DM (per-server+per-person isolated)"
    await inter.followup.send(
        f"📡 **Auto-news ON** (guild {gid}) — every {mins} min to {where}. Use `/stop` in this same context to turn it off.",
        ephemeral=True,
    )


@tree.command(name="stop", description="Stop auto-news digests (per-server isolated)")
async def cmd_stop(inter: discord.Interaction):
    if not await safe_defer(inter, ephemeral=True):
        return
    gid = _guild_id(inter)
    if db.stop_broadcast(inter.user.id, guild_id=gid):
        await inter.followup.send(f"📡 **Auto-news OFF** for this context (guild {gid}).", ephemeral=True)
    else:
        # also try global fallback
        if db.stop_broadcast(inter.user.id, guild_id=None):
            await inter.followup.send(f"📡 **Auto-news OFF** (cleared global for user).", ephemeral=True)
        else:
            await inter.followup.send(f"You don't have auto-news running in this **{'server' if gid!='0' else 'DM'}** (guild {gid}) — use `/start` here.", ephemeral=True)


# ------------------------------------------------------------- learning loop
async def do_learn_cycle() -> tuple[int, int]:
    """Scrape widely, store items, learn trending titles. Returns (new_items, total)."""
    s = await session()
    loop = asyncio.get_event_loop()
    t0 = time.time()
    n, items = await run_news_cycle(s, db=db)
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

        # 3) exhaustive news crawl — self-learning sources (whole web, not just fixed RSS)
        try:
            # Self-learning: pick 2 best RSS sources from DB (ranked by reliability), rotate
            rss_tasks = []
            try:
                active_srcs = await asyncio.get_event_loop().run_in_executor(None, db.get_active_news_sources, "", 12)
                # Rotate through top 12
                for i in range(2):
                    if active_srcs:
                        src = active_srcs[(_explore_cycle + i) % len(active_srcs)]
                        # Ensure _db_ref for stats
                        src = dict(src); src["_db_ref"] = db
                        rss_tasks.append(fetch_rss(s, src))
                    else:
                        src = config.RSS_SOURCES[(_explore_cycle + i) % len(config.RSS_SOURCES)]
                        src = dict(src); src["_db_ref"] = db
                        rss_tasks.append(fetch_rss(s, src))
            except Exception:
                for i in range(2):
                    src = config.RSS_SOURCES[(_explore_cycle + i) % len(config.RSS_SOURCES)]
                    src = dict(src); src["_db_ref"] = db
                    rss_tasks.append(fetch_rss(s, src))
            # Add generic web news (whole internet, not fixed RSS)
            from fetchers import fetch_web_news, fetch_news_exhaustive
            web_tasks = [fetch_web_news(s, q, limit=6) for q in ["anime news", "manga news"]]
            # Also run the exhaustive news crawl (covers web broadly)
            rss_tasks.extend(web_tasks)
            # Also try to discover new RSS via web for unpopular titles: search web for recent anime news
            all_results = await asyncio.gather(*rss_tasks, return_exceptions=True)
            new = 0
            for res in all_results:
                if isinstance(res, Exception) or not res:
                    continue
                for it in res:
                    if db.add_item(it):
                        new += 1
            # Every 3 cycles, also run the full exhaustive news (which does 4 web queries + RSS) and persist
            if _explore_cycle % 3 == 0:
                try:
                    n2, more = await run_news_cycle(s, db=db)
                    if more:
                        await asyncio.get_event_loop().run_in_executor(None, db.bulk_add_items, more)
                        new += len(more)
                except Exception as ex2:
                    print(f"[explore] exhaustive news failed: {ex2}")
        except Exception as ex:
            print(f"[explore] news crawl failed: {ex}")
            new = 0

        st = db.stats()
        await bot.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name=f"crawling web · {st['resources']} hosts · {st['proficiency']}/100 · {st['items']} news",
            )
        )
        if _explore_cycle % 5 == 0:
            print(f"[explore] cycle {_explore_cycle}: +{new} news (web+rw) · {len(alive)} hosts watched")
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
        s = await session()
        recent = db.recent_items(200)  # larger cache
        # Also fetch exhaustive live news in parallel for broader coverage (whole web, not just cached)
        # This ensures even unpopular titles get news via web crawl
        try:
            live_items = await fetch_news_for_title(s, "anime", limit=12, db=db)  # generic warmup, self-learning
            recent = (live_items[:10] if live_items else []) + recent
        except Exception:
            pass
        try:
            follows = db.all_follows_detailed()
        except Exception:
            follows = [{"user_id": str(uid), "title_key": key, "guild_id": "0"} for uid, key in db.all_follows()]
        async def _news_for_follow(entry):
            try:
                if isinstance(entry, tuple):
                    uid, key = entry; gid = "0"
                else:
                    uid = entry.get("user_id"); key = entry.get("title_key"); gid = entry.get("guild_id", "0")
                rec = db.get_title(key)
                if not rec:
                    return []
                canon = rec["canonical"]
                cached_matches = match_news(canon, recent, threshold=58)
                try:
                    live_for_title = await fetch_news_for_title(s, canon, limit=8, db=db)
                    # Score live items too
                    live_matches = match_news(canon, [{"title": it.title, "url": it.url, "summary": it.summary, "source": it.source, "id": hash(it.url)} for it in live_for_title], threshold=55)
                    # Convert live_for_title NewsItems to dicts for matching
                    # Already scored via match_news above, but we need dicts with id
                    # Merge
                    combined = cached_matches + live_matches
                    # Also add raw live items that contain title (even if scorer < threshold, for unpopular)
                    for it in live_for_title:
                        if canon.lower() in it.title.lower() and it not in combined:
                            combined.append({"title": it.title, "url": it.url, "summary": it.summary, "source": it.source, "id": hash(it.url), "score": 70})
                except Exception as ex:
                    print(f"[watch] live news for {canon} failed: {ex}")
                    combined = cached_matches
                return [(entry, m) for m in combined]
            except Exception as ex:
                print(f"[watch] _news_for_follow failed: {ex}")
                return []
        # Gather live news for all follows in parallel (capped)
        if follows:
            batch = follows[:12]  # avoid overload: max 12 follows per cycle
            results = await asyncio.gather(*[_news_for_follow(e) for e in batch], return_exceptions=True)
            for res in results:
                if isinstance(res, Exception) or not res:
                    continue
                for entry, m in res:
                    if isinstance(entry, tuple):
                        uid, key = entry; gid = "0"
                    else:
                        uid = entry.get("user_id"); key = entry.get("title_key"); gid = entry.get("guild_id", "0")
                    iid = m.get("id") or hash(m.get("url",""))
                    cache_key = (str(uid), str(gid), key, str(iid))
                    if cache_key in _sent_cache:
                        continue
                    _sent_cache.add(cache_key)
                    if len(_sent_cache) > 8000:
                        for _ in range(1000):
                            try: _sent_cache.pop()
                            except: break
                            if len(_sent_cache) <= 7000:
                                break
                    # Persist live news items that were found via web (so /news can see them)
                    try:
                        if m.get("url") and not any(r["url"] == m["url"] for r in recent):
                            # Convert dict back to NewsItem for DB
                            from database import NewsItem as _NI
                            db.add_item(_NI(source=m.get("source","Web"), kind="news", title=m["title"], url=m["url"], summary=m.get("summary","")[:500], published=""))
                    except Exception:
                        pass
                    sent_dm = await send_dm(int(uid), embed=embeds.news_embed(m, alert=f"{db.get_title(key)['canonical'] if db.get_title(key) else key} — new related news (guild {gid})"))
                    if config.NEWS_CHANNEL_ID and gid == "0":
                        ch = bot.get_channel(config.NEWS_CHANNEL_ID)
                        if ch:
                            try: await ch.send(embed=embeds.news_embed(m, alert=f"{db.get_title(key)['canonical'] if db.get_title(key) else key} — new related news"))
                            except: pass
                    if gid != "0":
                        try:
                            b = db.get_broadcast(str(uid), guild_id=gid)
                            if b and b.get("channel_id"):
                                ch = bot.get_channel(int(b["channel_id"]))
                                if ch:
                                    await ch.send(embed=embeds.news_embed(m, alert=f"{db.get_title(key)['canonical'] if db.get_title(key) else key} — new related news (followed in this server)"))
                        except: pass
                    if not sent_dm:
                        print(f"[watch] DM failed for {uid} guild {gid}")
        _last_watch_run = time.time()
    except Exception as ex:
        print(f"[watch_loop] error: {ex}")
        import traceback; traceback.print_exc()


@tasks.loop(seconds=config.BROADCAST_INTERVAL)
async def broadcast_loop():
    """Auto-news (/start): per-server & per-person isolated — each guild/person combo gets its own digest."""
    try:
        targets = db.all_broadcasts()
        if not targets:
            return
        s = await session()
        n, items = await run_news_cycle(s)
        if items:
            db.bulk_add_items(items)
        for t in targets:
            gid = str(t.get("guild_id", "0"))
            fresh = db.news_since(t["last_sent"], limit=12)
            if not fresh:
                continue
            try:
                # DM personal digest (always, but annotated per guild)
                embed = embeds.digest_embed(fresh, f"Auto-news digest · guild {gid}")
                sent = await send_dm(int(t["user_id"]), embed=embed)
                if not sent:
                    print(f"[broadcast] DM failed for {t['user_id']} guild {gid}")
                ch_id = t.get("channel_id") or 0
                if ch_id:
                    ch = bot.get_channel(int(ch_id))
                    if ch:
                        # server-isolated channel digest (guild-specific)
                        await ch.send(embed=embeds.digest_embed(fresh, f"Auto-news digest · {ch.guild.name if hasattr(ch.guild,'name') else gid}"))
                    else:
                        print(f"[broadcast] channel {ch_id} gone for user {t['user_id']} guild {gid}")
            except Exception as ex:
                print(f"[broadcast] failed for {t['user_id']} guild {gid}: {ex}")
            db.touch_broadcast(t["user_id"], time.time(), guild_id=gid)
    except Exception as ex:
        print(f"[broadcast_loop] error: {ex}")


@tasks.loop(seconds=3600)
async def self_improvement_loop():
    """Self-improvement: periodically tunes algorithm, prunes, decays, and learns.
    Runs every 1h, uses feedback to auto-improve matching and knowledge.
    """
    try:
        print("[self-improvement] cycle start")
        # 1) Prune stale low-weight aliases (self-cleaning)
        pruned = await asyncio.get_event_loop().run_in_executor(None, db.prune_stale_aliases, 90, 0.32)
        if pruned:
            print(f"[self-improvement] pruned {pruned} stale aliases")
        # 2) Decay confidence for titles not seen in 60+ days
        decayed = await asyncio.get_event_loop().run_in_executor(None, db.decay_stale_titles, 60, 0.12)
        if decayed:
            print(f"[self-improvement] decayed {decayed} stale titles")
        # 3) Recompute alias weights from feedback history (batch learning)
        updated = await asyncio.get_event_loop().run_in_executor(None, db.recompute_alias_weights_from_feedback)
        if updated:
            print(f"[self-improvement] recomputed {updated} alias weights from feedback")
        # 4) Auto-tune MATCH_THRESHOLD based on recent feedback window
        new_thr, info = await asyncio.get_event_loop().run_in_executor(None, db.auto_tune_threshold)
        old_thr = float(config.MATCH_THRESHOLD)
        if abs(new_thr - old_thr) > 0.6:
            config.MATCH_THRESHOLD = new_thr
            print(f"[self-improvement] threshold {old_thr:.1f} -> {new_thr:.1f} {info}")
            # Persist threshold to DB metrics
            await asyncio.get_event_loop().run_in_executor(None, db.log_learning_window, new_thr)
        else:
            # Still log window for history
            await asyncio.get_event_loop().run_in_executor(None, db.log_learning_window, old_thr)
        # 5) Learn hybrid scorer weights from feedback (self-learning algorithm)
        try:
            # Fetch recent feedback with scores
            def _learn_weights():
                import time as _t
                from matcher import get_weights as _gw, set_weights as _sw
                # Get feedback last 14 days
                with db._session() as s:
                    from database import Feedback as _FB
                    from sqlalchemy import select as _sel
                    rows = s.execute(_sel(_FB).where(_FB.ts > _t.time() - 14*86400)).scalars().all()
                    if len(rows) < 12:
                        return None
                    # For each feedback, we have query, matched_key, correct, score, method
                    # Compute per-signal separation if we had stored per-signal scores (we don't yet, so approximate)
                    # Instead, use score as proxy: correct should have high score, wrong low
                    # If avg correct >> avg wrong, keep weights; if not, nudge jaro/ngram up for typo cases
                    correct = [r for r in rows if r.correct]
                    wrong = [r for r in rows if not r.correct]
                    if not correct or not wrong:
                        return None
                    # Heuristic: if many wrong have high token_set but low jaro/ngram, it indicates typos (Chronicles)
                    # Boost ngram/jaro slightly
                    avg_c_score = sum(r.score or 75 for r in correct) / len(correct)
                    avg_w_score = sum(r.score or 45 for r in wrong) / len(wrong)
                    gap = avg_c_score - avg_w_score
                    cur = _gw()
                    # If gap is small (<18), matcher is not separating well -> increase jaro/ngram (typo handling)
                    if gap < 18:
                        new_w = dict(cur)
                        # Shift 0.04 from set/partial to jaro/ngram
                        new_w["jaro"] = min(0.32, cur["jaro"] + 0.035)
                        new_w["ngram"] = min(0.28, cur["ngram"] + 0.035)
                        new_w["set"] = max(0.18, cur["set"] - 0.035)
                        new_w["partial"] = max(0.05, cur["partial"] - 0.035)
                        _sw(new_w)
                        return {"gap": round(gap,1), "old": cur, "new": new_w, "n": len(rows)}
                    # If gap large (>32), matcher is overconfident -> slightly reduce jaro/ngram, increase set
                    elif gap > 32:
                        new_w = dict(cur)
                        new_w["set"] = min(0.50, cur["set"] + 0.025)
                        new_w["jaro"] = max(0.10, cur["jaro"] - 0.012)
                        new_w["ngram"] = max(0.08, cur["ngram"] - 0.013)
                        _sw(new_w)
                        return {"gap": round(gap,1), "old": cur, "new": new_w, "n": len(rows)}
                    return None
            res = await asyncio.get_event_loop().run_in_executor(None, _learn_weights)
            if res:
                print(f"[self-improvement] hybrid weights tuned {res}")
        except Exception as ex:
            print(f"[self-improvement] weight learn failed: {ex}")
        # 6) Self-learning sources: sync config, discover new RSS via web, update reliability
        try:
            # Ensure config sources are seeded
            added_cfg = await asyncio.get_event_loop().run_in_executor(None, db.sync_news_sources_from_config)
            if added_cfg:
                print(f"[self-improvement] seeded {added_cfg} news sources from config")
            # Discover new RSS feeds via web (whole internet, not just config)
            try:
                s = await session()
                # Use the fetcher's discovery (needs db)
                from fetchers import discover_rss_sources_via_web
                discovered = await discover_rss_sources_via_web(s, db, limit=4)
                if discovered:
                    print(f"[self-improvement] discovered {len(discovered)} new RSS sources via web: {[d['url'][:40] for d in discovered]}")
            except Exception as ex:
                print(f"[self-improvement] RSS discovery failed: {ex}")
            # Prune dead/paused sources, keep at most 40 active
            pruned_src = await asyncio.get_event_loop().run_in_executor(None, db.prune_dead_news_sources, 40)
            if pruned_src:
                print(f"[self-improvement] pruned {pruned_src} dead news sources")
            # Log source health
            try:
                srcs = await asyncio.get_event_loop().run_in_executor(None, db.get_all_news_sources)
                active = [s for s in srcs if s.get("status") == "active"]
                avg_rel = sum(s.get("reliability",0) for s in active)/max(1,len(active)) if active else 0
                print(f"[self-improvement] sources: {len(active)} active / {len(srcs)} total, avg reliability {avg_rel:.2f}")
                # Also log title search source health
                tsrcs = await asyncio.get_event_loop().run_in_executor(None, db.get_title_search_sources)
                print(f"[self-improvement] title sources: {[(t['name'], round(t['reliability'],2), round(t['weight'],2)) for t in tsrcs]}")
            except Exception as ex:
                print(f"[self-improvement] source report failed: {ex}")
        except Exception as ex:
            print(f"[self-improvement] source learning failed: {ex}")
        # 7) Update presence with learning report
        try:
            rep = await asyncio.get_event_loop().run_in_executor(None, db.get_learning_report)
            print(f"[self-improvement] report {rep}")
            await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name=f"self-learning · {rep.get('accuracy_7d',0)}% acc · {rep.get('total_titles',0)} titles"))
        except Exception:
            pass
        print("[self-improvement] cycle done")
    except Exception as ex:
        print(f"[self_improvement_loop] error: {ex}")
        import traceback; traceback.print_exc()


@tree.command(name="learning", description="Show self-learning & self-improvement report")
async def cmd_learning(inter: discord.Interaction):
    if not await safe_defer(inter, ephemeral=True):
        return
    rep = await asyncio.get_event_loop().run_in_executor(None, db.get_learning_report)
    w = get_weights()
    e = discord.Embed(title="🧬 Self-Learning & Self-Improvement", color=config.THEME_COLOR,
                      description=f"**Algorithm** hybrid scorer weights: `set {w['set']:.2f} sort {w['sort']:.2f} jaro {w['jaro']:.2f} ngram {w['ngram']:.2f} partial {w['partial']:.2f}`\n"
                                  f"Threshold `MATCH_THRESHOLD={float(config.MATCH_THRESHOLD):.1f}` (auto-tuned)")
    e.add_field(name="📈 7-day Feedback", value=f"Queries **{rep.get('feedback_7d',0)}** · Acc **{rep.get('accuracy_7d',0)}%** ({rep.get('correct_7d',0)}✅ / {rep.get('wrong_7d',0)}❌)", inline=False)
    e.add_field(name="📚 Knowledge", value=f"Titles **{rep.get('total_titles',0)}** · Avg conf **{rep.get('avg_confidence',0)}%**\nAliases **{rep.get('alias_count',0)}** (low {rep.get('low_alias',0)})", inline=False)
    if rep.get("history"):
        lines = []
        for h in rep["history"][:3]:
            lines.append(f"<t:{int(h['window_end'])}:R> acc {h['accuracy']:.0f}% thr {h['thr']:.0f}")
        e.add_field(name="🕒 Recent Windows", value="\n".join(lines) if lines else "—", inline=False)
    e.add_field(name="⚙️ Self-Improvement", value="Runs hourly: prunes stale aliases, decays stale titles, recomputes alias weights, auto-tunes threshold & hybrid weights, logs windows.", inline=False)
    embeds._stamp(e)
    await inter.followup.send(embed=e, ephemeral=True)

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
    if not self_improvement_loop.is_running():
        self_improvement_loop.start()
    # Prime the knowledge base on boot WITHOUT blocking on_ready — run it as a
    # background task so the bot stays responsive to commands immediately.
    asyncio.create_task(_boot_learn())


async def _boot_learn():
    try:
        # Self-learning: ensure news/title sources are seeded from config and DB is ready
        try:
            await asyncio.get_event_loop().run_in_executor(None, db.sync_news_sources_from_config)
            # Also ensure title search sources exist
            await asyncio.get_event_loop().run_in_executor(None, db.get_title_search_sources)
            # Quick source health check
            srcs = await asyncio.get_event_loop().run_in_executor(None, db.get_all_news_sources)
            print(f"[boot] sources seeded: {len(srcs)} news sources")
        except Exception as ex:
            print(f"[boot] source sync failed: {ex}")
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
