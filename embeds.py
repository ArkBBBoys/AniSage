"""Pretty Discord embeds. One place to keep the visual style consistent.

Design language
---------------
* Accent colors follow the media type: anime = AniList blue, manga = green,
  manhwa = violet (fallback: THEME_COLOR).
* Confidence / proficiency values render as text progress bars (block chars
  inside code ticks keep alignment on every client).
* Every embed carries a small branded footer + UTC timestamp so DMs read as
  a single consistent "feed".
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import discord

import config

BRAND = "AniSage"
BAR_FULL = "█"
BAR_EMPTY = "░"
BAR_WIDTH = 10

COLORS = {
    "anime": 0x02A9FF,  # AniList blue
    "manga": 0x2EBD59,  # fresh green
    "manhwa": 0xB18CFF,  # violet
}

DIVIDER = "\u200b"  # zero-width space: keeps empty embed fields visible


def _media_color(media_type: str | None) -> int:
    return COLORS.get((media_type or "").lower(), config.THEME_COLOR)


def _coerce_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return []
    return []


def _trim(s: str, n: int = 200) -> str:
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _stamp(e: discord.Embed, note: str | None = None) -> None:
    """Branded footer + UTC timestamp — the visual anchor of every embed."""
    text = note if note else BRAND
    e.set_footer(text=text)
    e.timestamp = datetime.now(timezone.utc)


def bar(value: float, width: int = BAR_WIDTH) -> str:
    """Text progress bar for 0-100 values, e.g. 78 -> `███████░░░` **78%**."""
    value = max(0.0, min(100.0, float(value)))
    filled = round(value / 100 * width)
    blocks = BAR_FULL * filled + BAR_EMPTY * (width - filled)
    return f"`{blocks}` **{value:.0f}%**"


def _links_field(e: discord.Embed, name: str, links: list[str]) -> None:
    """Add a link field, trimmed to Discord's 1024-char field cap."""
    value = "\n".join(links[:5])
    if len(value) > 1024:
        value = value[:1023].rstrip() + "…"
    if value:
        e.add_field(name=name, value=value, inline=False)


def header_embed(title: str, subtitle: str | None = None,
                 media_type: str | None = None) -> discord.Embed:
    """Compact banner embed used to open a DM burst (digest, trending, export)."""
    e = discord.Embed(title=title, color=_media_color(media_type))
    if subtitle:
        e.description = subtitle
    _stamp(e)
    return e


def _valid_image_url(url: str) -> str | None:
    """Normalize/validate an image URL; None when it can't render in Discord.

    Discord silently drops broken or unsupported images, so we pre-check:
    http(s) only, no SVG, no data URIs, no tracking pixels.
    """
    if not url:
        return None
    u = url.strip()
    if u.startswith("//"):
        u = "https:" + u
    low = u.lower()
    if not low.startswith(("http://", "https://")):
        return None
    if low.endswith(".svg") or "data:" in low or "pixel" in low:
        return None
    return u


def news_embed(item: dict, alert: str | None = None) -> discord.Embed:
    e = discord.Embed(
        title=_trim(item.get("title", "Untitled"), 256),
        url=item.get("url") or None,
        description=_trim(item.get("summary", ""), 500) or None,
        color=_media_color(item.get("media_type", "unknown")),
    )
    url = item.get("url") or None
    if alert:
        e.set_author(name=f"🔔 {alert}", url=url)
        footer_note = f"{item.get('source', BRAND)} · {BRAND}"
    else:
        e.set_author(name=item.get("source", "AniSage"), url=url)
        footer_note = BRAND
    img = _valid_image_url(item.get("image") or "")
    if img:
        e.set_image(url=img)
    mt = item.get("media_type", "unknown")
    if mt != "unknown":
        e.add_field(name="Type", value=mt.title(), inline=True)
    if item.get("published"):
        e.add_field(name="Published", value=_trim(str(item["published"]), 64), inline=True)
    _stamp(e, note=footer_note)
    return e


def search_result_embed(rec: dict, query: str) -> discord.Embed:
    mt = (rec.get("media_type") or "unknown").lower()
    e = discord.Embed(
        title=rec.get("canonical", "Unknown"),
        url=rec.get("url") or None,
        description=f"Matched **“{query}”** · via *{rec.get('match_method', 'fuzzy')}*",
        color=_media_color(mt),
    )
    img = _valid_image_url(rec.get("image") or "")
    if img:
        e.set_image(url=img)
    e.add_field(name="Kind", value=mt.title(), inline=True)
    if rec.get("season") is not None or rec.get("unit"):
        bits = []
        if rec.get("season") is not None:
            bits.append(f"Season {rec['season']}")
        if rec.get("unit"):
            num, label = rec["unit"]
            bits.append(f"{label} {num}")
        e.add_field(name="Requested", value=" · ".join(bits), inline=True)
    e.add_field(name="Confidence", value=bar(rec.get("confidence", 0)), inline=True)
    if rec.get("anilist_id"):
        e.add_field(name="AniList ID", value=rec["anilist_id"], inline=True)
    if rec.get("mal_id"):
        e.add_field(name="MAL ID", value=rec["mal_id"], inline=True)
    _links_field(e, "▶️ Watch (legal)", _coerce_list(rec.get("watch_links")))
    _links_field(e, "📖 Read (legal)", _coerce_list(rec.get("read_links")))
    aliases = _coerce_list(rec.get("aliases"))
    if aliases:
        e.add_field(name="Known aliases",
                    value=", ".join(aliases[:8]), inline=False)
    _stamp(e)
    return e


def item_search_embed(items: list[dict], query: str) -> discord.Embed:
    e = discord.Embed(
        title=f"🔎 Related news — “{query}”",
        color=config.THEME_COLOR,
        description=f"{len(items)} match(es) from the knowledge base",
    )
    img = _valid_image_url(items[0].get("image") or "") if items else None
    if img:
        e.set_thumbnail(url=img)
    for i, it in enumerate(items[:10], 1):
        e.add_field(
            name=f"{i}. {it.get('source')}",
            value=f"[{_trim(it.get('title', ''), 100)}]({it.get('url')}) · "
                  f"score **{it.get('score')}**",
            inline=False,
        )
    _stamp(e)
    return e


def stats_embed(stats: dict) -> discord.Embed:
    e = discord.Embed(
        title="🧠 AniSage — Knowledge & Proficiency",
        color=config.THEME_COLOR,
    )
    e.add_field(
        name="📚 Knowledge base",
        value=f"**{stats['items']}** news items\n**{stats['titles']}** titles learned",
        inline=False,
    )
    e.add_field(
        name="🎯 Performance",
        value=f"Match accuracy **{stats['accuracy']}%** · "
              f"avg confidence **{stats['avg_confidence']}%**",
        inline=False,
    )
    e.add_field(name="🆓 Free hosts (EverythingMoe)", value=stats["resources"], inline=True)
    e.add_field(name="💬 Human feedback", value=stats["feedback"], inline=True)
    e.add_field(name="🚀 Proficiency", value=bar(stats["proficiency"]), inline=False)
    _stamp(e)
    return e


def digest_embed(items: list[dict], label: str) -> discord.Embed:
    e = discord.Embed(
        title=f"📰 AniSage Digest — {label}",
        color=config.THEME_COLOR,
        description=f"{len(items)} fresh stories · newest first",
    )
    img = _valid_image_url(items[0].get("image") or "") if items else None
    if img:
        e.set_image(url=img)
    lines = []
    for i, it in enumerate(items[:12], 1):
        title = _trim(it.get("title", ""), 90)
        src = it.get("source", "Unknown")
        lines.append(f"**{i}.** [{title}]({it.get('url')}) — {src}")
    e.description = f"{e.description}\n\n" + "\n".join(lines)
    _stamp(e)
    return e


def learned_chunk_embed(lines: list[str], page: int, total_pages: int) -> discord.Embed:
    e = discord.Embed(
        title="📚 Knowledge base",
        color=config.THEME_COLOR,
        description="\n".join(_trim(l, 300) for l in lines),
    )
    _stamp(e, note=f"{BRAND} · page {page}/{total_pages}")
    return e


def export_chunk_embed(title: str, lines: list[str]) -> discord.Embed:
    e = discord.Embed(title=title, color=config.THEME_COLOR,
                      description="\n".join(lines) if lines else "—")
    _stamp(e)
    return e
