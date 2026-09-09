"""Configuration & runtime settings for the AniSage self-learning Discord bot."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    """Read an int env var, falling back to default instead of crashing.

    A malformed value (e.g. a token pasted into OWNER_ID) logs a clear warning
    rather than raising ValueError and killing the bot at import time.
    """
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[config] WARNING: {name}={raw!r} is not a valid integer; "
              f"using default {default}. Check your .env.")
        return default


# ---------------------------------------------------------------------------
# Core identity
# ---------------------------------------------------------------------------
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")
# The personal owner (you). DM delivery & admin commands are restricted to this id.
# This MUST be your numeric Discord USER ID (digits only), NOT the bot token.
OWNER_ID = _env_int("OWNER_ID", 0)

# Optional channel where news digests are also posted (server side).
NEWS_CHANNEL_ID = _env_int("NEWS_CHANNEL_ID", 0)

# How often the self-learning loop scrapes + re-indexes (seconds).
LEARN_INTERVAL = _env_int("LEARN_INTERVAL", 900)  # 15 min

# How often the release watcher checks followed titles (seconds).
WATCH_INTERVAL = _env_int("WATCH_INTERVAL", 1800)  # 30 min

# How often /start auto-news digests are sent (seconds).
BROADCAST_INTERVAL = _env_int("BROADCAST_INTERVAL", 1800)  # 30 min

# Max items kept per source in memory between runs.
MAX_CACHE = _env_int("MAX_CACHE", 500)

# How many image-less news items per source get an og:image page fetch
# (bounded so scraping stays fast).
OG_IMAGE_LIMIT = _env_int("OG_IMAGE_LIMIT", 12)

# User-Agent used for scraping so sites don't 403 us immediately.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Confidence threshold (0-100) above which a match is considered "certain".
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "78"))

# ---------------------------------------------------------------------------
# News sources (RSS) — now broad, not just 8 fixed feeds.
# The bot also crawls the whole web via DDG (fetch_web_news) in parallel,
# so it’s not limited to these specific feeds.
# ---------------------------------------------------------------------------
RSS_SOURCES: list[dict] = [
    {"name": "Anime News Network", "url": "https://www.animenewsnetwork.com/all/rss.xml", "kind": "news"},
    {"name": "Crunchyroll News", "url": "https://www.crunchyroll.com/news/rss", "kind": "news"},
    {"name": "r/anime", "url": "https://www.reddit.com/r/anime/hot/.rss", "kind": "community"},
    {"name": "r/manga", "url": "https://www.reddit.com/r/manga/hot/.rss", "kind": "community"},
    {"name": "r/manhwa", "url": "https://www.reddit.com/r/manhwa/hot/.rss", "kind": "community"},
    {"name": "MangaUpdates", "url": "https://www.mangaupdates.com/rss.php", "kind": "news"},
    {"name": "Comicbook.com Anime", "url": "https://comicbook.com/anime/feed/", "kind": "news"},
    {"name": "Siliconera", "url": "https://www.siliconera.com/feed/", "kind": "news"},
    # Additional broad sources — whole-web coverage, not just the original 8
    {"name": "MyAnimeList News", "url": "https://myanimelist.net/rss/news.xml", "kind": "news"},
    {"name": "Anime Corner", "url": "https://animecorner.me/feed/", "kind": "news"},
    {"name": "Anime Trending News", "url": "https://anitrendz.net/news/feed/", "kind": "news"},
    {"name": "Manga Tokyo", "url": "https://manga.tokyo/feed/", "kind": "news"},
    {"name": "OtakuKart", "url": "https://otakukart.com/feed/", "kind": "news"},
    {"name": "Anime UK News", "url": "https://animeuknews.net/feed/", "kind": "news"},
]

# Extra sites that are scraped directly (HTML) for release pages.
SCRAPE_TARGETS: list[dict] = [
    {"name": "LiveChart", "url": "https://www.livechart.me/spring-2026/tv", "kind": "anime", "selector": "a.title"},
    {"name": "AniList Trending", "url": "https://anilist.co/search/anime", "kind": "anime", "selector": ""},
]

# External APIs
ANILIST_URL = "https://graphql.anilist.co"
JIKAN_URL = "https://api.jikan.moe/v4"

# ---------------------------------------------------------------------------
# EverythingMoe — a curated public index of (mostly free) anime/manga/manhwa
# streaming & reading sites. We treat it as a *valuable resource* and learn the
# currently-live free hosts from it. We only ever link; we never host or fetch
# the media itself.
# ---------------------------------------------------------------------------
EVERYTHINGMOE_BASE = "https://everythingmoe.com"
EMOE_SECTIONS = [
    ("anime", "anime"),
    ("manga", "manga"),
    ("manhwa", "manhwa"),
]
# How often the free-host index is refreshed (seconds).
EMOE_REFRESH = _env_int("EMOE_REFRESH", 3600)  # 1h

# Embed theme color (AniList blue)
THEME_COLOR = 0x02A9FF

# ---------------------------------------------------------------------------
# Concurrency / parallelism tunables (override via env).
# These bound fan-out so exhaustive crawls go fast WITHOUT 429s/timeouts.
# ---------------------------------------------------------------------------
# Max parallel aiohttp fetches inside one news cycle (RSS + web + scrape).
FETCH_CONCURRENCY = _env_int("FETCH_CONCURRENCY", 16)
# Max parallel og:image page fetches per RSS source (was serial).
OG_IMAGE_CONCURRENCY = _env_int("OG_IMAGE_CONCURRENCY", 8)
# Max parallel blocking DDG searches (DDGS is sync + rate-limited).
DDG_CONCURRENCY = _env_int("DDG_CONCURRENCY", 6)
# Max parallel title-search sources inside fetch_exhaustive_search.
SEARCH_CONCURRENCY = _env_int("SEARCH_CONCURRENCY", 12)
# Max parallel Discord DM sends in a burst (Discord rate-limits hard).
DM_CONCURRENCY = _env_int("DM_CONCURRENCY", 4)
# Worker threads for CPU-bound fuzzy scoring (rapidfuzz releases the GIL).
MATCHER_WORKERS = _env_int("MATCHER_WORKERS", 8)
