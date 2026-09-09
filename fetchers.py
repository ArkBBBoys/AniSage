"""Fetchers: RSS news, AniList GraphQL, Jikan (MAL) + generic HTML scraping.

Everything is async and shares one aiohttp session. The bot treats these as
*raw experience* -- it scrapes widely, then the knowledge DB distills it.
"""
from __future__ import annotations

import asyncio
import functools
import re
import time as _time

import feedparser
from bs4 import BeautifulSoup

import concurrency
import config
from concurrency import DDG_SEM, HOST_SEM, HTTP_SEM, OG_SEM, RSS_SEM
from database import NewsItem, TitleRecord
from matcher import normalize, parse_query, score_pair

HEADERS = {"User-Agent": config.USER_AGENT}


def _loop_now() -> float:
    try:
        return asyncio.get_running_loop().time()
    except RuntimeError:
        return _time.monotonic()


async def _parse_html(text: str):
    """Parse HTML in a worker thread so the event loop never blocks on BS4."""
    return await concurrency.run_blocking(BeautifulSoup, text, "html.parser")


async def _parse_feed(text: str):
    """Parse an RSS feed in a worker thread (feedparser is CPU-bound)."""
    return await concurrency.run_blocking(feedparser.parse, text)


def _summary_text(html: str) -> str:
    """Blocking summary extraction (runs in worker threads via map)."""
    if not html:
        return ""
    try:
        return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    except Exception:
        return ""


def _html_img_url(html: str) -> str:
    """First real <img> URL inside a summary/content blob.

    Skips data URIs, SVGs and tracking pixels -- none of them render in Discord.
    """
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for img in soup.find_all("img"):
        src = (img.get("src") or "").strip()
        low = src.lower()
        if not src.startswith("http"):
            continue
        if low.startswith("data:") or low.endswith(".svg") or "pixel" in low:
            continue
        return src
    return ""


async def _og_image(session, url: str) -> str:
    """Read the og:image meta tag from an article page (RSS often omits images).

    Semaphore-capped: callers fan out with gather_capped instead of serial await.
    """
    if not url or not url.startswith("http"):
        return ""
    try:
        async with OG_SEM:
            async with session.get(url, headers=HEADERS, timeout=15) as r:
                if r.status != 200:
                    return ""
                text = await r.text()
        soup = await _parse_html(text)
        for prop in ("og:image", "og:image:url", "twitter:image"):
            tag = soup.find("meta", attrs={"property": prop}) or \
                soup.find("meta", attrs={"name": prop})
            if tag and tag.get("content"):
                return tag["content"].strip()
    except Exception as ex:
        print(f"[og-image] {url[:60]} failed: {ex}")
    return ""


# --------------------------------------------------------------------- RSS — now self-learning
def _parse_rss_entry(e: dict) -> tuple[str, str, str, str, str]:
    """Blocking RSS entry parse (summary + image). Runs in worker threads."""
    title = (e.get("title") or "").strip()
    link = e.get("link") or ""
    summary = ""
    for key in ("summary", "description"):
        if key in e and e[key]:
            summary = _summary_text(e[key])
            if summary:
                break
    img = ""
    try:
        if "media_content" in e and e["media_content"]:
            img = e["media_content"][0].get("url", "") or ""
    except Exception:
        pass
    if not img:
        img = _html_img_url(e.get("summary") or "")
    if not img and e.get("content"):
        try:
            img = _html_img_url(e["content"][0].get("value", ""))
        except Exception:
            pass
    pub = e.get("published") or e.get("updated") or ""
    return title, link, summary[:600], img, pub


async def fetch_rss(session, source: dict) -> list[NewsItem]:
    out: list[NewsItem] = []
    t0 = _loop_now()
    url = source.get("url", "")
    try:
        async with RSS_SEM:
            async with session.get(url, headers=HEADERS, timeout=20) as r:
                text = await r.text()
        parsed = await _parse_feed(text)
        entries = list(parsed.entries[: config.MAX_CACHE])
        # Threaded entry parsing: BS4 summary/img extraction is CPU-bound and
        # was serial per entry. Batch it across the blocking pool.
        if entries and len(entries) > 8:
            loop = asyncio.get_running_loop()

            def _parse_all(es):
                import concurrent.futures as _cf
                with _cf.ThreadPoolExecutor(max_workers=8) as pool:
                    return list(pool.map(_parse_rss_entry, es))

            parsed_rows = await loop.run_in_executor(
                concurrency.BLOCKING_EXECUTOR, _parse_all, entries)
        else:
            parsed_rows = [_parse_rss_entry(e) for e in entries]
        for title, link, summary, img, pub in parsed_rows:
            out.append(NewsItem(
                source=source["name"], kind=source["kind"], title=title,
                url=link, summary=summary, image=img, published=pub,
            ))
        # Parallel og:image fan-out (was: serial await per item, each up to
        # 15s). Capped by OG_SEM inside _og_image + gather limit.
        needy = [it for it in out if not it.image][: config.OG_IMAGE_LIMIT]
        if needy:
            imgs = await asyncio.gather(
                *(_og_image(session, it.url) for it in needy),
                return_exceptions=True)
            for it, im in zip(needy, imgs):
                if isinstance(im, str) and im:
                    it.image = im
        # Self-learning: update source stats via DB if available (passed via source dict)
        try:
            db = source.get("_db_ref")
            if db and url:
                latency = (_loop_now() - t0) * 1000
                # Run in DB pool to not block
                await concurrency.run_db(
                    db.update_news_source_stats, url, True, len(out), latency)
        except Exception:
            pass
    except Exception as ex:
        print(f"[rss] {source.get('name', url)} failed: {ex}")
        try:
            db = source.get("_db_ref") if isinstance(source, dict) else None
            if db and url:
                await concurrency.run_db(
                    db.update_news_source_stats, url, False, 0, 0)
        except Exception:
            pass
    return out


async def fetch_many_rss(session, sources: list[dict]) -> list[NewsItem]:
    """Parallel RSS fan-out across sources (capped by RSS_SEM)."""
    if not sources:
        return []
    results = await asyncio.gather(
        *(fetch_rss(session, s) for s in sources), return_exceptions=True)
    out: list[NewsItem] = []
    for res in results:
        if isinstance(res, Exception) or not res:
            continue
        out.extend(res)
    return out

async def discover_rss_sources_via_web(session, db, limit: int = 6) -> list[dict]:
    """Self-learning: discover new RSS feeds by crawling the web (DDG).

    Parallel: DDG queries fan out together, candidate probes fan out together.
    """
    candidates: list[dict] = []
    queries = [
        "anime news RSS feed xml",
        "manga news RSS feed",
    ]
    try:
        # Use DDG web search to find RSS URLs — parallel, semaphore-capped.
        def _ddg_search(q_):
            try:
                from ddgs import DDGS as _DDGS
            except ImportError:
                from duckduckgo_search import DDGS as _DDGS  # type: ignore
            with _DDGS() as ddgs:
                return list(ddgs.text(q_, max_results=8))

        async def _one_query(q):
            try:
                async with DDG_SEM:
                    res = await concurrency.run_blocking(_ddg_search, q)
                return res or []
            except Exception as ex:
                print(f"[discover] query {q!r} failed: {ex}")
                return []

        ddg_results = await asyncio.gather(*(_one_query(q) for q in queries))
        for res in ddg_results:
            if isinstance(res, Exception) or not res:
                continue
            for r in res:
                href = r.get("href") or r.get("url") or ""
                title = r.get("title") or ""
                if not href or len(href) < 10:
                    continue
                low = href.lower()
                # Heuristic: RSS-like URL
                if any(x in low for x in [".xml", "/rss", "/feed", "rss.xml", "atom.xml", "feed.xml"]):
                    candidates.append({"url": href, "name": title[:60] or href[:40], "kind": "news", "discovered_via": "web_discovery"})
                # Also extract any RSS link from body snippet
                body = r.get("body") or ""
                # Find URLs in body that look like RSS
                for m in re.findall(r"https?://[^\s\"']+\.(?:xml|rss)", body):
                    candidates.append({"url": m, "name": m[:50], "kind": "news", "discovered_via": "web_discovery"})
        # Also try to scrape known anime news hubs for <link rel=alternate type=application/rss+xml>
        hubs = ["https://www.animenewsnetwork.com", "https://myanimelist.net", "https://anitrendz.net"]
        async def _hub_links(hub: str):
            try:
                async with HTTP_SEM:
                    async with session.get(hub, headers=HEADERS, timeout=12) as r:
                        if r.status != 200:
                            return []
                        html = await r.text()
                soup = await _parse_html(html)
                found = []
                for link in soup.find_all("link", attrs={"type": re.compile(r"rss|atom", re.I)}):
                    href = link.get("href") or ""
                    if href.startswith("/"):
                        href = hub.rstrip("/") + href
                    if href and href.startswith("http"):
                        found.append({"url": href, "name": f"Discovered from {hub}", "kind": "news", "discovered_via": "hub_parse"})
                return found
            except Exception:
                return []

        hub_results = await asyncio.gather(*(_hub_links(h) for h in hubs[:1]))
        for found in hub_results:
            if isinstance(found, list):
                candidates.extend(found)
        # De-dup
        seen = set()
        uniq = []
        for c in candidates:
            if c["url"] not in seen:
                seen.add(c["url"])
                uniq.append(c)
        # Test each candidate with a quick fetch — parallel, capped.

        async def _probe(cand: dict):
            try:
                async with HTTP_SEM:
                    async with session.get(cand["url"], headers=HEADERS, timeout=10) as r:
                        txt = await r.text()
                # Quick check: contains <rss or <feed or <channel>
                low = txt[:2000].lower()
                if "<rss" in low or "<feed" in low or "<channel" in low:
                    # Try full parse to ensure it yields items
                    parsed = await _parse_feed(txt)
                    if len(parsed.entries) >= 1:
                        return cand
            except Exception:
                pass
            return None

        probes = await asyncio.gather(*(_probe(c) for c in uniq[:8]))
        tested = [c for c in probes if isinstance(c, dict)]
        tested = tested[:limit]
        # Persist to DB via self-learning
        if tested:
            try:
                added = await concurrency.run_db(db.discover_new_sources_from_web, tested)
                print(f"[discover] added {added} new RSS sources via web")
            except Exception as ex:
                print(f"[discover] db add failed: {ex}")
        return tested[:limit]
    except Exception as ex:
        print(f"[discover] failed: {ex}")
        return []


# ----------------------------------------------------------------- AniList
async def _anilist(session, query: str, variables: dict) -> dict:
    try:
        async with HTTP_SEM:
            async with session.post(
                config.ANILIST_URL, json={"query": query, "variables": variables},
                headers={**HEADERS, "Content-Type": "application/json"}, timeout=20,
            ) as r:
                data = await r.json()
    except Exception as ex:
        print(f"[anilist] request failed: {ex}")
        return {}
    if isinstance(data, dict) and data.get("errors"):
        print(f"[anilist] API error: {data['errors'][0].get('message')}")
        return {}
    return data or {}


def _media_type(m) -> str:
    t = m.get("type")
    if t == "ANIME":
        return "anime"
    country = m.get("countryOfOrigin")
    if country == "KR":
        return "manhwa"
    if country == "CN":
        return "manhua"
    return "manga"


def _links(m) -> tuple[list[str], list[str]]:
    watch, read = [], []
    for lk in m.get("externalLinks") or []:
        url = lk.get("url")
        site = (lk.get("site") or "").lower()
        if not url:
            continue
        if any(k in site for k in ("crunchyroll", "funimation", "netflix", "hulu",
                                    "disney", "prime", "hidive", "bilibili", "youtube")):
            watch.append(url)
        elif any(k in site for k in ("mangaplus", "viz", "webtoon", "kodansha",
                                     "shueisha", "yenpress", "comikey", "azuki")):
            read.append(url)
        else:
            read.append(url)
    # official site url always legal
    if m.get("siteUrl"):
        if _media_type(m) == "anime":
            watch.append(m["siteUrl"])
        else:
            read.append(m["siteUrl"])
    return watch, read


_TRENDING_Q = """
query ($type: MediaType, $per: Int) {
  Page(perPage: $per) {
    media(sort: TRENDING_DESC, type: $type) {
      id type countryOfOrigin title { romaji english native }
      coverImage { large } bannerImage
      siteUrl externalLinks { url site }
    }
  }
}"""

_SEARCH_Q = """
query ($search: String, $type: MediaType) {
  Page(perPage: 5) {
    media(search: $search, type: $type) {
      id type countryOfOrigin title { romaji english native }
      coverImage { large } bannerImage
      siteUrl externalLinks { url site }
    }
  }
}"""


def _image(m) -> str:
    """AniList cover/banner image; covers are tiny, banner is the wide one."""
    cover = (m.get("coverImage") or {}).get("large") or ""
    return cover or (m.get("bannerImage") or "")


async def fetch_anilist_trending(session, media_type: str = "ANIME", per: int = 25) -> list[TitleRecord]:
    data = await _anilist(session, _TRENDING_Q, {"type": media_type, "per": per})
    recs: list[TitleRecord] = []
    try:
        for m in data["data"]["Page"]["media"]:
            titles = m["title"]
            canonical = titles.get("english") or titles.get("romaji") or titles.get("native") or ""
            if not canonical:
                continue
            aliases = [t for t in titles.values() if t and t != canonical]
            watch, read = _links(m)
            recs.append(TitleRecord(
                key=normalize(canonical), canonical=canonical,
                media_type=_media_type(m), external_id=str(m["id"]),
                anilist_id=str(m["id"]), image=_image(m), aliases=aliases,
                watch_links=watch, read_links=read,
            ))
    except Exception as ex:
        print(f"[anilist] trending failed: {ex}")
    return recs


async def fetch_anilist_search(session, query: str, media_type: str = "ANIME") -> dict | None:
    data = await _anilist(session, _SEARCH_Q, {"search": query, "type": media_type})
    try:
        media = data["data"]["Page"]["media"]
        if not media:
            return None
        m = media[0]
        titles = m["title"]
        canonical = titles.get("english") or titles.get("romaji") or titles.get("native") or ""
        watch, read = _links(m)
        return {
            "canonical": canonical, "key": normalize(canonical),
            "media_type": _media_type(m), "anilist_id": str(m["id"]),
            "image": _image(m),
            "watch_links": watch, "read_links": read,
            "site_url": m.get("siteUrl", ""),
        }
    except Exception as ex:
        print(f"[anilist] search failed: {ex}")
        return None


# ------------------------------------------------------------------- Jikan
async def fetch_jikan_search(session, query: str, kind: str = "anime") -> dict | None:
    url = f"{config.JIKAN_URL}/{kind}"
    try:
        async with HTTP_SEM:
            async with session.get(url, params={"q": query, "limit": 5},
                                   headers=HEADERS, timeout=20) as r:
                if r.status != 200:
                    print(f"[jikan] {query!r} {kind} status {r.status}")
                    return None
                data = await r.json()
        if not data.get("data"):
            return None
        # score all 5 and pick best via matcher — threaded batch.
        from matcher import score_many as _sm
        rows = list(data["data"][:5])

        def _best_sync():
            best = None
            best_s = -1.0
            for m in rows:
                cand = m.get("title") or m.get("title_english") or m.get("title_japanese") or ""
                if not cand:
                    continue
                variants = [cand]
                alt = m.get("title_english") or ""
                if alt and alt != cand:
                    variants.append(alt)
                scores = _sm(query, variants)
                s = max(scores) if scores else 0.0
                if s > best_s:
                    best_s = s
                    best = m
            return best, best_s

        try:
            best, best_s = await concurrency.run_cpu(_best_sync)
        except Exception:
            best, best_s = None, -1
        if not best:
            return None
        m = best
        canonical = m.get("title") or m.get("title_english") or ""
        # prefer english if available for display
        if m.get("title_english"):
            canonical = m["title_english"]
        links = [m.get("url")] if m.get("url") else []
        # also collect title synonyms for alias learning
        aliases = []
        for k in ("title_english","title_japanese","title_synonyms"):
            v = m.get(k)
            if isinstance(v, list):
                aliases.extend([x for x in v if x and x != canonical])
            elif isinstance(v, str) and v and v != canonical:
                aliases.append(v)
        return {
            "canonical": canonical, "key": normalize(canonical),
            "media_type": kind, "mal_id": str(m.get("mal_id", "")),
            "watch_links": links, "read_links": links,
            "site_url": m.get("url", ""),
            "aliases": aliases,
            "score": best_s,
        }
    except Exception as ex:
        print(f"[jikan] search failed: {ex}")
        return None


# ------------------------------------------------------------------- Kitsu (independent of MAL/AniList)
async def fetch_kitsu_search(session, query: str, kind: str = "anime") -> dict | None:
    """Kitsu.io search — works even when AniList/Jikan/MAL are down.
    Returns same dict shape as other searchers, with score."""
    url = f"https://kitsu.io/api/edge/{kind}"
    params = {"filter[text]": query, "page[limit]": "5"}
    k_headers = {"User-Agent": config.USER_AGENT, "Accept": "application/vnd.api+json"}
    try:
        async with HTTP_SEM:
            async with session.get(url, params=params, headers=k_headers, timeout=15) as r:
                if r.status != 200:
                    # Kitsu returns 200 even for no results, so non-200 is real error
                    return None
                data = await r.json()
        items = data.get("data") or []
        if not items:
            return None
        from matcher import score_many as _sm

        def _best_sync():
            best = None
            best_s = -1.0
            best_cand = ""
            for entry in items:
                attr = entry.get("attributes") or {}
                cand = attr.get("canonicalTitle") or ""
                titles = attr.get("titles") or {}
                # try all title variants
                variants = [cand] + [v for v in titles.values() if v and v != cand]
                # also slug
                slug = attr.get("slug") or ""
                if slug:
                    variants.append(slug.replace("-", " "))
                variants = [v for v in variants if v]
                if not variants:
                    continue
                scores = _sm(query, variants)
                s = max(scores) if scores else 0.0
                if s > best_s:
                    best_s = s
                    best = entry
                    best_cand = cand
            return best, best_s, best_cand

        try:
            best, best_s, best_cand = await concurrency.run_cpu(_best_sync)
        except Exception:
            best, best_s, best_cand = None, -1, ""
        if not best:
            return None
        attr = best.get("attributes") or {}
        canonical = attr.get("canonicalTitle") or best_cand
        titles = attr.get("titles") or {}
        aliases = [v for v in titles.values() if v and v != canonical]
        # also abbreviatedTitles
        for ab in attr.get("abbreviatedTitles") or []:
            if ab and ab not in aliases and ab != canonical:
                aliases.append(ab)
        slug = attr.get("slug") or ""
        img = ""
        try:
            poster = attr.get("posterImage") or {}
            img = poster.get("large") or poster.get("original") or ""
        except Exception:
            pass
        links = []
        # Kitsu site url
        if best.get("links", {}).get("self"):
            # links.self is API url, build web url from slug/id
            links = [f"https://kitsu.io/{kind}/{attr.get('slug') or best.get('id')}"]
        return {
            "canonical": canonical, "key": normalize(canonical),
            "media_type": kind,  # keep requested kind
            "anilist_id": "", "mal_id": "",
            "image": img,
            "watch_links": links if kind == "anime" else [],
            "read_links": links if kind != "anime" else [],
            "site_url": links[0] if links else "",
            "aliases": aliases,
            "score": best_s,
        }
    except Exception as ex:
        print(f"[kitsu] search failed: {ex}")
        return None


async def fetch_ddg_titles(session, query: str, kind: str = "anime") -> list[dict]:
    """Generic web search via DuckDuckGo (ddgs library) — finds titles even when APIs are down.
    Uses the open web, not just fixed APIs. Runs DDGS().text() in a worker thread.

    Parallel: all query variants fan out together (capped by DDG_SEM) instead of
    serially awaiting each variant.
    """
    # Build queries: original + with kind context (anime/manga) — try both
    queries = [query]
    if kind and kind.lower() not in query.lower():
        queries.append(f"{query} {kind}")
    # also try without “the” for better web hit
    if query.lower().startswith("the "):
        queries.append(query[4:])

    def _ddg_search(q_):
        try:
            from ddgs import DDGS as _DDGS
        except ImportError:
            from duckduckgo_search import DDGS as _DDGS  # type: ignore
        with _DDGS() as ddgs:
            return list(ddgs.text(q_, max_results=5))

    async def _one(q):
        try:
            async with DDG_SEM:
                return await concurrency.run_blocking(_ddg_search, q)
        except Exception as ex:
            print(f"[ddg-titles] {query!r} {kind} q={q!r} failed: {ex}")
            return []

    ddg_batches = await asyncio.gather(*(_one(q) for q in queries))
    results_all = []
    for ddg_results in ddg_batches:
        if isinstance(ddg_results, Exception) or not ddg_results:
            continue
        for r in ddg_results[:5]:
            title = (r.get("title") or "").strip()
            href = (r.get("href") or "").strip()
            body = (r.get("body") or "").strip()
            if not title or len(title) < 3:
                continue
            # Two variants: raw title and cleaned
            variants = [title]
            clean = re.sub(r"\s*[-–|]\s*(Wikipedia|MyAnimeList|AniList|Kitsu|Crunchyroll|Fandom|IMDb).*$", "", title, flags=re.I).strip()
            clean = re.sub(r"\s*\(.*(TV|anime|manga).*\)\s*$", "", clean, flags=re.I).strip()
            if clean and clean != title and 2 <= len(clean) <= 80:
                variants.append(clean)
            # also use body snippet first sentence as potential title?
            for v in variants:
                if len(v) < 2 or len(v) > 90:
                    continue
                # filter obvious non-title results: if title is generic like “Wikipedia”
                if v.lower() in ("wikipedia", "fandom", "myanimelist"):
                    continue
                results_all.append({"canonical": v, "key": normalize(v), "media_type": kind, "aliases": [title] if title != v else [], "watch_links": [href] if href else [], "read_links": [], "site_url": href, "image": "", "body": body})
        if results_all and len(results_all) >= 3:
            break  # enough
    # Fallback to HTML scrape if ddgs gave nothing (rare) — parallel mirrors.
    if not results_all:
        async def _html_mirror(base_url: str):
            try:
                async with HTTP_SEM:
                    async with session.get(base_url, params={"q": query}, headers=HEADERS, timeout=12) as r:
                        if r.status != 200:
                            return []
                        html = await r.text()
                soup = await _parse_html(html)
                anchors = soup.select("a.result__a") or soup.find_all("a", href=True)
                tmp = []
                for a in anchors[:6]:
                    title = a.get_text(strip=True)
                    href = a.get("href") or ""
                    if not title or 3 > len(title) or len(title) > 90:
                        continue
                    if "uddg=" in href:
                        try:
                            import urllib.parse as up
                            qs = up.parse_qs(up.urlparse(href).query)
                            if "uddg" in qs:
                                href = up.unquote(qs["uddg"][0])
                        except Exception:
                            pass
                    tmp.append({"canonical": title, "key": normalize(title), "media_type": kind, "aliases": [], "watch_links": [href] if href else [], "read_links": [], "site_url": href, "image": ""})
                return tmp
            except Exception:
                return []
        mirror_results = await asyncio.gather(
            *(_html_mirror(u) for u in ["https://duckduckgo.com/html/", "https://html.duckduckgo.com/html/"]))
        for tmp in mirror_results:
            if tmp:
                results_all.extend(tmp)
                break
    return results_all[:8]

async def fetch_mal_scrape(session, query: str, kind: str = "anime") -> list[dict]:
    """Direct MAL HTML scrape — bypasses Jikan when it’s down.
    Scrapes https://myanimelist.net/anime.php?q=… and extracts titles.
    Returns list of candidates.
    """
    base = "https://myanimelist.net"
    cat = "anime" if kind == "anime" else "manga"
    url = f"{base}/{cat}.php"
    params = {"q": query, "cat": cat}
    try:
        async with HTTP_SEM:
            async with session.get(url, params=params, headers=HEADERS, timeout=15) as r:
                if r.status != 200:
                    return []
                html = await r.text()
        soup = await _parse_html(html)
        results = []
        # MAL search results are in table with .hoverinfo_trigger or a[href*=/anime/] etc.
        for a in soup.select("a.hoverinfo_trigger")[:8]:
            title = a.get_text(strip=True)
            href = a.get("href") or ""
            if not title or len(title) < 2:
                continue
            # image from nearby img
            img = ""
            parent = a.find_parent("tr")
            if parent:
                img_tag = parent.find("img")
                if img_tag and img_tag.get("data-src"):
                    img = img_tag["data-src"]
                elif img_tag and img_tag.get("src"):
                    img = img_tag["src"]
            results.append({"canonical": title, "key": normalize(title), "media_type": kind, "aliases": [], "watch_links": [href] if href else [], "read_links": [], "site_url": href, "image": img})
        # fallback: any anime/manga links
        if not results:
            for a in soup.select(f"a[href*='/{cat}/']")[:8]:
                title = a.get_text(strip=True)
                if not title or len(title) > 80 or len(title) < 2:
                    continue
                href = a.get("href") or ""
                if "/anime/" not in href and "/manga/" not in href:
                    continue
                results.append({"canonical": title, "key": normalize(title), "media_type": kind, "aliases": [], "watch_links": [], "read_links": [], "site_url": href, "image": ""})
        return results
    except Exception as ex:
        print(f"[mal-scrape] {query!r} {kind} failed: {ex}")
        return []

async def fetch_web_news(session, query: str = "anime news", limit: int = 12) -> list[NewsItem]:
    """Generic web news via DuckDuckGo — not limited to RSS.
    Uses ddgs news/text search to crawl the whole web (not just fixed RSS).

    Blocking DDGS call is semaphore-capped and off the event loop.
    """
    items: list[NewsItem] = []
    # Try ddgs news first (most relevant)
    try:
        def _ddg_news(q_):
            try:
                from ddgs import DDGS as _DDGS
            except ImportError:
                from duckduckgo_search import DDGS as _DDGS  # type: ignore
            with _DDGS() as ddgs:
                # news search
                try:
                    return list(ddgs.news(q_, max_results=limit))
                except Exception:
                    return list(ddgs.text(q_ + " news", max_results=limit))
        async with DDG_SEM:
            ddg_res = await concurrency.run_blocking(_ddg_news, query)
        for r in (ddg_res or [])[:limit]:
            title = (r.get("title") or "").strip()
            href = (r.get("url") or r.get("href") or "").strip()
            body = (r.get("body") or "").strip()[:500]
            if not title or not href:
                continue
            # news results have date
            date = r.get("date") or r.get("published") or ""
            items.append(NewsItem(source=f"Web:{query}", kind="news", title=title, url=href, summary=body, published=str(date)))
        if items:
            return items[:limit]
    except Exception as ex:
        print(f"[web-news-ddgs] {query!r} failed: {ex}")
    # Fallback to HTML scrape
    try:
        url = "https://duckduckgo.com/html/"
        params = {"q": query + " news"}
        async with HTTP_SEM:
            async with session.get(url, params=params, headers=HEADERS, timeout=12) as r:
                if r.status != 200:
                    return items
                html = await r.text()
        soup = await _parse_html(html)
        for a in soup.select("a.result__a")[:limit]:
            title = a.get_text(strip=True)
            href = a.get("href") or ""
            snippet = ""
            parent = a.find_parent("div", class_="result")
            if parent:
                snip = parent.select_one(".result__snippet")
                if snip:
                    snippet = snip.get_text(" ", strip=True)[:500]
            if not title or not href or "duckduckgo.com" in href:
                continue
            items.append(NewsItem(source=f"Web:{query}", kind="news", title=title, url=href, summary=snippet, published=""))
        return items[:limit]
    except Exception as ex:
        print(f"[web-news] {query!r} failed: {ex}")
        return items

async def fetch_exhaustive_search(session, query: str, preferred_type: str = "anime", db=None) -> dict | None:
    """Truly exhaustive parallel search — self-learning sources.
    Fires ALL available sources in parallel (AniList, Jikan, Kitsu, DDG web, MAL scrape),
    weights candidates by self-learned source reliability (from TitleSearchSource),
    collects every candidate, scores each against the original query via matcher.

    Parallel: source fan-out is semaphore-capped, DB weight fetch + stats writes
    are off the event loop, candidate scoring is threaded.
    """
    # Self-learning: get source weights if db available (off the event loop)
    src_weights = {"anilist": 1.0, "kitsu": 1.0, "jikan": 1.0, "ddg": 1.0, "mal": 1.0}
    if db is not None:
        try:
            sdicts = await concurrency.run_db(db.get_title_search_sources)
            for sdict in sdicts or []:
                name = sdict.get("name", "")
                w = float(sdict.get("weight", 1.0))
                if name in src_weights:
                    src_weights[name] = w
                # Map anilist -> anilist, etc. DDG weight also applies to ddg titles
        except Exception:
            pass
    try:
        clean, _, _ = parse_query(query)
    except Exception:
        clean = query
    q = clean or query
    # Fire all sources in parallel — with per-source timing for self-learning
    tasks = []
    task_names = []

    def _track(name, coro):
        async def _wrapper():
            t0 = _time.monotonic()
            try:
                res = await coro
                latency = (_time.monotonic() - t0) * 1000
                # Success if res has canonical or list non-empty
                success = False
                if isinstance(res, dict) and res.get("canonical"):
                    success = True
                elif isinstance(res, list) and len(res) > 0:
                    success = True
                elif res is not None and not isinstance(res, Exception):
                    # For anilist single dict, already handled
                    success = bool(res)
                if db is not None:
                    try:
                        # Map task name to source key
                        key = name.split("_")[0]  # anilist_anime -> anilist
                        if key in ["anilist", "kitsu", "jikan", "ddg", "mal"]:
                            await concurrency.run_db(db.update_title_source_stats, key, success, latency)
                    except Exception:
                        pass
                return res
            except Exception as ex:
                if db is not None:
                    try:
                        key = name.split("_")[0]
                        if key in src_weights:
                            await concurrency.run_db(db.update_title_source_stats, key, False, 0)
                    except Exception:
                        pass
                raise ex
        return _wrapper()

    # Wrap each source with tracking
    tasks.append(_track("anilist_anime", fetch_anilist_search(session, q, "ANIME"))); task_names.append("anilist")
    tasks.append(_track("anilist_manga", fetch_anilist_search(session, q, "MANGA"))); task_names.append("anilist")
    tasks.append(_track("jikan_anime", fetch_jikan_search(session, q, "anime"))); task_names.append("jikan")
    tasks.append(_track("jikan_manga", fetch_jikan_search(session, q, "manga"))); task_names.append("jikan")
    tasks.append(_track("kitsu_anime", fetch_kitsu_search(session, q, "anime"))); task_names.append("kitsu")
    tasks.append(_track("kitsu_manga", fetch_kitsu_search(session, q, "manga"))); task_names.append("kitsu")
    tasks.append(_track("ddg_anime", fetch_ddg_titles(session, q, "anime"))); task_names.append("ddg")
    tasks.append(_track("ddg_manga", fetch_ddg_titles(session, q, "manga"))); task_names.append("ddg")
    tasks.append(_track("mal_anime", fetch_mal_scrape(session, q, "anime"))); task_names.append("mal")
    tasks.append(_track("mal_manga", fetch_mal_scrape(session, q, "manga"))); task_names.append("mal")

    # Also try a secondary web search variant: query without stopwords
    q_norm = normalize(q)
    if q_norm and q_norm != q.lower():
        tasks.append(_track("ddg_norm", fetch_ddg_titles(session, q_norm, preferred_type))); task_names.append("ddg")
        tasks.append(_track("mal_norm", fetch_mal_scrape(session, q_norm, preferred_type))); task_names.append("mal")

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Threaded candidate scoring: score_pair is CPU-bound (5x rapidfuzz per
    # variant). Score all variants in one worker-pool batch instead of
    # serially blocking the loop.
    def _score_candidates_sync():
        from matcher import score_many
        scored: list[tuple[float, dict]] = []
        # Flatten variants first, then batch-score per query string.
        flat: list[tuple[int, str, str]] = []  # (result_idx, which_query, variant)
        flat_items: list[dict] = []
        flat_src_w: list[float] = []
        for idx, res in enumerate(results):
            if isinstance(res, Exception) or res is None:
                continue
            src_name = task_names[idx] if idx < len(task_names) else "unknown"
            src_key = src_name.split("_")[0] if "_" in src_name else src_name
            src_w = src_weights.get(src_key, 1.0) if src_key in src_weights else 1.0
            if src_key in ["ddg", "mal"] and "norm" in src_name:
                src_w = src_weights.get(src_key, 1.0) * 0.95
            items = res if isinstance(res, list) else [res]
            for r in items:
                if not r or not r.get("canonical"):
                    continue
                flat_items.append(r)
                flat_src_w.append(src_w)
        # Per-item best over variants via threaded score_many
        for r, src_w in zip(flat_items, flat_src_w):
            variants = [r.get("canonical", "")] + (r.get("aliases", []) or [])
            if r.get("key"):
                variants.append(r["key"].replace("-", " "))
            variants = [v for v in variants if v]
            if not variants:
                continue
            scores_q = score_many(query, variants)
            best_sc = max(scores_q) if scores_q else 0.0
            if q != query:
                scores_q2 = score_many(q, variants)
                if scores_q2:
                    best_sc = max(best_sc, max(scores_q2))
            if r.get("media_type") == preferred_type:
                best_sc += 1.5
            if "score" in r and isinstance(r["score"], (int, float)):
                best_sc = max(best_sc, float(r["score"]) * 0.95)
            best_sc = best_sc * (0.78 + 0.44 * src_w)
            scored.append((best_sc, r))
        return scored

    try:
        candidates = await concurrency.run_cpu(_score_candidates_sync)
    except Exception:
        candidates = []
        for idx, res in enumerate(results):
            if isinstance(res, Exception) or res is None:
                continue
            src_name = task_names[idx] if idx < len(task_names) else "unknown"
            src_key = src_name.split("_")[0] if "_" in src_name else src_name
            src_w = src_weights.get(src_key, 1.0)
            items = res if isinstance(res, list) else [res]
            for r in items:
                if r and r.get("canonical"):
                    candidates.append((50.0 * (0.78 + 0.44 * src_w), r))

    # Generic acronym expansion — no hardcoding: for short acronyms like “BTTH”, “JJK”, use DDG to discover the full title
    q_is_acronym = (len(q.strip()) <= 5 and q.strip().isupper() and q.strip().isalpha()) or (len(q.split()) == 1 and 2 <= len(q.strip()) <= 5 and q.strip().isupper())
    if q_is_acronym:
        # Always try to expand acronyms via web, even if current best looks high (Reddit posts often score 100 for acronyms but are not anime titles)
        # We will filter Reddit noise later, but first discover the true expansion.
        # Parallel: expansion lookups (recursive exhaustive + kitsu anime/manga)
        # fan out together instead of serial awaits.
        try:
            ddg_acr = await fetch_ddg_titles(session, q, preferred_type)
            expanded = ""
            expanded_cand = None
            for cand in ddg_acr:
                title = cand.get("canonical") or ""
                if len(title) >= 10 and normalize(title) != normalize(q) and not any(x in title.lower() for x in ["reddit", "quora", "btthj"]):
                    exp = re.sub(r"\s*-\s*Wikipedia.*$", "", title, flags=re.I).strip()
                    exp = re.sub(r"\s*\|\s*Fandom.*$", "", exp, flags=re.I).strip()
                    if len(exp) >= 8 and exp.lower() != q.lower():
                        expanded, expanded_cand = exp, cand
                        break
            if expanded:
                exp_res, r_anime, r_manga = await asyncio.gather(
                    fetch_exhaustive_search(session, expanded, preferred_type, db),
                    fetch_kitsu_search(session, expanded, "anime"),
                    fetch_kitsu_search(session, expanded, "manga"),
                    return_exceptions=True)
                try:
                    if isinstance(exp_res, dict) and exp_res.get("canonical"):
                        sc_exp = score_pair(query, exp_res["canonical"])
                        bonus = 14 if len(expanded) > 10 else 0
                        boosted = max(sc_exp, 102 + bonus, score_pair(expanded, exp_res["canonical"]) * 0.97)
                        candidates.append((boosted, exp_res))
                except Exception:
                    pass
                for r2 in (r_anime, r_manga):
                    try:
                        if isinstance(r2, dict) and r2.get("canonical"):
                            candidates.append((score_pair(query, r2["canonical"]) + 18, r2))
                            candidates.append((score_pair(expanded, r2["canonical"]) + 5, r2))
                    except Exception:
                        pass
                if expanded_cand is not None and (
                        "wikipedia" in (expanded_cand.get("site_url", "") or "").lower()
                        or "fandom" in (expanded_cand.get("site_url", "") or "").lower()
                        or len(expanded.split()) >= 2):
                    ddg_sc = 103.0
                    candidates.append((ddg_sc, {"canonical": expanded, "key": normalize(expanded), "media_type": preferred_type, "aliases": [q, expanded_cand.get("canonical", "")], "watch_links": [], "read_links": [], "site_url": expanded_cand.get("site_url", ""), "image": "", "score": ddg_sc}))
        except Exception as ex:
            print(f"[acronym-expand] {q!r} failed: {ex}")

    # Also try broader variant without leading “the” if nothing strong
    if not candidates or max(s for s,_ in candidates) < 58:
        if q.lower().startswith("the "):
            alt_q = q[4:].strip()
            if alt_q and len(alt_q) >= 3:
                try:
                    alt_res = await fetch_exhaustive_search(session, alt_q, preferred_type, db)
                    if alt_res:
                        sc_alt = score_pair(query, alt_res["canonical"]) * 0.92
                        candidates.append((sc_alt, alt_res))
                except Exception:
                    pass

    # Filter out obvious non-title DDG noise for acronym queries: drop Reddit/YouTube/Quora generic if a better Wikipedia/Fandom candidate exists
    if q_is_acronym:
        has_good = any("wikipedia" in (c[1].get("site_url","").lower()) or "fandom" in (c[1].get("site_url","").lower()) or "myanimelist" in (c[1].get("site_url","").lower()) for c in candidates)
        if has_good:
            candidates = [c for c in candidates if not (
                any(x in c[1].get("site_url","").lower() for x in ["reddit","youtube","youtu.be","quora","tiktok","twitter","x.com"])
                and c[0] > 85
            )]
            # also drop overly generic short titles that are just the acronym plus site name
            candidates = [c for c in candidates if not (
                c[1].get("canonical","").lower().strip() in ["btth - youtube", "btthj", "btth"] and c[0] > 90
            )]

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    # Generic web-noise filter: for any query, if best is low-quality Reddit/YouTube and a better non-noise exists, prefer it
    # This fixes “Unknown XYZ” returning Reddit 74 instead of None, and ensures unpopular titles aren’t masked by Reddit posts
    best_s, best_r = candidates[0]
    if any(x in (best_r.get("site_url") or "").lower() for x in ["reddit.com","youtube.com","youtu.be","quora.com"]) and best_s < 82:
        for s2, r2 in candidates[1:]:
            if not any(x in (r2.get("site_url") or "").lower() for x in ["reddit.com","youtube.com","youtu.be","quora.com"]):
                if s2 >= 55:
                    best_s, best_r = s2, r2
                    break
        else:
            # all remaining are also noise or low score — for truly unknown queries, return None instead of Reddit
            if best_s < 75:
                return None
    if best_s < 42:
        return None
    best_r["score"] = round(best_s, 1)
    # ensure aliases include original query for future alias hit
    best_r.setdefault("aliases", [])
    if normalize(query) != best_r.get("key"):
        try:
            al = best_r.get("aliases",[]) or []
            if query not in al and normalize(query) not in [normalize(a) for a in al]:
                best_r["aliases"] = al + [query]
        except Exception:
            pass
    return best_r


# ------------------------------------------------------------ generic scrape
async def scrape_html(session, target: dict) -> list[NewsItem]:
    out: list[NewsItem] = []
    try:
        async with HTTP_SEM:
            async with session.get(target["url"], headers=HEADERS, timeout=20) as r:
                html = await r.text()
        soup = await _parse_html(html)
        sel = target.get("selector")
        nodes = soup.select(sel) if sel else soup.select("a")
        for n in nodes[: config.MAX_CACHE]:
            title = n.get_text(strip=True)
            href = n.get("href", "")
            if not title or not href:
                continue
            if not href.startswith("http"):
                href = target["url"].rstrip("/") + "/" + href.lstrip("/")
            out.append(NewsItem(
                source=target["name"], kind=target.get("kind", "web"),
                title=title, url=href,
            ))
    except Exception as ex:
        print(f"[scrape] {target['name']} failed: {ex}")
    return out


# ------------------------------------------------------- EverythingMoe index
async def _fetch_emoe_section(session, section: str, kind: str) -> list[dict]:
    """One EverythingMoe section page (parallel fan-out helper)."""
    url = f"{config.EVERYTHINGMOE_BASE}/section/{section}"
    rows: list[dict] = []
    try:
        async with HTTP_SEM:
            async with session.get(url, headers=HEADERS, timeout=20) as r:
                if r.status != 200:
                    return []
                html = await r.text()
        soup = await _parse_html(html)
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not href.startswith("/s/"):
                continue
            slug = href.split("/s/")[-1].strip("/")
            if not slug:
                continue
            name = a.get_text(strip=True) or slug
            ctx = (a.parent.get_text(" ", strip=True) if a.parent else "")
            ctx_low = ctx.lower()
            if "graveyard" in ctx_low or "dead" in ctx_low:
                status = "graveyard"
            else:
                status = "alive"
            note = ""
            if "moved to" in ctx_low:
                note = ctx.split("moved to")[-1].strip(" .")
            rows.append({
                "slug": slug, "name": name, "kind": kind,
                "page_url": config.EVERYTHINGMOE_BASE + href,
                "status": status, "note": note,
            })
    except Exception as ex:
        print(f"[emoe] section {section} failed: {ex}")
    return rows


async def fetch_everythingmoe_index(session) -> list[dict]:
    """Scrape EverythingMoe's section pages for the currently-live free hosts.

    Returns a list of {slug, name, kind, page_url, status, note}. We only
    collect the EverythingMoe *page* for each host (the outbound link is
    JS-gated). EverythingMoe annotates dead hosts ("… is moved to Graveyard"),
    which lets the bot detect URL death and self-heal by rediscovering them.

    Parallel: all sections fetched together instead of serially.
    """
    section_results = await asyncio.gather(
        *(_fetch_emoe_section(session, section, kind)
          for section, kind in config.EMOE_SECTIONS),
        return_exceptions=True)
    out: list[dict] = []
    for res in section_results:
        if isinstance(res, Exception) or not res:
            continue
        out.extend(res)
    # de-dup by slug+kind (keep graveyard flag if any copy says so)
    seen = {}
    for d in out:
        k = (d["slug"], d["kind"])
        if k in seen:
            if d["status"] == "graveyard":
                seen[k]["status"] = "graveyard"
                seen[k]["note"] = d["note"] or seen[k]["note"]
        else:
            seen[k] = d
    return list(seen.values())


# ---------------------------------------------- direct-host discovery (links)
_JUNK_DOMAINS = {
    "everythingmoe.com", "github.com", "discord.com", "discord.gg",
    "twitter.com", "x.com", "reddit.com", "youtube.com", "patreon.com",
    "paypal.com", "megaup.net", "megaup.cc", "anonfiles.com", "buymeacoffee.com",
    "ko-fi.com", "google.com", "gstatic.com", "cloudflare.com", "w3.org",
    # interstitial / bypass / ad domains that sit in front of real hosts
    "aibrowsingapp.com", "get.aibrowsingapp.com", "browsebypass.com",
    "bypass.city", "shorturl.at", "bit.ly", "tinyurl.com", "adf.ly",
    "linkvertise.com", "loot-link.com", "rekonise.com",
}


def extract_domains(html: str) -> list[str]:
    found = re.findall(r"https?://([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", html)
    out, seen = [], set()
    for d in found:
        d = d.lower()
        if d in _JUNK_DOMAINS or d.endswith(".everythingmoe.com"):
            continue
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


async def resolve_host_domain(session, page_url: str) -> str | None:
    """From an EverythingMoe /s/ page, find the host's REAL current domain.

    Free-stream domains move constantly; this picks the first reachable one so
    the bot can link/crawl the actual site directly instead of only via the index.

    Parallel: all candidate domain probes fan out together (capped by HOST_SEM)
    and the first reachable in page order wins.
    """
    try:
        async with HTTP_SEM:
            async with session.get(page_url, headers=HEADERS, timeout=20) as r:
                html = await r.text()
    except Exception:
        return None
    domains = [d for d in extract_domains(html)
               if d not in _JUNK_DOMAINS and not d.endswith(".everythingmoe.com")]
    if not domains:
        return None

    async def _probe(dom: str) -> str | None:
        for scheme in ("https", "http"):
            base = f"{scheme}://{dom}"
            try:
                # Do NOT follow redirects: a 3xx still proves the domain resolves,
                # and following would land us on an interstitial/bypass page.
                async with HOST_SEM:
                    async with session.get(base, headers=HEADERS, timeout=10,
                                           allow_redirects=False) as r2:
                        if r2.status in (200, 301, 302, 307, 308):
                            return dom
            except Exception:
                continue
        return None

    results = await asyncio.gather(*(_probe(d) for d in domains[:12]),
                                   return_exceptions=True)
    for dom, res in zip(domains, results):
        if res == dom:
            return dom
    return None


async def learn_host_search(session, host: str) -> tuple[str | None, str | None]:
    """Learn a host's REAL search template (param name included) and the param.

    Returns (template, param), e.g. ('https://animekai.cc/search?q={q}', 'q') or
    ('https://aniwatch.to/search?keyword={q}', 'keyword'). The bot reads the
    site's actual <form> so it stores the *exact* query param that host expects
    -- never assumes 'q='. We only store the search *page link*, never
    episode/stream/video URLs.
    """
    import urllib.parse as up

    base = f"https://{host}" if "://" not in host else host
    try:
        async with session.get(base, headers=HEADERS, timeout=15,
                               allow_redirects=True) as r:
            html = await r.text()
            base = str(r.url).rstrip("/")
    except Exception:
        return None, None

    soup = await _parse_html(html)
    _QUERY_HINTS = ("q", "query", "search", "keyword", "term", "s", "name", "title")

    # 1) Learn from the site's real search <form> (method GET only -> deep-linkable)
    for form in soup.find_all("form"):
        method = (form.get("method") or "get").lower()
        if method != "get":
            continue
        action = form.get("action") or ""
        action_url = up.urljoin(base + "/", action) if action else base
        text_input = None
        for i in form.find_all("input"):
            t = (i.get("type") or "text").lower()
            n = i.get("name") or ""
            if t in ("text", "search", "") and n:
                text_input = n
                if any(k in n.lower() for k in _QUERY_HINTS):
                    break
        if not text_input:
            continue
        tmpl = f"{action_url}?{text_input}={{q}}"
        if await _verify_search(session, tmpl):
            return tmpl, text_input
        # keep as best-effort even if JS-rendered (form is authoritative)
        return tmpl, text_input

    # 2) Heuristic fallback: try common paths x common params, verify — parallel.
    probes = [f"{base}{path}?{param}={{q}}"
              for path in ("/search", "/browse", "/find", "/anime", "/manga")
              for param in _QUERY_HINTS]
    try:
        hits = await asyncio.gather(*(_verify_search(session, t) for t in probes))
        for tmpl, ok in zip(probes, hits):
            if ok is True:
                try:
                    param = tmpl.split("?")[1].split("=")[0]
                except Exception:
                    param = ""
                return tmpl, param
    except Exception:
        pass
    return None, None


async def _verify_search(session, tmpl: str) -> bool:
    """GET the template with a probe query; accept if it returns 200.

    JS-rendered result pages won't contain the literal query in raw HTML, so a
    200 is enough proof the endpoint exists (the form is the real authority).
    Semaphore-capped for parallel fallback probing.
    """
    url = tmpl.format(q="naruto")
    try:
        async with HOST_SEM:
            async with session.get(url, headers=HEADERS, timeout=12,
                                   allow_redirects=True) as r:
                return r.status == 200
    except Exception:
        return False


async def fetch_news_for_title(session, title: str, limit: int = 8, db=None) -> list[NewsItem]:
    """Exhaustive news search for a specific title — parallel web crawl.
    Self-learning: uses DB-ranked RSS (whole web, not fixed) + DDG + title-aware.

    Parallel: web/DDG queries + per-source RSS fan out together; relevance
    ranking is threaded.
    """
    queries = [f"{title} anime news", f"{title} manga news", title]
    tasks = []
    for q in queries:
        tasks.append(fetch_web_news(session, q, limit=limit))
        tasks.append(fetch_ddg_titles(session, q, "anime"))

    async def _one_rss(src: dict):
        try:
            if db and isinstance(src, dict):
                src = dict(src)
                src["_db_ref"] = db
            items = await fetch_rss(session, src)
            tl = title.lower()
            return [it for it in items
                    if tl in it.title.lower() or tl in (it.summary or "").lower()]
        except Exception:
            return []

    # Use self-learned RSS sources ordered by reliability — off the loop.
    try:
        if db is not None:
            srcs = await concurrency.run_db(_get_active_rss_sources_sync, db, "", 6)
        else:
            srcs = config.RSS_SOURCES[:4]
    except Exception:
        srcs = config.RSS_SOURCES[:4]
    for src in (srcs or [])[:4]:
        async def _wrap(s=src):
            return await _one_rss(s)
        tasks.append(_wrap())
    results = await asyncio.gather(*tasks, return_exceptions=True)
    items: list[NewsItem] = []
    for res in results:
        if isinstance(res, Exception) or res is None:
            continue
        lst = res if isinstance(res, list) else []
        for it in lst:
            if isinstance(it, NewsItem):
                items.append(it)
            elif isinstance(it, dict) and it.get("canonical"):
                # DDG title result -> convert to NewsItem
                items.append(NewsItem(source=f"Web:{title}", kind="news", title=it["canonical"], url=it.get("site_url",""), summary=it.get("body","")[:400], published=""))
    # De-dup and rank by title relevance to query — threaded scoring.
    seen = set()
    uniq = []
    pre = []
    for it in items:
        if it.url and it.url in seen:
            continue
        if it.url:
            seen.add(it.url)
        pre.append(it)
    if pre:
        try:
            from matcher import score_many as _score_many

            def _score_all():
                return _score_many(title, [it.title for it in pre])

            scores = await concurrency.run_cpu(_score_all)
        except Exception:
            scores = [score_pair(title, it.title) for it in pre]
        tl = title.lower()
        scored_pairs = []
        for it, sc in zip(pre, scores):
            if isinstance(sc, Exception):
                continue
            # Quick relevance filter: title should contain at least one word from query or be high scoring
            if sc < 35 and tl not in it.title.lower():
                # Allow if it's from web news for that exact title query (already filtered by DDG)
                if not it.source.startswith("Web:"):
                    continue
            scored_pairs.append((sc, it))
        # Sort by relevance to title
        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        uniq = [it for _, it in scored_pairs]
    return uniq[:limit]


def _get_active_rss_sources_sync(db, kind: str = "", limit: int = 12) -> list[dict]:
    """Sync helper so fetch_news_for_title can fetch sources via run_db."""
    return _get_active_rss_sources(db, kind=kind, limit=limit)

def _get_active_rss_sources(db=None, kind: str = "", limit: int = 12) -> list[dict]:
    """Self-learning: prefer DB-learned RSS sources ordered by score, fallback to config."""
    if db is not None:
        try:
            # Ensure DB is seeded from config on first run
            try:
                db.sync_news_sources_from_config()
            except: pass
            srcs = db.get_active_news_sources(kind=kind, limit=limit)
            if srcs:
                # Attach db ref for stats update in fetch_rss
                for s in srcs:
                    s["_db_ref"] = db
                return srcs
        except Exception as ex:
            print(f"[sources] DB get_active failed: {ex}")
    # Fallback to config
    return list(config.RSS_SOURCES[:limit])

async def fetch_news_exhaustive(session, limit: int = 20, db=None) -> list[NewsItem]:
    """Truly exhaustive news crawl — parallel across all web sources.
    Self-learning: uses DB-ranked RSS sources (by reliability) + whole-web DDG.
    Not limited to 3 fixed web queries; rotates through many and also discovers via DDG.

    Parallel: sources fetched with a concurrency cap (FETCH_CONCURRENCY) so a
    18-way fan-out goes fast without connection storms.
    """
    # Base: self-learned RSS (ordered by reliability) + scrape — DB off-loop.
    try:
        if db is not None:
            rss_srcs = await concurrency.run_db(_get_active_rss_sources_sync, db, "", 10)
        else:
            rss_srcs = list(config.RSS_SOURCES[:10])
    except Exception:
        rss_srcs = list(config.RSS_SOURCES[:10])
    try:
        cap = max(4, int(getattr(config, "FETCH_CONCURRENCY", 16)))
    except Exception:
        cap = 16
    sem = asyncio.Semaphore(cap)

    async def _capped(coro):
        async with sem:
            return await coro

    tasks = [_capped(fetch_rss(session, s)) for s in (rss_srcs or [])]
    tasks += [_capped(scrape_html(session, t)) for t in config.SCRAPE_TARGETS]
    # Web: many diverse queries to cover whole anime/manga web, not just “anime news”
    web_queries = [
        "anime news", "manga news", "manhwa news",
        "anime episode release", "manga chapter release",
        "anime trailer news", "anime film news",
        "weekly shonen jump news", "crunchyroll news",
        "anime trending news"
    ]
    # Pick 4 queries per cycle deterministically but rotate
    import random
    random.seed(int(_time.monotonic()) // 900)  # rotate every 15 min
    chosen = random.sample(web_queries, k=4)
    for q in chosen:
        tasks.append(_capped(fetch_web_news(session, q, limit=8)))
    # Also do DDG title searches for trending titles (to catch news via web titles)
    for q in ["anime 2026", "manga 2026"]:
        tasks.append(_capped(fetch_ddg_titles(session, q, "anime")))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    items: list[NewsItem] = []
    for res in results:
        if isinstance(res, Exception) or res is None:
            continue
        lst = res if isinstance(res, list) else []
        for it in lst:
            if isinstance(it, NewsItem):
                items.append(it)
            elif isinstance(it, dict) and it.get("canonical"):
                items.append(NewsItem(source="Web:trending", kind="news", title=it["canonical"], url=it.get("site_url",""), summary=it.get("body","")[:400], published=""))
    # De-dup
    seen = set()
    uniq = []
    for it in items:
        if it.url and it.url in seen:
            continue
        if it.url:
            seen.add(it.url)
        uniq.append(it)
    return uniq

# ------------------------------------------------------------------ runner
async def run_news_cycle(session, db=None) -> tuple[int, list[NewsItem]]:
    """Fetch ALL news — exhaustive parallel crawl (not just specific feeds).
    Self-learning: uses DB-ranked sources and updates reliability.
    """
    items = await fetch_news_exhaustive(session, limit=30, db=db)
    # De-dup already done, just return
    return len(items), items


if __name__ == "__main__":
    import aiohttp

    async def _t():
        async with aiohttp.ClientSession() as s:
            n, items = await run_news_cycle(s)
            print("news items:", n)
            recs = await fetch_anilist_trending(s, "ANIME", 5)
            print("trending:", [r.canonical for r in recs])

    asyncio.run(_t())
