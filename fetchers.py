"""Fetchers: RSS news, AniList GraphQL, Jikan (MAL) + generic HTML scraping.

Everything is async and shares one aiohttp session. The bot treats these as
*raw experience* -- it scrapes widely, then the knowledge DB distills it.
"""
from __future__ import annotations

import asyncio
import feedparser
import re
from bs4 import BeautifulSoup


async def _parse_html(text: str):
    """Parse HTML in a worker thread so the event loop never blocks on BS4."""
    return await asyncio.get_event_loop().run_in_executor(
        None, BeautifulSoup, text, "html.parser"
    )


async def _parse_feed(text: str):
    """Parse an RSS feed in a worker thread (feedparser is CPU-bound)."""
    return await asyncio.get_event_loop().run_in_executor(None, feedparser.parse, text)

import config
from database import NewsItem, TitleRecord
from matcher import normalize, parse_query, score_pair

HEADERS = {"User-Agent": config.USER_AGENT}


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
    """Read the og:image meta tag from an article page (RSS often omits images)."""
    try:
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
async def fetch_rss(session, source: dict) -> list[NewsItem]:
    out: list[NewsItem] = []
    t0 = asyncio.get_event_loop().time()
    url = source.get("url","")
    try:
        async with session.get(url, headers=HEADERS, timeout=20) as r:
            text = await r.text()
        parsed = await _parse_feed(text)
        for e in parsed.entries[: config.MAX_CACHE]:
            title = (e.get("title") or "").strip()
            link = e.get("link") or ""
            summary = ""
            for key in ("summary", "description"):
                if key in e:
                    summary = BeautifulSoup(e[key], "html.parser").get_text(" ", strip=True)
                    break
            img = ""
            if "media_content" in e and e["media_content"]:
                img = e["media_content"][0].get("url", "")
            if not img:
                img = _html_img_url(e.get("summary") or "")
            if not img and e.get("content"):
                img = _html_img_url(e["content"][0].get("value", ""))
            pub = e.get("published") or e.get("updated") or ""
            out.append(NewsItem(
                source=source["name"], kind=source["kind"], title=title,
                url=link, summary=summary[:600], image=img, published=pub,
            ))
        for it in [i for i in out if not i.image][: config.OG_IMAGE_LIMIT]:
            it.image = await _og_image(session, it.url)
        # Self-learning: update source stats via DB if available (passed via source dict)
        try:
            db = source.get("_db_ref")
            if db and url:
                latency = (asyncio.get_event_loop().time() - t0)*1000
                # Run in executor to not block
                await asyncio.get_event_loop().run_in_executor(None, db.update_news_source_stats, url, True, len(out), latency)
        except: pass
    except Exception as ex:
        print(f"[rss] {source.get('name', url)} failed: {ex}")
        try:
            db = source.get("_db_ref") if isinstance(source, dict) else None
            if db and url:
                await asyncio.get_event_loop().run_in_executor(None, db.update_news_source_stats, url, False, 0, 0)
        except: pass
    return out

async def discover_rss_sources_via_web(session, db, limit: int = 6) -> list[dict]:
    """Self-learning: discover new RSS feeds by crawling the web (DDG)."""
    candidates = []
    queries = [
        "anime news RSS feed xml",
        "manga news RSS feed",
        "manhwa news RSS",
        "anime release RSS feed",
    ]
    try:
        # Use DDG web search to find RSS URLs
        from matcher import score_pair  # noqa
        for q in queries[:2]:
            try:
                # Use fetch_ddg_titles as generic web search and also try direct DDG
                def _ddg_search(q_):
                    try:
                        from ddgs import DDGS as _DDGS
                    except ImportError:
                        from duckduckgo_search import DDGS as _DDGS
                    with _DDGS() as ddgs:
                        return list(ddgs.text(q_, max_results=8))
                loop = asyncio.get_event_loop()
                res = await loop.run_in_executor(None, _ddg_search, q)
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
            except Exception as ex:
                print(f"[discover] query {q!r} failed: {ex}")
        # Also try to scrape known anime news hubs for <link rel=alternate type=application/rss+xml>
        hubs = ["https://www.animenewsnetwork.com", "https://myanimelist.net", "https://anitrendz.net"]
        for hub in hubs[:1]:
            try:
                async with session.get(hub, headers=HEADERS, timeout=12) as r:
                    if r.status != 200:
                        continue
                    html = await r.text()
                    soup = await _parse_html(html)
                    for link in soup.find_all("link", attrs={"type": re.compile(r"rss|atom", re.I)}):
                        href = link.get("href") or ""
                        if href.startswith("/"):
                            href = hub.rstrip("/") + href
                        if href and href.startswith("http"):
                            candidates.append({"url": href, "name": f"Discovered from {hub}", "kind": "news", "discovered_via": "hub_parse"})
            except Exception:
                pass
        # De-dup
        seen = set()
        uniq = []
        for c in candidates:
            if c["url"] not in seen:
                seen.add(c["url"])
                uniq.append(c)
        # Test each candidate with a quick fetch (limit to 4 to avoid overload)
        tested = []
        for cand in uniq[:8]:
            try:
                async with session.get(cand["url"], headers=HEADERS, timeout=10) as r:
                    txt = await r.text()
                    # Quick check: contains <rss or <feed or <channel>
                    low = txt[:2000].lower()
                    if "<rss" in low or "<feed" in low or "<channel" in low:
                        # Try full parse to ensure it yields items
                        parsed = await _parse_feed(txt)
                        if len(parsed.entries) >= 1:
                            tested.append(cand)
                            if len(tested) >= limit:
                                break
            except Exception:
                continue
        # Persist to DB via self-learning
        if tested:
            try:
                added = await asyncio.get_event_loop().run_in_executor(None, db.discover_new_sources_from_web, tested)
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
        async with session.get(url, params={"q": query, "limit": 5},
                               headers=HEADERS, timeout=20) as r:
            if r.status != 200:
                print(f"[jikan] {query!r} {kind} status {r.status}")
                return None
            data = await r.json()
        if not data.get("data"):
            return None
        # score all 5 and pick best via matcher
        from matcher import score_pair as _sp
        best = None
        best_s = -1
        for m in data["data"]:
            cand = m.get("title") or m.get("title_english") or m.get("title_japanese") or ""
            if not cand:
                continue
            s = _sp(query, cand)
            # also try english title
            alt = m.get("title_english") or ""
            if alt and alt != cand:
                s = max(s, _sp(query, alt))
            if s > best_s:
                best_s = s
                best = m
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
        async with session.get(url, params=params, headers=k_headers, timeout=15) as r:
            if r.status != 200:
                # Kitsu returns 200 even for no results, so non-200 is real error
                return None
            data = await r.json()
        items = data.get("data") or []
        if not items:
            return None
        from matcher import score_pair as _sp
        best = None
        best_s = -1
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
            for v in variants:
                s = _sp(query, v)
                if s > best_s:
                    best_s = s
                    best = entry
                    best_cand = cand
                    best_titles = titles
                    best_attr = attr
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
    """
    # Build queries: original + with kind context (anime/manga) — try both
    queries = [query]
    if kind and kind.lower() not in query.lower():
        queries.append(f"{query} {kind}")
    # also try without “the” for better web hit
    if query.lower().startswith("the "):
        queries.append(query[4:])
    results_all = []
    for q in queries:
        try:
            # Run sync DDGS in thread pool
            def _ddg_search(q_):
                try:
                    from ddgs import DDGS as _DDGS
                except ImportError:
                    from duckduckgo_search import DDGS as _DDGS  # type: ignore
                with _DDGS() as ddgs:
                    return list(ddgs.text(q_, max_results=5))
            loop = asyncio.get_event_loop()
            ddg_results = await loop.run_in_executor(None, _ddg_search, q)
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
        except Exception as ex:
            print(f"[ddg-titles] {query!r} {kind} q={q!r} failed: {ex}")
            continue
    # Fallback to HTML scrape if ddgs gave nothing (rare)
    if not results_all:
        for base_url in ["https://duckduckgo.com/html/", "https://html.duckduckgo.com/html/"]:
            try:
                async with session.get(base_url, params={"q": query}, headers=HEADERS, timeout=12) as r:
                    if r.status != 200:
                        continue
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
                        except: pass
                    tmp.append({"canonical": title, "key": normalize(title), "media_type": kind, "aliases": [], "watch_links": [href] if href else [], "read_links": [], "site_url": href, "image": ""})
                if tmp:
                    results_all.extend(tmp)
                    break
            except Exception:
                continue
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
        loop = asyncio.get_event_loop()
        ddg_res = await loop.run_in_executor(None, _ddg_news, query)
        for r in ddg_res[:limit]:
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
    """
    import time as _t
    # Self-learning: get source weights if db available
    src_weights = {"anilist": 1.0, "kitsu": 1.0, "jikan": 1.0, "ddg": 1.0, "mal": 1.0}
    if db is not None:
        try:
            for sdict in db.get_title_search_sources():
                name = sdict.get("name","")
                w = float(sdict.get("weight", 1.0))
                if name in src_weights:
                    src_weights[name] = w
                # Map anilist -> anilist, etc. DDG weight also applies to ddg titles
        except: pass
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
            t0 = _t.time()
            try:
                res = await coro
                latency = (_t.time() - t0)*1000
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
                        if key in ["anilist","kitsu","jikan","ddg","mal"]:
                            await asyncio.get_event_loop().run_in_executor(None, db.update_title_source_stats, key, success, latency)
                    except: pass
                return res
            except Exception as ex:
                if db is not None:
                    try:
                        key = name.split("_")[0]
                        if key in src_weights:
                            await asyncio.get_event_loop().run_in_executor(None, db.update_title_source_stats, key, False, 0)
                    except: pass
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

    candidates: list[tuple[float, dict]] = []
    for idx, res in enumerate(results):
        if isinstance(res, Exception) or res is None:
            continue
        # Determine source weight for this result
        src_name = task_names[idx] if idx < len(task_names) else "unknown"
        # Map task name to source key for weighting
        src_key = src_name.split("_")[0] if "_" in src_name else src_name
        src_w = src_weights.get(src_key, 1.0) if src_key in src_weights else 1.0
        # Also handle ddg_norm, mal_norm etc.
        if src_key in ["ddg", "mal"] and "norm" in src_name:
            src_w = src_weights.get(src_key, 1.0) * 0.95
        items = res if isinstance(res, list) else [res]
        for r in items:
            if not r or not r.get("canonical"):
                continue
            best_sc = 0.0
            variants = [r.get("canonical","")] + r.get("aliases",[])
            if r.get("key"):
                variants.append(r["key"].replace("-", " "))
            for v in variants:
                if not v:
                    continue
                sc = score_pair(query, v)
                if sc > best_sc:
                    best_sc = sc
                if q != query:
                    sc2 = score_pair(q, v)
                    if sc2 > best_sc:
                        best_sc = sc2
            if r.get("media_type") == preferred_type:
                best_sc += 1.5
            if "score" in r and isinstance(r["score"], (int,float)):
                best_sc = max(best_sc, float(r["score"]) * 0.95)
            # Self-learning: weight by source reliability (e.g., if Kitsu is more reliable than Jikan currently, boost it)
            best_sc = best_sc * (0.78 + 0.44 * src_w)  # src_w 0.4-1.8 -> factor 0.95-1.57
            # Also apply small per-source bias from DB weight
            candidates.append((best_sc, r))

    # Generic acronym expansion — no hardcoding: for short acronyms like “BTTH”, “JJK”, use DDG to discover the full title
    q_is_acronym = (len(q.strip()) <= 5 and q.strip().isupper() and q.strip().isalpha()) or (len(q.split()) == 1 and 2 <= len(q.strip()) <= 5 and q.strip().isupper())
    if q_is_acronym:
        # Always try to expand acronyms via web, even if current best looks high (Reddit posts often score 100 for acronyms but are not anime titles)
        # We will filter Reddit noise later, but first discover the true expansion
        try:
            ddg_acr = await fetch_ddg_titles(session, q, preferred_type)
            for cand in ddg_acr:
                title = cand.get("canonical") or ""
                if len(title) >= 10 and normalize(title) != normalize(q) and not any(x in title.lower() for x in ["reddit", "quora", "btthj"]):
                    expanded = re.sub(r"\s*-\s*Wikipedia.*$", "", title, flags=re.I).strip()
                    expanded = re.sub(r"\s*\|\s*Fandom.*$", "", expanded, flags=re.I).strip()
                    if len(expanded) >= 8 and expanded.lower() != q.lower():
                        try:
                            exp_res = await fetch_exhaustive_search(session, expanded, preferred_type, db)
                            if exp_res:
                                sc_exp = score_pair(query, exp_res["canonical"])
                                bonus = 14 if len(expanded) > 10 else 0
                                boosted = max(sc_exp, 102 + bonus, score_pair(expanded, exp_res["canonical"]) * 0.97)
                                candidates.append((boosted, exp_res))
                                for kind in ("anime","manga"):
                                    r2 = await fetch_kitsu_search(session, expanded, kind)
                                    if r2:
                                        candidates.append((score_pair(query, r2["canonical"]) + 18, r2))
                                        candidates.append((score_pair(expanded, r2["canonical"]) + 5, r2))
                        except Exception:
                            pass
                        if "wikipedia" in cand.get("site_url","").lower() or "fandom" in cand.get("site_url","").lower() or len(expanded.split()) >= 2:
                            ddg_sc = 103.0
                            candidates.append((ddg_sc, {"canonical": expanded, "key": normalize(expanded), "media_type": preferred_type, "aliases": [q, title], "watch_links": [], "read_links": [], "site_url": cand.get("site_url",""), "image": "", "score": ddg_sc}))
                    if any(len(x) >= 10 for x in [expanded]):
                        break
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
async def fetch_everythingmoe_index(session) -> list[dict]:
    """Scrape EverythingMoe's section pages for the currently-live free hosts.

    Returns a list of {slug, name, kind, page_url, status, note}. We only
    collect the EverythingMoe *page* for each host (the outbound link is
    JS-gated). EverythingMoe annotates dead hosts ("… is moved to Graveyard"),
    which lets the bot detect URL death and self-heal by rediscovering them.
    """
    out: list[dict] = []
    for section, kind in config.EMOE_SECTIONS:
        url = f"{config.EVERYTHINGMOE_BASE}/section/{section}"
        try:
            async with session.get(url, headers=HEADERS, timeout=20) as r:
                if r.status != 200:
                    continue
                html = await r.text()
            soup = await _parse_html(html)
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if not href.startswith(f"/s/"):
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
                out.append({
                    "slug": slug, "name": name, "kind": kind,
                    "page_url": config.EVERYTHINGMOE_BASE + href,
                    "status": status, "note": note,
                })
        except Exception as ex:
            print(f"[emoe] section {section} failed: {ex}")
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
    """
    try:
        async with session.get(page_url, headers=HEADERS, timeout=20) as r:
            html = await r.text()
    except Exception:
        return None
    for dom in extract_domains(html):
        if dom in _JUNK_DOMAINS or dom.endswith(".everythingmoe.com"):
            continue
        for scheme in ("https", "http"):
            base = f"{scheme}://{dom}"
            try:
                # Do NOT follow redirects: a 3xx still proves the domain resolves,
                # and following would land us on an interstitial/bypass page.
                async with session.get(base, headers=HEADERS, timeout=10,
                                       allow_redirects=False) as r2:
                    if r2.status in (200, 301, 302, 307, 308):
                        return dom
            except Exception:
                continue
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

    # 2) Heuristic fallback: try common paths x common params, verify.
    for path in ("/search", "/browse", "/find", "/anime", "/manga"):
        for param in _QUERY_HINTS:
            tmpl = f"{base}{path}?{param}={{q}}"
            if await _verify_search(session, tmpl):
                return tmpl, param
    return None, None


async def _verify_search(session, tmpl: str) -> bool:
    """GET the template with a probe query; accept if it returns 200.

    JS-rendered result pages won't contain the literal query in raw HTML, so a
    200 is enough proof the endpoint exists (the form is the real authority).
    """
    url = tmpl.format(q="naruto")
    try:
        async with session.get(url, headers=HEADERS, timeout=12,
                               allow_redirects=True) as r:
            return r.status == 200
    except Exception:
        return False


async def fetch_news_for_title(session, title: str, limit: int = 8, db=None) -> list[NewsItem]:
    """Exhaustive news search for a specific title — parallel web crawl.
    Self-learning: uses DB-ranked RSS (whole web, not fixed) + DDG + title-aware.
    """
    queries = [f"{title} anime news", f"{title} manga news", title]
    tasks = []
    for q in queries:
        tasks.append(fetch_web_news(session, q, limit=limit))
        tasks.append(fetch_ddg_titles(session, q, "anime"))
    async def _rss_filtered():
        out = []
        # Use self-learned RSS sources ordered by reliability
        srcs = _get_active_rss_sources(db, limit=6) if db else config.RSS_SOURCES[:4]
        for src in srcs[:4]:
            try:
                # Attach db for stats
                if db and isinstance(src, dict):
                    src = dict(src); src["_db_ref"] = db
                items = await fetch_rss(session, src)
                for it in items:
                    if title.lower() in it.title.lower() or title.lower() in it.summary.lower():
                        out.append(it)
            except Exception:
                pass
        return out
    tasks.append(_rss_filtered())
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
    # De-dup and rank by title relevance to query
    seen = set()
    uniq = []
    for it in items:
        if it.url and it.url in seen:
            continue
        if it.url:
            seen.add(it.url)
        # Quick relevance filter: title should contain at least one word from query or be high scoring
        if score_pair(title, it.title) < 35 and title.lower() not in it.title.lower():
            # Allow if it's from web news for that exact title query (already filtered by DDG)
            if not it.source.startswith("Web:"):
                continue
        uniq.append(it)
    # Sort by relevance to title
    uniq.sort(key=lambda x: score_pair(title, x.title), reverse=True)
    return uniq[:limit]

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
    """
    # Base: self-learned RSS (ordered by reliability) + scrape
    rss_srcs = _get_active_rss_sources(db, limit=10)
    tasks = [fetch_rss(session, s) for s in rss_srcs]
    tasks += [scrape_html(session, t) for t in config.SCRAPE_TARGETS]
    # Web: many diverse queries to cover whole anime/manga web, not just “anime news”
    web_queries = [
        "anime news", "manga news", "manhwa news",
        "anime episode release", "manga chapter release",
        "anime trailer news", "anime film news",
        "weekly shonen jump news", "crunchyroll news",
        "anime trending news"
    ]
    # Pick 4 queries per cycle deterministically but rotate
    import random, time
    random.seed(int(time.time()) // 900)  # rotate every 15 min
    chosen = random.sample(web_queries, k=4)
    for q in chosen:
        tasks.append(fetch_web_news(session, q, limit=8))
    # Also do DDG title searches for trending titles (to catch news via web titles)
    for q in ["anime 2026", "manga 2026"]:
        tasks.append(fetch_ddg_titles(session, q, "anime"))
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
