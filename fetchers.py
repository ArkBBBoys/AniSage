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
from matcher import normalize

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


# --------------------------------------------------------------------- RSS
async def fetch_rss(session, source: dict) -> list[NewsItem]:
    out: list[NewsItem] = []
    try:
        async with session.get(source["url"], headers=HEADERS, timeout=20) as r:
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
    except Exception as ex:  # network/parse errors are logged, not fatal
        print(f"[rss] {source['name']} failed: {ex}")
    return out


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
        async with session.get(url, params={"q": query, "limit": 1},
                               headers=HEADERS, timeout=20) as r:
            data = await r.json()
        if not data.get("data"):
            return None
        m = data["data"][0]
        canonical = m.get("title") or m.get("title_english") or ""
        links = [m.get("url")] if m.get("url") else []
        return {
            "canonical": canonical, "key": normalize(canonical),
            "media_type": kind, "mal_id": str(m.get("mal_id", "")),
            "watch_links": links, "read_links": links,
            "site_url": m.get("url", ""),
        }
    except Exception as ex:
        print(f"[jikan] search failed: {ex}")
        return None


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


# ------------------------------------------------------------------ runner
async def run_news_cycle(session) -> tuple[int, list[NewsItem]]:
    """Fetch all RSS + scrape targets. Returns (count, items)."""
    tasks = [fetch_rss(session, s) for s in config.RSS_SOURCES]
    tasks += [scrape_html(session, t) for t in config.SCRAPE_TARGETS]
    results = await asyncio.gather(*tasks)
    items: list[NewsItem] = []
    for group in results:
        items.extend(group)
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
