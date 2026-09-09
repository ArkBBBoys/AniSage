"""Matched-based search engine.

This is deliberately NOT keyword/substring search. It uses token-set fuzzy
matching (rapidfuzz) plus a learned alias table so "JJK", "jujutsu", or
"Jujutsu Kaisen season 2" all resolve to the *same* canonical title with a
confidence score. Queries can carry a season ("Season 2", "S2") and/or an
episode/chapter/volume marker ("chapter 150", "ch.150", "S02E05") -- the
marker is parsed out so the base title still matches, then re-applied as a
boost when ranking news items so season/chapter-specific stories win.
"""
from __future__ import annotations

import concurrent.futures
import functools
import os
import re
import unicodedata
from rapidfuzz import fuzz

_STOP = {
    "the", "a", "an", "of", "and", "to", "in", "on", "for", "with", "season",
    "episode", "ep", "chapter", "ch", "vol", "volume", "manga", "manhwa",
    "manhua", "anime", "raw", "scan", "scanlation", "dub", "sub", "english",
    "release", "read", "watch", "online", "free", "pdf",
}

# (?!\d) instead of \b after the number so "S02E05" still parses season 2
_SEASON_RE = re.compile(r"\b(?:season|s|part|stage)\s*(\d{1,3})(?!\d)", re.IGNORECASE)
_CHAPTER_RE = re.compile(
    r"\b(chapter|ch|episode|ep|vol|volume|issue|e)\.?\s*(\d{1,5})\b",
    re.IGNORECASE,
)
_UNIT_LABELS = {
    "chapter": "Chapter", "ch": "Chapter",
    "episode": "Episode", "ep": "Episode", "e": "Episode",
    "vol": "Volume", "volume": "Volume", "issue": "Issue",
}


def parse_query(query: str) -> tuple[str, int | None, tuple[int, str] | None]:
    """Split a query into (clean_title, season, (unit_number, unit_label)).

    Examples
    --------
    "Jujutsu Kaisen Season 2"      -> ("Jujutsu Kaisen", 2, None)
    "Solo Leveling chapter 150"    -> ("Solo Leveling", None, (150, "Chapter"))
    "S02E05"                       -> ("", 2, (5, "Episode"))
    "Chainsaw Man vol. 12"         -> ("Chainsaw Man", None, (12, "Volume"))
    """
    if not query:
        return "", None, None
    season = None
    ms = _SEASON_RE.search(query)
    if ms:
        season = int(ms.group(1))
        query = query[: ms.start()] + " " + query[ms.end():]
    unit = None
    mc = _CHAPTER_RE.search(query)
    if mc:
        unit = (int(mc.group(2)), _UNIT_LABELS.get(mc.group(1).lower(), "Chapter"))
        query = query[: mc.start()] + " " + query[mc.end():]
    return " ".join(query.split()), season, unit


def season_in(text: str) -> int | None:
    """Season number mentioned in a news/title string, if any."""
    m = _SEASON_RE.search(text)
    return int(m.group(1)) if m else None


def unit_in(text: str) -> tuple[int, str] | None:
    """(number, label) mentioned in a news/title string, if any."""
    m = _CHAPTER_RE.search(text)
    return (int(m.group(2)), _UNIT_LABELS.get(m.group(1).lower(), "Chapter")) if m else None


def _normalize_uncached(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    toks = [t for t in text.split() if t and t not in _STOP]
    return " ".join(toks)


@functools.lru_cache(maxsize=16384)
def normalize(text: str) -> str:
    """Cached normalize: pure function, called thousands of times per match."""
    return _normalize_uncached(text if isinstance(text, str) else "")


def tokenize(text: str) -> list[str]:
    return normalize(text).split()


def _ngram_set(s: str, n: int = 2) -> set[str]:
    """Character n-gram set for TFIDF-like cosine."""
    s = s.replace(" ", "_")
    if len(s) < n:
        return {s}
    return {s[i:i+n] for i in range(len(s)-n+1)}

def _ngram_cosine(a: str, b: str, n: int = 2) -> float:
    """0-100 n-gram cosine, robust for typos like Chronicales vs Chronicles."""
    if not a or not b:
        return 0.0
    sa, sb = _ngram_set(a, n), _ngram_set(b, n)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    denom = (len(sa) * len(sb)) ** 0.5
    if denom == 0:
        return 0.0
    return (inter / denom) * 100

def _jaro_winkler(a: str, b: str) -> float:
    """Jaro-Winkler via rapidfuzz, 0-100."""
    try:
        from rapidfuzz.distance import JaroWinkler
        return JaroWinkler.normalized_similarity(a, b) * 100
    except Exception:
        try:
            return fuzz.WRatio(a, b)
        except Exception:
            return 0.0

# Learned weights for hybrid scorer — self-improves via feedback
# Default: token_set 0.35, token_sort 0.22, partial 0.10, jaro 0.18, ngram 0.15
# These are tuned by self_improvement_loop in main.py based on feedback accuracy
_LEARNED_WEIGHTS = {"set": 0.35, "sort": 0.22, "partial": 0.10, "jaro": 0.18, "ngram": 0.15}
_WEIGHT_HISTORY: list[dict] = []

def get_weights() -> dict:
    return dict(_LEARNED_WEIGHTS)

def set_weights(new: dict):
    global _LEARNED_WEIGHTS
    for k in _LEARNED_WEIGHTS:
        if k in new:
            _LEARNED_WEIGHTS[k] = max(0.05, min(0.60, float(new[k])))
    # renormalize to 1.0
    s = sum(_LEARNED_WEIGHTS.values())
    if s:
        for k in _LEARNED_WEIGHTS:
            _LEARNED_WEIGHTS[k] /= s

def score_pair(query: str, candidate: str) -> float:
    """Hybrid scorer — conservative, typo-tolerant, acronym-aware, self-learning weighted.

    Combines 5 signals with learned weights:
      token_set (reordering), token_sort, partial, jaro-winkler (typos), ngram cosine (Chronicles)
    Handles acronyms generically and penalizes false positives.
    """
    # Acronym fast-path: e.g., BTTH (2-5 upper) vs “Battle Through the Heavens”
    q_stripped = query.strip()
    if 2 <= len(q_stripped) <= 5 and q_stripped.isupper() and q_stripped.isalpha():
        c_words = candidate.strip().split()
        initials = "".join(w[0].upper() for w in c_words if w and w[0].isalpha())
        if initials == q_stripped:
            return 96.0
        if f"({q_stripped})" in candidate.upper() or f" {q_stripped} " in f" {candidate.upper()} ":
            return 92.0
    q, c = normalize(query), normalize(candidate)
    if not q or not c:
        return 0.0
    if q == c:
        return 100.0
    q_toks = set(q.split())
    c_toks = set(c.split())
    if not q_toks or not c_toks:
        return 0.0
    overlap = len(q_toks & c_toks)
    # Core signals
    s_set = fuzz.token_set_ratio(q, c)
    s_sort = fuzz.token_sort_ratio(q, c)
    s_part = fuzz.partial_ratio(q, c)
    s_jaro = _jaro_winkler(q, c)
    s_ngram = _ngram_cosine(q, c, n=2)
    # Guard for very short queries with no overlap
    if overlap == 0 and len(q_toks) <= 2:
        base = max(s_set, s_sort * 0.95, s_jaro * 0.9, s_ngram * 0.9)
        length_penalty = min(15, abs(len(q) - len(c)) * 0.25)
        # also penalize ngram/jaro if no overlap
        return max(0.0, base - length_penalty)
    w = _LEARNED_WEIGHTS
    # Weighted hybrid — learned weights sum to 1.0
    hybrid = (w["set"] * s_set + w["sort"] * s_sort + w["partial"] * s_part + w["jaro"] * s_jaro + w["ngram"] * s_ngram)
    # Boost if jaro+ngram agree (typo case like Chronicales vs Chronicles: ngram high, jaro high, but token_set low)
    if s_ngram > 75 and s_jaro > 75 and s_set < 60:
        hybrid = max(hybrid, 0.45 * s_ngram + 0.45 * s_jaro + 0.10 * s_set)
    if len(q_toks) <= 2:
        # short query: ignore partial
        hybrid = (w["set"] * s_set + w["sort"] * s_sort + w["jaro"] * s_jaro + w["ngram"] * s_ngram) / (1 - w["partial"])
    if overlap == 0:
        hybrid = min(hybrid, 58.0)
    # Final: max of hybrid and raw set (set is most reliable for reordering)
    return max(s_set * 0.88, hybrid)


def _aliases_for(rec: dict) -> list[str]:
    """Coerce rec['aliases'] (str JSON or list) to list of strings."""
    import json
    a = rec.get("aliases")
    if not a:
        return []
    if isinstance(a, list):
        return [str(x) for x in a if x]
    if isinstance(a, str):
        try:
            j = json.loads(a)
            if isinstance(j, list):
                return [str(x) for x in j if x]
        except Exception:
            pass
        # fallback: comma separated
        return [s.strip() for s in a.split(",") if s.strip()]
    return []

def _best_score_for_title(query_clean: str, rec: dict) -> float:
    """Score query against canonical + every alias, return max."""
    candidates = [rec.get("canonical", "")] + _aliases_for(rec)
    best = 0.0
    for cand in candidates:
        s = score_pair(query_clean, cand)
        if s > best:
            best = s
    return best

def _matcher_workers(n: int | None = None) -> int:
    """Thread count for parallel scoring (env MATCHER_WORKERS, default 8)."""
    try:
        default = int(os.getenv("MATCHER_WORKERS", "8"))
    except ValueError:
        default = 8
    if n is not None:
        return max(1, min(n, default))
    return max(1, default)


def score_many(query: str, candidates: list[str],
               max_workers: int | None = None) -> list[float]:
    """Threaded batch scoring — same results as serial score_pair, ~Nx faster.

    rapidfuzz releases the GIL, so threads give real speedup. Small batches
    (<16) stay serial to avoid thread overhead.
    """
    cands = list(candidates or [])
    if not cands:
        return []
    if len(cands) < 16:
        return [score_pair(query, c) for c in cands]
    workers = _matcher_workers(max_workers)
    # Chunk to keep tasks coarse-grained (fewer futures, less overhead).
    chunk = max(8, -(-len(cands) // (workers * 4)))
    chunks = [cands[i:i + chunk] for i in range(0, len(cands), chunk)]

    def _score_chunk(ch):
        return [score_pair(query, c) for c in ch]

    out: list[float] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(_score_chunk, chunks):
            out.extend(res)
    return out


def _score_titles_chunk(args) -> list[tuple[float, float, int]]:
    """Worker: score a chunk of (canonical+aliases) title rows.

    Rows are flat (idx, canonical, aliases, confidence) tuples as built by
    _fuzzy_best_threaded. Returns [(adjusted, raw, index)] per title.
    """
    q_clean, rows = args
    res = []
    for idx, canonical, aliases, confidence in rows:
        best = 0.0
        if canonical:
            best = score_pair(q_clean, canonical)
        for al in aliases:
            if not al:
                continue
            s = score_pair(q_clean, al)
            if s > best:
                best = s
                if best >= 99.5:  # early exit on near-perfect
                    break
        adj = best * 0.92 + (confidence or 0.0) * 0.08
        res.append((adj, best, idx))
    return res


def _fuzzy_best_threaded(q_clean: str, titles: list[dict],
                         max_workers: int | None = None):
    """Parallel fuzzy scan over title rows. Returns (best_rec, best_score, best_raw)."""
    if not titles:
        return None, 0.0, 0.0
    workers = _matcher_workers(max_workers)
    if len(titles) < 40 or workers <= 1:
        best = None
        best_score = 0.0
        best_raw = 0.0
        for rec in titles:
            raw = _best_score_for_title(q_clean, rec)
            s = raw * 0.92 + (rec.get("confidence") or 0.0) * 0.08
            if s > best_score:
                best_score, best_raw, best = s, raw, rec
        return best, best_score, best_raw
    # Pre-extract light tuples to keep workers pickling cheap (threads share
    # memory, but smaller tuples still help cache locality).
    light = []
    for i, rec in enumerate(titles):
        light.append((i, rec.get("canonical", ""), _aliases_for(rec),
                      rec.get("confidence") or 0.0))
    chunk = max(16, -(-len(light) // (workers * 4)))
    chunks = [light[i:i + chunk] for i in range(0, len(light), chunk)]
    payloads = [(q_clean, ch) for ch in chunks]
    best_score = 0.0
    best_raw = 0.0
    best_idx = -1
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(_score_titles_chunk, payloads):
            for adj, raw, idx in res:
                if adj > best_score:
                    best_score, best_raw, best_idx = adj, raw, idx
    if best_idx < 0:
        return None, 0.0, 0.0
    return titles[best_idx], best_score, best_raw


def _match_title_from_rows(q_clean: str, qn: str, titles: list[dict],
                           db, threshold: float, season, unit) -> dict | None:
    """Stage 1b + stage 2 over an already-fetched title list (no extra DB I/O)."""
    # Stage 1b: exact token containment over aliases (cheap, serial — early exit).
    try:
        for rec in titles:
            for al in _aliases_for(rec) + [rec.get("canonical", "")]:
                an = normalize(al)
                if not an:
                    continue
                if qn == an or qn in an.split() or an in qn.split():
                    rec2 = db.get_title(rec["key"]) if hasattr(db, "get_title") else rec
                    if rec2:
                        rec2["score"] = 92.0
                        rec2["match_method"] = "alias_token"
                        rec2["season"] = season
                        rec2["unit"] = unit
                        return rec2
    except Exception:
        pass
    best, best_score, best_raw = _fuzzy_best_threaded(q_clean, titles)
    if best is not None and best_score >= threshold:
        # Light rows (all_titles_for_match) lack media_type/image/links — hydrate.
        full = None
        if hasattr(db, "get_title") and best.get("media_type") is None:
            try:
                full = db.get_title(best["key"])
            except Exception:
                full = None
        rec_out = full if full else best
        rec_out["score"] = round(best_score, 1)
        rec_out["raw_score"] = round(best_raw, 1)
        rec_out["match_method"] = "fuzzy"
        rec_out["season"] = season
        rec_out["unit"] = unit
        return rec_out
    return None


def _fetch_match_rows(db) -> list[dict]:
    """Prefer the lightweight match projection; fallback to full rows."""
    try:
        if hasattr(db, "all_titles_for_match"):
            rows = db.all_titles_for_match()
            if rows:
                return rows
    except Exception:
        pass
    return db.all_titles()


def match_title(query: str, db, threshold: float = 78.0) -> dict | None:
    """Return the best matching title record, or None.

    Two stage:
      1. Exact normalized alias hit (instant, high confidence).
          Also handles partial alias containment (query inside alias).
      2. Fuzzy token-set scoring over canonical + aliases, with confidence
          nudge for learned titles.

    Returns None when below threshold — caller must trigger live search rather
    than presenting a low-confidence mis-match. Season/chapter markers are
    parsed out first and re-attached to the returned record.
    """
    q_clean, season, unit = parse_query(query)
    qn = normalize(q_clean)
    if not q_clean or not qn:
        return None
    # Stage 1a: exact normalized alias hit (instant)
    if qn:
        row = db.alias_lookup(qn)
        if row:
            rec = db.get_title(row[0])
            if rec:
                rec["score"] = min(100.0, 90.0 + row[1] * 2)
                rec["match_method"] = "alias"
                rec["season"] = season
                rec["unit"] = unit
                return rec
    # Stage 1b + 2 over a single fetched snapshot (one DB round-trip, then
    # threaded scoring -- previously this scanned all_titles() twice).
    # Uses the lightweight projection (key/canonical/aliases/confidence) for
    # ~3-5x less SQLite I/O, then hydrates the winner via get_title.
    try:
        titles = _fetch_match_rows(db)
    except Exception:
        return None
    return _match_title_from_rows(q_clean, qn, titles, db, threshold, season, unit)
    # Below threshold => no local match; let caller fall back to live APIs.
    # Do NOT return low-confidence mis-match (was bug causing wrong titles like
    # "Dr. Stone" -> random isekai). Return None so live search is exhaustive.
    return None


def _news_score_one(args) -> tuple[float, int]:
    """Worker for threaded match_news: returns (adjusted_score, index)."""
    q_clean, season, unit, title = args
    s = score_pair(q_clean, title or "")
    if season is not None:
        ise = season_in(title or "")
        if ise == season:
            s += 18
        elif ise is not None:
            s -= 25
    if unit is not None:
        iu = unit_in(title or "")
        if iu and iu[0] == unit[0]:
            s += 22
        elif iu is not None:
            s -= 25
    return s


def match_news(query: str, items: list[dict], threshold: float = 60.0) -> list[dict]:
    """Rank news items against a query.

    Season/chapter markers in the query boost stories that mention the SAME
    season/chapter and sink stories that mention a DIFFERENT one, so
    "Jujutsu Kaisen Season 2" surfaces season-2 news, not season-1 recaps.

    Threaded: titles are scored in a worker pool when the list is large
    (>=30 items), otherwise serial to avoid thread overhead.
    """
    q_clean, season, unit = parse_query(query)
    if not items:
        return []
    titles = [(it.get("title", "") if isinstance(it, dict) else "") for it in items]
    if len(items) >= 30:
        workers = _matcher_workers()
        payloads = [(q_clean, season, unit, ti) for ti in titles]
        chunk = max(8, -(-len(payloads) // (workers * 4)))
        chunks = [payloads[i:i + chunk] for i in range(0, len(payloads), chunk)]

        def _chunk_scores(ch):
            return [_news_score_one(a) for a in ch]

        import concurrent.futures as _cf
        scores: list[float] = []
        with _cf.ThreadPoolExecutor(max_workers=workers) as pool:
            for res in pool.map(_chunk_scores, chunks):
                scores.extend(res)
    else:
        scores = [score_pair(q_clean, ti) for ti in titles]
        if season is not None or unit is not None:
            adj = []
            for s, it in zip(scores, items):
                title = it.get("title", "") if isinstance(it, dict) else ""
                if season is not None:
                    ise = season_in(title or "")
                    if ise == season:
                        s += 18
                    elif ise is not None:
                        s -= 25
                if unit is not None:
                    iu = unit_in(title or "")
                    if iu and iu[0] == unit[0]:
                        s += 22
                    elif iu is not None:
                        s -= 25
                adj.append(s)
            scores = adj
    scored = []
    for it, s in zip(items, scores):
        if s >= threshold:
            it = dict(it)
            it["score"] = round(s, 1)
            scored.append(it)
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


# ------------------------------------------------- async (non-blocking)
# These keep the Discord event loop responsive: DB I/O goes to the DB pool
# and CPU scoring goes to the CPU pool. Sync match_title/match_news above
# stay for background threads and tests.

async def amatch_title(query: str, db, threshold: float = 78.0) -> dict | None:
    """Async match_title: alias lookup + one bulk fetch off-loop, threaded score."""
    import asyncio as _aio
    q_clean, season, unit = parse_query(query)
    qn = normalize(q_clean)
    if not q_clean or not qn:
        return None
    loop = _aio.get_running_loop()
    try:
        import concurrency as _conc
        row = await loop.run_in_executor(_conc.DB_EXECUTOR, db.alias_lookup, qn)
    except Exception:
        row = None
    if row:
        try:
            import concurrency as _conc2
            rec = await loop.run_in_executor(_conc2.DB_EXECUTOR, db.get_title, row[0])
        except Exception:
            rec = None
        if rec:
            rec["score"] = min(100.0, 90.0 + row[1] * 2)
            rec["match_method"] = "alias"
            rec["season"] = season
            rec["unit"] = unit
            return rec

    def _fetch_rows_sync():
        return _fetch_match_rows(db)

    try:
        import concurrency as _conc3
        titles = await loop.run_in_executor(_conc3.DB_EXECUTOR, _fetch_rows_sync)
    except Exception:
        return None
    if not titles:
        return None
    # Stage 1b containment is cheap; run it inline before threading.
    try:
        for rec in titles:
            for al in _aliases_for(rec) + [rec.get("canonical", "")]:
                an = normalize(al)
                if not an:
                    continue
                if qn == an or qn in an.split() or an in qn.split():
                    import concurrency as _conc4
                    rec2 = await loop.run_in_executor(_conc4.DB_EXECUTOR, db.get_title, rec["key"])
                    if rec2:
                        rec2["score"] = 92.0
                        rec2["match_method"] = "alias_token"
                        rec2["season"] = season
                        rec2["unit"] = unit
                        return rec2
                    break
    except Exception:
        pass
    try:
        import concurrency as _conc5
        best, best_score, best_raw = await loop.run_in_executor(
            _conc5.CPU_EXECUTOR, _fuzzy_best_threaded_sync, q_clean, titles)
    except Exception:
        return None
    if best is not None and best_score >= threshold:
        # Hydrate light rows to full records for embeds/links.
        if best.get("media_type") is None:
            try:
                import concurrency as _conc6
                full = await loop.run_in_executor(_conc6.DB_EXECUTOR, db.get_title, best["key"])
                if full:
                    best = full
            except Exception:
                pass
        best["score"] = round(best_score, 1)
        best["raw_score"] = round(best_raw, 1)
        best["match_method"] = "fuzzy"
        best["season"] = season
        best["unit"] = unit
        return best
    return None


def _fuzzy_best_threaded_sync(q_clean: str, titles: list[dict]):
    """Picklable sync entry point for CPU pool (calls threaded scorer)."""
    return _fuzzy_best_threaded(q_clean, titles)


async def amatch_news(query: str, items: list[dict], threshold: float = 60.0) -> list[dict]:
    """Async match_news: scoring off the event loop in the CPU pool."""
    if not items:
        return []
    import asyncio as _aio2
    loop = _aio2.get_running_loop()
    try:
        import concurrency as _conc6
        return await loop.run_in_executor(
            _conc6.CPU_EXECUTOR, match_news, query, list(items), threshold)
    except Exception:
        return match_news(query, items, threshold)


async def ascore_many(query: str, candidates: list[str]) -> list[float]:
    """Async batch scoring off the event loop."""
    if not candidates:
        return []
    import asyncio as _aio3
    loop = _aio3.get_running_loop()
    try:
        import concurrency as _conc7
        return await loop.run_in_executor(
            _conc7.CPU_EXECUTOR, score_many, query, list(candidates))
    except Exception:
        return score_many(query, candidates)
