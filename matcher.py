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


def normalize(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    toks = [t for t in text.split() if t and t not in _STOP]
    return " ".join(toks)


def tokenize(text: str) -> list[str]:
    return normalize(text).split()


def score_pair(query: str, candidate: str) -> float:
    """Combined fuzzy score (0-100)."""
    q, c = normalize(query), normalize(candidate)
    if not q or not c:
        return 0.0
    if q == c:
        return 100.0
    # token-set handles word reordering / extra words robustly
    s1 = fuzz.token_set_ratio(q, c)
    s2 = fuzz.partial_ratio(q, c)
    s3 = fuzz.token_sort_ratio(q, c)
    return max(s1, s2, 0.9 * s3)


def match_title(query: str, db, threshold: float = 78.0) -> dict | None:
    """Return the best matching title record, or None.

    Two stage:
      1. Exact normalized alias hit (instant, high confidence).
      2. Fuzzy token-set scoring over all learned canonical titles.

    Season/chapter markers are parsed out first ("JJK season 2" resolves to
    the base title "Jujutsu Kaisen"); the found markers ride along on the
    returned record as rec["season"] / rec["unit"] for display + news ranking.
    """
    q_clean, season, unit = parse_query(query)
    qn = normalize(q_clean)
    if qn:
        row = db.alias_lookup(qn)
        if row:
            rec = db.get_title(row[0])
            if rec:
                rec["score"] = min(100.0, 90.0 + row[1])
                rec["match_method"] = "alias"
                rec["season"] = season
                rec["unit"] = unit
                return rec

    best = None
    best_score = 0.0
    for rec in db.all_titles():
        s = score_pair(q_clean, rec["canonical"])
        # confidence from prior learning nudges borderline cases
        s = s * 0.92 + (rec["confidence"] or 0.0) * 0.08
        if s > best_score:
            best_score = s
            best = rec
    if best and best_score >= threshold:
        best["score"] = round(best_score, 1)
        best["match_method"] = "fuzzy"
        best["season"] = season
        best["unit"] = unit
        return best
    if best:  # return best-effort even below threshold, flagged
        best["score"] = round(best_score, 1)
        best["match_method"] = "fuzzy"
        best["season"] = season
        best["unit"] = unit
        return best
    return None


def match_news(query: str, items: list[dict], threshold: float = 60.0) -> list[dict]:
    """Rank news items against a query.

    Season/chapter markers in the query boost stories that mention the SAME
    season/chapter and sink stories that mention a DIFFERENT one, so
    "Jujutsu Kaisen Season 2" surfaces season-2 news, not season-1 recaps.
    """
    q_clean, season, unit = parse_query(query)
    scored = []
    for it in items:
        s = score_pair(q_clean, it["title"])
        if season is not None:
            ise = season_in(it["title"])
            if ise == season:
                s += 18
            elif ise is not None:
                s -= 25
        if unit is not None:
            iu = unit_in(it["title"])
            if iu and iu[0] == unit[0]:
                s += 22
            elif iu is not None:
                s -= 25
        if s >= threshold:
            it = dict(it)
            it["score"] = round(s, 1)
            scored.append(it)
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored
