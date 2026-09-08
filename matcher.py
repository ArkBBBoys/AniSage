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
        # Stage 1b: alias substring containment (handles "JJK" -> "jujutsu kaisen" alias)
        # If query is substring of any alias or vice versa with high token overlap,
        # treat as alias hit before fuzzy scan.
        # We scan aliases table directly for near-exact containment to catch short acronyms
        try:
            # cheap scan over alias table for containment (small table)
            import json as _json
            # db doesn't expose alias scan, so do fuzzy containment via all_titles aliases
            for rec in db.all_titles():
                for al in _aliases_for(rec) + [rec.get("canonical","")]:
                    an = normalize(al)
                    if not an:
                        continue
                    if qn == an or qn in an.split() or an in qn.split():
                        # exact token containment
                        rec2 = db.get_title(rec["key"])
                        if rec2:
                            rec2["score"] = 92.0
                            rec2["match_method"] = "alias_token"
                            rec2["season"] = season
                            rec2["unit"] = unit
                            return rec2
        except Exception:
            pass

    best = None
    best_score = 0.0
    best_raw = 0.0
    for rec in db.all_titles():
        raw = _best_score_for_title(q_clean, rec)
        # confidence nudge (0.08 weight) only helps borderline, not strong false positives
        s = raw * 0.92 + (rec.get("confidence") or 0.0) * 0.08
        if s > best_score:
            best_score = s
            best_raw = raw
            best = rec
    if best and best_score >= threshold:
        best["score"] = round(best_score, 1)
        best["raw_score"] = round(best_raw, 1)
        best["match_method"] = "fuzzy"
        best["season"] = season
        best["unit"] = unit
        return best
    # Below threshold => no local match; let caller fall back to live APIs.
    # Do NOT return low-confidence mis-match (was bug causing wrong titles like
    # "Dr. Stone" -> random isekai). Return None so live search is exhaustive.
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
