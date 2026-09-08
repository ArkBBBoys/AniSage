"""Self-learning knowledge store (SQLite via SQLAlchemy ORM).

The bot never just "forgets" what it scraped. Every item, every title, every
alias and every piece of user feedback is persisted so the matching engine
gets *better* the more it runs -- that is the "self-learning" loop.

Thread-safety: the connection is shared between the event-loop thread and the
worker threads used for bulk writes, so the engine opens with
check_same_thread=False and EVERY access is serialized with a single (reentrant)
lock. WAL journal mode means outside readers (e.g. a DB viewer) can always read
while the bot writes -- no more "database is locked".
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import (
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    event,
    func,
    select,
    update,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.sql import text

from matcher import normalize

DB_PATH = Path(__file__).parent / "anisage.db"

# One (reentrant) lock for the single process-wide KnowledgeDB. RLock is
# REQUIRED because learn_title/record_feedback call _set_alias, which is itself
# @_locked -- a plain Lock would deadlock on the nested acquire.
_DB_LOCK = threading.RLock()


def _locked(func):
    """Serialize every DB method call so the shared connection is never hit
    concurrently from two threads."""
    def wrapper(self, *args, **kwargs):
        with _DB_LOCK:
            return func(self, *args, **kwargs)
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


@dataclass
class NewsItem:
    source: str
    kind: str
    title: str
    url: str
    summary: str = ""
    image: str = ""
    published: str = ""
    media_type: str = "unknown"
    external_id: str = ""
    anilist_id: str = ""
    mal_id: str = ""


@dataclass
class TitleRecord:
    key: str
    canonical: str
    media_type: str = "unknown"
    external_id: str = ""
    anilist_id: str = ""
    mal_id: str = ""
    image: str = ""
    aliases: list[str] = None  # type: ignore
    watch_links: list[str] = None  # type: ignore
    read_links: list[str] = None  # type: ignore
    times_seen: int = 0
    confidence: float = 0.0

    def __post_init__(self):
        self.aliases = self.aliases or []
        self.watch_links = self.watch_links or []
        self.read_links = self.read_links or []


# ------------------------------------------------------------------- ORM
class Base(DeclarativeBase):
    pass


def _obj_dict(obj) -> dict:
    """ORM instance -> plain dict (drops SQLAlchemy's instance-state attr)."""
    d = obj.__dict__.copy()
    d.pop("_sa_instance_state", None)
    return d


class Item(Base):
    __tablename__ = "items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String, default="")
    kind: Mapped[str] = mapped_column(String, default="")
    title: Mapped[str] = mapped_column(String, default="")
    url: Mapped[str] = mapped_column(String, unique=True, default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    image: Mapped[str] = mapped_column(Text, default="")
    published: Mapped[str] = mapped_column(String, default="")
    fetched_at: Mapped[float] = mapped_column(Float, default=0.0)
    media_type: Mapped[str] = mapped_column(String, default="unknown")
    external_id: Mapped[str] = mapped_column(String, default="")
    anilist_id: Mapped[str] = mapped_column(String, default="")
    mal_id: Mapped[str] = mapped_column(String, default="")


class Title(Base):
    __tablename__ = "titles"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    canonical: Mapped[str] = mapped_column(String, default="")
    media_type: Mapped[str] = mapped_column(String, default="unknown")
    external_id: Mapped[str] = mapped_column(String, default="")
    anilist_id: Mapped[str] = mapped_column(String, default="")
    mal_id: Mapped[str] = mapped_column(String, default="")
    image: Mapped[str] = mapped_column(Text, default="")
    aliases: Mapped[str] = mapped_column(Text, default="[]")
    watch_links: Mapped[str] = mapped_column(Text, default="[]")
    read_links: Mapped[str] = mapped_column(Text, default="[]")
    first_seen: Mapped[float] = mapped_column(Float, default=0.0)
    last_seen: Mapped[float] = mapped_column(Float, default=0.0)
    times_seen: Mapped[int] = mapped_column(Integer, default=1)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    # Self-learning: Bayesian feedback tracking
    feedback_correct: Mapped[int] = mapped_column(Integer, default=0)
    feedback_wrong: Mapped[int] = mapped_column(Integer, default=0)
    last_feedback_ts: Mapped[float] = mapped_column(Float, default=0.0)
    # For self-improvement: embedding-like ngram signature for quick re-rank
    ngram_sig: Mapped[str] = mapped_column(Text, default="")


class Alias(Base):
    __tablename__ = "aliases"

    alias: Mapped[str] = mapped_column(String, primary_key=True)
    title_key: Mapped[str] = mapped_column(String, default="")
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    # Self-learning: per-alias performance
    hits_correct: Mapped[int] = mapped_column(Integer, default=0)
    hits_wrong: Mapped[int] = mapped_column(Integer, default=0)
    last_used: Mapped[float] = mapped_column(Float, default=0.0)
    # Decay: alias ages out if not used
    created_at: Mapped[float] = mapped_column(Float, default=0.0)


class LearningMetrics(Base):
    """Self-improvement: tracks matcher performance over time for auto-tuning."""
    __tablename__ = "learning_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    window_start: Mapped[float] = mapped_column(Float, default=0.0)
    window_end: Mapped[float] = mapped_column(Float, default=0.0)
    total_queries: Mapped[int] = mapped_column(Integer, default=0)
    correct_hits: Mapped[int] = mapped_column(Integer, default=0)
    wrong_hits: Mapped[int] = mapped_column(Integer, default=0)
    avg_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    threshold_used: Mapped[float] = mapped_column(Float, default=78.0)
    # Learned weights snapshot
    weights_json: Mapped[str] = mapped_column(Text, default="{}")

class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(Text, default="")
    matched_key: Mapped[str] = mapped_column(String, default="")
    correct: Mapped[int] = mapped_column(Integer, default=0)
    ts: Mapped[float] = mapped_column(Float, default=0.0)
    # For self-learning: what matcher method was used and what score
    match_method: Mapped[str] = mapped_column(String, default="")
    score: Mapped[float] = mapped_column(Float, default=0.0)


class ScrapeLog(Base):
    __tablename__ = "scrape_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String, default="")
    count: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    ok: Mapped[int] = mapped_column(Integer, default=0)
    ts: Mapped[float] = mapped_column(Float, default=0.0)


class Follow(Base):
    __tablename__ = "follows"
    # Per-guild + per-user isolation: same user can follow same title in different servers
    # Old schema was PK(user_id, title_key) with no guild. New adds guild_id.
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String, index=True)
    title_key: Mapped[str] = mapped_column(String, index=True)
    guild_id: Mapped[str] = mapped_column(String, default="0", index=True)  # "0" = DM / global, otherwise guild snowflake
    created_at: Mapped[float] = mapped_column(Float, default=0.0)

    __table_args__ = (
        UniqueConstraint("user_id", "title_key", "guild_id", name="uq_follow_user_title_guild"),
    )


class Broadcast(Base):
    """Per-user auto-news: who wants it, which guild channel to post in, and
    when the last digest went out (so each cycle only sends fresh news).
    Now per-guild isolated: user can have separate auto-news in each server + DM.
    """

    __tablename__ = "broadcasts"
    # logical key: user_id + guild_id (guild_id "0" = DM personal)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String, index=True)
    guild_id: Mapped[str] = mapped_column(String, default="0", index=True)
    channel_id: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[float] = mapped_column(Float, default=0.0)
    last_sent: Mapped[float] = mapped_column(Float, default=0.0)

    __table_args__ = (
        UniqueConstraint("user_id", "guild_id", name="uq_broadcast_user_guild"),
    )


class Resource(Base):
    __tablename__ = "resources"

    slug: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, default="")
    kind: Mapped[str] = mapped_column(String, default="")
    page_url: Mapped[str] = mapped_column(Text, default="")
    domain: Mapped[str] = mapped_column(String, default="")
    search_url: Mapped[str] = mapped_column(Text, default="")
    search_param: Mapped[str] = mapped_column(String, default="")
    status: Mapped[str] = mapped_column(String, default="alive")
    note: Mapped[str] = mapped_column(Text, default="")
    last_seen: Mapped[float] = mapped_column(Float, default=0.0)
    last_checked: Mapped[float] = mapped_column(Float, default=0.0)
    dead_count: Mapped[int] = mapped_column(Integer, default=0)

class NewsSource(Base):
    """Self-learning news source — discovered and curated by the bot itself."""
    __tablename__ = "news_sources"

    url: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, default="")
    kind: Mapped[str] = mapped_column(String, default="news")  # news, community, manga, etc.
    status: Mapped[str] = mapped_column(String, default="active")  # active, dead, paused
    reliability: Mapped[float] = mapped_column(Float, default=0.5)  # 0.0-1.0 self-learned
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    fail_count: Mapped[int] = mapped_column(Integer, default=0)
    avg_items: Mapped[float] = mapped_column(Float, default=0.0)
    avg_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    last_checked: Mapped[float] = mapped_column(Float, default=0.0)
    last_success: Mapped[float] = mapped_column(Float, default=0.0)
    discovered_via: Mapped[str] = mapped_column(String, default="config")  # config, web_discovery, user
    score: Mapped[float] = mapped_column(Float, default=0.5)  # composite for ranking
    consecutive_fails: Mapped[int] = mapped_column(Integer, default=0)

class TitleSearchSource(Base):
    """Self-learning title search source reliability (AniList, Kitsu, Jikan, DDG, MAL)."""
    __tablename__ = "title_search_sources"

    name: Mapped[str] = mapped_column(String, primary_key=True)  # anilist, kitsu, jikan, ddg, mal
    reliability: Mapped[float] = mapped_column(Float, default=0.5)
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    fail_count: Mapped[int] = mapped_column(Integer, default=0)
    avg_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    last_checked: Mapped[float] = mapped_column(Float, default=0.0)
    last_success: Mapped[float] = mapped_column(Float, default=0.0)
    consecutive_fails: Mapped[int] = mapped_column(Integer, default=0)
    weight: Mapped[float] = mapped_column(Float, default=1.0)  # for candidate scoring


def _make_engine(path: Path):
    # check_same_thread=False REQUIRED because bulk writes run in a worker
    # thread; the RLock above guarantees serialized access. timeout=30 = wait
    # out brief lock holders (e.g. a DB viewer) instead of erroring instantly.
    engine = create_engine(
        f"sqlite:///{path.as_posix()}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )

    # WAL mode: readers (VS Code SQLite viewer, backup tools) never block on
    # the bot's writes, and vice versa. synchronous=NORMAL = durable enough for
    # a bot while keeping writes fast.
    @event.listens_for(engine, "connect")
    def _set_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

    return engine


class KnowledgeDB:
    def __init__(self, path: Path | str = DB_PATH):
        self.path = Path(path)
        # Ensure the directory exists; sqlite creates the file on connect, but
        # if the parent dir is missing the open would fail. Make it bulletproof.
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        self.engine = _make_engine(self.path)
        Base.metadata.create_all(self.engine)
        self._session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self._migrate_resources_columns()
        self._migrate_titles_columns()
        self._migrate_guild_columns()
        self._migrate_learning_columns()

    def _migrate_titles_columns(self):
        """Best-effort upgrade for pre-existing DBs missing the image column."""
        try:
            with self._session() as s:
                s.execute(text("ALTER TABLE titles ADD COLUMN image TEXT DEFAULT ''"))
                s.commit()
        except Exception:
            pass  # column already exists (or table absent)

    def _migrate_resources_columns(self):
        """Best-effort upgrade for pre-existing DBs missing newer columns."""
        for col, typ in [
            ("status", "TEXT DEFAULT 'alive'"),
            ("note", "TEXT DEFAULT ''"),
            ("last_checked", "REAL"),
            ("dead_count", "INTEGER DEFAULT 0"),
            ("domain", "TEXT DEFAULT ''"),
            ("search_url", "TEXT DEFAULT ''"),
            ("search_param", "TEXT DEFAULT ''"),
        ]:
            try:
                with self._session() as s:
                    s.execute(text(f"ALTER TABLE resources ADD COLUMN {col} {typ}"))
                    s.commit()
            except Exception:
                pass  # column already exists (or table absent)

    def _migrate_guild_columns(self):
        """Migrate follows/broadcasts to per-guild isolation.
        Handles old DBs where tables were PK(user_id,title_key) and PK(user_id).
        Adds guild_id + created_at where missing, and rebuilds PKs if needed via
        table recreation (safe, preserves data).
        """
        def _cols(table: str) -> set[str]:
            try:
                with self._session() as s:
                    rows = s.execute(text(f"PRAGMA table_info({table})")).fetchall()
                    return {r[1] for r in rows}
            except Exception:
                return set()
        def _sql(table: str) -> str:
            try:
                with self._session() as s:
                    row = s.execute(text(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}'")).fetchone()
                    return row[0] if row and row[0] else ""
            except Exception:
                return ""
        # -- follows: add guild_id + created_at if missing
        f_cols = _cols("follows")
        if f_cols:
            try:
                with self._session() as s:
                    if "guild_id" not in f_cols:
                        s.execute(text("ALTER TABLE follows ADD COLUMN guild_id TEXT DEFAULT '0'"))
                        s.commit()
                        print("[migrate] follows.guild_id added")
                    if "created_at" not in f_cols:
                        s.execute(text("ALTER TABLE follows ADD COLUMN created_at REAL DEFAULT 0"))
                        s.commit()
            except Exception:
                pass
            # need to handle old PK (user_id,title_key) vs new surrogate id + unique
            sql = _sql("follows")
            # if old schema still has no 'id' column, it is pre-migration; rebuild
            if "id" not in f_cols:
                try:
                    with self._session() as s:
                        # detect if we can just add id via recreate (sqlite ALTER can't add PK)
                        # create new table, copy, replace
                        s.execute(text("""
                            CREATE TABLE IF NOT EXISTS follows_new (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                user_id TEXT NOT NULL,
                                title_key TEXT NOT NULL,
                                guild_id TEXT DEFAULT '0',
                                created_at REAL DEFAULT 0
                            )
                        """))
                        # copy existing rows, ignore duplicate due to new unique
                        s.execute(text("""
                            INSERT OR IGNORE INTO follows_new (user_id, title_key, guild_id, created_at)
                            SELECT user_id, title_key, COALESCE(guild_id, '0'), COALESCE(created_at, 0)
                            FROM follows
                        """))
                        s.execute(text("DROP TABLE follows"))
                        s.execute(text("ALTER TABLE follows_new RENAME TO follows"))
                        s.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_follow_user_title_guild ON follows(user_id, title_key, guild_id)"))
                        s.execute(text("CREATE INDEX IF NOT EXISTS ix_follows_user_id ON follows(user_id)"))
                        s.execute(text("CREATE INDEX IF NOT EXISTS ix_follows_title_key ON follows(title_key)"))
                        s.execute(text("CREATE INDEX IF NOT EXISTS ix_follows_guild_id ON follows(guild_id)"))
                        s.commit()
                        print("[migrate] follows table rebuilt for per-guild isolation")
                except Exception as ex:
                    try:
                        s.rollback()
                    except: pass
                    print(f"[migrate] follows rebuild failed: {ex}")
        # -- broadcasts: add guild_id if missing, and ensure id surrogate
        b_cols = _cols("broadcasts")
        if b_cols:
            try:
                with self._session() as s:
                    if "guild_id" not in b_cols:
                        s.execute(text("ALTER TABLE broadcasts ADD COLUMN guild_id TEXT DEFAULT '0'"))
                        s.commit()
                        print("[migrate] broadcasts.guild_id added")
            except Exception:
                pass
            sql = _sql("broadcasts")
            if "id" not in b_cols:
                try:
                    with self._session() as s:
                        s.execute(text("""
                            CREATE TABLE IF NOT EXISTS broadcasts_new (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                user_id TEXT NOT NULL,
                                guild_id TEXT DEFAULT '0',
                                channel_id INTEGER DEFAULT 0,
                                created_at REAL DEFAULT 0,
                                last_sent REAL DEFAULT 0,
                                UNIQUE(user_id, guild_id)
                            )
                        """))
                        # copy old rows: old had PK user_id only; assign guild_id '0' for DM/global
                        # if old had channel_id etc, preserve
                        # need to handle old schema which had no guild_id; use 0
                        s.execute(text("""
                            INSERT OR IGNORE INTO broadcasts_new (user_id, guild_id, channel_id, created_at, last_sent)
                            SELECT user_id, '0', channel_id, created_at, last_sent FROM broadcasts
                        """))
                        s.execute(text("DROP TABLE broadcasts"))
                        s.execute(text("ALTER TABLE broadcasts_new RENAME TO broadcasts"))
                        s.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_broadcast_user_guild ON broadcasts(user_id, guild_id)"))
                        s.execute(text("CREATE INDEX IF NOT EXISTS ix_broadcasts_user_id ON broadcasts(user_id)"))
                        s.execute(text("CREATE INDEX IF NOT EXISTS ix_broadcasts_guild_id ON broadcasts(guild_id)"))
                        s.commit()
                        print("[migrate] broadcasts table rebuilt for per-guild isolation")
                except Exception as ex:
                    try: s.rollback()
                    except: pass
                    print(f"[migrate] broadcasts rebuild failed: {ex}")

    def _migrate_learning_columns(self):
        """Self-learning upgrade: add Bayesian feedback columns, decay, ngram sig."""
        # Titles: feedback_correct, feedback_wrong, last_feedback_ts, ngram_sig
        for col, typ in [
            ("feedback_correct", "INTEGER DEFAULT 0"),
            ("feedback_wrong", "INTEGER DEFAULT 0"),
            ("last_feedback_ts", "REAL DEFAULT 0"),
            ("ngram_sig", "TEXT DEFAULT ''"),
        ]:
            try:
                with self._session() as s:
                    s.execute(text(f"ALTER TABLE titles ADD COLUMN {col} {typ}"))
                    s.commit()
            except Exception:
                pass
        # Aliases: hits_correct, hits_wrong, last_used, created_at
        for col, typ in [
            ("hits_correct", "INTEGER DEFAULT 0"),
            ("hits_wrong", "INTEGER DEFAULT 0"),
            ("last_used", "REAL DEFAULT 0"),
            ("created_at", "REAL DEFAULT 0"),
        ]:
            try:
                with self._session() as s:
                    s.execute(text(f"ALTER TABLE aliases ADD COLUMN {col} {typ}"))
                    s.commit()
            except Exception:
                pass
        # Feedback: match_method, score
        for col, typ in [
            ("match_method", "TEXT DEFAULT ''"),
            ("score", "REAL DEFAULT 0"),
        ]:
            try:
                with self._session() as s:
                    s.execute(text(f"ALTER TABLE feedback ADD COLUMN {col} {typ}"))
                    s.commit()
            except Exception:
                pass
        # Ensure learning_metrics table exists (create_all already did)
        try:
            with self._session() as s:
                s.execute(text("SELECT 1 FROM learning_metrics LIMIT 1"))
        except Exception:
            pass

    # ------------------------------------------------------------------ items
    @_locked
    def add_item(self, it: NewsItem) -> bool:
        try:
            with self._session() as s:
                stmt = sqlite_insert(Item).values(
                    source=it.source, kind=it.kind, title=it.title, url=it.url,
                    summary=it.summary, image=it.image, published=it.published,
                    fetched_at=time.time(), media_type=it.media_type,
                    external_id=it.external_id, anilist_id=it.anilist_id,
                    mal_id=it.mal_id,
                ).on_conflict_do_nothing(index_elements=["url"])
                res = s.execute(stmt)
                s.commit()
                return res.rowcount > 0
        except IntegrityError:
            return False

    @_locked
    def bulk_add_items(self, items: list[NewsItem]) -> int:
        """Insert many items in ONE transaction (one commit) — avoids blocking."""
        rows = [
            {
                "source": it.source, "kind": it.kind, "title": it.title,
                "url": it.url, "summary": it.summary, "image": it.image,
                "published": it.published, "fetched_at": time.time(),
                "media_type": it.media_type, "external_id": it.external_id,
                "anilist_id": it.anilist_id, "mal_id": it.mal_id,
            }
            for it in items
        ]
        with self._session() as s:
            stmt = sqlite_insert(Item).on_conflict_do_nothing(index_elements=["url"])
            # executemany results are IteratorResult (no rowcount); count real
            # inserts via the raw connection's total_changes delta instead.
            raw = s.connection().connection.driver_connection
            before = raw.total_changes
            s.execute(stmt, rows)
            s.commit()
            return raw.total_changes - before

    @_locked
    def recent_items(self, limit: int = 25, media_type: str = "") -> list[dict]:
        with self._session() as s:
            stmt = select(Item).order_by(Item.fetched_at.desc()).limit(limit)
            if media_type:
                stmt = stmt.where(Item.media_type == media_type)
            return [_obj_dict(o) for o in s.execute(stmt).scalars().all()]

    @_locked
    def items_for_title(self, key: str, limit: int = 20) -> list[dict]:
        with self._session() as s:
            rows = s.execute(
                select(Item).where(Item.title.like(f"%{key}%"))
                .order_by(Item.fetched_at.desc()).limit(limit)
            ).scalars().all()
            return [_obj_dict(o) for o in rows]

    # ----------------------------------------------------------------- titles
    def _bayesian_confidence(self, correct: int, wrong: int, times_seen: int, base: float = 5.0, last_seen: float = 0, alias_weights: list[float] | None = None, source_quality: float = 0.5) -> float:
        """Upgraded Bayesian confidence — self-learning, recency-aware, alias-aware.

        Prior is adaptive: alpha/beta learned from global accuracy (self-improving).
        Components:
          - Posterior from Beta-Binomial (correct/wrong) with learned prior
          - Frequency: log1p(times_seen) with diminishing returns
          - Recency: boost if seen recently, decay if stale
          - Alias quality: avg alias weight
          - Source quality: 0.0-1.0 (AniList 0.95, Kitsu 0.9, DDG 0.6)
          - Feedback velocity: correct streak bonus
        """
        import math
        now = time.time()
        # Adaptive prior from global feedback (self-learning)
        # If overall accuracy high, prior is stronger
        try:
            with self._session() as s:
                total_fb = s.execute(select(func.count(Feedback.id))).scalar() or 0
                if total_fb >= 10:
                    good = s.execute(select(func.coalesce(func.sum(Feedback.correct), 0))).scalar() or 0
                    global_acc = good / max(1, total_fb)
                    # Prior adapts: high accuracy → stronger prior (alpha up)
                    alpha = 1.8 + global_acc * 1.8  # 1.8-3.6
                    beta = 1.2 + (1 - global_acc) * 1.4  # 1.2-2.6
                else:
                    alpha, beta = 2.0, 1.0
        except:
            alpha, beta = 2.0, 1.0
        # Posterior mean
        total_fb_title = correct + wrong
        if total_fb_title > 0:
            posterior = (correct + alpha) / (total_fb_title + alpha + beta)
            # Uncertainty penalty: high variance when few samples → shrink toward prior
            # Use Wilson-like adjustment: posterior * (1 - 1/(n+4))
            shrink = 1 - 1 / (total_fb_title + 4)
            posterior = 0.5 * (1 - shrink) + posterior * shrink
        else:
            posterior = 0.62  # slight pessimism for unseen
        # Frequency: log scale, but with soft cap and diminishing after 30
        freq = 4.5 * math.log1p(max(0, times_seen))
        if times_seen > 30:
            freq += 1.2 * math.log1p(times_seen - 30)
        freq = min(22.0, freq)
        # Recency: boost if seen in last 7 days, decay if >45 days
        recency = 0.0
        if last_seen:
            days = (now - last_seen) / 86400
            if days <= 7:
                recency = 3.5 * (1 - days/7)  # 0-3.5 boost
            elif days > 45:
                recency = -min(10.0, (days - 45) * 0.14)
        # Alias quality: avg weight of aliases for this title
        alias_bonus = 0.0
        if alias_weights:
            try:
                avg_w = sum(alias_weights) / len(alias_weights)
                # Weight 1.0 is neutral, >1.0 is good, <1.0 is poor
                alias_bonus = (avg_w - 1.0) * 6.5
                alias_bonus = max(-6, min(8, alias_bonus))
            except:
                pass
        # Source quality bonus
        source_bonus = (source_quality - 0.5) * 8  # -4 to +4
        # Feedback velocity: streak of correct
        fb_velocity = 0
        if correct > 0 and wrong == 0 and total_fb_title >= 3:
            fb_velocity = min(6, (correct - 2) * 1.2)
        # Small base from times_seen and source
        raw = base + posterior * 48 + freq + recency + alias_bonus + source_bonus + fb_velocity
        # Confidence should also reflect that titles with no feedback but many sightings are still moderately confident
        if total_fb_title == 0 and times_seen >= 5:
            raw = max(raw, 22 + min(18, times_seen * 1.1))
        return max(0.0, min(100.0, raw))

    def _ngram_sig_for(self, text: str) -> str:
        """Simple 3-gram signature for self-improvement re-rank."""
        try:
            from matcher import normalize
            norm = normalize(text)
            if not norm:
                return ""
            grams = sorted({norm[i:i+3] for i in range(max(1, len(norm)-2))})
            return "|".join(grams[:24])
        except Exception:
            return ""

    @_locked
    def learn_title(self, rec: TitleRecord):
        if not rec.key:
            return
        now = time.time()
        # Precompute ngram sig for this canonical
        sig = self._ngram_sig_for(rec.canonical)
        with self._session() as s:
            existing = s.get(Title, rec.key)
            if existing:
                old_conf = float(existing.confidence or 0)
                old_last = float(existing.last_seen or now)
                # Self-learning: apply decay if stale (>30 days not seen) BEFORE updating last_seen
                days_stale = (now - old_last) / 86400
                if days_stale > 30:
                    decay = min(12.0, (days_stale - 30) * 0.38)
                    existing.confidence = max(0.0, old_conf - decay)
                    old_conf = float(existing.confidence)
                existing.times_seen += 1
                existing.last_seen = now
                # Gather alias weights for this title for confidence bonus
                try:
                    alias_list = json.loads(existing.aliases or "[]")
                    alias_weights = []
                    for al in alias_list[:6]:
                        norm = normalize(al)
                        if not norm:
                            continue
                        arow = s.get(Alias, norm)
                        if arow and arow.weight:
                            alias_weights.append(float(arow.weight))
                    canon_norm = normalize(existing.canonical)
                    if canon_norm:
                        arow = s.get(Alias, canon_norm)
                        if arow and arow.weight:
                            alias_weights.append(float(arow.weight))
                except:
                    alias_weights = []
                source_q = 0.55
                if rec.anilist_id:
                    source_q = 0.96
                elif rec.mal_id:
                    source_q = 0.88
                elif rec.image:
                    source_q = 0.78
                elif rec.watch_links or rec.read_links:
                    source_q = 0.70
                new_conf = self._bayesian_confidence(
                    existing.feedback_correct or 0,
                    existing.feedback_wrong or 0,
                    existing.times_seen,
                    base=old_conf * 0.15 + 7.8,
                    last_seen=existing.last_seen,
                    alias_weights=alias_weights or None,
                    source_quality=source_q,
                )
                # Self-learning: re-seen must at least slightly increase (evidence), not decrease
                # For high conf >85, small +0.4; for low <40, bigger +1.8
                min_boost = 0.45 + max(0, (68 - old_conf) * 0.018)
                existing.confidence = min(100.0, max(new_conf, old_conf + min_boost))
                # Slight extra boost for being re-seen via web (evidence)
                existing.confidence = min(100.0, existing.confidence + 0.35)
                if rec.media_type and rec.media_type != "unknown":
                    existing.media_type = rec.media_type
                if rec.anilist_id:
                    existing.anilist_id = rec.anilist_id
                if rec.mal_id:
                    existing.mal_id = rec.mal_id
                if rec.image:
                    existing.image = rec.image
                # Merge links/aliases without duplication, keep most recent
                try:
                    wl = set(json.loads(existing.watch_links or "[]"))
                    wl.update(rec.watch_links or [])
                    existing.watch_links = json.dumps(sorted(wl)[:12])
                except: pass
                try:
                    rl = set(json.loads(existing.read_links or "[]"))
                    rl.update(rec.read_links or [])
                    existing.read_links = json.dumps(sorted(rl)[:12])
                except: pass
                try:
                    al = set(json.loads(existing.aliases or "[]"))
                    al.update(rec.aliases or [])
                    existing.aliases = json.dumps(sorted(al)[:24])
                except: pass
                if sig:
                    existing.ngram_sig = sig
            else:
                # New title: start with modest prior, but give bonus if it has image/links (evidence)
                # Source quality for new title
                source_q = 0.55
                if rec.anilist_id:
                    source_q = 0.96
                elif rec.mal_id:
                    source_q = 0.88
                elif rec.image:
                    source_q = 0.78
                elif rec.watch_links or rec.read_links:
                    source_q = 0.70
                bonus = 0
                if rec.image: bonus += 2
                if rec.watch_links: bonus += 1.5
                if rec.read_links: bonus += 1.5
                init_conf = self._bayesian_confidence(0, 0, 1, base=5.0 + bonus, last_seen=now, alias_weights=None, source_quality=source_q)
                s.add(Title(
                    key=rec.key, canonical=rec.canonical, media_type=rec.media_type,
                    external_id=rec.external_id, anilist_id=rec.anilist_id,
                    mal_id=rec.mal_id, image=rec.image,
                    aliases=json.dumps(rec.aliases),
                    watch_links=json.dumps(rec.watch_links),
                    read_links=json.dumps(rec.read_links),
                    first_seen=now, last_seen=now, times_seen=1, confidence=init_conf,
                    feedback_correct=0, feedback_wrong=0, last_feedback_ts=0,
                    ngram_sig=sig,
                ))
            s.commit()
        # Self-learning alias graph: canonical with higher weight, aliases with learned weight
        # Weight now reflects past success: weight = 1.8 + log1p(hits_correct) - 0.7*hits_wrong
        self._set_alias(rec.key, rec.key, weight=2.2)
        for a in rec.aliases:
            # Alias weight decays if it previously caused wrong hits
            self._set_alias(a, rec.key, weight=1.0)

    @_locked
    def _set_alias(self, alias: str, title_key: str, weight: float = 1.0):
        norm = normalize(alias)
        if not norm:
            return
        now = time.time()
        with self._session() as s:
            existing = s.get(Alias, norm)
            if existing:
                # Self-learning: weight is not just max, but Bayesian from hits
                # If alias was previously wrong, decay; if correct, boost
                # Keep max of provided weight and learned, but also apply time decay
                days_since_use = (now - (existing.last_used or existing.created_at or now)) / 86400
                decay = 0
                if days_since_use > 45 and (existing.hits_correct or 0) == 0:
                    decay = min(0.45, (days_since_use - 45) * 0.008)
                learned = 1.0 + max(0, (existing.hits_correct or 0) * 0.55 - (existing.hits_wrong or 0) * 0.75) - decay
                # Blend provided weight with learned
                blended = max(weight, learned, existing.weight * 0.92 + 0.08 * weight)
                existing.weight = min(3.6, max(0.25, blended))
                existing.title_key = title_key
                existing.last_used = now
            else:
                s.add(Alias(alias=norm, title_key=title_key, weight=weight, hits_correct=0, hits_wrong=0, last_used=now, created_at=now))
            s.commit()

    @_locked
    def get_title(self, key: str) -> dict | None:
        with self._session() as s:
            obj = s.get(Title, key)
            return _obj_dict(obj) if obj else None

    @_locked
    def alias_lookup(self, normalized: str) -> tuple[str, float] | None:
        """Resolve a normalized alias to (title_key, weight) under the DB lock."""
        with self._session() as s:
            obj = s.get(Alias, normalized)
            return (obj.title_key, obj.weight) if obj else None

    @_locked
    def all_follows(self) -> list[tuple[int, str]]:
        """Every (user_id, title_key) pair from the follows table, under the lock."""
        with self._session() as s:
            rows = s.execute(
                select(Follow.user_id, Follow.title_key).distinct()
            ).mappings().all()
            return [(int(r["user_id"]), r["title_key"]) for r in rows]

    @_locked
    def all_titles(self) -> list[dict]:
        with self._session() as s:
            rows = s.execute(
                select(Title).order_by(Title.times_seen.desc(), Title.confidence.desc())
            ).scalars().all()
            return [_obj_dict(o) for o in rows]

    @_locked
    def title_count(self) -> int:
        with self._session() as s:
            return s.execute(select(func.count(Title.key))).scalar() or 0

    # -------------------------------------------------------------- resources
    @_locked
    def upsert_resource(self, slug: str, name: str, kind: str, page_url: str,
                        status: str = "alive", note: str = ""):
        now = time.time()
        with self._session() as s:
            stmt = sqlite_insert(Resource).values(
                slug=slug, name=name, kind=kind, page_url=page_url,
                status=status, note=note, last_seen=now, last_checked=now,
            ).on_conflict_do_update(
                index_elements=["slug"],
                set_={
                    "name": name, "kind": kind, "page_url": page_url,
                    "status": status, "note": note,
                    "last_seen": now, "last_checked": now,
                },
            )
            s.execute(stmt)
            s.commit()

    @_locked
    def bulk_upsert_resources(self, rows: list[dict]):
        """Insert/update many resources in ONE transaction — avoids blocking."""
        now = time.time()
        data = [
            {
                "slug": r["slug"], "name": r["name"], "kind": r["kind"],
                "page_url": r["page_url"], "status": r.get("status", "alive"),
                "note": r.get("note", ""), "last_seen": now, "last_checked": now,
            }
            for r in rows
        ]
        with self._session() as s:
            stmt = sqlite_insert(Resource).on_conflict_do_update(
                index_elements=["slug"],
                set_={
                    "name": Resource.name, "kind": Resource.kind,
                    "page_url": Resource.page_url, "status": Resource.status,
                    "note": Resource.note, "last_seen": Resource.last_seen,
                    "last_checked": Resource.last_checked,
                },
            )
            s.execute(stmt, data)
            s.commit()

    @_locked
    def update_resource_host(self, slug: str, domain: str, search_url: str, search_param: str = ""):
        with self._session() as s:
            s.execute(
                update(Resource).where(Resource.slug == slug).values(
                    domain=domain, search_url=search_url, search_param=search_param,
                    last_checked=time.time(),
                )
            )
            s.commit()

    @_locked
    def mark_resource_dead(self, slug: str, note: str = ""):
        with self._session() as s:
            s.execute(
                update(Resource).where(Resource.slug == slug).values(
                    status="dead", dead_count=Resource.dead_count + 1,
                    note=note, last_checked=time.time(),
                )
            )
            s.commit()

    @_locked
    def mark_resource_alive(self, slug: str):
        with self._session() as s:
            s.execute(
                update(Resource).where(Resource.slug == slug).values(
                    status="alive", last_checked=time.time(),
                )
            )
            s.commit()

    @_locked
    def get_resources(self, kind: str = "", status: str = "") -> list[dict]:
        with self._session() as s:
            stmt = select(Resource)
            if kind and status:
                stmt = stmt.where(Resource.kind == kind, Resource.status == status) \
                           .order_by(Resource.name)
            elif kind:
                stmt = stmt.where(Resource.kind == kind).order_by(Resource.name)
            elif status:
                stmt = stmt.where(Resource.status == status) \
                           .order_by(Resource.kind, Resource.name)
            else:
                stmt = stmt.order_by(Resource.kind, Resource.name)
            return [_obj_dict(o) for o in s.execute(stmt).scalars().all()]

    @_locked
    def find_resource_by_name(self, name: str) -> dict | None:
        with self._session() as s:
            obj = s.execute(
                select(Resource).where(Resource.name.like(f"%{name}%"))
                .order_by(Resource.last_seen.desc()).limit(1)
            ).scalars().first()
            return _obj_dict(obj) if obj else None

    @_locked
    def resource_count(self) -> int:
        with self._session() as s:
            return s.execute(select(func.count(Resource.slug))).scalar() or 0

    @_locked
    def alive_resource_count(self) -> int:
        with self._session() as s:
            return s.execute(
                select(func.count(Resource.slug)).where(Resource.status == "alive")
            ).scalar() or 0

    # --------------------------------------------------------------- feedback (self-learning)
    @_locked
    def record_feedback(self, query: str, matched_key: str, correct: bool, match_method: str = "", score: float = 0.0):
        now = time.time()
        norm_q = normalize(query)
        with self._session() as s:
            s.add(Feedback(
                query=query, matched_key=matched_key,
                correct=1 if correct else 0, ts=now,
                match_method=match_method or "", score=score or 0.0,
            ))
            # Bayesian title update: increment correct/wrong counters and recompute confidence
            title = s.get(Title, matched_key)
            if title is not None:
                if correct:
                    title.feedback_correct = (title.feedback_correct or 0) + 1
                else:
                    title.feedback_wrong = (title.feedback_wrong or 0) + 1
                title.last_feedback_ts = now
                old_conf = float(title.confidence or 0)
                # Gather alias weights for this title for confidence bonus
                alias_weights = []
                alias_for_decay = None
                try:
                    alias_list = json.loads(title.aliases or "[]")
                    for al in alias_list[:6]:
                        norm = normalize(al)
                        if not norm:
                            continue
                        arow = s.get(Alias, norm)
                        if arow and arow.weight:
                            alias_weights.append(float(arow.weight))
                    canon_norm = normalize(title.canonical)
                    if canon_norm:
                        arow = s.get(Alias, canon_norm)
                        if arow and arow.weight:
                            alias_weights.append(float(arow.weight))
                except:
                    alias_weights = []
                try:
                    if norm_q:
                        alias_for_decay = s.get(Alias, norm_q)
                except:
                    alias_for_decay = None
                source_q = 0.62
                if title.anilist_id:
                    source_q = 0.96
                elif title.mal_id:
                    source_q = 0.88
                elif title.image:
                    source_q = 0.78
                new_conf = self._bayesian_confidence(
                    title.feedback_correct or 0,
                    title.feedback_wrong or 0,
                    title.times_seen or 1,
                    base=old_conf * 0.14 + 7.2,
                    last_seen=title.last_seen or now,
                    alias_weights=alias_weights or None,
                    source_quality=source_q,
                )
                # Self-learning: correct must always increase from old, wrong must decrease
                # Use momentum: correct +1.5 to +3.2 depending on streak, wrong -1.2 to -2.8
                if correct:
                    # At least +1.4, more if low confidence, less if already 95+
                    boost = 1.6 + max(0, (75 - old_conf) * 0.035)  # low titles get bigger boost
                    if (title.feedback_correct or 0) >= 3 and (title.feedback_wrong or 0) == 0:
                        boost += 0.9  # perfect streak bonus
                    title.confidence = min(100.0, max(new_conf, old_conf + boost))
                else:
                    penalty = 1.3 + max(0, (old_conf - 60) * 0.018)  # high conf penalized more
                    if alias_for_decay and (alias_for_decay.hits_wrong or 0) > (alias_for_decay.hits_correct or 0):
                        penalty += 0.8
                    title.confidence = max(0.0, min(new_conf, old_conf - penalty))
                    # Also ensure not too low if still has corrects
                    if (title.feedback_correct or 0) > (title.feedback_wrong or 0):
                        title.confidence = max(12.0, title.confidence)
                if not title.ngram_sig:
                    title.ngram_sig = self._ngram_sig_for(title.canonical)
            # Alias performance tracking
            if norm_q:
                alias = s.get(Alias, norm_q)
                if alias is not None:
                    if correct:
                        alias.hits_correct = (alias.hits_correct or 0) + 1
                        alias.weight = min(3.8, (alias.weight or 1.0) + 0.32)
                    else:
                        alias.hits_wrong = (alias.hits_wrong or 0) + 1
                        # Decay weight more aggressively for wrong, but not below 0.25
                        alias.weight = max(0.25, (alias.weight or 1.0) - 0.42)
                    alias.last_used = now
                    # If alias was wrong and weight very low, consider reassigning? Keep for now.
                elif correct:
                    # If correct and alias didn't exist, create it with high weight (new learning)
                    # This is handled below via _set_alias, but we also want to record hits
                    pass
            s.commit()
        # Self-learning alias graph: create/update alias with performance-aware weight
        if correct:
            # Strong positive: weight 3.0 will be blended with learned above, but ensure at least 2.8
            self._set_alias(query, matched_key, weight=3.0)
            # Also decay competing aliases that might be wrong? Not needed now.
        else:
            # Negative feedback: record wrong hit for that alias if exists, and slightly demote
            # We already updated hits_wrong above, but also ensure alias weight decays
            try:
                # Touch alias to apply decay even if it was not found via norm_q (e.g., typo)
                self._set_alias(query, matched_key, weight=0.45)
            except Exception:
                pass

    # --------------------------------------------------------------- self-improvement
    @_locked
    def prune_stale_aliases(self, max_age_days: int = 90, min_weight: float = 0.32) -> int:
        """Self-improvement: prune aliases that are stale, low weight, and never correct."""
        now = time.time()
        cutoff = now - max_age_days * 86400
        pruned = 0
        try:
            with self._session() as s:
                q = select(Alias).where(
                    (Alias.weight < min_weight) &
                    ((Alias.last_used == 0) | (Alias.last_used < cutoff)) &
                    (Alias.hits_correct == 0)
                )
                stale = s.execute(q).scalars().all()
                for a in stale:
                    # Don't prune canonical self-aliases (alias == title_key)
                    if a.alias == a.title_key:
                        continue
                    s.delete(a)
                    pruned += 1
                if pruned:
                    s.commit()
        except Exception:
            pass
        return pruned

    @_locked
    def decay_stale_titles(self, stale_days: int = 60, decay_per_day: float = 0.12) -> int:
        """Self-improvement: decay confidence for titles not seen in a while."""
        now = time.time()
        decayed = 0
        try:
            with self._session() as s:
                cutoff = now - stale_days * 86400
                q = select(Title).where(Title.last_seen < cutoff)
                for t in s.execute(q).scalars().all():
                    days = (now - (t.last_seen or now)) / 86400 - stale_days
                    dec = min(18.0, max(0, days * decay_per_day))
                    new_conf = max(0.0, (t.confidence or 0) - dec)
                    # Don't decay below 2 if it has correct feedback
                    if (t.feedback_correct or 0) > 0:
                        new_conf = max(4.0, new_conf)
                    if abs(new_conf - (t.confidence or 0)) > 0.3:
                        t.confidence = new_conf
                        decayed += 1
                if decayed:
                    s.commit()
        except Exception:
            pass
        return decayed

    @_locked
    def auto_tune_threshold(self) -> tuple[float, dict]:
        """Self-improvement: tune MATCH_THRESHOLD based on recent feedback window."""
        import config
        now = time.time()
        window = 7 * 86400  # last 7 days
        try:
            with self._session() as s:
                fb = s.execute(select(Feedback).where(Feedback.ts > now - window)).scalars().all()
                if len(fb) < 8:
                    return float(config.MATCH_THRESHOLD), {"reason": "not enough feedback", "n": len(fb)}
                # Group by score bucket
                correct_scores = [f.score for f in fb if f.correct]
                wrong_scores = [f.score for f in fb if not f.correct]
                if not correct_scores or not wrong_scores:
                    return float(config.MATCH_THRESHOLD), {"reason": "one-sided", "n": len(fb)}
                avg_c = sum(correct_scores) / len(correct_scores)
                avg_w = sum(wrong_scores) / len(wrong_scores)
                # Ideal threshold is midway, but biased slightly to precision
                ideal = (avg_c * 0.62 + avg_w * 0.38)
                # Clamp 58-84
                ideal = max(58.0, min(84.0, ideal))
                # Smooth: move 22% toward ideal
                cur = float(config.MATCH_THRESHOLD)
                new_thr = cur * 0.78 + ideal * 0.22
                return round(new_thr, 1), {"avg_correct": round(avg_c,1), "avg_wrong": round(avg_w,1), "ideal": round(ideal,1), "n": len(fb)}
        except Exception as ex:
            return float(config.MATCH_THRESHOLD), {"error": str(ex)}

    @_locked
    def recompute_alias_weights_from_feedback(self) -> int:
        """Self-improvement: recompute alias weights from feedback history (batch)."""
        updated = 0
        try:
            with self._session() as s:
                # For each alias, recompute from feedback
                aliases = s.execute(select(Alias)).scalars().all()
                for a in aliases:
                    # Count feedback for this alias (query normalized == alias)
                    fb = s.execute(select(Feedback).where(Feedback.query == a.alias)).scalars().all()
                    if not fb:
                        continue
                    c = sum(1 for f in fb if f.correct)
                    w = len(fb) - c
                    # Bayesian weight: 1.0 + 0.5*c - 0.7*w, clamped
                    new_w = 1.0 + c * 0.52 - w * 0.68
                    # Boost if recent correct
                    if c > 0:
                        new_w = max(new_w, 1.35)
                    new_w = max(0.25, min(3.8, new_w))
                    if abs(new_w - (a.weight or 1.0)) > 0.08:
                        a.weight = new_w
                        a.hits_correct = c
                        a.hits_wrong = w
                        updated += 1
                if updated:
                    s.commit()
        except Exception:
            pass
        return updated

    @_locked
    def get_learning_report(self) -> dict:
        """Self-improvement: snapshot for /stats and logs."""
        try:
            with self._session() as s:
                total_titles = s.execute(select(func.count(Title.key))).scalar() or 0
                avg_conf = s.execute(select(func.coalesce(func.avg(Title.confidence), 0.0))).scalar() or 0.0
                # Feedback window 7d
                now = time.time()
                fb7 = s.execute(select(Feedback).where(Feedback.ts > now - 7*86400)).scalars().all()
                c7 = sum(1 for f in fb7 if f.correct)
                w7 = len(fb7) - c7
                acc7 = (c7 / len(fb7) * 100) if fb7 else 0.0
                # Alias quality
                alias_cnt = s.execute(select(func.count(Alias.alias))).scalar() or 0
                low_alias = s.execute(select(func.count(Alias.alias)).where(Alias.weight < 0.6)).scalar() or 0
                # Learning metrics history
                hist = s.execute(select(LearningMetrics).order_by(LearningMetrics.window_end.desc()).limit(5)).scalars().all()
                hist_list = [{"window_end": h.window_end, "accuracy": (h.correct_hits/max(1,h.total_queries)*100) if h.total_queries else 0, "thr": h.threshold_used} for h in hist]
                return {
                    "total_titles": total_titles,
                    "avg_confidence": round(avg_conf,1),
                    "feedback_7d": len(fb7),
                    "accuracy_7d": round(acc7,1),
                    "correct_7d": c7,
                    "wrong_7d": w7,
                    "alias_count": alias_cnt,
                    "low_alias": low_alias,
                    "history": hist_list,
                }
        except Exception as ex:
            return {"error": str(ex)}

    @_locked
    def log_learning_window(self, threshold_used: float):
        """Persist a learning window for self-improvement history."""
        now = time.time()
        window = 7*86400
        try:
            with self._session() as s:
                fb = s.execute(select(Feedback).where(Feedback.ts > now - window)).scalars().all()
                total = len(fb)
                correct = sum(1 for f in fb if f.correct)
                wrong = total - correct
                avg_conf = s.execute(select(func.coalesce(func.avg(Title.confidence), 0.0))).scalar() or 0.0
                # Weights snapshot
                try:
                    import json as _json
                    from matcher import get_weights
                    wj = _json.dumps(get_weights())
                except: wj = "{}"
                s.add(LearningMetrics(
                    window_start=now - window, window_end=now,
                    total_queries=total, correct_hits=correct, wrong_hits=wrong,
                    avg_confidence=float(avg_conf), threshold_used=float(threshold_used),
                    weights_json=wj,
                ))
                s.commit()
        except Exception:
            pass

    # --------------------------------------------------------------- self-learning sources
    @_locked
    def sync_news_sources_from_config(self) -> int:
        """Self-learning bootstrap: ensure config RSS_SOURCES are in DB with scoring."""
        try:
            import config as _cfg
            added = 0
            with self._session() as s:
                for src in _cfg.RSS_SOURCES:
                    url = src.get("url","").strip()
                    if not url:
                        continue
                    existing = s.get(NewsSource, url)
                    if existing:
                        # Update name/kind if changed, but keep learned reliability
                        if existing.name != src.get("name",""):
                            existing.name = src.get("name","")
                        continue
                    s.add(NewsSource(
                        url=url, name=src.get("name",""), kind=src.get("kind","news"),
                        status="active", reliability=0.55, success_count=0, fail_count=0,
                        avg_items=3.0, avg_latency_ms=0, last_checked=0, last_success=0,
                        discovered_via="config", score=0.55, consecutive_fails=0
                    ))
                    added += 1
                if added:
                    s.commit()
            # Also ensure title search sources exist
            with self._session() as s:
                for name in ["anilist","kitsu","jikan","ddg","mal"]:
                    if not s.get(TitleSearchSource, name):
                        s.add(TitleSearchSource(name=name, reliability=0.5, weight=1.0))
                s.commit()
            return added
        except Exception:
            return 0

    @_locked
    def get_active_news_sources(self, kind: str = "", limit: int = 12) -> list[dict]:
        """Get active news sources ordered by self-learned score (reliability)."""
        with self._session() as s:
            stmt = select(NewsSource).where(NewsSource.status == "active")
            if kind and kind != "all":
                stmt = stmt.where(NewsSource.kind == kind)
            # Order by score desc, then reliability
            stmt = stmt.order_by(NewsSource.score.desc(), NewsSource.reliability.desc()).limit(limit)
            rows = s.execute(stmt).scalars().all()
            # Fallback to config if DB empty
            if not rows:
                try:
                    import config as _cfg
                    return [dict(url=src["url"], name=src["name"], kind=src["kind"], reliability=0.5, score=0.5) for src in _cfg.RSS_SOURCES[:limit]]
                except:
                    return []
            return [_obj_dict(r) for r in rows]

    @_locked
    def get_all_news_sources(self) -> list[dict]:
        with self._session() as s:
            rows = s.execute(select(NewsSource)).scalars().all()
            return [_obj_dict(r) for r in rows]

    @_locked
    def update_news_source_stats(self, url: str, success: bool, item_count: int = 0, latency_ms: float = 0):
        """Self-learning: update reliability based on fetch result."""
        now = time.time()
        try:
            with self._session() as s:
                src = s.get(NewsSource, url)
                if not src:
                    return
                src.last_checked = now
                if success:
                    src.success_count = (src.success_count or 0) + 1
                    src.consecutive_fails = 0
                    src.last_success = now
                    # Update avg_items with EMA
                    prev = src.avg_items or 0
                    src.avg_items = prev * 0.78 + item_count * 0.22 if prev else float(item_count)
                    # Latency EMA
                    if latency_ms:
                        prev_lat = src.avg_latency_ms or latency_ms
                        src.avg_latency_ms = prev_lat * 0.82 + latency_ms * 0.18
                    # Reliability: success rate with smoothing + recency
                    total = (src.success_count or 0) + (src.fail_count or 0)
                    succ_rate = (src.success_count or 0) / max(1, total)
                    # Boost for high avg_items (productive) and low latency
                    prod_boost = min(0.12, (src.avg_items or 0) * 0.015)
                    latency_penalty = min(0.08, (src.avg_latency_ms or 0) / 8000)
                    new_rel = 0.60 * succ_rate + 0.25 * (1 - min(1, (src.fail_count or 0)/12)) + prod_boost - latency_penalty
                    src.reliability = max(0.05, min(0.98, new_rel * 0.88 + (src.reliability or 0.5) * 0.12))
                    # Score for ranking: reliability * 0.7 + normalized avg_items *0.2 + recency 0.1
                    recency = 1.0 if (now - (src.last_success or 0)) < 86400 else 0.7 if (now - src.last_success) < 3*86400 else 0.4
                    src.score = max(0.05, min(0.98, src.reliability * 0.68 + min(1, (src.avg_items or 0)/8) * 0.18 + recency * 0.14))
                    if src.score < 0.18 and (src.fail_count or 0) >= 4:
                        src.status = "dead"
                else:
                    src.fail_count = (src.fail_count or 0) + 1
                    src.consecutive_fails = (src.consecutive_fails or 0) + 1
                    total = (src.success_count or 0) + (src.fail_count or 0)
                    succ_rate = (src.success_count or 0) / max(1, total)
                    src.reliability = max(0.05, min(0.98, succ_rate * 0.65 + (src.reliability or 0.5) * 0.35 - 0.08))
                    src.score = max(0.05, src.reliability * 0.75)
                    # Auto-pause after 3 consecutive fails, dead after 7
                    if src.consecutive_fails >= 7:
                        src.status = "dead"
                    elif src.consecutive_fails >= 3:
                        src.status = "paused"
                s.commit()
        except Exception:
            pass

    @_locked
    def discover_new_sources_from_web(self, candidates: list[dict]) -> int:
        """Add newly discovered RSS URLs from web search (self-learning)."""
        added = 0
        now = time.time()
        try:
            with self._session() as s:
                for c in candidates:
                    url = (c.get("url") or "").strip()
                    if not url or not url.startswith("http"):
                        continue
                    # Only RSS-like URLs
                    low = url.lower()
                    if not any(x in low for x in [".xml", "/rss", "/feed", "rss.", "feed"]):
                        # Allow if discovered via RSS discovery and looks like feed
                        if "discovered_via" not in c or c.get("kind") != "news":
                            continue
                    if s.get(NewsSource, url):
                        continue
                    # Basic validation: must be http and not junk
                    if any(d in low for d in ["duckduckgo.com", "google.com", "youtube.com"]):
                        continue
                    s.add(NewsSource(
                        url=url, name=c.get("name", url[:40]), kind=c.get("kind","news"),
                        status="active", reliability=0.45, success_count=0, fail_count=0,
                        avg_items=2.0, avg_latency_ms=0, last_checked=0, last_success=0,
                        discovered_via=c.get("discovered_via","web_discovery"), score=0.45, consecutive_fails=0
                    ))
                    added += 1
                if added:
                    s.commit()
        except Exception:
            pass
        return added

    @_locked
    def get_title_search_sources(self) -> list[dict]:
        with self._session() as s:
            rows = s.execute(select(TitleSearchSource).order_by(TitleSearchSource.weight.desc())).scalars().all()
            if not rows:
                # Seed defaults
                for name in ["anilist","kitsu","jikan","ddg","mal"]:
                    if not s.get(TitleSearchSource, name):
                        s.add(TitleSearchSource(name=name, reliability=0.5, weight=1.0))
                s.commit()
                rows = s.execute(select(TitleSearchSource).order_by(TitleSearchSource.weight.desc())).scalars().all()
            return [_obj_dict(r) for r in rows]

    @_locked
    def update_title_source_stats(self, name: str, success: bool, latency_ms: float = 0):
        now = time.time()
        try:
            with self._session() as s:
                src = s.get(TitleSearchSource, name)
                if not src:
                    src = TitleSearchSource(name=name, reliability=0.5, weight=1.0)
                    s.add(src)
                src.last_checked = now
                if success:
                    src.success_count = (src.success_count or 0) + 1
                    src.consecutive_fails = 0
                    src.last_success = now
                    # EMA latency
                    if latency_ms:
                        prev = src.avg_latency_ms or latency_ms
                        src.avg_latency_ms = prev * 0.8 + latency_ms * 0.2
                    total = (src.success_count or 0) + (src.fail_count or 0)
                    succ_rate = (src.success_count or 0) / max(1, total)
                    # Weight for candidate scoring: higher for reliable + fast
                    latency_factor = max(0.7, 1 - (src.avg_latency_ms or 0)/6000)
                    src.reliability = max(0.05, min(0.97, succ_rate * 0.75 + (src.reliability or 0.5)*0.25))
                    src.weight = max(0.4, min(1.8, src.reliability * 1.1 * latency_factor + 0.35))
                else:
                    src.fail_count = (src.fail_count or 0) + 1
                    src.consecutive_fails = (src.consecutive_fails or 0) + 1
                    total = (src.success_count or 0) + (src.fail_count or 0)
                    succ_rate = (src.success_count or 0) / max(1, total)
                    src.reliability = max(0.05, succ_rate * 0.6 + (src.reliability or 0.5)*0.4 - 0.07)
                    src.weight = max(0.35, src.reliability * 0.9 + 0.25)
                    # No auto-dead for title sources, just lower weight
                s.commit()
        except Exception:
            pass

    @_locked
    def prune_dead_news_sources(self, max_keep: int = 40) -> int:
        """Self-cleaning: keep at most max_keep active sources, prune lowest score dead/paused."""
        pruned = 0
        try:
            with self._session() as s:
                # Count active
                active_cnt = s.execute(select(func.count(NewsSource.url)).where(NewsSource.status == "active")).scalar() or 0
                if active_cnt > max_keep:
                    # Demote lowest score actives to paused
                    rows = s.execute(select(NewsSource).where(NewsSource.status == "active").order_by(NewsSource.score.asc()).limit(active_cnt - max_keep)).scalars().all()
                    for r in rows:
                        r.status = "paused"
                        pruned += 1
                # Delete very old dead with low reliability
                cutoff = time.time() - 45*86400
                dead_rows = s.execute(select(NewsSource).where((NewsSource.status == "dead") & (NewsSource.last_checked < cutoff) & (NewsSource.reliability < 0.22))).scalars().all()
                for r in dead_rows:
                    s.delete(r)
                    pruned += 1
                if pruned:
                    s.commit()
        except Exception:
            pass
        return pruned

    # --------------------------------------------------------------- scrapelog
    @_locked
    def log_scrape(self, source: str, count: int, duration_ms: float, ok: bool):
        with self._session() as s:
            s.add(ScrapeLog(
                source=source, count=count, duration_ms=duration_ms,
                ok=1 if ok else 0, ts=time.time(),
            ))
            s.commit()

    # ----------------------------------------------------------------- follows (per-guild isolated)
    @_locked
    def follow(self, user_id: str, title_key: str, guild_id: str | int | None = None):
        gid = str(guild_id) if guild_id not in (None, 0, "0") else "0"
        if guild_id is None:
            gid = "0"
        with self._session() as s:
            s.execute(
                sqlite_insert(Follow).values(
                    user_id=str(user_id), title_key=title_key, guild_id=str(gid), created_at=time.time(),
                ).on_conflict_do_nothing(index_elements=["user_id", "title_key", "guild_id"])
            )
            s.commit()

    @_locked
    def unfollow(self, user_id: str, title_key: str, guild_id: str | int | None = None):
        gid = str(guild_id) if guild_id not in (None, "") else "0"
        with self._session() as s:
            # if guild_id provided, only remove that guild's follow; if None, remove all guilds' copies
            if guild_id is not None:
                s.execute(
                    delete(Follow).where(
                        Follow.user_id == str(user_id), Follow.title_key == title_key, Follow.guild_id == str(gid))
                )
            else:
                s.execute(
                    delete(Follow).where(
                        Follow.user_id == str(user_id), Follow.title_key == title_key)
                )
            s.commit()

    @_locked
    def followed(self, user_id: str, guild_id: str | int | None = None) -> list[str]:
        with self._session() as s:
            stmt = select(Follow.title_key).where(Follow.user_id == str(user_id))
            if guild_id is not None:
                gid = str(guild_id)
                stmt = stmt.where(Follow.guild_id == gid)
            rows = s.execute(stmt).scalars().all()
            return list(rows)

    @_locked
    def all_follows(self) -> list[tuple[int, str]]:
        """Every (user_id, title_key) pair from the follows table, under the lock.
        For watch_loop we need guild-aware variant too."""
        with self._session() as s:
            rows = s.execute(
                select(Follow.user_id, Follow.title_key).distinct()
            ).mappings().all()
            return [(int(r["user_id"]), r["title_key"]) for r in rows]

    @_locked
    def all_follows_detailed(self) -> list[dict]:
        """Guild-aware follows for per-server delivery."""
        with self._session() as s:
            rows = s.execute(select(Follow)).scalars().all()
            return [_obj_dict(o) for o in rows]

    # ------------------------------------------------------------- broadcasts (per-guild isolated)
    @_locked
    def set_broadcast(self, user_id: str, channel_id: int = 0, guild_id: str | int | None = None):
        """Enable auto-news per guild; each server/person combo is isolated."""
        gid = str(guild_id) if guild_id not in (None, 0, "0", "") else "0"
        if guild_id is None and channel_id == 0:
            gid = "0"  # DM personal
        elif guild_id is None and channel_id != 0:
            # infer guild from channel? caller should pass guild_id; fallback to 0
            gid = "0"
        now = time.time()
        with self._session() as s:
            # try composite unique user_id+guild_id
            existing = s.execute(select(Broadcast).where(Broadcast.user_id == str(user_id), Broadcast.guild_id == str(gid))).scalars().first()
            if existing:
                existing.channel_id = channel_id
                existing.guild_id = str(gid)
                if not existing.last_sent:
                    existing.last_sent = now
            else:
                s.add(Broadcast(
                    user_id=str(user_id), guild_id=str(gid), channel_id=channel_id,
                    created_at=now, last_sent=now,
                ))
            s.commit()

    @_locked
    def stop_broadcast(self, user_id: str, guild_id: str | int | None = None) -> bool:
        with self._session() as s:
            if guild_id is not None:
                gid = str(guild_id)
                objs = s.execute(select(Broadcast).where(Broadcast.user_id == str(user_id), Broadcast.guild_id == gid)).scalars().all()
                if not objs:
                    return False
                for o in objs:
                    s.delete(o)
            else:
                # stop all for user across all guilds
                objs = s.execute(select(Broadcast).where(Broadcast.user_id == str(user_id))).scalars().all()
                if not objs:
                    return False
                for o in objs:
                    s.delete(o)
            s.commit()
            return True

    @_locked
    def all_broadcasts(self) -> list[dict]:
        with self._session() as s:
            rows = s.execute(select(Broadcast)).scalars().all()
            return [_obj_dict(o) for o in rows]

    @_locked
    def get_broadcast(self, user_id: str, guild_id: str | int | None) -> dict | None:
        gid = str(guild_id) if guild_id not in (None, "") else "0"
        with self._session() as s:
            obj = s.execute(select(Broadcast).where(Broadcast.user_id == str(user_id), Broadcast.guild_id == gid)).scalars().first()
            return _obj_dict(obj) if obj else None

    @_locked
    def touch_broadcast(self, user_id: str, ts: float, guild_id: str | int | None = None):
        with self._session() as s:
            if guild_id is not None:
                obj = s.execute(select(Broadcast).where(Broadcast.user_id == str(user_id), Broadcast.guild_id == str(guild_id))).scalars().first()
            else:
                # fallback: first match
                obj = s.execute(select(Broadcast).where(Broadcast.user_id == str(user_id))).scalars().first()
            if obj:
                obj.last_sent = ts
                s.commit()

    @_locked
    def news_since(self, ts: float, limit: int = 12) -> list[dict]:
        """Fresh items ingested after ts (for the next auto-news digest)."""
        with self._session() as s:
            rows = s.execute(
                select(Item).where(Item.fetched_at > ts)
                .order_by(Item.fetched_at.desc()).limit(limit)
            ).scalars().all()
            return [_obj_dict(o) for o in rows]

    # ------------------------------------------------------------------ stats — upgraded proficiency (self-learning, multi-factor)
    @_locked
    def stats(self) -> dict:
        import math, time as _time
        now = _time.time()
        with self._session() as s:
            items = s.execute(select(func.count(Item.id))).scalar() or 0
            titles = s.execute(select(func.count(Title.key))).scalar() or 0
            resources = s.execute(select(func.count(Resource.slug))).scalar() or 0
            alive_res = s.execute(select(func.count(Resource.slug)).where(Resource.status == "alive")).scalar() or 0
            total_fb = s.execute(select(func.count(Feedback.id))).scalar() or 0
            good = s.execute(select(func.coalesce(func.sum(Feedback.correct), 0))).scalar() or 0
            avg_conf = s.execute(select(func.coalesce(func.avg(Title.confidence), 0.0))).scalar() or 0.0
            avg_times = s.execute(select(func.coalesce(func.avg(Title.times_seen), 0.0))).scalar() or 0.0
            # Recent velocity: titles/items in last 7 days
            week_ago = now - 7*86400
            recent_titles = s.execute(select(func.count(Title.key)).where(Title.first_seen > week_ago)).scalar() or 0
            recent_items = s.execute(select(func.count(Item.id)).where(Item.fetched_at > week_ago)).scalar() or 0
            recent_fb = s.execute(select(func.count(Feedback.id)).where(Feedback.ts > week_ago)).scalar() or 0
            # Alias quality
            alias_cnt = s.execute(select(func.count(Alias.alias))).scalar() or 0
            high_alias = s.execute(select(func.count(Alias.alias)).where(Alias.weight >= 1.8)).scalar() or 0
        # --- Mastery: avg confidence 0-100
        mastery = max(0.0, min(100.0, float(avg_conf)))
        # --- Accuracy: with Beta(2,2) smoothing so 0 feedback = 50% neutral, not 0
        # Prior 50% prevents new bots looking incompetent
        acc_raw = (good / total_fb * 100) if total_fb else 50.0
        # Smoothed accuracy: (good+2)/(total+4)*100
        acc_smooth = ((good + 2) / (total_fb + 4) * 100) if total_fb else 50.0
        # Blend raw and smooth 70/30, but if total_fb <5, use smooth
        accuracy = acc_smooth if total_fb < 12 else (acc_raw * 0.72 + acc_smooth * 0.28)
        # --- Breadth: log scale up to 1000 titles
        breadth = 100 * (math.log10(max(1, titles) + 1) / math.log10(1001))
        breadth = max(0.0, min(100.0, breadth))
        # --- Depth: items per title + avg times_seen
        # Ideal: 12 items per title and avg times_seen 8
        ipp = (items / max(1, titles)) if titles else 0
        depth_items = min(100.0, (ipp / 12) * 100)
        depth_freq = min(100.0, (float(avg_times) / 8) * 100)
        depth = depth_items * 0.62 + depth_freq * 0.38
        # --- Vitality: resources alive + recent activity
        vitality_res = (alive_res / max(1, resources) * 100) if resources else 0
        vitality_recent = min(100.0, (recent_items / 28) * 100)  # 28 items/week = 100
        vitality = vitality_res * 0.65 + vitality_recent * 0.35
        # --- Velocity: learning speed last 7 days
        # Titles: 7 per week = 100, items: 40 per week = 100
        vel_titles = min(100.0, (recent_titles / 7) * 100)
        vel_items = min(100.0, (recent_items / 40) * 100)
        # Feedback velocity also matters
        vel_fb = min(100.0, (recent_fb / 8) * 100)
        velocity = vel_titles * 0.42 + vel_items * 0.38 + vel_fb * 0.20
        # --- Alias quality
        alias_quality = (high_alias / max(1, alias_cnt) * 100) if alias_cnt else 0
        # --- Composite proficiency (weights sum to 1.0)
        # More weight to mastery+accuracy as core, but breadth/depth/vitality/velocity matter for growth
        # Also add small alias quality bonus
        proficiency = (
            mastery * 0.30 +
            accuracy * 0.25 +
            breadth * 0.15 +
            depth * 0.10 +
            vitality * 0.10 +
            velocity * 0.07 +
            alias_quality * 0.03
        )
        # Self-learning boost: if recent accuracy high (>78) and titles growing, add momentum
        momentum = 0
        if accuracy > 78 and recent_titles >= 2:
            momentum = min(4.5, (accuracy - 78) * 0.12 + recent_titles * 0.35)
        proficiency = min(100.0, proficiency + momentum)
        # Ensure proficiency is not 0 when there's at least some data
        if (titles + items) > 0 and proficiency < 8:
            proficiency = 8 + min(12, (titles * 0.6 + items * 0.08))
        return {
            "items": items, "titles": titles, "resources": resources, "alive_resources": alive_res,
            "feedback": total_fb, "feedback_7d": recent_fb,
            "accuracy": round(acc_raw, 1), "accuracy_smooth": round(acc_smooth,1), "avg_confidence": round(float(avg_conf), 1),
            "avg_times_seen": round(float(avg_times),2),
            "breadth": round(breadth,1), "depth": round(depth,1), "vitality": round(vitality,1), "velocity": round(velocity,1), "alias_quality": round(alias_quality,1),
            "proficiency": round(proficiency,1),
            "proficiency_breakdown": {
                "mastery": round(mastery,1), "accuracy": round(accuracy,1), "breadth": round(breadth,1),
                "depth": round(depth,1), "vitality": round(vitality,1), "velocity": round(velocity,1), "alias_quality": round(alias_quality,1),
                "momentum": round(momentum,1)
            }
        }

    @_locked
    def close(self):
        self.engine.dispose()


if __name__ == "__main__":
    db = KnowledgeDB()
    print("stats:", db.stats())
    db.close()