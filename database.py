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
    PrimaryKeyConstraint,
    String,
    Text,
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
    aliases: Mapped[str] = mapped_column(Text, default="[]")
    watch_links: Mapped[str] = mapped_column(Text, default="[]")
    read_links: Mapped[str] = mapped_column(Text, default="[]")
    first_seen: Mapped[float] = mapped_column(Float, default=0.0)
    last_seen: Mapped[float] = mapped_column(Float, default=0.0)
    times_seen: Mapped[int] = mapped_column(Integer, default=1)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)


class Alias(Base):
    __tablename__ = "aliases"

    alias: Mapped[str] = mapped_column(String, primary_key=True)
    title_key: Mapped[str] = mapped_column(String, default="")
    weight: Mapped[float] = mapped_column(Float, default=1.0)


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(Text, default="")
    matched_key: Mapped[str] = mapped_column(String, default="")
    correct: Mapped[int] = mapped_column(Integer, default=0)
    ts: Mapped[float] = mapped_column(Float, default=0.0)


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
    __table_args__ = (PrimaryKeyConstraint("user_id", "title_key"),)

    user_id: Mapped[str] = mapped_column(String)
    title_key: Mapped[str] = mapped_column(String)


class Broadcast(Base):
    """Per-user auto-news: who wants it, which guild channel to post in, and
    when the last digest went out (so each cycle only sends fresh news)."""

    __tablename__ = "broadcasts"

    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    channel_id: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[float] = mapped_column(Float, default=0.0)
    last_sent: Mapped[float] = mapped_column(Float, default=0.0)


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
    @_locked
    def learn_title(self, rec: TitleRecord):
        # Titles that normalize to "" (e.g. native-only names) would collide on
        # the PRIMARY KEY and raise IntegrityError -- skip them.
        if not rec.key:
            return
        now = time.time()
        with self._session() as s:
            existing = s.get(Title, rec.key)
            if existing:
                existing.times_seen += 1
                existing.last_seen = now
                existing.confidence = min(100.0, (existing.confidence or 0.0) + 2.0)
                if rec.media_type:
                    existing.media_type = rec.media_type
                if rec.anilist_id:
                    existing.anilist_id = rec.anilist_id
                if rec.mal_id:
                    existing.mal_id = rec.mal_id
                existing.watch_links = json.dumps(
                    rec.watch_links or json.loads(existing.watch_links or "[]"))
                existing.read_links = json.dumps(
                    rec.read_links or json.loads(existing.read_links or "[]"))
                existing.aliases = json.dumps(sorted(set(
                    rec.aliases or json.loads(existing.aliases or "[]"))))
            else:
                s.add(Title(
                    key=rec.key, canonical=rec.canonical, media_type=rec.media_type,
                    external_id=rec.external_id, anilist_id=rec.anilist_id,
                    mal_id=rec.mal_id, aliases=json.dumps(rec.aliases),
                    watch_links=json.dumps(rec.watch_links),
                    read_links=json.dumps(rec.read_links),
                    first_seen=now, last_seen=now, times_seen=1, confidence=5.0,
                ))
            s.commit()
        # register canonical + aliases (separate sessions, still under the lock)
        self._set_alias(rec.key, rec.key, weight=2.0)
        for a in rec.aliases:
            self._set_alias(a, rec.key, weight=1.0)

    @_locked
    def _set_alias(self, alias: str, title_key: str, weight: float = 1.0):
        norm = normalize(alias)
        if not norm:
            return
        with self._session() as s:
            existing = s.get(Alias, norm)
            if existing:
                existing.weight = max(existing.weight, weight)
                existing.title_key = title_key
            else:
                s.add(Alias(alias=norm, title_key=title_key, weight=weight))
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

    # --------------------------------------------------------------- feedback
    @_locked
    def record_feedback(self, query: str, matched_key: str, correct: bool):
        with self._session() as s:
            s.add(Feedback(
                query=query, matched_key=matched_key,
                correct=1 if correct else 0, ts=time.time(),
            ))
            # reinforce alias when a human confirms a match
            if correct:
                s.execute(
                    update(Title).where(Title.key == matched_key).values(
                        confidence=func.min(100.0, Title.confidence + 5.0))
                )
            else:
                s.execute(
                    update(Title).where(Title.key == matched_key).values(
                        confidence=func.max(0.0, Title.confidence - 3.0))
                )
            s.commit()
        if correct:
            self._set_alias(query, matched_key, weight=3.0)

    # --------------------------------------------------------------- scrapelog
    @_locked
    def log_scrape(self, source: str, count: int, duration_ms: float, ok: bool):
        with self._session() as s:
            s.add(ScrapeLog(
                source=source, count=count, duration_ms=duration_ms,
                ok=1 if ok else 0, ts=time.time(),
            ))
            s.commit()

    # ----------------------------------------------------------------- follows
    @_locked
    def follow(self, user_id: str, title_key: str):
        with self._session() as s:
            s.execute(
                sqlite_insert(Follow).values(
                    user_id=str(user_id), title_key=title_key,
                ).on_conflict_do_nothing()
            )
            s.commit()

    @_locked
    def unfollow(self, user_id: str, title_key: str):
        with self._session() as s:
            s.execute(
                delete(Follow).where(
                    Follow.user_id == str(user_id), Follow.title_key == title_key)
            )
            s.commit()

    @_locked
    def followed(self, user_id: str) -> list[str]:
        with self._session() as s:
            rows = s.execute(
                select(Follow.title_key).where(Follow.user_id == str(user_id))
            ).scalars().all()
            return list(rows)

    # ------------------------------------------------------------- broadcasts
    @_locked
    def set_broadcast(self, user_id: str, channel_id: int = 0):
        """Enable auto-news for a user; (re)targets the guild channel."""
        now = time.time()
        with self._session() as s:
            existing = s.get(Broadcast, str(user_id))
            if existing:
                existing.channel_id = channel_id
                if not existing.last_sent:
                    existing.last_sent = now
            else:
                s.add(Broadcast(
                    user_id=str(user_id), channel_id=channel_id,
                    created_at=now, last_sent=now,
                ))
            s.commit()

    @_locked
    def stop_broadcast(self, user_id: str) -> bool:
        with self._session() as s:
            obj = s.get(Broadcast, str(user_id))
            if not obj:
                return False
            s.delete(obj)
            s.commit()
            return True

    @_locked
    def all_broadcasts(self) -> list[dict]:
        with self._session() as s:
            rows = s.execute(select(Broadcast)).scalars().all()
            return [_obj_dict(o) for o in rows]

    @_locked
    def touch_broadcast(self, user_id: str, ts: float):
        with self._session() as s:
            obj = s.get(Broadcast, str(user_id))
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

    # ------------------------------------------------------------------ stats
    @_locked
    def stats(self) -> dict:
        with self._session() as s:
            items = s.execute(select(func.count(Item.id))).scalar() or 0
            titles = s.execute(select(func.count(Title.key))).scalar() or 0
            resources = s.execute(select(func.count(Resource.slug))).scalar() or 0
            total_fb = s.execute(select(func.count(Feedback.id))).scalar() or 0
            good = s.execute(
                select(func.coalesce(func.sum(Feedback.correct), 0))
            ).scalar() or 0
            avg_conf = s.execute(
                select(func.coalesce(func.avg(Title.confidence), 0.0))
            ).scalar() or 0.0
        accuracy = (good / total_fb * 100) if total_fb else 0.0
        # proficiency blends experience (titles seen) with human-verified accuracy
        proficiency = round(min(100.0, avg_conf * 0.5 + accuracy * 0.5), 1)
        return {
            "items": items, "titles": titles, "resources": resources,
            "feedback": total_fb,
            "accuracy": round(accuracy, 1), "avg_confidence": round(avg_conf, 1),
            "proficiency": proficiency,
        }

    @_locked
    def close(self):
        self.engine.dispose()


if __name__ == "__main__":
    db = KnowledgeDB()
    print("stats:", db.stats())
    db.close()