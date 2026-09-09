"""Shared concurrency infra: thread pools, semaphores, tuned aiohttp sessions.

Why this exists
---------------
The bot mixes three kinds of work:

* async I/O (aiohttp fetches, Discord sends) — best with asyncio.gather
* blocking I/O (duckduckgo_search/DDGS, feedparser, BeautifulSoup) — must run
  in threads or the event loop stalls
* CPU-bound fuzzy scoring (rapidfuzz over hundreds of titles) — also threaded
  (rapidfuzz releases the GIL, so threads give real speedup)

Centralising executors + semaphores here keeps every module using the same
bounded parallelism instead of spawning unbounded tasks or hammering the
default executor.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import os

import aiohttp

# ------------------------------------------------------------------ executors
# Blocking web libs (DDGS, feedparser, BS4 page fetches). Sized for I/O wait,
# not CPU — threads mostly sleep on sockets.


def _cpu_count() -> int:
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4


BLOCKING_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=int(os.getenv("ANISAGE_BLOCKING_WORKERS", "16")),
    thread_name_prefix="anisage-blocking",
)
# CPU-bound fuzzy scoring. rapidfuzz releases the GIL so threads scale.
CPU_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=int(os.getenv("ANISAGE_CPU_WORKERS", str(min(16, _cpu_count() * 2)))),
    thread_name_prefix="anisage-cpu",
)
# SQLite is serialized by a single RLock anyway — a small pool avoids piling
# hundreds of queued DB closures onto the default executor.
DB_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=int(os.getenv("ANISAGE_DB_WORKERS", "4")),
    thread_name_prefix="anisage-db",
)

# ----------------------------------------------------------------- semaphores
# Bound outbound fan-out so an exhaustive crawl doesn't open 30 sockets at
# once (429s / timeouts) or blow the DDG rate limit.


def _sema_n(env_key: str, default: int) -> int:
    try:
        n = int(os.getenv(env_key, str(default)))
    except ValueError:
        n = default
    return max(1, n)


class _LoopSemaphore:
    """asyncio.Semaphore that stays correct across event loops.

    Plain module-level Semaphores bind to the first loop that uses them and
    raise "bound to a different event loop" on any other loop (tests,
    restarts, multiple bots in one process). This wrapper keeps one real
    Semaphore per running loop and delegates ``async with`` to it, so call
    sites stay unchanged.
    """

    def __init__(self, env_key: str, default: int):
        self._env_key = env_key
        self._default = default
        self._n = _sema_n(env_key, default)
        self._semas: dict = {}

    @property
    def _value(self) -> int:  # compat: tests read the configured bound
        return self._n

    def _for_loop(self) -> asyncio.Semaphore:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        sema = self._semas.get(loop)
        if sema is None:
            sema = asyncio.Semaphore(self._n)
            self._semas[loop] = sema
        return sema

    async def __aenter__(self):
        return await self._for_loop().__aenter__()

    async def __aexit__(self, *exc):
        return await self._for_loop().__aexit__(*exc)


def _sema(env_key: str, default: int) -> _LoopSemaphore:
    return _LoopSemaphore(env_key, default)


HTTP_SEM = _sema("ANISAGE_HTTP_CONCURRENCY", 24)   # generic aiohttp GETs
RSS_SEM = _sema("ANISAGE_RSS_CONCURRENCY", 10)     # parallel RSS feeds
OG_SEM = _sema("ANISAGE_OG_CONCURRENCY", 8)        # og:image page fetches
DDG_SEM = _sema("ANISAGE_DDG_CONCURRENCY", 6)      # DDGS blocking searches
SEARCH_SEM = _sema("ANISAGE_SEARCH_CONCURRENCY", 12)  # exhaustive title search
NEWS_SEM = _sema("ANISAGE_NEWS_CONCURRENCY", 10)   # per-title news fan-out
DM_SEM = _sema("ANISAGE_DM_CONCURRENCY", 4)        # Discord DM bursts
HOST_SEM = _sema("ANISAGE_HOST_CONCURRENCY", 8)    # host domain probes

# ------------------------------------------------------------- loop helpers

def _running_loop() -> asyncio.AbstractEventLoop:
    """get_running_loop with a fallback for sync contexts (tests, __main__)."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.get_event_loop()


async def run_blocking(fn, *args, **kwargs):
    """Run sync blocking I/O (DDGS, feedparser, BS4) in the blocking pool."""
    loop = _running_loop()
    call = functools.partial(fn, *args, **kwargs)
    return await loop.run_in_executor(BLOCKING_EXECUTOR, call)


async def run_cpu(fn, *args, **kwargs):
    """Run CPU-bound work (rapidfuzz scoring) in the CPU pool."""
    loop = _running_loop()
    call = functools.partial(fn, *args, **kwargs)
    return await loop.run_in_executor(CPU_EXECUTOR, call)


async def run_db(fn, *args, **kwargs):
    """Run a DB callable off the event loop (DB has its own lock)."""
    loop = _running_loop()
    call = functools.partial(fn, *args, **kwargs)
    return await loop.run_in_executor(DB_EXECUTOR, call)


async def gather_capped(coros, limit_sem: asyncio.Semaphore, return_exceptions: bool = True):
    """asyncio.gather with a semaphore cap: ``coros`` is an iterable of coroutines."""
    async def _wrap(coro):
        async with limit_sem:
            return await coro

    return await asyncio.gather(*(_wrap(c) for c in coros),
                                return_exceptions=return_exceptions)


async def map_blocking(fn, items, sem: asyncio.Semaphore | None = None):
    """Apply sync ``fn`` to ``items`` concurrently in the blocking pool."""
    loop = _running_loop()

    async def _one(item):
        if sem is not None:
            async with sem:
                return await loop.run_in_executor(BLOCKING_EXECUTOR, functools.partial(fn, item))
        return await loop.run_in_executor(BLOCKING_EXECUTOR, functools.partial(fn, item))

    return await asyncio.gather(*(_one(it) for it in items), return_exceptions=True)


async def map_cpu(fn, items):
    """Apply sync ``fn`` to ``items`` concurrently in the CPU pool."""
    loop = _running_loop()

    async def _one(item):
        return await loop.run_in_executor(CPU_EXECUTOR, functools.partial(fn, item))

    return await asyncio.gather(*(_one(it) for it in items), return_exceptions=True)


def cpu_map_sync(fn, items, max_workers: int | None = None):
    """Sync threaded map for use *inside* an executor worker (nested fan-out).

    ``fn`` must be thread-safe (rapidfuzz scoring is). Returns list in order,
    with exceptions captured as-is (caller filters).
    """
    items = list(items)
    if not items:
        return []
    if len(items) == 1:
        try:
            return [fn(items[0])]
        except Exception as ex:  # keep ordering contract
            return [ex]
    workers = max_workers or min(len(items), CPU_EXECUTOR._max_workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))


# ------------------------------------------------------------ http session

def make_timeout(total: float = 20.0, connect: float = 8.0, sock_read: float = 12.0) -> aiohttp.ClientTimeout:
    return aiohttp.ClientTimeout(total=total, connect=connect, sock_read=sock_read)


def make_connector(limit: int = 64, limit_per_host: int = 8) -> aiohttp.TCPConnector:
    return aiohttp.TCPConnector(
        limit=limit,
        limit_per_host=limit_per_host,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
    )


def make_session(limit: int = 64, limit_per_host: int = 8,
                 total_timeout: float = 20.0) -> aiohttp.ClientSession:
    """Tuned session: bounded connections, DNS cache, sane timeouts."""
    return aiohttp.ClientSession(
        connector=make_connector(limit, limit_per_host),
        timeout=make_timeout(total=total_timeout),
    )
