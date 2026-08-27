"""SQLite storage.

The raw document store is **append-only and content-hashed**. That is the one
storage decision worth defending: it means the classifier can be changed and
re-run over history, and it means `fetched_at` (not the publisher's claimed
`published_at`, which is frequently wrong or backdated) is the honest record of
when we actually saw something.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "semimon.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_documents (
    id           TEXT PRIMARY KEY,          -- sha256 of (source, url|title)
    source       TEXT NOT NULL,             -- 'usgs', 'edgar', 'gdelt', 'rss:digitimes', ...
    source_type  TEXT NOT NULL,             -- 'hard_sensor' | 'narrative'
    url          TEXT,
    title        TEXT,
    body         TEXT,                      -- headline/summary only for paywalled press
    published_at TEXT,
    fetched_at   TEXT NOT NULL,
    payload      TEXT                       -- source-specific JSON
);
CREATE INDEX IF NOT EXISTS idx_raw_fetched ON raw_documents(fetched_at);
CREATE INDEX IF NOT EXISTS idx_raw_source  ON raw_documents(source);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT NOT NULL REFERENCES raw_documents(id),
    kind        TEXT NOT NULL,              -- 'earthquake','filing','regulation','news',...
    occurred_at TEXT,
    node_ids    TEXT NOT NULL,              -- JSON list of registry node ids
    stages      TEXT NOT NULL,              -- JSON list of stage ids
    criticality REAL DEFAULT 0,
    summary     TEXT,
    UNIQUE(doc_id)
);
CREATE INDEX IF NOT EXISTS idx_events_when ON events(occurred_at);

CREATE TABLE IF NOT EXISTS clusters (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    label      TEXT,
    created_at TEXT NOT NULL,
    doc_ids    TEXT NOT NULL               -- JSON list
);

CREATE TABLE IF NOT EXISTS classifications (
    cluster_id     INTEGER PRIMARY KEY REFERENCES clusters(id),
    relevant       INTEGER NOT NULL,
    commodity      TEXT,                   -- JSON list
    risk_type      TEXT,
    severity       INTEGER,
    confidence     REAL,
    is_speculative INTEGER,
    evidence_quote TEXT,
    entities       TEXT,                   -- JSON list of node ids
    propagation    TEXT,
    horizon        TEXT,
    model          TEXT,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS market_bars (
    ticker TEXT NOT NULL,
    date   TEXT NOT NULL,
    close  REAL,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS alerts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id INTEGER REFERENCES clusters(id),
    headline   TEXT NOT NULL,
    body       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    kind       TEXT NOT NULL DEFAULT 'event'   -- 'event' | 'digest'
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def doc_id(source: str, key: str) -> str:
    return hashlib.sha256(f"{source}\x00{key}".encode("utf-8")).hexdigest()[:32]


@contextmanager
def connect(path: Path | str = DEFAULT_DB) -> Iterator[sqlite3.Connection]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def store_documents(conn: sqlite3.Connection, docs: Iterable[dict]) -> int:
    """Insert documents, ignoring ones already seen. Returns the count of new rows."""
    new = 0
    for doc in docs:
        identifier = doc.get("id") or doc_id(doc["source"], doc.get("url") or doc["title"])
        cur = conn.execute(
            """INSERT OR IGNORE INTO raw_documents
               (id, source, source_type, url, title, body, published_at, fetched_at, payload)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                identifier,
                doc["source"],
                doc.get("source_type", "narrative"),
                doc.get("url"),
                doc.get("title"),
                doc.get("body"),
                doc.get("published_at"),
                doc.get("fetched_at") or utcnow(),
                json.dumps(doc.get("payload", {}), ensure_ascii=False),
            ),
        )
        new += cur.rowcount
    return new


def store_event(conn: sqlite3.Connection, doc_id_: str, kind: str, occurred_at: Optional[str],
                node_ids: list[str], stages: list[str], criticality: float,
                summary: str) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO events
           (doc_id, kind, occurred_at, node_ids, stages, criticality, summary)
           VALUES (?,?,?,?,?,?,?)""",
        (doc_id_, kind, occurred_at, json.dumps(node_ids), json.dumps(stages),
         criticality, summary),
    )


def unclustered_documents(conn: sqlite3.Connection, since: str) -> list[sqlite3.Row]:
    """Documents fetched since `since` that are not yet in any cluster."""
    clustered: set[str] = set()
    for row in conn.execute("SELECT doc_ids FROM clusters"):
        clustered.update(json.loads(row["doc_ids"]))
    rows = conn.execute(
        "SELECT * FROM raw_documents WHERE fetched_at >= ? ORDER BY fetched_at", (since,)
    ).fetchall()
    return [r for r in rows if r["id"] not in clustered]


def jload(value: Any, default=None):
    if not value:
        return default if default is not None else []
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default if default is not None else []
