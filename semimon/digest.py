"""Collection run and digest rendering.

The digest is the default surface, not event alerts. That is a deliberate
choice: a synthesis-and-slow-burn monitor genuinely has quiet weeks, and a tool
that goes silent for five weeks reads as broken even when it is working
correctly. So the digest always carries content - graph state, chokepoint
pressure, scheduled data releases - and event alerts are the exception that
interrupts it.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Sequence

from . import cluster as clustering
from . import db as store
from .classify import RiskClassification, get_classifier
from .market import Market
from .registry import Registry, load_registry
from .sensors.hard import collect_hard
from .sensors.narrative import collect_narrative

# A cluster at or above this severity, confirmed rather than speculative,
# interrupts rather than waiting for the weekly digest.
ALERT_SEVERITY = 4


def collect(registry: Registry, db_path: Path | str = store.DEFAULT_DB,
            days: int = 7) -> int:
    """Run every sensor and append new documents. Returns count of new rows."""
    print("collecting...")
    docs = collect_hard(registry, days=days) + collect_narrative(registry)
    with store.connect(db_path) as conn:
        new = store.store_documents(conn, docs)
        # Hard-sensor documents arrive pre-resolved, so record them as events
        # immediately - they need no language model to be meaningful.
        for doc in docs:
            if doc.get("source_type") != "hard_sensor":
                continue
            node_ids = (doc.get("payload") or {}).get("node_ids") or []
            if not node_ids:
                continue
            identifier = store.doc_id(doc["source"], doc.get("url") or doc["title"])
            criticality = max(
                [registry.criticality(n) for n in node_ids] or [0.0])
            store.store_event(
                conn, identifier, doc["source"], doc.get("published_at"),
                node_ids, registry.stages_for(node_ids), criticality,
                doc.get("title", ""),
            )
    print(f"  {len(docs)} fetched, {new} new")
    return new


def _next_twse_revenue_date(today: Optional[datetime] = None) -> str:
    """Taiwan requires monthly revenue disclosure by the 10th. A scheduled,
    free hard-data release worth knowing is coming."""
    today = today or datetime.now(timezone.utc)
    if today.day < 10:
        target = today.replace(day=10)
    else:
        nxt = (today.replace(day=28) + timedelta(days=7))
        target = nxt.replace(day=10)
    return target.strftime("%Y-%m-%d")


def _graph_state(registry: Registry, top: int = 6) -> list[tuple[str, float]]:
    """Tightest chokepoints, so the reader keeps a mental model of where the
    chain actually binds."""
    scored = [(registry[n].name, registry[n].concentration)
              for n in registry.nodes
              if registry[n].type == "stage"]
    return sorted(scored, key=lambda pair: pair[1], reverse=True)[:top]


def build(registry: Registry, db_path: Path | str = store.DEFAULT_DB,
          days: int = 7, offline: bool = False,
          skip_market: bool = False) -> str:
    """Cluster, classify, annotate and render. Returns markdown."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(
        timespec="seconds")
    classifier = get_classifier(registry, force_offline=offline)
    market = None if skip_market else Market()

    with store.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM raw_documents WHERE fetched_at >= ? ORDER BY fetched_at",
            (cutoff,)).fetchall()
        docs = [dict(row) for row in rows]
        for doc in docs:
            doc["payload"] = store.jload(doc.get("payload"), {})

        groups = clustering.cluster_documents(docs)
        print(f"  {len(docs)} documents -> {len(groups)} clusters")

        entries = []
        for indices in groups:
            members = [docs[i] for i in indices]
            text = "\n\n".join(
                f"{m.get('title', '')}\n{m.get('body') or ''}".strip()
                for m in members)[:8000]
            candidates: list[str] = []
            for member in members:
                for node_id in (member.get("payload") or {}).get("node_ids", []):
                    if node_id not in candidates:
                        candidates.append(node_id)

            result: Optional[RiskClassification] = classifier.classify(
                text, candidates)
            if result is None or not result.relevant:
                continue

            propagation = ""
            for node_id in (result.entities or candidates):
                if node_id in registry.nodes:
                    propagation = registry.explain(node_id)
                    if propagation:
                        break

            market_note = ""
            if market is not None:
                # Feed dates arrive as ISO, RFC-822 ("Thu, 27 Aug 2026 ...") and
                # bare YYYY-MM-DD, so normalise rather than slicing the string.
                when = (clustering._parsed_time(members[0].get("published_at"))
                        or clustering._parsed_time(members[0].get("fetched_at")))
                tickers = registry.tickers(result.entities or candidates)
                if when and tickers:
                    market_note = market.annotate(tickers, when.strftime("%Y-%m-%d"))

            body = classifier.draft(text, result, propagation, market_note,
                                    clustering.originating_sources(docs, indices))
            entries.append({
                "label": clustering.cluster_label(docs, indices),
                "classification": result,
                "propagation": propagation,
                "market": market_note,
                "body": body,
                "sources": clustering.originating_sources(docs, indices),
                "urls": [m.get("url") for m in members if m.get("url")][:4],
                "size": len(members),
            })

        entries.sort(
            key=lambda e: (e["classification"].severity,
                           e["classification"].confidence),
            reverse=True)
        markdown = render(registry, entries, days)
        conn.execute(
            "INSERT INTO alerts (headline, body, created_at, kind) VALUES (?,?,?,?)",
            (f"Digest {datetime.now(timezone.utc):%Y-%m-%d}", markdown,
             store.utcnow(), "digest"))
    return markdown


def render(registry: Registry, entries: Sequence[dict], days: int) -> str:
    now = datetime.now(timezone.utc)
    lines = [
        f"# RAM & GPU supply-chain digest",
        f"_{(now - timedelta(days=days)):%Y-%m-%d} to {now:%Y-%m-%d}_",
        "",
    ]

    urgent = [e for e in entries
              if e["classification"].severity >= ALERT_SEVERITY
              and not e["classification"].is_speculative]
    if urgent:
        lines += ["## Alerts", ""]
        for entry in urgent:
            lines += _entry_block(entry)

    routine = [e for e in entries if e not in urgent]
    lines += ["## This period", ""]
    if routine:
        for entry in routine[:12]:
            lines += _entry_block(entry)
    elif not urgent:
        lines += ["Nothing met the relevance threshold this period. "
                  "That is a normal outcome, not a failure - the sources below "
                  "were all polled successfully.", ""]

    lines += ["## Graph state", "",
              "Tightest chokepoints in the tracked chain:", ""]
    for name, concentration in _graph_state(registry):
        bar = "#" * int(round(concentration * 20))
        lines.append(f"- `{concentration:.2f}` {bar:<20} {name}")
    lines += [
        "",
        f"- Next Taiwan monthly revenue disclosures: **{_next_twse_revenue_date()}** "
        "(TSMC, ASE, Nanya, Winbond - free, scheduled hard data)",
        f"- Tracking {len(registry.by_type('company'))} companies, "
        f"{len(registry.by_type('site'))} sites, "
        f"{len(registry.by_type('route'))} routes",
        "",
        "---",
        "_Awareness tool. Not investment advice, and no directional view is "
        "implied by anything above._",
    ]
    return "\n".join(lines)


def _entry_block(entry: dict) -> list[str]:
    result = entry["classification"]
    flags = [f"severity {result.severity}/5",
             "speculative" if result.is_speculative else "confirmed",
             result.risk_type.replace("_", " "),
             "/".join(result.commodity)]
    lines = [f"### {entry['label']}", "",
             f"`{' | '.join(flags)}`  ·  {entry['size']} source(s)", "",
             entry["body"], ""]
    if entry["propagation"]:
        lines += [f"**Path:** {entry['propagation']}", ""]
    if entry["market"]:
        lines += [f"**Market:** {entry['market']}", ""]
    if result.evidence_quote:
        lines += [f"> {result.evidence_quote}", ""]
    if entry["urls"]:
        lines += ["Sources: " + " · ".join(f"<{u}>" for u in entry["urls"]), ""]
    return lines
