"""Narrative sensors - GDELT and trade-press RSS.

This is where the noise lives. A GDELT query for "memory chip" returns video
game walkthroughs alongside Micron guidance. Two defences, in order:

1. **Registry prefilter** (here, free) - a document that resolves to no node in
   the supply chain never reaches the language model at all.
2. **LLM relevance classification** (later, costs money) - only for documents
   that survived step 1.

Doing it in that order is what keeps the running cost near zero.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import yaml
from lxml import etree

from ..registry import Registry
from .base import BROWSER_UA, fetch, fetch_json, qs, safe

SOURCES = Path(__file__).resolve().parent.parent.parent / "config" / "sources.yaml"

# Keywords that make a document plausibly about supply rather than, say, a
# product review or a video game. Applied alongside entity resolution.
SUPPLY_TERMS = [
    "shortage", "supply", "disruption", "capacity", "production", "output",
    "fab", "foundry", "packaging", "substrate", "wafer", "yield", "allocation",
    "lead time", "price", "prices", "pricing", "export", "tariff", "sanction",
    "earthquake", "fire", "outage", "strike", "typhoon", "flood", "contamination",
    "shipment", "shipping", "freight", "cargo", "port", "customs", "quake",
    "halt", "suspend", "shut", "delay", "constraint", "sold out", "ramp",
]


def load_sources(path: Path | str = SOURCES) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def looks_supply_related(text: str) -> bool:
    lowered = (text or "").lower()
    return any(term in lowered for term in SUPPLY_TERMS)


def prefilter(docs: Iterable[dict], registry: Registry) -> list[dict]:
    """Drop documents that mention no supply-chain entity, or that mention one
    only incidentally with no supply language.

    Annotates survivors with resolved `node_ids` so downstream stages do not
    repeat the work.
    """
    kept = []
    for doc in docs:
        blob = f"{doc.get('title', '')} {doc.get('body', '')}"
        node_ids = registry.resolve(blob)
        if not node_ids:
            continue
        if not looks_supply_related(blob):
            continue
        payload = dict(doc.get("payload") or {})
        payload["node_ids"] = node_ids
        doc["payload"] = payload
        kept.append(doc)
    return kept


# ------------------------------------------------------------------ GDELT

def gdelt(query: str, timespan: str = "7d", maxrecords: int = 250,
          name: str = "gdelt") -> list[dict]:
    url = qs("https://api.gdeltproject.org/api/v2/doc/doc", {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": maxrecords,
        "timespan": timespan,
        "sort": "datedesc",
    })
    raw = fetch(url, headers=BROWSER_UA).decode("utf-8", "replace").strip()
    if not raw:
        return []
    import json
    try:
        data = json.loads(raw)
    except ValueError:
        # GDELT answers malformed queries with an HTML error page, not a 4xx.
        return []

    docs = []
    for article in data.get("articles", []):
        seen = article.get("seendate", "")
        published = None
        if len(seen) >= 15:
            try:
                published = datetime.strptime(seen, "%Y%m%dT%H%M%SZ").replace(
                    tzinfo=timezone.utc).isoformat(timespec="seconds")
            except ValueError:
                published = None
        docs.append({
            "source": f"gdelt:{name}",
            "source_type": "narrative",
            "url": article.get("url"),
            "title": (article.get("title") or "").strip(),
            "body": "",           # GDELT artlist gives no body; headline only
            "published_at": published,
            "payload": {"domain": article.get("domain"),
                        "country": article.get("sourcecountry"),
                        "query": name},
        })
    return docs


def collect_gdelt(config: dict) -> list[dict]:
    section = config.get("gdelt", {})
    docs: list[dict] = []
    for entry in section.get("queries", []):
        docs += safe(f"gdelt:{entry['name']}", gdelt, entry["query"],
                     timespan=section.get("timespan", "7d"),
                     maxrecords=section.get("maxrecords", 250),
                     name=entry["name"])
    return docs


# ------------------------------------------------------------------ RSS

def _text(element, *paths: str) -> str:
    for path in paths:
        found = element.find(path)
        if found is not None:
            if found.text:
                return found.text.strip()
            href = found.get("href")
            if href:
                return href.strip()
    return ""


def rss(url: str, name: str, paywalled: bool = False) -> list[dict]:
    """Parse RSS 2.0 or Atom with lxml. No feedparser dependency.

    For paywalled sources we keep the headline and link only - never the body,
    even when the feed volunteers a full-text description.
    """
    raw = fetch(url, headers=BROWSER_UA)
    parser = etree.XMLParser(recover=True, resolve_entities=False, no_network=True)
    root = etree.fromstring(raw, parser=parser)
    if root is None:
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    items = root.findall(".//item") or root.findall(".//atom:entry", ns)

    docs = []
    for item in items:
        title = _text(item, "title", "atom:title") or _text(item, "{*}title")
        link = _text(item, "link", "atom:link") or _text(item, "{*}link")
        if not title:
            continue
        body = ""
        if not paywalled:
            body = (_text(item, "description", "atom:summary")
                    or _text(item, "{*}summary"))[:1200]
            body = re.sub(r"<[^>]+>", " ", body)
            body = re.sub(r"\s+", " ", body).strip()
        published = (_text(item, "pubDate", "atom:published")
                     or _text(item, "{*}updated") or None)
        docs.append({
            "source": f"rss:{name}",
            "source_type": "narrative",
            "url": link or None,
            "title": title,
            "body": body,
            "published_at": published,
            "payload": {"feed": name, "paywalled": paywalled},
        })
    return docs


def collect_rss(config: dict) -> list[dict]:
    docs: list[dict] = []
    for feed in config.get("rss", []):
        docs += safe(f"rss:{feed['name']}", rss, feed["url"], feed["name"],
                     feed.get("paywalled", False))
    return docs


def collect_narrative(registry: Registry, config: Optional[dict] = None) -> list[dict]:
    config = config or load_sources()
    raw = collect_gdelt(config) + collect_rss(config)
    kept = prefilter(raw, registry)
    print(f"  narrative: {len(raw)} fetched -> {len(kept)} passed registry prefilter")
    return kept
