"""Hard sensors - structured, deterministic, no language model involved.

These are the highest-signal-per-byte sources in the system. A magnitude-6.5
quake 40km from Hsinchu is an alert from arithmetic; it does not need to be
read. Everything here returns the same document shape the narrative sensors
produce, so downstream stages do not care where a document came from.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from ..registry import Registry
from .base import fetch_json, qs, safe

# Below this, fabs do not even pause. A flat magnitude floor plus a flat radius
# produced 28 hits in 30 days on live data, almost all of them M4.5-5 events
# 100km from a site - i.e. noise. Relevance depends on magnitude *and* distance,
# so the radius scales with magnitude instead.
QUAKE_MIN_MAG = 5.5
QUAKE_RADIUS_CAP_KM = 350.0
# Global prefilter only; the real gate is felt_radius_km() per event.
QUAKE_QUERY_RADIUS_KM = QUAKE_RADIUS_CAP_KM


def felt_radius_km(magnitude: float) -> float:
    """Distance within which a quake plausibly disrupts fab operations.

    Calibrated against Hualien 2024 (M7.4 -> ~152km, which covers the Hsinchu
    and Taichung clusters that actually evacuated) and against the fact that a
    M5.5 barely registers past ~45km.
    """
    return min(QUAKE_RADIUS_CAP_KM, 30.0 * max(0.0, magnitude - 4.0) ** 1.5)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


# ------------------------------------------------------------------ USGS

def usgs_quakes(registry: Registry, days: int = 7,
                min_magnitude: float = QUAKE_MIN_MAG,
                radius_km: Optional[float] = None,
                start: Optional[str] = None, end: Optional[str] = None) -> list[dict]:
    """Earthquakes near any located node.

    Queried globally then joined locally against the registry rather than making
    one request per site - 30 sites would otherwise mean 30 requests for the
    same data.
    """
    now = datetime.now(timezone.utc)
    params = {
        "format": "geojson",
        "starttime": start or _iso(now - timedelta(days=days)),
        "endtime": end or _iso(now + timedelta(days=1)),
        "minmagnitude": min_magnitude,
        "orderby": "time",
    }
    data = fetch_json(qs("https://earthquake.usgs.gov/fdsnws/event/1/query", params))

    docs = []
    for feature in data.get("features", []):
        props = feature["properties"]
        lon, lat = feature["geometry"]["coordinates"][:2]
        magnitude = props.get("mag")
        if magnitude is None:
            continue
        # Magnitude-scaled unless the caller pins a radius (backtest replays do).
        reach = radius_km if radius_km is not None else felt_radius_km(magnitude)
        nearby = registry.near(lat, lon, reach)
        if not nearby:
            continue
        occurred = datetime.fromtimestamp(props["time"] / 1000, tz=timezone.utc)
        affected = ", ".join(f"{n.name} ({d:.0f}km)" for n, d in nearby[:6])
        docs.append({
            "source": "usgs",
            "source_type": "hard_sensor",
            "url": props.get("url"),
            "title": f"M{props['mag']} earthquake - {props.get('place')}",
            "body": f"Magnitude {props['mag']} at depth "
                    f"{feature['geometry']['coordinates'][2]}km. "
                    f"Supply-chain sites within {reach:.0f}km: {affected}.",
            "published_at": occurred.isoformat(timespec="seconds"),
            "payload": {
                "magnitude": props["mag"],
                "lat": lat, "lon": lon,
                "nearby": [{"node": n.id, "name": n.name, "km": d} for n, d in nearby],
                "node_ids": [n.id for n, _ in nearby],
            },
        })
    return docs


# ------------------------------------------------------------------ SEC EDGAR

# 8-K items that can plausibly signal a supply disruption. Most 8-Ks are
# governance boilerplate; filtering by item code keeps the noise down.
MATERIAL_FORMS = {"8-K", "6-K"}


def edgar_filings(registry: Registry, days: int = 14) -> list[dict]:
    """Recent material filings for registry companies with a SEC CIK.

    8-Ks are legally mandated disclosure of material events, timestamped to the
    minute. When a fab has a fire, this beats the trade press.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    docs = []
    for node in registry.by_type("company"):
        cik = node.filing_ids.get("sec_cik")
        if not cik:
            continue
        url = f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
        data = fetch_json(url)
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        for i, form in enumerate(forms):
            if form not in MATERIAL_FORMS:
                continue
            filed = recent["filingDate"][i]
            if filed < cutoff:
                continue
            accession = recent["accessionNumber"][i].replace("-", "")
            doc_url = (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                       f"{accession}/{recent['primaryDocument'][i]}")
            items = recent.get("items", [""] * len(forms))[i]
            docs.append({
                "source": "edgar",
                "source_type": "hard_sensor",
                "url": doc_url,
                "title": f"{node.name} files {form}"
                         + (f" (items {items})" if items else ""),
                "body": f"{node.name} filed a {form} on {filed}."
                        + (f" Reported items: {items}." if items else ""),
                "published_at": filed,
                "payload": {"node_ids": [node.id], "form": form, "items": items,
                            "cik": cik},
            })
    return docs


# ------------------------------------------------------------- Federal Register

# Everything these agencies publish is potentially in scope, so query by agency
# rather than by keyword. Bare term search matches the full document text and
# returned "Small Business Size Standards" and generic TSCA chemical rules.
TRADE_AGENCIES = [
    "industry-and-security-bureau",       # BIS - the export-control publisher
    "foreign-assets-control-office",      # OFAC
    "trade-representative-office-of-united-states",
]

# Post-filter: an agency-scoped hit still has to look like it concerns this
# supply chain. Checked against title + abstract only, never full text.
RELEVANCE_TERMS = [
    "semiconductor", "integrated circuit", "advanced computing", "wafer",
    "memory", "dram", "nand", "hbm", "lithograph", "entity list",
    "electronic design automation", "gallium", "germanium", "rare earth",
    "export administration regulations", "chip",
]

FR_FIELDS = ["title", "html_url", "publication_date", "abstract", "agencies", "type"]


def _fr_relevant(title: str, abstract: str) -> bool:
    blob = f"{title} {abstract}".lower()
    return any(term in blob for term in RELEVANCE_TERMS)


def federal_register(days: int = 30) -> list[dict]:
    """Export-control and trade rulemaking. Frequently lands here before it is
    reported anywhere.

    Two passes: everything from the trade agencies, plus a keyword sweep across
    all agencies to catch action taken somewhere unexpected. Both are filtered
    on title/abstract relevance.
    """
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    base = "https://www.federalregister.gov/api/v1/documents.json"
    queries = [
        {"conditions[agencies][]": TRADE_AGENCIES},
        {"conditions[term]": "\"semiconductor manufacturing equipment\" OR "
                             "\"advanced computing\" OR \"entity list\""},
    ]

    docs: list[dict] = []
    seen: set[str] = set()
    for extra in queries:
        params = {
            "conditions[publication_date][gte]": since,
            "per_page": 50,
            "order": "newest",
            "fields[]": FR_FIELDS,
        }
        params.update(extra)
        data = fetch_json(qs(base, params))
        for item in data.get("results", []):
            url = item.get("html_url")
            title = item.get("title", "") or ""
            abstract = (item.get("abstract") or "")
            if not url or url in seen or not _fr_relevant(title, abstract):
                continue
            seen.add(url)
            agencies = ", ".join(a.get("name", "") for a in item.get("agencies", []))
            docs.append({
                "source": "federal_register",
                "source_type": "hard_sensor",
                "url": url,
                "title": title[:400],
                "body": abstract[:2000],
                "published_at": item.get("publication_date"),
                "payload": {"agencies": agencies, "type": item.get("type")},
            })
    return docs


# ------------------------------------------------------------------ collection

def collect_hard(registry: Registry, days: int = 7) -> list[dict]:
    docs: list[dict] = []
    docs += safe("usgs", usgs_quakes, registry, days=days)
    docs += safe("edgar", edgar_filings, registry, days=max(days, 14))
    docs += safe("federal_register", federal_register, days=max(days, 30))
    return docs
