"""JSON API + static file server for the CHOKEPOINT web UI.

Thin translation layer only: every number on the page traces back to
`digest.classify_period()`, `registry.py` graph traversal, or the sensors -
this module does no independent judgement of its own. It shapes that data
into the JSON the frontend (`web/`) renders, and nothing else.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml
from flask import Flask, jsonify, request, send_from_directory

from . import classify as classify_mod
from . import db as store
from . import digest as digest_mod
from .registry import Registry, load_registry

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
SOURCES_CONFIG = Path(__file__).resolve().parent.parent / "config" / "sources.yaml"

# The six sensors the top bar counts: three hard sensors plus GDELT (as one)
# plus the RSS trade-press feed pool plus market annotation.
SENSOR_COUNT = 6


def _source_note(domain: str) -> str:
    """'headline only · paywalled' vs 'full text', from sources.yaml, falling
    back to a generic label when the domain isn't a configured RSS feed."""
    try:
        raw = yaml.safe_load(SOURCES_CONFIG.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return "documents"
    for feed in raw.get("rss", []):
        if feed.get("name") == domain:
            return "headline only · paywalled" if feed.get("paywalled") else "full text"
    if domain == "gdelt":
        return "narrative sensor"
    return "documents"


def _hops(registry: Registry, entity_id: str) -> list[dict]:
    """The propagation path as discrete hops, for the UI's dot-and-line trail.

    Mirrors `Registry.explain()` but returns structured steps instead of one
    sentence, so the frontend can render the graph traversal directly.
    """
    if not entity_id or entity_id not in registry.nodes:
        return []
    node = registry[entity_id]
    stages = registry.stages_for([entity_id])
    if not stages:
        return [{"name": node.name, "lag": "origin"}]

    hops = []
    if node.type != "stage":
        hops.append({"name": node.id, "lag": "origin"})

    path = registry.critical_path(stages[0])
    for step in path:
        concentration = registry[step.stage].concentration
        if step.cum_lag_max == 0:
            # cum_lag 0 means this step is the root stage itself: "origin" if
            # nothing was added yet (the entity *is* a stage), otherwise it is
            # the first stage the origin feeds, shown by how tight it is.
            lag = "origin" if not hops else f"conc. {concentration:.2f}"
        else:
            lag = f"{step.cum_lag_min}-{step.cum_lag_max} wk"
        hops.append({"name": step.stage, "lag": lag})
    return hops


def _entry_to_json(registry: Registry, entry: dict, index: int) -> dict:
    result = entry["classification"]
    commodities = [c for c in result.commodity if c != "neither"]
    if len(commodities) >= 2:
        commodity = "BOTH"
    elif commodities:
        commodity = commodities[0].upper()
    else:
        commodity = "NONE"

    when = entry.get("when")
    date = when.strftime("%Y-%m-%d") if when else ""

    sources = [{"name": name, "note": _source_note(name)}
               for name in entry["sources"]]

    return {
        "id": store.doc_id("digest-entry", entry["label"] or str(index)),
        "severity": result.severity,
        "speculative": result.is_speculative,
        "commodity": commodity,
        "confidence": round(result.confidence, 2),
        "riskType": result.risk_type.replace("_", " ").upper(),
        "date": date,
        "docCount": entry["size"],
        "horizon": result.horizon.upper(),
        "title": entry["label"],
        "pathShort": entry["propagation"] or "",
        "draft": entry["body"],
        "quote": result.evidence_quote,
        "market": entry["market"] or "No abnormal return above threshold, or no tracked ticker involved.",
        "hops": _hops(registry, entry["propagation_entity"]),
        "sources": sources,
        "urls": entry["urls"],
    }


def _classifier_label(offline: bool) -> str:
    """Same decision `get_classifier()` makes, without paying for a second
    Anthropic client just to read its class name."""
    if offline:
        return "heuristic (offline)"
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return "heuristic (no API key)"
    return classify_mod.MODEL


def _chokepoints(registry: Registry) -> list[dict]:
    top = digest_mod.graph_state(registry, top=6)
    peak = top[0][1] if top else 1.0
    return [{"name": name, "concentration": round(score, 2),
             "pct": round(100 * score / peak) if peak else 0}
            for name, score in top]


def build_app(db_path: Path | str = store.DEFAULT_DB) -> Flask:
    app = Flask(__name__, static_folder=None)
    registry_holder: dict[str, Registry] = {}

    def registry() -> Registry:
        if "r" not in registry_holder:
            registry_holder["r"] = load_registry()
        return registry_holder["r"]

    @app.get("/api/digest")
    def api_digest():
        days = int(request.args.get("days", 7))
        offline = request.args.get("offline", "").lower() in ("1", "true", "yes")
        no_market = request.args.get("no_market", "").lower() in ("1", "true", "yes")
        reg = registry()

        started = time.time()
        entries = digest_mod.classify_period(
            reg, db_path, days=days, offline=offline, skip_market=no_market)
        elapsed = round(time.time() - started, 2)

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(
            timespec="seconds")
        with store.connect(db_path) as conn:
            total_docs = conn.execute(
                "SELECT COUNT(*) FROM raw_documents").fetchone()[0]
            docs_in_window = conn.execute(
                "SELECT COUNT(*) FROM raw_documents WHERE fetched_at >= ?",
                (cutoff,)).fetchone()[0]
            last_fetched = conn.execute(
                "SELECT MAX(fetched_at) FROM raw_documents").fetchone()[0]

        open_severe = sum(
            1 for e in entries
            if e["classification"].severity >= digest_mod.ALERT_SEVERITY
            and not e["classification"].is_speculative)

        return jsonify({
            "meta": {
                "days": days,
                "rangeStart": None,
                "rangeEnd": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "classifier": _classifier_label(offline),
                "sensorCount": SENSOR_COUNT,
                "totalDocuments": total_docs,
                "documentsInWindow": docs_in_window,
                "surfacedThisPeriod": len(entries),
                "openSevere": open_severe,
                "nodesTracked": len(reg.nodes),
                "edgesTracked": len(reg.edges),
                "lastPollAt": last_fetched,
                "nextScheduledDate": digest_mod._next_twse_revenue_date(),
                "nextScheduledNote": "Taiwan monthly revenue disclosures - TSMC, ASE, Nanya, Winbond.",
                "buildSeconds": elapsed,
            },
            "chokepoints": _chokepoints(reg),
            "entries": [_entry_to_json(reg, e, i) for i, e in enumerate(entries)],
        })

    @app.post("/api/run")
    def api_run():
        reg = registry()
        days = int(request.json.get("days", 7)) if request.is_json else 7
        digest_mod.collect(reg, db_path, days=days)
        return api_digest()

    @app.get("/api/graph/<node_id>")
    def api_graph(node_id: str):
        reg = registry()
        if node_id not in reg.nodes:
            return jsonify({"error": "unknown node"}), 404
        return jsonify({
            "id": node_id,
            "name": reg[node_id].name,
            "criticality": reg.criticality(node_id),
            "explain": reg.explain(node_id),
            "hops": _hops(reg, node_id),
        })

    @app.get("/")
    def landing_page():
        return send_from_directory(WEB_DIR, "landing.html")

    @app.get("/app")
    @app.get("/app/")
    def app_page():
        return send_from_directory(WEB_DIR, "index.html")

    @app.get("/<path:path>")
    def static_files(path: str):
        target = WEB_DIR / path
        if not target.exists() or target.is_dir():
            return send_from_directory(WEB_DIR, "landing.html")
        return send_from_directory(WEB_DIR, path)

    return app


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="serve the CHOKEPOINT web UI")
    parser.add_argument("--db", default=str(store.DEFAULT_DB))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = build_app(args.db)
    print(f"CHOKEPOINT serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
