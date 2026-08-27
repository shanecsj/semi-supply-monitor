"""Offline test suite. No network, no API key.

Live-network acceptance checks live in `python -m semimon.cli verify` instead,
so this suite stays fast and deterministic.
"""

from __future__ import annotations

import pytest

from semimon.classify import HeuristicClassifier, _quake_severity
from semimon.cluster import cluster_documents, originating_sources
from semimon.registry import haversine_km, load_registry


@pytest.fixture(scope="module")
def registry():
    return load_registry()


# ------------------------------------------------------------------ registry

def test_graph_loads_and_is_acyclic(registry):
    assert len(registry.nodes) > 50
    assert registry._find_cycle() is None


def test_every_site_has_coordinates(registry):
    assert all(n.lat is not None and n.lon is not None
               for n in registry.by_type("site"))


def test_sites_resolve_inside_their_country(registry):
    # Rough bounding boxes; catches a transposed lat/lon or a sign error.
    boxes = {
        "TW": (21.5, 25.5, 119.5, 122.5),
        "KR": (33.0, 39.0, 125.0, 130.0),
        "JP": (30.0, 46.0, 128.0, 146.0),
        "US": (24.0, 50.0, -125.0, -66.0),
        "NL": (50.5, 53.7, 3.3, 7.3),
        "CN": (18.0, 54.0, 73.0, 135.0),
    }
    for node in registry.by_type("site"):
        lo_lat, hi_lat, lo_lon, hi_lon = boxes[node.country]
        assert lo_lat <= node.lat <= hi_lat, f"{node.id} latitude"
        assert lo_lon <= node.lon <= hi_lon, f"{node.id} longitude"


def test_haversine_known_distance():
    # Hsinchu to Tainan, ~185km
    assert 170 < haversine_km(24.77, 121.00, 23.10, 120.28) < 200


# -------------------------------------------------------- entity resolution

@pytest.mark.parametrize("text,expected", [
    ("SK Hynix says Icheon fab output unaffected", "hynix_icheon"),
    ("Shin-Etsu halts shipments after Naoetsu fire", "shinetsu_naoetsu"),
    ("Micron confirms Taichung operations resumed", "micron_taichung"),
    ("Kioxia suspends Yokkaichi output", "kioxia_yokkaichi"),
    ("Port of Kaohsiung strike enters second week", "port_kaohsiung"),
])
def test_resolves_sites_from_bare_location(registry, text, expected):
    assert expected in registry.resolve(text)


def test_abstains_on_ambiguous_location(registry):
    # "Taichung" is both Micron and TSMC; with no company named, bind neither.
    assert registry.resolve("Explosion near Taichung industrial park") == []


def test_site_match_implies_parent_but_not_reverse(registry):
    assert "micron" in registry.resolve("Micron Taichung fab")
    assert "micron_taichung" not in registry.resolve("Micron reported earnings")


def test_us_only_ticker_filter(registry):
    tickers = registry.tickers(["sk_hynix", "micron", "nvidia"])
    assert "MU" in tickers and "NVDA" in tickers
    assert not any("." in t for t in tickers)   # 000660.KS excluded


# ------------------------------------------------------------- propagation

def test_propagation_reaches_downstream_with_lag(registry):
    path = registry.explain("shinetsu_naoetsu")
    assert "Photoresist" in path or "Silicon wafer" in path
    assert "wk)" in path


def test_criticality_ranks_cowos_above_commodity_osat(registry):
    assert registry.criticality("cowos") > registry.criticality("osat")


def test_terminal_stage_has_no_downstream(registry):
    assert registry.downstream("retail_supply") == []


# -------------------------------------------------------------- clustering

def test_paraphrases_collapse_to_one_cluster():
    docs = [{"title": f"Earthquake halts output at SK Hynix Icheon fab, report {i}",
             "payload": {"node_ids": ["hynix_icheon"]}} for i in range(20)]
    assert len(cluster_documents(docs)) == 1


def test_different_companies_do_not_merge():
    docs = [
        {"title": "Samsung delays fab expansion in Pyeongtaek",
         "payload": {"node_ids": ["samsung"]}},
        {"title": "TSMC delays fab expansion in Hsinchu",
         "payload": {"node_ids": ["tsmc"]}},
    ]
    assert len(cluster_documents(docs)) == 2


def test_distinct_sources_not_mention_count():
    docs = [{"title": "x", "payload": {"domain": "digitimes.com"}},
            {"title": "x", "payload": {"domain": "digitimes.com"}},
            {"title": "x", "payload": {"domain": "eetimes.com"}}]
    assert originating_sources(docs, [0, 1, 2]) == ["digitimes.com", "eetimes.com"]


# ------------------------------------------------------------ classification

def test_quake_severity_scales_with_magnitude():
    assert _quake_severity("Magnitude 7.4 at depth 40km") == 5
    assert _quake_severity("Magnitude 5.8 at depth 61km") == 2
    assert _quake_severity("no magnitude here") is None


def test_reported_items_is_not_logistics(registry):
    """Regression: an unanchored `port` matched "Reported items:" and
    classified every SEC 8-K as a logistics event."""
    result = HeuristicClassifier(registry).classify(
        "Micron Technology filed a 8-K on 2026-08-26. Reported items: 5.02,9.01.",
        ["micron"])
    assert result.risk_type != "logistics"


def test_real_logistics_still_classifies(registry):
    result = HeuristicClassifier(registry).classify(
        "Port of Kaohsiung strike halts container shipping", ["port_kaohsiung"])
    assert result.risk_type == "logistics"


def test_speculation_flagged(registry):
    classifier = HeuristicClassifier(registry)
    speculative = classifier.classify(
        "Analysts expect DRAM prices could rise next quarter", ["micron"])
    confirmed = classifier.classify(
        "Fire halts production at Kioxia Yokkaichi", ["kioxia_yokkaichi"])
    assert speculative.is_speculative
    assert not confirmed.is_speculative


def test_heuristic_never_claims_high_confidence(registry):
    result = HeuristicClassifier(registry).classify(
        "Fire halts production at Kioxia Yokkaichi", ["kioxia_yokkaichi"])
    assert result.confidence <= 0.5


# ------------------------------------------------------------------ prefilter

def test_prefilter_drops_unrelated(registry):
    from semimon.sensors.narrative import prefilter
    docs = [
        {"title": "All memory chip locations in Beast of Reincarnation", "body": ""},
        {"title": "Micron halts Taichung output after quake", "body": ""},
    ]
    kept = prefilter(docs, registry)
    assert len(kept) == 1
    assert "Micron" in kept[0]["title"]
