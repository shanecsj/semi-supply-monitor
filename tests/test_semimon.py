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

# ------------------------------------------------------- opencode classifier

VALID = {"relevant": True, "commodity": ["RAM"], "risk_type": "fab_incident",
         "severity": 4, "confidence": 0.8, "is_speculative": False,
         "evidence_quote": "fire halted output", "entities": ["kioxia"],
         "horizon": "2-4 weeks", "summary": "Fire at Kioxia."}


class FakeBackend:
    """Stands in for OpenCode Go so provider logic is testable without a key."""
    model = "glm-5.3"

    def __init__(self, *replies, raises=None):
        self.replies = list(replies)
        self.raises = raises
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        if self.raises:
            raise self.raises
        return self.replies.pop(0)


@pytest.mark.parametrize("wrapper", [
    "{raw}",                                   # bare
    "```json\n{raw}\n```",                     # fenced
    "Here is the classification:\n\n{raw}\n",  # prefaced
])
def test_extract_json_tolerates_model_formatting(wrapper):
    import json
    from semimon.classify import _extract_json
    assert _extract_json(wrapper.format(raw=json.dumps(VALID))) == VALID


def test_extract_json_returns_none_on_prose():
    from semimon.classify import _extract_json
    assert _extract_json("I cannot answer that.") is None


def test_opencode_classifier_parses_fenced_json():
    import json
    from semimon.classify import OpenCodeClassifier
    c = OpenCodeClassifier(backend=FakeBackend(f"```json\n{json.dumps(VALID)}\n```"))
    result = c.classify("Fire at Kioxia Yokkaichi", ["kioxia"])
    assert result.risk_type == "fab_incident" and result.severity == 4


def test_opencode_classifier_retries_once_on_invalid():
    import json
    from semimon.classify import OpenCodeClassifier
    bad = json.dumps({**VALID, "severity": 99})    # out of ge/le bounds
    backend = FakeBackend(bad, json.dumps(VALID))
    c = OpenCodeClassifier(backend=backend)
    assert c.classify("x", ["kioxia"]) is not None
    assert backend.calls == 2


def test_opencode_classifier_gives_up_after_one_retry():
    from semimon.classify import OpenCodeClassifier
    backend = FakeBackend("nope", "still nope")
    assert OpenCodeClassifier(backend=backend).classify("x", ["kioxia"]) is None
    assert backend.calls == 2


def test_auth_failure_trips_breaker_and_falls_back(registry):
    """Regression: a bad key fired one doomed request per cluster, burning 25
    requests of subscription quota to learn the same thing 25 times."""
    from semimon.classify import HeuristicClassifier, OpenCodeClassifier
    backend = FakeBackend(raises=RuntimeError("url: HTTP 401 Invalid API key."))
    c = OpenCodeClassifier(backend=backend, fallback=HeuristicClassifier(registry))
    first = c.classify("Fire at Kioxia Yokkaichi", ["kioxia_yokkaichi"])
    second = c.classify("Quake near Hsinchu. Magnitude 6.5", ["tsmc_hsinchu"])
    assert backend.calls == 1          # breaker tripped; no second API call
    assert first is not None and second is not None   # heuristic still answers


def test_both_providers_share_one_draft_prompt():
    """A change of backend must not silently change the no-advice constraint."""
    from semimon.classify import RiskClassification, _draft_prompt
    prompt = _draft_prompt("text", RiskClassification(**VALID), "A -> B", "", ["x"])
    assert "Do not give trading or investment advice" in prompt
    assert "A -> B" in prompt


# ----------------------------------------------------------------------- chat

def test_upstream_path_for_consumer_only_node(registry):
    """NVIDIA supplies nothing, so a downstream walk is empty. The useful view
    is what feeds it."""
    path = registry.explain("nvidia")
    assert "->" in path and path.endswith("NVIDIA")
    assert "CoWoS" in path


def test_upstream_lag_bounds_are_ordered(registry):
    """Regression: rebasing cumulative lags produced inverted bounds ("87-29wk")."""
    path = registry.critical_upstream("gpu_module")
    assert path
    assert all(s.cum_lag_min <= s.cum_lag_max for s in path)
    # Lag shrinks as the path approaches the target stage.
    assert [s.cum_lag_max for s in path] == sorted(
        [s.cum_lag_max for s in path], reverse=True)


def test_downstream_unaffected_by_upstream_support(registry):
    assert "HBM stack" in registry.explain("hynix_icheon")


def test_build_context_includes_documents_and_paths(registry):
    from semimon.chat import Retrieved, build_context
    hits = [Retrieved(1, "SK Hynix Icheon fab halted", "body", "rss:x",
                      "https://e.com", "Thu, 27 Aug 2026 10:00:00 +0000", 0.9)]
    context = build_context(registry, "what happened at Hynix?", hits)
    assert "[1] SK Hynix Icheon fab halted" in context
    assert "(2026-08-27)" in context          # RFC-822 parsed, not sliced
    assert "PROPAGATION PATHS" in context
    assert "QUESTION" in context


def test_extractive_backend_does_not_invent(registry):
    """Without a key the fallback must return retrieval, not fluent prose."""
    from semimon.chat import ExtractiveBackend
    out = ExtractiveBackend().complete(
        [{"role": "user", "content": "DOCUMENTS\n\n[1] Kioxia halts output\n"}])
    assert "Kioxia halts output" in out
    assert "retrieval only" in out


def test_system_prompt_forbids_ungrounded_answers():
    from semimon.chat import SYSTEM_PROMPT
    lowered = SYSTEM_PROMPT.lower()
    assert "only from" in lowered
    assert "cite" in lowered
    assert "investment advice" in lowered


def test_prefilter_drops_unrelated(registry):
    from semimon.sensors.narrative import prefilter
    docs = [
        {"title": "All memory chip locations in Beast of Reincarnation", "body": ""},
        {"title": "Micron halts Taichung output after quake", "body": ""},
    ]
    kept = prefilter(docs, registry)
    assert len(kept) == 1
    assert "Micron" in kept[0]["title"]
