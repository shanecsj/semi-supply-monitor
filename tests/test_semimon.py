"""Offline test suite. No network, no API key.

Live-network acceptance checks live in `python -m semimon.cli verify` instead,
so this suite stays fast and deterministic.
"""

from __future__ import annotations

import os

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


# --------------------------------------------------------------------- dotenv

def test_dotenv_parses_shapes(tmp_path, monkeypatch):
    from semimon.dotenv import load_dotenv
    (tmp_path / ".env").write_text(
        "# comment\n"
        "\n"
        "PLAIN=value\n"
        "export EXPORTED=exported\n"
        'QUOTED="has spaces"\n'
        "SINGLE='single'\n"
        "TRAILING=value # inline comment\n",
        encoding="utf-8")
    for key in ("PLAIN", "EXPORTED", "QUOTED", "SINGLE", "TRAILING"):
        monkeypatch.delenv(key, raising=False)
    applied = load_dotenv(tmp_path / ".env")
    assert applied["PLAIN"] == "value"
    assert applied["EXPORTED"] == "exported"
    assert applied["QUOTED"] == "has spaces"
    assert applied["SINGLE"] == "single"
    assert applied["TRAILING"] == "value"


def test_dotenv_skips_unfilled_placeholder(tmp_path, monkeypatch):
    """An untouched .env must not set the key to the literal placeholder,
    which would produce a confusing 401 instead of 'no credentials'."""
    from semimon.dotenv import load_dotenv
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        "OPENCODE_API_KEY=<paste-your-opencode-go-key-here>\n", encoding="utf-8")
    assert load_dotenv(tmp_path / ".env") == {}
    assert "OPENCODE_API_KEY" not in os.environ


def test_real_environment_wins_over_dotenv(tmp_path, monkeypatch):
    from semimon.dotenv import load_dotenv
    monkeypatch.setenv("SEMIMON_CONTACT", "real@example.com")
    (tmp_path / ".env").write_text("SEMIMON_CONTACT=file@example.com\n",
                                   encoding="utf-8")
    load_dotenv(tmp_path / ".env")
    assert os.environ["SEMIMON_CONTACT"] == "real@example.com"


def test_missing_dotenv_is_not_an_error(tmp_path):
    from semimon.dotenv import load_dotenv
    assert load_dotenv(tmp_path / "nope.env") == {}


def test_rfc822_alphabetic_timezone_parses():
    """Regression: DigiTimes stamps "GMT", not "+0000". strptime's %z rejects
    alphabetic zones, so every date from the corpus's highest-volume source was
    silently dropped - degrading clustering, market annotation, and the dates
    shown to the chat model."""
    from semimon.cluster import _parsed_time
    got = _parsed_time("Thu, 27 Aug 2026 04:21:01 GMT")
    assert got is not None and got.year == 2026 and got.day == 27


@pytest.mark.parametrize("stamp", [
    "Tue, 25 Aug 2026 07:00:00 +0000",
    "Wed, 26 Aug 2026 00:54:37 +0200",
    "2026-08-25",
    "2026-08-22T17:00:39+00:00",
])
def test_other_feed_date_formats_still_parse(stamp):
    from semimon.cluster import _parsed_time
    assert _parsed_time(stamp) is not None


# ---------------------------------------------------------------- latest brief

def test_latest_brief_buckets_consumer_nodes_as_gpu(registry):
    """Regression: NVIDIA supplies nothing, so stages_for() is empty for it and
    every Nvidia story fell into 'Other' instead of GPU."""
    from semimon.chat import latest_brief
    docs = [{"title": "Nvidia caps growth on supply", "source": "rss:x",
             "published_at": "Wed, 26 Aug 2026 12:00:00 GMT",
             "payload": {"node_ids": ["nvidia"]}}]
    out = latest_brief(registry, docs)
    assert "GPU:" in out and "Nvidia caps growth" in out


def test_latest_brief_separates_ram_gpu_and_shipping(registry):
    from semimon.chat import latest_brief
    docs = [
        {"title": "Micron HBM update", "source": "rss:x",
         "published_at": "Wed, 26 Aug 2026 12:00:00 GMT",
         "payload": {"node_ids": ["micron"]}},
        {"title": "Kaohsiung port strike", "source": "rss:x",
         "published_at": "Wed, 26 Aug 2026 11:00:00 GMT",
         "payload": {"node_ids": ["port_kaohsiung"]}},
    ]
    out = latest_brief(registry, docs)
    assert "RAM:" in out and "Shipping:" in out
    assert out.index("RAM:") < out.index("Shipping:")


def test_latest_brief_needs_no_model_or_network(registry):
    """The opener must be purely local - that is the whole reason it exists."""
    from semimon.chat import latest_brief
    assert "No documents" in latest_brief(registry, [])


def test_gdelt_disabled_by_default():
    """GDELT failed on every run and each failure costs ~84s of timeouts."""
    from semimon.sensors.narrative import collect_gdelt, load_sources
    assert load_sources()["gdelt"].get("enabled") is False
    assert collect_gdelt(load_sources()) == []


# --------------------------------------------------------------- answer cache

class _CountingBackend:
    model = "fake-model"

    def __init__(self):
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        return f"answer {self.calls}"


def _session(tmp_path, registry, backend, use_cache=True):
    from semimon import db as store
    from semimon.chat import ChatSession
    db = tmp_path / "t.db"
    with store.connect(db) as conn:
        store.store_documents(conn, [{
            "source": "rss:x", "url": "https://e.com/1", "title": "Micron HBM news",
            "body": "supply tight", "published_at": "Wed, 26 Aug 2026 12:00:00 GMT",
            "payload": {"node_ids": ["micron"]},
        }])
    return ChatSession(registry, db, backend=backend, use_cache=use_cache)


def test_second_identical_question_is_served_from_cache(tmp_path, registry):
    backend = _CountingBackend()
    session = _session(tmp_path, registry, backend)
    first, _ = session.ask("what is happening with HBM?")
    second, _ = session.ask("what is happening with HBM?")
    assert first == second
    assert backend.calls == 1        # the model was asked exactly once


def test_cache_key_is_case_and_whitespace_insensitive(tmp_path, registry):
    backend = _CountingBackend()
    session = _session(tmp_path, registry, backend)
    session.ask("What is happening with HBM?")
    session.ask("  what is HAPPENING with hbm?  ")
    assert backend.calls == 1


def test_new_documents_invalidate_cached_answers(tmp_path, registry):
    """A cached answer is only valid for the corpus it was built on - otherwise
    yesterday's answer gets replayed as though it were current."""
    from semimon import db as store
    backend = _CountingBackend()
    session = _session(tmp_path, registry, backend)
    session.ask("what is happening with HBM?")

    # Collect a genuinely new document, exactly as a refresh would.
    with store.connect(session.db_path) as conn:
        store.store_documents(conn, [{
            "source": "rss:x", "url": "https://e.com/2",
            "title": "SK Hynix Icheon output update", "body": "new news",
            "published_at": "Thu, 27 Aug 2026 09:00:00 GMT",
            "payload": {"node_ids": ["sk_hynix"]},
        }])
    session.retriever.load()

    session.ask("what is happening with HBM?")
    assert backend.calls == 2


def test_backend_errors_are_never_cached(tmp_path, registry):
    backend = _CountingBackend()
    session = _session(tmp_path, registry, backend)
    session._cache_put("q", "[backend error] HTTP 500")
    assert session._cache_get("q") is None


def test_no_cache_flag_disables_caching(tmp_path, registry):
    backend = _CountingBackend()
    session = _session(tmp_path, registry, backend, use_cache=False)
    session.ask("what is happening with HBM?")
    session.ask("what is happening with HBM?")
    assert backend.calls == 2
