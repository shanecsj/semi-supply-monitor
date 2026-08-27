"""Layer 0 - the entity registry and dependency graph.

This is the spine of the monitor. Two jobs:

1. **Entity resolution** - given a chunk of free text (a headline, a filing, a
   press release), find which nodes in the supply chain it is talking about.
2. **Propagation** - given a node, walk the dependency graph downstream and
   produce the sentence that explains *why an event there matters*, with
   cumulative lag. That sentence is the whole point of the digest.

Deliberately dependency-light: pyyaml and the standard library.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Optional

import yaml

DEFAULT_REGISTRY = Path(__file__).resolve().parent.parent / "config" / "entities.yaml"

# Aliases shorter than this are only matched as standalone capitalised tokens,
# because "TOK", "ASE" and "AMD" collide with ordinary words and abbreviations.
SHORT_ALIAS_LEN = 4


@dataclass
class Node:
    id: str
    type: str
    name: str
    aliases: list[str] = field(default_factory=list)
    country: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    parent: Optional[str] = None
    tickers: list[str] = field(default_factory=list)
    filing_ids: dict = field(default_factory=dict)
    concentration: float = 0.0
    mode: Optional[str] = None
    # stage linkage, by node type
    supplies: list[str] = field(default_factory=list)   # company -> stages
    consumes: list[str] = field(default_factory=list)   # company -> stages
    produces: list[str] = field(default_factory=list)   # site    -> stages
    carries: list[str] = field(default_factory=list)    # route   -> stages

    @property
    def stages(self) -> list[str]:
        """Every stage this node touches on the supply side."""
        return list(dict.fromkeys(self.supplies + self.produces + self.carries))


@dataclass
class Edge:
    src: str
    dst: str
    lag_min: int
    lag_max: int


@dataclass
class PathStep:
    stage: str
    name: str
    cum_lag_min: int
    cum_lag_max: int


class Registry:
    def __init__(self, path: Path | str = DEFAULT_REGISTRY):
        self.path = Path(path)
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8"))

        self.nodes: dict[str, Node] = {}
        for entry in raw["nodes"]:
            node = Node(
                id=entry["id"],
                type=entry["type"],
                name=entry["name"],
                aliases=entry.get("aliases", []),
                country=entry.get("country"),
                lat=entry.get("lat"),
                lon=entry.get("lon"),
                parent=entry.get("parent"),
                tickers=entry.get("tickers", []),
                filing_ids=entry.get("filing_ids", {}) or {},
                concentration=entry.get("concentration", 0.0),
                mode=entry.get("mode"),
                supplies=entry.get("supplies", []),
                consumes=entry.get("consumes", []),
                produces=entry.get("produces", []),
                carries=entry.get("carries", []),
            )
            self.nodes[node.id] = node

        self.edges: list[Edge] = [
            Edge(e["from"], e["to"], e["lag_weeks"][0], e["lag_weeks"][1])
            for e in raw["edges"]
        ]
        self._out: dict[str, list[Edge]] = {}
        self._in: dict[str, list[Edge]] = {}
        for edge in self.edges:
            self._out.setdefault(edge.src, []).append(edge)
            self._in.setdefault(edge.dst, []).append(edge)

        self._alias_patterns = self._build_alias_patterns()
        self._site_locations = self._build_site_locations()
        self._validate()

    # ------------------------------------------------------------------ setup

    def _build_alias_patterns(self) -> list[tuple[re.Pattern, str]]:
        """Longest aliases first, so 'Taiwan Semiconductor Manufacturing' wins
        over a bare 'Taiwan' substring in some other node."""
        pairs: list[tuple[str, str]] = []
        for node in self.nodes.values():
            names = set(node.aliases)
            if node.type in ("site", "route"):
                names.add(node.name)
            for alias in names:
                if alias:
                    pairs.append((alias, node.id))
        pairs.sort(key=lambda p: len(p[0]), reverse=True)

        patterns = []
        for alias, node_id in pairs:
            escaped = re.escape(alias).replace(r"\ ", r"\s+").replace(r"\-", r"[-\s]")
            if len(alias) < SHORT_ALIAS_LEN:
                # Case-sensitive for short acronyms; "AMD" yes, "amd" in a URL no.
                patterns.append((re.compile(rf"\b{escaped}\b"), node_id))
            else:
                patterns.append((re.compile(rf"\b{escaped}\b", re.IGNORECASE), node_id))
        return patterns

    def _build_site_locations(self) -> dict[str, list[str]]:
        """Map bare place names -> site ids.

        Real headlines say "SK Hynix said its Icheon fab", not "SK Hynix Icheon",
        so matching the full site name misses nearly every genuine mention. We
        strip the parent company out of each site name and keep what is left as
        a location token.

        Ambiguity is resolved at match time, not here: "Icheon" is unique so it
        stands alone, while "Yokkaichi" (Kioxia *and* JSR) and "Taichung"
        (Micron *and* TSMC) only bind when the parent company is also named.
        """
        noise = {"fab", "ap", "backend", "cowos", "inc", "plant", "site", "technology"}
        locations: dict[str, list[str]] = {}
        for node in self.by_type("site"):
            text = node.name
            parent = self.nodes.get(node.parent) if node.parent else None
            if parent:
                for label in [parent.name] + parent.aliases:
                    text = re.sub(re.escape(label), " ", text, flags=re.IGNORECASE)
            for token in re.split(r"[^\w\-]+", text):
                token = token.strip("-")
                if len(token) < 4 or token.lower() in noise or token.isdigit():
                    continue
                locations.setdefault(token.lower(), []).append(node.id)
        return locations

    def _validate(self) -> None:
        ids = set(self.nodes)
        for edge in self.edges:
            if edge.src not in ids or edge.dst not in ids:
                raise ValueError(f"edge references unknown node: {edge}")
        for node in self.nodes.values():
            if node.parent and node.parent not in ids:
                raise ValueError(f"{node.id}: unknown parent {node.parent}")
            for stage in node.stages + node.consumes:
                if stage not in ids:
                    raise ValueError(f"{node.id}: unknown stage {stage}")
            if node.type in ("site", "route") and (node.lat is None or node.lon is None):
                raise ValueError(f"{node.id}: {node.type} needs lat/lon")
        if self._find_cycle():
            raise ValueError(f"dependency graph has a cycle: {self._find_cycle()}")

    def _find_cycle(self) -> Optional[list[str]]:
        WHITE, GREY, BLACK = 0, 1, 2
        colour = {n: WHITE for n in self.nodes}

        def visit(node_id: str, trail: list[str]) -> Optional[list[str]]:
            colour[node_id] = GREY
            for edge in self._out.get(node_id, []):
                if colour[edge.dst] == GREY:
                    return trail + [node_id, edge.dst]
                if colour[edge.dst] == WHITE:
                    found = visit(edge.dst, trail + [node_id])
                    if found:
                        return found
            colour[node_id] = BLACK
            return None

        for node_id in self.nodes:
            if colour[node_id] == WHITE:
                found = visit(node_id, [])
                if found:
                    return found
        return None

    # ------------------------------------------------- accessors & resolution

    def __getitem__(self, node_id: str) -> Node:
        return self.nodes[node_id]

    def by_type(self, node_type: str) -> list[Node]:
        return [n for n in self.nodes.values() if n.type == node_type]

    @property
    def located(self) -> list[Node]:
        """Nodes with coordinates - everything the geospatial sensors join against."""
        return [n for n in self.nodes.values() if n.lat is not None]

    def resolve(self, text: str) -> list[str]:
        """Return node ids mentioned in `text`, most specific first.

        A site match implies its parent company, but not the reverse: "Micron
        Taichung" resolves to both the site and Micron, while a bare "Micron"
        resolves only to the company.
        """
        if not text:
            return []
        hits: list[str] = []
        for pattern, node_id in self._alias_patterns:
            if node_id in hits:
                continue
            if pattern.search(text):
                hits.append(node_id)
                parent = self.nodes[node_id].parent
                if parent and parent not in hits:
                    hits.append(parent)

        # Second pass: bare place names. A unique location binds on its own; an
        # ambiguous one binds only when its parent company is already resolved.
        for token, site_ids in self._site_locations.items():
            if not re.search(rf"\b{re.escape(token)}\b", text, re.IGNORECASE):
                continue
            for site_id in site_ids:
                if site_id in hits:
                    continue
                parent = self.nodes[site_id].parent
                if len(site_ids) == 1 or (parent and parent in hits):
                    hits.append(site_id)
                    if parent and parent not in hits:
                        hits.append(parent)
        return hits

    def stages_for(self, node_ids: Iterable[str]) -> list[str]:
        """The supply-side stages touched by a set of resolved nodes."""
        stages: list[str] = []
        for node_id in node_ids:
            node = self.nodes.get(node_id)
            if node is None:
                continue
            if node.type == "stage":
                stages.append(node.id)
            for stage in node.stages:
                if stage not in stages:
                    stages.append(stage)
        return stages

    # ------------------------------------------------------------ propagation

    def downstream(self, stage_id: str, max_depth: int = 6) -> list[list[PathStep]]:
        """All downstream paths from a stage, with cumulative lag ranges."""
        if stage_id not in self.nodes:
            return []
        paths: list[list[PathStep]] = []

        def walk(current: str, trail: list[PathStep], depth: int) -> None:
            outs = self._out.get(current, [])
            if not outs or depth >= max_depth:
                if len(trail) > 1:
                    paths.append(trail)
                return
            for edge in outs:
                step = PathStep(
                    stage=edge.dst,
                    name=self.nodes[edge.dst].name,
                    cum_lag_min=trail[-1].cum_lag_min + edge.lag_min,
                    cum_lag_max=trail[-1].cum_lag_max + edge.lag_max,
                )
                walk(edge.dst, trail + [step], depth + 1)

        root = PathStep(stage_id, self.nodes[stage_id].name, 0, 0)
        walk(stage_id, [root], 0)
        return paths

    def critical_path(self, stage_id: str) -> list[PathStep]:
        """The single most consequential downstream path from a stage.

        Scored by the concentration of the stages it traverses - a path through
        CoWoS (0.95, effectively single-sourced) outranks one through commodity
        OSAT (0.35), because that is where a disruption actually binds.
        """
        paths = self.downstream(stage_id)
        if not paths:
            return []
        return max(
            paths,
            key=lambda p: sum(self.nodes[s.stage].concentration for s in p) / len(p),
        )

    def upstream(self, stage_id: str, max_depth: int = 6) -> list[list[PathStep]]:
        """All inbound paths feeding a stage, ordered source -> stage.

        The mirror of `downstream`. Needed because consumer-side nodes (NVIDIA,
        AMD) supply nothing, so a downstream walk from them is empty and the
        useful question is "what feeds this?" rather than "what does it feed?".
        """
        if stage_id not in self.nodes:
            return []
        paths: list[list[PathStep]] = []

        def walk(current: str, trail: list[PathStep], depth: int) -> None:
            ins = self._in.get(current, [])
            if not ins or depth >= max_depth:
                if len(trail) > 1:
                    # Reverse so the path reads source -> ... -> stage. Each
                    # step's accumulated lag is already its distance to the
                    # target stage, so no rebasing is needed (an earlier attempt
                    # to rebase produced inverted bounds like "87-29wk").
                    paths.append(list(reversed(trail)))
                return
            for edge in ins:
                step = PathStep(
                    stage=edge.src,
                    name=self.nodes[edge.src].name,
                    cum_lag_min=trail[-1].cum_lag_min + edge.lag_min,
                    cum_lag_max=trail[-1].cum_lag_max + edge.lag_max,
                )
                walk(edge.src, trail + [step], depth + 1)

        root = PathStep(stage_id, self.nodes[stage_id].name, 0, 0)
        walk(stage_id, [root], 0)
        return paths

    def critical_upstream(self, stage_id: str) -> list[PathStep]:
        paths = self.upstream(stage_id)
        if not paths:
            return []
        return max(
            paths,
            key=lambda p: sum(self.nodes[s.stage].concentration for s in p) / len(p),
        )

    def explain(self, node_id: str) -> str:
        """One-line propagation sentence, the core of a digest entry.

        e.g. "Shin-Etsu Naoetsu -> Photoresist -> DRAM die (3-8wk) ->
              HBM stack (4-11wk) -> CoWoS (5-13wk) -> GPU module (6-16wk)"
        """
        node = self.nodes.get(node_id)
        if node is None:
            return ""
        stages = self.stages_for([node_id])
        if not stages:
            # Consumer-only node (NVIDIA, AMD): show what feeds it instead.
            if node.consumes:
                path = self.critical_upstream(node.consumes[0])
                if path:
                    rendered = " -> ".join(
                        s.name if s.cum_lag_max == 0
                        else f"{s.name} ({s.cum_lag_min}-{s.cum_lag_max}wk upstream)"
                        for s in path)
                    return f"{rendered} -> {node.name}"
            return node.name
        path = self.critical_path(stages[0])
        if not path:
            return f"{node.name} -> {self.nodes[stages[0]].name}"
        parts = [node.name] if node.type != "stage" else []
        for step in path:
            if step.cum_lag_max == 0:
                parts.append(step.name)
            else:
                parts.append(f"{step.name} ({step.cum_lag_min}-{step.cum_lag_max}wk)")
        return " -> ".join(parts)

    def criticality(self, node_id: str) -> float:
        """0..1 - how much a disruption here binds the chain.

        Concentration of the node itself, lifted toward the tightest chokepoint
        it feeds, so a low-concentration node that is the sole path into CoWoS
        still scores high.
        """
        node = self.nodes.get(node_id)
        if node is None:
            return 0.0
        own = node.concentration
        if node.type != "stage":
            own = max(
                [self.nodes[s].concentration for s in node.stages] or [0.0]
            )
        downstream_peak = 0.0
        for stage in self.stages_for([node_id]):
            for path in self.downstream(stage):
                for step in path:
                    downstream_peak = max(
                        downstream_peak, self.nodes[step.stage].concentration
                    )
        return round(min(1.0, 0.6 * own + 0.4 * downstream_peak), 3)

    # -------------------------------------------------------------- geospatial

    def near(self, lat: float, lon: float, radius_km: float) -> list[tuple[Node, float]]:
        """Located nodes within `radius_km`, nearest first. Drives the USGS and
        weather joins."""
        out = []
        for node in self.located:
            distance = haversine_km(lat, lon, node.lat, node.lon)
            if distance <= radius_km:
                out.append((node, round(distance, 1)))
        return sorted(out, key=lambda pair: pair[1])

    def tickers(self, node_ids: Iterable[str], us_only: bool = True) -> list[str]:
        """Tickers for the market annotation. US-listed only by default, since
        that is what the digest can meaningfully annotate."""
        out: list[str] = []
        for node_id in node_ids:
            node = self.nodes.get(node_id)
            if node is None:
                continue
            candidates = node.tickers
            if node.parent and not candidates:
                candidates = self.nodes[node.parent].tickers
            for ticker in candidates:
                if us_only and ("." in ticker):
                    continue
                if ticker not in out:
                    out.append(ticker)
        return out


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


@lru_cache(maxsize=1)
def load_registry(path: str = str(DEFAULT_REGISTRY)) -> Registry:
    return Registry(path)
