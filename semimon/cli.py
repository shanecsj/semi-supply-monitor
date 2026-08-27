"""Command line interface.

    python -m semimon.cli collect              # poll every sensor
    python -m semimon.cli digest               # cluster, classify, render
    python -m semimon.cli run                  # collect then digest
    python -m semimon.cli verify               # acceptance checks
    python -m semimon.cli graph <node_id>      # propagation path
    python -m semimon.cli resolve "<text>"     # entity resolution
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import db as store
from . import digest as digest_mod
from .registry import load_registry


def _force_utf8() -> None:
    """Windows consoles default to cp1252 and this tool prints Japanese and
    Taiwanese place names constantly ("Minami-Soma", "Xi'an"). Without this the
    first earthquake near a Japanese fab crashes the process on a UnicodeEncodeError.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def cmd_collect(args) -> int:
    registry = load_registry()
    digest_mod.collect(registry, args.db, days=args.days)
    return 0


def cmd_digest(args) -> int:
    registry = load_registry()
    markdown = digest_mod.build(registry, args.db, days=args.days,
                                offline=args.offline, skip_market=args.no_market)
    if args.out:
        Path(args.out).write_text(markdown, encoding="utf-8")
        print(f"\nwritten to {args.out}")
    else:
        print()
        print(markdown)
    return 0


def cmd_run(args) -> int:
    registry = load_registry()
    digest_mod.collect(registry, args.db, days=args.days)
    return cmd_digest(args)


def cmd_graph(args) -> int:
    registry = load_registry()
    if args.node not in registry.nodes:
        print(f"unknown node: {args.node}")
        print("\ntry one of:")
        for node in list(registry.nodes.values())[:200]:
            print(f"  {node.id:24s} {node.type:8s} {node.name}")
        return 1
    node = registry[args.node]
    print(f"{node.name}  [{node.type}]")
    print(f"  criticality : {registry.criticality(node.id):.3f}")
    print(f"  path        : {registry.explain(node.id)}")
    if node.tickers:
        print(f"  tickers     : {', '.join(node.tickers)}")
    stages = registry.stages_for([node.id])
    if stages:
        print(f"\n  all downstream paths from {registry[stages[0]].name}:")
        for path in registry.downstream(stages[0]):
            trail = " -> ".join(
                f"{s.name}({s.cum_lag_min}-{s.cum_lag_max}w)" if s.cum_lag_max
                else s.name for s in path)
            print(f"    {trail}")
    return 0


def cmd_resolve(args) -> int:
    registry = load_registry()
    hits = registry.resolve(args.text)
    if not hits:
        print("no entities resolved")
        return 0
    for node_id in hits:
        node = registry[node_id]
        print(f"  {node_id:24s} {node.type:8s} {node.name}")
    print(f"\n  stages : {registry.stages_for(hits)}")
    print(f"  tickers: {registry.tickers(hits)}")
    return 0


def cmd_verify(args) -> int:
    """Acceptance checks from the plan. Live network, no API key required."""
    from .sensors import hard

    registry = load_registry()
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    print("registry")
    check("graph loads and is acyclic", len(registry.nodes) > 0)
    check("no site missing coordinates",
          all(n.lat is not None for n in registry.by_type("site")))
    leaves = [n for n in registry.by_type("stage")
              if registry.downstream(n.id) or n.id == "retail_supply"]
    check("every stage has a propagation path or is terminal",
          len(leaves) == len(registry.by_type("stage")),
          f"{len(leaves)}/{len(registry.by_type('stage'))}")

    print("\nentity resolution")
    cases = [
        ("SK Hynix says Icheon fab output unaffected", "hynix_icheon"),
        ("Shin-Etsu halts shipments after Naoetsu fire", "shinetsu_naoetsu"),
        ("Micron confirms Taichung operations resumed", "micron_taichung"),
    ]
    for text, expected in cases:
        check(f"resolves {expected}", expected in registry.resolve(text))
    check("abstains on ambiguous location with no company",
          registry.resolve("Explosion near Taichung industrial park") == [])

    print("\nUSGS - Hualien 2024-04-03 replay")
    try:
        quakes = hard.usgs_quakes(registry, start="2024-04-02", end="2024-04-05",
                                 min_magnitude=7.0)
        big = [q for q in quakes if q["payload"]["magnitude"] >= 7.0]
        nodes = set(big[0]["payload"]["node_ids"]) if big else set()
        check("M7.4 event retrieved", bool(big))
        for expected in ("tsmc_hsinchu", "tsmc_taichung", "micron_taichung"):
            check(f"flags {expected}", expected in nodes)
    except Exception as exc:  # noqa: BLE001
        check("USGS reachable", False, str(exc)[:80])

    print("\nclustering")
    from . import cluster as clustering
    synthetic = [{"title": f"Earthquake halts output at SK Hynix Icheon fab, "
                           f"report {i}", "payload": {"node_ids": ["hynix_icheon"]}}
                 for i in range(20)]
    groups = clustering.cluster_documents(synthetic)
    check("20 paraphrases collapse to 1 cluster", len(groups) == 1,
          f"got {len(groups)}")

    print("\nmarket annotation")
    if args.no_market:
        print("  [SKIP] market (--no-market)")
    else:
        from .market import Market
        value = Market().abnormal_return("MU", "2024-04-03")
        check("MU abnormal return on Hualien day", value is not None,
              f"{value:+.2f}%" if value is not None else "no data")

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("all checks passed")
    return 0


def main(argv=None) -> int:
    _force_utf8()
    parser = argparse.ArgumentParser(prog="semimon",
                                     description="RAM & GPU supply-chain news monitor")
    parser.add_argument("--db", default=str(store.DEFAULT_DB))
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="poll every sensor")
    collect.add_argument("--days", type=int, default=7)
    collect.set_defaults(func=cmd_collect)

    for name, fn in (("digest", cmd_digest), ("run", cmd_run)):
        node = sub.add_parser(name, help="build the digest"
                              if name == "digest" else "collect then digest")
        node.add_argument("--days", type=int, default=7)
        node.add_argument("--offline", action="store_true",
                          help="force the heuristic classifier")
        node.add_argument("--no-market", action="store_true")
        node.add_argument("--out", help="write markdown to a file")
        node.set_defaults(func=fn)

    graph = sub.add_parser("graph", help="show a node's propagation path")
    graph.add_argument("node")
    graph.set_defaults(func=cmd_graph)

    resolve = sub.add_parser("resolve", help="entity-resolve some text")
    resolve.add_argument("text")
    resolve.set_defaults(func=cmd_resolve)

    verify = sub.add_parser("verify", help="run acceptance checks")
    verify.add_argument("--no-market", action="store_true")
    verify.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
