# semi-supply-monitor

Tracks supplier and shipping news for **RAM (DRAM/NAND)** and **GPU/accelerator**
supply chains. Classifies risk, explains the propagation path, and drafts an alert.

## What this is, and what it is not

It is an **attention filter**. It exists so you can know what is happening to these
supply chains without reading 500 articles a day.

It is **not** a trading signal generator, and this is a deliberate design decision
rather than a missing feature. During the 2024-04-03 Hualien M7.4 earthquake:

| Ticker | Abnormal return, day 1 vs SOXX | Role |
|---|---|---|
| **MU** | **+4.0%** | memory supplier — rival capacity impaired |
| TSM | +1.0% | the disrupted party (fear → same-day relief) |
| **NVDA** | **−0.8%** (→ −4% day 2) | consumer of TSMC/HBM capacity |

Two lessons are baked into the code. The market repriced inside **one session**, so
there is no speed edge on free daily data. And the *sign* is the opposite of the
naive intuition — a supply disruption is bullish for the disrupted chain's sellers,
because scarcity is pricing power. A monitor that scored "risk" as a magnitude and
implied a direction would have been confidently backwards. So this tool reports
**what happened and how it propagates**, and takes no directional view at all.

## Quickstart

```powershell
cd D:\path	o\semi-supply-monitor
& "path	o\envs\semi\python.exe" -m semimon.cli
```

That is the whole app. No subcommand needed — it refreshes if the corpus is
stale, prints the latest headlines grouped by RAM / GPU / Shipping, and drops
you at a prompt to ask about any of it.

```
  corpus is 9 min old; skipping refresh
  corpus: 28 documents

RAM:
  Aug 27  Kioxia weighs third Kitakami fab as NAND demand rises  (digitimes)
  Aug 27  SK Hynix questions PIM as thermal, packaging limits constrain HBM  (digitimes)
GPU:
  Aug 27  Nvidia reportedly shifts HBM4 mix toward 8-high as memory supply stays tight  (digitimes)

  ask a question about any of this ('exit' to quit, 'latest' to re-list)

>
```

### Speed, honestly

| | |
|---|---|
| Warm start to headlines | **~2s** |
| Cold start (stale corpus, refreshes first) | **~5s** |
| Asking a question (hosted LLM) | **10–25s, and it varies 2–3× run to run** |

The headline view is deliberately **local and model-free** — retrieval, entity
resolution and the dependency graph all run on your machine in milliseconds.
That is why "what changed?" is instant. A hosted LLM call to OpenCode Go costs
10–25 seconds no matter what you do to the prompt (measured across seven models;
the variance is server-side, not prompt-side), so the model is reserved for
questions you actually choose to ask.

Two things got it from ~5 minutes to ~2 seconds:

- **GDELT is off by default.** It failed on every single run from a residential
  IP — SSL handshake timeouts, connection resets, 429s — and each failed query
  burns ~84s in timeouts and backoff. RSS supplies every useful document anyway.
  Set `enabled: true` in `config/sources.yaml` or `SEMIMON_USE_GDELT=1` to try it.
- **Collectors run concurrently and the corpus is cached.** A refresh is ~3s, and
  it only happens when the corpus is older than 20 minutes.

### Commands

```bash
python -m semimon.cli                       # chat (the default)
python -m semimon.cli "what changed in HBM"  # one-shot question
python -m semimon.cli --no-refresh          # never touch the network
python -m semimon.cli --refresh             # force a refresh
python -m semimon.cli collect               # poll sensors only
python -m semimon.cli digest --out d.md     # full classified digest (slow: ~30min)
python -m semimon.cli verify                # health check
python -m semimon.cli graph hynix_icheon    # propagation path for a node
```

`digest` is the original batch report. It classifies every cluster with the LLM,
which at 10–25s per call takes ~30 minutes. It is no longer on the interactive
path.

## Architecture

```
config/entities.yaml ──> registry (75 nodes, 24 edges, propagation paths)
                              │
   ┌──────────────────────────┼──────────────────────────┐
hard sensors            narrative sources          market annotation
(USGS, EDGAR,           (GDELT, trade RSS)         (Yahoo daily bars)
 Federal Register)
   └──────────────────────────┼──────────────────────────┘
                              ▼
             raw_documents (append-only, content-hashed)
                              ▼
             registry prefilter  ← drops ~92% before any LLM call
                              ▼
             TF-IDF clustering (48h window, shared-entity gate)
                              ▼
             LLM classification (structured output) + drafting
                              ▼
                      digest / alerts
```

The **registry prefilter is the cost control**. A document mentioning no supply-chain
entity, or mentioning one with no supply language, never reaches the model. On live
data that is 265 documents in, 14 out.

The **entity graph is the differentiator**. It turns "Shin-Etsu had a fire" into
`Shin-Etsu Naoetsu → Photoresist → Logic wafer (3-8wk) → CoWoS (5-14wk) → GPU
module (6-17wk)`. That sentence is what makes a digest entry worth reading, and it
is pure graph traversal — no model involved.

## Data sources (all free)

| Source | What it catches | Notes |
|---|---|---|
| **USGS** | Quakes near fabs | Magnitude-scaled radius; M7.4 → 188km |
| **SEC EDGAR** | 8-K/6-K material events | Needs a real `User-Agent` or 403s |
| **Federal Register** | Export controls, Entity List | Agency-scoped to BIS/OFAC/USTR |
| **GDELT 2.0** | Global news | Rate-limited; see Known issues |
| **Trade RSS** | DigiTimes, EE Times, Tom's, etc. | Carries most of the narrative load |
| **Yahoo Finance** | Daily bars for annotation | Replaces stooq, which is now bot-gated |

Shipping is tracked as **news and chokepoint events, not telemetry** — RAM and GPUs
move by air (Taoyuan/Incheon → Anchorage, Memphis), and free air-freight telemetry
does not exist. Free ocean data covers the finished-server leg, the least relevant
part of this chain.

## Known issues

- **GDELT is unreliable from a single IP.** Expect intermittent `429`, connection
  resets and SSL handshake timeouts even at 12s spacing between queries. Failures
  are non-fatal — a run degrades to RSS-only rather than dying. If you need GDELT
  consistently, spread queries across a longer schedule rather than one burst.
- **The heuristic classifier is genuinely blunt.** It reports `confidence: 0.35` and
  `horizon: unknown` so a keyless digest never reads as though a model wrote it.
- **`stooq` is dead as a data source** — it now serves a JavaScript proof-of-work
  anti-bot challenge instead of CSV. Yahoo sits behind a `BarProvider` protocol, so
  swapping to Tiingo/Finnhub is one class.

## Legal

- GDELT and US/KR/TW government data: open or public domain.
- **Paywalled trade press: headline, URL and timestamp only.** `rss()` refuses to
  store bodies for feeds marked `paywalled: true`, even when the feed volunteers
  full text.
- Yahoo Finance terms disallow redistribution — fine locally, swap it if hosted.
- Public-data-only means no MNPI exposure.

## Layout

```
config/entities.yaml     registry: companies, fabs, stages, routes, edges
config/sources.yaml      GDELT queries and RSS feeds
semimon/registry.py      graph, entity resolution, propagation
semimon/sensors/         base (throttling), hard, narrative
semimon/cluster.py       TF-IDF story clustering
semimon/classify.py      LLM + heuristic classifiers
semimon/market.py        abnormal-return annotation
semimon/chat.py          grounded RAG chat over the corpus (OpenCode Go)
semimon/dotenv.py        minimal .env loader, no dependency
semimon/digest.py        pipeline and markdown rendering
semimon/cli.py           command line
tests/                   pytest suite
```
