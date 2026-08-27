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

```bash
conda create -n semi python=3.12 requests pyyaml pandas numpy scikit-learn lxml -y
conda activate semi
pip install anthropic

export SEMIMON_CONTACT="you@example.com"        # required by SEC EDGAR, else 403
python -m semimon.cli verify                    # acceptance checks, no API key needed
python -m semimon.cli run --out digest.md       # collect + build digest
```

`SEMIMON_CONTACT` goes into the User-Agent SEC requires. It is not hardcoded, so
the repo carries no personal address — set it or the EDGAR sensor will be rejected.

Without `ANTHROPIC_API_KEY` the pipeline still runs end-to-end using a deterministic
heuristic classifier — blunter, explicitly low-confidence, and labelled as such in
the output. Set the key to use the model:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m semimon.cli run --out digest.md
```

## Chat

Ask questions about the news the monitor has collected.

```bash
export OPENCODE_API_KEY=...                     # https://opencode.ai/go, $10/month
python -m semimon.cli chat                      # REPL
python -m semimon.cli chat "what is happening with HBM supply"
python -m semimon.cli chat --list-models        # live model ids, no key needed
```

**This is grounded retrieval, not a chatbot.** The model may only use documents
this system actually collected, must cite them by number, and is instructed to say
what is missing rather than fill a gap from general knowledge. For a supply-chain
tool that constraint is the point: a fluent invention about a fab fire is worse than
no answer, because the reader cannot tell the two apart.

Three things enter the context window — retrieved documents (TF-IDF plus an
entity-overlap boost), **propagation paths from the dependency graph**, and the
question. The graph is what lets it answer *"why would an HBM problem affect Nvidia
GPUs?"* when no single article says so: the answer is `SK Hynix → DRAM die → HBM
stack (1-3wk) → CoWoS (2-5wk) → GPU module`, which is graph traversal, not news.
Consumer-side nodes like NVIDIA and AMD supply nothing downstream, so they get the
*upstream* view of what feeds them.

Backend is [OpenCode Go](https://opencode.ai/go) via its OpenAI-compatible endpoint
(`https://opencode.ai/zen/go/v1/chat/completions`), behind a `ChatBackend` protocol
so the provider is swappable. Default model `glm-5.3`; override with
`SEMIMON_CHAT_MODEL` (e.g. `deepseek-v4-flash` for a higher request quota,
`kimi-k3` or `deepseek-v4-pro` for harder questions). Without a key it degrades to
extractive retrieval — it shows you the matching documents and declines to
synthesise, rather than faking an answer.

Note this is independent of the Anthropic classifier used for the digest; the two
providers do not interact.

### Other commands

```bash
python -m semimon.cli graph hynix_icheon        # propagation path for one node
python -m semimon.cli resolve "SK Hynix Icheon fab halted"
python -m semimon.cli collect --days 14         # poll sensors only
python -m semimon.cli digest --offline          # force heuristic classifier
```

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
semimon/digest.py        pipeline and markdown rendering
semimon/cli.py           command line
tests/                   pytest suite
```
