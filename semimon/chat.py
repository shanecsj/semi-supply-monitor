"""Ask questions about the collected news.

Grounded retrieval, not a general chatbot. The model may only use documents this
system actually collected, and must cite them. That constraint is the whole
design: for a supply-chain tool a confident invention about a fab fire is worse
than no answer at all, because the reader has no way to tell the two apart.

Three things go into the context window, in descending order of trust:

1. **Retrieved documents** - TF-IDF over the corpus, boosted by entity overlap.
2. **Graph propagation paths** for entities named in the question. This is what
   lets the chat answer "why does an Ibiden problem matter for GPUs?" when no
   single article says so - the answer is in the dependency graph, not the news.
3. **The question.**

Backend is OpenCode Go (https://opencode.ai/go), a hosted subscription exposing
an OpenAI-compatible endpoint. Kept behind `ChatBackend` so the provider is
swappable; the Anthropic classifier in classify.py is untouched and independent.
"""

from __future__ import annotations

import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import db as store
from .registry import Registry
from .sensors.base import FetchError, post_json

# OpenAI-compatible endpoint. The Anthropic-format one is /v1/messages, and
# /v1/models lists live model ids without authentication.
OPENCODE_BASE = os.environ.get("OPENCODE_BASE_URL", "https://opencode.ai/zen/go/v1")
DEFAULT_MODEL = os.environ.get("SEMIMON_CHAT_MODEL", "glm-5.3")

# How many documents reach the context window. Enough to answer across a few
# stories, small enough that the model cannot hide a fabrication among them.
TOP_K = 12

SYSTEM_PROMPT = """\
You answer questions about RAM (DRAM/NAND) and GPU/accelerator supply chains for \
an analyst using a news monitoring tool.

Hard rules:
- Answer ONLY from the numbered documents and propagation paths provided. They are \
the entire world for this question.
- Cite every factual claim with its document number, like [3]. A sentence stating \
a fact with no citation is a bug.
- If the documents do not answer the question, say so plainly and state what is \
missing. Never fill a gap with general knowledge about the industry, even when \
you are confident it is correct - the user cannot tell your recall from the corpus.
- Distinguish what was reported from what was speculated. "DigiTimes reports X" and \
"an analyst expects X" are different claims.
- Use the propagation paths to explain downstream consequences and timing. They come \
from a curated dependency graph and are more reliable than inference from headlines.
- No investment advice and no directional market calls. This is an awareness tool.
- Be concise. Three short paragraphs at most unless asked for detail.
"""


@dataclass
class Retrieved:
    index: int          # 1-based citation number
    title: str
    body: str
    source: str
    url: Optional[str]
    published_at: Optional[str]
    score: float


class ChatBackend(Protocol):
    def complete(self, messages: list[dict]) -> str: ...


class OpenCodeGo:
    """OpenCode Go - $10/month hosted subscription, OpenAI-compatible.

    Model ids come from GET /v1/models (unauthenticated). Note the ids are bare
    here (`glm-5.3`); the `opencode-go/` prefix seen in OpenCode's own config
    files is for its TUI, not for this endpoint.
    """

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL,
                 base_url: str = OPENCODE_BASE):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def complete(self, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,     # grounded Q&A, not creative writing
            "max_tokens": 1200,
        }
        data = post_json(
            f"{self.base_url}/chat/completions", payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        choices = data.get("choices") or []
        if not choices:
            raise FetchError(f"no choices in response: {str(data)[:200]}")
        return (choices[0].get("message") or {}).get("content", "").strip()

    def stream(self, messages: list[dict]):
        """Yield text deltas. Same request as complete(), streamed."""
        from .sensors.base import post_stream
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1200,
        }
        yield from post_stream(
            f"{self.base_url}/chat/completions", payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    @staticmethod
    def models(base_url: str = OPENCODE_BASE) -> list[str]:
        """Live model ids. This endpoint needs no authentication."""
        from .sensors.base import fetch_json
        listing = fetch_json(f"{base_url.rstrip('/')}/models")
        return [m["id"] for m in listing.get("data", [])]


class ExtractiveBackend:
    """No-key fallback: returns the retrieved passages without synthesis.

    Deliberately not an imitation of an answer. It says what it found and lets
    the reader do the synthesis, rather than producing fluent prose that no
    model actually reasoned about.
    """

    def complete(self, messages: list[dict]) -> str:
        user = messages[-1]["content"]
        docs = re.findall(r"^\[(\d+)\]\s+(.+)$", user, re.MULTILINE)
        if not docs:
            return ("No documents in the corpus matched that question. "
                    "Try `python -m semimon.cli collect` first.")
        lines = ["No OPENCODE_API_KEY set, so this is retrieval only - the most "
                 "relevant items in the corpus, unsynthesised:", ""]
        for number, title in docs[:6]:
            lines.append(f"  [{number}] {title[:160]}")
        return "\n".join(lines)


class Retriever:
    """TF-IDF retrieval with an entity-overlap boost.

    No embedding API on purpose: it keeps the tool free and offline-capable, and
    for a corpus of a few thousand headlines lexical retrieval is competitive.
    The entity boost is what makes it work well here - a question about "Hynix"
    should surface Hynix documents even when the wording shares little else.
    """

    def __init__(self, registry: Registry, db_path: Path | str = store.DEFAULT_DB):
        self.registry = registry
        self.db_path = db_path
        self._docs: list[dict] = []
        self._matrix = None
        self._vectorizer: Optional[TfidfVectorizer] = None

    def load(self, days: Optional[int] = None) -> int:
        from datetime import datetime, timedelta, timezone
        query = "SELECT * FROM raw_documents"
        params: tuple = ()
        if days:
            cutoff = (datetime.now(timezone.utc)
                      - timedelta(days=days)).isoformat(timespec="seconds")
            query += " WHERE fetched_at >= ?"
            params = (cutoff,)
        query += " ORDER BY fetched_at DESC"
        with store.connect(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()
        self._docs = []
        for row in rows:
            doc = dict(row)
            doc["payload"] = store.jload(doc.get("payload"), {})
            self._docs.append(doc)
        if not self._docs:
            return 0
        texts = [f"{d.get('title', '')} {d.get('body') or ''}" for d in self._docs]
        self._vectorizer = TfidfVectorizer(stop_words="english",
                                           ngram_range=(1, 2), sublinear_tf=True)
        self._matrix = self._vectorizer.fit_transform(texts)
        return len(self._docs)

    def search(self, question: str, top_k: int = TOP_K) -> list[Retrieved]:
        if not self._docs or self._vectorizer is None:
            return []
        vector = self._vectorizer.transform([question])
        scores = cosine_similarity(vector, self._matrix)[0]

        # Entity boost: documents about an entity named in the question are
        # relevant even when they share little vocabulary with it.
        asked = set(self.registry.resolve(question))
        if asked:
            for i, doc in enumerate(self._docs):
                overlap = asked & set((doc.get("payload") or {}).get("node_ids", []))
                if overlap:
                    scores[i] += 0.30 + 0.05 * min(len(overlap), 4)

        ranked = sorted(range(len(self._docs)), key=lambda i: scores[i], reverse=True)
        out: list[Retrieved] = []
        for i in ranked[:top_k]:
            if scores[i] <= 0.01:
                continue
            doc = self._docs[i]
            out.append(Retrieved(
                index=len(out) + 1,
                title=(doc.get("title") or "").strip(),
                body=(doc.get("body") or "").strip()[:700],
                source=doc.get("source") or "",
                url=doc.get("url"),
                published_at=doc.get("published_at"),
                score=round(float(scores[i]), 3),
            ))
        return out


def build_context(registry: Registry, question: str,
                  hits: Sequence[Retrieved]) -> str:
    """Assemble the grounded prompt: documents, then graph paths, then question."""
    from .cluster import _parsed_time

    blocks = ["DOCUMENTS", ""]
    for hit in hits:
        # Feed dates arrive as ISO, RFC-822 and bare dates; slicing the raw
        # string renders "Wed, 26 Au". Parse, then format.
        when = _parsed_time(hit.published_at)
        stamp = f" ({when:%Y-%m-%d})" if when else ""
        blocks.append(f"[{hit.index}] {hit.title}{stamp}  -- {hit.source}")
        if hit.body:
            blocks.append(textwrap.indent(textwrap.fill(hit.body, 88), "    "))
        blocks.append("")

    # Graph context for anything named in the question or surfaced by retrieval.
    # Scan every hit, not just the head. The entity that explains a question is
    # often deeper in the ranking than the ones that merely match its wording.
    mentioned: list[str] = list(registry.resolve(question))
    for hit in hits:
        for node_id in registry.resolve(f"{hit.title} {hit.body}"):
            if node_id not in mentioned:
                mentioned.append(node_id)

    paths = []
    for node_id in mentioned[:8]:
        explanation = registry.explain(node_id)
        if explanation and "->" in explanation:
            paths.append(f"  {explanation}")
    if paths:
        blocks += ["PROPAGATION PATHS (from the curated dependency graph)", ""]
        blocks += paths
        blocks.append("")

    blocks += ["QUESTION", "", question]
    return "\n".join(blocks)


def get_backend(force_offline: bool = False) -> ChatBackend:
    key = os.environ.get("OPENCODE_API_KEY")
    if force_offline or not key:
        if not force_offline:
            print("  backend: extractive (no OPENCODE_API_KEY; "
                  "subscribe at https://opencode.ai/go)")
        return ExtractiveBackend()
    print(f"  backend: opencode-go / {DEFAULT_MODEL}")
    return OpenCodeGo(key, model=DEFAULT_MODEL)


# The question a bare `semimon chat` answers before handing over the prompt.
LATEST_QUESTION = ("What is the latest on RAM and GPU supply and shipping? "
                   "At most 4 short bullets, one line each, newest first. "
                   "Cite [n]. No preamble, no headings.")


def latest_brief(registry: Registry, docs: Sequence[dict], limit: int = 8) -> str:
    """The newest headlines, grouped, with zero model involvement.

    This exists because a hosted LLM call to OpenCode Go costs 10-25s and swings
    by 2-3x run to run, which is not a "latest news" experience. Retrieval,
    entity resolution and the dependency graph are all local and instant, and
    they already answer "what changed?" - the model is only needed once you want
    to *ask* something. So the opener is deterministic and the model is on demand.
    """
    from .cluster import _parsed_time

    RAM = {"dram_die", "nand_die", "hbm_stack", "dram_module", "ssd"}
    GPU = {"logic_wafer", "cowos", "gpu_module"}
    SHIP = {"route", }

    dated = []
    for doc in docs:
        when = _parsed_time(doc.get("published_at")) or _parsed_time(
            doc.get("fetched_at"))
        if when:
            dated.append((when, doc))
    dated.sort(key=lambda pair: pair[0], reverse=True)

    buckets: dict[str, list[str]] = {"RAM": [], "GPU": [], "Shipping": [],
                                     "Other": []}
    for when, doc in dated[:limit * 3]:
        node_ids = (doc.get("payload") or {}).get("node_ids") or []
        stages = set(registry.stages_for(node_ids))
        # Consumer-side nodes (NVIDIA, AMD) supply nothing, so stages_for is
        # empty for them and every Nvidia story fell into "Other". Fold in what
        # they consume as well.
        for node_id in node_ids:
            node = registry.nodes.get(node_id)
            if node is not None:
                stages.update(node.consumes)
        is_route = any(registry.nodes[n].type == "route"
                       for n in node_ids if n in registry.nodes)
        key = ("Shipping" if is_route else
               "RAM" if stages & RAM else
               "GPU" if stages & GPU else "Other")
        if len(buckets[key]) >= limit:
            continue
        source = (doc.get("source") or "").replace("rss:", "")
        buckets[key].append(
            f"  {when:%b %d}  {(doc.get('title') or '').strip()[:96]}  ({source})")

    lines = []
    for key in ("RAM", "GPU", "Shipping", "Other"):
        if buckets[key]:
            lines.append(f"{key}:")
            lines += buckets[key]
            lines.append("")
    if not lines:
        return "No documents in the corpus yet."
    return "\n".join(lines).rstrip()

# Corpus older than this triggers a refresh on chat start. Long enough that
# consecutive questions cost no network, short enough that "latest" is honest.
STALE_AFTER_MINUTES = 20


def ensure_fresh(registry: Registry, db_path: Path | str = store.DEFAULT_DB,
                 max_age_minutes: float = STALE_AFTER_MINUTES,
                 force: bool = False) -> None:
    """Refresh the corpus only when it is stale.

    Chat has to feel instant, and a warm corpus needs no network at all. When a
    refresh is needed it runs every source concurrently, which costs ~3s rather
    than the ~5 minutes a sequential run with GDELT enabled used to take.
    """
    from .digest import collect, corpus_age_minutes

    age = corpus_age_minutes(db_path)
    if not force and age is not None and age <= max_age_minutes:
        print(f"  corpus is {age:.0f} min old; skipping refresh")
        return
    if age is None:
        print("  corpus empty; collecting...")
    else:
        print(f"  corpus is {age:.0f} min old; refreshing...")
    collect(registry, db_path, parallel=True)


class ChatSession:
    """Multi-turn session. History is carried, but every turn re-retrieves.

    Re-retrieving per turn rather than once at the start matters: follow-ups
    change the subject ("what about NAND?") and a session pinned to the first
    question's documents would answer confidently from the wrong corpus.
    """

    def __init__(self, registry: Registry, db_path: Path | str = store.DEFAULT_DB,
                 backend: Optional[ChatBackend] = None, days: Optional[int] = None):
        self.registry = registry
        self.retriever = Retriever(registry, db_path)
        self.count = self.retriever.load(days=days)
        self.backend = backend or get_backend()
        self.history: list[dict] = []

    def _messages(self, question: str, hits: Sequence[Retrieved]) -> list[dict]:
        context = build_context(self.registry, question, hits)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += self.history[-6:]          # keep the tail, not the world
        messages.append({"role": "user", "content": context})
        return messages

    def ask(self, question: str) -> tuple[str, list[Retrieved]]:
        hits = self.retriever.search(question)
        try:
            answer = self.backend.complete(self._messages(question, hits))
        except FetchError as exc:
            return f"[backend error] {exc}", hits
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        return answer, hits

    def ask_stream(self, question: str):
        """Yield (chunk, hits) as the answer arrives; hits on the first yield.

        Falls back to a single blocking call when the backend cannot stream, so
        the offline extractive path still works unchanged.
        """
        hits = self.retriever.search(question)
        messages = self._messages(question, hits)
        pieces: list[str] = []
        streamer = getattr(self.backend, "stream", None)
        try:
            if streamer is None:
                text = self.backend.complete(messages)
                pieces.append(text)
                yield text, hits
            else:
                first = True
                for piece in streamer(messages):
                    pieces.append(piece)
                    yield piece, (hits if first else [])
                    first = False
        except FetchError as exc:
            yield f"[backend error] {exc}", hits
            return
        answer = "".join(pieces)
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
