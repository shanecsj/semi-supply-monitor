"""Clustering - collapse the same story told by forty outlets into one item.

This is the single biggest cost lever in the system. One earthquake generates
dozens of near-identical headlines; classifying each separately would multiply
the language-model bill by the redundancy factor and produce a digest that
reads like a stutter.

TF-IDF cosine rather than embeddings, on purpose: it needs no API, no model
download and no network, it is deterministic (so the same inputs always give
the same digest), and for near-duplicate headline detection it performs about
as well as anything heavier.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Above this cosine similarity two headlines are treated as the same story.
# Tuned so that paraphrases of one event merge while two genuinely different
# events at the same company stay apart.
SIMILARITY_THRESHOLD = 0.42

# Stories more than this far apart in time are separate even if worded alike -
# "Micron cuts guidance" is a different event in March and in September.
WINDOW_HOURS = 48


class _Union:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)


def _parsed_time(value: Optional[str]) -> Optional[datetime]:
    """Parse the many date formats feeds emit.

    `email.utils.parsedate_to_datetime` handles RFC-822 properly, including
    alphabetic zone names. That matters: DigiTimes stamps "Thu, 27 Aug 2026
    04:21:01 GMT", and strptime's %z accepts only numeric offsets like +0000, so
    a hand-rolled RFC-822 pattern silently dropped every date from the corpus's
    highest-volume source - degrading clustering, market annotation, and the
    dates shown to the chat model.
    """
    if not value:
        return None
    text = value.strip()
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(text)
        if dt is not None:
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        pass
    for parse in (
        lambda t: datetime.fromisoformat(t.replace("Z", "+00:00")),
        lambda t: datetime.strptime(t, "%Y-%m-%d"),
    ):
        try:
            dt = parse(text)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
    return None


def cluster_documents(docs: Sequence[dict],
                      threshold: float = SIMILARITY_THRESHOLD,
                      window_hours: int = WINDOW_HOURS) -> list[list[int]]:
    """Group documents into stories. Returns lists of indices into `docs`.

    Two documents merge when they are textually similar, within the time
    window, and share at least one resolved supply-chain entity. That last
    condition matters: "Samsung delays fab" and "TSMC delays fab" are lexically
    close but are not the same story.
    """
    if not docs:
        return []
    if len(docs) == 1:
        return [[0]]

    texts = [f"{d.get('title', '')} {d.get('body', '') or ''}".strip() for d in docs]
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), min_df=1, sublinear_tf=True,
    )
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        # Every document was stop-words only.
        return [[i] for i in range(len(docs))]

    similarity = cosine_similarity(matrix)
    times = [_parsed_time(d.get("published_at")) or _parsed_time(d.get("fetched_at"))
             for d in docs]
    entities = [set((d.get("payload") or {}).get("node_ids") or []) for d in docs]

    union = _Union(len(docs))
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            if similarity[i][j] < threshold:
                continue
            if entities[i] and entities[j] and not (entities[i] & entities[j]):
                continue
            if times[i] and times[j]:
                if abs((times[i] - times[j]).total_seconds()) > window_hours * 3600:
                    continue
            union.union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(len(docs)):
        groups.setdefault(union.find(i), []).append(i)

    # Biggest stories first - a cluster of twelve outlets is more likely to
    # matter than a lone blog post.
    return sorted(groups.values(), key=len, reverse=True)


def cluster_label(docs: Sequence[dict], indices: Iterable[int]) -> str:
    """Human-readable name for a cluster: the earliest headline in it.

    Earliest rather than most-common, because the first outlet to report is the
    one worth crediting and the later ones are usually rewrites.
    """
    members = list(indices)
    if not members:
        return ""
    ordered = sorted(
        members,
        key=lambda i: (_parsed_time(docs[i].get("published_at"))
                       or datetime.max.replace(tzinfo=timezone.utc)),
    )
    return (docs[ordered[0]].get("title") or "").strip()[:300]


def originating_sources(docs: Sequence[dict], indices: Iterable[int]) -> list[str]:
    """Distinct originating outlets in a cluster.

    Counted distinctly on purpose. Trade press launders one DigiTimes rumour
    through a dozen outlets until repetition looks like corroboration; counting
    mentions would reward that, counting distinct sources does not.
    """
    seen: list[str] = []
    for i in indices:
        domain = ((docs[i].get("payload") or {}).get("domain")
                  or docs[i].get("source") or "")
        if domain and domain not in seen:
            seen.append(domain)
    return seen
