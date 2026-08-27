"""Risk classification and alert drafting.

Two language-model jobs, and only two:

1. **Classify a cluster** into a structured risk record. Structured outputs, so
   we get a validated object rather than prose to regex.
2. **Draft the alert**, which is genuinely a writing task.

Everything else in this system is deterministic. The model is a means here, not
the point - it is used where it beats a regex and kept out of the path where it
does not.

`HeuristicClassifier` is a deliberate offline fallback: the whole pipeline runs
and produces a digest with no API key at all, just with blunter judgement. That
keeps the system testable and makes the model's marginal contribution visible
rather than assumed.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Literal, Optional, Protocol, Sequence

from pydantic import BaseModel, Field

from .registry import Registry

MODEL = os.environ.get("SEMIMON_MODEL", "claude-opus-5")

RiskType = Literal[
    "natural_disaster", "fab_incident", "export_control", "capacity_allocation",
    "logistics", "labor", "demand_shock", "pricing", "other",
]


class RiskClassification(BaseModel):
    """The structured record the digest is built from."""

    relevant: bool = Field(
        description="True only if this concerns RAM or GPU supply, production, "
                    "logistics, or the policy environment around them. Product "
                    "reviews, benchmarks and gaming articles are not relevant."
    )
    commodity: List[Literal["RAM", "GPU", "both", "neither"]]
    risk_type: RiskType
    severity: int = Field(ge=1, le=5, description="1 trivial, 5 major disruption")
    confidence: float = Field(ge=0.0, le=1.0)
    is_speculative: bool = Field(
        description="True for analyst forecasts, rumours and expectations. "
                    "False for confirmed events that have already happened."
    )
    evidence_quote: str = Field(
        description="A short verbatim quote from the source supporting the "
                    "classification. Never paraphrase or invent."
    )
    entities: List[str] = Field(
        description="Registry node ids from the supplied candidate list only."
    )
    horizon: str = Field(description="When effects would be felt, e.g. '2-6 weeks'")
    summary: str = Field(description="One neutral sentence describing what happened.")


SYSTEM_PROMPT = """\
You classify news about RAM (DRAM/NAND) and GPU/accelerator supply chains for a \
monitoring tool. The reader wants situational awareness, not trading advice.

Rules:
- Ground every field in the supplied text. If the text does not support a field, \
choose the most conservative value rather than inferring.
- `evidence_quote` must be verbatim from the source. Never fabricate one.
- Distinguish a confirmed event from an expectation. An analyst predicting higher \
DRAM prices is speculative; a fab fire is not.
- Severity reflects supply impact, not how dramatic the headline is. A routine \
capacity expansion announcement is severity 1-2 even when the numbers are large.
- Only use entity ids from the candidate list you are given.
- Be willing to mark things irrelevant. Most articles that mention a chip company \
are not about supply.
"""


class Classifier(Protocol):
    def classify(self, cluster_text: str, candidates: Sequence[str]
                 ) -> Optional[RiskClassification]: ...


def _draft_prompt(cluster_text: str, classification: "RiskClassification",
                  propagation: str, market_note: str,
                  sources: Sequence[str]) -> str:
    """Shared by every provider, so a change of backend cannot quietly change
    the editorial voice or drop the no-advice constraint."""
    stance = "SPECULATIVE" if classification.is_speculative else "CONFIRMED"
    return f"""\
Write a short supply-chain alert for a reader tracking RAM and GPU supply.

What happened: {classification.summary}
Risk type: {classification.risk_type} | Severity {classification.severity}/5 | {stance}
Propagation path: {propagation or 'not established'}
Expected horizon: {classification.horizon}
Market reaction: {market_note or 'none observed'}
Distinct sources: {', '.join(sources[:6])}

Source material:
{cluster_text[:4000]}

Write 3-5 sentences. Lead with what happened and who it affects. Use the \
propagation path to explain why it matters downstream. State plainly if this is \
unconfirmed. Do not give trading or investment advice. Do not speculate beyond \
the source material."""


class LLMClassifier:
    def __init__(self, model: str = MODEL, effort: str = "low"):
        import anthropic  # imported lazily so the offline path needs no SDK
        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort

    def classify(self, cluster_text: str, candidates: Sequence[str]
                 ) -> Optional[RiskClassification]:
        prompt = (
            f"Candidate entity ids: {', '.join(candidates) or '(none)'}\n\n"
            f"Source material:\n{cluster_text[:6000]}"
        )
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_format=RiskClassification,
            output_config={"effort": self.effort},
        )
        # Safety classifiers can decline; stop_reason must be checked before
        # touching content.
        if getattr(response, "stop_reason", None) == "refusal":
            return None
        return response.parsed_output

    def draft(self, cluster_text: str, classification: RiskClassification,
              propagation: str, market_note: str, sources: Sequence[str]) -> str:
        prompt = _draft_prompt(cluster_text, classification, propagation,
                               market_note, sources)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        if getattr(response, "stop_reason", None) == "refusal":
            return classification.summary
        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()


# ---------------------------------------------------------------- opencode go

def _extract_json(text: str) -> Optional[dict]:
    """Pull a JSON object out of a completion.

    Open-weight models fence their JSON, preface it with "Here is the...", or
    both. Anthropic's structured outputs make this unnecessary; the
    OpenAI-compatible path does not guarantee it, so parse defensively rather
    than trusting `response_format` to have been honoured.
    """
    import json

    if not text:
        return None
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        return json.loads(candidate)
    except ValueError:
        pass
    # Fall back to the outermost brace-balanced span.
    start = candidate.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(candidate)):
        if candidate[i] == "{":
            depth += 1
        elif candidate[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(candidate[start:i + 1])
                except ValueError:
                    return None
    return None


class OpenCodeClassifier:
    """Classify via OpenCode Go instead of Anthropic.

    The Anthropic path uses `messages.parse` with a Pydantic output_format, which
    validates server-side. OpenCode Go exposes an OpenAI-compatible endpoint
    whose upstream models (GLM, Kimi, DeepSeek) do not reliably support strict
    JSON-schema mode, so the schema goes in the prompt, the response is parsed
    tolerantly, and Pydantic validates locally - with one corrective retry.
    """

    def __init__(self, backend=None, model: Optional[str] = None,
                 fallback: Optional["HeuristicClassifier"] = None):
        from .chat import DEFAULT_MODEL, OpenCodeGo
        if backend is None:
            key = os.environ.get("OPENCODE_API_KEY")
            if not key:
                raise RuntimeError("OPENCODE_API_KEY is not set")
            backend = OpenCodeGo(key, model=model or DEFAULT_MODEL)
        self.backend = backend
        self.model = getattr(backend, "model", model or "opencode-go")
        self.fallback = fallback
        self._disabled = False

    def _trip(self, exc: Exception) -> bool:
        """Stop calling the API after an unrecoverable error.

        A bad key fails identically for every cluster, so without this a digest
        run fires one doomed request per cluster - 25 requests charged against
        the subscription rate limit to learn the same thing 25 times.
        """
        text = str(exc)
        if "HTTP 401" in text or "HTTP 403" in text:
            self._disabled = True
            print(f"  [warn] OpenCode auth rejected; "
                  f"{'using heuristic for the rest of this run' if self.fallback else 'skipping remaining clusters'}")
            return True
        return False

    def _schema_prompt(self, cluster_text: str, candidates: Sequence[str]) -> str:
        import json
        schema = RiskClassification.model_json_schema()
        return (
            f"Candidate entity ids: {', '.join(candidates) or '(none)'}\n\n"
            f"Source material:\n{cluster_text[:6000]}\n\n"
            f"Return a single JSON object matching this schema, and nothing else "
            f"- no prose, no code fence:\n{json.dumps(schema, indent=1)[:2500]}"
        )

    def classify(self, cluster_text: str, candidates: Sequence[str]
                 ) -> Optional[RiskClassification]:
        if self._disabled:
            return self.fallback.classify(cluster_text, candidates) if self.fallback else None
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._schema_prompt(cluster_text, candidates)},
        ]
        for attempt in range(2):
            try:
                raw = self.backend.complete(messages)
            except Exception as exc:  # noqa: BLE001
                if self._trip(exc):
                    return (self.fallback.classify(cluster_text, candidates)
                            if self.fallback else None)
                print(f"  [warn] opencode classify failed: {str(exc)[:120]}")
                return None
            parsed = _extract_json(raw)
            if parsed is not None:
                try:
                    return RiskClassification(**parsed)
                except Exception as exc:  # noqa: BLE001 - pydantic ValidationError
                    if attempt == 0:
                        messages.append({"role": "assistant", "content": raw[:800]})
                        messages.append({
                            "role": "user",
                            "content": f"That did not validate: {str(exc)[:400]}. "
                                       f"Return only the corrected JSON object.",
                        })
                        continue
                    return None
            if attempt == 0:
                messages.append({"role": "assistant", "content": raw[:800]})
                messages.append({"role": "user",
                                 "content": "Return only a JSON object, nothing else."})
        return None

    def draft(self, cluster_text: str, classification: RiskClassification,
              propagation: str, market_note: str, sources: Sequence[str]) -> str:
        if self._disabled:
            return (self.fallback.draft(cluster_text, classification, propagation,
                                        market_note, sources)
                    if self.fallback else classification.summary)
        prompt = _draft_prompt(cluster_text, classification, propagation,
                               market_note, sources)
        try:
            return self.backend.complete(
                [{"role": "user", "content": prompt}]).strip()
        except Exception as exc:  # noqa: BLE001
            if not self._trip(exc):
                print(f"  [warn] opencode draft failed: {str(exc)[:120]}")
            if self.fallback:
                return self.fallback.draft(cluster_text, classification,
                                           propagation, market_note, sources)
            return classification.summary


# ------------------------------------------------------------------ offline

# Word boundaries are load-bearing here. An unanchored `port` matched "Reported
# items:" in every SEC 8-K body, which classified all of them as logistics.
_RISK_PATTERNS: list[tuple[str, RiskType, int]] = [
    (r"\b(earthquake|quake|seismic|typhoon|flood|tsunami)\b", "natural_disaster", 4),
    (r"\b(fire|explosion|contamination|blackout|shutdown)\b|power outage",
     "fab_incident", 4),
    (r"export control|entity list|\b(sanction|tariffs?|embargo)\b|\bban on\b",
     "export_control", 4),
    (r"\b(allocation|shortage|constrained?)\b|sold out|capacity crunch",
     "capacity_allocation", 3),
    (r"\b(ports?|freight|cargo|shipping|customs|logistics)\b", "logistics", 3),
    (r"\b(strike|union|walkout)\b|labou?r dispute", "labor", 3),
    (r"\b(pricing|prices)\b|contract price|spot price", "pricing", 2),
    (r"\b(demand|guidance)\b|order cut", "demand_shock", 2),
]

_MAGNITUDE = re.compile(r"\bmagnitude\s+([0-9]\.[0-9])\b", re.IGNORECASE)


def _quake_severity(text: str) -> Optional[int]:
    """Grade an earthquake by magnitude rather than by the word 'earthquake'.

    Without this every seismic event scored 4 and a routine M5.8 at 61km depth
    was promoted into the alerts section.
    """
    found = _MAGNITUDE.search(text)
    if not found:
        return None
    magnitude = float(found.group(1))
    if magnitude >= 7.0:
        return 5
    if magnitude >= 6.5:
        return 4
    if magnitude >= 6.0:
        return 3
    if magnitude >= 5.5:
        return 2
    return 1

_SPECULATIVE = re.compile(
    r"\b(expect|forecast|predict|may |could |reportedly|rumou?r|analyst|likely|"
    r"weigh|consider|plan to|set to)\b", re.IGNORECASE)


class HeuristicClassifier:
    """Deterministic fallback. No network, no key, blunt but honest.

    Marks everything low-confidence so a digest built without a model never
    reads as though it had one.
    """

    def __init__(self, registry: Registry):
        self.registry = registry

    def classify(self, cluster_text: str, candidates: Sequence[str]
                 ) -> Optional[RiskClassification]:
        text = cluster_text.lower()
        risk_type: RiskType = "other"
        severity = 2
        for pattern, kind, sev in _RISK_PATTERNS:
            if re.search(pattern, text):
                risk_type, severity = kind, sev
                break
        if risk_type == "natural_disaster":
            graded = _quake_severity(cluster_text)
            if graded is not None:
                severity = graded

        stages = self.registry.stages_for(candidates)
        ram = {"dram_die", "nand_die", "hbm_stack", "dram_module", "ssd"}
        gpu = {"logic_wafer", "cowos", "gpu_module"}
        has_ram, has_gpu = bool(ram & set(stages)), bool(gpu & set(stages))
        commodity = (["both"] if has_ram and has_gpu
                     else ["RAM"] if has_ram
                     else ["GPU"] if has_gpu else ["neither"])

        sentence = next((s.strip() for s in re.split(r"(?<=[.!?])\s+", cluster_text)
                         if s.strip()), cluster_text[:200])

        return RiskClassification(
            relevant=bool(candidates) and risk_type != "other",
            commodity=commodity,
            risk_type=risk_type,
            severity=severity,
            confidence=0.35,               # never pretend to model-grade judgement
            is_speculative=bool(_SPECULATIVE.search(cluster_text)),
            evidence_quote=sentence[:300],
            entities=list(candidates)[:8],
            horizon="unknown (heuristic)",
            summary=sentence[:280],
        )

    def draft(self, cluster_text: str, classification: RiskClassification,
              propagation: str, market_note: str, sources: Sequence[str]) -> str:
        bits = [classification.summary]
        if propagation:
            bits.append(f"Propagation: {propagation}.")
        if classification.is_speculative:
            bits.append("Reported as expectation rather than confirmed event.")
        if market_note:
            bits.append(market_note + ".")
        bits.append(f"Sources: {', '.join(sources[:5])}.")
        return " ".join(bits)


def has_credentials() -> bool:
    """Whether the Anthropic SDK will find credentials.

    An unset ANTHROPIC_API_KEY does NOT mean there are none. The SDK also reads
    an OAuth profile written by `ant auth login`, and a bare Anthropic() client
    works from that with no env var set - so checking env vars alone silently
    downgrades an authenticated user to the heuristic classifier.
    """
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config" / "anthropic").is_dir()


def get_classifier(registry: Registry, force_offline: bool = False):
    """Pick a classifier and say which one out loud.

    Order: explicit override, then OpenCode Go, then Anthropic, then heuristic.
    OpenCode is preferred when its key is present so a single subscription can
    drive both the digest and the chat; set SEMIMON_CLASSIFIER=anthropic to
    invert that.

    Announcing the choice matters: silently degrading to worse judgement is how
    you end up trusting a digest that no model ever read.
    """
    if force_offline:
        print("  classifier: heuristic (offline, forced)")
        return HeuristicClassifier(registry)

    choice = os.environ.get("SEMIMON_CLASSIFIER", "auto").lower()
    if choice == "heuristic":
        print("  classifier: heuristic (SEMIMON_CLASSIFIER=heuristic)")
        return HeuristicClassifier(registry)

    want_opencode = choice == "opencode" or (
        choice == "auto" and os.environ.get("OPENCODE_API_KEY"))
    if want_opencode:
        try:
            classifier = OpenCodeClassifier(
                fallback=HeuristicClassifier(registry))
            print(f"  classifier: opencode-go / {classifier.model}")
            return classifier
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] OpenCode classifier unavailable ({exc})")

    if choice in ("auto", "anthropic") and has_credentials():
        try:
            classifier = LLMClassifier()
            print(f"  classifier: {MODEL}")
            return classifier
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] Anthropic classifier unavailable ({exc})")

    print("  classifier: heuristic (no model credentials found - set "
          "OPENCODE_API_KEY, or ANTHROPIC_API_KEY)")
    return HeuristicClassifier(registry)
