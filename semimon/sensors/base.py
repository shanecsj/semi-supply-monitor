"""Shared fetch plumbing for every sensor.

Politeness is not optional here. SEC EDGAR returns 403 to any client without a
descriptive User-Agent carrying a contact address, and throttles aggressively
past ~10 req/s. GDELT rate-limits silently. So: one place that sets headers,
backs off, and never lets a single dead source take down a collection run.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

# SEC EDGAR requires a User-Agent carrying a real contact address and returns
# 403 without one. Set SEMIMON_CONTACT to your own email before using the EDGAR
# sensor - the default below is a placeholder and SEC may reject it.
CONTACT = os.environ.get(
    "SEMIMON_CONTACT",
    "semi-supply-monitor/0.1 (set SEMIMON_CONTACT to your email)",
)

DEFAULT_HEADERS = {
    "User-Agent": CONTACT,
    "Accept-Encoding": "gzip, deflate",
}

BROWSER_UA = {"User-Agent": "Mozilla/5.0 (compatible; semi-supply-monitor/0.1)"}


class FetchError(RuntimeError):
    pass


# Minimum seconds between requests to the same host.
#
# GDELT is the reason this exists. Six back-to-back DOC API queries earn a mix
# of HTTP 429, connection resets and SSL handshake timeouts - and naive retries
# make it strictly worse by tripling the request count. Roughly one query every
# five seconds is what it actually tolerates.
HOST_MIN_INTERVAL = {
    "api.gdeltproject.org": 12.0,
    "data.sec.gov": 0.2,            # SEC asks for <=10 req/s
    "www.federalregister.gov": 0.5,
}

_last_request: dict[str, float] = {}


def _throttle(url: str) -> None:
    host = urllib.parse.urlparse(url).netloc
    interval = HOST_MIN_INTERVAL.get(host, 0.0)
    if interval:
        elapsed = time.monotonic() - _last_request.get(host, 0.0)
        if elapsed < interval:
            time.sleep(interval - elapsed)
    _last_request[host] = time.monotonic()


def fetch(url: str, headers: Optional[dict] = None, timeout: int = 25,
          retries: int = 3, backoff: float = 3.0) -> bytes:
    """GET with per-host throttling and retry/backoff.

    Raises FetchError once retries are exhausted. Honours Retry-After on 429.
    """
    merged = dict(DEFAULT_HEADERS)
    if headers:
        merged.update(headers)
    last: Optional[Exception] = None
    for attempt in range(retries):
        _throttle(url)
        try:
            request = urllib.request.Request(url, headers=merged)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    import gzip
                    data = gzip.decompress(data)
                return data
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code == 429:
                wait = float(exc.headers.get("Retry-After") or backoff * (attempt + 2))
                time.sleep(min(wait, 30.0))
                continue
            # Other client errors will not fix themselves.
            if exc.code in (400, 401, 403, 404):
                break
            time.sleep(backoff * (attempt + 1))
        except Exception as exc:  # noqa: BLE001 - the network is a swamp
            last = exc
            time.sleep(backoff * (attempt + 1))
    raise FetchError(f"{url}: {last}")


def fetch_json(url: str, headers: Optional[dict] = None, **kwargs) -> Any:
    return json.loads(fetch(url, headers=headers, **kwargs).decode("utf-8", "replace"))


def post_json(url: str, payload: dict, headers: Optional[dict] = None,
              timeout: int = 120) -> Any:
    """POST JSON and parse the JSON response.

    Separate from `fetch` because inference calls want a long timeout and must
    not be silently retried - a retried chat completion is a second charge
    against the subscription quota, not a free do-over.
    """
    merged = {"Content-Type": "application/json", "User-Agent": CONTACT}
    if headers:
        merged.update(headers)
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=merged, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise FetchError(f"{url}: HTTP {exc.code} {detail}") from exc
    except Exception as exc:  # noqa: BLE001
        raise FetchError(f"{url}: {exc}") from exc


def post_stream(url: str, payload: dict, headers: Optional[dict] = None,
                timeout: int = 120):
    """POST and yield server-sent-event text deltas as they arrive.

    Total time is unchanged, but the first token lands in a second or two
    instead of the reader staring at a blank screen for the length of the whole
    answer. For a chat app that is the difference between "instant" and "slow".
    """
    import json as _json

    merged = {"Content-Type": "application/json", "User-Agent": CONTACT,
              "Accept": "text/event-stream"}
    if headers:
        merged.update(headers)
    body = _json.dumps({**payload, "stream": True}).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=merged, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    chunk = _json.loads(data)
                except ValueError:
                    continue
                for choice in chunk.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        yield piece
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise FetchError(f"{url}: HTTP {exc.code} {detail}") from exc
    except Exception as exc:  # noqa: BLE001
        raise FetchError(f"{url}: {exc}") from exc


def qs(base: str, params: dict) -> str:
    # doseq=True so list values become repeated keys (fields[]=a&fields[]=b)
    # rather than a stringified Python list, which Federal Register 400s on.
    return f"{base}?{urllib.parse.urlencode(params, doseq=True)}"


def safe(_label: str, _fn, *args, **kwargs) -> list:
    """Run a collector, log and swallow its failure.

    A collection run touches ~10 independent sources; one being down must not
    cost us the other nine. Leading underscores so that forwarded kwargs named
    `name` or `fn` do not collide with these parameters.
    """
    try:
        return _fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] {_label}: {type(exc).__name__}: {exc}")
        return []
