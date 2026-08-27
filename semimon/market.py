"""Market annotation.

Deliberately small. This is *not* a signal generator and does not score, rank,
or take a direction. It answers one question per digest entry - "did anyone
else think this mattered?" - and that is the only feedback loop the monitor has
about whether it is surfacing things that turn out to be real.

A note on why the number is computed against a sector ETF rather than raw:
during the Hualien earthquake MU moved +4.3% on the day, of which only +4.0%
was specific to MU; the rest was the whole sector drifting. Raw returns would
credit the monitor with noise.

`stooq` was the original plan here and is no longer usable - it now serves a
JavaScript proof-of-work anti-bot challenge instead of CSV. Hence the provider
interface: swapping to a keyed source (Tiingo, Finnhub) is one class.
"""

from __future__ import annotations

import json
import urllib.parse
from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol

from .sensors.base import BROWSER_UA, fetch

SECTOR_CONTROL = "SOXX"


class BarProvider(Protocol):
    def bars(self, ticker: str, start: str, end: str) -> dict[str, float]:
        """Daily closes keyed by YYYY-MM-DD."""


class YahooProvider:
    """Free, no API key, works for US and international listings.

    Terms disallow redistribution - fine on your own machine, swap it out if
    this ever becomes a hosted product.
    """

    BASE = "https://query1.finance.yahoo.com/v8/finance/chart/"

    def bars(self, ticker: str, start: str, end: str) -> dict[str, float]:
        p1 = int(datetime.strptime(start, "%Y-%m-%d")
                 .replace(tzinfo=timezone.utc).timestamp())
        p2 = int(datetime.strptime(end, "%Y-%m-%d")
                 .replace(tzinfo=timezone.utc).timestamp()) + 86400
        url = (f"{self.BASE}{urllib.parse.quote(ticker)}"
               f"?period1={p1}&period2={p2}&interval=1d")
        payload = json.loads(fetch(url, headers=BROWSER_UA).decode("utf-8", "replace"))
        results = (payload.get("chart") or {}).get("result") or []
        if not results:
            return {}
        result = results[0]
        stamps = result.get("timestamp") or []
        closes = (result["indicators"]["quote"][0].get("close")
                  if result.get("indicators") else []) or []
        out: dict[str, float] = {}
        for stamp, close in zip(stamps, closes):
            if close is None:
                continue
            day = datetime.fromtimestamp(stamp, tz=timezone.utc).strftime("%Y-%m-%d")
            out[day] = float(close)
        return out


class Market:
    def __init__(self, provider: Optional[BarProvider] = None):
        self.provider = provider or YahooProvider()
        self._cache: dict[tuple[str, str, str], dict[str, float]] = {}

    def _bars(self, ticker: str, start: str, end: str) -> dict[str, float]:
        key = (ticker, start, end)
        if key not in self._cache:
            try:
                self._cache[key] = self.provider.bars(ticker, start, end)
            except Exception:  # noqa: BLE001 - annotation is best-effort
                self._cache[key] = {}
        return self._cache[key]

    def abnormal_return(self, ticker: str, on: str,
                        control: str = SECTOR_CONTROL) -> Optional[float]:
        """Percent move on `on` minus the sector's move that day.

        Returns None when either series lacks the date - a holiday, a delisting,
        or simply a date before the ticker existed. None means "no annotation",
        never zero, so the digest can stay silent rather than imply flatness.
        """
        try:
            anchor = datetime.strptime(on, "%Y-%m-%d")
        except (ValueError, TypeError):
            # Callers pass through publisher-supplied dates, which arrive in
            # every format imaginable. No annotation beats a crash.
            return None
        window_start = (anchor - timedelta(days=10)).strftime("%Y-%m-%d")
        window_end = (anchor + timedelta(days=2)).strftime("%Y-%m-%d")

        subject = self._bars(ticker, window_start, window_end)
        index = self._bars(control, window_start, window_end)
        if on not in subject or on not in index:
            return None

        def prior(series: dict[str, float]) -> Optional[float]:
            earlier = sorted(d for d in series if d < on)
            return series[earlier[-1]] if earlier else None

        base_subject, base_index = prior(subject), prior(index)
        if not base_subject or not base_index:
            return None
        move = (subject[on] / base_subject - 1.0) * 100.0
        sector = (index[on] / base_index - 1.0) * 100.0
        return round(move - sector, 2)

    def annotate(self, tickers: list[str], on: str) -> str:
        """One-line annotation for a digest entry, or '' when nothing to say."""
        parts = []
        for ticker in tickers[:4]:
            value = self.abnormal_return(ticker, on)
            if value is not None and abs(value) >= 1.0:
                parts.append(f"{ticker} {value:+.1f}%")
        return ("vs sector: " + ", ".join(parts)) if parts else ""
