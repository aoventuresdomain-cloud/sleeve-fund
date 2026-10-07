"""Shared harness for the O17 pre-build xfails (test_o17a_xfails.py, test_o17b_xfails.py). Synthetic data only: no
venue is called. Binance USD-M is the perp under test because it is the venue whose own settled rates are charged
(markets.native_terms: funding_venue set, funding_rate 0.01% is only the fallback). Copy this file into tests/ beside the
two test files.

- `binance` fixture: the funding store in a temp dir, Binance's contract limits offline, and the venue's funding loader
  replaced (paper asks it directly) so nothing leaves the machine.
- `backtest(...)`: a perp backtest of `o17win` on hourly bars, with the settled rates you write with `write_rates`.
- `paper(...)`: the same strategy run as paper (Recorder + replay: the paper runtime, tick by tick, 30 s ticks), with
  a publication time for each settled rate, so "the venue has not published it yet" is real simulated time.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("BACKTEST_ISOLATE", "0")

from nautilus_trader.model import AggressorSide, Price, Quantity, QuoteTick, TradeId, TradeTick  # noqa: E402

from sleeve_fund import funding, markets  # noqa: E402
from sleeve_fund.paper.recorder import Recorder  # noqa: E402
from sleeve_fund.research.replay import replay  # noqa: E402
from sleeve_fund.research.runner import run_backtest  # noqa: E402
from sleeve_fund.store import Store  # noqa: E402
from sleeve_fund.strategies import REGISTRY  # noqa: E402
from sleeve_fund.strategies.base import LongFlatConfig, LongFlatStrategy  # noqa: E402
from sleeve_fund.venues import binance_contract, venue  # noqa: E402

BASELINE = markets.LOW_FEE_PERP.funding_rate  # 0.01%: the fallback native_terms carries for a missing rate
assert BASELINE == 0.0001
PAIR = "BTC/USDT"
NEVER = pd.Timestamp.max.tz_localize("UTC")
PERP = {"market": "perp", "allow_short": True}
_REAL_RATES = funding.rates  # the store reader itself, before any test wraps it

INFO = {"symbols": [  # as tests/test_binance.py
    {"symbol": "BTCUSDT", "pair": "BTCUSDT", "contractType": "PERPETUAL", "status": "TRADING", "baseAsset": "BTC",
     "quoteAsset": "USDT", "pricePrecision": 2, "quantityPrecision": 3,
     "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                 {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                 {"filterType": "MIN_NOTIONAL", "notional": "100"}]}]}


def utc(s) -> pd.Timestamp:
    t = pd.Timestamp(s)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def ms(t) -> int:
    return int(utc(t).timestamp() * 1000)


class O17WinConfig(LongFlatConfig):
    def __init__(self, *, windows=(), **kw) -> None:
        super().__init__(**kw)
        self.windows = [tuple(w) for w in windows]


class O17Win(LongFlatStrategy):
    """On side s while open_ns <= the decision bar's close < close_ns, for each [open_ns, close_ns, s]; flat otherwise."""

    last = None  # the most recent instance: its clock is the simulated "now" the fake venue publishes against

    def __init__(self, config: O17WinConfig) -> None:
        super().__init__(config)
        self.windows = config.windows
        O17Win.last = self

    def want_long(self, bar):
        return None

    def want_side(self, bar):
        for o, c, s in self.windows:
            if o <= bar.ts_event < c:
                return s
        return 0


class O17WeightConfig(O17WinConfig):
    pass


class O17Weight(O17Win):
    """A target weight w while open_ns <= the decision bar's close < close_ns, for each [open_ns, close_ns, w]; flat
    otherwise. Raising the weight while held is an ADD. Refused on a perp at c7a73f1 (E13-6: check_perp_sizing)."""

    def target_weight(self, bar):
        for o, c, w in self.windows:
            if o <= bar.ts_event < c:
                return float(w)
        return 0.0

    want_side = LongFlatStrategy.want_side  # the side from the weight (long while above 0), not O17Win's windows


def win(*spans) -> dict:
    """Params for O17Win: spans of (open, close, side) as UTC strings."""
    return {"windows": [[utc(o).value, utc(c).value, s] for o, c, s in spans], **PERP}


@pytest.fixture(autouse=True)
def _o17win(monkeypatch):
    monkeypatch.setitem(REGISTRY, "o17win", (O17Win, O17WinConfig))
    monkeypatch.setitem(REGISTRY, "o17weight", (O17Weight, O17WeightConfig))


@pytest.fixture(autouse=True)
def _no_swallowed_strategy_errors(capfd, request):
    """As tests/conftest.py: the engine swallows exceptions raised in strategy callbacks, so fail on any it logs. A test
    marked strategy_errors checks the handler errors itself (BacktestResult.handler_error_count), inside the test."""
    yield
    out, err = capfd.readouterr()
    if request.node.get_closest_marker("strategy_errors"):
        return
    for line in (out + err).splitlines():
        if ("Python " in line and ("failed" in line or "raised exception" in line)) or "strategy handler " in line:
            pytest.fail(f"a strategy callback raised inside the engine: {line}")


@pytest.fixture
def binance(tmp_path, monkeypatch):
    monkeypatch.setattr(funding, "DEFAULT_ROOT", tmp_path / "funding")
    b = venue("binance")
    monkeypatch.setattr(b, "contract", lambda pair: binance_contract(pair, get_json=lambda url: INFO))
    monkeypatch.setattr(b, "funding_loader", lambda pair, start: [])  # the venue has nothing unless a test says so
    return b


def write_rates(rates: dict) -> None:
    """Keep these settled rates (settlement time -> rate; NaN is written as JSON NaN) in the funding store."""
    p = funding._path("BINANCE", PAIR)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"rates": [[ms(t), r] for t, r in sorted(rates.items())]}))
    funding._cache.clear()


def settlements(start, end) -> list[pd.Timestamp]:
    """Binance's settlement times (00, 08, 16 UTC) in (start, end]."""
    return [pd.Timestamp(t) for t in markets.funding_times(utc(start).to_pydatetime(), utc(end).to_pydatetime(),
                                                          venue("binance").funding_hours)]


def hourly_bars(start, end) -> pd.DataFrame:
    idx = pd.date_range(utc(start), utc(end), freq="1h")[1:]
    c = 60_000.0 + np.arange(len(idx))
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e9}, index=idx)


def backtest(b, bars, params, risk_profile="balanced"):
    inst = b.instrument("BTC", "USDT")
    return run_backtest("o17win", bars, inst, params, starting_capital=10_000, risk_profile=risk_profile,
                        bar_minutes=60, half_spread=0)


def journal():
    """The paper journal: Postgres when TEST_DATABASE_URL is set (the production engine), else in-memory SQLite."""
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        from sleeve_fund.store import make_engine, metadata

        engine = make_engine(url)
        metadata.drop_all(engine)
        return Store(engine=engine)
    return Store.in_memory()


def paper(tmp_path, monkeypatch, b, params, start, minutes, published: dict | None = None, rates: dict | None = None,
          step: int = 5, price=None, stored: dict | None = None, name: str = "w", store=None,
          tick_seconds: int = 30, strategy: str = "o17win") -> dict:
    """Run `params` as paper from `start` for `minutes`, a trade and a quote every `step` seconds on a slow ramp.

    rates: settlement time -> the venue's settled rate. published: settlement time -> when the venue publishes it
    (default: at the settlement; NEVER: not at all): a direct venue request (the loader) sees it from then on.
    stored: settlement time -> when the collector has it in the store (funding.rates); default: as published.
    name / store: the strategy's name and the journal to run into (default: a fresh one), so two strategies on the
    same instrument can share a journal. tick_seconds: paper's tick (30 s, as deployed).

    price: seconds from `start` -> trade price (default: 60,000 rising 0.01 a second).

    Returns fills, funding rows, events (oldest first) and the last equity mark, read before the journal is reused."""
    rates = {utc(k): v for k, v in (rates or {}).items()}
    published = {utc(k): utc(v) for k, v in (published or {}).items()}
    stored = {**published, **{utc(k): utc(v) for k, v in (stored or {}).items()}}
    write_rates(rates)

    def now():
        s = O17Win.last
        return utc(s.clock.utc_now()) if s is not None else utc(start)

    def visible(t, when):
        return when.get(t, t) <= now()

    def rates_as_stored(venue_name, pair, root=None):
        s = _REAL_RATES(venue_name, pair, root)
        return s[[visible(t, stored) for t in s.index]] if len(s) else s

    monkeypatch.setattr(funding, "rates", rates_as_stored)
    monkeypatch.setattr(b, "funding_loader", lambda pair, since: [(ms(t), r) for t, r in sorted(rates.items())
                                                                  if ms(t) >= since and visible(t, published)])
    O17Win.last = None

    inst = b.instrument("BTC", "USDT")
    fees = markets.fees_for(params, b.fees, "BINANCE")
    path = f"{tmp_path}/paper-{name}-{abs(hash((str(params), start, minutes, str(published), str(stored))))}.jsonl.gz"
    rec = Recorder(path)
    rec.meta = {"balances": ["10000.00 USDT"],
                "sleeve": {"name": name, "strategy": strategy, "instrument": PAIR, "bar_spec": "1-MINUTE-LAST-INTERNAL",
                           "starting_balance": 10_000, "risk_profile": "aggressive", "params": params,
                           "max_notional": None, "maker_fee": str(fees.maker), "taker_fee": str(fees.taker),
                           "tick_seconds": tick_seconds}}
    rec.start(inst)
    t0 = utc(start)
    for i, s in enumerate(range(0, minutes * 60, step)):
        px = round(price(s) if price is not None else 60_000 + s * 0.01, 1)
        t = t0.value + s * 1_000_000_000
        rec.trade(TradeTick(inst.id, Price(px, 1), Quantity(1.0, 3), AggressorSide.BUY if i % 2 else AggressorSide.SELL,
                            TradeId(str(i)), t, t + 1000))
        rec.quote(QuoteTick(inst.id, Price(px - 0.1, 1), Price(px + 0.1, 1), Quantity(1, 3), Quantity(1, 3),
                            t + 2000, t + 3000))
    rec.close()
    store = store if store is not None else journal()
    _, fills = replay(path, with_fills=True, store=store)
    out = {"fills": [(utc(f["ts"]), f["side"], float(f["qty"])) for f in fills],
           "funding": list(reversed(store.funding(name))),
           # every strategy's events and those of no strategy (an alert per instrument need not name one)
           "events": list(reversed(store.events(None, limit=5000))),
           "equity": store.equity_series(name)[-1]}
    for r in out["funding"]:
        r["ts"] = utc(r["ts"])
    for e in out["events"]:
        e["ts"] = utc(e["ts"])
    return out


def kinds(events, kind):
    return [e for e in events if e["kind"] == kind]


def mark_at(t, start, step=5, price=None) -> float:
    """The paper harness's last trade price at or before `t` (the settlement-time mark)."""
    s = int((utc(t) - utc(start)).total_seconds()) // step * step
    return round(price(s) if price is not None else 60_000 + s * 0.01, 1)


def isnan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)
