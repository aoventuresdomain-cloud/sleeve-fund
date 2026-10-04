"""Fast sanity checks: the engine's arithmetic, one rule at a time, in well under a minute.

The end-to-end review replays whole strategies through every screen. These checks sit beside it and
catch the cheap, embarrassing mistakes quickly: a fee that doesn't follow the size of the trade, a
maker fill charged the taker rate, books that don't add up, a position left open after a full exit,
an R that isn't the stop distance, and paper entering or exiting differently from the backtest on
the same prices.

They trade a probe strategy defined here (not one of the shipped strategies, which come and go): it
goes long for `period` decision bars, then flat for `period`, on the clock, so paper and the backtest
decide on exactly the same bars. Run them alone with:  pytest -m sanity
"""

from __future__ import annotations

from datetime import timezone

import numpy as np
import pandas as pd
import pytest
from nautilus_trader.model import Bar

from sleeve_fund import markets, risk
from sleeve_fund.instruments import BOOK_SHARE
from sleeve_fund.paper.recorder import Recorder
from sleeve_fund.research.replay import replay
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.strategies import REGISTRY
from sleeve_fund.strategies.base import LongFlatConfig, LongFlatStrategy
from sleeve_fund.venues import venue

pytestmark = pytest.mark.sanity

K = venue("KRAKEN")
MAKER, TAKER = float(K.fees.maker), float(K.fees.taker)
HALF = 0.0005  # a 5 bp half spread on orders that take liquidity
CENT = 0.0100001  # fees are kept to the cent with the rounding carried to the next (S-1): one fee may be a cent out

# Three price scales: a large price, a sub-dollar one and a sub-cent one.
INSTRUMENTS = {
    "BTC": (K.instrument("BTC", "USD", price_precision=1), 60_000.0),
    "XRP": (K.instrument("XRP", "USD", price_precision=5), 0.5),
    "DOGE": (K.instrument("DOGE", "USD", price_precision=7), 0.08),
}


class ProbeConfig(LongFlatConfig):
    def __init__(self, *, period: int = 5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.period = period


class Probe(LongFlatStrategy):
    """Long for `period` bars, flat for `period` bars, by the bar's close time, so every path decides
    on the same bars whatever it has seen before."""

    def __init__(self, config: ProbeConfig) -> None:
        super().__init__(config)
        self.step = config.bar_type.spec.timedelta.total_seconds() * 1e9
        self.period = config.period

    def want_long(self, bar: Bar) -> bool | None:
        return (int(bar.ts_event // self.step) // self.period) % 2 == 1


@pytest.fixture(autouse=True)
def _probe(monkeypatch):
    monkeypatch.setitem(REGISTRY, "probe", (Probe, ProbeConfig))


def _bars(px, n=240, minutes=60, vol=None, seed=1, sigma=0.003):
    rng = np.random.default_rng(seed)
    c = px * np.exp(np.cumsum(rng.normal(0, sigma, n)))
    o = np.r_[px, c[:-1]]
    idx = pd.date_range("2025-01-01", periods=n, freq=f"{minutes}min", tz="UTC") + pd.Timedelta(minutes=minutes)
    volume = vol if vol is not None else 1e9 / px  # deep: the volume cap never binds unless asked to
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.001, "low": np.minimum(o, c) * 0.999,
                         "close": c, "volume": volume}, index=idx)


def _run(prices, inst, params=None, capital=10_000.0, profile="aggressive", minutes=60, **kw):
    params = {"period": 5, **(params or {})}
    return run_backtest("probe", prices, inst, params, starting_capital=capital, risk_profile=profile,
                        bar_minutes=minutes, half_spread=kw.pop("half_spread", HALF), **kw)


def _entries(j):
    return [o for o in j.orders_.values() if o["intent"] == "entry" and o["filled_qty"] > 0]


def _notional(fills):
    return sum(f["qty"] * f["price"] for f in fills)


# --- fees -------------------------------------------------------------------------------------------


def _rate(order):
    """What the venue charges this order on its notional, as the journal books it: the maker fee on a
    post-only order, else the taker fee plus the half spread (a backtest books the spread as a cost)."""
    return MAKER if order["order_type"] == "POST-ONLY LIMIT" else TAKER + HALF


def _assert_every_fee(j):
    assert j.fills_, "no fills"
    for f in j.fills_:
        expected = f["qty"] * f["price"] * _rate(j.orders_[f["order_id"]])
        assert f["fee"] == pytest.approx(expected, abs=CENT), (f, expected)
    # The rounding is carried from fee to fee, so however many fills, the total stays within a cent: a
    # one-cent tolerance per fee must not hide a bias that builds up.
    exact = sum(f["qty"] * f["price"] * _rate(j.orders_[f["order_id"]]) for f in j.fills_)
    assert sum(f["fee"] for f in j.fills_) == pytest.approx(exact, abs=0.0100001), (len(j.fills_), exact)


@pytest.mark.parametrize("capital", [100.0, 10_000.0, 1_000_000.0])
@pytest.mark.parametrize("name", list(INSTRUMENTS))
def test_every_fill_pays_its_rate_on_its_own_notional(name, capital):
    inst, px = INSTRUMENTS[name]
    j = _run(_bars(px), inst, capital=capital).journal
    _assert_every_fee(j)
    total = sum(f["fee"] for f in j.fills_)
    assert total == pytest.approx(_notional(j.fills_) * (TAKER + HALF), rel=1e-3, abs=CENT * len(j.fills_))


# Each sizing knob, at two settings: the entry's notional must follow the knob by the expected ratio,
# and the fees must follow the notional. "Sized by" must name the knob, so the PM can see what set it.
SIZING = {
    "starting capital": (dict(capital=10_000.0), dict(capital=100_000.0), 10.0, "aggressive risk profile cap"),
    "risk profile": (dict(profile="conservative"), dict(profile="aggressive"), 0.50 / 0.20, "risk profile cap"),
    "largest order": (dict(params={"max_notional": 400.0}), dict(params={"max_notional": 4_000.0}), 10.0,
                      "largest order cap"),
    "risk per trade": (dict(params={"stop_loss": 0.05, "risk_per_trade": 0.002}),
                       dict(params={"stop_loss": 0.05, "risk_per_trade": 0.01}), 5.0, "risk per trade"),
    "volume": (dict(capital=1e6, vol=2.0), dict(capital=1e6, vol=20.0), 10.0, "share of the bar's volume"),
}


@pytest.mark.parametrize("knob", list(SIZING))
def test_fees_follow_the_size_whatever_sets_it(knob):
    small, large, ratio, sized_by = SIZING[knob]
    inst, px = INSTRUMENTS["BTC"]
    runs = []
    for setting in (small, large):
        setting = dict(setting)
        prices = _bars(px, vol=setting.pop("vol", None))
        runs.append(_run(prices, inst, **setting).journal)
    first = [_entries(j)[0] for j in runs]
    for o in first:
        assert sized_by in o["signal"]["sized_by"], o["signal"]
    size = [o["filled_qty"] * o["avg_px"] for o in first]
    assert size[1] / size[0] == pytest.approx(ratio, rel=0.01), size
    fees = [sum(f["fee"] for f in j.fills_) for j in runs]
    traded = [_notional(j.fills_) for j in runs]
    assert fees[1] / fees[0] == pytest.approx(traded[1] / traded[0], rel=0.01), (fees, traded)
    for f, t in zip(fees, traded):
        assert f / t == pytest.approx(TAKER + HALF, rel=1e-3), (f, t)


def _maker_run(px, inst, vol_per_minute, capital=10_000.0):
    """15-minute decisions on 1-minute execution bars, post-only orders waiting 5 minutes."""
    minutes = _bars(px, n=15 * 120, minutes=1, vol=vol_per_minute, sigma=0.0006)
    decisions = minutes.resample("15min", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    return _run(decisions, inst, {"maker_wait_minutes": 5}, capital=capital, minutes=15,
                exec_prices=minutes, exec_minutes=1)


@pytest.mark.usefixtures("maker_on")
@pytest.mark.parametrize("name", list(INSTRUMENTS))
def test_post_only_fills_pay_the_maker_fee_and_market_fills_the_taker_fee(name):
    inst, px = INSTRUMENTS[name]
    j = _maker_run(px, inst, vol_per_minute=1e9 / px).journal
    kinds = {j.orders_[f["order_id"]]["order_type"] for f in j.fills_}
    assert "POST-ONLY LIMIT" in kinds
    _assert_every_fee(j)
    maker = [f for f in j.fills_ if j.orders_[f["order_id"]]["order_type"] == "POST-ONLY LIMIT"]
    assert sum(f["fee"] for f in maker) / _notional(maker) == pytest.approx(MAKER, rel=1e-3)


@pytest.mark.usefixtures("maker_on")
@pytest.mark.parametrize("capital", [96.0, 200.0, 20_000.0, 1_000_000.0])
def test_post_only_orders_filled_in_slices_still_pay_the_maker_rate_overall(capital):
    """Thin minutes: each post-only order fills a slice at a time (BOOK_SHARE of what trades), so small
    accounts get many small fills. Rounding each to the cent must not move the total off the rate."""
    inst, px = INSTRUMENTS["BTC"]
    j = _maker_run(px, inst, vol_per_minute=capital / px / 4, capital=capital).journal
    maker = [f for f in j.fills_ if j.orders_[f["order_id"]]["order_type"] == "POST-ONLY LIMIT"]
    assert len(maker) > len({f["order_id"] for f in maker}), "expected slices"
    _assert_every_fee(j)
    assert sum(f["fee"] for f in maker) / _notional(maker) == pytest.approx(MAKER, rel=0.01)


# --- the books --------------------------------------------------------------------------------------


def _held(fills):
    return sum(f["qty"] if f["side"] == "BUY" else -f["qty"] for f in fills)


def _cash(start, fills):
    return start + sum((f["qty"] * f["price"]) * (1 if f["side"] == "SELL" else -1) - f["fee"] for f in fills)


@pytest.mark.usefixtures("maker_on")
@pytest.mark.parametrize("name", list(INSTRUMENTS))
@pytest.mark.parametrize("maker", [False, True])
def test_books_add_up_and_every_full_exit_leaves_nothing(name, maker):
    inst, px = INSTRUMENTS[name]
    res = _maker_run(px, inst, vol_per_minute=1e9 / px) if maker else _run(_bars(px), inst)
    j = res.journal
    lot = float(inst.size_increment)
    # Once an exit (and any rest of it sent at market) has filled, nothing is left: not a lot, not dust.
    intent = lambda f: j.orders_[f["order_id"]]["intent"]  # noqa: E731
    held, exits = 0.0, 0
    for i, f in enumerate(j.fills_):
        held += f["qty"] if f["side"] == "BUY" else -f["qty"]
        after = j.fills_[i + 1] if i + 1 < len(j.fills_) else None
        if intent(f) != "entry" and (after is None or intent(after) == "entry"):
            exits += 1
            assert abs(held) < lot / 2, (f, held)
    assert exits >= 5
    # Cash and position from the fills are the journal's own, and equity is cash plus the position
    # at the last price.
    last = j.equity[-1]
    assert last["qty"] == pytest.approx(_held(j.fills_), abs=lot / 2)
    # The venue keeps cash to the cent on every fill (its notional and its fee), the journal's fills
    # don't: at most a cent apart per fill.
    assert last["cash"] == pytest.approx(_cash(10_000.0, j.fills_), abs=0.01 * len(j.fills_))
    assert last["equity"] == pytest.approx(last["cash"] + last["qty"] * last["price"], abs=0.01)
    # The backtest's own equity curve ends at the same value.
    assert float(res.equity.iloc[-1]) == pytest.approx(last["equity"], rel=1e-5)


def test_research_equity_is_cash_plus_position_from_the_fills_report():
    """The research path (no risk profile) reports from the venue's fills; the same arithmetic holds."""
    inst, px = INSTRUMENTS["BTC"]
    prices = _bars(px)
    res = run_backtest("probe", prices, inst, {"period": 5}, starting_capital=10_000, bar_minutes=60, half_spread=HALF)
    f = res.fills
    qty = f["filled_qty"].astype(float) * np.where(f["side"].astype(str).str.endswith("BUY"), 1, -1)
    cash = 10_000 - (qty * f["avg_px"].astype(float)).sum() - res.fees_paid
    equity = cash + qty.sum() * prices["close"].iloc[-1]
    assert float(res.equity.iloc[-1]) == pytest.approx(equity, abs=0.01 * len(f) + 0.01)
    assert res.fees_paid / (f["filled_qty"].astype(float) * f["avg_px"].astype(float)).sum() == pytest.approx(TAKER, rel=1e-3)


# --- R ----------------------------------------------------------------------------------------------


def _drops(px, stop, n=200, minutes=60):
    """Bars that drift up, with a clean drop through any stop every 10 bars: the stop fills at its level."""
    c = [px]
    for i in range(1, n):
        c.append(c[-1] * (1 - 2.5 * stop if i % 10 == 7 else 1.0005))
    c = np.array(c)
    o = np.r_[px, c[:-1]]
    idx = pd.date_range("2025-01-01", periods=n, freq=f"{minutes}min", tz="UTC") + pd.Timedelta(minutes=minutes)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.0001, "low": np.minimum(o, c) * 0.9999,
                         "close": c, "volume": 1e9 / px}, index=idx)


@pytest.mark.parametrize("risk", [0.005, 0.02])
def test_one_r_is_the_loss_at_the_stop_and_a_stop_loses_one_r(risk):
    inst, px = INSTRUMENTS["BTC"]
    stop = 0.02
    j = _run(_drops(px, stop), inst, {"period": 2, "stop_loss": stop, "risk_per_trade": risk}).journal
    entries = _entries(j)
    assert entries
    for o in entries:
        s = o["signal"]
        loss = stop + (TAKER + HALF) + (1 - stop) * (TAKER + HALF)
        assert s["risk_amount"] == pytest.approx(o["filled_qty"] * s["close"] * loss, abs=0.01), s
    stopped = [o for o in j.orders_.values() if o["intent"] == "stop_loss" and o["status"] == "filled"]
    assert len(stopped) >= 3
    by_order = {}
    for f in j.fills_:
        by_order.setdefault(f["order_id"], []).append(f)
    for o in stopped:
        entry = max((e for e in entries if e["ts"] <= o["ts"]), key=lambda e: e["ts"])
        cost = sum(f["qty"] * f["price"] + f["fee"] for f in by_order[entry["order_id"]])
        got = sum(f["qty"] * f["price"] - f["fee"] for f in by_order[o["order_id"]])
        realised_r = (got - cost) / entry["signal"]["risk_amount"]
        assert realised_r == pytest.approx(-1.0, abs=0.02), (entry, o, realised_r)
        # And the stop sold at its level, not somewhere else on the bar.
        assert o["avg_px"] == pytest.approx(entry["avg_px"] * (1 - stop), rel=2e-4), o


# --- paper and the backtest on the same prices ------------------------------------------------------

START = 1_759_449_600_000_000_000  # 2025-10-03 00:00 UTC, in nanoseconds
SPREAD = 12.0  # $12 wide around each BTC trade
TICK_INST = K.instrument("BTC", "USD", price_precision=1)


def _fees(params):
    return markets.fees_for(params, K.fees)


def _record(path, prices, params, profile="aggressive", size=1.0, strategy="probe"):
    """A paper session with one trade a second, the quote following each trade. It charges what the paper
    node would: the venue's fees on spot, the market's on a perpetual (paper.config.SleeveConfig.fees)."""
    from nautilus_trader.model import AggressorSide, Price, Quantity, QuoteTick, TradeId, TradeTick

    rec = Recorder(path)
    rec.meta = {"balances": ["10000.00 USD"],
                "sleeve": {"name": "sanity", "strategy": strategy, "instrument": "BTC/USD",
                           "bar_spec": "1-MINUTE-LAST-INTERNAL", "starting_balance": 10_000,
                           "risk_profile": profile, "params": params, "max_notional": None,
                           "maker_fee": str(_fees(params).maker), "taker_fee": str(_fees(params).taker),
                           "tick_seconds": 30}}
    rec.start(TICK_INST)
    stamps = []
    for s, px in enumerate(np.round(prices, 1)):
        t = START + s * 1_000_000_000
        rec.trade(TradeTick(TICK_INST.id, Price(px, 1), Quantity(size, 8),
                            AggressorSide.BUY if s % 2 else AggressorSide.SELL, TradeId(str(s)), t, t + 1000))
        rec.quote(QuoteTick(TICK_INST.id, Price(px - SPREAD / 2, 1), Price(px + SPREAD / 2, 1), Quantity(1, 8),
                            Quantity(1, 8), t + 2000, t + 3000))
        stamps.append(t)
    rec.close()
    return pd.Series(np.round(prices, 1), index=pd.to_datetime(stamps, utc=True))


def _per_order(fills, intents):
    """(side, intent, minute, qty, all-in price, fee) per order, in fill order. The all-in price
    includes the fee and spread: paper pays the spread in its price and the backtest with its fee."""
    out = {}
    for f in fills:
        side, qty, notional, fee, _ = out.get(f["order_id"], (f["side"], 0.0, 0.0, 0.0, None))
        out[f["order_id"]] = (side, qty + f["qty"], notional + f["qty"] * f["price"], fee + f["fee"], f["ts"])
    rows = []
    for k, (side, qty, notional, fee, ts) in out.items():
        t = pd.Timestamp(ts).tz_convert(timezone.utc)
        minute = t if t == t.floor("1min") else t.ceil("1min")
        rows.append((side, intents[k], minute, qty, (notional + fee if side == "BUY" else notional - fee) / qty, fee))
    return rows


def _paper_and_backtest(tmp_path, prices, params, strategy="probe", profile="aggressive"):
    trades = _record(tmp_path / "s.jsonl.gz", prices, params, profile=profile, strategy=strategy)
    orders, fills = replay(tmp_path / "s.jsonl.gz", with_fills=True)
    paper = _per_order(fills, {o["order_id"]: o["intent"] for o in orders})
    bars = trades.resample("1min", closed="left", label="right").ohlc()
    bars["volume"] = 60 / BOOK_SHARE  # the same trades as liquidity on both paths
    res = run_backtest(strategy, bars, TICK_INST, params=params, starting_capital=10_000, risk_profile=profile,
                       bar_minutes=1, half_spread=SPREAD / 2 / float(prices[0]))
    j = res.journal
    return paper, _per_order(j.fills_, {k: o["intent"] for k, o in j.orders_.items()})


def test_paper_and_backtest_enter_and_exit_at_the_same_prices(tmp_path):
    """Signal entries and exits are market orders on both paths: same minute, same size, and the same
    all-in price to within rounding (0.3 bp). The fees then match too."""
    s = np.arange(180 * 60)
    prices = 60_000 * (1 + 0.01 * np.sin(s / 700) + 0.001 * np.sin(s / 11))
    paper, bt = _paper_and_backtest(tmp_path, prices, {"period": 7})
    assert len(paper) >= 20
    assert [r[:3] for r in paper] == [r[:3] for r in bt]
    for p, b in zip(paper, bt):
        assert b[3] == pytest.approx(p[3], rel=2e-3), (p, b)
        assert abs(b[4] / p[4] - 1) * 1e4 <= 0.3, (p, b)
    fee_p, fee_b = sum(r[5] for r in paper), sum(r[5] for r in bt)
    # Paper's fee is the venue's alone (its price carries the spread); the backtest's includes the spread.
    spread_b = sum(r[3] * SPREAD / 2 for r in bt)
    assert fee_b - spread_b == pytest.approx(fee_p, rel=0.005), (fee_p, fee_b, spread_b)


def test_paper_and_backtest_take_profit_and_stop_in_the_same_minute(tmp_path):
    """With a stop and a target: paper sells on the trade through the level, the backtest at the level,
    so prices may differ by a few bp (the same bounds as tests/test_tick_bar_parity.py), never the minute."""
    s = np.arange(240 * 60)
    prices = 60_000 * (1 + 0.012 * np.sin(s / 500) + 0.0015 * np.sin(s / 13))
    params = {"period": 9, "stop_loss": 0.004, "take_profit": 0.025}
    paper, bt = _paper_and_backtest(tmp_path, prices, params)
    intents = {r[1] for r in paper}
    assert {"entry", "stop_loss"} <= intents or {"entry", "take_profit"} <= intents, intents
    assert [r[:2] for r in paper] == [r[:2] for r in bt]
    tol = {"entry": (-0.3, 0.3), "exit": (-0.3, 0.3), "stop_loss": (-2.5, 7.0), "take_profit": (-6.0, 1.0)}
    for p, b in zip(paper, bt):
        assert b[3] == pytest.approx(p[3], rel=2e-3), (p, b)
        lo, hi = tol[p[1]]
        assert lo <= (b[4] / p[4] - 1) * 1e4 <= hi, (p, b)
        # Signal orders in the same minute; a stop or target in the same minute or the next (paper's
        # stop sits off its fill at the ask, the backtest's off the bar's trade price: half a spread apart).
        late = (b[2] - p[2]).total_seconds()
        assert late == 0 if p[1] in ("entry", "exit") else 0 <= late <= 60, (p, b)


# --- the shipped test strategies as probes -----------------------------------------------------------

# Prices that swing about 3% over an hour or so with small wiggles, drifting up so ping-pong's buy near a
# top still sees its 1% rise: enough for ping-pong's 1% rise and 0.5% dip, and for RSI(14) on minute bars to reach its bands both ways.
TEST_STRATEGIES = {
    "ping_pong": ({"rise": 0.01, "dip": 0.005},
                  lambda s: 60_000 * (1 + 0.015 * np.sin(s / 600) + 0.001 * np.sin(s / 17) + 2e-6 * s)),
    # Wilder's RSI(14), the standard one, is smoother than the engine's exponential RSI it once traded on
    # (review round 11, M11-1): a 1.2% swing over about 40 minutes takes it past both bands.
    "rsi_bands": ({},
                  lambda s: 60_000 * (1 + 0.006 * np.sin(s / 400) + 0.0008 * np.sin(s / 29))),
}


@pytest.mark.parametrize("strategy", list(TEST_STRATEGIES))
def test_test_strategies_enter_exit_and_pay_fees_alike_in_paper_and_backtest(tmp_path, strategy):
    """The two test strategies (the PM's probes) on the same trades: paper replayed tick by tick and
    the backtest on minute bars send the same orders in the same minute, at the same size and all-in
    price to within 0.3 bp, and pay the same fees."""
    params, path = TEST_STRATEGIES[strategy]
    prices = path(np.arange(240 * 60))
    paper, bt = _paper_and_backtest(tmp_path, prices, params, strategy=strategy, profile="balanced")
    assert len(paper) >= 6, paper
    assert [r[:3] for r in paper] == [r[:3] for r in bt]
    for p, b in zip(paper, bt):
        assert b[3] == pytest.approx(p[3], rel=2e-3), (p, b)
        gap = (b[4] / p[4] - 1) * 1e4
        if p[1] in RISK_EXITS:
            # At 2x leverage the probe can hit the daily-loss pause. Paper closes on the trade that breaches
            # it, the backtest at the close of the minute it judged at its worst price: the same minute and
            # size, a price a few bp kinder to the backtest at most (a known gap, like a stop's).
            assert -15 <= gap * (1 if p[0] == "BUY" else -1) <= 0.3, (p, b)
        else:
            assert abs(gap) <= 0.3, (p, b)
    spread_b = sum(r[3] * SPREAD / 2 for r in bt)
    assert sum(r[5] for r in bt) - spread_b == pytest.approx(sum(r[5] for r in paper), rel=0.005)
    for side, intent, _, qty, px, fee in paper:  # paper's fee is the venue's taker fee alone
        assert fee == pytest.approx(qty * (px - fee / qty if side == "BUY" else px + fee / qty) * TAKER, abs=CENT)


@pytest.mark.parametrize("strategy", list(TEST_STRATEGIES))
def test_test_strategies_book_every_fee_and_end_flat_after_each_exit(strategy):
    params, path = TEST_STRATEGIES[strategy]
    s = np.arange(0, 600 * 60, 60)
    c = path(s)
    o = np.r_[c[0], c[:-1]]
    idx = pd.date_range("2025-01-01", periods=len(c), freq="1min", tz="UTC") + pd.Timedelta(minutes=1)
    prices = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.0002, "low": np.minimum(o, c) * 0.9998,
                           "close": c, "volume": 1e9 / 60_000}, index=idx)
    j = run_backtest(strategy, prices, TICK_INST, params, starting_capital=10_000, risk_profile="balanced",
                     bar_minutes=1, half_spread=HALF).journal
    assert len(j.fills_) >= 6
    _assert_every_fee(j)
    held = 0.0
    for i, f in enumerate(j.fills_):
        held += f["qty"] if f["side"] == "BUY" else -f["qty"]
        after = j.fills_[i + 1] if i + 1 < len(j.fills_) else None
        if f["side"] == "SELL" and (after is None or after["side"] == "BUY"):
            assert abs(held) < float(TICK_INST.size_increment) / 2, (f, held)


# --- market orders by default ------------------------------------------------------------------------


def test_maker_first_orders_are_refused_while_switched_off(monkeypatch):
    """Maker-first is behind an off switch (SLEEVE_MAKER_ORDERS): asking for it is an error, not a
    silent post-only order at a fee the PM didn't choose."""
    monkeypatch.delenv("SLEEVE_MAKER_ORDERS", raising=False)
    inst, px = INSTRUMENTS["BTC"]
    with pytest.raises(ValueError, match="switched off"):
        _run(_bars(px, n=60, minutes=1, vol=1.0), inst, {"maker_wait_minutes": 5}, minutes=1)


@pytest.mark.parametrize("strategy", ["probe", *TEST_STRATEGIES])
def test_every_order_goes_at_market_and_pays_the_taker_fee_by_default(tmp_path, monkeypatch, strategy):
    """With the switch off, every order on both paths (entries, exits, stops and targets) takes
    liquidity: nothing post-only or limit, and every fill pays the taker fee."""
    monkeypatch.delenv("SLEEVE_MAKER_ORDERS", raising=False)
    params, path = TEST_STRATEGIES.get(strategy, ({"period": 7, "stop_loss": 0.004, "take_profit": 0.02},
                                                  lambda s: 60_000 * (1 + 0.01 * np.sin(s / 700) + 0.001 * np.sin(s / 11))))
    trades = _record(tmp_path / "s.jsonl.gz", path(np.arange(180 * 60)), params, profile="balanced", strategy=strategy)
    orders, fills = replay(tmp_path / "s.jsonl.gz", with_fills=True)
    bars = trades.resample("1min", closed="left", label="right").ohlc()
    bars["volume"] = 60 / BOOK_SHARE
    j = run_backtest(strategy, bars, TICK_INST, params=params, starting_capital=10_000, risk_profile="balanced",
                     bar_minutes=1, half_spread=HALF).journal
    for name, os_ in (("paper", orders), ("backtest", list(j.orders_.values()))):
        filled = [o for o in os_ if float(o["filled_qty"]) > 0]
        assert len(filled) >= 4, (name, os_)
        # A backtest's stop rests at the venue as a stop-market order: it still takes liquidity at the taker fee.
        kinds = {(o["intent"], o["order_type"]) for o in filled}
        assert all(t == "MARKET" or (i == "stop_loss" and t == "STOP") for i, t in kinds), (name, kinds)
    for f in fills:
        assert f["fee"] == pytest.approx(f["qty"] * f["price"] * TAKER, abs=CENT), f
    _assert_every_fee(j)


# --- restarts ----------------------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    from sleeve_fund.store import Store

    s = Store(f"sqlite:///{tmp_path}/t.db")
    s.create_sleeve(name="s1", strategy="probe", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                    starting_balance=10_000, risk_profile="balanced")
    return s


def _restarted(store, t):
    from sleeve_fund.paper.runtime import SleeveRuntime

    rt = SleeveRuntime(store, "s1", now=lambda: t[0])
    rt.on_start(TAKER)
    return rt


@pytest.mark.parametrize("command,reason", [("pause", "PM pause"), ("flatten", "PM flatten"),
                                            ("flatten", "Book kill switch: stop everything")])
def test_a_pm_stop_survives_any_number_of_restarts_until_resumed(store, command, reason):
    """Pause, flatten and the book kill switch (a flatten per strategy) have no end time: restarts (a
    settings reload, a stale feed, a crash) never lift them; only a resume does (round 10, B10-3)."""
    from datetime import datetime, timedelta

    t = [datetime(2025, 10, 3, 12, tzinfo=timezone.utc)]
    rt = _restarted(store, t)
    store.command("s1", command, reason)
    rt.tick(equity=10_000, cash=10_000, qty=0.0, price=60_000)
    for _ in range(3):
        t[0] += timedelta(days=2)  # long enough that a daily-loss pause would have ended
        rt = _restarted(store, t)
        rt.tick(equity=10_000, cash=10_000, qty=0.0, price=60_000)
        assert store.sleeve("s1").status == "paused" and not rt.can_open()
    store.command("s1", "resume", "carry on")
    rt.tick(equity=10_000, cash=10_000, qty=0.0, price=60_000)
    assert store.sleeve("s1").status == "running" and rt.can_open()


@pytest.mark.parametrize("reason", ["PM flatten", "Book kill switch: stop everything"])
def test_a_flatten_cut_short_by_a_restart_still_sells_the_position(store, reason):
    """The kill switch promises cash. If the process stops between sending the sell and its fill (the
    moments a kill switch is pressed are the ones processes fall over in), the restart must sell again."""
    from datetime import datetime, timedelta

    t = [datetime(2025, 10, 3, 12, tzinfo=timezone.utc)]
    rt = _restarted(store, t)
    store.command("s1", "flatten", reason)
    assert rt.tick(equity=10_000, cash=7_000, qty=0.05, price=60_000) == "flatten"
    t[0] += timedelta(minutes=1)
    rt = _restarted(store, t)  # the sell never filled
    assert rt.tick(equity=10_000, cash=7_000, qty=0.05, price=60_000) == "flatten"
    rt = _restarted(store, t)  # sold this time: the next restart owes nothing
    assert rt.tick(equity=10_000, cash=10_000, qty=0.0, price=60_000) is None


def test_a_restart_after_a_resume_never_sells_a_position_the_strategy_took_since(store):
    """The other side of S-3: once the PM resumes, a later restart must not replay the old flatten and
    sell a position the strategy opened afterwards."""
    from datetime import datetime, timedelta

    t = [datetime(2025, 10, 3, 12, tzinfo=timezone.utc)]
    rt = _restarted(store, t)
    store.command("s1", "flatten", "Book kill switch: stop everything")
    rt.tick(equity=10_000, cash=7_000, qty=0.05, price=60_000)
    store.command("s1", "resume", "carry on")
    rt.tick(equity=10_000, cash=10_000, qty=0.0, price=60_000)
    for _ in range(2):
        t[0] += timedelta(minutes=1)
        rt = _restarted(store, t)
        assert rt.tick(equity=10_000, cash=7_000, qty=0.05, price=60_000) is None
        assert store.sleeve("s1").status == "running"


def test_a_daily_loss_pause_cut_short_sells_again_but_not_once_it_has_expired(store):
    """A daily-loss pause flattens for 24 hours. A restart inside the pause owes the sell; one after it
    has ended owes nothing, and the strategy trades again."""
    from datetime import datetime, timedelta

    t = [datetime(2025, 10, 3, 12, tzinfo=timezone.utc)]
    rt = _restarted(store, t)
    rt.tick(equity=10_000, cash=7_000, qty=0.05, price=60_000)
    t[0] += timedelta(minutes=1)
    assert rt.tick(equity=9_000, cash=7_000, qty=0.05, price=40_000) == "flatten"  # a 10% day
    assert store.sleeve("s1").status == "paused" and store.sleeve("s1").paused_until is not None
    t[0] += timedelta(minutes=1)
    assert _restarted(store, t).tick(equity=9_000, cash=7_000, qty=0.05, price=40_000) == "flatten"
    t[0] += timedelta(hours=25)
    rt = _restarted(store, t)
    assert rt.tick(equity=9_000, cash=7_000, qty=0.05, price=40_000) is None and rt.can_open()


# --- long and short on a perpetual ----------------------------------------------------------------

PERP = {"market": "perp", "allow_short": True}
PERP_FEES = markets.LOW_FEE_PERP
RISK_EXITS = {"risk_pause", "risk_halt", "liquidation", "liquidation_cut"}


class ProbeLS(Probe):
    """Long for `period` decision bars, then short for `period`, then flat for `period`, on the clock."""

    def want_side(self, bar):
        return (1, -1, 0)[(int(bar.ts_event // self.step) // self.period) % 3]


class ProbeShort(Probe):
    """Short from the first bar, and stays short."""

    def want_side(self, bar):
        return -1


class ProbeLong(Probe):
    """Long from the first bar, and stays long."""

    def want_side(self, bar):
        return 1


@pytest.fixture(autouse=True)
def _probe_ls(monkeypatch):
    monkeypatch.setitem(REGISTRY, "probe_ls", (ProbeLS, ProbeConfig))
    monkeypatch.setitem(REGISTRY, "probe_short", (ProbeShort, ProbeConfig))
    monkeypatch.setitem(REGISTRY, "probe_long", (ProbeLong, ProbeConfig))


def _ls_bars(closes, minutes=60, start="2025-10-03"):
    c = np.asarray(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    idx = pd.date_range(start, periods=len(c), freq=f"{minutes}min", tz="UTC") + pd.Timedelta(minutes=minutes)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c,
                         "volume": 1e12 / float(c[0])}, index=idx)


LS_CASES = {
    "probe_ls": ({"period": 7}, lambda s: 60_000 * (1 + 0.01 * np.sin(s / 700) + 0.001 * np.sin(s / 11))),
    **TEST_STRATEGIES,
}


@pytest.mark.parametrize("strategy", list(LS_CASES))
def test_long_and_short_on_a_perp_enter_exit_and_pay_fees_alike_in_paper_and_backtest(tmp_path, strategy):
    """The PM's two test strategies and the probe, long and short on the simulated low-fee perpetual: paper
    replayed tick by tick and the backtest on minute bars send the same orders (shorts included) in the
    same minute, at the same size and all-in price to within 0.3 bp, and pay the perp's taker fee."""
    params, path = LS_CASES[strategy]
    params = {**params, **PERP}
    paper, bt = _paper_and_backtest(tmp_path, path(np.arange(240 * 60)), params, strategy=strategy,
                                    profile="balanced")
    # The backtest's last bar closes on the last trade; paper's would close on a trade after it, which never
    # comes. An order on that bar alone is an edge of the recording, not a difference.
    end = pd.Timestamp(START, tz="UTC") + pd.Timedelta(minutes=240)
    bt = [r for r in bt if r[2] < end]
    assert ("SELL", "entry") in {r[:2] for r in paper}, paper  # it went short
    assert ("BUY", "exit") in {r[:2] for r in paper}, paper  # and bought the short back
    assert [r[:3] for r in paper] == [r[:3] for r in bt]
    for p, b in zip(paper, bt):
        assert b[3] == pytest.approx(p[3], rel=2e-3), (p, b)
        gap = (b[4] / p[4] - 1) * 1e4
        if p[1] in RISK_EXITS:
            # At 2x leverage the probe can hit the daily-loss pause. Paper closes on the trade that breaches
            # it, the backtest at the close of the minute it judged at its worst price: the same minute and
            # size, a price a few bp kinder to the backtest at most (a known gap, like a stop's).
            assert -15 <= gap * (1 if p[0] == "BUY" else -1) <= 0.3, (p, b)
        else:
            assert abs(gap) <= 0.3, (p, b)
    spread_b = sum(r[3] * SPREAD / 2 for r in bt)
    assert sum(r[5] for r in bt) - spread_b == pytest.approx(sum(r[5] for r in paper), rel=0.005)
    taker = float(PERP_FEES.fees.taker)
    for side, _, _, qty, px, fee in paper:
        assert fee == pytest.approx(qty * (px - fee / qty if side == "BUY" else px + fee / qty) * taker, abs=CENT)


def test_paper_sells_short_at_the_bid_and_buys_it_back_at_the_ask(tmp_path):
    """A short sale takes the bid and its cover takes the ask, like any market order: never the other side
    of the book, which would hand the short the spread. (The recording quotes $6 either side of each trade.)"""
    params = {"period": 7, **PERP}
    prices = LS_CASES["probe_ls"][1](np.arange(180 * 60))
    trades = _record(tmp_path / "s.jsonl.gz", prices, params, profile="balanced", strategy="probe_ls")
    orders, fills = replay(tmp_path / "s.jsonl.gz", with_fills=True)
    intent = {o["order_id"]: o["intent"] for o in orders}
    shorts = [f for f in fills if intent[f["order_id"]] == "entry" and f["side"] == "SELL"]
    assert shorts
    for f in fills:
        ts = pd.Timestamp(f["ts"]).tz_convert(timezone.utc)
        last = trades[trades.index < ts].iloc[-1]  # the quote in force follows the last trade before the fill
        adverse = (f["price"] - last) * (1 if f["side"] == "BUY" else -1)
        assert adverse == pytest.approx(SPREAD / 2, abs=0.11), (intent[f["order_id"]], f, last)


def _trips(j):
    from sleeve_fund.research.metrics import trades

    return trades(list(j.fills_), shorts=True)


def test_every_trip_long_or_short_makes_its_price_move_less_its_fees():
    """P&L sign: a long gains on a rise, a short on a fall, each by quantity x the move less its fees."""
    s = np.arange(0, 400 * 60, 60)
    c = 60_000 * (1 + 0.03 * np.sin(s / 9_000) + 0.004 * np.sin(s / 700))
    j = run_backtest("probe_ls", _ls_bars(c, minutes=60), TICK_INST, {"period": 5, **PERP}, starting_capital=10_000,
                     risk_profile="balanced", bar_minutes=60, half_spread=HALF).journal
    trips = _trips(j)
    assert {t["side"] for t in trips} == {1, -1}, trips
    for t in trips:
        move = t["side"] * t["qty"] * (t["exit_px"] - t["entry_px"])
        assert t["pnl"] == pytest.approx(move - t["fees"], abs=0.02), t
        assert (t["pnl"] + t["fees"] > 0) == (t["side"] * (t["exit_px"] - t["entry_px"]) > 0), t


def test_funding_a_long_pays_and_a_short_receives_and_the_books_add_up():
    """Funding every 8 hours at 0.01% of the position's value: a long pays it, a short receives it. Equity
    is the opening cash, every fill's cash flow and fee, the funding, and the position at the close."""
    s = np.arange(0, 24 * 9 * 3600, 3600)
    c = 60_000 * (1 + 0.01 * np.sin(s / 50_000))
    bars = _ls_bars(c, minutes=60)
    res = run_backtest("probe_ls", bars, TICK_INST, {"period": 24, **PERP}, starting_capital=10_000,
                       risk_profile="balanced", bar_minutes=60, half_spread=HALF)
    j = res.journal
    entries = [o for o in j.orders_.values() if o["intent"] == "entry"]
    assert entries and all(o["signal"]["sized_by"] == "2x leverage cap" for o in entries)  # funding on 2x
    held, cash = 0.0, 10_000.0
    for f in j.fills_:
        sign = 1 if f["side"] == "BUY" else -1
        held += sign * f["qty"]
        cash -= sign * f["qty"] * f["price"] + f["fee"]
    rows = j.funding_
    # One payment at every funding time a position was held through: none missed, none doubled.
    held_at, pos = [], 0.0
    for f in j.fills_:
        pos += f["qty"] if f["side"] == "BUY" else -f["qty"]
        held_at.append((pd.Timestamp(f["ts"]), pos))
    marks = markets.funding_times(bars.index[0].to_pydatetime(), bars.index[-1].to_pydatetime(), PERP_FEES.funding_hours)
    owed = [t for t in marks if abs(next((q for ts, q in reversed(held_at) if ts < t), 0.0)) > 0]
    assert sorted(pd.Timestamp(r["ts"]) for r in rows) == [pd.Timestamp(t) for t in owed]
    longs = [r for r in rows if r["qty"] > 0]
    shorts = [r for r in rows if r["qty"] < 0]
    assert longs and shorts
    for r in rows:
        assert r["amount"] == pytest.approx(-r["qty"] * r["price"] * float(PERP_FEES.funding_rate), rel=1e-6), r
    assert all(r["amount"] < 0 for r in longs) and all(r["amount"] > 0 for r in shorts)
    total = sum(r["amount"] for r in rows)
    assert res.equity.iloc[-1] == pytest.approx(cash + total + held * c[-1], abs=0.05)


def _swing(n=400):
    s = np.arange(0, n * 60, 60)
    return _ls_bars(60_000 * (1 + 0.03 * np.sin(s / 9_000) + 0.004 * np.sin(s / 700)), minutes=60)


@pytest.mark.parametrize("profile", ["conservative", "balanced", "aggressive"])
def test_a_perp_entry_takes_the_leverage_cap_and_its_liquidation_price_is_where_the_margin_runs_out(profile):
    """Sizing on a perp (PM decision, 4 Oct): every entry, long or short, is the profile's leverage cap of
    equity (less the cash buffer and fee, never over it), and the liquidation price it reports is where
    equity falls to the maintenance margin: below the entry for a long, above it for a short, at the
    textbook (1 -/+ 1/leverage) / (1 -/+ maintenance) of the entry, and outside the profile's minimum
    distance. A long at 1x is fully paid for and has none."""
    prof = risk.PROFILES[profile]
    j = run_backtest("probe_ls", _swing(), TICK_INST, {"period": 5, **PERP}, starting_capital=10_000,
                     risk_profile=profile, bar_minutes=60, half_spread=HALF).journal
    entries = [o for o in j.orders_.values() if o["intent"] == "entry"]
    assert {o["side"] for o in entries} == {"BUY", "SELL"}
    m = float(PERP_FEES.maintenance_margin)
    for o in entries:
        sig = o["signal"]
        assert sig["sized_by"] == f"{prof.max_leverage:g}x leverage cap", sig
        if o["side"] == "BUY" and prof.max_leverage <= 1:
            assert "liquidation_px" not in sig, sig
            continue
        lev, liq, close = sig["leverage"], sig["liquidation_px"], sig["close"]
        assert prof.max_leverage * 0.97 <= lev <= prof.max_leverage, sig
        side = 1 if o["side"] == "BUY" else -1
        assert (liq < close) if side > 0 else (liq > close), sig
        assert liq / close == pytest.approx((1 - side / lev) / (1 - side * m), rel=2e-3), sig
        assert abs(liq / close - 1) >= prof.min_liquidation_distance, sig


@pytest.mark.parametrize(("profile", "stop", "refused"), [
    ("balanced", 0.20, False), ("balanced", 0.30, True),  # 2x: liquidation ~50% away, stop at most half
    ("aggressive", 0.10, False), ("aggressive", 0.20, True),  # 3x: ~33% away
])
def test_an_entry_whose_stop_sits_past_the_profiles_share_of_the_way_to_liquidation_is_refused(profile, stop,
                                                                                                refused):
    """With leverage binding, the stop-to-liquidation rule decides: a stop further than the profile's share
    of the distance to liquidation refuses the entry; one inside it enters, and its stop then sits between
    the entry and the liquidation price, at most that share of the way."""
    prof = risk.PROFILES[profile]
    j = run_backtest("probe_ls", _swing(), TICK_INST, {"period": 5, "stop_loss": stop, **PERP},
                     starting_capital=10_000, risk_profile=profile, bar_minutes=60, half_spread=HALF).journal
    entries = [o for o in j.orders_.values() if o["intent"] == "entry"]
    if refused:
        assert not entries and not j.fills_
        return
    assert entries
    for o in entries:
        sig = o["signal"]
        close, liq = sig["close"], sig["liquidation_px"]
        side = 1 if o["side"] == "BUY" else -1
        stop_px = close * (1 - side * sig["stop_frac"])
        assert (liq < stop_px < close) if side > 0 else (close < stop_px < liq), sig
        assert abs(stop_px - close) <= prof.stop_to_liquidation * abs(liq - close) + 1e-6, sig


@pytest.mark.parametrize("profile", ["conservative", "balanced", "aggressive"])
def test_a_rally_against_a_short_is_bought_back_by_the_guards_before_the_venue_would_liquidate(profile):
    """A short caught in a steady rally to three times its price: the daily-loss pause and the drawdown halt
    (or, failing those, the liquidation guard) buy the whole short back at market, the book ends flat, and
    the venue never liquidates it."""
    c = np.r_[np.full(10, 60_000.0), np.linspace(60_000, 180_000, 200)]
    res = run_backtest("probe_short", _ls_bars(c, minutes=60), TICK_INST, PERP, starting_capital=10_000, risk_profile=profile, bar_minutes=60,
                       half_spread=HALF)
    j = res.journal
    orders = sorted(j.orders_.values(), key=lambda o: o["id"])
    fills = {f["order_id"]: f for f in j.fills_}
    assert "risk_halt" in {o["intent"] for o in orders} or "liquidation_cut" in {o["intent"] for o in orders}
    for opened, closed in zip(orders[::2], orders[1::2]):  # each short, then what bought it back
        assert (opened["side"], opened["intent"]) == ("SELL", "entry"), opened
        assert closed["side"] == "BUY" and closed["intent"] in ("risk_pause", "risk_halt", "liquidation_cut")
        assert closed["filled_qty"] == pytest.approx(opened["filled_qty"])  # the whole short
        assert fills[closed["order_id"]]["price"] < opened["signal"]["liquidation_px"], (opened, closed)
    assert "liquidation" not in {o["intent"] for o in orders}  # the venue never took it
    assert abs(_held(j.fills_)) < float(TICK_INST.size_increment) / 2
    assert res.equity.iloc[-1] > 0


@pytest.mark.parametrize("profile", ["balanced", "aggressive"])
def test_a_crash_under_a_leveraged_long_is_sold_by_the_guards_before_the_venue_would_liquidate(profile):
    """The mirror of the rally: a long at 2x or 3x in a steady fall to a third of its price is sold in
    full by the guards at a price above its liquidation price, the book ends flat with equity left."""
    c = np.r_[np.full(10, 60_000.0), np.linspace(60_000, 20_000, 200)]
    res = run_backtest("probe_long", _ls_bars(c, minutes=60), TICK_INST, PERP, starting_capital=10_000,
                       risk_profile=profile, bar_minutes=60, half_spread=HALF)
    j = res.journal
    orders = sorted(j.orders_.values(), key=lambda o: o["id"])
    fills = {f["order_id"]: f for f in j.fills_}
    assert orders
    for opened, closed in zip(orders[::2], orders[1::2]):
        assert (opened["side"], opened["intent"]) == ("BUY", "entry"), opened
        assert closed["side"] == "SELL" and closed["intent"] in RISK_EXITS - {"liquidation"}, closed
        assert closed["filled_qty"] == pytest.approx(opened["filled_qty"])
        assert fills[closed["order_id"]]["price"] > opened["signal"]["liquidation_px"], (opened, closed)
    assert abs(_held(j.fills_)) < float(TICK_INST.size_increment) / 2
    assert res.equity.iloc[-1] > 0


def test_every_perp_exit_closes_to_exactly_zero_and_the_next_entry_rests_its_stop_and_target():
    """B11-2: a perp exit closes the whole position, to zero in the venue's exact quantities, not to a float
    residue a lot below it; so the entry that follows, on either side, opens a fresh position and rests its
    own stop and target on the far side, the whole of its quantity, every time."""
    from decimal import Decimal

    j = run_backtest("probe_ls", _swing(), TICK_INST, {"period": 5, "stop_loss": 0.05, "take_profit": 0.2, **PERP},
                     starting_capital=10_000, risk_profile="balanced", bar_minutes=60, half_spread=HALF).journal
    orders = sorted(j.orders_.values(), key=lambda o: o["id"])
    intent = {o["order_id"]: o["intent"] for o in orders}
    held = Decimal(0)
    for f in sorted(j.fills_, key=lambda f: f["id"]):
        held += Decimal(str(f["qty"])) * (1 if f["side"] == "BUY" else -1)
        if intent[f["order_id"]] != "entry":
            assert held == 0, (f, held)  # every exit leaves exactly nothing
    entries = [i for i, o in enumerate(orders) if o["intent"] == "entry"]
    assert len(entries) > 10 and {orders[i]["side"] for i in entries} == {"BUY", "SELL"}
    for i in entries:
        e = orders[i]
        assert e["signal"].get("stop_frac") == pytest.approx(0.05), e
        far = "SELL" if e["side"] == "BUY" else "BUY"
        legs = {o["intent"]: o for o in orders[i + 1:i + 3]}
        assert set(legs) == {"stop_loss", "take_profit"}, (e, orders[i + 1:i + 3])
        assert all(o["side"] == far and o["qty"] == pytest.approx(e["filled_qty"]) for o in legs.values()), legs


@pytest.mark.parametrize(("strategy", "gap"), [("probe_short", 120_000.0), ("probe_long", 25_000.0)])
@pytest.mark.parametrize("profile", ["balanced", "aggressive"])
def test_a_gap_through_the_liquidation_price_is_liquidated_in_full_and_trades_no_more(profile, strategy, gap):
    """M11-3: a price that jumps past the liquidation price between bars, with no bar near it for the guards
    to act on, is liquidated: one order with the 'liquidation' intent closes the whole position, the book is
    flat after it, and the strategy is halted (no entry after it)."""
    c = np.r_[np.full(10, 60_000.0), np.full(20, gap)]
    res = run_backtest(strategy, _ls_bars(c, minutes=60), TICK_INST, PERP, starting_capital=10_000,
                       risk_profile=profile, bar_minutes=60, half_spread=HALF)
    j = res.journal
    orders = sorted(j.orders_.values(), key=lambda o: o["id"])
    assert [o["intent"] for o in orders] == ["entry", "liquidation"], orders
    opened, closed = orders
    assert closed["side"] != opened["side"] and closed["filled_qty"] == pytest.approx(opened["filled_qty"])
    liq = opened["signal"]["liquidation_px"]
    assert (gap > liq) if opened["side"] == "SELL" else (gap < liq)  # the gap went past it
    assert abs(_held(j.fills_)) < float(TICK_INST.size_increment) / 2


@pytest.mark.xfail(strict=True, reason="sanity 5 Oct: a gap past liquidation books a loss beyond the strategy's "
                                       "isolated margin (equity goes negative); reported to the build thread")
@pytest.mark.parametrize(("strategy", "gap"), [("probe_short", 120_000.0), ("probe_long", 25_000.0)])
def test_isolated_margin_a_liquidation_never_loses_more_than_the_strategys_equity(strategy, gap):
    """Isolated margin (long/short verdict default): a liquidated position loses at most the margin behind
    it, the strategy's equity; the venue's insurance fund takes any shortfall past the bankruptcy price. So
    a strategy's equity never goes below zero, and never pulls the rest of the fund down with it."""
    c = np.r_[np.full(10, 60_000.0), np.full(20, gap)]
    res = run_backtest(strategy, _ls_bars(c, minutes=60), TICK_INST, PERP, starting_capital=10_000,
                       risk_profile="aggressive", bar_minutes=60, half_spread=HALF)
    assert res.equity.min() >= 0, res.equity.min()


def _wilder(closes, n=14):
    """Wilder's RSI, written out from its definition, independent of the engine's indicators."""
    out = [None] * len(closes)
    ch = np.diff(np.asarray(closes, dtype=float))
    g, l = np.clip(ch, 0, None), np.clip(-ch, 0, None)
    if len(ch) < n:
        return out
    ag, al = g[:n].mean(), l[:n].mean()
    for k in range(n, len(ch) + 1):
        if k > n:
            ag, al = (ag * (n - 1) + g[k - 1]) / n, (al * (n - 1) + l[k - 1]) / n
        out[k] = 100.0 if al == 0 and ag > 0 else 50.0 if al == ag == 0 else 100 - 100 / (1 + ag / al)
    return out


def test_rsi_bands_trades_on_the_standard_rsi_at_its_bands():
    """M11-1: rsi_bands decides on Wilder's RSI(14), the one the chart draws and that 'RSI' means: every
    order's journaled RSI equals RSI recomputed from the closes, and each obeys its band (long in at 30 or
    under and out at 55 or over; short in at 70 or over and out at 50 or under)."""
    s = np.arange(0, 240 * 60, 60)
    c = np.round(TEST_STRATEGIES["rsi_bands"][1](s), 1)  # at the instrument's price precision
    bars = _ls_bars(c, minutes=1)
    j = run_backtest("rsi_bands", bars, TICK_INST, PERP, starting_capital=10_000, risk_profile="balanced",
                     bar_minutes=1, half_spread=HALF).journal
    rsi = dict(zip(bars.index, _wilder(c)))
    orders = sorted(j.orders_.values(), key=lambda o: o["id"])
    seen = set()
    for o in orders:
        if o["intent"] not in ("entry", "exit"):
            continue
        r = o["signal"]["rsi"]
        assert r == pytest.approx(rsi[pd.Timestamp(o["ts"])], abs=1e-6), o
        band = {("BUY", "entry"): r <= 30, ("SELL", "exit"): r >= 55,
                ("SELL", "entry"): r >= 70, ("BUY", "exit"): r <= 50}[(o["side"], o["intent"])]
        assert band, o
        seen.add((o["side"], o["intent"]))
    assert seen == {("BUY", "entry"), ("SELL", "exit"), ("SELL", "entry"), ("BUY", "exit")}, seen


@pytest.mark.parametrize("reason", ["PM flatten", "Book kill switch: stop everything"])
def test_a_flatten_of_a_short_cut_short_by_a_restart_buys_it_back(store, reason):
    """S-3 on a short: the flatten is owed again after a restart while a short is still held."""
    from datetime import datetime, timedelta

    t = [datetime(2025, 10, 3, 12, tzinfo=timezone.utc)]
    rt = _restarted(store, t)
    store.command("s1", "flatten", reason)
    assert rt.tick(equity=10_000, cash=13_000, qty=-0.05, price=60_000) == "flatten"
    t[0] += timedelta(minutes=1)
    assert _restarted(store, t).tick(equity=10_000, cash=13_000, qty=-0.05, price=60_000) == "flatten"

