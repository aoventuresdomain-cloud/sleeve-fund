"""Funding at the venue's own settlements, on the position held at the settlement instant (QA round on #151,
quant-review/v2-p1/open-interest-151.md: P1-O1 and P1-O2, both BLOCKERS). Adapted from QA's own tests: the same
fixtures and assertions, with the two findings' xfails lifted now they are fixed. Synthetic data, no venue called."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from nautilus_trader.model import AggressorSide, Price, Quantity, QuoteTick, TradeId, TradeTick

from sleeve_fund import funding, markets
from sleeve_fund.instruments import BOOK_SHARE
from sleeve_fund.paper.recorder import Recorder
from sleeve_fund.research.replay import replay
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.store import Store
from sleeve_fund.strategies import REGISTRY
from sleeve_fund.strategies.base import LongFlatConfig, LongFlatStrategy
from sleeve_fund.venues import binance_contract, venue


def utc(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


K = venue("kraken")
INST = K.instrument("BTC", "USD", price_precision=1)
PERP = {"market": "perp", "allow_short": True}
RATE = markets.LOW_FEE_PERP.funding_rate  # the simulated perp's fixed 0.01%


class WinConfig(LongFlatConfig):
    def __init__(self, *, open_at: int = 0, close_at: int = 0, side: int = 1, **kw) -> None:
        super().__init__(**kw)
        self.open_at, self.close_at, self.side = open_at, close_at, side


class Win(LongFlatStrategy):
    """On `side` while open_at <= the decision bar's close < close_at, flat otherwise (UTC ns)."""

    def __init__(self, config: WinConfig) -> None:
        super().__init__(config)
        self.o, self.c, self.sd = config.open_at, config.close_at, config.side

    def want_long(self, bar):
        return None

    def want_side(self, bar):
        return self.sd if self.o <= bar.ts_event < self.c else 0


@pytest.fixture(autouse=True)
def _win(monkeypatch):
    monkeypatch.setitem(REGISTRY, "win", (Win, WinConfig))


def ns(s: str) -> int:
    return utc(s).value


def paper_and_backtest(tmp, start, minutes, prices, params, bar_minutes=1):
    """The same trades replayed as paper (tick by tick, 30 s ticks, data-driven) and as a backtest on
    `bar_minutes` decision bars with 1-minute execution bars. Returns (paper fills, paper funding, bt fills,
    bt funding), funding as (ts, qty, price, amount)."""
    fees = markets.fees_for(params, K.fees)
    spec = f"{bar_minutes // 60}-HOUR-LAST-INTERNAL" if bar_minutes % 60 == 0 else f"{bar_minutes}-MINUTE-LAST-INTERNAL"
    path = f"{tmp}/s.jsonl.gz"
    rec = Recorder(path)
    rec.meta = {"balances": ["10000.00 USD"],
                "sleeve": {"name": "w", "strategy": "win", "instrument": "BTC/USD", "bar_spec": spec,
                           "starting_balance": 10_000, "risk_profile": "aggressive", "params": params,
                           "max_notional": None, "maker_fee": str(fees.maker), "taker_fee": str(fees.taker),
                           "tick_seconds": 30}}
    rec.start(INST)
    t0 = utc(start)
    when = t0 + pd.to_timedelta(np.arange(minutes * 60), unit="s")
    for s, px in enumerate(np.round(prices, 1)):
        t = t0.value + s * 1_000_000_000
        rec.trade(TradeTick(INST.id, Price(px, 1), Quantity(1.0, 8), AggressorSide.BUY if s % 2 else AggressorSide.SELL,
                            TradeId(str(s)), t, t + 1000))
        rec.quote(QuoteTick(INST.id, Price(px - 6, 1), Price(px + 6, 1), Quantity(1, 8), Quantity(1, 8), t + 2000, t + 3000))
    rec.close()
    store = Store.in_memory()
    _, fills = replay(path, with_fills=True, store=store)
    pf = [(pd.Timestamp(f["ts"]), f["side"], f["qty"]) for f in fills]
    pfund = [(pd.Timestamp(r["ts"]), r["qty"], r["price"], r["amount"]) for r in reversed(store.funding("w"))]
    trades = pd.Series(np.round(prices, 1), index=when)
    one = trades.resample("1min", closed="left", label="right").ohlc()
    one["volume"] = 60 / BOOK_SHARE
    dec = trades.resample(f"{bar_minutes}min", closed="left", label="right", origin="start_day").ohlc()
    dec["volume"] = 60 / BOOK_SHARE
    res = run_backtest("win", dec, INST, params=params, starting_capital=10_000, risk_profile="aggressive",
                       bar_minutes=bar_minutes, exec_prices=one if bar_minutes > 1 else None, half_spread=6 / 60_000)
    j = res.journal
    bf = [(pd.Timestamp(f["ts"]), f["side"], f["qty"]) for f in j.fills_]
    bfund = [(pd.Timestamp(r["ts"]), r["qty"], r["price"], r["amount"]) for r in j.funding_]
    return pf, pfund, bf, bfund


def _ramp(minutes):
    return 60_000 + np.arange(minutes * 60) * 0.5  # 0.5 a second: the mark tells which second was used


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.no_open_risk_limit
def test_a_position_closed_before_settlement_pays_nothing_in_paper_or_backtest(tmp_path, side):
    params = {"open_at": ns("2025-10-03 07:52"), "close_at": ns("2025-10-03 07:59"), "side": side, **PERP}
    pf, pfund, bf, bfund = paper_and_backtest(tmp_path, "2025-10-03 07:50", 20, _ramp(20), params)
    assert pf[-1][0] < utc("2025-10-03 08:00") and bf[-1][0] < utc("2025-10-03 08:00")
    assert pfund == [] and bfund == []


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.no_open_risk_limit
def test_a_position_opened_at_or_after_settlement_pays_nothing_in_paper_or_backtest(tmp_path, side):
    params = {"open_at": ns("2025-10-03 08:00"), "close_at": ns("2025-10-03 08:05"), "side": side, **PERP}
    pf, pfund, bf, bfund = paper_and_backtest(tmp_path, "2025-10-03 07:50", 20, _ramp(20), params)
    assert pf[0][0] >= utc("2025-10-03 08:00") and bf[0][0] >= utc("2025-10-03 08:00")
    assert pfund == [] and bfund == []


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("close_at", ["2025-10-03 08:00", "2025-10-03 08:05"])  # exit on the settlement bar, or later
@pytest.mark.no_open_risk_limit
def test_a_position_held_across_pays_rate_times_notional_once_longs_pay_shorts_receive(tmp_path, side, close_at):
    params = {"open_at": ns("2025-10-03 07:52"), "close_at": ns(close_at), "side": side, **PERP}
    pf, pfund, bf, bfund = paper_and_backtest(tmp_path, "2025-10-03 07:50", 20, _ramp(20), params)
    for fund in (pfund, bfund):
        (ts, qty, price, amount), = fund
        assert ts == utc("2025-10-03 08:00") and np.sign(qty) == side
        assert price == pytest.approx(60_299.5)  # the last trade at or before 08:00:00 (07:59:59)
        assert amount == pytest.approx(-qty * price * RATE)
        assert (amount < 0) == (side > 0)  # positive rate: the long pays, the short receives
    assert [r[1:] for r in pfund] == pytest.approx([r[1:] for r in bfund])  # paper and backtest alike


def test_paper_charges_a_long_held_at_settlement_and_stopped_out_five_seconds_later(tmp_path):
    """3-hour decision bars (08:00 is not a bar close), so only paper's 30-second tick settles funding. Long from the
    06:00 bar, stop hit at 08:00:05. The recording starts at 03:00:13, so the 06:00 bar is whole (a part bar is
    degraded, board 5a) and ticks fall at :13 and :43."""
    start, minutes = "2025-10-03 03:00:13", 304
    t = utc(start) + pd.to_timedelta(np.arange(minutes * 60), unit="s")
    prices = np.where(t < utc("2025-10-03 08:00:05"), 60_000 + np.arange(minutes * 60) * 0.01, 58_800.0)
    params = {"open_at": ns("2025-10-03 06:00"), "close_at": ns("2025-10-03 12:00"), "side": 1, "stop_loss": 0.005,
              **PERP}
    pf, pfund, bf, bfund = paper_and_backtest(tmp_path, start, minutes, prices, params, bar_minutes=180)
    assert pf[-1][0] == utc("2025-10-03 08:00:05")  # held at 08:00:00, closed after it
    assert [r[0] for r in bfund] == [utc("2025-10-03 08:00")]  # the backtest charges it
    assert [r[0] for r in pfund] == [utc("2025-10-03 08:00")]  # paper charges it too (P1-O2 fixed)
    assert [r[1:] for r in pfund] == pytest.approx([r[1:] for r in bfund])  # the same position, price and amount


@pytest.fixture
def binance_funding_8h_then_4h(tmp_path, monkeypatch):
    """Settled rates every 8 h to 4 Oct 16:00, then every 4 h from 20:00 (the venue changed the interval)."""
    from test_binance import INFO

    monkeypatch.setattr(funding, "DEFAULT_ROOT", tmp_path)
    b = venue("binance")
    monkeypatch.setattr(b, "contract", lambda pair: binance_contract(pair, get_json=lambda url: INFO))
    times = (pd.date_range("2025-10-03 00:00", "2025-10-04 16:00", freq="8h", tz="UTC")
             .append(pd.date_range("2025-10-04 20:00", "2025-10-06 00:00", freq="4h", tz="UTC")))
    rates = [[int(t.timestamp() * 1000), 0.0001 * (1 + i % 3)] for i, t in enumerate(times)]
    p = funding._path("BINANCE", "BTC/USDT")
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"rates": rates}))
    idx = pd.date_range("2025-10-03 00:00", "2025-10-06 00:00", freq="1h", tz="UTC")[1:]
    c = 60_000.0 + np.arange(len(idx))
    bars = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e9}, index=idx)
    return b.instrument("BTC", "USDT"), bars, times, dict(zip(times, (r for _, r in rates)))


def _held_throughout(inst, bars, side):
    params = {"open_at": ns("2025-10-03 01:00"), "close_at": ns("2030-01-01"), "side": side, **PERP}
    r = run_backtest("win", bars, inst, params, starting_capital=10_000, risk_profile="balanced", bar_minutes=60,
                     half_spread=0)
    return r.journal.funding_


@pytest.mark.parametrize("side", [1, -1])
def test_no_settlement_is_charged_twice_at_an_interval_change_and_each_at_its_own_rate(binance_funding_8h_then_4h, side):
    inst, bars, times, rate = binance_funding_8h_then_4h
    rows = _held_throughout(inst, bars, side)
    ts = [pd.Timestamp(r["ts"]) for r in rows]
    assert len(ts) == len(set(ts)) and set(ts) <= set(times)
    for r in rows:
        assert r["rate"] == pytest.approx(rate[pd.Timestamp(r["ts"])])  # the rate in force at that settlement
        assert r["amount"] == pytest.approx(-r["qty"] * r["price"] * r["rate"], rel=1e-6)


@pytest.mark.parametrize("side", [1, -1])
def test_after_a_change_from_8h_to_4h_every_new_settlement_is_charged(binance_funding_8h_then_4h, side):
    inst, bars, times, _ = binance_funding_8h_then_4h
    rows = _held_throughout(inst, bars, side)
    owed = [t for t in times if utc("2025-10-03 01:00") < t <= bars.index[-1]]
    charged = [pd.Timestamp(r["ts"]) for r in rows]
    assert len(owed) == 13 and charged == owed


def test_settlement_times_follow_the_venues_records_and_carry_its_latest_interval_on():
    """Before the first record: the fixed hours. Between records: the venue's own times. Past the newest: its
    latest interval, so paper charges a settlement the venue hasn't published yet at the right time."""
    t = lambda s: utc(s).to_pydatetime()  # noqa: E731
    rec = pd.Series(0.0001, index=pd.DatetimeIndex([utc("2025-10-04 08:00:00.004"), utc("2025-10-04 16:00"),
                                                    utc("2025-10-04 20:00")]))
    got = markets.settlement_times(t("2025-10-03 23:00"), t("2025-10-05 05:00"), (0, 8, 16), rec)
    assert got == [t("2025-10-04 00:00"), t("2025-10-04 08:00"), t("2025-10-04 16:00"), t("2025-10-04 20:00"),
                   t("2025-10-05 00:00"), t("2025-10-05 04:00")]
    assert markets.settlement_times(t("2025-10-04 08:00"), t("2025-10-04 16:00"), (0, 8, 16), rec) == [
        t("2025-10-04 16:00")]  # (after, until]: a settlement already charged isn't charged again
    assert markets.settlement_times(t("2025-10-04 01:00"), t("2025-10-04 17:00"), (0, 8, 16), None) == [
        t("2025-10-04 08:00"), t("2025-10-04 16:00")]


def test_a_foreseen_settlement_the_venue_skips_is_not_a_settlement_once_a_newer_record_lands():
    """4-hourly records, then the venue goes back to 8-hourly: 04:00 was foreseen from the 4 h step, but once the
    08:00 record is kept it is no settlement (CR minor 1: no phantom charge). A single record (a new listing)
    falls back to the fixed hours after it."""
    t = lambda s: utc(s).to_pydatetime()  # noqa: E731
    four = pd.Series(0.0001, index=pd.DatetimeIndex([utc("2025-10-04 20:00"), utc("2025-10-05 00:00")]))
    assert markets.settlement_times(t("2025-10-05 00:00"), t("2025-10-05 05:00"), (0, 8, 16), four) == [
        t("2025-10-05 04:00")]  # foreseen: paper waits for its record
    eight = pd.concat([four, pd.Series(0.0001, index=pd.DatetimeIndex([utc("2025-10-05 08:00")]))])
    assert markets.settlement_times(t("2025-10-05 00:00"), t("2025-10-05 09:00"), (0, 8, 16), eight) == [
        t("2025-10-05 08:00")]
    one = pd.Series(0.0001, index=pd.DatetimeIndex([utc("2025-10-05 08:00")]))
    assert markets.settlement_times(t("2025-10-05 07:00"), t("2025-10-06 01:00"), (0, 8, 16), one) == [
        t("2025-10-05 08:00"), t("2025-10-05 16:00"), t("2025-10-06 00:00")]


def test_paper_waits_a_foreseen_settlement_until_a_newer_record_could_drop_it():
    """CR minor 1: a time foreseen from the 4 h step waits one more interval than a published one, so the venue's
    08:00 record (back to 8-hourly) lands and drops 04:00 before any baseline is charged for it."""
    from datetime import timedelta

    t = lambda s: utc(s).to_pydatetime()  # noqa: E731
    four = pd.Series(0.0001, index=pd.DatetimeIndex([utc("2025-10-04 20:00"), utc("2025-10-05 00:00")]))
    wait = timedelta(minutes=15)
    assert markets.settlement_wait(t("2025-10-05 04:00"), four, wait) == timedelta(hours=4, minutes=30)  # and the store's refresh after the 08:00 record
    assert markets.settlement_wait(t("2025-10-05 00:00"), four, wait) == wait  # a recorded time: the usual wait
    assert markets.settlement_wait(t("2025-10-05 04:00"), None, wait) == wait  # no records: the fixed hours'


def test_a_fill_on_a_gap_pays_the_settlements_held_through_before_it(prices, instrument):
    """CR minor 2, then the Advisor's rule (c) (QA P1-D9): daily bars, a short stopped out by a gap at the next bar's
    open. A bars-only fill on a gap took the open's price, so it is held to the open: the settlement at the open is
    settled before the fill is booked, and none inside the bar is (the short would have received them). The trades'
    P&L still adds up to equity."""
    from sleeve_fund.research.metrics import fills_to_rows, trades
    from test_long_short import _gapped

    closes = [100.0, 100.5, 100.8, 101.5, 101.5, 400.0, 400.0, 400.0]
    res = run_backtest("ping_pong", _gapped(prices, closes), instrument, PERP, half_spread=0, risk_profile="aggressive")
    shut = pd.Timestamp(res.fills.sort_values("ts_last")["ts_last"].iloc[-1])
    charged = {pd.Timestamp(f["ts"]): f["amount"] for f in res.funding}
    opened = shut - pd.Timedelta(days=1)
    assert opened in charged and charged[opened] > 0  # held to the open; a short receives at a positive rate
    assert not [ts for ts in charged if opened < ts <= shut]  # none inside the bar it gapped out in
    trips = trades(fills_to_rows(res.fills), True, res.funding, res.insurance)
    assert sum(t["pnl"] for t in trips) == pytest.approx(res.equity.iloc[-1] - res.starting_capital, abs=0.05)
