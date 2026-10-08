"""P2-7 M4: the portfolio backtest runner. Several Strategies in one engine on venue clones, P2-2's gate over one
MemoryLedger, the fund marked every hour, and the Independent Quant Advisor's rulings on its risk choices (8 Oct 03:05
UK, R1 to R5), each pinned here."""

from decimal import Decimal as D

import numpy as np
import pandas as pd
import pytest
from nautilus_trader.model import Venue

from sleeve_fund import open_risk
from sleeve_fund.data import validate_ohlcv
from sleeve_fund.instruments import FeeSchedule, spot_pair
from sleeve_fund.research import portfolio_run
from sleeve_fund.research.portfolio_run import LegSpec, run_portfolio
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.risk import PortfolioProfile

FEES = FeeSchedule(D("0.001"), D("0.002"))
LOOSE = PortfolioProfile(gross=100, net_instrument=100, margin=100, open_risk=100, drawdown=0.99, daily_loss=0.99)
PERP = {"market": "perp", "allow_short": True}
COLS = ["side", "filled_qty", "avg_px"]


def _hourly(days: int, seed: int, vol: float = 0.007, minutes: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = days * 24 * 60 // minutes
    close = 10_000 * np.exp(np.cumsum(rng.normal(0, vol * (minutes / 60) ** 0.5, n)))
    open_ = np.concatenate([[10_000], close[:-1]])
    wick = np.abs(rng.normal(0, vol / 2, n))
    step = pd.Timedelta(minutes=minutes)
    idx = pd.date_range("2018-01-01", periods=n, freq=step, tz="UTC") + step
    return validate_ohlcv(pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * (1 + wick),
                                        "low": np.minimum(open_, close) * (1 - wick), "close": close,
                                        "volume": 1_000.0}, index=pd.DatetimeIndex(idx, name="timestamp")))


def _daily(h: pd.DataFrame) -> pd.DataFrame:
    day = (h.index - pd.Timedelta(1, "ns")).floor("D") + pd.Timedelta("1D")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return validate_ohlcv(h.groupby(day).agg(agg).rename_axis("timestamp"))


def _spec(name: str, seed: int, params: dict | None = None, days: int = 140, warm: int = 20, model: str = "rsi_cross",
          **kw) -> LegSpec:
    h = _hourly(days, seed)
    d = _daily(h)
    w, window = d.iloc[:warm], d.iloc[warm:]
    inst = spot_pair("BTC", "USD", FEES, Venue("KRAKEN"))
    return LegSpec(name, model, inst, window, h[h.index > w.index[-1]], w, params=dict(params or {}), **kw)


def _alone(s: LegSpec):
    return run_backtest(s.strategy, s.prices, s.instrument, params=s.params, starting_capital=s.capital,
                        exec_prices=s.exec_prices, exec_minutes=s.exec_minutes, warmup_prices=s.warmup_prices)


def _same_fills(a, b) -> bool:
    return a.fills[COLS].reset_index(drop=True).equals(b.fills[COLS].reset_index(drop=True))


BUSY = {"model": "ping_pong"}  # trades every few days, so the pause and the halt have entries to act on
HALT_AT_8 = PortfolioProfile(drawdown=0.08, daily_loss=0.99)  # the halt's level, low enough to fire in this window


def _halted():
    return run_portfolio([_spec("a", 2, **BUSY), _spec("b", 10, PERP, **BUSY)], HALT_AT_8)


@pytest.fixture(scope="module")
def halted():
    return _halted()


def test_with_room_to_spare_each_strategy_trades_exactly_as_it_does_alone():
    specs = [_spec("a", 2), _spec("b", 10, PERP)]
    run = run_portfolio(specs, LOOSE)
    for s in specs:
        alone, joined = _alone(s), run.legs[s.name]
        assert alone.fills is not None and len(alone.fills), "the model must trade for this to test anything"
        assert _same_fills(joined, alone)
        # A gated order fills at its close + 1 ns, so the close it was decided on is marked before the fill (P2-7a
        # trap 2: parity keys on the bar); every other close is the same.
        filled = set(pd.to_datetime(alone.fills["ts_last"]))
        keep = [t for t in alone.equity.index if t not in filled]
        assert joined.equity[keep].round(6).equals(alone.equity[keep].round(6))
    assert run.halt is None and not run.data_gaps
    assert all(d["outcome"] == "approved" for d in run.decisions) and run.decisions


def test_the_book_is_the_whole_fund_unallocated_cash_included():
    run = run_portfolio([_spec("a", 2), _spec("b", 10, PERP)], LOOSE, unallocated=5_000.0)
    assert run.book.iloc[0] == 25_000.0


# --- R1: an hourly mark grid, whatever the Strategies trade -------------------------------------------------------

def test_r1_the_book_is_marked_every_hour_though_every_strategy_decides_daily():
    s = _spec("a", 2)
    run = run_portfolio([s], LOOSE)
    hours = pd.date_range(s.exec_prices.index[0], s.exec_prices.index[-1], freq="1h")
    assert run.book.index.equals(hours)  # not just the daily closes, nor only the closes with an entry


def test_r1_a_run_without_bars_of_an_hour_or_less_is_refused():
    s = _spec("a", 2)
    with pytest.raises(ValueError, match="every hour"):
        run_portfolio([LegSpec(**{**s.__dict__, "exec_minutes": 240})], LOOSE)
    with pytest.raises(ValueError, match="every hour"):
        run_portfolio([LegSpec(**{**s.__dict__, "exec_prices": s.exec_prices.iloc[:0]})], LOOSE)


def _hours(lows: list[float], closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2018-01-02 01:00", periods=len(lows), freq="1h", tz="UTC")
    return pd.DataFrame({"high": [c * 1.01 for c in closes], "low": lows, "close": closes}, index=idx)


def test_r1_a_cross_inside_an_hour_that_no_mark_saw_is_counted_as_intrabar_halt_risk():
    h = _hours([100, 80, 100], [100, 100, 100])  # the second hour trades 20% down and closes where it opened
    marks = {int(t.value): (100.0, [("a", 1.0, 100.0)]) for t in h.index}
    book = pd.Series(100.0, index=h.index)
    worst, risk = portfolio_run._intrabar(book, marks, {"a": h}, PortfolioProfile(), None, [])
    assert worst == pytest.approx(0.2) and risk == {"halt": 1, "pause": 1}
    h2 = _hours([100, 95, 100], [100, 100, 100])  # 5% inside the hour: the pause only
    worst, risk = portfolio_run._intrabar(book, marks, {"a": h2}, PortfolioProfile(), None, [])
    assert worst == pytest.approx(0.05) and risk == {"halt": 0, "pause": 1}
    seen = [{"ts": h.index[1], "reason": "x"}]  # a mark that day did pause: no intrabar risk left to report
    assert portfolio_run._intrabar(book, marks, {"a": h2}, PortfolioProfile(), None, seen)[1]["pause"] == 0


def test_r1_the_result_reports_the_worst_intrabar_drawdown_beside_the_marks(halted):
    marked = float((1 - halted.book / halted.book.cummax()).max())
    assert halted.worst_intrabar_drawdown >= marked - 1e-9 and set(halted.intrabar_halt_risk) == {"halt", "pause"}


# --- R2 and R3: the halt flattens every Strategy, at the next tradable price, and the fund stays flat -------------

def test_r2_a_halt_flattens_every_strategy_in_one_pass_and_its_cost_is_its_own_line(halted):
    assert halted.halt is not None and halted.halt["drawdown"] >= 0.08
    at = halted.halt["ts"]
    flattens = []
    for name, r in halted.legs.items():
        for oid, row in r.fills.iterrows():
            d = r.decisions[oid]
            if d["intent"] == "risk_halt" and d["reason"].startswith(portfolio_run.HALTED):
                flattens.append((name, oid, row))
    assert flattens
    for r in halted.legs.values():  # every Strategy holding anything at the halt is flat after it
        held = r.fills[pd.to_datetime(r.fills["ts_last"]) <= at + pd.Timedelta(1, "ns")]
        signed = sum((1 if str(x["side"]).endswith("BUY") else -1) * float(x["filled_qty"]) for _, x in held.iterrows())
        assert abs(signed) < 1e-9
    assert all(pd.Timestamp(row["ts_last"]) - at <= pd.Timedelta(1, "ns") for _, _, row in flattens)
    assert halted.halt_flatten_cost > 0
    fees = sum(float(str(row["commissions"][0] if isinstance(row["commissions"], list) else row["commissions"])
                     .split()[0]) for _, _, row in flattens)
    assert halted.halt_flatten_cost >= fees  # fees and the half spread


def test_r3_after_a_halt_the_fund_stays_flat_and_every_figure_covers_the_whole_window(halted):
    at = halted.halt["ts"]
    later = [d for d in halted.decisions if pd.Timestamp(d["ts"]) > at]
    assert later and all(d["outcome"] == "rejected" and d["limit"] == "halt" for d in later)
    assert sum(by.get("halt", 0) for by in halted.refusals.values()) == len(later)
    for r in halted.legs.values():
        assert r.equity.index[-1] == _spec("a", 2, **BUSY).prices.index[-1]  # never cut at the halt
        assert all(pd.Timestamp(t) <= at + pd.Timedelta(1, "ns") for t in r.fills["ts_last"])
    assert halted.book.index[-1] == _spec("a", 2, **BUSY).exec_prices.index[-1]
    assert halted.book[halted.book.index > at].nunique() == 1  # flat: nothing moves the book after the flatten


# --- R4: the daily pause flattens nothing -----------------------------------------------------------------------

def _hourly_spec(name: str, seed: int, params: dict | None = None, days: int = 60, warm: int = 20) -> LegSpec:
    """ping_pong deciding every hour on half-hour bars, so entries fall inside a day's pause."""
    h = _hourly(days, seed, minutes=30)
    hour = (h.index - pd.Timedelta(1, "ns")).floor("h") + pd.Timedelta("1h")
    d = validate_ohlcv(h.groupby(hour).agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                            "volume": "sum"}).rename_axis("timestamp"))
    w, window = d.iloc[:warm * 24], d.iloc[warm * 24:]
    inst = spot_pair("BTC", "USD", FEES, Venue("KRAKEN"))
    return LegSpec(name, "ping_pong", inst, window, h[h.index > w.index[-1]], w, exec_minutes=30, bar_minutes=60,
                   params=dict(params or {}))


def test_r4_a_daily_pause_blocks_entries_until_midnight_and_flattens_nothing():
    run = run_portfolio([_hourly_spec("a", 2), _hourly_spec("b", 10, PERP)],
                        PortfolioProfile(open_risk=1.0, drawdown=0.99, daily_loss=0.02))
    assert run.pauses and run.halt is None
    paused = [d for d in run.decisions if d["limit"] == "pause"]
    assert paused and all(d["outcome"] == "rejected" for d in paused)
    for d in paused:  # each inside a pause's day, before its 00:00 UTC
        at = pd.Timestamp(d["ts"])
        assert any(p["ts"] <= at < p["until"] for p in run.pauses)
    assert any(p["ts"].hour not in (0,) for p in run.pauses)  # found on the hourly marks, not only at a day's close
    for r in run.legs.values():
        assert not [o for o in r.decisions.values() if o["intent"] == "risk_halt"]
        held = [pd.Timestamp(t) for t in r.fills["ts_last"]]
        for p in run.pauses:  # nothing is sold because of the pause itself
            assert not [t for t in held if abs(t - p["ts"]) <= pd.Timedelta(1, "ns")]


# --- R5: an order the gate can't measure is refused, and inside the window it is a data gap -----------------------

def test_r5_a_run_without_fourteen_days_of_warm_up_is_refused():
    s = _spec("a", 2)
    with pytest.raises(ValueError, match="warm-up"):
        run_portfolio([LegSpec(**{**s.__dict__, "warmup_prices": s.warmup_prices.iloc[-10:]})], LOOSE)


def test_r5_with_the_warm_up_the_atr_exists_from_the_windows_first_day():
    run = run_portfolio([_spec("a", 2)], LOOSE)
    assert not run.data_gaps and not [d for d in run.decisions if d["outcome"] == "rejected"]


def test_r5_a_refusal_inside_the_window_for_want_of_a_figure_is_reported_as_a_data_gap(monkeypatch):
    monkeypatch.setattr(open_risk, "daily_atr_lookup", lambda prices: {})  # no ATR at all: stopless can't be measured
    s = _spec("a", 2)
    run = run_portfolio([s], LOOSE)
    assert run.data_gaps and all(g["strategy"] == "a" and "daily ATR" in g["why"] for g in run.data_gaps)
    assert all(pd.Timestamp(g["ts"]) >= s.prices.index[0] for g in run.data_gaps)
    # CR F220-1: counted with the reason, beside the leg's own entry_refused count
    assert run.refusals == {"a": {"unmeasurable": len(run.data_gaps)}}
    assert run.legs["a"].portfolio_gate == {"entry_refused": len(run.data_gaps)}
    assert not run.legs["a"].fills is not None or run.legs["a"].fills.empty


# --- 00:00 UTC (Advisor 8 Oct 03:25 UK, MUST): old day's final mark, then the new day's start, then decisions -------

def test_at_midnight_the_old_day_is_marked_first_so_no_decision_sees_a_stale_daily_start_or_the_old_pause():
    from sleeve_fund.research.portfolio import PortfolioGate

    hour = 3_600_000_000_000
    midnight = int(pd.Timestamp("2018-01-03", tz="UTC").value)
    equity = {midnight - hour: 100, midnight: 97, midnight + hour: 95}  # the book at each hour's close
    gate = PortfolioGate(lambda ts: (D(equity[max(t for t in equity if t <= ts)]), ()), PortfolioProfile())
    gate.mark(midnight - hour + 1)  # 23:00: the old day starts at 100
    order = {"side": 1, "qty": D(1), "price": D(10), "stop_frac": 0.02, "step": D("0.001"), "min_qty": D("0.001"),
             "instrument": "BTC", "leverage": None, "atr_pct": None}
    decided = gate("a", order, midnight + 1)  # the gate pass at 00:00 + 1 ns runs before the hourly mark here
    assert decided.decision.outcome == "approved"  # the old day's 3% loss doesn't block the new day
    st = gate.ledger.state()
    assert st.day_start_equity == D(97)  # the new day starts from the 00:00 close, not the 23:00 mark
    assert gate.mark(midnight + hour + 1) is None  # 95 is 2.1% below 97: no pause (5% below a stale 100 would be)


def test_a_run_is_one_trials_row_of_kind_portfolio_run_listing_each_members_trial_id(halted):
    from sleeve_fund.store import check_trial

    rows = halted.trials_rows("synthetic-hourly", member_trial_ids={"a": "t-a"})
    run, *members = rows
    check_trial(run)
    assert run["source"] == "backtest" and run["kind"] == "portfolio_run" and run["family"] == "portfolio"
    listed = {m["name"]: m["trial_id"] for m in __import__("json").loads(run["settings"])["members"]}
    assert listed == {"a": "t-a", "b": members[0]["id"]} and len(members) == 1  # b had no row yet: it gets one
    assert run["sharpe"] == pytest.approx(halted.summary()["sharpe"])
    assert halted.summary()["days"] == len(halted.book.resample("1D").last()) - 1  # the whole window (R3)
