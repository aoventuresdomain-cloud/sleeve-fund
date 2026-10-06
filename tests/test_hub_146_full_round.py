"""QA PR #146, full round on 07d4c83 (Head of QA, 6 Oct 2026), moved ~19:50 to Advisor #146 L12 FINAL (19:30, 19:38,
19:40): a backtest or outage-replay target books at the level less max(half spread, 0.05 %), taker, never the open
(replay row marked modelled); paper live books the real fill and records the level; an exact touch fills. Those
pins are plain: they fail on 9be8f62, which predates the ruling (the bare level). Original scope: NA-2 as built, the Advisor's three conditions on it
(18:55), D13 point 6 (18:36), the adversarial hub probes, an outage across a funding settlement, and a liquidation
found by the outage replay (Advisor 18:17).

Builds on test_hub_146_qa.py (same folder; its probe strategy, hub-fed paper replay and backtest reference). A test
that passes pins what is true on 07d4c83; a finding is a strict xfail whose reason starts with its id (P1-L9..).
Run from the worktree root, as the master file:

  cd /tmp/claude-0/p146r && PYTHONDONTWRITEBYTECODE=1 BACKTEST_ISOLATE=0 PYTHONPATH=/tmp/claude-0/p146r \
    /tmp/claude-0/v/bin/python -m pytest -q -p no:cacheprovider -rxXs \
    /mnt/project-files/sleeve-fund/quant-review/v2-p1/hub-146-scripts/test_hub_146_full_round.py
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import test_hub_146_qa as qa
from test_hub_146_qa import (  # noqa: F401 - _probe is the autouse fixture that registers the probe strategy
    BASE, M, S, SPREAD, START, TP_SLIP, _probe, adverse, favourable, first_exit, flat_prices, is_modelled, minute,
    minutes_of, paper, recover_from, shape, tp_model)

HALF = SPREAD / 2 / BASE
CENT = 0.0100001

SETUPS = [("spot-long", False, "aggressive", 1), ("perp-1x-long", True, "conservative", 1),
          ("perp-3x-long", True, "aggressive", 1), ("perp-3x-short", True, "aggressive", -1)]
IDS = [s[0] for s in SETUPS]
PERPS = [s for s in SETUPS if s[1]]
PERP_IDS = [s[0] for s in PERPS]


def bt_res(prices, *, enter=5, leave=60, side=1, minutes=1, perp=False, profile="aggressive", stop=0.01, tp=0.02,
           extra=None, after=0):
    """run_backtest on the minutes of `prices` (as qa.backtest), returning the result itself."""
    from sleeve_fund.instruments import BOOK_SHARE
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.venues import venue

    inst = venue("KRAKEN").instrument("BTC", "USD", price_precision=1)
    params = {"enter": enter, "leave": leave, "side": side, "after": after, **(extra or {})}
    if stop:
        params["stop_loss"] = stop
    if tp:
        params["take_profit"] = tp
    if perp:
        params.update(market="perp", allow_short=True)
    m = minutes_of(prices)
    bars = m.set_axis(m.index + pd.Timedelta(minutes=1))
    bars["volume"] = 60 / BOOK_SHARE
    kw = {}
    if minutes > 1:
        dec = bars.resample(f"{minutes}min", label="right", closed="right").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
        kw = {"exec_prices": bars, "exec_minutes": 1}
        bars = dec
    res = run_backtest("probe", bars, inst, params=params, starting_capital=10_000, risk_profile=profile,
                       bar_minutes=minutes, half_spread=HALF, **kw)
    return res, params


def orders_of(res):
    j = res.journal
    orders = list(j.orders_.values()) if isinstance(j.orders_, dict) else list(j.orders_)
    return sorted(orders, key=lambda o: (o["ts"], o["id"]))


def fills_of(res, order_id):
    return [f for f in res.journal.fills_ if f["order_id"] == order_id]


def bt_seq(res):
    by = {}
    for f in res.journal.fills_:
        q, n, ts = by.get(f["order_id"], (0.0, 0.0, f["ts"]))
        by[f["order_id"]] = (q + f["qty"], n + f["qty"] * f["price"], ts)
    intents = {o["order_id"]: (o["intent"], o["side"]) for o in orders_of(res)}
    out = []
    for oid, (q, n, ts) in by.items():
        t = pd.Timestamp(ts).tz_convert(timezone.utc)
        out.append((*intents[oid], t if t == t.floor("1min") else t.ceil("1min"), n / q, q))
    return sorted(out, key=lambda r: r[2])


def commission(rep) -> float:
    """The fills report's commission for one order: a list of quote-currency Money strings ("12.34 USD")."""
    c = rep["commissions"]
    return sum(float(str(m).split()[0]) for m in (c if isinstance(c, (list, tuple)) else [c]))


def wick(side, k, depth=0.03, width=15 / 60):
    """Flat, with a wick `depth` in the position's favour for `width` minutes from minute k, then back."""
    return shape(flat_prices(40), k, k + width, favourable(side, depth))


# ================================================== NA-2 as built: a booked target, the report, journal and cash


@pytest.mark.parametrize("label, perp, profile, side", SETUPS, ids=IDS)
def test_na2_a_booked_target_agrees_to_the_cent_in_report_journal_and_cash(label, perp, profile, side):
    """A wick 3 % in favour through the 2 % target inside the minute to 00:11, back at the entry by its close. The
    target is booked at its level less max(half spread, 0.05 %) (Advisor L12 FINAL 19:30; it was the bare level),
    the taker fee through the fee model; the report and the journal show the same price and fee, and the run's final
    equity is the journal's cash to the cent."""
    from sleeve_fund import markets
    from sleeve_fund.store import replay_book
    from sleeve_fund.venues import venue

    res, params = bt_res(wick(side, 10.4), side=side, perp=perp, profile=profile, leave=30)
    orders = orders_of(res)
    tps = [o for o in orders if o["intent"] == "take_profit"]
    assert len(tps) == 1 and [o["intent"] for o in orders].count("exit") == 0, [o["intent"] for o in orders]
    tp, entry = tps[0], next(o for o in orders if o["intent"] == "entry")
    (ef,) = fills_of(res, entry["order_id"])
    (tf,) = fills_of(res, tp["order_id"])
    level = round(ef["price"] * (1 + side * 0.02), 1)
    booked = tp_model(level, side)
    assert tf["price"] == pytest.approx(booked, abs=0.1), (tf, booked)  # to the tick
    assert pd.Timestamp(tf["ts"]) == minute(11)  # stamped at the minute it traded in
    taker = float(markets.fees_for(params, venue("KRAKEN").fees).taker)
    qty = tf["qty"]
    assert tf["fee"] == pytest.approx(qty * tf["price"] * taker, abs=CENT)  # taker alone: the slippage is in the price
    rep = res.fills.loc[tp["order_id"]]
    assert float(rep["avg_px"]) == pytest.approx(tf["price"], abs=CENT / qty)
    assert commission(rep) == pytest.approx(tf["fee"], abs=CENT)
    # the report's proceeds (price x qty less commission) equal the journal's to the cent
    sign = -side  # the closing fill's side: a sell closes a long
    j_cash = -sign * qty * tf["price"] - tf["fee"]
    r_cash = -sign * qty * float(rep["avg_px"]) - commission(rep)
    assert r_cash == pytest.approx(j_cash, abs=2 * CENT)
    book = replay_book(res.journal.fills_, 10_000, res.journal.funding_total("backtest"),
                       res.journal.insurance_total("backtest"))
    pnl = side * qty * (tf["price"] - ef["price"]) - ef["fee"] - tf["fee"]
    assert float(res.equity.iloc[-1]) == pytest.approx(10_000 + pnl, abs=2 * CENT)
    assert book["qty"] == pytest.approx(0, abs=1e-12)
    assert book["cash"] == pytest.approx(float(res.equity.iloc[-1]), abs=2 * CENT)


@pytest.mark.parametrize("label, perp, profile, side", SETUPS, ids=IDS)
def test_l20_a_booked_target_uses_max_half_spread_or_5bp_as_taker_slippage(label, perp, profile, side):
    """P1-L20 flipped by Advisor L12 FINAL (19:30, supersedes 19:00 "TP pays no half spread"): a backtest target fills
    at the level less max(half spread, 0.05 %), the taker fee on that price, no half spread charged on top."""
    from sleeve_fund import markets
    from sleeve_fund.venues import venue

    res, params = bt_res(wick(side, 10.4), side=side, perp=perp, profile=profile, leave=30)
    tp = next(o for o in orders_of(res) if o["intent"] == "take_profit")
    (tf,) = fills_of(res, tp["order_id"])
    taker = float(markets.fees_for(params, venue("KRAKEN").fees).taker)
    level = Price_level(res, side)
    assert TP_SLIP == 0.0005  # this harness's half spread is 0.01 %, so the 0.05 % floor applies
    assert tf["price"] == pytest.approx(level * (1 - side * TP_SLIP), abs=0.1), (tf["price"], level)
    assert tf["fee"] == pytest.approx(tf["qty"] * tf["price"] * taker, abs=CENT)
    assert float(res.fills.loc[tp["order_id"]]["avg_px"]) == pytest.approx(tf["price"], abs=CENT / tf["qty"])


def Price_level(res, side, tp=0.02):
    """The target's level from the entry's journal fill, to the tick."""
    entry = next(o for o in orders_of(res) if o["intent"] == "entry")
    (ef,) = fills_of(res, entry["order_id"])
    return round(ef["price"] * (1 + side * tp), 1)


@pytest.mark.parametrize("label, perp, profile, side", SETUPS, ids=IDS)
def test_na2_a_minute_touching_stop_and_target_takes_the_stop_whichever_came_first(label, perp, profile, side):
    """Target first (3 % in favour 00:10:05-00:10:15), then the stop (2 % against 00:10:30-00:10:40), the open nearer
    the target (+1.8 %): the stop. (The master file's L7 pins this too.) Its price moved to the D13 pin below
    (Advisor 20:39: the backtest books a stop at its level less max(half spread, 0.05 %))."""
    seq = bt_seq(bt_res(_stop_after_target(side), side=side, perp=perp, profile=profile, leave=30)[0])
    ex = first_exit(seq)
    assert ex[0] == "stop_loss" and ex[2] == minute(11), seq
    assert [r[0] for r in seq].count("take_profit") == 0


def _stop_after_target(side):
    p = flat_prices(40).copy()
    p[600:660] = BASE * favourable(side, 0.018)
    p[605:615] = BASE * favourable(side, 0.03)
    p[630:640] = BASE * adverse(side, 0.02)
    return p


@pytest.mark.xfail(strict=True, reason="NA-1/D13 (Advisor 20:39): the backtest books a stop at its level less "
                   "max(half spread, 0.05 %); stop slippage lands with D13, which stacks on #146: not built yet")
@pytest.mark.parametrize("label, perp, profile, side", SETUPS, ids=IDS)
def test_d13_the_backtest_books_the_stop_at_its_level_less_the_slippage_floor(label, perp, profile, side):
    seq = bt_seq(bt_res(_stop_after_target(side), side=side, perp=perp, profile=profile, leave=30)[0])
    ex, level = first_exit(seq), seq[0][3] * (1 - side * 0.01)
    assert ex[0] == "stop_loss" and abs(ex[3] / qa.tp_model(level, side) - 1) < 1e-4, (ex, qa.tp_model(level, side))


def _open_past_target_then_stop(side, k=10):
    """The minute k..k+1 opens 2.5 % in favour (past the 2 % target) and then trades 1.7 % against (through the 1 %
    stop), where it closes; flat after."""
    p = flat_prices(40).copy()
    p[k * 60:k * 60 + 30] = BASE * favourable(side, 0.025)
    p[k * 60 + 30:(k + 1) * 60] = BASE * adverse(side, 0.017)
    return p


@pytest.mark.parametrize("label, perp, profile, side", SETUPS, ids=IDS)
def test_l9_a_minute_opening_past_the_target_books_the_target_at_its_trigger_not_the_stop(label, perp, profile, side):
    """P1-L9 (D13 18:36 pt 6, NA-2 cond. 3): the open through the target takes the target first, the stop never.
    Mark removed ~19:50: the order is fixed on 9be8f62 (ScheduleFeeModel.open_targets + _rebook_as_target, checked in
    the code); the price is now L12 FINAL's level less TP_SLIP, never the open (9be8f62 still books the bare level)."""
    res, _ = bt_res(_open_past_target_then_stop(side), side=side, perp=perp, profile=profile, leave=30)
    seq = bt_seq(res)
    ex = first_exit(seq)
    entry = seq[0][3]
    assert ex[0] == "take_profit" and abs(ex[3] / tp_model(entry * (1 + side * 0.02), side) - 1) < 1e-4, seq  # L12 FINAL


@pytest.mark.parametrize("label, perp, profile, side", [SETUPS[0], SETUPS[3]], ids=[IDS[0], IDS[3]])
def test_d13_with_1m_execution_bars_a_15m_bar_opening_past_the_target_books_the_target(label, perp, profile, side):
    """Where the 1-minute data shows the order (a 15m decision bar, its first minute opens past the target, the stop
    is reached minutes later), the backtest books the target at its trigger: the residual is only the 1-minute bar
    that holds both."""
    p = flat_prices(60).copy()
    p[15 * 60:16 * 60] = BASE * favourable(side, 0.025)
    p[18 * 60:19 * 60] = BASE * adverse(side, 0.017)
    res, _ = bt_res(p, side=side, perp=perp, profile=profile, leave=50, minutes=15, enter=5)
    seq = bt_seq(res)
    ex = first_exit(seq)
    assert ex[0] == "take_profit" and ex[2] == minute(16), seq
    assert abs(ex[3] / tp_model(seq[0][3] * (1 + side * 0.02), side) - 1) < 1e-4, seq  # L12 FINAL: never the open


@pytest.mark.parametrize("label, perp, profile, side", SETUPS, ids=IDS)
def test_paper_replay_of_a_minute_opening_past_the_target_takes_the_target(label, perp, profile, side):
    """The same minute missed by the hub (away 00:06-00:12) and replayed in paper: the target first (D13 pt 6), at the
    level less TP_SLIP, the row marked as a modelled price (L12 FINAL 19:30: no real fill in a replay)."""
    p = _open_past_target_then_stop(side, k=7)
    p[8 * 60:] = BASE * (1 + 1e-7 * np.arange(8 * 60, 40 * 60))
    run = paper(p, side=side, perp=perp, profile=profile, tp=0.02, leave=25, gone=qa.GONE, away=qa.AWAY,
                back_at=qa.BACK)
    ex = first_exit(run.sequence())
    entry = qa.fill_px(run, "entry")
    assert ex[0] == "take_profit" and abs(ex[3] / tp_model(entry * (1 + side * 0.02), side) - 1) < 1e-4, run.sequence()
    (o,) = [o for o in run.orders if o["intent"] == "take_profit"]
    assert is_modelled(o["signal"]), o["signal"]


# ======================================= Advisor 18:55 condition 1: exactly one exit with a signal exit or a reversal


@pytest.mark.parametrize("label, perp, profile, side", SETUPS, ids=IDS)
def test_c1_the_target_on_the_bar_of_a_signal_exit_is_the_only_exit(label, perp, profile, side):
    """The signal leaves at the bar closing 00:11 (leave=11) and the target trades inside that bar: one closing
    order, the target, and the position ends flat (never the other side)."""
    res, _ = bt_res(wick(side, 10.4), side=side, perp=perp, profile=profile, leave=11)
    seq = bt_seq(res)
    closes = [r for r in seq if r[0] != "entry"]
    assert [r[0] for r in closes] == ["take_profit"], seq
    assert sum(r[4] for r in closes) == pytest.approx(seq[0][4]), seq  # sold exactly what was bought
    assert len([r for r in seq if r[0] == "entry"]) == 1, seq


@pytest.mark.parametrize("label, perp, profile, side", PERPS, ids=PERP_IDS)
def test_c1_the_target_on_the_bar_of_a_reversal_is_the_only_close_and_the_new_side_is_sized_whole(label, perp, profile,
                                                                                                  side):
    """The signal turns to the other side at the bar closing 00:11 (after=-side) and the target trades inside it: one
    close (the target), and the new side opens once, at the size an entry from flat gets at that equity."""
    res, _ = bt_res(wick(side, 10.4), side=side, perp=perp, profile=profile, leave=11, after=-side)
    seq = bt_seq(res)
    orders = orders_of(res)
    closes = [r for r in seq if r[0] != "entry"]
    assert [r[0] for r in closes] == ["take_profit"], seq
    entries = [o for o in orders if o["intent"] == "entry"]
    assert len(entries) == 2, seq
    new = entries[1]
    assert new["side"] == ("SELL" if side > 0 else "BUY")
    # never netted through: the first close took the first entry's whole quantity, no more
    assert closes[0][4] == pytest.approx(seq[0][4])
    sig = new["signal"]
    from sleeve_fund.venues import venue

    lot = float(venue("KRAKEN").instrument("BTC", "USD", price_precision=1).size_increment)
    assert new["qty"] == pytest.approx(sig["budget"] / sig["close"], abs=lot + 5e-6 * new["qty"]), (new["qty"], sig)
    first = [o for o in orders if o["intent"] == "entry"][0]
    assert sig["sized_by"] == first["signal"]["sized_by"], (sig, first["signal"])  # the same limit, from flat


@pytest.mark.parametrize("label, perp, profile, side", PERPS, ids=PERP_IDS)
def test_l10_a_reversal_on_the_targets_bar_opens_the_new_side_on_the_same_bar_as_paper(label, perp, profile, side):
    res, _ = bt_res(wick(side, 10.4), side=side, perp=perp, profile=profile, leave=11, after=-side)
    bt_new = [r for r in bt_seq(res) if r[0] == "entry"][1]
    run = paper(wick(side, 10.4), side=side, perp=perp, profile=profile, tp=0.02, leave=11,
                extra={"after": -side})
    pp_new = [r for r in run.sequence() if r[0] == "entry"][1]
    assert bt_new[2] == pp_new[2], (bt_new, pp_new)


# ================================================ condition 2: the target's fee is paper's order type (taker today)


@pytest.mark.parametrize("label, perp, profile, side", SETUPS, ids=IDS)
def test_c2_the_target_is_a_market_order_in_paper_and_both_charge_the_taker_rate(label, perp, profile, side):
    from sleeve_fund import markets
    from sleeve_fund.venues import venue

    run = paper(wick(side, 10.4), side=side, perp=perp, profile=profile, tp=0.02, leave=30)
    (o,) = [o for o in run.orders if o["intent"] == "take_profit"]
    assert o["order_type"] == "MARKET", o
    (f,) = [f for f in run.fills if f["order_id"] == o["order_id"]]
    params = {"market": "perp"} if perp else {}
    taker = float(markets.fees_for(params, venue("KRAKEN").fees).taker)
    assert f["fee"] == pytest.approx(f["qty"] * f["price"] * taker, abs=CENT)
    res, _ = bt_res(wick(side, 10.4), side=side, perp=perp, profile=profile, leave=30)
    tp = next(o for o in orders_of(res) if o["intent"] == "take_profit")
    (tf,) = fills_of(res, tp["order_id"])
    rep = res.fills.loc[tp["order_id"]]
    assert commission(rep) == pytest.approx(float(rep["filled_qty"]) * tf["price"] * taker, abs=CENT)


def records_level(signal: dict | None, level: float) -> bool:
    """Paper's live target row records the target's level (L12 FINAL 19:30), as some number equal to it; the
    price, close and entry fields don't count."""
    return any(isinstance(v, (int, float)) and k not in ("price", "close", "entry_px") and abs(v / level - 1) < 2e-6
               for k, v in (signal or {}).items())


# ============================== condition 4: stamped at the bar it traded in; funding by D9 (no in-bar credit)


def _funding_bars(side, wick_in_bar: bool):
    """3-hour bars from 00:00: flat at 60,000; the bar 06:00-09:00 (it holds the 08:00 settlement) trades 3 % in
    favour (through the 2 % target) when wick_in_bar."""
    idx = pd.date_range("2025-10-03 03:00", periods=8, freq="180min", tz="UTC")
    c = np.full(len(idx), BASE)
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e6}, index=idx)
    if wick_in_bar:
        col = "high" if side > 0 else "low"
        df.loc[pd.Timestamp("2025-10-03 09:00", tz="UTC"), col] = BASE * favourable(side, 0.03)
    return df


def _funding_run(side, wick_in_bar, monkeypatch, rate=0.0001):
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.strategies.base import LongFlatStrategy
    from sleeve_fund.venues import venue

    monkeypatch.setattr(LongFlatStrategy, "_funding_rate", lambda self, terms, ts, now, *_: rate)  # harness: #155 passes the wait too
    inst = venue("KRAKEN").instrument("BTC", "USD", price_precision=1)
    params = {"enter": 180, "leave": 10**6, "side": side, "take_profit": 0.02, "market": "perp",
              "allow_short": True}
    return run_backtest("probe", _funding_bars(side, wick_in_bar), inst, params=params, starting_capital=10_000,
                        risk_profile="aggressive", bar_minutes=180, half_spread=HALF)


@pytest.mark.parametrize("side", [1], ids=["long-pays"])
def test_c4_a_target_in_a_bar_holding_a_settlement_is_charged_it_when_it_is_a_cost(side, monkeypatch):
    """A long with a positive rate pays funding. The target trades somewhere in 06:00-09:00, which holds the 08:00
    settlement: when it traded is unknown, so the cost is charged (the worse outcome, D9). Stamped 09:00."""
    res = _funding_run(side, True, monkeypatch)
    tp = next(o for o in orders_of(res) if o["intent"] == "take_profit")
    (f,) = fills_of(res, tp["order_id"])
    assert pd.Timestamp(f["ts"]) == pd.Timestamp("2025-10-03 09:00", tz="UTC")
    costs = [r for r in res.journal.funding_ if pd.Timestamp(r["ts"]) == pd.Timestamp("2025-10-03 08:00", tz="UTC")]
    assert len(costs) == 1 and costs[0]["amount"] < 0, res.journal.funding_


@pytest.mark.parametrize("side", [-1], ids=["short-receives"])
def test_l11_a_target_in_a_bar_holding_a_settlement_takes_no_in_bar_credit(side, monkeypatch):
    res = _funding_run(side, True, monkeypatch)
    tp = next(o for o in orders_of(res) if o["intent"] == "take_profit")
    (f,) = fills_of(res, tp["order_id"])
    assert pd.Timestamp(f["ts"]) == pd.Timestamp("2025-10-03 09:00", tz="UTC")
    credits = [r for r in res.journal.funding_ if pd.Timestamp(r["ts"]) == pd.Timestamp("2025-10-03 08:00", tz="UTC")]
    assert credits == [], credits


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_c4_with_no_target_the_08_00_settlement_is_charged_on_the_held_position(side, monkeypatch):
    """Control: the same bars without the wick charge (or credit) 08:00 on the position held through it."""
    res = _funding_run(side, False, monkeypatch)
    at8 = [r for r in res.journal.funding_ if pd.Timestamp(r["ts"]) == pd.Timestamp("2025-10-03 08:00", tz="UTC")]
    assert len(at8) == 1 and (at8[0]["amount"] < 0) == (side > 0), res.journal.funding_


# ================================================ condition 5: backtest and paper show the same target fill level


@pytest.mark.parametrize("label, perp, profile, side", SETUPS, ids=IDS)
def test_c5_a_target_traded_through_shows_the_same_fill_in_paper_and_backtest(label, perp, profile, side):
    """The price trades smoothly through the target (about 1 bp a second). L12 FINAL (19:30, 19:38): paper books the
    real fill and records the level; the backtest books the level less TP_SLIP. Paper is never worse than the
    backtest here (a one-sided gap is positive slippage, not a failure), and no more than a few bp better."""
    p = flat_prices(40).copy()
    s = np.arange(600, 900)
    p[600:900] = BASE * (1 + side * 0.03 * (s - 600) / 300)  # a ramp to 3 % in favour over five minutes
    p[900:] = p[899]
    run = paper(p, side=side, perp=perp, profile=profile, tp=0.02, leave=30)
    res, _ = bt_res(p, side=side, perp=perp, profile=profile, leave=30)
    pp = next(r for r in run.sequence() if r[0] == "take_profit")
    bt = next(r for r in bt_seq(res) if r[0] == "take_profit")
    level = Price_level(res, side)
    assert bt[3] == pytest.approx(tp_model(level, side), abs=0.1), (bt, level)
    assert 0 <= side * (pp[3] - bt[3]) / bt[3] < 1e-3, (pp, bt)
    (o,) = [o for o in run.orders if o["intent"] == "take_profit"]
    assert records_level(o["signal"], qa.fill_px(run, "entry") * (1 + side * 0.02)), o["signal"]


@pytest.mark.parametrize("label, perp, profile, side", SETUPS, ids=IDS)
def test_l12_a_jump_through_the_target_paper_books_the_real_fill_the_backtest_the_modelled_level(label, perp, profile,
                                                                                                  side):
    """P1-L12 settled by Advisor L12 FINAL (19:30, 19:38): phase 1 targets are market-on-touch. On a jump through the
    target paper books the real fill (61,500 for a ~61,200 target) and records the level; the backtest books the
    level less TP_SLIP, never the open. The gap is positive slippage in fills-vs-model, not a failure."""
    p = flat_prices(40).copy()
    p[630:] = BASE * favourable(side, 0.025)  # one trade at +0 %, the next at +2.5 %: no trade between
    run = paper(p, side=side, perp=perp, profile=profile, tp=0.02, leave=30)
    res, _ = bt_res(p, side=side, perp=perp, profile=profile, leave=30)
    pp = next(r for r in run.sequence() if r[0] == "take_profit")
    bt = next(r for r in bt_seq(res) if r[0] == "take_profit")
    level = Price_level(res, side)
    assert bt[3] == pytest.approx(tp_model(level, side), abs=0.1), (bt, level)  # never the 61,500 open
    assert abs(pp[3] / (BASE * favourable(side, 0.025)) - 1) < 3e-4, pp  # the real fill at the jump
    (o,) = [o for o in run.orders if o["intent"] == "take_profit"]
    assert records_level(o["signal"], qa.fill_px(run, "entry") * (1 + side * 0.02)), o["signal"]


@pytest.mark.parametrize("label, perp, profile, side", [SETUPS[0], SETUPS[3]], ids=[IDS[0], IDS[3]])
def test_l12_an_exact_touch_of_the_target_fills_in_the_backtest(label, perp, profile, side):
    """Advisor 19:40 (phase 1, supersedes 19:16 (1)): a bar whose extreme only TOUCHES the level (high == level for a
    long) fills, at the level less TP_SLIP with the taker fee, as paper's market-on-touch trigger does. Built in two
    passes: the first finds the entry fill and so the level to the tick; the second touches it exactly at 00:10:25."""
    first, _ = bt_res(flat_prices(40), side=side, perp=perp, profile=profile, leave=30)
    level = Price_level(first, side)
    p = flat_prices(40).copy()
    p[625:635] = level
    res, _ = bt_res(p, side=side, perp=perp, profile=profile, leave=30)
    assert Price_level(res, side) == level  # the same entry
    assert float(minutes_of(p).iloc[10]["high" if side > 0 else "low"]) == level  # an exact touch, nothing through
    tp = [r for r in bt_seq(res) if r[0] == "take_profit"]
    assert tp and tp[0][2] == minute(11), bt_seq(res)
    assert tp[0][3] == pytest.approx(tp_model(level, side), abs=0.1), (tp, level)


# ========================================== a post-only entry and the target in one bar (maker orders, behind a flag)


def test_a_post_only_entry_with_a_stop_books_no_target_from_a_high_before_its_fill(monkeypatch):
    """Guard (incidental: the stop sent on the entry's fill is in flight, so _bar_target sees _busy on that bar)."""
    _maker_entry_then_target(monkeypatch, stop=0.01)


def test_l13_a_post_only_entry_without_a_stop_books_no_target_from_a_high_before_its_fill(monkeypatch):
    """P1-L13, mark removed ~19:50: fixed on 9be8f62 (_bar_target skips the bar the entry filled in, _opened_seq ==
    fee_model.bar_seq; checked in the code)."""
    _maker_entry_then_target(monkeypatch, stop=None)


def _maker_entry_then_target(monkeypatch, stop):
    monkeypatch.setenv("SLEEVE_MAKER_ORDERS", "1")
    p = flat_prices(40).copy()
    p[300:310] = BASE * 1.015  # the minute to 00:06 opens nearer its high
    p[310:330] = BASE * 1.03  # the high (through 2 % from any entry near 60,000)
    p[330:360] = BASE * 0.998  # then down through the post-only buy's limit, where it closes
    p[360:] = BASE * 0.998
    res, _ = bt_res(p, side=1, perp=False, profile="aggressive", leave=30, enter=5, minutes=5, stop=stop,
                    extra={"maker_wait_minutes": 1})
    seq = bt_seq(res)
    assert seq and seq[0][0] == "entry", seq
    assert not [r for r in seq if r[0] == "take_profit" and r[2] == seq[0][2]], seq


# ================================== CR on a7fb808 (2): a slice filled mid-bar on the gap bar is never the target


def _bt_vol(prices, vol, **params):
    """A spot backtest on 5m decision bars with 1m execution bars, the minute volumes overridden ({minute index: units}),
    so a post-only entry fills in slices."""
    from sleeve_fund.instruments import BOOK_SHARE
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.venues import venue

    inst = venue("KRAKEN").instrument("BTC", "USD", price_precision=1)
    m = minutes_of(prices)
    bars = m.set_axis(m.index + pd.Timedelta(minutes=1))
    bars["volume"] = 60 / BOOK_SHARE
    for k, v in vol.items():
        bars.iloc[k, bars.columns.get_loc("volume")] = v / BOOK_SHARE
    dec = bars.resample("5min", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    return run_backtest("probe", dec, inst, params=params, starting_capital=10_000, risk_profile="aggressive",
                        bar_minutes=5, half_spread=HALF, exec_prices=bars, exec_minutes=1)


def test_l21_a_slice_filled_mid_bar_on_the_gap_bar_is_never_booked_as_the_target(monkeypatch):
    """Spot long, post-only buy at ~59,995.7 from 00:05 (4-minute wait). Slice 1 (0.0075) fills in the thin minute to
    00:06, so the stop and target are set then (target ~61,195.6). The minute to 00:08 OPENS through the target
    (61,300): slice 1 is the target's at the open. The price then falls through the limit (slice 2 fills mid-bar, after
    the open) and on through the stop, where it closes. Slice 2 never had a target to take at that open: it is
    stopped, never journaled or priced as the take-profit."""
    monkeypatch.setenv("SLEEVE_MAKER_ORDERS", "1")
    p = flat_prices(40).copy()
    p[300:360] = 59_950.0
    p[360:420] = 60_000.0
    p[420:440] = 61_300.0
    p[440:460] = 59_900.0
    p[460:] = 58_900.0
    res = _bt_vol(p, {5: 0.0075}, enter=5, leave=30, side=1, after=0, stop_loss=0.01, take_profit=0.02,
                  maker_wait_minutes=4)
    orders = {o["order_id"]: o for o in orders_of(res)}
    fills = sorted(res.journal.fills_, key=lambda f: (f["ts"], f["id"]))
    entry = [f for f in fills if orders[f["order_id"]]["intent"] == "entry"]
    before = sum(f["qty"] for f in entry if pd.Timestamp(f["ts"]) < minute(8))
    total = sum(f["qty"] for f in entry)
    # slice 1 before the gap bar (its size is the thin minute's share), slice 2 on it: the scenario held
    assert 0 < before < total and {pd.Timestamp(f["ts"]) for f in entry} == {minute(6), minute(8)}, entry
    tp_qty = sum(f["qty"] for f in fills if orders[f["order_id"]]["intent"] == "take_profit")
    assert tp_qty <= before + 1e-12, [(orders[f["order_id"]]["intent"], f["qty"], f["price"]) for f in fills]


# ===================================================================== adversarial hub probes (unseen by PE1)


def _hb_decoder(monkeypatch):
    """The client's heartbeat path: a {"t": "hb"} message flushes held minutes (HubDataClient does this)."""
    from sleeve_fund.paper import hub_client

    class HbDecoder(hub_client.Decoder):
        def __call__(self, m, now_ns):
            if m.get("t") == "hb":
                out = self.flush(now_ns)
                return out or None
            return super().__call__(m, now_ns)

    monkeypatch.setattr(hub_client, "Decoder", HbDecoder)


def _hb_decoder_with_status(monkeypatch):
    """_hb_decoder, plus a HubStatus shared by the decoder and the strategy, as paper.node wires it (HoQA harness
    fix for the L16 condition, 5f28360; the same as tests/test_hub_gaps.py's TimelineStatus). The harness decodes
    the whole feed before the run, so the status answers as the hub client's would have at the strategy's clock."""
    from sleeve_fund.paper import hub_client
    from sleeve_fund.strategies import base

    class TimelineStatus(hub_client.HubStatus):
        def __init__(self) -> None:
            super().__init__()
            self.history: list[tuple[int, dict]] = []
            self.strategy = None

        def record(self, now_ns: int) -> None:
            self.history.append((now_ns, {k: set(v) for k, v in self.lost.items()}))

        def unfilled(self, iid: str) -> list[int]:
            now, snap = self.strategy.clock.timestamp_ns(), {}
            for at, lost in self.history:
                if at > now:
                    break
                snap = lost
            return sorted(snap.get(iid, ()))

    status = TimelineStatus()

    class HbDecoder(hub_client.Decoder):
        def __init__(self, *a, **k):
            super().__init__(*a, **k, status=status)

        def __call__(self, m, now_ns):
            out = (self.flush(now_ns) or None) if m.get("t") == "hb" else super().__call__(m, now_ns)
            status.record(now_ns)
            return out

    attach = base.LongFlatStrategy.attach_runtime

    def wired(self, runtime):
        self.hub_status, status.strategy = status, self
        return attach(self, runtime)

    monkeypatch.setattr(hub_client, "Decoder", HbDecoder)
    monkeypatch.setattr(base.LongFlatStrategy, "attach_runtime", wired)
    return status


def _with(schedule_fn, monkeypatch):
    monkeypatch.setattr(qa, "schedule", schedule_fn)


def _venue_drop_schedule(*, refill_delay=0, drop_filled=False, drop_gap=False, hb=True, second=None):
    """The hub loses its venue 00:06-00:10; the first live minute (to 00:11) is published first, with the gap message
    just before it; the refill of 00:07-00:10 comes refill_delay ns after the live minute, then "filled" (unless
    dropped). Heartbeats every 5 s. second=(a, b, live_close, refill_at): a second gap (minutes closing in (a, b]),
    announced before its live minute live_close, refilled at refill_at, then its "filled"."""

    def sched(mins, iid, lag=S // 2, away=None, back_at=None, refill_after_live=False, lost=()):
        a, b = START + 6 * M, START + 10 * M
        first_live = START + 11 * M
        refill_at = first_live + lag + 1 + refill_delay
        out = []
        for ts, r in mins.iterrows():
            close = int(ts.value) + M
            if a < close <= b:
                out.append((refill_at, qa.hub_msg(iid, close, r, refilled=True), close))
            elif second is not None and second[0] < close <= second[1]:
                out.append((second[3], qa.hub_msg(iid, close, r, refilled=True), close))
            else:
                out.append((close + lag, qa.hub_msg(iid, close, r), close))
        span = {"id": iid, "since": a + M, "until": b}
        if not drop_gap:
            out.append((first_live + lag, {"t": "gap", **span}, first_live - 1))
        if not drop_filled:
            out.append((refill_at, {"t": "filled", **span}, b + 1))
        if second is not None:
            s2 = {"id": iid, "since": second[0] + M, "until": second[1]}
            out.append((second[2] + lag, {"t": "gap", **s2}, second[2] - 1))
            out.append((second[3], {"t": "filled", **s2}, second[1] + 1))
        if hb:
            for t in range(START, START + len(mins) * M, 5 * S):
                out.append((t + 3, {"t": "hb", "ts": t, "venue_up": True}, -1))
        out.sort(key=lambda x: (x[0], x[2]))
        return [(at, m) for at, m, _ in out]

    return sched


def _drop_prices(side, n=30):
    """2 % against 00:06:30-00:06:50 (through the 1 % stop), recovered: only the refilled minute to 00:07 shows it."""
    return shape(flat_prices(n), 6.5, 6 + 50 / 60, adverse(side, 0.02))


DROP_GONE = frozenset(range(6 * 60, 10 * 60))
PROBE_SETUPS = [("spot-long", False, "aggressive", 1), ("perp-3x-short", True, "aggressive", -1)]
PROBE_IDS = [s[0] for s in PROBE_SETUPS]


def _no_store(mins):
    return recover_from(mins, available=lambda close: False)


@pytest.mark.parametrize("label, perp, profile, side", PROBE_SETUPS, ids=PROBE_IDS)
def test_probe_l2_hold_filled_arrives_in_time_the_stop_runs(label, perp, profile, side, monkeypatch):
    """Control for the probes: refill 3 s after the live minute, then "filled": the held minute waits, the refills
    reach the strategy, the stop runs."""
    _hb_decoder(monkeypatch)
    _with(_venue_drop_schedule(refill_delay=3 * S), monkeypatch)
    p = _drop_prices(side)
    run = paper(p, side=side, perp=perp, profile=profile, leave=20, gone=DROP_GONE, recover=_no_store(minutes_of(p)))
    ex = first_exit(run.sequence())
    assert ex[0] == "stop_loss" and ex[2] <= minute(12), run.sequence()


@pytest.mark.parametrize("delay_s", [25, 45], ids=["filled-after-25s", "filled-after-45s"])
@pytest.mark.parametrize("label, perp, profile, side", PROBE_SETUPS, ids=PROBE_IDS)
def test_l14_refills_landing_after_the_20s_flush_still_have_their_stop_checked(label, perp, profile, side, delay_s,
                                                                                 monkeypatch):
    _hb_decoder(monkeypatch)
    _with(_venue_drop_schedule(refill_delay=delay_s * S), monkeypatch)
    p = _drop_prices(side)
    run = paper(p, side=side, perp=perp, profile=profile, leave=20, gone=DROP_GONE, recover=_no_store(minutes_of(p)))
    ex = first_exit(run.sequence())
    assert ex[0] == "stop_loss" and ex[2] <= minute(13), run.sequence()


def test_l14_refills_landing_after_the_flush_still_reach_the_indicators_once_in_order(monkeypatch):
    p = qa.WAVY[:30 * 60]
    clean = paper(p, enter=10**6, leave=10**6 + 1, stop=None).bars
    _hb_decoder(monkeypatch)
    _with(_venue_drop_schedule(refill_delay=45 * S), monkeypatch)
    run = paper(p, enter=10**6, leave=10**6 + 1, stop=None, gone=DROP_GONE, recover=_no_store(minutes_of(p)))
    qa._no_dupes_in_order(run.bars)
    assert run.bars == clean


@pytest.mark.parametrize("label, perp, profile, side", PROBE_SETUPS, ids=PROBE_IDS)
def test_probe_filled_never_arrives_the_held_minute_is_released_by_the_flush_within_20s(label, perp, profile, side,
                                                                                         monkeypatch):
    """No "filled" and no refill at all: the live minute to 00:11 is held, then released by the heartbeat flush
    within 20-25 s (never held for good), and later minutes flow."""
    from sleeve_fund.paper import hub_client

    _hb_decoder(monkeypatch)
    _with(_venue_drop_schedule(refill_delay=10**6 * S, drop_filled=True), monkeypatch)
    p = _drop_prices(side)
    run = paper(p, side=side, perp=perp, profile=profile, leave=20, gone=DROP_GONE, recover=_no_store(minutes_of(p)))
    ts = [b[0] for b in run.bars]
    assert START + 11 * M in ts and START + 12 * M in ts, ts[-12:]
    assert hub_client.HOLD_SECONDS == 20


@pytest.mark.parametrize("label, perp, profile, side", PROBE_SETUPS, ids=PROBE_IDS)
def test_l15_a_lost_gap_message_still_has_the_refilled_stop_checked(label, perp, profile, side, monkeypatch):
    _hb_decoder(monkeypatch)
    _with(_venue_drop_schedule(refill_delay=3 * S, drop_gap=True), monkeypatch)
    p = _drop_prices(side)
    run = paper(p, side=side, perp=perp, profile=profile, leave=20, gone=DROP_GONE, recover=_no_store(minutes_of(p)))
    ex = first_exit(run.sequence())
    assert ex[0] == "stop_loss" and ex[2] <= minute(13), run.sequence()


@pytest.mark.parametrize("label, perp, profile, side", PROBE_SETUPS, ids=PROBE_IDS)
def test_l14b_two_gaps_back_to_back_both_refills_are_checked(label, perp, profile, side, monkeypatch):
    """Gap A (00:07-00:10) announced before the live minute to 00:11, held; the venue drops again at 00:11, gap B
    (00:12-00:13) is announced before the live minute to 00:14, then A's refill (slow, 00:14:05), A filled, B's refill,
    B filled. The stop is crossed inside A (00:06:30-00:06:50)."""
    _hb_decoder(monkeypatch)
    sched = _venue_drop_schedule(refill_delay=3 * M + 4 * S, hb=False,
                                 second=(START + 11 * M, START + 13 * M, START + 14 * M, START + 14 * M + 6 * S))
    _with(sched, monkeypatch)
    p = _drop_prices(side)
    gone = DROP_GONE | frozenset(range(11 * 60, 13 * 60))
    run = paper(p, side=side, perp=perp, profile=profile, leave=20, gone=gone, recover=_no_store(minutes_of(p)))
    ex = first_exit(run.sequence())
    assert ex[0] == "stop_loss" and ex[2] <= minute(15), run.sequence()


def test_probe_a_gap_on_one_instrument_never_holds_anothers_minutes():
    from sleeve_fund.paper.hub_client import Decoder

    d = Decoder("1-MINUTE-LAST-EXTERNAL", recover=lambda iid, a, b: [])
    mins = minutes_of(flat_prices(20))
    x, y = "BTC/USD.KRAKEN", "ETH/USD.KRAKEN"
    got, held_x_while_y_flowed = [], False
    for k, (ts, r) in enumerate(mins.iterrows()):
        close = int(ts.value) + M
        if k == 7:  # X's venue dropped 00:05-00:07: its gap announced before its live minute to 00:08
            d({"t": "gap", "id": x, "since": START + 6 * M, "until": START + 7 * M}, close)
        for iid in (x, y):
            if iid == x and START + 5 * M < close <= START + 7 * M:
                continue  # X's minutes to 00:06 and 00:07: refill not here yet
            out = d(qa.hub_msg(iid, close, r), close + S)
            got += out if isinstance(out, list) else [out] if out is not None else []
        if k == 7:
            held_x_while_y_flowed = x in d.held and any(
                b.ts_event == close and str(b.bar_type.instrument_id) == y for b in got)
    ys = [b.ts_event for b in got if str(b.bar_type.instrument_id) == y]
    assert ys == [int(ts.value) + M for ts in mins.index], "Y was held by X's gap"
    assert held_x_while_y_flowed


def _hub_restart_schedule(seeded: bool, refill_delay=2 * S):
    """The hub process restarts: down 00:06:20 to 00:08:30 (no messages to the client). Seeded (the store's last
    minute known): the first bar after it, the minute to 00:09, is a part bar, so a stand-in gap from the first
    missing minute to it is announced, refilled and filled, then the minute to 00:10 live. Unseeded (Gaps fresh):
    only the stand-in minute (since=close=00:09) is announced and refilled."""

    def sched(mins, iid, lag=S // 2, away=None, back_at=None, refill_after_live=False, lost=()):
        down, up = START + 6 * M + 20 * S, START + 8 * M + 30 * S
        stand_in = START + 9 * M
        since = START + 7 * M if seeded else stand_in
        refill_at = stand_in + lag + refill_delay
        out = []
        for ts, r in mins.iterrows():
            close = int(ts.value) + M
            if close + lag <= down:
                out.append((close + lag, qa.hub_msg(iid, close, r), close))
            elif since <= close <= stand_in:
                out.append((refill_at, qa.hub_msg(iid, close, r, refilled=True), close))
            elif close > stand_in:
                out.append((close + lag, qa.hub_msg(iid, close, r), close))
        out.append((stand_in + lag, {"t": "gap", "id": iid, "since": since, "until": stand_in}, since - 1))
        out.append((refill_at, {"t": "filled", "id": iid, "since": since, "until": stand_in}, stand_in + 1))
        for t in range(up, START + len(mins) * M, 5 * S):
            out.append((t + 3, {"t": "hb", "ts": t, "venue_up": True}, -1))
        out.sort(key=lambda x: (x[0], x[2]))
        return [(at, m) for at, m, _ in out]

    return sched


HUB_DOWN_GONE = frozenset(range(6 * 60 + 20, 8 * 60 + 30))


@pytest.mark.parametrize("label, perp, profile, side", PROBE_SETUPS, ids=PROBE_IDS)
@pytest.mark.parametrize("refill_s", [2, 70], ids=["refill-2s", "refill-after-the-next-live-minute"])
def test_probe_hub_restart_seeded_the_stop_in_the_down_minutes_runs_and_nothing_is_held_for_good(
        label, perp, profile, side, refill_s, monkeypatch):
    """The production hub seeds its gap state from the store (hub.__main__.last_closes): the restart's gap covers
    00:07-00:09. The stop is crossed 00:06:30-00:06:50 (no trade reaches the client then)."""
    _hb_decoder(monkeypatch)
    _with(_hub_restart_schedule(True, refill_s * S), monkeypatch)
    p = _drop_prices(side)
    run = paper(p, side=side, perp=perp, profile=profile, leave=20, gone=HUB_DOWN_GONE,
                recover=_no_store(minutes_of(p)))
    ex = first_exit(run.sequence())
    assert ex[0] == "stop_loss" and ex[2] <= minute(12), run.sequence()
    ts = [b[0] for b in run.bars]
    assert START + 10 * M in ts and START + 15 * M in ts  # the minutes after flow: never held for good


@pytest.mark.xfail(strict=True, reason="P1-L16 MINOR (#146, adversarial: hub restart with fresh gap state): when "
                   "the hub doesn't know an instrument's last stored minute (an instrument picked up by discover() "
                   "after start, or the store unreadable at start), only the stand-in minute is announced and "
                   "refilled; the minutes the hub was down are never refilled, and on 1m the strategy replays only "
                   "the bar it decides on, so a stop crossed in them is lost")
@pytest.mark.parametrize("label, perp, profile, side", PROBE_SETUPS, ids=PROBE_IDS)
def test_l16_hub_restart_unseeded_the_stop_in_the_down_minutes_runs(label, perp, profile, side, monkeypatch):
    _hb_decoder(monkeypatch)
    _with(_hub_restart_schedule(False), monkeypatch)
    p = _drop_prices(side)
    run = paper(p, side=side, perp=perp, profile=profile, leave=20, gone=HUB_DOWN_GONE,
                recover=_no_store(minutes_of(p)))
    ex = first_exit(run.sequence())
    assert ex[0] == "stop_loss" and ex[2] <= minute(12), run.sequence()


def _hub_events(run, minute_text):
    """Warnings, errors or incidents (journal or the decoder's reports, which paper forwards as alerts) naming a
    minute of the gap."""
    return [e for e in run.events if e.get("level", "warning") in ("warning", "error")
            and minute_text in (e.get("message") or "")]


@pytest.mark.parametrize("label, perp, profile, side", PROBE_SETUPS, ids=PROBE_IDS)
def test_l16_condition_after_an_unseeded_hub_restart_no_entry_until_the_gap_is_refilled(label, perp, profile, side,
                                                                                         monkeypatch):
    """HoQA condition for deferring P1-L16 to the DA: an unseeded hub restart leaves 00:07-00:08 unrefilled (never
    refilled here). The strategy, flat and due to enter from 00:10, opens NOTHING on that instrument while the gap is
    unrefilled, and a warning, alert or incident names the gap."""
    _hb_decoder_with_status(monkeypatch)  # HubStatus wired, as paper.node does (harness fix, 5f28360)
    _with(_hub_restart_schedule(False), monkeypatch)
    p = flat_prices(30)
    run = paper(p, side=side, perp=perp, profile=profile, enter=10, leave=25, gone=HUB_DOWN_GONE,
                recover=_no_store(minutes_of(p)))
    assert _hub_events(run, "00:07"), [(e["kind"], e.get("message", "")[:90]) for e in run.events][-12:]
    assert not [o for o in run.orders if o["intent"] == "entry"], run.sequence()


@pytest.mark.parametrize("label, perp, profile, side", PROBE_SETUPS, ids=PROBE_IDS)
def test_l16_condition_after_an_unseeded_hub_restart_the_open_position_keeps_its_stop(label, perp, profile, side,
                                                                                    monkeypatch):
    """Same restart with a position open from 00:05 (the gap holds nothing adverse): the stop still works afterwards,
    2 % against 00:14:30-00:14:50 is stopped out by 00:15."""
    _hb_decoder_with_status(monkeypatch)  # HubStatus wired, as paper.node does (harness fix, 5f28360)
    _with(_hub_restart_schedule(False), monkeypatch)
    p = shape(flat_prices(30), 14.5, 14 + 50 / 60, adverse(side, 0.02))
    run = paper(p, side=side, perp=perp, profile=profile, leave=25, gone=HUB_DOWN_GONE,
                recover=_no_store(minutes_of(p)))
    ex = first_exit(run.sequence())
    assert ex[0] == "stop_loss" and ex[2] <= minute(15), run.sequence()


def test_probe_no_entry_is_decided_on_the_held_minute_before_its_refills_land(monkeypatch):
    """The entry is due on the bar to 00:11, the gap's live minute. It is decided only once the refills of 00:07-00:10
    have reached the strategy, on the same minute as the backtest."""
    _hb_decoder(monkeypatch)
    _with(_venue_drop_schedule(refill_delay=8 * S), monkeypatch)
    p = flat_prices(30)
    run = paper(p, enter=11, leave=20, stop=0.01, gone=DROP_GONE, recover=_no_store(minutes_of(p)))
    (e,) = [o for o in run.orders if o["intent"] == "entry"]
    decided = pd.Timestamp(e["ts"])
    assert decided >= pd.Timestamp(START + 11 * M + 8 * S, unit="ns", tz="UTC"), decided
    accepted = [b[0] for b in run.bars]
    assert all(START + k * M in accepted for k in range(7, 12)), accepted[:15]
    bt = qa.backtest(p, enter=11, leave=20)
    pp = run.sequence()[0]
    assert pp[:2] == bt[0][:2] and abs((pp[2] - bt[0][2]).total_seconds()) <= 60, (pp, bt[0])


def test_l17_no_entry_on_a_minute_released_by_the_flush_with_its_gap_unfilled(monkeypatch):
    _hb_decoder(monkeypatch)
    _with(_venue_drop_schedule(refill_delay=10**6 * S, drop_filled=True), monkeypatch)
    p = flat_prices(30)
    run = paper(p, enter=11, leave=20, stop=0.01, gone=DROP_GONE, recover=_no_store(minutes_of(p)))
    decided = [pd.Timestamp(o["ts"]) for o in run.orders if o["intent"] == "entry"]
    assert not decided or decided[0] >= minute(12), decided  # not on the flushed minute to 00:11


# ==================================================================== an outage across a funding settlement


def _funding_paper(monkeypatch, *, side, stop_in_outage: bool, rate=0.0001):
    """Perp 3x, entered at 07:55; hub away 07:56-08:02 (refilled 08:02:05), so the 08:00 settlement falls in the
    outage. The clock starts at 07:50 (the probe's minutes count from there)."""
    from sleeve_fund.strategies.base import LongFlatStrategy

    start = int(pd.Timestamp("2025-10-03 07:50", tz="UTC").value)
    monkeypatch.setattr(qa, "START", start)
    monkeypatch.setattr(LongFlatStrategy, "_funding_rate", lambda self, terms, ts, now, *_: rate)  # harness: #155 passes the wait too
    p = flat_prices(40)
    if stop_in_outage:
        p = shape(p, 7.5, 7 + 50 / 60, adverse(side, 0.02))  # 07:57:30-07:57:50 through the 1 % stop
    run = paper(p, side=side, perp=True, profile="aggressive", leave=35, gone=frozenset(range(6 * 60, 12 * 60)),
                away=(start + 6 * M, start + 12 * M), back_at=start + 12 * M + 5 * S)
    return run, start


def _funding_rows(run):
    return run.store.funding("q146")


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_outage_across_a_settlement_charges_08_00_once_on_the_held_position(side, monkeypatch):
    run, start = _funding_paper(monkeypatch, side=side, stop_in_outage=False)
    rows = [r for r in _funding_rows(run) if pd.Timestamp(r["ts"]) == pd.Timestamp("2025-10-03 08:00", tz="UTC")]
    assert len(rows) == 1, _funding_rows(run)
    entry = next(r for r in run.sequence() if r[0] == "entry")
    assert rows[0]["amount"] == pytest.approx(-side * entry[3] * abs(rows[0]["qty"]) * 0.0001, rel=2e-3)


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_l19_outage_stop_booked_before_the_settlement_pays_and_receives_no_08_00_funding(side, monkeypatch):
    """The replay books the stop in the minute to 07:58; the venue's resting stop would have closed the position
    before 08:00, so no 08:00 funding is charged or credited, though the exit is sent at 08:02."""
    run, start = _funding_paper(monkeypatch, side=side, stop_in_outage=True)
    ex = first_exit(run.sequence())
    assert ex[0] == "stop_loss", run.sequence()
    rows = [r for r in _funding_rows(run) if pd.Timestamp(r["ts"]) == pd.Timestamp("2025-10-03 08:00", tz="UTC")]
    assert rows == [], rows


# ================================================== NA-3: a liquidation found by the replay opens an incident, halts


# P1-L18: passes with #155's incident and halt on the liquidation fill (PE2): its xfail mark removed
@pytest.mark.parametrize("path", ["reconnect", "restart"])
@pytest.mark.parametrize("label, perp, profile, side", qa.LIQ_SETUPS, ids=qa.LIQ_IDS)
def test_l18_a_liquidation_found_by_the_replay_opens_an_incident_and_halts(path, label, perp, profile, side):
    p = shape(flat_prices(30), 7.0, 8.0, adverse(side, qa.LIQ_DEPTH[profile]))
    run = qa._outage(path, p, side=side, perp=perp, profile=profile, stop=0.10, qty=0.25)
    assert first_exit(run.sequence())[0] == "liquidation", run.sequence()
    assert run.kinds("incident"), [e["kind"] for e in run.events][-15:]
    assert run.store.sleeve("q146").status == "halted"


@pytest.mark.parametrize("path", ["reconnect", "restart"])
@pytest.mark.parametrize("label, perp, profile, side", qa.LIQ_SETUPS, ids=qa.LIQ_IDS)
def test_na3_a_liquidation_found_by_the_replay_is_journaled_as_an_error_and_never_re_entered(path, label, perp,
                                                                                              profile, side):
    p = shape(flat_prices(30), 7.0, 8.0, adverse(side, qa.LIQ_DEPTH[profile]))
    run = qa._outage(path, p, side=side, perp=perp, profile=profile, stop=0.10, qty=0.25)
    liq = [e for e in run.events if e["kind"] == "liquidation"]
    assert liq and liq[0]["level"] == "error", liq
    seq = run.sequence()
    k = next(i for i, r in enumerate(seq) if r[0] == "liquidation")
    assert not [r for r in seq[k + 1:] if r[0] == "entry"], seq
