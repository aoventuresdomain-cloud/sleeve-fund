"""TP-EXIT (PM 14:42 UK 21 Oct: every Strategy gets a stop-loss and a take-profit; Advisor's fill rule FINAL 14:48 UK):
the rule builder's take-profit exit, `take_profit = {r = N}` (N times R, R the entry's distance to its stop, frozen at
entry) or `take_profit = {level = "<operand>"}` (a level read from closed bars). Cells by QD from the Advisor's eight
points; PE4 owns the engine code and may not edit these without a TEC.

1. touch fires: a long's target fires on a bar whose high reaches it (>=), an exact touch included; a short's on the low.
2. a taker exit: booked at the level less max(half spread, 0.05%) (instruments.target_fill_px), plus the taker fee.
3. never better than that, a bar that opened through the target included.
4. a bar that reached both the stop and the target takes the stop.
5. R is frozen at entry: a trail that moves the stop doesn't move the target.
6. the level form reads closed bars only, and a level on the wrong side of the entry when set never fires.
7. the exit is recorded as kind "take_profit", with its level.
8. check_definition refuses a take_profit without a stop.
Plus the must-keeps: one parity cell across the three paths (backtest bar, paper tick, replay_missed) on an
exact-touch bar, and a definition without a take_profit trading exactly as before.

Red first on 96e5fbf3: check_definition refuses the take_profit key (LEVEL_EXITS = stop, trail, exit_at_level)."""

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.instruments import FeeSchedule, target_fill_px
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.strategies.base import replay_missed
from sleeve_fund.strategies.definitions import check_definition, to_params
from sleeve_fund.venues import venue

INST = venue("KRAKEN").instrument("BTC", "USD")
TAKER = Decimal("0.00055")
FEES = FeeSchedule(maker=Decimal("0.0002"), taker=TAKER)
REASON = "A test case, written before it is run."
STOP = 2.0  # the stop sits this far from the entry close: R


def _defn(take_profit, side="long", stop=STOP, trail=None, blocks=None):
    """An always-true entry on `side`, a stop STOP from the close, and the take-profit under test."""
    sign = 1 if side == "long" else -1
    exits = {"stop": {"level": {"sub" if sign > 0 else "add": ["close", stop]}}}
    if trail is not None:
        exits["trail"] = {"level": {"sub" if sign > 0 else "add": ["close", trail]}}
    if take_profit is not None:
        exits["take_profit"] = take_profit
    return {"version": 1, "reason": REASON, "blocks": blocks or {},
            side: {"entry": {"left": "close", "op": ">", "right": 0}}, "exits": exits}


def _bars(n=40, at=100.0):
    """1-minute bars flat at `at`, deep enough that an order doesn't walk the book."""
    idx = pd.date_range("2025-10-03 00:01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"open": at, "high": at, "low": at, "close": at, "volume": 1e9}, index=idx)


def _set(df, i, **ohlc):
    for k, v in ohlc.items():
        df.iloc[i, df.columns.get_loc(k)] = v
    return df


def _run(defn, df, side="long", half_spread=0.0):
    params = to_params(defn)
    if side == "short":
        params = {**params, "market": "perp", "allow_short": True}
    return run_backtest("rules", df, INST, params=params, bar_minutes=1, half_spread=half_spread, fees=FEES)


def _exits(res):
    """Each exit in fill order: (intent, its bar's close time, fill price, fee over notional, decision signal)."""
    f = res.fills.sort_values("ts_last", kind="stable")
    out = []
    for o, r in f.iterrows():
        d = res.decisions[o]
        if d["intent"] == "entry":
            continue
        fee = float(str(r["commissions"][0] if isinstance(r["commissions"], list) else r["commissions"]).split()[0])
        out.append((d["intent"], r["ts_last"], float(r["avg_px"]), fee / (float(r["filled_qty"]) * float(r["avg_px"])),
                    d.get("signal") or {}))
    return out


def _first(res, intent="take_profit"):
    return next((e for e in _exits(res) if e[0] == intent), None)


def _target(entry=100.0, n=3, side=1, stop=None):
    """{r = n} (Advisor's ruling, HoE 15:28 UK): the entry fill plus (a short: less) n times R, R = |the entry fill -
    the initial stop level|, in price, frozen at entry; costs go on the fills. With no spread on these deep bars the
    entry fills at the close, 100, and the stop sits at 98 (102): 3R = 106 (94)."""
    stop = 100.0 - side * STOP if stop is None else stop
    return entry + side * n * abs(entry - stop)


def _entry_fill(res) -> float:
    f = res.fills.sort_values("ts_last", kind="stable")
    return next(float(r["avg_px"]) for o, r in f.iterrows() if res.decisions[o]["intent"] == "entry")


# ---- 1. touch fires -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("side", ["long", "short"])
def test_the_target_fires_on_a_bar_that_touches_it_exactly(side):
    sign = 1 if side == "long" else -1
    level = _target(side=sign)
    df = _set(_bars(), 20, **({"high": level} if sign > 0 else {"low": level}))
    tp = _first(_run(_defn({"r": 3}, side), df, side))
    assert tp is not None and tp[1] == df.index[20], _exits(_run(_defn({"r": 3}, side), df, side))


@pytest.mark.parametrize("side", ["long", "short"])
def test_a_bar_short_of_the_target_by_one_tick_doesnt_fire(side):
    sign = 1 if side == "long" else -1
    near = _target(side=sign) - sign * 0.1
    df = _set(_bars(), 20, **({"high": near} if sign > 0 else {"low": near}))
    assert _first(_run(_defn({"r": 3}, side), df, side)) is None


# ---- 2. a taker exit at the level less the slippage, plus the taker fee ----------------------------------------------

@pytest.mark.parametrize("side, half_spread", [("long", 0.0), ("long", 0.001), ("short", 0.0), ("short", 0.001)])
def test_the_target_is_booked_at_its_level_less_the_takers_slippage_plus_the_taker_fee(side, half_spread):
    sign = 1 if side == "long" else -1
    level = _target(side=sign)
    df = _set(_bars(), 20, **({"high": level + sign * 1} if sign > 0 else {"low": level + sign * 1}))
    res = _run(_defn({"r": 3}, side), df, side, half_spread)
    _, _, px, fee, sig = _first(res)
    level = sig["target_px"]
    # From the entry fill, not the decision close: with a spread the fill sits off the close, and R with it.
    assert level == pytest.approx(_target(_entry_fill(res), side=sign), abs=0.01)  # on the price step
    assert half_spread == 0 or abs(level - _target(side=sign)) > 0.1
    assert px == pytest.approx(float(target_fill_px(level, sign > 0, half_spread)), rel=1e-6)
    assert px == pytest.approx(level * (1 - sign * max(half_spread, 0.0005)), rel=1e-6)
    assert fee == pytest.approx(float(TAKER), rel=1e-3)


# ---- 3. never better than the level, a gap through it included ------------------------------------------------------

@pytest.mark.parametrize("side", ["long", "short"])
def test_a_bar_that_opens_through_the_target_books_it_at_the_level_never_the_better_open(side):
    sign = 1 if side == "long" else -1
    level, through = _target(side=sign), _target(side=sign) + sign * 4
    df = _set(_bars(), 20, open=through, high=max(through, 100) if sign > 0 else 100,
              low=min(through, 100) if sign < 0 else 100, close=through)
    _, at, px, _, _ = _first(_run(_defn({"r": 3}, side), df, side))
    assert at == df.index[20]
    assert px == pytest.approx(float(target_fill_px(level, sign > 0, 0.0)), rel=1e-6)
    assert sign * (px - level) <= 0


# ---- 4. stop first on a shared bar ----------------------------------------------------------------------------------

@pytest.mark.parametrize("side", ["long", "short"])
def test_a_bar_through_both_the_stop_and_the_target_takes_the_stop(side):
    sign = 1 if side == "long" else -1
    df = _set(_bars(), 20, high=100 + 10 if sign > 0 else 100 + 3, low=100 - 3 if sign > 0 else 100 - 10)
    exits = _exits(_run(_defn({"r": 3}, side), df, side))
    assert exits and exits[0][0] == "stop_loss" and exits[0][1] == df.index[20], exits


# ---- 5. R frozen at entry -------------------------------------------------------------------------------------------

def test_a_trail_that_moves_the_stop_doesnt_move_the_target():
    """The trail ratchets the stop up with the closes (to 103 by bar 19); a target re-read from the moved stop would
    sit nearer, at 103 + 3 x 2 = 109 from the close or 100 + 3 x (100 - 103) below the entry: it stays at 106."""
    df = _bars()
    for i, c in enumerate([101.0, 102.0, 103.0, 104.0, 105.0], start=15):
        _set(df, i, high=c, close=c, open=c - 1, low=c - 1)
    for i in range(20, 40):
        _set(df, i, open=105.0, high=105.0, low=105.0, close=105.0)
    _set(df, 30, high=106.0)
    res = _run(_defn({"r": 3}, trail=STOP), df)
    tp = _first(res)
    assert tp is not None and tp[1] == df.index[30], _exits(res)
    assert tp[4]["target_px"] == pytest.approx(_target())


# ---- 6. the level form: closed bars only, never from the wrong side -------------------------------------------------

def _level_defn(level):
    return _defn({"level": level}, blocks={"m": {"kind": "sma", "period": 1}})


def test_the_level_form_fires_on_the_level_the_last_closed_bar_set():
    """The level is the last close plus 6. Bar 20 trades to 106.5 and closes at 101: the closed bar before it set the
    level at 106, so it fires; a level read from bar 20's own close (107) wouldn't."""
    df = _set(_bars(), 20, high=106.5, close=101.0)
    tp = _first(_run(_level_defn({"add": ["m", 6]}), df))
    assert tp is not None and tp[1] == df.index[20] and tp[4]["target_px"] == pytest.approx(106.0)


def test_the_level_form_waits_for_a_bar_after_the_close_that_moved_it():
    """Bar 19 closes at 103, moving the level to 109 for bar 20; bar 20 trades to 107 and doesn't fire."""
    df = _set(_set(_bars(), 19, high=103.0, close=103.0), 20, open=103.0, high=107.0, low=103.0, close=103.0)
    assert _first(_run(_level_defn({"add": ["m", 6]}), df)) is None


def test_a_level_on_the_wrong_side_of_the_entry_when_set_never_fires():
    """Below a long's entry it would be a stop, not a target: none is set, and a rally doesn't take it."""
    df = _bars()
    for i in range(5, 40):
        _set(df, i, open=100 + i * 0.5 - 0.5, high=100 + i * 0.5, low=100 + i * 0.5 - 0.5, close=100 + i * 0.5)
    exits = _exits(_run(_level_defn({"sub": ["m", 1]}), df))
    assert not any(e[0] == "take_profit" for e in exits), exits


# ---- 7. the exit kind is recorded -------------------------------------------------------------------------------------

def test_the_exit_is_recorded_as_a_take_profit_with_its_level_and_r():
    df = _set(_bars(), 20, high=107.0)
    res = _run(_defn({"r": 3}), df)
    intent, _, _, _, sig = _first(res)
    assert intent == "take_profit" and sig["target_px"] == pytest.approx(106.0)
    assert sig["entry_px"] == pytest.approx(100.0)


# ---- 8. check_definition --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("tp", [{"r": 3}, {"level": "high"}])
def test_check_definition_accepts_a_take_profit_beside_a_stop(tp):
    check_definition(_defn(tp), bar_spec="1-MINUTE-LAST-INTERNAL")


@pytest.mark.parametrize("tp", [{"r": 3}, {"level": "high"}])
def test_check_definition_refuses_a_take_profit_without_a_stop(tp):
    d = _defn(tp)
    del d["exits"]["stop"]
    with pytest.raises(ValueError, match="take_profit needs a stop"):
        check_definition(d, bar_spec="1-MINUTE-LAST-INTERNAL")


@pytest.mark.parametrize("tp", [{"r": 0}, {"r": -1}, {"r": 3, "level": "high"}, {}])
def test_check_definition_refuses_a_take_profit_that_isnt_one_r_or_one_level(tp):
    with pytest.raises(ValueError, match="take_profit"):
        check_definition(_defn(tp), bar_spec="1-MINUTE-LAST-INTERNAL")


def test_a_definitions_take_profit_beside_the_studys_take_profit_r_is_refused():
    """One target per trade: the study-wide take_profit_r on top of a definition's own take_profit is refused."""
    with pytest.raises(ValueError, match="take_profit_r"):
        run_backtest(
            "rules", _bars(), INST, params={**to_params(_defn({"r": 3})), "take_profit_r": 3}, bar_minutes=1,
            half_spread=0.0, fees=FEES)


def test_the_sheet_says_the_take_profit_is_in_price_r(tmp_path):
    from sleeve_fund.research import tearsheet
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.study import run_study
    from sleeve_fund.strategies.rules import SPEC as RULES

    defn = {"version": 1, "reason": REASON, "blocks": {"fast": {"kind": "ema", "period": 10},
                                                      "slow": {"kind": "ema", "period": 30},
                                                      "atr": {"kind": "atr", "period": 14}},
            "long": {"entry": {"left": "fast", "op": "crosses_above", "right": "slow"},
                     "exit": {"left": "fast", "op": "<", "right": "slow"}},
            "exits": {"stop": {"level": {"sub": ["close", {"mul": [3, "atr"]}]}}, "take_profit": {"r": 3}}}
    ledger = IdeaLedger(tmp_path / "l.jsonl")
    r = run_study(RULES, _wicky(), venue("KRAKEN").instrument("BTC", "USD", price_precision=1), dataset="syn", ledger=ledger, synthetic=True,
                  holdout_days=60, train_days=240, test_days=60, default_params=to_params(defn))
    assert "take-profit at 3R (price)" in tearsheet.render(r, ledger)


def _wicky(seed=5, days=500):
    """4h candles whose lows often wick 2-5% below and close back up."""
    idx = pd.date_range("2022-01-01 04:00", periods=days * 6, freq="4h", tz="UTC")
    rng = np.random.default_rng(seed)
    c = np.round(30_000 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(idx))
                                           + 0.006 * np.sin(np.arange(len(idx)) / 60))), 1)
    o = np.r_[c[0], c[:-1]]
    wick = np.where(rng.random(len(idx)) < 0.3, rng.uniform(0.02, 0.05, len(idx)), 0.002)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.002, "low": np.minimum(o, c) * (1 - wick),
                         "close": c, "volume": 1e9}, index=idx)


# ---- must-keep: three-path parity on an exact-touch bar ---------------------------------------------------------------

def test_the_backtest_the_paper_tick_and_replay_take_the_same_exact_touch_bar(tmp_path):
    """One recorded paper session (a trade a second), replayed tick by tick, against the same trades as one-minute
    bars and against replay_missed over that minute: each takes the target in the minute one trade touches it
    exactly, at the same level. The level is read from a first backtest (the entry fills at the ask, so where R is
    measured from is the engine's), then one trade is put exactly on it."""
    import test_tick_bar_parity as tbp

    half = tbp.SPREAD / 2 / 60_000
    params = to_params(_defn({"r": 3}, stop=1_200.0))  # the stop 1,200 (2%) below the entry close

    def session(touch, name):
        prices = np.full(40 * 60, 60_000.0)
        prices[25 * 60 + 30] = touch
        path = tmp_path / f"{name}.jsonl.gz"
        trades = tbp._record(path, prices, "rules", params, "aggressive")
        bars = trades.resample("1min", closed="left", label="right").ohlc()
        bars["volume"] = 1e9
        res = run_backtest("rules", bars, tbp.INST, params=params, starting_capital=10_000, risk_profile="aggressive",
                           bar_minutes=1, half_spread=half, fees=FEES)
        return path, bars, res

    level = _first(session(64_000.0, "probe")[2])[4]["target_px"]
    assert round(level, 1) == pytest.approx(level, abs=1e-9)  # on the price step, so a trade can touch it exactly
    path, bars, res = session(level, "touch")
    bt = _first(res)
    minute = bars.index[bars["high"] >= level][0]
    assert bt is not None and bt[1] == minute and bt[4]["target_px"] == pytest.approx(level)
    orders, _ = tbp.replay(path, with_fills=True)
    paper = [o for o in orders if o["intent"] == "take_profit"]
    assert paper and tbp._minute(pd.Timestamp(paper[0]["ts"]).to_pydatetime()) == minute.to_pydatetime()
    assert paper[0]["signal"]["target_px"] == pytest.approx(level, abs=0.05)
    row = bars.loc[minute]
    hit = replay_missed([(minute.value, row["open"], row["high"], row["low"])], 1, {"stop_loss": 58_800.0}, {},
                        level, float(target_fill_px(level, True, half)))
    # The report folds the fee's cent rounding into the fill price (D13 pin e: within 1.5 cents over the quantity).
    f = res.fills.sort_values("ts_last", kind="stable")
    qty = next(float(r["filled_qty"]) for o, r in f.iterrows() if res.decisions[o]["intent"] == "take_profit")
    assert hit is not None and hit[0] == "take_profit"
    assert hit[1] == pytest.approx(bt[2], abs=1.5 * 0.0100001 / qty + 1e-9)


# ---- must-keep: no take_profit, no change ---------------------------------------------------------------------------

def test_a_definition_without_a_take_profit_trades_exactly_as_before():
    """Pinned on 96e5fbf3, before TP-EXIT: the same stop-only definition on the same wicky bars makes the same exits
    at the same prices."""
    rng = np.random.default_rng(7)
    df = _bars(n=400)
    c = np.round(100 * np.exp(np.cumsum(rng.normal(0, 0.004, len(df)))), 2)
    df["open"], df["close"] = np.r_[100.0, c[:-1]], c
    df["high"] = np.maximum(df["open"], c) * (1 + rng.uniform(0, 0.01, len(df)))
    df["low"] = np.minimum(df["open"], c) * (1 - rng.uniform(0, 0.01, len(df)))
    exits = _exits(_run(_defn(None), df))
    assert (len(exits), {e[0] for e in exits}) == PINNED_COUNT
    assert round(sum(e[2] for e in exits), 6) == PINNED_SUM


PINNED_COUNT = (11, {"stop_loss"})  # run on 96e5fbf3
PINNED_SUM = 993.822852
