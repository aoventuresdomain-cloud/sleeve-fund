"""Head of QA: edge-case acceptance tests for v2 P1-5 (rule builder) and its board 9b extras, written BEFORE the
build as STRICT xfails. Platform Engineer 1 makes each pass, then removes its xfail mark, before QA round 1. Copy
this file into tests/ to run it (it uses the suite's own helpers and fixtures).

Sources (every trading-behaviour expectation cites one):
- [S]   v2/strategy-builder-spec.md, Step 3 and build item P1-5 (Done when), guardrails.
- [HF]  v2/hoe-feasibility.md answers 2 and 3: TOML definitions in configs/definitions/, copied into the strategy's
        params with a version and content hash at creation; trials register keyed by that hash; a new `rules`
        model, a LongFlatStrategy subclass in REGISTRY.
- [B6]  eng-board.md row 6: block chaining, time stop, exit at indicator level, DA-4 lineage.
- [B9b] eng-board.md row 9b: A4 arithmetic (move > 2 x ATR, Z-score), A6 calendar and time-of-day, A7 stop from a
        structure level, B3 extra slippage on breakout candles, A11 ratcheting channel trailing stop.
- [ADV] eng-board.md, Advisor 15:35 (ATR): port the models on simple ATR with an exact trade match; Wilder switch =
        new trials-register variant; builder offers "ATR" (Wilder default) + "ATR (simple average)".
- [A5]  quant-review/v2-p1/atr-149.md P1-A5: the rule evaluator must gate on `settled`, not `initialized`.
- [HoE] eng-board.md test-corrections log, 6 Oct 16:05: venue warm-up capped at 720 bars, so ATR length >= 37 shows
        "Short warm-up"; rule builder P1-5 to surface this.
- [MA7] quant-review/v2-p1/atr-157.md m-A7 (coordinator, 6 Oct): a warm-up that MAX_WARMUP_BARS can never load is
        refused by the definition schema, and the reason names the cap.
- [DA4] data-architect/da4-lineage-note.md "Done when" 1 and 3 (signal v1 payload).
- [DA8] sleeve_fund/migrations/DATA_MODEL.md "Keys and links" + data-architect/charter.md decision 4.
- [P14] v2/p1-4-multi-timeframe-design.md rules 2-3 (slower candle visible only once closed).
- [ENG] engine conventions pinned by today's suite and reused per [S] "Exits reuse what exists today":
        tests/test_backtest.py::test_stop_fills_at_the_open_when_price_gaps_through, rsi_cross time_stop_bars.
- [R]   Advisor rulings 6 Oct 16:40, amended 16:43 (via the coordinator; README "Advisor rulings"): 5.1 the settled
        gate wins; exact match judged after the longest warm-up, on preloaded history; legacy trades on unsettled
        indicators reported and flagged, never silently changed. 5.2 a time stop is wall-clock by default (first bar
        at or after the deadline); a port may set "count bars" and the rsi_cross port does. 5.3 an indicator-level
        exit may rest in the bar at the level set at the previous close, or check on the close; the current bar's own
        level is never used inside that bar; an explicit option rests at the previous closed bar's level. 5.4 a
        trigger exactly at expiry counts; a fresh setup re-arms and restarts the expiry; trigger and cancel at the
        same instant: cancel wins. 5.5 a gate reads the decision time, the bar's CLOSE; each gate declares its
        timezone, UTC by default, local with DST for sessions. 5.6 B3 applies only to entries filled on the bar where
        the breakout first becomes true; stop exits never pay it; other exits normal slippage. 5.7 the A11 trail
        rests in the bar on the channel up to the previous closed bar. 5.8 A7 uses bars fully closed before the
        entry fill (the entry bar is included for a fill at its close). 5.9 all/any members are put in canonical
        order (sorted, duplicates removed) before hashing. 4.4 a venue's daily anchor is part of the hash.
        Advisor 17:07 (via the coordinator): a level resting inside bar k (A11 trail, A7 structure stop, a B3 breakout
        level used in the next bar) is the channel of the n most recent CLOSED bars, k-n to k-1, the bar just closed
        included; the rule builder defines it so and does not rely on the library block's offset. At bar i's close a
        channel output therefore takes in bar i, and a breakout compares close[i] with {"prev": channel}.
Not pinned: Researcher R2 first-touch (a follow-up after P1-5 core, board backlog).

ASSUMED INTERFACE (nothing below is built yet; the assertions depend only on behaviour, and every name lives in the
adapter section, so the engineer adapts the adapter, never the assertions):
- Module: the first importable of sleeve_fund.strategies.rules, sleeve_fund.rules, sleeve_fund.definitions.
- Checking a definition: the first callable of check_definition / validate_definition / validate /
  load_definition / parse_definition / load, given the definition (a dict as TOML parses it) and optionally
  bar_spec=...; raises ValueError with the reason on refusal; whatever it returns (or warns) is the "notes".
- Hash: definition_hash / content_hash / hash_definition(defn, venue=...) -> str, else the checked result's
  `content_hash`; `venue` (if the function takes it) brings in the venue's daily anchor [R 4.4].
- Warm-up preload in a backtest: run_backtest takes the bars before the test window as warm-up, under the first of
  warmup_prices / warmup / preload / history, fed without trading as paper's history warm-up is [R 5.1].
- Legacy unsettled trades [R 5.1b]: the backtest result has an attribute whose name contains "unsettled" holding
  their count, and each such fill's decision carries a truthy key containing "unsettled".
- Signals of exits and gates (SHAPE): exits.time_stop {"minutes": n} (wall-clock) or {"bars": n, "count": "bars"};
  exits.exit_at_level {"level": id, "rest": "previous_close" | "current_bar"}; a time operand may carry "tz".
- Strategy params for a definition: to_params / strategy_params / params_for(defn), else {"definition": defn}.
- The model is REGISTRY["rules"] [HF]; definitions of the two ported models are TOML files in configs/definitions/
  whose names contain "rsi_bands" and "rsi_pullback" [HF][S].
- The builder's block catalogue: some attribute of the module holds the offered labels, among them "ATR" and
  "ATR (simple average)" [ADV].
- ASSUMED DEFINITION SHAPE: the helpers D/B/C/... below are the only place the schema is spelled out. When the
  real schema differs, change those helpers (and the few literal definitions marked SHAPE), not the asserts.

Every import of P1-5 code happens inside a test body, so on main every test XFAILs rather than errors."""

import copy
import importlib
import inspect
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
START = 1_759_449_600_000_000_000  # 2025-10-03 00:00 UTC (a Friday), as tests/test_replay.py
PERP = {"market": "perp", "allow_short": True}
RULES = "rules"  # [HF] answer 3



# ==== ADAPTER: the only place that names P1-5 code =================================================================

def _api():
    for name in ("sleeve_fund.strategies.rules", "sleeve_fund.rules", "sleeve_fund.definitions"):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name is None or not name.startswith(exc.name):
                raise  # the module exists but something it imports does not
    raise ImportError("P1-5: no rule-builder module yet")


def _fn(*names):
    api = _api()
    for n in names:
        f = getattr(api, n, None)
        if callable(f):
            return f
    raise ImportError(f"P1-5: none of {names} in {api.__name__}")


def _call(f, defn, **ctx):
    accepted = inspect.signature(f).parameters
    kw = {k: v for k, v in ctx.items() if k in accepted or any(p.kind == p.VAR_KEYWORD for p in accepted.values())}
    return f(defn, **kw)


def _check(defn, **ctx):
    """Check a definition; returns (result, notes text). Raises ValueError with the reason on a refusal."""
    f = _fn("check_definition", "validate_definition", "validate", "load_definition", "parse_definition", "load")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _call(f, copy.deepcopy(defn), **ctx)
    return result, " | ".join([_text(result)] + [str(w.message) for w in caught])


def _text(x, depth=0) -> str:
    if depth > 4:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return " ".join(_text(v, depth + 1) for v in x.values())
    if isinstance(x, (list, tuple, set)):
        return " ".join(_text(v, depth + 1) for v in x)
    if hasattr(x, "__dict__"):
        return " ".join(_text(v, depth + 1) for v in vars(x).values()) + " " + str(x)
    return str(x)


def _hash(defn) -> str:
    try:
        return _fn("definition_hash", "content_hash", "hash_definition")(copy.deepcopy(defn))
    except ImportError:
        result, _ = _check(defn)
        return getattr(result, "content_hash", None) or result["content_hash"]


def _params(defn) -> dict:
    try:
        return _fn("to_params", "strategy_params", "params_for")(copy.deepcopy(defn))
    except ImportError:
        _api()
        return {"definition": copy.deepcopy(defn)}


def _hash_on(defn, venue: str) -> str:
    """The hash of a definition run on `venue` [R 4.4]."""
    try:
        f = _fn("definition_hash", "content_hash", "hash_definition")
    except ImportError:
        return _hash(defn)
    return _call(f, copy.deepcopy(defn), venue=venue)


def _preload_kw() -> str:
    """run_backtest's parameter for warm-up bars fed before the window without trading [R 5.1a]."""
    _api()
    from sleeve_fund.research.runner import run_backtest

    names = inspect.signature(run_backtest).parameters
    for n in ("warmup_prices", "warmup", "preload", "history"):
        if n in names:
            return n
    raise ImportError("P1-5: run_backtest takes no warm-up preload yet")


def _port_definition(model: str):
    """The definition file the engineer wrote for a legacy model's port [HF][S]."""
    _api()
    files = sorted((REPO / "configs" / "definitions").glob(f"*{model}*.toml"))
    assert files, f"no configs/definitions/*{model}*.toml"
    try:
        return _fn("load_definition_file", "load_file", "read_definition")(files[0])
    except ImportError:
        import tomllib

        return tomllib.loads(files[0].read_text())


# ==== ASSUMED DEFINITION SHAPE (SHAPE: adapt here when the schema lands) ===========================================

def D(blocks, long=None, short=None, exits=None, **extra) -> dict:
    d = {"version": 1, "name": "qa-acceptance", "reason": "QA acceptance case, written before it is run.",
         "blocks": blocks}
    if long:
        d["long"] = long
    if short:
        d["short"] = short
    if exits:
        d["exits"] = exits
    d.update(extra)
    return d


def B(kind, timeframe=None, input=None, **settings) -> dict:  # noqa: A002 - the schema's word
    b = {"kind": kind, **settings}
    if timeframe:
        b["timeframe"] = timeframe
    if input:
        b["input"] = input  # block chaining [B6]: this block is fed another block's output
    return b


def C(left, op, right) -> dict:
    """One condition. Operands: a number, "close"/"open"/"high"/"low"/"volume", a block id or "id.output",
    {"prev": x}, {"add"/"sub"/"mul"/"div": [a, b]} (A4), {"time": "hour"|"weekday"} in UTC (A6)."""
    return {"left": left, "op": op, "right": right}


def ALL(*rules) -> dict:
    return {"all": list(rules)}


def SETUP(setup, trigger, expire_bars, expire_timeframe, cancel=None) -> dict:
    r = {"setup": setup, "trigger": trigger, "expire_after": {"bars": expire_bars, "timeframe": expire_timeframe}}
    if cancel:
        r["cancel"] = cancel
    return r


def SIDE(entry, exit=None, **extra) -> dict:  # noqa: A002
    s = {"entry": entry}
    if exit:
        s["exit"] = exit
    s.update(extra)
    return s


ALWAYS = C("close", ">", 0)
NEVER = C("close", "<", 0)


# ==== data and runs =================================================================================================

def _minutes(n=4000, seed=5, start=START, dips=True) -> pd.DataFrame:
    """1-minute bars stamped at their close: a drifting walk with sharp heavy-volume dips every 170 minutes (so both
    legacy RSI models trade on it: measured on main, rsi_pullback 12 fills, rsi_bands 62)."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.00008, 0.0012, n)
    vol = rng.uniform(0.5, 1.5, n)
    if dips:
        for k in range(300, n, 170):
            r[k:k + 6] -= 0.0025
            vol[k:k + 6] *= 4
    c = 60_000 * np.exp(np.cumsum(r))
    o = np.concatenate([[c[0]], c[:-1]])
    h = np.maximum(o, c) * (1 + rng.uniform(0, 4e-4, n))
    lo = np.minimum(o, c) * (1 - rng.uniform(0, 4e-4, n))
    idx = pd.date_range(pd.Timestamp(start, tz="UTC") + pd.Timedelta(minutes=1), periods=n, freq="1min")
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": vol * 1e3}, index=idx)


def _neutral_start(df: pd.DataFrame, bars: int = 2100) -> pd.DataFrame:
    """The first `bars` bars made neutral (closes alternating 0.05% either side, steady volume): no band, dip or
    volume spike, so neither a legacy model nor its port can act before every block has settled. RSI(14) settles at
    140 bars and EMA(200) at 2,000 [A5]; today's rsi_bands first trades at bar 34 and rsi_pullback at bar 984 on
    _minutes(), on values not yet settled (see README, Needs Advisor)."""
    df = df.copy()
    base = float(df["close"].iloc[bars])
    c = base * (1 + 0.0005 * np.where(np.arange(bars) % 2, 1.0, -1.0))
    o = np.concatenate([[base], c[:-1]])
    df.iloc[:bars, df.columns.get_indexer(["open", "high", "low", "close", "volume"])] = np.column_stack(
        [o, np.maximum(o, c), np.minimum(o, c), c, np.full(bars, 1e3)])
    df.iloc[bars, df.columns.get_loc("open")] = c[-1]
    return df


def _daily(prices, closes, lows=None, opens=None) -> pd.DataFrame:
    from test_backtest import _path

    df = _path(prices, closes)  # open = the close before; high and low span open and close
    for i, v in (opens or {}).items():
        df.iloc[i, df.columns.get_loc("open")] = v
    df["high"] = df[["open", "close"]].max(axis=1)
    df["low"] = df[["open", "close"]].min(axis=1)
    for i, v in (lows or {}).items():  # a wick below the body
        df.iloc[i, df.columns.get_loc("low")] = v
    return df


def _run(defn, df, instrument, minutes=1, params=None, **kw):
    from sleeve_fund.research.runner import run_backtest

    return run_backtest(RULES, df, instrument, params={**_params(defn), **(params or {})}, bar_minutes=minutes,
                        half_spread=0, **kw)


def _fills(res) -> list[tuple]:
    """(intent, side, close time, qty, price) per fill, in time order."""
    if res.fills is None or len(res.fills) == 0:  # a case that rightly trades nothing (HoE/Platform 1, 6 Oct)
        return []
    f = res.fills.sort_values("ts_last")
    return [(res.decisions[o]["intent"], f.loc[o, "side"], f.loc[o, "ts_last"], float(f.loc[o, "filled_qty"]),
             float(f.loc[o, "avg_px"])) for o in f.index]


def _entries(res):
    return [x for x in _fills(res) if x[0] == "entry"]


def _stream(block, df) -> list:
    """The block's values after each bar, None until it reads as settled (the reference the evaluator must keep)."""
    out = []
    for ts, r in df.iterrows():
        block.update_ohlcv(r.open, r.high, r.low, r.close, r.volume, ts.value)
        out.append(dict(block.values) if block.settled else None)
    return out


def _store(tmp_path):
    from sleeve_fund.store import Store

    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        from sleeve_fund.store import make_engine, metadata

        engine = make_engine(url)
        metadata.drop_all(engine)
        return Store(engine=engine)
    return Store(f"sqlite:///{tmp_path}/t.db")


# ==== settled, never initialized [A5] =============================================================================

@pytest.mark.parametrize("block, settled_after", [
    (B("rsi", period=14), "rsi"),
    (B("sma", period=20), "sma"),
    (B("atr_sma", period=14), "atr_sma"),
    (B("vwap", anchor="day"), "vwap"),
    ("chained", "chained"),  # [B6] block chaining: SMA(3) of RSI(5) settles only once its input has
])
def test_the_evaluator_never_acts_on_an_unsettled_indicator(block, settled_after, instrument):
    """A rule that is true whenever the block has any value at all. rsi, sma, atr_sma and day vwap read
    `initialized` (and values) before `settled` [A5], so an evaluator on `initialized` enters early."""
    from sleeve_fund.strategies.indicators import Rsi, Sma, make_block

    df = _minutes(n=2200)
    if block == "chained":
        defn = D({"r": B("rsi", period=5), "x": B("sma", period=3, input="r")}, long=SIDE(C("x", ">", -1), NEVER))
        rsi, sma, first = Rsi(5), Sma(3), None
        for i, (ts, r) in enumerate(df.iterrows()):
            rsi.update_ohlcv(r.open, r.high, r.low, r.close, r.volume, ts.value)
            if rsi.settled:
                sma.update_raw(rsi.value)
                if sma.settled and first is None:
                    first = i
    else:
        defn = D({"x": block}, long=SIDE(C("x", ">", -1), NEVER))
        ref = _stream(make_block(block["kind"], **{k: v for k, v in block.items() if k != "kind"}), df)
        first = next(i for i, v in enumerate(ref) if v is not None)
    (entry, *_) = _entries(_run(defn, df, instrument))
    assert entry[2] >= df.index[first], f"entered at {entry[2]}, before {settled_after} settled at {df.index[first]}"


# ==== no look-ahead ================================================================================================

def test_a_scrambled_future_gives_identical_decisions(instrument):
    """The prefix check of tests/test_indicators_ref.py::test_no_look_ahead, on decisions: every fill up to the cut
    is the same when the bars after it are the real ones or a scramble of them. The definition chains blocks, reads
    a 15-minute timeframe, a swing-confirmed block and a gate."""
    df = _minutes(n=3000)
    defn = D({"rsi": B("rsi", period=14), "ema": B("ema", period=50), "bb": B("bollinger", period=20),
              "rsi15": B("rsi", period=5, timeframe="15m"), "div": B("rsi_divergence"),
              "rsi_avg": B("sma", period=3, input="rsi")},
             long=SIDE(ALL(C("rsi", "crosses_above", 35), {"only_when": C("rsi15", "<", 70), "then": ALWAYS}),
                       {"any": [C("bb.pct_b", ">", 0.9), C("rsi_avg", ">", 65), C("div.bearish", ">", 0)]}))
    full = _fills(_run(defn, df, instrument))
    assert len(full) >= 6, "the case needs trades to compare"
    rng = np.random.default_rng(1)
    for cut in (900, 1700, 2500):
        future = df.iloc[cut:].copy()
        future[:] = future.to_numpy()[rng.permutation(len(future))]
        future["high"] = future[["open", "high", "close"]].max(axis=1)
        future["low"] = future[["open", "low", "close"]].min(axis=1)
        scrambled = _fills(_run(defn, pd.concat([df.iloc[:cut], future]), instrument))
        seen = df.index[cut - 1]
        assert [x for x in scrambled if x[2] <= seen] == [x for x in full if x[2] <= seen]


# ==== time stop across missing bars [B6] ==========================================================================

def test_a_time_stop_is_wall_clock_by_default_and_missing_bars_never_lengthen_a_hold(instrument):
    """[R 5.2] A 6-minute time stop on 1-minute bars with 600 minutes missing: every leg ends on the first bar at or
    after entry + 6 minutes, also when fewer than 6 bars came in between (a bar count would hold on)."""
    df = _minutes(n=3000)
    rng = np.random.default_rng(4)
    df = df.drop(df.index[np.sort(rng.choice(np.arange(100, 2950), 600, replace=False))])
    defn = D({"rsi": B("rsi", period=5)}, long=SIDE(C("rsi", "crosses_above", 30), NEVER),
             exits={"time_stop": {"minutes": 6}})
    fills = _fills(_run(defn, df, instrument))
    trips = list(zip(fills[0::2], fills[1::2]))
    assert len(trips) >= 10 and all(e[0] == "entry" and x[1] == "SELL" for e, x in trips)
    spanned_a_hole = 0
    for e, x in trips:
        deadline = e[2] + pd.Timedelta(minutes=6)
        assert x[2] == df.index[df.index >= deadline][0], (e[2], x[2])
        spanned_a_hole += ((df.index > e[2]) & (df.index <= x[2])).sum() < 6
    assert spanned_a_hole, "the case needs a hold with missing bars inside it"


def _exit_candle_reentries(fills):
    """Indices of entries that open, in the exit's own direction, at the close time of the fill that flattened."""
    pos, last_exit, out = 0.0, None, []
    for i, (_intent, side, ts, qty, _px) in enumerate(fills):
        before, pos = pos, pos + (qty if side == "BUY" else -qty)
        if abs(pos) < 1e-12:
            pos = 0.0
        if before and not pos:
            last_exit = (ts, before > 0)
        elif not before and pos and last_exit == (ts, pos > 0):
            out.append(i)
    return out


def _renewed_holds(fills, df, bars):
    """Indices of exits more than `bars` decision bars after their entry: the time stop fired and the same side was
    taken again on that candle, which nets to no trade, so the hold just carries on (PE1, 6 Oct 22:57)."""
    return [i for i in range(1, len(fills)) if fills[i - 1][0] == "entry" and fills[i][0] != "entry"
            and ((df.index > fills[i - 1][2]) & (df.index <= fills[i][2])).sum() > bars]


# P1-LEG-RE-1 parity pin, moved here from the LEG-RE master (HoE 7 Oct: #172 merges before #161)
# (it replaces test_the_rsi_cross_port_counts_bars_and_matches_rsi_cross_exactly_across_missing_bars, old copy md5
# c4e6deae; HoE OK 7 Oct 10:36 UK). UNMARKED control: passes once #172's legacy fix and #161's port are both on main.
def _same_side_reentries(fills, minutes):
    """Entries in a leg's own direction that fill on the decision candle (close - minutes, close] in which that leg's
    flattening fill happened. Candle-aware, so it holds for 15-minute and 4-hour decisions with 1-minute execution."""
    def candle(t):
        return pd.Timestamp(t).ceil(f"{minutes}min")

    out, last = [], None
    pos = 0.0
    for i, (_intent, side, t, qty) in enumerate(fills):
        before, pos = pos, pos + (qty if side == "BUY" else -qty)
        if abs(pos) < 1e-9:
            pos = 0.0
        if before and not pos:
            last = (candle(t), before > 0)
        elif not before and pos and last == (candle(t), pos > 0):
            out.append(i)
    return out


def test_the_rsi_cross_port_matches_rsi_cross_exactly_again(instrument):
    """The p1-5 rsi_cross pin 1 flips back to exact parity when LEG-RE lands (HoQA 22:40: it was amended to 'matches
    until legacy re-enters on an exit candle'). Same data as that pin: 80 rising bars, 60 minutes missing, time stop
    6 bars counted. Exact equality of every fill (intent, side, time, quantity, price). Passes today because legacy and
    the port both swallow the exit alike; after LEG-RE both must stop re-entering and still agree to the last fill."""
    from sleeve_fund.research.runner import run_backtest

    rng = np.random.default_rng(9)
    df = _minutes(n=3000)
    rise = np.linspace(0.98, 1.0, 81)[:-1]
    df.iloc[:80, df.columns.get_indexer(["open", "high", "low", "close"])] = (
        df["close"].iloc[80] * rise[:, None] * np.array([1.0, 1.0001, 0.9999, 1.0001]))
    df = df.drop(df.index[np.sort(rng.choice(np.arange(200, 2900), 60, replace=False))])
    legacy = run_backtest("rsi_cross", df, instrument, {"rsi_period": 5, "time_stop_bars": 6, "trend_sma": 0, **PERP},
                          bar_minutes=1, half_spread=0)
    defn = D({"rsi": B("rsi", period=5)},
             long=SIDE(C("rsi", "crosses_above", 30), C("rsi", ">=", 55)),
             short=SIDE(C("rsi", "crosses_below", 70), C("rsi", "<=", 50)),
             exits={"time_stop": {"bars": 6, "count": "bars"}})
    try:
        got, leg = _fills(_run(defn, df, instrument, params=PERP)), _fills(legacy)
    except ImportError as e:  # a head without the rule builder
        raise AssertionError(f"not built: {e}") from e
    assert len(leg) > 100 and got == leg
    assert _same_side_reentries([(x[0], x[1], x[2], x[3]) for x in leg], 1) == []


# ==== exit at an indicator level, gapped through [B6] =============================================================

def test_an_exit_at_an_indicator_level_gapped_through_fills_on_that_bar_no_better_than_its_open(prices, instrument):
    """A long held above its 20-bar SMA; one bar opens at 85 and closes at 80, far below the SMA near 107. By
    default the exit may rest at the level set at the previous close or check on the close [R 5.3]: either way it
    exits on that bar, at the open or worse, never at the level it gapped through."""
    closes = [100.0 + 0.5 * i for i in range(30)] + [80.0] * 5
    df = _daily(prices, closes, opens={30: 85.0})
    defn = D({"sma": B("sma", period=20)}, long=SIDE(ALWAYS), exits={"exit_at_level": {"level": "sma"}})
    res = _run(defn, df, instrument, minutes=1440)
    (first_exit, *_) = [x for x in _fills(res) if x[1] == "SELL"]
    assert first_exit[2] == df.index[30]
    assert float(df["low"].iloc[30]) <= first_exit[4] <= 85.0


def _rising_with_wick(prices, low15):
    """Daily closes 100, 101, ... 119 (open = the close before). SMA(5) at the close of bar 14 is 112 and of bar 15
    is 113. Bar 15 opens at 114, closes at 115, and dips to `low15`."""
    return _daily(prices, [100.0 + i for i in range(20)], lows={15: low15})


@pytest.mark.parametrize("rest", [None, "previous_close"])
def test_an_exit_at_an_indicator_level_never_uses_the_current_bars_own_level(rest, prices, instrument):
    """[R 5.3] Bar 15 dips to 112.5: under the level set at the previous close (112) and above the close (115), but
    below the SMA that includes bar 15's own close (113). Resting at the previous close's level or checking on the
    close, nothing exits; only the forbidden current-bar level would."""
    level = {"level": "sma"} if rest is None else {"level": "sma", "rest": rest}
    defn = D({"sma": B("sma", period=5)}, long=SIDE(ALWAYS), exits={"exit_at_level": level})
    res = _run(defn, _rising_with_wick(prices, 112.5), instrument, minutes=1440)
    assert _entries(res) and not [x for x in _fills(res) if x[1] == "SELL"]


def test_the_explicit_resting_exit_fills_at_the_previous_closes_level_and_current_bar_resting_is_refused(prices,
                                                                                                       instrument):
    """[R 5.3] Bar 15 dips to 111, under the 112 set at bar 14's close, and closes at 115: the resting exit fills
    at 112 inside bar 15. A definition asking to rest on the current bar's own level is refused."""
    defn = D({"sma": B("sma", period=5)}, long=SIDE(ALWAYS),
             exits={"exit_at_level": {"level": "sma", "rest": "previous_close"}})
    df = _rising_with_wick(prices, 111.0)
    (sell, *_) = [x for x in _fills(_run(defn, df, instrument, minutes=1440)) if x[1] == "SELL"]
    assert sell[2] == df.index[15] and sell[4] == pytest.approx(112.0)
    bad = D({"sma": B("sma", period=5)}, long=SIDE(ALWAYS),
            exits={"exit_at_level": {"level": "sma", "rest": "current_bar"}})
    with pytest.raises(ValueError):
        _check(bad, bar_spec="1-DAY-LAST-EXTERNAL")


# ==== setup then trigger on a slower timeframe [S][P14] ===========================================================

def _setup_path():
    """1-minute closes from 00:00 (15-minute candles close at :15, :30, :45, :00). The 00:15-00:30 candle spikes to
    115 inside (00:20) but closes at 104: a peek at the forming candle would arm it. The 00:30-00:45 candle closes
    at 111: armed at 00:45. Then 109 (no trigger: above 108; no new arming: below 110)."""
    c = np.full(180, 100.0)
    c[15:30] = 104.0
    c[19] = 115.0  # 00:20, inside a forming candle
    c[23] = 107.0  # 00:24: a trigger level while nothing is armed
    c[30:45] = 109.5
    c[44] = 111.0  # 00:45 closes the candle at 111
    c[45:] = 109.0
    return c


@pytest.mark.parametrize("case", ["fires", "expired", "cancelled", "at_expiry", "re_armed", "trigger_and_cancel"])
def test_setup_then_trigger_arms_only_on_a_closed_slower_candle_fires_inside_its_expiry_and_not_after(case,
                                                                                                    instrument):
    """[S] setup-then-trigger: a condition on the closed 15m candle arms a short, a 1m condition fires it while
    armed, the setup expires after four 15m candles (default) or when a cancel condition on the 15m candle comes
    first. The levels are prices, not the PM's RSI, so each case is exact; the PM's own definition is checked to be
    accepted in test_the_pms_example_definition_is_accepted. [R 5.4]: a trigger exactly at the expiry (01:45, four
    candles after 00:45) counts; a fresh setup (01:30) re-arms and restarts the expiry (to 02:30); a trigger and a
    cancel on the same 1m close (01:00, which closes a 15m candle): the cancel wins."""
    c = _setup_path()
    t = lambda m: pd.Timestamp(START, tz="UTC") + pd.Timedelta(minutes=m)  # noqa: E731
    cancel_below = 105.0
    if case == "fires":
        c[64] = 107.5  # 01:05, one candle after arming
    elif case == "expired":
        c[129] = 107.5  # 02:10: four candles after 00:45 is 01:45
    elif case == "cancelled":
        c[45:60] = 109.2  # 01:00 closes at 109.2: below the cancel level of 109.5 set for this case
        c[64] = 107.5
        cancel_below = 109.5
    elif case == "at_expiry":
        c[104] = 107.5  # 01:45: the fourth 15m close after arming, inclusive
    elif case == "re_armed":
        c[89] = 111.0  # 01:30 closes a 15m candle at 111: a fresh setup, the expiry restarts to 02:30
        c[129] = 107.5  # 02:10: past the first expiry, inside the second
    else:
        c[59] = 104.0  # 01:00: the 1m close triggers (<= 108) and the 15m candle it closes cancels (< 105)
        c[64] = 107.5
    idx = pd.date_range(t(1), periods=len(c), freq="1min")
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e6}, index=idx)
    defn = D({"c15": B("sma", period=1, timeframe="15m")},
             short=SIDE(SETUP(C("c15", ">=", 110.0), C("close", "<=", 108.0), 4, "15m",
                              cancel=C("c15", "<", cancel_below)), C("close", ">=", 1000.0)))
    entries = _entries(_run(defn, df, instrument, params=PERP))
    assert all(e[2] > t(45) for e in entries), "nothing may fire before the setup's 15m candle has closed"
    want = {"fires": [("SELL", t(65))], "at_expiry": [("SELL", t(105))], "re_armed": [("SELL", t(130))]}
    assert [(e[1], e[2]) for e in entries] == want.get(case, [])


def test_the_pms_example_definition_is_accepted():
    """[S] PM, 5 Oct 15:53: 15m RSI(14) at or above 85 arms a short; 1m RSI(14) at or below 70 enters; it expires
    after four 15m candles or when the 15m RSI drops back below 70 first; decided on 1m closes. SHAPE."""
    defn = D({"rsi15": B("rsi", period=14, timeframe="15m"), "rsi1": B("rsi", period=14)},
             short=SIDE(SETUP(C("rsi15", ">=", 85), C("rsi1", "<=", 70), 4, "15m", cancel=C("rsi15", "<", 70)),
                        C("rsi1", "<=", 30)))
    _check(defn, bar_spec="1-MINUTE-LAST-INTERNAL")


# ==== refusals with a reason ======================================================================================

@pytest.mark.parametrize("why", ["cycle", "unknown_block", "unknown_reference", "period_max", "slower_size"])
def test_an_invalid_definition_is_refused_with_its_reason(why):
    from sleeve_fund.strategies.indicators import PERIOD_MAX

    rule = SIDE(C("a", ">", 0))
    defn, named = {
        "cycle": (D({"a": B("sma", period=3, input="b"), "b": B("ema", period=3, input="a")}, long=rule), ["a", "b"]),
        "unknown_block": (D({"a": B("rsi_magic", period=3)}, long=rule), ["rsi_magic"]),
        "unknown_reference": (D({"a": B("rsi", period=3)}, long=SIDE(C("rsi2", ">", 0))), ["rsi2"]),
        "period_max": (D({"a": B("sma", period=PERIOD_MAX + 1)}, long=rule), [f"{PERIOD_MAX}", f"{PERIOD_MAX:,}"]),
        "slower_size": (D({"a": B("sma", period=3, timeframe="25m")}, long=rule), ["25"]),  # [P14] rule 2
    }[why]
    with pytest.raises(ValueError) as refused:
        _check(defn, bar_spec="1-MINUTE-LAST-INTERNAL")
    msg = str(refused.value)
    if why == "cycle":
        assert all(n in msg for n in named), msg
    else:
        assert any(n in msg for n in named), msg


def _past_cap(kind: str, cap: int) -> tuple[int, int]:
    """The longest period of `kind` whose warm-up loads within `cap`, and the next one, which can't."""
    from sleeve_fund.strategies.indicators import BLOCKS, PERIOD_MAX

    cls = BLOCKS[kind]
    lo, hi = 2, PERIOD_MAX
    assert cls.warmup_bars(period=hi) > cap, f"{kind}: every period loads"
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if cls.warmup_bars(period=mid) <= cap else (lo, mid)
    return lo, hi


@pytest.mark.parametrize("kind", ["atr", "ema", "rsi", "atr_sma"])
def test_a_block_whose_warm_up_can_never_load_is_refused_naming_the_cap(kind):
    """[MA7]: Atr's period limit is PERIOD_MAX, but its warm-up past MAX_WARMUP_BARS can never load, so it never
    settles. Generalised to every block: the period one past the loadable one is refused, naming the cap; the
    longest loadable period is accepted."""
    from sleeve_fund.paper.config import MAX_WARMUP_BARS

    ok, never = _past_cap(kind, MAX_WARMUP_BARS)
    with pytest.raises(ValueError) as refused:
        _check(D({"x": B(kind, period=never)}, long=SIDE(C("x", ">", 0))), bar_spec="1-MINUTE-LAST-INTERNAL")
    msg = str(refused.value)
    assert f"{MAX_WARMUP_BARS:,}" in msg or str(MAX_WARMUP_BARS) in msg, msg
    _check(D({"x": B(kind, period=ok)}, long=SIDE(C("x", ">", 0))), bar_spec="1-MINUTE-LAST-INTERNAL")


def test_a_warm_up_past_the_venues_720_candles_but_loadable_says_short_warm_up_and_is_not_refused():
    from sleeve_fund.paper.config import MAX_WARMUP_BARS, VENUE_WARMUP_BARS
    from sleeve_fund.strategies.indicators import BLOCKS

    period = 100
    assert VENUE_WARMUP_BARS < BLOCKS["atr"].warmup_bars(period=period) <= MAX_WARMUP_BARS
    _, notes = _check(D({"x": B("atr", period=period)}, long=SIDE(C("x", ">", 0))), bar_spec="1-HOUR-LAST-EXTERNAL")
    assert "short warm-up" in notes.lower(), notes


# ==== ATR naming [ADV] ============================================================================================

def test_atr_is_wilders_and_atr_simple_average_is_offered_beside_it(tmp_path, instrument):
    """The builder's catalogue offers both labels, and a definition's `atr` reads Wilder's ATR while `atr_sma` reads
    the simple one: their values in the order's lineage payload [DA4] match the library blocks at that bar."""
    from sleeve_fund.paper.runtime import SleeveRuntime
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.strategies.indicators import Atr, AtrSma

    labels = _text({k: v for k, v in vars(_api()).items() if not k.startswith("__")})
    assert "ATR (simple average)" in labels and "ATR" in labels.replace("ATR (simple average)", "")
    # The engine reads prices at the instrument's precision, so the reference must read the same (Platform 1, 6 Oct).
    df = _minutes(n=400).round({k: instrument.price_precision for k in ("open", "high", "low", "close")})
    defn = D({"w": B("atr", period=14), "s": B("atr_sma", period=14)},
             long=SIDE(ALL(C("w", ">", 0), C("s", ">", 0)), NEVER))
    store = _store(tmp_path)
    store.create_sleeve(name="atr-check", strategy=RULES, instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params=_params(defn))
    run_backtest(RULES, df, instrument, params=_params(defn), bar_minutes=1, half_spread=0,
                 runtime=SleeveRuntime(store, "atr-check", tick_seconds=60))
    (entry,) = [o for o in store.orders("atr-check") if o["intent"] == "entry"]
    at = df.index.get_indexer([pd.Timestamp(entry["ts"]).floor("1min")])[0]
    wilder, simple = _stream(Atr(14), df.iloc[: at + 1])[-1], _stream(AtrSma(14), df.iloc[: at + 1])[-1]
    blocks = entry["signal"]["blocks"]
    assert blocks["w"] == pytest.approx(wilder["value"], rel=1e-9)
    assert blocks["s"] == pytest.approx(simple["value"], rel=1e-9)
    assert wilder["value"] != pytest.approx(simple["value"], rel=1e-6)  # the case tells them apart


# ==== ports of the legacy models [S][ADV] =========================================================================

def _replayed(tmp_path, df, strategy, params):
    """The paper path: a session recorded from these 1-minute bars (four trades a minute: open, high, low, close),
    replayed through the paper runtime. Returns what the journal compares (side, intent, type, qty, time)."""
    from nautilus_trader.model import AggressorSide, Price, Quantity, QuoteTick, TradeId, TradeTick

    from sleeve_fund.paper.recorder import Recorder
    from sleeve_fund.research.replay import comparable, replay
    from sleeve_fund.venues import venue

    inst = venue("KRAKEN").instrument("BTC", "USD", price_precision=1)
    path = tmp_path / f"{strategy}.jsonl.gz"
    rec = Recorder(path)
    rec.meta = {"balances": ["10000.00 USD"],
                "sleeve": {"name": "port", "strategy": strategy, "instrument": "BTC/USD",
                           "bar_spec": "1-MINUTE-LAST-INTERNAL", "starting_balance": 10_000,
                           "risk_profile": "balanced", "params": params, "max_notional": None, "maker_fee": "0.004",
                           "taker_fee": "0.008", "tick_seconds": 30}}
    rec.start(inst)
    n = 0
    for ts, r in df.iterrows():
        t0 = ts.value - 60_000_000_000
        for k, px in enumerate((r.open, r.high, r.low, r.close)):
            t = t0 + (k * 15 + 1) * 1_000_000_000
            rec.trade(TradeTick(inst.id, Price(round(px, 1), 1), Quantity(round(r.volume / 4e3, 6), 8),
                                AggressorSide.BUY if n % 2 else AggressorSide.SELL, TradeId(str(n)), t, t + 1000))
            rec.quote(QuoteTick(inst.id, Price(round(px, 1) - 0.5, 1), Price(round(px, 1) + 0.5, 1), Quantity(1, 8),
                                Quantity(1, 8), t + 2000, t + 3000))
            n += 1
    rec.close()
    return comparable(replay(path))


@pytest.mark.parametrize("path", ["backtest", "paper_replay"])
@pytest.mark.parametrize("model", ["rsi_bands", "rsi_pullback"])
def test_each_legacy_model_rebuilt_as_a_definition_makes_exactly_its_trades(model, path, tmp_path, instrument):
    from sleeve_fund.research.runner import run_backtest

    defn = _port_definition(model)
    if path == "backtest":  # [R 5.1a] the longest warm-up (EMA(200): 2,000 bars) preloaded before the window
        full = _minutes(n=6100)
        pre, df = full.iloc[:2100], full.iloc[2100:]
        kw = {_preload_kw(): pre}
        legacy = _fills(run_backtest(model, df, instrument, bar_minutes=1, half_spread=0, **kw))
        port = _fills(_run(defn, df, instrument, **kw))
        assert len(legacy) >= 10, "the case needs trades to compare"
        assert [x[:3] for x in port] == [x[:3] for x in legacy]  # intent, side and bar: exactly
        assert [x[3:] for x in port] == pytest.approx([x[3:] for x in legacy], rel=1e-12)  # qty and price
    else:  # replay() takes no warm-up, so the window opens on 2,100 neutral bars instead [R 5.1]
        df = _neutral_start(_minutes(n=3600))
        legacy = _replayed(tmp_path / "legacy", df, model, {})
        port = _replayed(tmp_path / "port", df, RULES, _params(defn))
        assert len(legacy) >= 2, "the case needs orders to compare"
        assert pd.Timestamp(legacy[0][4]) > df.index[2100], "the legacy model acted inside the neutral warm-up"
        assert port == legacy


def test_legacy_trades_on_unsettled_indicators_are_counted_and_flagged_not_changed(instrument):
    """[R 5.1b] Today rsi_bands first trades at bar 34 of _minutes(), on an RSI(14) that settles at bar 140 (it
    reads `initialized` [A5]). Its result still holds that trade (measured on main), says how many fills were
    decided on unsettled values, and flags exactly those."""
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.strategies.indicators import Rsi

    df = _minutes(n=4000)
    rsi, settled_at = Rsi(14), None
    for i, c in enumerate(df["close"]):
        rsi.update_raw(float(c))
        if rsi.settled:
            settled_at = i
            break
    res = run_backtest("rsi_bands", df, instrument, bar_minutes=1, half_spread=0)
    fills = res.fills.sort_values("ts_last")
    assert fills["ts_last"].iloc[0] == df.index[34], "legacy results are not changed"
    early = [o for o, ts in zip(fills.index, fills["ts_last"]) if ts < df.index[settled_at]]
    early_entries = [o for o in early if res.decisions[o]["intent"] == "entry"]
    assert early_entries
    counts = [getattr(res, a) for a in dir(res) if "unsettled" in a.lower() and not a.startswith("__")]
    assert any(c in (len(early), len(early_entries)) for c in counts if isinstance(c, int)), counts

    def flagged(o):
        return any("unsettled" in str(k).lower() and v for k, v in res.decisions[o].items())

    assert [o for o in fills.index if flagged(o)] == early


# ==== definition identity: hash [HF], DA-4, DA-8 =================================================================

def test_the_definition_hash_is_stable_and_follows_the_content():
    defn = D({"rsi": B("rsi", period=14), "ema": B("ema", period=200)},
             long=SIDE(ALL(C("rsi", "<", 30), C("close", ">", "ema")), C("rsi", ">=", 55)))
    same = _hash(defn)
    assert isinstance(same, str) and len(same) >= 8
    assert _hash(copy.deepcopy(defn)) == same

    def reversed_keys(x):  # a TOML table is an unordered map: the same content written in another order
        if isinstance(x, dict):
            return {k: reversed_keys(x[k]) for k in reversed(list(x))}
        if isinstance(x, list):
            return [reversed_keys(v) for v in x]
        return x

    assert _hash(reversed_keys(defn)) == same
    reordered = copy.deepcopy(defn)  # [R 5.9] all/any in canonical order: sorted, duplicates removed
    reordered["long"]["entry"]["all"] = list(reversed(reordered["long"]["entry"]["all"]))
    assert _hash(reordered) == same
    doubled = copy.deepcopy(defn)
    doubled["long"]["entry"]["all"].append(copy.deepcopy(doubled["long"]["entry"]["all"][0]))
    assert _hash(doubled) == same
    changed = copy.deepcopy(defn)
    changed["long"]["entry"]["all"][0]["right"] = 25
    assert _hash(changed) != same  # a new setting is a new variant for the trials register


def test_the_venues_daily_anchor_is_part_of_the_definition_hash(monkeypatch):
    import dataclasses
    from datetime import time, timedelta

    from sleeve_fund.venues import VENUES

    defn = D({"d": B("sma", period=20, timeframe="1d")}, long=SIDE(C("close", ">", "d")))
    profile = VENUES["KRAKEN"]
    names = [f.name for f in dataclasses.fields(profile) if "anchor" in f.name.lower()]
    assert names, "no daily anchor property"
    at_midnight = _hash_on(defn, "KRAKEN")
    now = getattr(profile, names[0])
    eight = (timedelta(hours=8) if isinstance(now, timedelta) else time(8, 0) if isinstance(now, time)
             else "08:00" if isinstance(now, str) else type(now)(480))
    monkeypatch.setattr(profile, names[0], eight)
    assert _hash_on(defn, "KRAKEN") != at_midnight


def test_every_rules_order_carries_the_lineage_payload(tmp_path, instrument):
    from sleeve_fund.paper.runtime import SleeveRuntime
    from sleeve_fund.research.runner import run_backtest

    df = _minutes(n=1500)
    defn = D({"rsi": B("rsi", period=5), "rsi15": B("rsi", period=3, timeframe="15m")},
             long=SIDE(ALL(C("rsi", "crosses_above", 35), C("rsi15", ">", 0)), C("rsi", ">=", 60)))
    store = _store(tmp_path)
    store.create_sleeve(name="lineage", strategy=RULES, instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params=_params(defn))
    run_backtest(RULES, df, instrument, params=_params(defn), bar_minutes=1, half_spread=0,
                 runtime=SleeveRuntime(store, "lineage", tick_seconds=60))
    orders = [o for o in store.orders("lineage", limit=1000) if o["intent"] in ("entry", "exit")]
    assert len(orders) >= 4
    for o in orders:
        sig = o["signal"]
        assert sig["v"] == 1 and sig["definition"] == _hash(defn)
        assert isinstance(sig["rule"], str) and sig["rule"]
        assert {"rsi", "rsi15"} <= set(sig["blocks"]) and all(np.isfinite(float(v)) for v in sig["blocks"].values())
        assert {"1m", "15m"} <= set(sig["bars"])
        for first, last in sig["bars"].values():
            assert pd.Timestamp(first) <= pd.Timestamp(last) <= pd.Timestamp(o["ts"])
        assert sig["data"]["source"] in ("hub", "store", "venue_rest")
        # DA-4 owner ruling (DA 21:54, CR minor 1 on #161): None (unknown) until the provenance read lands, never a
        # hard-coded False claiming a clean range nobody checked. When provenance lands this pin moves to `is False`
        # (this case refills nothing), with a refilled case pinned `is True`.
        assert sig["data"]["refilled_in_range"] is None


def test_a_rules_strategy_keeps_its_definition_identity_and_new_tables_key_on_an_immutable_id():
    from sleeve_fund.store import metadata

    defn = D({"rsi": B("rsi", period=14)}, long=SIDE(C("rsi", "<", 30), C("rsi", ">=", 55)))
    params = _params(defn)
    assert _hash(defn) in _text(params) and "1" in _text(params)  # version and content hash travel with it
    baseline = {  # main 0ceac6f, plus the tables other open items add
        "account_keys", "account_retired", "accounts", "alert_acks", "backtests", "commands", "decisions",
        "demo_mirror", "equity", "events", "exit_plans", "fee_schedules", "feed_seen", "fills", "funding",
        "history_requests", "insurance", "mirror_requests", "orders", "signal_state", "sleeve_accounts",
        "sleeve_archive", "sleeve_venues", "sleeves", "spreads", "strategy_reset_holds", "strategy_resets", "trials",
        "order_timing", "holdout_locks", "research_registrations", "research_runs", "research_ideas",
        "study_returns"}
    for table in metadata.sorted_tables:
        if table.name in baseline or not any(w in table.name for w in ("rule", "definition")):
            continue
        cols = {c.name: c for c in table.columns}
        assert "sleeve" not in cols, f"{table.name} links a strategy by name"
        for c in table.columns:
            for fk in c.foreign_keys:
                assert fk.column.table.name != "sleeves" or fk.column.name == "id", f"{table.name}.{c.name}"


# ==== board 9b extras ==============================================================================================

def _move_rows(df, every=211, size=0.006):
    """Big one-bar up-moves sprinkled over the walk, so `move > 2 x ATR` and a Z-score above 2 occur."""
    df = df.copy()
    for k in range(400, len(df), every):
        df.iloc[k:, df.columns.get_indexer(["open", "high", "low", "close"])] *= 1 + size
        df.iloc[k, df.columns.get_loc("open")] = df["close"].iloc[k - 1]
        df.iloc[k, df.columns.get_loc("low")] = min(df["low"].iloc[k], df["open"].iloc[k])
    return df


@pytest.mark.parametrize("form", ["move_vs_atr", "z_score"])
def test_arithmetic_rules_enter_exactly_where_the_reference_says(form, instrument):
    from sleeve_fund.strategies.indicators import Atr, Bollinger

    df = _move_rows(_minutes(n=2500, dips=False))
    close, prev = df["close"].to_numpy(), df["close"].shift(1).to_numpy()
    if form == "move_vs_atr":
        ref = _stream(Atr(14), df)
        hit = [v is not None and close[i] - prev[i] > 2 * v["value"] for i, v in enumerate(ref)]
        defn = D({"atr": B("atr", period=14)},
                 long=SIDE(C({"sub": ["close", {"prev": "close"}]}, ">", {"mul": [2, "atr"]}), ALWAYS))
    else:
        ref = _stream(Bollinger(20, 2.0), df)
        hit = [v is not None and (close[i] - v["mid"]) / ((v["upper"] - v["mid"]) / 2.0) > 2 for i, v in enumerate(ref)]
        z = {"div": [{"sub": ["close", "bb.mid"]}, {"div": [{"sub": ["bb.upper", "bb.mid"]}, 2.0]}]}
        defn = D({"bb": B("bollinger", period=20, k=2.0)}, long=SIDE(C(z, ">", 2.0), ALWAYS))
    want = [df.index[i] for i, h in enumerate(hit) if h]
    got = [e[2] for e in _entries(_run(defn, df, instrument))]
    assert len(want) >= 3 and got and got[0] == want[0] and set(got) <= set(want)


def test_calendar_and_time_of_day_conditions_gate_entries(instrument):
    """Long from 08:00 to 12:00 on a weekday, flat outside; the gates name no timezone, so UTC [R 5.5]. The data run
    from Friday 00:00 to Sunday: one leg, decided on the 08:00 close and ended on the 12:00 close."""
    df = _minutes(n=3 * 1440 - 10, dips=False)
    hour, weekday = {"time": "hour"}, {"time": "weekday"}
    defn = D({}, long=SIDE(ALL(C(hour, ">=", 8), C(hour, "<", 12), C(weekday, "<", 5)),
                           {"any": [C(hour, ">=", 12), C(hour, "<", 8)]}))
    fills = _fills(_run(defn, df, instrument))
    t = lambda h, m=0: pd.Timestamp(START, tz="UTC") + pd.Timedelta(hours=h, minutes=m)  # noqa: E731
    assert [(x[0], x[2]) for x in fills] == [("entry", t(8)), ("exit", t(12))]


def test_a_local_time_gate_follows_daylight_saving(instrument):
    """[R 5.5] The same 08:00-12:00 weekday gate in Europe/London, over the end of summer time (26 Oct 2025): on
    Friday 24 Oct (BST) it decides on the 07:00 and 11:00 UTC closes, on Monday 27 Oct (GMT) on 08:00 and 12:00."""
    start = pd.Timestamp("2025-10-24", tz="UTC")
    df = _minutes(n=4 * 1440, start=start.value, dips=False)
    hour = {"time": "hour", "tz": "Europe/London"}
    weekday = {"time": "weekday", "tz": "Europe/London"}
    defn = D({}, long=SIDE(ALL(C(hour, ">=", 8), C(hour, "<", 12), C(weekday, "<", 5)),
                           {"any": [C(hour, ">=", 12), C(hour, "<", 8)]}))
    fills = _fills(_run(defn, df, instrument))
    at = lambda d, h: start + pd.Timedelta(days=d, hours=h)  # noqa: E731
    assert [(x[0], x[2]) for x in fills] == [("entry", at(0, 7)), ("exit", at(0, 11)),
                                             ("entry", at(3, 8)), ("exit", at(3, 12))]


def test_a_stop_at_a_structure_level_fills_at_that_level(prices, instrument):
    """The stop is the 10-bar low, as the rule builder defines a channel: the n most recent CLOSED bars [R 5.7 as
    clarified 17:07], never the library block's own offset. The entry (close above 101.5) fills at bar 12's close, so
    bar 12 counts [R 5.8]: the level is the low of bars 3-12, which takes in bar 12's own wick to 94. Leaving the
    entry bar out (bars 2-11, as for an intrabar fill) would give 95. A later bar opens at 105 and trades down to 90
    inside: the stop fills at its level [ENG]."""
    closes = [100.0, 99, 98, 97, 96, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 90, 90]
    df = _daily(prices, closes, lows={12: 94.0})
    with_entry_bar, without = float(df["low"].iloc[3:13].min()), float(df["low"].iloc[2:12].min())
    assert (with_entry_bar, without) == (94.0, 95.0)  # the case tells the two readings apart
    defn = D({"dc": B("donchian", period=10)}, long=SIDE(C("close", ">", 101.5)),
             exits={"stop": {"level": "dc.lower"}})
    fills = _fills(_run(defn, df, instrument, minutes=1440))
    assert fills[0][:3] == ("entry", "BUY", df.index[12])
    sells = [x for x in fills if x[1] == "SELL"]
    assert sells[0][2] == df.index[16] and sells[0][4] == pytest.approx(with_entry_bar)
    assert sells[0][4] != pytest.approx(without)


def test_breakout_slippage_costs_only_entries_on_the_breakout_candle(instrument):
    """A Donchian breakout (close above the high of the 20 bars before it) with a structure stop at the 20-bar low,
    run with 0 and with 25 bp of breakout slippage: the same trades at the same bars (the stop level does not depend on
    the fill); an entry on the bar where the breakout first became true is 25 bp dearer, any other entry and every exit
    (signal or stop) costs the same."""
    df = _minutes(n=6000, dips=False)
    high, close = df["high"].to_numpy(), df["close"].to_numpy()
    true = np.array([i >= 20 and close[i] > high[i - 20:i].max() for i in range(len(df))])
    first_true = {df.index[i] for i in range(1, len(df)) if true[i] and not true[i - 1]}
    # [R 17:07] a channel is the n most recent CLOSED bars: at bar i's close dc.upper takes in bar i itself, so the
    # breakout compares close[i] with the channel as it stood at bar i-1's close, the 20 bars before i (= `true`).
    # This reverses Platform 1's 6 Oct edit, which relied on the library block's offset.
    side = SIDE(C("close", ">", {"prev": "dc.upper"}), C("close", "<", {"prev": "dc.lower"}), breakout=True)
    exits = {"stop": {"level": "dc.lower"}}
    plain = _fills(_run(D({"dc": B("donchian", period=20)}, long=side, exits=exits), df, instrument))
    slipped = _fills(_run(D({"dc": B("donchian", period=20)}, long=side, exits=exits,
                            costs={"breakout_slippage_bp": 25}), df, instrument))
    assert [x[:3] for x in slipped] == [x[:3] for x in plain]
    assert sum(x[0] == "entry" and x[2] in first_true for x in plain) >= 3
    assert any(x[1] == "SELL" and "stop" in x[0] for x in plain), "the case needs a stop exit"
    for a, b in zip(plain, slipped):
        on_breakout = a[0] == "entry" and a[2] in first_true
        # Since #156 (m-G7) the commission's rounding cent is carried in the fill price, so a price can sit up to about
        # 1 cent / qty off the exact figure (seen 3e-7 and 1.2e-6 relative): 2 cents / qty allowed, far below 25 bp
        # (QA, HoE 21:41, PE1's proposal).
        want, tol = a[4] * (1.0025 if on_breakout else 1.0), 0.02 / min(a[3], b[3])
        assert tol < 0.1 * 0.0025 * a[4], ("set-up: a fill too small for the cent allowance to stay below 25 bp", a, b)
        assert b[4] == pytest.approx(want, rel=0, abs=tol), (a, b)


def test_a_channel_trailing_stop_ratchets_up_and_never_follows_the_channel_down(prices, instrument):
    """[R 5.7, clarified 17:07] The trail rests inside bar k on the channel of the n most recent CLOSED bars, k-n to
    k-1 (the bar just closed included), whatever offset the library block uses.
    (1) The 5-bar low, rising to 110: resting in bar 11 it is the low of bars 6-10, 105, so bar 11's wick to 103
    fills it at 105. One bar stale (bars 5-9) would be 104; taking in bar 11 itself (bars 7-11) would be 103 and
    not fill on bar 11 at all. Either off-by-one fails.
    (2) The ratchet, on the 10-bar channel's mid (a channel low cannot fall without first hitting its stop): bar 6
    spikes to 121; entered at bar 12's close (112), the resting mid rises 111.5, 112, 112.5, 113.5 (bars 13-16), then
    falls to 111 in bar 17 once the spike has left. The stop stays at 113.5, so bar 17's wick to 112.5 fills it at
    113.5. A stop that followed the mid down to 111 would not exit at all."""
    closes = [100.0 + i for i in range(11)] + [111.0, 112.0, 113.0, 104.0, 104.0]
    df = _daily(prices, closes, lows={11: 103.0})
    low = df["low"]
    level, stale, ahead = float(low.iloc[6:11].min()), float(low.iloc[5:10].min()), float(low.iloc[7:12].min())
    assert (level, stale, ahead) == (105.0, 104.0, 103.0)
    defn = D({"dc": B("donchian", period=5)}, long=SIDE(ALWAYS),
             exits={"trail": {"level": "dc.lower", "ratchet": True}})
    sells = [x for x in _fills(_run(defn, df, instrument, minutes=1440)) if x[1] == "SELL"]
    assert sells and sells[0][2] == df.index[11], "a stop resting on bars 7-11 (103) would not fill on bar 11"
    assert sells[0][4] != pytest.approx(stale), "one bar stale: the channel must take in bar 10, the bar just closed"
    assert sells[0][4] == pytest.approx(level)

    df = _daily(prices, [100.0 + i for i in range(22)], lows={17: 112.5}, opens={6: 121.0})
    mid = [(float(df["high"].iloc[k - 10:k].max()) + float(df["low"].iloc[k - 10:k].min())) / 2 for k in (16, 17)]
    assert mid == [113.5, 111.0]  # the channel's mid falls between bar 16 and bar 17
    defn = D({"dc": B("donchian", period=10)}, long=SIDE(C("close", ">", 111.5)),
             exits={"trail": {"level": "dc.mid", "ratchet": True}})
    fills = _fills(_run(defn, df, instrument, minutes=1440))
    assert fills[0][:3] == ("entry", "BUY", df.index[12])
    sell = next(x for x in fills if x[1] == "SELL")
    assert sell[2] == df.index[17] and sell[4] == pytest.approx(113.5)
