"""P1-D13 bars-only stop fills: tests written by QA BEFORE the build (pre-build, on main c7a73f1). The open ones are
strict xfails: the Quant Developer makes each one pass and removes its mark before hand-over. The plain tests pin
what already holds and must keep holding through the fix.

Rulings (advisor-rulings.md; the latest wins):
- "P1-D13 bars-only stop fills (Advisor 6 Oct 18:16, to QA)": (1) G1 never judges a bars-only run with resting exits
  (stops, TPs, channel exits): needs 1m execution bars or "not judged: no 1-minute execution data". (2) Every bars-only
  run (preview/study/lab): a stop the bar traded through fills pessimistic: gap at open -> the open; else worse of
  trigger and the bar adverse extreme; then usual stop slippage. (3) Label "bars-only: stop fills pessimistic".
  (4) One-sided parity: bars-only end equity <= 1m end equity (fee tolerance only); fixtures test_d13 + short mirror
  + gap-at-open.
- "P1-D13 open points (Advisor 6 Oct 18:36, to QA; supersedes 18:16 where different)", with the ~18:45 follow-up:
  (a) target G1 fix: fetch 1m bars only for decision bars whose range touches a resting level (stop, TP, resting
  entry, liquidation), exact; budget counts touched bars only. (b) until it ships: OOS windows AND holdout on 1m,
  in-sample may be bars-only pessimistic, labelled. (c) 5m pessimistic as a ONE-WAY test (pass counts; fail re-run at
  1m before dropping); one-sided parity on a 1m sample of every (c) pass. TP fills at trigger, never improvement.
  Stop slippage max(half spread, 0.05%) REPLACES the half spread, backtest = paper; no spread data -> 0.05%. Label
  "execution bars N min: pessimistic fills" for coarser-than-1m. Liquidation check always uses the bar adverse
  extreme. Open past the target -> target first at trigger, stop never triggers; else adverse-first (NA-2). Label
  whenever any fill OR liquidation check relied on intrabar order; engine sets the flag. Falsifier gap: QD study
  code -> trials register. Resting entry + its stop inside one bar: entry fills then stopped out.
- "D13 last seven (Advisor 6 Oct 19:00)": TP pays no half spread (superseded at 19:30, below); liquidation and risk
  stops are resting exits for G1; windows match 1m exactly; liquidation-only label "(liquidation check)"; no label
  unless a level lay in a bar's range; the liquidation MAJOR is the merge precondition.
- "D13 last three (Advisor 6 Oct 19:16)": an exact target touch does not fill (superseded at 19:40, below), the label
  applies; every window starts flat: a STATE rule enters at its first close, a CROSSOVER rule waits for its next
  cross, never force an entry. TP fee: settled by #146 (6fc758a), paper's target is a market exit at its level paying
  TAKER.
- "#146 L12 FINAL (Advisor 6 Oct 19:30)": the backtest books EVERY target fill (touch or gap) at the level minus taker
  slippage max(half spread, 0.05%), with the taker fee; never price improvement, never the bar open. Ordering
  unchanged: open through the target -> target first, stop never triggers.
- "Exact target touch, phase 1 (Advisor 6 Oct 19:40)": an exact touch (high == level) FILLS at the level minus
  max(half spread, 0.05%), taker fee; 19:16 (1) becomes the before-G2 resting-limit rule.
- "Backtest taker fill price (Advisor 6 Oct 20:55)": taker fills pay the spread IN THE PRICE, levels derive from the
  fill, and the spread leaves the separate cost line (no double charge). Entries at bar close: mid +/- half spread,
  NO 0.05% floor. The floor stays for stops, market-on-touch targets and replayed exits. The fee-ladder break-even
  moves only through the level effect.
Also used: NA-2 (17:00) adverse-first; P1-D3 (16:20) isolated gap loss = the whole position margin + fees.

ASSUMED INTERFACES (adapt the names, never the assertions):
- BacktestResult.labels: a list of strings, set by the engine (a plain string `label` is accepted too). A run with
  no execution bars where a fill or a liquidation check relied on intrabar order carries
  "bars-only: stop fills pessimistic"; one on N-minute execution bars (N > 1) carries
  "execution bars N min: pessimistic fills". preview.run returns it anywhere in its dict; the tear sheet prints it.
- StudyResult.not_judged contains "no 1-minute execution data" for a study with resting exits and no 1-minute bars
  for its OOS windows; on 5-minute bars a would-be FAIL is NOT JUDGED and says it must be re-run at 1 minute.
- The Research page path (research.run.run_store_study) reads 1-minute bars through HistoryStore.read(..., 1, ...);
  the touched-bars fix (a) is measured by the 1-minute rows it reads there.
- StudyResult.bars_only_gap: the falsifier gap, (1m end equity - bars-only end equity) / starting capital, also
  written by the study to its ledger (the rows the trials register imports) under the key "bars_only_gap".

Run from a checkout (see README.md):
  TEST_DATABASE_URL=... BACKTEST_ISOLATE=0 python -m pytest -q -p no:cacheprovider <this file>
"""
import dataclasses
import json
import math
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
# lib155 and test_degraded_155_qa: beside this folder, or where QA keeps them when this file is copied into tests/.
D155 = next((p for p in (HERE.parent / "v2-p1" / "degraded-155-scripts",
                         Path("/mnt/project-files/sleeve-fund/quant-review/v2-p1/degraded-155-scripts"))
             if (p / "test_degraded_155_qa.py").exists()), None)
assert D155 is not None, "test_degraded_155_qa.py (QA's #155 file) not found"
sys.path.insert(0, str(D155))
sys.path.insert(0, os.path.join(os.getcwd(), "tests"))  # the repo's test helpers, run from the checkout
os.environ.setdefault("BACKTEST_ISOLATE", "0")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from lib155 import backtest, binance_inst, kraken_inst, quiet, resample, synth_1m  # noqa: E402
# The #155 QA file's helpers, by import. `_reg` is its autouse fixture (registers qa_t, puts the funding store in
# tmp_path, refuses venue calls); importing it makes it autouse here too. E rests a stop ENTRY.
from test_degraded_155_qa import PERP, E, EConfig, T, TConfig, _reg, ns, put_rates  # noqa: E402,F401
from sleeve_fund.strategies import REGISTRY  # noqa: E402

REASON = "QA P1-D13: not built yet (Advisor 18:16)"
xf = pytest.mark.xfail(strict=True, reason=REASON)

LABEL = "bars-only: stop fills pessimistic"
LABEL_5M = "execution bars 5 min: pessimistic fills"
LABEL_LIQ = LABEL + " (liquidation check)"  # 19:00: liquidation check only
NO_1M = "no 1-minute execution data"
H = 0.001  # the half spread in most runs here: above the 0.05% floor
FLOOR = 0.0005  # stop slippage floor


def slip(h):
    """Stop slippage (18:36): max(half spread, 0.05%), REPLACING the half spread."""
    return max(h, FLOOR)


TICK = 0.1
P0 = 60_000.0
FLAT = (P0, P0, P0, P0)


class S(T):
    """qa_t for spot (long only): long from the bar closing at `at` until `until`."""

    def want_long(self, bar):
        return self._cfg.at <= bar.ts_event < self._cfg.until


class ES(E):
    """E (a resting stop entry placed at the close of `place_at`) that also plans its stop_loss, so the stop rests
    the moment the entry fills, as a strategy's own entry would."""

    def on_bar(self, bar):
        if bar.ts_event == self._cfg.place_at:
            self._stop_frac = self._cfg.stop_loss
        super().on_bar(bar)


class XConfig(TConfig):
    def __init__(self, *, level: float = 0.0, **kw):
        super().__init__(**kw)
        self.level = level


class X(T):
    """A CROSSOVER rule (19:16): wants `side` on the bar whose close crosses up through `level` from at or below it,
    and otherwise asks for no change (None), so it holds what it has. From flat with the price already above the
    level it never enters until the next cross."""

    def want_side(self, bar):
        close = bar.close.as_double()
        prev, self._prev = getattr(self, "_prev", None), close
        return self._cfg.side if prev is not None and prev <= self._cfg.level < close else None


class PConfig(TConfig):
    def __init__(self, *, period: int = 5, **kw):
        super().__init__(**kw)
        self.period = period


class P(T):
    """Spot, signal exits only (nothing rests): long for `period` days, flat for `period` days, by the UTC day
    number of the decision bar's close."""

    def want_long(self, bar):
        return (bar.ts_event // 86_400_000_000_000 // self._cfg.period) % 2 == 0


@pytest.fixture(autouse=True)
def _mine(monkeypatch):
    monkeypatch.setitem(REGISTRY, "qa_d13s", (S, TConfig))
    monkeypatch.setitem(REGISTRY, "qa_d13e", (ES, EConfig))
    monkeypatch.setitem(REGISTRY, "qa_d13x", (X, XConfig))
    monkeypatch.setitem(REGISTRY, "qa_d13p", (P, PConfig))


# ---------------------------------------------------------------------------------------------------------------
# Helpers: hand-made daily bars. Entry at the close of the 2nd bar (60,000); the 4th bar is the one under test.
# ---------------------------------------------------------------------------------------------------------------

def _days(rows, start="2025-10-01"):
    idx = pd.date_range(pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=1), periods=len(rows), freq="1D")
    df = pd.DataFrame([tuple(map(float, r)) for r in rows], columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1e9  # never short of volume: every fill is whole
    return df


def _run(test_bar, side=1, perp=True, after=None, h=H, **exits):
    """Bars-only run (no execution bars), research mode (no risk profile), as a study's runs are."""
    after = after if after is not None else (test_bar[3],) * 4
    d = _days([FLAT, FLAT, FLAT, test_bar, after])
    params = {**(PERP if perp else {}), "at": d.index[1].value, "side": side, **exits}
    r = backtest(binance_inst() if perp else kraken_inst(), d, strategy="qa_t" if perp else "qa_d13s", params=params,
                 minutes=1440, profile=None, half_spread=h)
    return r, d


def _fills(r):
    """[(intent, ts, side, avg_px after the spread, qty)] oldest first."""
    out = []
    for coid, f in r.fills.iterrows():
        if float(f["filled_qty"]) > 0:
            out.append((r.decisions.get(str(coid), {}).get("intent"), pd.Timestamp(f["ts_last"]), str(f["side"]),
                        float(f["avg_px"]), float(f["filled_qty"])))
    return sorted(out, key=lambda x: x[1])


def _exit(r):
    fills = _fills(r)
    assert not r.handler_errors, r.handler_errors
    assert fills and fills[0][0] == "entry", fills  # setup: the entry filled
    exits = fills[1:]
    assert len(exits) == 1, fills  # one exit
    return exits[0]


def _stop_px(base, side, h=H):
    """A stop's fill at `base` after its slippage, which replaces the half spread (a long's exit sells)."""
    return base * (1 - slip(h)) if side > 0 else base * (1 + slip(h))


def _near(px, want):
    return abs(px - want) <= TICK


def _booked(side, h=H):
    """20:55: a market entry at the bar's close books at mid +/- the half spread, no floor (60,060 / 59,940 at 0.1%)."""
    return P0 * (1 + side * h)


def _stop_level(side, h=H, frac=0.02):
    """20:55 "levels derive from the fill": a 2% stop from the BOOKED entry (58,858.8 long / 61,138.8 short at 0.1%)."""
    return _booked(side, h) * (1 - side * frac)


def _tp_level(side, h=H, frac=0.03):
    """A 3% target from the BOOKED entry (61,861.8 long / 58,141.8 short at 0.1%; 61,812.36 long at 0.02%)."""
    return _booked(side, h) * (1 + side * frac)


def _touching(level, side):
    """A bar extreme that just reaches a stop at `level` whatever tick rounding the build applies to it: a long's low
    floored to the tick, a short's high ceiled (so within one tick past the level, never short of it). The fill is
    then pinned on this extreme (the worse of the trigger and the extreme), so the build's rounding of the level
    cannot move it."""
    return math.floor(level / TICK) * TICK if side > 0 else math.ceil(level / TICK) * TICK


# ---------------------------------------------------------------------------------------------------------------
# The stop fill on bars only (18:16 (2); slippage 18:36)
# ---------------------------------------------------------------------------------------------------------------

THROUGH = {
    # The stop (2% from the booked entry: 58,858.8 long, 61,138.8 short) is traded through inside the bar, to
    # 57,000 / 63,000;
    # the bar then closes back on the right side of the stop, so only the extreme says how far it went.
    "long-perp": (1, True, (60_000, 60_050, 57_000, 59_900), 57_000.0),
    "short-perp": (-1, True, (60_000, 63_000, 59_950, 60_100), 63_000.0),
    "long-spot-1x": (1, False, (60_000, 60_050, 57_000, 59_900), 57_000.0),
}


@pytest.mark.parametrize("case", list(THROUGH))
def test_a_stop_traded_through_inside_the_bar_fills_at_the_bars_extreme_plus_slippage(case):
    """Not a gap: the worse of the trigger and the bar's adverse extreme, then the slippage. Today: the trigger."""
    side, perp, bar, extreme = THROUGH[case]
    r, d = _run(bar, side=side, perp=perp, stop_loss=0.02)
    intent, ts, _, px, _ = _exit(r)
    assert intent == "stop_loss" and d.index[2] <= ts <= d.index[3], (intent, ts)  # on the bar under test
    assert _near(px, _stop_px(extreme, side)), (px, _stop_px(extreme, side), "trigger", _stop_level(side))


GAP = {
    # The bar opens beyond the stop (57,000 / 63,000) and trades further still (56,500 / 63,500): the open.
    "long-perp": (1, True, (57_000, 57_100, 56_500, 56_800), 57_000.0),
    "short-perp": (-1, True, (63_000, 63_500, 62_900, 63_200), 63_000.0),
    "long-spot-1x": (1, False, (57_000, 57_100, 56_500, 56_800), 57_000.0),
}


@pytest.mark.parametrize("case", list(GAP))
def test_a_stop_gapped_at_the_open_fills_at_the_open_not_the_extreme(case):
    """"gap at open -> the open", then the slippage (= the 0.1% half spread here). Holds today."""
    side, perp, bar, opened = GAP[case]
    r, _ = _run(bar, side=side, perp=perp, stop_loss=0.02)
    intent, _, _, px, _ = _exit(r)
    assert intent == "stop_loss"
    assert _near(px, _stop_px(opened, side)), (px, _stop_px(opened, side))


TOUCH = {
    # The extreme equals the trigger exactly: 2% from the BOOKED entry (20:55), 58,858.8 / 61,138.8.
    "long-perp": (1, (60_000, 60_050, _touching(_stop_level(1), 1), 59_000), _stop_level(1)),
    "short-perp": (-1, (60_000, _touching(_stop_level(-1), -1), 59_950, 61_000), _stop_level(-1)),
}


@pytest.mark.parametrize("case", list(TOUCH))
def test_a_stop_just_touched_fills_at_its_trigger_plus_slippage(case):
    """The worse of the trigger and the extreme, when they are (to the tick) equal, then the slippage: about
    58,858.8 x 0.999 = 58,799.94 / 61,138.8 x 1.001 = 61,199.94, pinned on the bar's extreme (the level floored /
    ceiled to the tick, so the build's rounding of the level cannot move it). The trigger is 2% from the booked entry
    (20:55; v9: it was from mid, 58,800 / 61,200). Today the stop rests at 58,800 / 61,200 (from mid), which these
    bars never reach: no exit."""
    side, bar, trigger = TOUCH[case]
    extreme = bar[2] if side > 0 else bar[1]  # the level, to the tick (within 0.1 past it)
    r, _ = _run(bar, side=side, stop_loss=0.02)
    intent, _, _, px, _ = _exit(r)
    assert intent == "stop_loss"
    assert _near(px, _stop_px(extreme, side)), (px, _stop_px(extreme, side), "level", trigger)


SLIP_FLOOR = {
    # half spread given -> what the stop pays: the 0.05% floor below it.
    "zero-spread": 0.0,
    "spread-below-the-floor": 0.0002,
}


@pytest.mark.parametrize("case", list(SLIP_FLOOR))
def test_a_stop_pays_at_least_the_005pct_slippage_floor(case):
    """18:36: stop slippage max(half spread, 0.05%) replaces the half spread. A just-touched long stop (2% from the
    booked entry: 58,800 at 0%, 58,811.76 at 0.02%; the bar's low is that level floored to the tick) fills at the low
    x (1 - 0.05%): about 58,770.6 / 58,782.3. Today: 58,800 (0%; no slippage at all) or no exit (0.02%: the stop
    rests at 58,800 from mid, below the bar's low)."""
    h = SLIP_FLOOR[case]
    low = _touching(_stop_level(1, h), 1)
    r, _ = _run((60_000, 60_050, low, 59_000), side=1, h=h, stop_loss=0.02)
    intent, _, _, px, _ = _exit(r)
    assert intent == "stop_loss"
    assert _near(px, low * (1 - FLOOR)), (px, low * (1 - FLOOR))


def test_with_no_spread_data_a_stop_pays_005pct():
    """18:36 "no spread data -> 0.05%": no half spread passed, so the venue's assumption (Binance: 0.01%) applies to
    other orders: the entry books at 60,006 (20:55), the 2% stop rests at 58,805.88, and a just-touched stop (low
    58,805.8) pays 0.05%: about 58,776.4. Today the stop rests at 58,800 (from mid), below the bar's low: no exit."""
    from sleeve_fund import markets
    from sleeve_fund.venues import venue as venue_profile

    params = {**PERP, "side": 1, "stop_loss": 0.02}
    assumed = markets.half_spread_for(params, venue_profile("BINANCE").assumed_half_spread, "BINANCE")
    assert 0 < assumed < FLOOR, assumed  # setup: the assumption is under the floor
    low = _touching(_stop_level(1, assumed), 1)
    d = _days([FLAT, FLAT, FLAT, (60_000, 60_050, low, 59_000), (59_000,) * 4])
    r = backtest(binance_inst(), d, strategy="qa_t", minutes=1440, profile=None, half_spread=None,
                 params={**params, "at": d.index[1].value})
    intent, _, _, px, _ = _exit(r)
    assert intent == "stop_loss"
    assert _near(px, low * (1 - FLOOR)), (px, low * (1 - FLOOR))


# ---------------------------------------------------------------------------------------------------------------
# Entries at bar close (Advisor 20:55 "Backtest taker fill price")
# ---------------------------------------------------------------------------------------------------------------

UNDER = 0.0002  # a half spread under the 0.05% floor: entries must NOT be floored (stops and targets are)


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_a_market_entry_at_bar_close_fills_at_mid_plus_the_half_spread_with_no_floor(side):
    """20:55 (1): a market entry at the bar's close fills IN THE PRICE at mid +/- the half spread, with NO 0.05% floor:
    at a 0.02% half spread, 60,012 long / 59,988 short (60,030 / 59,970 would be the floor). No double charge: the
    venue fee is the commission alone (taker x qty x price, the spread not in it), and the drop in equity at the entry
    bar's close equals the price effect plus that fee. v9 (QD, through the coordinator): the fee is on the BOOKED
    notional (taker x qty x booked price; at cent precision it cannot be told from mid here), and the spread is
    recorded separately as information, spread_paid = qty x |booked - mid|, never inside the fee and never charged
    again (the equity check). Holds today (the spread moved into the reported price; spread_paid already exists)."""
    r, d = _run(FLAT, side=side, h=UNDER)
    fills = _fills(r)
    assert fills and fills[0][0] == "entry", fills  # setup
    _, ts, _, px, qty = fills[0]
    mid = P0
    assert _near(px, mid * (1 + side * UNDER)), (px, mid * (1 + side * UNDER))
    fee, _ = _venue_fee(r, "entry")
    from sleeve_fund import markets
    from sleeve_fund.instruments import FeeSchedule

    inst = binance_inst()
    taker = float(markets.fees_for(r.params, FeeSchedule(inst.maker_fee, inst.taker_fee), str(inst.id.venue)).taker)
    assert fee == pytest.approx(taker * qty * px, abs=0.01), (fee, taker, qty, px)  # the fee, no spread in it
    assert r.spread_paid == pytest.approx(qty * abs(px - mid), abs=0.01), (r.spread_paid, qty * abs(px - mid))
    drop = r.starting_capital - float(r.equity.loc[d.index[1]])
    assert drop == pytest.approx(abs(px - mid) * qty + fee, abs=0.02), (drop, abs(px - mid) * qty, fee)


ENTRY_LEVELS = {
    # 0.02% half spread: the entry fills at 60,012 (long) / 59,988 (short). Levels from the FILL price: long stop
    # 2% = 58,811.76, short stop 61,187.76, long target 3% = 61,812.36. From mid they would be 58,800 / 61,200 /
    # 61,800, which these bars never reach (stops) or do reach (target).
    "long-stop-from-the-fill": (1, (60_000, 60_050, 58_811.0, 59_500), {"stop_loss": 0.02}, "stop_loss"),
    "short-stop-from-the-fill": (-1, (60_000, 61_188.5, 59_950, 60_500), {"stop_loss": 0.02}, "stop_loss"),
    "long-target-from-the-fill": (1, (60_000, 61_805.0, 59_950, 61_000), {"take_profit": 0.03}, None),
}


@pytest.mark.parametrize("case", list(ENTRY_LEVELS))
def test_stops_and_targets_derive_from_the_entrys_fill_price_not_mid(case):
    """20:55 (1): "levels derive from the fill". With the entry at 60,012 (a 0.02% half spread), a 2% stop rests at
    58,811.76, so a bar that trades to 58,811 stops it out (short mirror: 61,187.76, a bar to 61,188.5); a 3% target
    rests at 61,812.36, so a bar to 61,805 does NOT fill it. Today the levels derive from mid (58,800 / 61,200 /
    61,800): no stop, and the target fills."""
    side, bar, exits, want = ENTRY_LEVELS[case]
    r, _ = _run(bar, side=side, h=UNDER, **exits)
    fills = _fills(r)
    assert fills and fills[0][0] == "entry" and _near(fills[0][3], _booked(side, UNDER)), fills  # setup
    if want == "stop_loss":
        assert any(d.get("intent") == "stop_loss" for d in r.decisions.values()), r.decisions  # setup: it rested
    else:
        # setup: the target is live (a resting order today, a market-on-touch watch in the build, which journals
        # nothing until it triggers): the same run with the bar through 61,900 fills it
        ctl, _ = _run((bar[0], 61_900.0, bar[2], bar[3]), side=side, h=UNDER, **exits)
        assert [f[0] for f in _fills(ctl)] == ["entry", "take_profit"], _fills(ctl)
    got = [f[0] for f in fills[1:2]]
    assert got == ([want] if want else []), fills


def _tp_fill(level, side, h=H):
    """#146 L12 FINAL (19:30): the backtest books every target fill, touched inside the bar or gapped through, at the
    level MINUS taker slippage max(half spread, 0.05%) (a long's target sells), with the taker fee. Never price
    improvement, never the bar's open."""
    return level * (1 - slip(h)) if side > 0 else level * (1 + slip(h))


def _tp_px(side, h=H):
    """A 3% target's booked fill, the level derived from the BOOKED entry (20:55). Derived by QA, v9:
    long 0.1%: 60,000 x 1.001 = 60,060; x 1.03 = 61,861.8; x 0.999 = 61,799.94.
    short 0.1%: 60,000 x 0.999 = 59,940; x 0.97 = 58,141.8; x 1.001 = 58,199.94.
    long 0.02%: 60,000 x 1.0002 = 60,012; x 1.03 = 61,812.36; x 0.9995 (the floor) = 61,781.45."""
    return _tp_fill(_tp_level(side, h), side, h)


def _tp_never_better(px, level, side):
    return px <= level + TICK if side > 0 else px >= level - TICK


TP_INSIDE = {
    # 3% take-profit from the booked entry: 61,861.8 long, 58,141.8 short (0.1%). The open hasn't reached it; the bar
    # trades through it.
    "long-perp": (1, (60_000, 62_500, 59_950, 62_000)),
    "short-perp": (-1, (60_000, 60_050, 57_500, 58_000)),
}

TP_GAP = {
    # The bar opens past the take-profit (63,000 / 57,000, against 61,861.8 / 58,141.8).
    "long-perp": (1, (63_000, 63_500, 62_900, 63_200)),
    "short-perp": (-1, (57_000, 57_100, 56_500, 56_800)),
}

TP_PRICE = {
    # case: (inside or gap, TP_* key, half spread). H = 0.1% is over the floor, so it is the slippage itself.
    "inside-long-over": ("inside", "long-perp", H),
    "inside-short-over": ("inside", "short-perp", H),
    "gap-long-over": ("gap", "long-perp", H),
    "gap-short-over": ("gap", "short-perp", H),
    "inside-long-under": ("inside", "long-perp", UNDER),
    "gap-long-under": ("gap", "long-perp", UNDER),
}


@pytest.mark.parametrize("case", list(TP_PRICE))
def test_a_take_profit_books_at_its_level_minus_taker_slippage_never_at_the_open(case):
    """#146 L12 FINAL (19:30) and 20:55 (levels from the booked entry): every target fill, touched inside the bar or
    gapped through at the open, books at level x (1 -/+ max(half spread, 0.05%)), the level 3% from the booked entry:
    61,799.94 / 58,199.94 at a 0.1% half spread (over the floor), 61,781.45 at 0.02% (under it). A gap never books at
    the open. Today the level derives from mid: 61,738.21 / 58,258.21 (0.1%), 61,787.65 (0.02%: the half spread, not
    the floor). v9: these pins were derived from mid (61,738.2 / 58,258.2 / 61,769.1)."""
    kind, key, h = TP_PRICE[case]
    side, bar = (TP_INSIDE if kind == "inside" else TP_GAP)[key]
    r, _ = _run(bar, side=side, h=h, take_profit=0.03)
    intent, _, _, px, _ = _exit(r)
    assert intent == "take_profit", _fills(r)
    if kind == "gap":
        assert abs(px - bar[0]) > 100 * TICK, (px, bar[0])  # never the open
    assert _tp_never_better(px, _tp_level(side, h), side)
    assert _near(px, _tp_px(side, h)), (px, _tp_px(side, h))


def _venue_fee(r, intent):
    """(venue fee in quote currency, filled qty) of the order sent for `intent`: the commission alone, the half spread
    having been moved into the price (runner._spread_into_prices)."""
    for coid, f in r.fills.iterrows():
        if r.decisions.get(str(coid), {}).get("intent") == intent and float(f["filled_qty"]) > 0:
            entry = f["commissions"]
            first = (entry if isinstance(entry, (list, tuple)) else [entry])[0]
            return float(str(first).split()[0]), float(f["filled_qty"])
    raise AssertionError(f"no {intent} fill: {_fills(r)}")


@pytest.mark.parametrize("case", [f"inside-{c}" for c in TP_INSIDE] + [f"gap-{c}" for c in TP_GAP])
def test_a_take_profit_pays_the_taker_fee_on_its_level(case):
    """#146 (6fc758a) and L12 FINAL (19:30): the target pays TAKER, never maker, also when the open passed it, on the
    BOOKED notional (v9): taker x qty x booked price, to the cent. Holds today (a limit that is not post-only pays
    taker; at cent precision the booked and engine prices give the same fee here); the price is pinned above."""
    kind, name = case.split("-", 1)
    side, bar = (TP_INSIDE if kind == "inside" else TP_GAP)[name]
    r, _ = _run(bar, side=side, take_profit=0.03)
    assert _exit(r)[0] == "take_profit", _fills(r)  # setup
    from sleeve_fund import markets
    from sleeve_fund.instruments import FeeSchedule

    inst = binance_inst()
    fees = markets.fees_for(r.params, FeeSchedule(inst.maker_fee, inst.taker_fee), str(inst.id.venue))
    assert float(fees.maker) != float(fees.taker), fees  # setup: the two rates are told apart
    fee, qty = _venue_fee(r, "take_profit")
    px = _exit(r)[3]
    assert fee == pytest.approx(float(fees.taker) * qty * px, abs=0.01), (fee, float(fees.taker), qty, px)


OPEN_PAST_TP = {
    # Stop 2% and target 3% from the booked entry (58,858.8 / 61,861.8 long; 61,138.8 / 58,141.8 short). The bar
    # OPENS past the target and later trades through the stop as well.
    "long-open-nearer-the-high": (1, (63_000, 63_500, 58_000, 59_000)),
    "long-open-nearer-the-low": (1, (62_000, 66_500, 58_000, 59_000)),
    "short-open-nearer-the-low": (-1, (57_000, 62_000, 56_500, 61_000)),
    "short-open-nearer-the-high": (-1, (58_000, 62_000, 53_500, 61_000)),
}


@pytest.mark.parametrize("case", list(OPEN_PAST_TP))
def test_when_the_open_is_past_the_target_the_target_fills_first_and_the_stop_never_triggers(case):
    """18:36, unchanged by #146 L12 FINAL (19:30): "open past the target -> target first, stop never triggers";
    whichever extreme is nearer the open. The ordering only (one exit, the target, never better than its level, never
    the open); the price is pinned by the test below. Holds today (the open is matched first)."""
    side, bar = OPEN_PAST_TP[case]
    r, _ = _run(bar, side=side, stop_loss=0.02, take_profit=0.03)
    intent, _, _, px, _ = _exit(r)
    assert intent == "take_profit", _fills(r)
    assert _tp_never_better(px, _tp_level(side), side) and abs(px - bar[0]) > 100 * TICK, (px, bar[0])


@pytest.mark.parametrize("case", list(OPEN_PAST_TP))
def test_when_the_open_is_past_the_target_it_books_at_the_level_from_the_booked_entry(case):
    """L12 FINAL (19:30) and 20:55: the target that fills first books at its level (3% from the booked entry) minus
    the taker slippage: 61,799.94 long / 58,199.94 short. Today: 61,738.21 / 58,258.21 (the level from mid). v9:
    split from the ordering test above, whose price pin was derived from mid."""
    side, bar = OPEN_PAST_TP[case]
    r, _ = _run(bar, side=side, stop_loss=0.02, take_profit=0.03)
    intent, _, _, px, _ = _exit(r)
    assert intent == "take_profit", _fills(r)  # setup: the ordering, pinned above
    assert _near(px, _tp_px(side)), (px, _tp_px(side))


def test_a_channel_exit_traded_through_fills_at_the_bars_low():
    """Channel exits too: the stop resting at the lowest low of the last 3 bars (stop_swing_bars, 59,000) is traded
    through to 56,000 inside the bar. Today it fills at 59,000."""
    rows = [(60_000, 60_100, 59_000, 60_000), (60_000, 60_100, 59_500, 60_000), FLAT,
            (60_000, 60_050, 56_000, 59_900), (59_900,) * 4]
    d = _days(rows)
    r = backtest(binance_inst(), d, strategy="qa_t", minutes=1440, profile=None, half_spread=H,
                 params={**PERP, "at": d.index[1].value, "side": 1, "stop_swing_bars": 3})
    intent, _, _, px, _ = _exit(r)
    assert intent == "stop_loss"
    assert _near(px, _stop_px(56_000.0, 1)), (px, _stop_px(56_000.0, 1), "trigger 59,000")


BOTH = {
    # Stop 2% and take-profit 3% both inside the bar; the open (inside both) is nearer the target, so today's "extreme
    # nearer the open trades first" takes the target. NA-2: the stop, and it fills at the adverse extreme.
    "long-perp": (1, True, (61_000, 62_000, 58_000, 60_000), 58_000.0),
    "long-spot-1x": (1, False, (61_000, 62_000, 58_000, 60_000), 58_000.0),
    "short-perp": (-1, True, (59_000, 62_000, 58_000, 60_000), 62_000.0),
}


@pytest.mark.parametrize("case", list(BOTH))
def test_with_the_stop_and_the_target_both_inside_the_bar_the_stop_fills_first_at_the_extreme(case):
    """Adversarial: the open hasn't passed the target, so adverse-first (NA-2, 18:36). Today the target fills."""
    side, perp, bar, extreme = BOTH[case]
    r, _ = _run(bar, side=side, perp=perp, stop_loss=0.02, take_profit=0.03)
    intent, _, _, px, _ = _exit(r)
    assert intent == "stop_loss", (intent, px)
    assert _near(px, _stop_px(extreme, side)), (px, _stop_px(extreme, side))


ENTRY_AND_STOP = {
    # A resting stop entry placed at the 60,000 close (long: buy at 60,500; short: sell at 59,500), with a 2% stop
    # from its booked fill (a moving-market fill, so with the floor: 60,560.5 / 59,440.5 -> 59,349.29 / 60,629.31;
    # from mid it was 59,290 / 60,690). The next bar trades through both. Closing beyond the stop or back inside it.
    "long-closes-beyond-the-stop": (1, 60_500.0, (60_000, 61_000, 58_000, 59_000), 58_000.0),
    "long-closes-back-inside": (1, 60_500.0, (60_000, 61_000, 58_000, 60_800), 58_000.0),
    "short-closes-beyond-the-stop": (-1, 59_500.0, (60_000, 62_000, 59_000, 61_000), 62_000.0),
    "short-closes-back-inside": (-1, 59_500.0, (60_000, 62_000, 59_000, 59_200), 62_000.0),
}


@pytest.mark.parametrize("case", list(ENTRY_AND_STOP))
def test_a_resting_entry_and_its_stop_inside_one_bar_fill_then_stop_out(case):
    """18:36: "Resting entry + its stop inside one bar: entry fills then stopped out", pessimistically, on that bar.
    Today: closing beyond, the strategy's close check exits at the close (58,941 / 61,061); closing back inside,
    nothing stops it out at all."""
    side, trigger, bar, extreme = ENTRY_AND_STOP[case]
    d = _days([FLAT, FLAT, bar, (bar[3],) * 4, (bar[3],) * 4])
    r = backtest(binance_inst(), d, strategy="qa_d13e", minutes=1440, profile=None, half_spread=H,
                 params={**PERP, "at": d.index[2].value, "side": side, "place_at": d.index[1].value,
                         "trigger": trigger, "stop_loss": 0.02})
    fills = _fills(r)
    assert not r.handler_errors and fills and fills[0][0] == "entry", fills  # setup: the resting entry filled
    assert d.index[1] < fills[0][1] <= d.index[2], fills  # on the bar under test
    stops = [f for f in fills[1:] if d.index[1] < f[1] <= d.index[2]]
    assert [f[0] for f in stops] == ["stop_loss"], fills
    assert _near(stops[0][3], _stop_px(extreme, side)), (stops[0][3], _stop_px(extreme, side))


# --- the liquidation check: always the bar's adverse extreme ---------------------------------------------------

@pytest.fixture
def full_margin(monkeypatch):
    """As the repo's conftest: every profile puts the whole equity up as margin, so a perp sizes at its full leverage
    cap (aggressive: 3x) and liquidation is in reach."""
    from sleeve_fund import risk

    for name, p in list(risk.PROFILES.items()):
        monkeypatch.setitem(risk.PROFILES, name, dataclasses.replace(p, max_position_pct=1.0))


@pytest.fixture
def liquidation_only(monkeypatch):
    """full_margin, and the drawdown halt, daily pause and liquidation cut moved out of the way (99%, 99%, 0), so the
    only resting level a bar can reach is the liquidation price itself."""
    from sleeve_fund import risk

    for name, p in list(risk.PROFILES.items()):
        monkeypatch.setitem(risk.PROFILES, name, dataclasses.replace(
            p, max_position_pct=1.0, max_drawdown=0.99, daily_loss=0.99, min_liquidation_distance=0.0))


def _levered(low=30_000.0, close=59_000.0):
    """A 3x long perp on the aggressive profile, NO stop of its own, daily bars only. The 4th bar trades down to
    `low`, past the liquidation price (about 40,000), and closes back at `close`."""
    rows = [FLAT, FLAT, FLAT, (60_000, 60_050, low, close), (close,) * 4, (close,) * 4]
    d = _days(rows)
    r = backtest(binance_inst(), d, strategy="qa_t", minutes=1440, profile="aggressive", half_spread=H,
                 params={**PERP, "at": d.index[1].value, "side": 1})
    assert not r.handler_errors, r.handler_errors
    return r, d


def test_the_liquidation_check_uses_the_bars_adverse_extreme(full_margin):
    """18:36: "Liquidation check always uses the bar adverse extreme". The low (30,000) is past the liquidation price,
    so the position is liquidated on that bar and loses its whole isolated margin (P1-D3), however the bar closed.
    Today the resting daily-pause stop fills at its level (58,331) and the strategy keeps about 91% of its equity."""
    r, d = _levered()
    fills = _fills(r)
    entry = fills[0]
    assert entry[0] == "entry", fills
    margin = entry[3] * entry[4] / 3  # 3x on the aggressive profile, all of the equity as margin
    after = float(r.equity.loc[d.index[3]])
    assert after <= 10_000 - 0.95 * margin, (after, margin, fills)


# --- the same fill through the other two paths: the backtest page (preview) and a study -----------------------

def _preview_daily(n=70, stop_day=40, bar=(60_000, 60_050, 57_000, 59_900)):
    rows = [FLAT] * n
    rows[stop_day] = bar
    rows[stop_day + 1:] = [(bar[3],) * 4] * (n - stop_day - 1)
    return _days(rows, start="2025-06-01")


def _preview(daily, params, **kw):
    from sleeve_fund.dashboard import preview

    preview._history.clear()
    return quiet(lambda: preview.run("qa_d13s", "BTC/USD", params, venue="KRAKEN", fetch=lambda pair: daily, **kw))


def test_the_backtest_page_fills_a_stop_traded_through_inside_the_bar_at_its_low():
    """Preview (bars only: no stored minutes), at Kraken's assumed 0.05% half spread. Today: the 58,800 trigger."""
    daily = _preview_daily()
    d = _preview(daily, {"at": daily.index[30].value, "stop_loss": 0.02}, detail=True)
    h = d["spread"]["half"]
    sells = [f for f in d["fills"] if f["side"].endswith("SELL")]
    assert len(sells) == 1 and pd.Timestamp(sells[0]["t"]) <= daily.index[40], d["fills"]  # setup: the stop
    assert _near(sells[0]["price"], _stop_px(57_000.0, 1, h)), (sells[0]["price"], _stop_px(57_000.0, 1, h))


def test_a_study_fills_a_stop_traded_through_inside_the_bar_at_its_low():
    """A study: buy and hold with a 3% stop; on day 25 the price trades to 55,000 inside the bar and closes at 59,000.
    Today the full-period run fills at the 58,200 trigger."""
    r, _ = _study("handmade/stop")
    stops = [f for f in _fills(r.full_period) if f[0] == "stop_loss"]
    assert len(stops) == 1, _fills(r.full_period)  # setup: the stop traded once, on day 25
    assert _near(stops[0][3], _stop_px(55_000.0, 1)), (stops[0][3], "trigger 58,200")


# ---------------------------------------------------------------------------------------------------------------
# The label (18:16 (3); 18:36: set by the engine whenever a fill or a liquidation check relied on intrabar order;
# "execution bars N min: pessimistic fills" when the execution bars are coarser than 1 minute)
# ---------------------------------------------------------------------------------------------------------------

def _result_labels(res) -> str:
    labels = getattr(res, "labels", None) or []
    one = getattr(res, "label", None) or ""
    return " | ".join([*map(str, labels), str(one)])


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _strings(k)
            yield from _strings(v)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            yield from _strings(v)


def _labelled(d, label=LABEL) -> bool:
    return any(label in s for s in _strings(d))


def _tape(m1_days=8, start="2025-10-01"):
    """1-minute bars and the daily bars made from them."""
    m1 = synth_1m(days=m1_days, seed=13, start=start, vol_day=0.03)
    return m1, resample(m1, 1440)


def _dip_tape():
    """Six days of 1-minute bars at 60,000 (a cent of wiggle); at 10:03 on the 4th day one minute trades down to
    57,000 and closes back at 59,950. Daily and 5-minute bars made from it."""
    idx = pd.date_range("2025-10-01 00:01", periods=6 * 1440, freq="1min", tz="UTC")
    c = P0 + (np.arange(len(idx)) % 2) * 0.1
    m1 = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e9}, index=idx)
    at = pd.Timestamp("2025-10-04 10:04", tz="UTC")
    m1.loc[at, ["low", "close"]] = [57_000.0, 59_950.0]
    return m1, resample(m1, 1440), resample(m1, 5)


def test_a_bars_only_lab_result_whose_stop_filled_inside_a_bar_is_labelled():
    """run_backtest, no execution bars, the stop traded through inside the bar."""
    r, _ = _run(THROUGH["long-perp"][2], side=1, stop_loss=0.02)
    assert LABEL in _result_labels(r), f"no {LABEL!r} label: labels {getattr(r, 'labels', None)!r}"


def test_a_levered_perp_whose_risk_stop_lay_inside_the_bar_is_labelled(full_margin):
    """No stop of its own, but the risk guard's resting daily-pause stop (and the liquidation price) lay inside the
    bar's range (19:00: label only when some stop/target/entry/liquidation level was within the bar range)."""
    r, _ = _levered()
    assert LABEL in _result_labels(r), f"no {LABEL!r} label: labels {getattr(r, 'labels', None)!r}"


def test_when_only_the_liquidation_check_used_the_bar_the_label_says_liquidation_check(liquidation_only):
    """19:00: "label when only liquidation check used intrabar order: same label + '(liquidation check)'". The only
    level in reach is the liquidation price (about 40,201); the bar trades to 30,000. Today it is liquidated (at
    40,201) with no label."""
    r, _ = _levered()
    assert [f[0] for f in _fills(r)][:2] == ["entry", "liquidation"], _fills(r)  # setup: liquidated on that bar
    assert LABEL_LIQ in _result_labels(r), f"no {LABEL_LIQ!r} label: labels {getattr(r, 'labels', None)!r}"


def test_no_label_when_the_liquidation_level_stayed_outside_every_bars_range(liquidation_only):
    """19:00: NO label when no resting level lay inside any bar's range: the bar trades to 45,000, short of the
    liquidation price (about 40,201). Holds today."""
    r, _ = _levered(low=45_000.0)
    assert [f[0] for f in _fills(r)] == ["entry"], _fills(r)  # setup: held throughout
    assert "pessimistic" not in _result_labels(r)


def test_no_label_when_a_resting_stop_stayed_outside_every_bars_range():
    """19:00: a stop rested (58,800 on 1x spot) but every bar stayed between 59,900 and 60,100: no label. Holds
    today."""
    r, _ = _run((60_000, 60_100, 59_900, 60_000), side=1, perp=False, stop_loss=0.02)
    assert [f[0] for f in _fills(r)] == ["entry"] and any(
        d.get("intent") == "stop_loss" for d in r.decisions.values()), _fills(r)  # setup: the stop rested, untraded
    assert "pessimistic" not in _result_labels(r)


def _engine_tp_level(side, h=H, frac=0.03):
    """The target level exactly as the build computes it (94dc824 base.py _target_level):
    Price(entry_px x (1 + side x tp), price_precision).as_double(), with entry_px the BOOKED entry, itself a Price.
    Building the touching bar from this expression keeps the float hair (60,060 x 0.98 < 58,858.8) out of it."""
    from nautilus_trader.model import Price

    prec = binance_inst().price_precision
    booked = Price(P0 * (1 + side * h), prec).as_double()
    return Price(booked * (1 + side * frac), prec).as_double()


def _touch_bar(side, level, past=0.0):
    """A bar after the entry whose high (long) or low (short) is exactly `level`, or `past` beyond it; the open is
    inside it and it closes back inside it."""
    if side > 0:
        return (60_000.0, level + past, 59_950.0, 61_000.0)
    return (60_000.0, 60_050.0, level - past, 59_000.0)


def _exact_touch(side):
    """The exact-touch run, and its positive control: the same run with the bar 100 past the level fills the target
    (the target is live: a resting order on c7a73f1, a market-on-touch watch in the build that journals nothing until
    it triggers)."""
    level = _engine_tp_level(side)
    bar = _touch_bar(side, level)
    assert (bar[1] if side > 0 else bar[2]) == level, (bar, level)  # setup: the extreme IS the level, exactly
    assert abs(level - _tp_level(side)) < 1e-6, (level, _tp_level(side))  # setup: 61,861.8 / 58,141.8
    ctl, _ = _run(_touch_bar(side, level, past=100.0), side=side, take_profit=0.03)
    assert [f[0] for f in _fills(ctl)] == ["entry", "take_profit"], _fills(ctl)  # setup: a bar through it fills
    r, _ = _run(bar, side=side, take_profit=0.03)
    return r, level


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_an_exact_target_touch_fills_at_the_level_minus_taker_slippage(side):
    """Phase 1 exact touch (Advisor 19:40; supersedes 19:16 (1), which becomes the before-G2 resting-limit rule): the
    bar's high (low) is exactly the target, 3% from the BOOKED entry (20:55): 61,861.8 long / 58,141.8 short at a
    0.1% half spread, with no trade through it. The market-on-touch target FILLS, at the level x (1 -/+ max(0.1%,
    0.05%)) = 61,799.94 / 58,199.94, as [tp-l12]. Open until #155. v10: re-derived from the booked entry (it was the
    mid-derived 61,800); the "rested" setup check is replaced by a positive control. Today the target rests at
    61,800 / 58,200 (from mid), which this bar trades through: it fills at 61,738.21 / 58,258.21."""
    r, level = _exact_touch(side)
    assert [f[0] for f in _fills(r)] == ["entry", "take_profit"], _fills(r)
    px = _exit(r)[3]
    assert _near(px, _tp_fill(level, side)), (px, _tp_fill(level, side))


@pytest.mark.parametrize("side", [1, pytest.param(-1, marks=xf)], ids=["long", "short"])  # short: open until #155
def test_a_target_touched_exactly_inside_a_bar_is_labelled(side):
    """19:00 [lbl-range]: the label applies when a resting level lay inside a bar's range. The target (61,861.8 /
    58,141.8, from the booked entry) is exactly the bar's high (low): inside the range, so labelled (its fill is
    pinned by the test above). Open until #155. v10: re-derived as above. Today: no label."""
    r, _ = _exact_touch(side)
    assert LABEL in _result_labels(r), f"no {LABEL!r} label: labels {getattr(r, 'labels', None)!r}"


def test_a_lab_result_on_5_minute_execution_bars_is_labelled_with_its_bar_length():
    """18:36: "execution bars 5 min: pessimistic fills"; the stop traded through inside a 5-minute bar."""
    _, daily, m5 = _dip_tape()
    r = backtest(binance_inst(), daily, strategy="qa_t", minutes=1440, profile=None, half_spread=H,
                 params={**PERP, "at": daily.index[1].value, "side": 1, "stop_loss": 0.02}, exec_prices=m5,
                 exec_minutes=5)
    assert [f[0] for f in _fills(r)] == ["entry", "stop_loss"], _fills(r)  # setup
    assert LABEL_5M in _result_labels(r), f"no {LABEL_5M!r} label: labels {getattr(r, 'labels', None)!r}"


def test_on_5_minute_execution_bars_a_stop_traded_through_fills_at_the_bars_extreme():
    """(c) "5m pessimistic": the 5-minute bar holding the 57,000 minute is traded through: fill about 57,000.
    Today: the 58,800 trigger (minus the half spread)."""
    _, daily, m5 = _dip_tape()
    r = backtest(binance_inst(), daily, strategy="qa_t", minutes=1440, profile=None, half_spread=H,
                 params={**PERP, "at": daily.index[1].value, "side": 1, "stop_loss": 0.02}, exec_prices=m5,
                 exec_minutes=5)
    intent, _, _, px, _ = _exit(r)
    assert intent == "stop_loss" and _near(px, _stop_px(57_000.0, 1)), (px, _stop_px(57_000.0, 1))


def test_a_bars_only_spot_run_with_signal_exits_only_is_not_labelled():
    """No fill and no liquidation check relied on intrabar order (1x spot, market orders at closes): no label.
    Holds today."""
    _, daily = _tape()
    res = backtest(kraken_inst(), daily, strategy="qa_d13s", minutes=1440, profile=None, half_spread=H,
                   params={"at": daily.index[1].value, "until": daily.index[5].value})
    assert [f[0] for f in _fills(res)] == ["entry", "exit"], _fills(res)  # setup: in and out at closes
    assert LABEL not in _result_labels(res)


@pytest.mark.parametrize("exits", [{"stop_loss": 0.02}, {}], ids=["with-a-stop", "signal-only"])
def test_a_lab_result_on_1_minute_execution_bars_is_not_labelled(exits):
    """Holds today (there is no label at all)."""
    m1, daily = _tape()
    res = backtest(binance_inst(), daily, strategy="qa_t", minutes=1440, profile=None, half_spread=H,
                   params={**PERP, "at": daily.index[1].value, "side": 1, **exits}, exec_prices=m1, exec_minutes=1)
    assert LABEL not in _result_labels(res) and "pessimistic" not in _result_labels(res)


def test_a_bars_only_backtest_page_result_with_a_stop_filled_inside_a_bar_is_labelled():
    """Preview: the venue's daily candles, no stored minutes."""
    daily = _preview_daily()
    d = _preview(daily, {"at": daily.index[30].value, "stop_loss": 0.02}, detail=True)
    assert _labelled(d), f"no {LABEL!r} anywhere in the result; keys {sorted(d)}"


def test_a_bars_only_backtest_page_result_with_signal_exits_only_is_not_labelled():
    """Preview, 1x spot, in and out at closes. Holds today."""
    daily = _preview_daily()
    d = _preview(daily, {"at": daily.index[30].value, "until": daily.index[35].value}, detail=True)
    assert not _labelled(d)


def test_a_backtest_page_result_on_1_minute_execution_bars_is_not_labelled(monkeypatch, tmp_path):
    """Preview with stored minutes up to now and a risk profile: the page matches on 1-minute bars (61 days is under
    its 150,000-bar budget). Holds today."""
    from sleeve_fund import history
    from sleeve_fund.dashboard import preview

    now = pd.Timestamp.now(tz="UTC").floor("1D")
    m1 = synth_1m(days=61, seed=17, start=str(now - pd.Timedelta(days=61)), vol_day=0.03)
    opens = m1.set_axis(m1.index - pd.Timedelta(minutes=1))  # the store keeps open times
    monkeypatch.setattr(history, "DEFAULT_ROOT", tmp_path / "hist")
    history.HistoryStore(tmp_path / "hist").append("KRAKEN", "BTC/USD", opens, cursor="x")
    preview._history.clear()
    at = (now - pd.Timedelta(days=40)).value
    d = quiet(lambda: preview.run("qa_d13s", "BTC/USD", {"at": at, "stop_loss": 0.02}, venue="KRAKEN",
                                  risk_profile="balanced", detail=True))
    assert d["execution"]["matched_on"] == "1-minute", d["execution"]  # setup: matched on 1-minute bars
    assert not _labelled(d) and not _labelled(d, "pessimistic fills")


# ---------------------------------------------------------------------------------------------------------------
# G1 and the study label. Small studies (40 days, two folds), each run once and kept.
# ---------------------------------------------------------------------------------------------------------------

_STUDIES: dict = {}


def _synth_40():
    """40 days of 1-minute bars with one minute on day 25 dropping 10% below its close: any stop then trades."""
    m1 = synth_1m(days=40, seed=4, start="2025-03-01", vol_day=0.04)
    at = m1.index[25 * 1440 + 600]
    m1.loc[at, "low"] = round(float(m1.loc[at, "close"]) * 0.9, 1)
    return m1


def _study(key: str):
    """key: '<exec>/<exits>', exec in bars | 1m | 5m | handmade, exits in stop | tp | channel | none."""
    if key in _STUDIES:
        return _STUDIES[key]
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.study import run_study
    from sleeve_fund.strategies.buy_and_hold import SPEC

    exec_, ex = key.split("/")
    if exec_ == "handmade":
        rows = [FLAT] * 40
        rows[25] = (60_000, 60_050, 55_000, 59_000)
        rows[26:] = [(59_000,) * 4] * 14
        daily, exec_prices = _days(rows, start="2025-03-01"), None
    else:
        m1 = _synth_40()
        daily = resample(m1, 1440)
        exec_prices = {"bars": None, "1m": m1, "5m": resample(m1, 5)}[exec_]
    exits = {"stop": {"stop_loss": 0.03}, "tp": {"take_profit": 0.05}, "channel": {"stop_swing_bars": 5},
             "none": None}[ex]
    path = Path(tempfile.mkdtemp(prefix="d13-ledger-")) / "l.jsonl"
    ledger = IdeaLedger(path)
    r = quiet(lambda: run_study(SPEC, daily, kraken_inst(), dataset=f"d13-{exec_}-{ex}", ledger=ledger,
                                synthetic=True, holdout_days=0, train_days=20, test_days=10, exits=exits,
                                exec_prices=exec_prices, half_spread=H))
    _STUDIES[key] = (r, ledger)
    return r, ledger


def _g1(r, ledger):
    from sleeve_fund.research.tearsheet import g1_checks, g1_verdict, render

    return g1_verdict(g1_checks(r, ledger))[0], render(r, ledger)


def _rested(r, intents=("stop_loss", "take_profit")):
    return any(d.get("intent") in intents for d in r.full_period.decisions.values())


@pytest.mark.parametrize("ex", ["stop", "tp", "channel"])
def test_g1_does_not_judge_a_bars_only_study_with_resting_exits(ex):
    """(1), and (b): no 1-minute bars for the out-of-sample windows: not judged, with the reason."""
    from sleeve_fund.research.tearsheet import NOT_JUDGED

    r, ledger = _study(f"bars/{ex}")
    assert _rested(r), "setup: the exit rested"
    verdict, sheet = _g1(r, ledger)
    assert NO_1M in r.not_judged, f"not_judged is {r.not_judged!r}; G1 says {verdict}"
    assert verdict == NOT_JUDGED and "**G1: NOT JUDGED**" in sheet
    assert f"not judged: {NO_1M}" in sheet.lower()


def test_g1_does_not_judge_a_bars_only_study_of_a_model_that_rests_its_own_stop(tmp_path):
    """(1): the resting exit can be the model's own: dip_buy places a stop on every entry (stop_atr 2)."""
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.study import run_study
    from sleeve_fund.research.tearsheet import NOT_JUDGED
    from sleeve_fund.strategies.dip_buy import SPEC

    daily = synthetic_ohlcv(days=300, seed=3, start_price=60_000, drift=0.003, vol=0.03)  # an up-trend it dips in
    ledger = IdeaLedger(tmp_path / "l.jsonl")
    r = quiet(lambda: run_study(SPEC, daily, kraken_inst(), dataset="d13-dip", ledger=ledger, synthetic=True,
                                holdout_days=0, train_days=120, test_days=60, half_spread=H))
    assert _rested(r, ("stop_loss",)), "setup: dip_buy rested a stop"
    verdict, _ = _g1(r, ledger)
    assert NO_1M in r.not_judged and verdict == NOT_JUDGED, (r.not_judged, verdict)


def test_on_5_minute_execution_bars_a_g1_fail_is_not_final_until_re_run_at_1_minute():
    """(c) one-way: on 5-minute bars a pass counts, but a fail must be re-run at 1 minute before the idea is dropped.
    This study fails G1 today (FAIL, judged); it must read NOT JUDGED, saying to re-run at 1 minute."""
    from sleeve_fund.research.tearsheet import NOT_JUDGED

    r, ledger = _study("5m/stop")
    assert _rested(r), "setup: the stop rested"
    verdict, _ = _g1(r, ledger)
    assert verdict != "FAIL", (verdict, r.not_judged)
    assert verdict == NOT_JUDGED and "1-minute" in r.not_judged, (verdict, r.not_judged)


def test_g1_does_not_judge_a_bars_only_perp_study_whose_only_resting_levels_are_the_risk_guards(monkeypatch):
    """19:00: "liquidation level + risk-guard stops count as resting exits for G1 1m". qa_t long on the Binance perp
    under the balanced profile, no stop of its own: the risk guard rests its stop (drawdown halt, daily pause or
    liquidation cut) on every bar. Bars only: not judged. Today: judged."""
    from sleeve_fund.research import study
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.study import run_study

    # Set-up (HoQA, after #163): the synthetic perp has no stored funding rates, and since #163 funding's not-judged
    # reason comes first, so funding is judged here (as in the GAP-LIQ-CAP master) and the 1-minute reason is read.
    if hasattr(study, "_funding_check"):
        monkeypatch.setattr(study, "_funding_check", lambda r: ("PASS", "funding lifted by the D13 harness"))
    from sleeve_fund.research.tearsheet import NOT_JUDGED
    from sleeve_fund.strategies.base import IdeaSpec

    spec = IdeaSpec(name="qa_t", family="qa-d13", idea="QA: long throughout", rules="QA",
                    default_params={"side": 1, "at": 0, "allow_short": True})
    daily = resample(_synth_40(), 1440)
    ledger = IdeaLedger(Path(tempfile.mkdtemp(prefix="d13-ledger-")) / "l.jsonl")
    r = quiet(lambda: run_study(spec, daily, binance_inst(), dataset="d13-perp-guard", ledger=ledger, synthetic=True,
                                holdout_days=0, train_days=20, test_days=10, risk_profile="balanced", half_spread=H))
    assert _rested(r, ("risk_halt", "risk_pause", "liquidation_cut")), "setup: a risk-guard stop rested"
    assert not _rested(r), "setup: no stop or target of its own"
    verdict, _ = _g1(r, ledger)
    assert NO_1M in r.not_judged and verdict == NOT_JUDGED, (r.not_judged, verdict)


def test_g1_judges_a_study_with_resting_exits_on_1_minute_execution_bars():
    """With 1-minute execution bars the cap does not apply. Holds today; the fix must keep it."""
    from sleeve_fund.research.tearsheet import NOT_JUDGED

    r, ledger = _study("1m/stop")
    assert _rested(r), "setup: the stop rested"
    verdict, sheet = _g1(r, ledger)
    assert r.not_judged == "" and verdict != NOT_JUDGED, (r.not_judged, verdict)
    assert NO_1M not in sheet


def test_g1_judges_a_bars_only_study_with_signal_exits_only():
    """(1) is about resting exits: a bars-only 1x spot study whose exits are all on the signal is still judged.
    Holds today; the fix must keep it."""
    from sleeve_fund.research.tearsheet import NOT_JUDGED

    r, ledger = _study("bars/none")
    assert not _rested(r), "setup: nothing rested"
    verdict, sheet = _g1(r, ledger)
    assert r.not_judged == "" and verdict != NOT_JUDGED, (r.not_judged, verdict)
    assert NO_1M not in sheet


def test_a_bars_only_study_whose_stop_filled_inside_a_bar_is_labelled():
    """On the tear sheet ((b): in-sample may be bars-only, "labelled")."""
    r, ledger = _study("handmade/stop")
    _, sheet = _g1(r, ledger)
    assert LABEL in sheet


def test_a_study_on_5_minute_execution_bars_is_labelled_with_its_bar_length():
    r, ledger = _study("5m/stop")
    assert [f for f in _fills(r.full_period) if f[0] == "stop_loss"], "setup: a stop traded"
    _, sheet = _g1(r, ledger)
    assert LABEL_5M in sheet


@pytest.mark.parametrize("key", ["1m/stop", "bars/none"])
def test_a_study_with_no_intrabar_reliance_is_not_labelled(key):
    """1-minute execution bars, or bars only with 1x spot signal exits: no label. Holds today."""
    r, ledger = _study(key)
    _, sheet = _g1(r, ledger)
    assert LABEL not in sheet and "pessimistic fills" not in sheet


# ---------------------------------------------------------------------------------------------------------------
# The Research page over the 1-minute budget: (a) touched bars, (b) OOS + holdout on 1 minute, the falsifier gap.
# Two years of stored minutes (1.05 million: seven times the 150,000-bar budget), buy and hold with an 8% stop, no
# risk profile, train 180 / test 90 / holdout 60 days, holdout opened. Run once (about a minute) and kept.
# ---------------------------------------------------------------------------------------------------------------

_BIG: dict = {}


def _store_study():
    if _BIG:
        return _BIG
    from sleeve_fund import history
    from sleeve_fund.research import run as research_run
    from sleeve_fund.research import study

    now = pd.Timestamp.now(tz="UTC").floor("1D")
    m1 = synth_1m(days=730, seed=21, start=str(now - pd.Timedelta(days=730)), vol_day=0.03)
    root = Path(tempfile.mkdtemp(prefix="d13-store-"))
    hs = history.HistoryStore(root / "hist")
    hs.append("KRAKEN", "BTC/USD", m1.set_axis(m1.index - pd.Timedelta(minutes=1)), cursor="x")
    reads, real_read = [], hs.read

    def spy(venue, pair, minutes=1440, start=None, end=None):
        df = real_read(venue, pair, minutes, start=start, end=end)
        reads.append((minutes, len(df)))
        return df

    hs.read = spy
    seen, real_study = {}, study.run_study

    def keep(*a, **k):
        seen["result"], seen["args"] = real_study(*a, **k), (a, k)
        return seen["result"]

    study.run_study = keep
    try:
        req = research_run.StudyRequest(strategy="buy_and_hold", pair="BTC/USD", venue="KRAKEN", risk_profile=None,
                                        train_days=180, test_days=90, holdout_days=60, use_holdout=True,
                                        stop_loss=0.08)
        quiet(lambda: research_run.run_store_study(req, history=hs, ledger_path=root / "ledger.jsonl",
                                                   out_dir=root / "sheets"))
    finally:
        study.run_study = real_study
    _BIG.update(m1=m1, reads=reads, result=seen["result"], args=seen["args"], ledger=root / "ledger.jsonl",
                budget=research_run.EXEC_BAR_BUDGET)
    return _BIG


def _window_days(r, f):
    """The study's OOS returns on fold f's own test-window days (its first bar to its last), as G1 sees them."""
    first = f.train_end + pd.Timedelta(days=1)
    idx = r.oos_returns.index
    return r.oos_returns[(idx >= first) & (idx <= f.test_end)], first


def _flat_start(ref, first, bar):
    """A flat-start reference's daily returns on a window's own days. The window pays its own entry cost (QD /
    coordinator, 6 Oct): the reference starts flat ONE BAR BEFORE the window, and its first-day return is measured
    from that flat state (its starting capital). The engine books the entry a state rule sends at that bar's close at
    the same timestamp, so its mark there already carries the fee and spread; it is replaced by the flat starting
    capital, which puts that cost in the window's first day, not before the window."""
    from sleeve_fund.research.metrics import daily_returns, whole_days

    assert ref.equity.index[0] == first - bar, (ref.equity.index[0], first)  # setup: one bar before the window
    eq = ref.equity.copy()
    eq.iloc[0] = ref.starting_capital
    return whole_days(daily_returns(eq), first, bar)


def _same_days(got, want, what):
    """Compared on the window's own days: the reference must cover exactly those days, then match each one."""
    assert list(want.index) == list(got.index), (what, len(got), len(want), got.index[:2], want.index[:2])
    assert np.allclose(got.to_numpy(), want.to_numpy(), atol=1e-12), (what, got.head(), want.head())


def _reference(df, params, inst, h, m1):
    """The same run on every 1-minute bar."""
    from sleeve_fund.research.runner import run_backtest

    fine = m1[(m1.index > df.index[0] - pd.Timedelta(days=1)) & (m1.index <= df.index[-1])]
    return quiet(lambda: run_backtest("buy_and_hold", df, inst, params=params, exec_prices=fine, exec_minutes=1,
                                      half_spread=h, bar_minutes=1440))


@xf
def test_a_multi_year_study_with_a_stop_is_judged_on_touched_1_minute_bars_and_equals_the_full_1_minute_run():
    """(a), the target design: 1-minute bars fetched only for the decision bars whose range touches a resting level.
    The study is judged; its full-period run equals the run on every 1-minute bar (exact: fills and end equity); and
    the 1-minute bars it read stay within the budget. Today it reads no 1-minute bars (15-minute ones instead)."""
    from sleeve_fund.research.tearsheet import NOT_JUDGED
    from sleeve_fund.research.ledger import IdeaLedger

    big = _store_study()
    r, (a, k) = big["result"], big["args"]
    prices, inst = a[1], a[2]
    research = prices.iloc[:-60]
    assert [f[0] for f in _fills(r.full_period)].count("stop_loss") >= 1, "setup: the stop traded"
    assert r.not_judged == "", r.not_judged
    assert _g1(r, IdeaLedger(big["ledger"]))[0] != NOT_JUDGED
    ref = _reference(research, dict(k["exits"]), inst, k["half_spread"], big["m1"])
    assert [(i, t, round(p, 6)) for i, t, _, p, _ in _fills(r.full_period)] == \
        [(i, t, round(p, 6)) for i, t, _, p, _ in _fills(ref)]
    assert float(r.full_period.equity.iloc[-1]) == pytest.approx(float(ref.equity.iloc[-1]), abs=0.01)
    one_minute = sum(n for m, n in big["reads"] if m == 1)
    assert 0 < one_minute <= big["budget"], (one_minute, big["reads"])


def test_a_studys_out_of_sample_windows_and_holdout_equal_their_1_minute_runs():
    """(b), the interim rule (and true under (a) too): G1 stands on OOS windows and a holdout run on 1-minute bars,
    matching 1 minute exactly (19:00), each STARTED FLAT (19:16 (2)). buy_and_hold is a state rule, so each window
    enters on its first day. Each fold's test-window returns and the holdout equal 1-minute runs started flat ONE BAR
    BEFORE the window (the window pays its own entry cost: QD/coordinator, 6 Oct), compared on the window's own days
    (the holdout: 60). Today the windows continue the full path: where it was stopped out (8% stop) before a window,
    the window is flat, while the flat start enters on its first day."""
    from sleeve_fund.research.metrics import summary

    big = _store_study()
    r, (a, k) = big["result"], big["args"]
    prices, inst, h, params = a[1], a[2], k["half_spread"], dict(k["exits"])
    bar = pd.Timedelta(days=1)
    research = prices.iloc[:-60]
    assert len(r.folds) >= 3 and r.holdout is not None, (len(r.folds), r.holdout_withheld)  # setup
    start = 0
    for f in r.folds:
        got, first = _window_days(r, f)
        assert first == research.index[start + 180] and len(got) == 90, (first, len(got))  # setup: the fold's window
        ref = _reference(research.iloc[start + 179: start + 270], params, inst, h, big["m1"])
        assert [x[0] for x in _fills(ref)][:1] == ["entry"], _fills(ref)  # setup: from flat it enters
        _same_days(got, _flat_start(ref, first, bar), f.test_end)
        start += 90
    ref = _reference(prices.iloc[-61:], params, inst, h, big["m1"])
    days = _flat_start(ref, prices.index[-60], bar)
    assert len(days) == 60, len(days)  # setup: the holdout's own 60 days, the first carrying the entry cost
    want = summary(days)
    assert {k_: v for k_, v in r.holdout.items() if isinstance(v, float) and math.isfinite(v)} == \
        pytest.approx({k_: v for k_, v in want.items() if isinstance(v, float) and math.isfinite(v)}, abs=1e-12)


_FLAT: dict = {}


def _carry_study(kind="state"):
    """The Research page over the budget, built so the bars-only path and the 1-minute path carry DIFFERENT positions
    into the first out-of-sample window. 150 days of stored 1-minute bars on the Binance perp, flat at 60,000 and
    rising 0.05% a day (so nothing else ever trades). qa_d13e rests a stop entry at the day-40 close, 1% above it, with
    a 2% stop. On day 41, inside the one 5-minute bar 10:00-10:05: at 10:01 the price dips 3% (below where the stop
    will be), at 10:03 it rises 2% (through the entry), and it stays 1.5% up after. On 1-minute bars the dip comes
    BEFORE the entry: long, held into the window (day 60). Bars-only pessimistic (18:36): entry then stopped out: flat.
    Train 60, test 30, holdout 30 days; today it runs on 5-minute bars.

    kind "state" is that model (a STATE rule: from the day-41 close it wants long whatever the position). kind "cross"
    runs qa_d13x on the same tape instead (a CROSSOVER rule: long on the day-41 close, which crosses up through 1%
    above the day-40 close, and never another cross after it, so it holds; no stop)."""
    if kind in _FLAT:
        return _FLAT[kind]
    from sleeve_fund import history
    from sleeve_fund.research import run as research_run
    from sleeve_fund.research import study
    from sleeve_fund.strategies.base import IdeaSpec

    now = pd.Timestamp.now(tz="UTC").floor("1D")
    start = now - pd.Timedelta(days=150)
    idx = pd.date_range(start + pd.Timedelta(minutes=1), periods=150 * 1440, freq="1min")
    day = (idx - start) / pd.Timedelta(days=1)
    c = np.round(P0 * (1 + 0.0005 * day) + (np.arange(len(idx)) % 2) * 0.1, 1)
    m1 = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e9}, index=idx)
    d40 = start + pd.Timedelta(days=40)
    p40 = float(m1.loc[d40, "close"])
    after = m1.index > d40 + pd.Timedelta(hours=10, minutes=3)
    m1.loc[after, ["open", "high", "low", "close"]] = np.round(m1.loc[after, ["open", "high", "low", "close"]] * 1.015, 1)
    t = d40 + pd.Timedelta(hours=10)
    m1.loc[t + pd.Timedelta(minutes=1), ["low", "close"]] = [round(p40 * 0.97, 1), round(p40 * 0.99, 1)]
    m1.loc[t + pd.Timedelta(minutes=2), ["open", "high", "low", "close"]] = [round(p40 * 0.99, 1)] * 4
    m1.loc[t + pd.Timedelta(minutes=3), ["open", "low", "high", "close"]] = [round(p40 * 0.99, 1), round(p40 * 0.99, 1),
                                                                             round(p40 * 1.02, 1), round(p40 * 1.015, 1)]
    root = Path(tempfile.mkdtemp(prefix="d13-carry-"))
    hs = history.HistoryStore(root / "hist")
    hs.append("BINANCE", "BTC/USDT", m1.set_axis(m1.index - pd.Timedelta(minutes=1)), cursor="x")
    if kind == "state":
        name, stop = "qa_d13e", 0.02
        params = {"side": 1, "at": (d40 + pd.Timedelta(days=1)).value, "place_at": d40.value,
                  "trigger": round(p40 * 1.01, 1), "allow_short": True}
    else:
        name, stop = "qa_d13x", None
        params = {"side": 1, "level": round(p40 * 1.01, 1), "allow_short": True}
    spec = IdeaSpec(name=name, family="qa-d13", idea="QA: a resting entry", rules="QA", default_params=params,
                    param_grid={key: [v] for key, v in params.items()})  # one grid point: the folds trade it too
    seen, real_study, real_spec = {}, study.run_study, research_run.spec_of

    def keep(*a, **k):
        seen["result"], seen["args"] = real_study(*a, **k), (a, k)
        return seen["result"]

    study.run_study, research_run.spec_of = keep, (lambda name: spec)
    try:
        req = research_run.StudyRequest(strategy=name, pair="BTC/USDT", venue="BINANCE", risk_profile=None,
                                        train_days=60, test_days=30, holdout_days=30, use_holdout=True,
                                        stop_loss=stop)
        quiet(lambda: research_run.run_store_study(req, history=hs, ledger_path=root / "ledger.jsonl",
                                                   out_dir=root / "sheets"))
    finally:
        study.run_study, research_run.spec_of = real_study, real_spec
    _FLAT[kind] = dict(m1=m1, result=seen["result"], args=seen["args"], params=params, name=name)
    return _FLAT[kind]


def _one_minute(big, df):
    """The study's model run on every 1-minute bar over `df` (decision bars), started flat at df's first bar."""
    from sleeve_fund.research.runner import run_backtest

    (a, k), m1 = big["args"], big["m1"]
    params = {**big["params"], **k["exits"], "market": "perp"}
    bar = pd.Timedelta(days=1)
    fine = m1[(m1.index > df.index[0] - bar) & (m1.index <= df.index[-1])]
    return quiet(lambda: run_backtest(big["name"], df, a[2], params=params, exec_prices=fine, exec_minutes=1,
                                      half_spread=k["half_spread"], bar_minutes=1440))


def test_an_out_of_sample_window_with_a_state_rule_starts_flat_and_enters_at_its_first_close():
    """19:00 tightened by 19:16 (2): each OOS window (and the holdout) STARTS FLAT. A model with a STATE rule (a
    non-zero target weight from flat; qa_d13e wants long from the day-41 close) enters at the window's first close, so
    the first window's returns equal a 1-minute run started flat one bar before the window (so the window pays its own
    entry cost on its first day), compared on the window's own 30 days. Neither the bars-only path (flat, stopped out
    on day 41) nor the continuous 1-minute path (long since day 41) may be carried in. Today the window continues the
    continuous path: it equals the 1-minute run through training and test, not the flat start."""
    big = _carry_study("state")
    r, (a, _) = big["result"], big["args"]
    through, bar = a[1].iloc[:90], pd.Timedelta(days=1)
    got, first = _window_days(r, r.folds[0])
    assert first == through.index[60] and len(got) == 30, (first, len(got))  # setup: the first fold's window
    carried = _one_minute(big, through)
    assert [f[0] for f in _fills(carried)] == ["entry"], _fills(carried)  # setup: 1 minute is long into the window
    assert float(carried.exposure.loc[through.index[59]]) > 0
    fresh = _one_minute(big, through.iloc[59:])
    assert [f[0] for f in _fills(fresh)][:1] == ["entry"], _fills(fresh)  # setup: from flat it enters
    assert _fills(fresh)[0][1] >= through.index[59], _fills(fresh)  # setup: entered from flat at the window's start
    assert (got.abs() > 0).any(), got  # setup: the window traded a position
    _same_days(got, _flat_start(fresh, first, bar), "first window")


def test_an_out_of_sample_window_with_a_crossover_rule_stays_flat_until_its_next_cross():
    """19:16 (2): a model with a CROSSOVER rule (qa_d13x: long on the close that crosses up through the level, then no
    change) starts each window flat and stays flat until its next cross, even though the path before the window held a
    position: never force an entry. Here the only cross is on day 41, so the first window and the holdout are flat:
    every return is zero, on the window's own 30 days and the holdout's (opened: use_holdout). Today the window
    carries the in-sample long in."""
    big = _carry_study("cross")
    r, (a, _) = big["result"], big["args"]
    through = a[1].iloc[:90]
    assert r.holdout is not None, r.holdout_withheld  # setup: the holdout was opened
    got, first = _window_days(r, r.folds[0])
    assert first == through.index[60] and len(got) == 30, (first, len(got))  # setup: the first fold's window
    carried = _one_minute(big, through)
    assert [f[0] for f in _fills(carried)] == ["entry"], _fills(carried)  # setup: the prior path is long
    assert float(carried.exposure.loc[through.index[59]]) > 0
    assert _fills(_one_minute(big, through.iloc[59:])) == []  # setup: from flat the model never enters here
    assert (got.abs() < 1e-12).all(), got[got.abs() >= 1e-12].head()
    assert abs(float(r.holdout.get("total_return", 0.0))) < 1e-12, r.holdout


def test_the_study_records_the_bars_only_vs_1_minute_gap_for_the_falsifier():
    """18:36 falsifier: the study code writes the bars-only vs 1-minute gap (share of capital) to the trials register
    (here: the study's ledger, which the register imports). Bars-only is never better: the gap is >= 0 bar fees."""
    big = _store_study()
    r = big["result"]
    gap = getattr(r, "bars_only_gap", None)
    assert gap is not None and math.isfinite(gap) and gap >= -0.001, gap
    rows = [json.loads(line) for line in Path(big["ledger"]).read_text().splitlines() if line.strip()]
    assert any("bars_only_gap" in json.dumps(row) for row in rows), "no bars_only_gap row in the study's ledger"


# ---------------------------------------------------------------------------------------------------------------
# Fee ladder: the break-even moves only through the level effect (Advisor 20:55)
# ---------------------------------------------------------------------------------------------------------------

_LADDER: dict = {}
LADDER_TOL = 1e-5  # 0.001% per side; why: see the test


def _ladder_study(h):
    """run_study (its cost ladder over COST_LADDER and its verified break-even) on qa_d13p, Kraken spot, signal exits
    only: 160 synthetic days that rise 0.2% a day while the model is long and fall 0.2% a day while it is flat
    (5 days each), so every round trip makes about 1% gross. Train 20 / test 10, no holdout."""
    if h in _LADDER:
        return _LADDER[h]
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.study import run_study
    from sleeve_fund.strategies.base import IdeaSpec

    start = pd.Timestamp("2025-01-01", tz="UTC")
    idx = pd.date_range(start + pd.Timedelta(days=1), periods=160, freq="1D")
    closes, c = [], P0
    for t in idx:
        held = ((t - pd.Timedelta(days=1)).value // 86_400_000_000_000 // 5) % 2 == 0  # long through this bar
        c = round(c * (1.002 if held else 0.998), 1)
        closes.append(c)
    daily = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1e9}, index=idx)
    daily["open"] = daily["close"].shift(1).fillna(P0)
    daily["high"] = daily[["open", "close"]].max(axis=1)
    daily["low"] = daily[["open", "close"]].min(axis=1)
    spec = IdeaSpec(name="qa_d13p", family="qa-d13", idea="QA: periodic", rules="QA", default_params={"period": 5},
                    param_grid={"period": [5]})
    ledger = IdeaLedger(Path(tempfile.mkdtemp(prefix="d13-ladder-")) / "l.jsonl")
    res = quiet(lambda: run_study(spec, daily, kraken_inst(), dataset=f"d13-ladder-{h}", ledger=ledger,
                                  synthetic=True, holdout_days=0, train_days=20, test_days=10, half_spread=h))
    _LADDER[h] = res
    return res


def test_the_fee_ladder_break_even_moves_only_through_the_level_effect():
    """20:55 "fee-ladder break-even moves only through the level effect". Two studies at the SAME total cost:
    - in the price: a 0.1% half spread, charged in the fill price (the 20:55 rule);
    - as a cost line: no spread, the same 0.1% carried as part of the fee, so its break-even fee less 0.1% is the
      break-even with the spread on a separate cost line.
    The model has signal exits only, so nothing derives a level from the fill: the level-driven change in fills is
    ZERO, and the two break-evens must be equal. Each is the study's VERIFIED figure (re-run to a net return within
    BREAKEVEN_TOLERANCE).
    Tolerance 1e-5 (0.001% per side), why: (i) charged in the price, the fee is on the spread-shifted fill, so each
    side pays fee x 0.1% more (about 4e-6 at a 0.4% fee); (ii) each verified figure is within 0.01% of capital of
    zero net return, which over the ~15 round trips here (30 sides of ~1x notional) is about 3e-6 of fee. Holds today
    (the spread is charged on mid notional, exactly as a fee would be)."""
    in_price, as_line = _ladder_study(H), _ladder_study(0.0)
    got = []
    for res in (in_price, as_line):
        row = res.sensitivity.iloc[0]
        assert "verified" in str(row["breakeven"]), row["breakeven"]  # setup: a verified figure, inside the ladder
        assert int(row["round_trips"]) >= 10, row["round_trips"]  # setup: enough trades to locate it
        got.append(float(row["breakeven_fee"]))
    assert got[0] == pytest.approx(got[1] - H, abs=LADDER_TOL), (got[0], got[1] - H, got)


# ---------------------------------------------------------------------------------------------------------------
# One-sided parity, bars-only end equity <= 1-minute end equity, fee tolerance only (18:16 (4))
# ---------------------------------------------------------------------------------------------------------------

def _parity_runs(side: int, gap_at: str):
    """The #155 test_d13 fixture: daily bars only against the same tape on 1-minute execution bars. A 15-second tape
    from 1 to 5 Oct 2025 cycling 60,000-60,019.5; from `gap_at` every print moves 5% against the position (down for
    a long, up for a short). In at the 2 Oct close, a 2% stop; balanced profile; no spread (both runs then pay the
    0.05% stop slippage floor); settlements 8-hourly, then 4-hourly from 3 Oct 20:00."""
    times = (pd.date_range("2025-10-01 00:00", "2025-10-03 16:00", freq="8h", tz="UTC")
             .append(pd.date_range("2025-10-03 20:00", "2025-10-06 00:00", freq="4h", tz="UTC")))
    put_rates(times)
    sec = pd.date_range("2025-10-01 00:00:01", "2025-10-05 23:59:46", freq="15s", tz="UTC")
    px = np.round(60_000.0 + (np.arange(len(sec)) % 40) * 0.5, 1)
    gap = pd.Timestamp(gap_at, tz="UTC")
    px = np.where(sec >= gap, np.round(px * (1 - 0.05 * side), 1), px)
    tape = pd.Series(px, index=sec)
    day = tape.resample("1D", closed="left", label="right").ohlc()
    day["volume"] = 1e9
    one = tape.resample("1min", closed="left", label="right").ohlc()
    one["volume"] = 1e9
    params = {**PERP, "at": ns("2025-10-02 00:00"), "side": side, "stop_loss": 0.02, "tag": "d13"}
    bo = backtest(binance_inst(), day, strategy="qa_t", params=params, minutes=1440, profile="balanced", half_spread=0.0)
    rf = backtest(binance_inst(), day, strategy="qa_t", params=params, minutes=1440, profile="balanced",
                  half_spread=0.0, exec_prices=one, exec_minutes=1)
    return bo, rf


PARITY = {
    "repro-long-gap-inside-the-bar": (1, "2025-10-04 10:00:01"),
    "short-mirror": (-1, "2025-10-04 10:00:01"),
    "gap-at-the-open-long": (1, "2025-10-04 00:00:01"),
    "gap-at-the-open-short": (-1, "2025-10-04 00:00:01"),
}
OPEN_TODAY = set()  # repro-long-gap-inside-the-bar: bars-only $193 better on main 1582c3a, $5.23 worse on #178 ed7f484: passes for real


@pytest.mark.parametrize("case", [pytest.param(c, marks=xf) if c in OPEN_TODAY else c for c in PARITY])
def test_bars_only_end_equity_is_never_above_the_1_minute_run(case):
    """One-sided: bars-only <= 1-minute, with a tolerance of the two runs' fee difference only (a worse exit price
    pays a smaller fee)."""
    side, gap_at = PARITY[case]
    bo, rf = _parity_runs(side, gap_at)
    for r in (bo, rf):
        assert not r.handler_errors, r.handler_errors
        assert [f for f in _fills(r) if f[0] == "stop_loss"], _fills(r)  # setup: both runs stopped out
    tolerance = abs(bo.fees_paid - rf.fees_paid) + 0.01
    gap = float(bo.equity.iloc[-1]) - float(rf.equity.iloc[-1])
    assert gap <= tolerance, f"bars-only ends {gap:,.2f} better than the 1-minute run (tolerance {tolerance:,.2f})"
