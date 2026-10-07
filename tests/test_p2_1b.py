"""P2-1b: weight-sized perp models converted to entry-sized trades (Advisor's spec, 6 Oct 2026). Strict xfails written
by QA BEFORE the build, from the spec's Done-when lines. The engineer makes each pass (and removes its mark) before
handing over.

Sources (every trading-behaviour expectation cites one):
- [1b:N] v2/p2-1b-weight-to-entry-spec.md, line N. Scope (l.3): Donchian and the vol_target trend filter; "Their legacy
  weight versions stay as they are, on spot only". Conversion rule 1 (l.6-9) entries/exits from the signal; rule 2
  (l.10-12) size set once at entry by central sizing, smaller of stop and vol size, every cap, the half-liquidation
  rule, 1% default; rule 3 (l.13-16) the model's own resting exit level (Donchian: the opposite channel up to the
  previous closed bar, ruling 5.7) or the 2.5 x Wilder ATR fallback placed as a real order, the tighter applies;
  rule 4 (l.17-20) no resizing in the base variant, a 25% band in the second; rule 5 (l.21) turnover and fee drag
  against the legacy weight version on spot; Recording (l.23-26); Done when (l.28-33, DW1-DW5).
- [A17] the Advisor's P2-1b rulings, 6 Oct 17:09 (relayed by the coordinator), items 1-7: (1) ensemble exit when the
  weight returns to 0; a long's resting level is the LOWEST of the active thirds' opposite channels over the n most
  recent closed bars INCLUDING the bar just closed (17:07 reading (a)), the tighter of that and 2.5 x ATR applies;
  (2) the trend filter's slow-average exit stays a close-checked signal exit, never a touch, and its placed stop is
  the 2.5 x ATR fallback; (3) the 30% is relative, |converted - legacy| / |legacy| > 30%, and when |legacy| < 5% a
  1.5-point gap instead; (4) the sign-change rule is pinned on a synthetic weight series only; (5) the target is
  central sizing recomputed at each close x current weight / full weight, the band |target - held| / held > 25%;
  (6) after a stop-out, re-entry only once the weight has gone to 0 and turned on again; (7) labels are descriptive:
  assert on params (sizing, resize_band), not names.
- [DS] the Donchian SPEC rules (strategies/donchian.py): channels are on CLOSES ("the highest close ... the lowest
  close of the N/2 days before").
- [S39] v2/phase2-spec.md line ~39: "Weight-sized models are converted to this formula, which lifts today's E13-6
  refusal on perps" (moved here from P2-1 by the coordinator: "P2-1b, not P2-1").

ASSUMED INTERFACES (adapt names, never assertions):
- Opt-in, as P2-1: the strategy param sizing="central" converts a weight-sized model on a perp (spec l.3: the legacy
  weight version, without it, stays spot only and refused on perps).
- The band variant is the param resize_band=0.25; without it (None) the variant is "entry-sized, fixed". (The
  existing rebalance_band can't tell them apart: the vol_target trend filter defaults it to 0.25.)
- Decisions as on main: res.decisions[order_id] = {"intent", "reason", "signal"}; an entry's signal carries close,
  sized_by, stop_frac and liquidation_px; its stop is a following decision whose intent contains "stop" and whose
  signal carries "trigger". An add (band variant) is a decision with intent "entry" (spec l.19: "An add is an entry").
- The study records converted variants in the IdeaLedger with params that include sizing="central" (and resize_band
  for the band variant), under the legacy SPEC's idea and family; TrialsRegister.import_ledger carries them over.
- sleeve_fund.portfolio.conversion (pure, no engine): trades_from_weights(weights, stopped=()) -> list of
  (bar index, "entry" | "exit", side) from a weight series, with `stopped` the bars on which a stop filled;
  band_target(central_qty, weight, full_weight) -> Decimal; band_order(target_qty, held_qty, band=Decimal("0.25"))
  -> the signed order (0 inside the band); gross_gap_flag(converted, legacy) -> bool (the diagnostic).
- An exit decision's signal carries the model's state as on main (Donchian: share_long and long_<n>d).
- StudyResult.legacy_comparison = {"converted": {...}, "legacy": {...}, "investigate": bool}, each side with
  turnover, fee_drag and gross_return, the legacy side a spot run of the legacy weight version over the same period;
  render() prints both turnover and fee-drag figures beside each other.
"""

import math
from decimal import Decimal as D

import pytest

ITEM = "P2-1b weight-sized perp models to entry-sized trades"


def xf(done: str):
    return pytest.mark.xfail(strict=True, reason=f"{ITEM} Done-when: {done}")


DONCHIAN = {"market": "perp", "lookbacks": "10,20", "vol_target": 0.25, "vol_lookback_days": 20, "sizing": "central"}
TREND = {"market": "perp", "fast": 5, "slow": 20, "vol_target": 0.4, "vol_lookback_days": 10, "sizing": "central"}
MODELS = [("donchian", DONCHIAN), ("trend_filter", TREND)]
# Two up-trends and two falls after a flat start: several entries and exits, and volatility that moves. The flat start
# is 280 bars so ATR(14) has settled (#157: 20 lengths) before the first trend bar (QD, 7 Oct 01:56; data only).
CLOSES = ([100.0] * 280 + [100 + 1.5 * i for i in range(1, 41)] + [160 - 2 * i for i in range(1, 31)]
          + [100 + 2 * i for i in range(1, 41)] + [180 - 2.5 * i for i in range(1, 31)])


def _bt(prices, instrument, strategy, params, profile="balanced"):
    from sleeve_fund.research.runner import run_backtest
    from test_backtest import _path

    px = _path(prices, CLOSES)
    res = run_backtest(strategy, px, instrument, params, risk_profile=profile, half_spread=0)
    assert not res.handler_errors, res.handler_errors
    return px, res


def _orders(res):
    """(order_id, signed qty, price, decision) in fill order."""
    out = []
    for oid, r in res.fills.sort_values("ts_last").iterrows():
        q = float(r.filled_qty) * (1 if str(r.side) == "BUY" else -1)
        out.append((oid, q, float(r.avg_px), res.decisions.get(oid, {})))
    return out


def _wilder_atr(px, n=14):
    hi, lo, cl = px["high"], px["low"], px["close"]
    tr = (hi - lo).combine((hi - cl.shift()).abs(), max).combine((lo - cl.shift()).abs(), max)
    atr = [float("nan")] * len(tr)
    atr[n] = float(tr.iloc[1:n + 1].mean())
    for k in range(n + 1, len(tr)):
        atr[k] = (atr[k - 1] * (n - 1) + float(tr.iloc[k])) / n
    return atr


def _bar_of(px, res, oid, entry):
    """The decision bar of entry `oid`, found by its fill time, not its price: the path repeats prices across legs (QD,
    7 Oct 01:57). Set-up check: that bar (or the one before, if a fill is stamped at the next bar's open) closes at the
    entry's decision close."""
    import pandas as pd

    k = int(px.index.get_indexer([pd.Timestamp(res.fills.loc[oid, "ts_last"])], method="pad")[0])
    found = [j for j in (k, k - 1) if j >= 0 and abs(float(px["close"].iloc[j]) - entry) < 1e-9]
    assert found, ("set-up: no bar at the fill time closes at the entry's decision close", oid, k, entry)
    return found[0]


# --- the refusal lifts for a converted model only ----------------------------------------------------------------

@pytest.mark.parametrize("strategy, params", MODELS)
def test_the_weight_sized_perp_refusal_lifts_only_for_an_opted_in_model(strategy, params):
    from sleeve_fund.strategies import PERP_WEIGHT_REFUSAL, check_perp_sizing

    check_perp_sizing(strategy, params)
    legacy = {k: v for k, v in params.items() if k != "sizing"}
    with pytest.raises(ValueError, match=PERP_WEIGHT_REFUSAL):  # the legacy weight version stays spot only
        check_perp_sizing(strategy, legacy)
    check_perp_sizing(strategy, {**legacy, "market": "spot"})


# --- DW1/DW2: entries and exits from the signal, size fixed ------------------------------------------------------

@pytest.mark.parametrize("strategy, params", MODELS)
def test_a_converted_perp_model_trades_only_entries_and_exits(prices, instrument, strategy, params):
    _, res = _bt(prices, instrument, strategy, params)
    orders = _orders(res)
    assert any(d.get("intent") == "entry" for *_, d in orders), "no entry: the test path gave nothing to check"
    held, size = 0.0, None
    for oid, q, _, d in orders:
        intent = d.get("intent", "")
        assert intent == "entry" or intent == "exit" or "stop" in intent, (oid, intent)
        if abs(held) < 1e-12:  # flat: only an entry opens
            assert intent == "entry", (oid, intent)
            size = abs(q)
        else:  # held: every order closes all of it, at the size fixed at entry
            assert intent != "entry", ("an add or resize in the fixed variant", oid)
            assert abs(held + q) < 1e-9 and abs(q) == pytest.approx(size), (oid, held, q)
        held += q
    if strategy == "donchian":  # [A17-1] the trade ends when the weight returns to 0, not when one third leaves
        for oid, q, _, d in orders:
            if d.get("intent") == "exit":
                assert d["signal"]["share_long"] == 0, (oid, d["signal"])


@pytest.mark.parametrize("strategy, params", MODELS)
def test_a_converted_entry_is_sized_by_central_sizing_at_1pct_risk(prices, instrument, strategy, params):
    _, res = _bt(prices, instrument, strategy, params)
    oid, q, px, d = next(o for o in _orders(res) if o[3].get("intent") == "entry")
    sized_by = d["signal"]["sized_by"]
    assert any(w in sized_by for w in ("risk per trade", "volatil", "cap", "liquidation")), sized_by
    assert "weight" not in sized_by, sized_by
    assert abs(q) * px * d["signal"]["stop_frac"] <= 0.01 * 10_000 + 1e-6  # 1% of the 10,000 it starts with


@pytest.mark.parametrize("strategy, params", MODELS)
def test_the_band_variant_resizes_only_past_25pct_and_an_add_is_an_entry(prices, instrument, strategy, params):
    _, res = _bt(prices, instrument, strategy, {**params, "resize_band": 0.25})
    orders = _orders(res)
    held = 0.0
    for i, (oid, q, _, d) in enumerate(orders):
        new = held + q
        assert held * new >= -1e-12, ("one order crossed zero", oid)  # a sign change is an exit then an entry (l.8)
        if abs(held) > 1e-12 and abs(new) > 1e-12:  # a resize, not an open or a close
            assert abs(abs(new) - abs(held)) > 0.25 * abs(held) - 1e-9, (oid, held, new)  # never inside the band
            if abs(new) > abs(held):
                assert d.get("intent") == "entry" and d["signal"].get("sized_by"), (oid, d)
            else:
                assert d.get("intent") != "entry", (oid, d)
        held = new


# --- DW3: every trade has a placed stop within half the distance to liquidation ----------------------------------

def _stops_after(res, oid):
    ids = list(res.decisions)
    nxt = ids[ids.index(oid) + 1: ids.index(oid) + 4]
    return [res.decisions[o] for o in nxt if "stop" in res.decisions[o]["intent"] and "trigger" in
            res.decisions[o]["signal"]]


@pytest.mark.parametrize("band", [None, 0.25])
@pytest.mark.parametrize("strategy, params", MODELS)
def test_every_converted_trade_has_a_placed_stop_within_half_way_to_liquidation(prices, instrument, strategy, params,
                                                                                band):
    px, res = _bt(prices, instrument, strategy, {**params, **({"resize_band": band} if band else {})})
    atr = _wilder_atr(px)
    entries = [o for o in _orders(res) if o[3].get("intent") == "entry"]
    assert entries
    for oid, q, fill, d in entries:
        stops = _stops_after(res, oid)
        assert stops, ("no resting stop placed for", oid, d["intent"])
        entry, trigger = d["signal"]["close"], stops[0]["signal"]["trigger"]
        k = _bar_of(px, res, oid, entry)
        assert (trigger - entry) * q < 0, (entry, trigger, q)  # below a long's entry
        assert abs(entry - trigger) <= 2.5 * max(atr[k], atr[k - 1]) * 1.02, (entry, trigger, atr[k])
        liq = d["signal"]["liquidation_px"]
        assert abs(entry - trigger) <= 0.5 * abs(entry - liq) + 1e-9, (entry, trigger, liq)


def test_the_donchian_ensembles_stop_is_the_tighter_of_its_active_channels_and_the_atr(prices, instrument):
    px, res = _bt(prices, instrument, "donchian", DONCHIAN)
    atr, closes = _wilder_atr(px), list(px["close"])
    lookbacks = [int(n) for n in DONCHIAN["lookbacks"].split(",")]
    entries = [o for o in _orders(res) if o[3].get("intent") == "entry"]
    assert entries
    for oid, q, fill, d in entries:
        entry, (stop, *_) = d["signal"]["close"], _stops_after(res, oid)
        k = _bar_of(px, res, oid, entry)
        active = [n for n in lookbacks if d["signal"].get(f"long_{n}d")]
        assert active, d["signal"]
        level = min(min(closes[k - n // 2 + 1:k + 1]) for n in active)  # reading (a): the bar just closed counts
        want = [max(level, entry - 2.5 * a) for a in (atr[k], atr[k - 1])]  # a long: the tighter is the higher
        assert any(abs(stop["signal"]["trigger"] - w) <= 0.001 * entry for w in want), (stop["signal"]["trigger"],
                                                                                         want, active)


def test_the_trend_filters_placed_stop_is_the_atr_fallback_and_its_average_exit_is_on_the_close(prices, instrument):
    px, res = _bt(prices, instrument, "trend_filter", TREND)
    atr = _wilder_atr(px)
    orders = _orders(res)
    entries = [o for o in orders if o[3].get("intent") == "entry"]
    assert entries
    for oid, q, fill, d in entries:
        entry, stops = d["signal"]["close"], _stops_after(res, oid)
        assert len(stops) == 1, stops  # one resting order: the ATR stop, nothing at the average
        k = _bar_of(px, res, oid, entry)
        assert any(abs(stops[0]["signal"]["trigger"] - (entry - 2.5 * a)) <= 0.001 * entry for a in (atr[k], atr[k - 1]))
    for oid, q, fill, d in orders:
        if d.get("intent") == "exit":  # filled at a bar's close, not at a touched level inside the bar
            assert "trigger" not in d["signal"], d
            assert any(abs(fill - c) <= 0.002 * c for c in px["close"]), fill


# --- the conversion rules on synthetic series (pure) -------------------------------------------------------------

def test_a_sign_change_in_the_weight_is_an_exit_then_an_entry():
    from sleeve_fund.portfolio.conversion import trades_from_weights

    w = [0.0, 0.5, 0.8, -0.3, -0.6, 0.0, 0.0]
    assert trades_from_weights(w) == [(1, "entry", 1), (3, "exit", 1), (3, "entry", -1), (5, "exit", -1)]


def test_after_a_stop_out_re_entry_waits_for_the_weight_to_go_to_zero_and_back():
    from sleeve_fund.portfolio.conversion import trades_from_weights

    w = [0.0, 0.5, 0.5, 0.6, 0.5, 0.0, 0.4, 0.4]
    assert trades_from_weights(w, stopped={2}) == [(1, "entry", 1), (2, "exit", 1), (6, "entry", 1)]


def test_the_band_target_is_central_sizing_times_weight_over_full_weight():
    from sleeve_fund.portfolio.conversion import band_target

    assert band_target(D("2.000"), 0.3, 0.6) == D("1.000")
    assert band_target(D("2.000"), 0.6, 0.6) == D("2.000")


@pytest.mark.parametrize("target, order", [(D("1.250"), D(0)), (D("1.251"), D("0.251")),
                                           (D("0.750"), D(0)), (D("0.749"), D("-0.251"))])
def test_the_band_trades_only_past_25pct_of_the_size_held(target, order):
    from sleeve_fund.portfolio.conversion import band_order

    assert band_order(target, D("1.000"), D("0.25")) == order


@pytest.mark.parametrize("converted, legacy, flag", [
    (0.129, 0.10, False), (0.131, 0.10, True),  # 29% / 31% above
    (0.071, 0.10, False), (0.069, 0.10, True),  # 29% / 31% below
    (0.115, 0.10, False),  # 1.5 points but only 15%: relative applies at |legacy| >= 5%
    (0.034, 0.02, False), (0.036, 0.02, True),  # 1.4 / 1.6 points under the 5% floor
    (0.030, 0.02, False),  # 50% relative but only 1 point: the floor's gap applies
    (0.004, -0.010, False), (0.006, -0.010, True),  # a small negative legacy: 1.4 / 1.6 points
])
def test_the_gross_gap_diagnostic_is_30pct_relative_with_a_1_5_point_floor(converted, legacy, flag):
    from sleeve_fund.research.conversion import gross_gap_flag

    assert gross_gap_flag(converted, legacy) is flag


# --- DW4/DW5: recording, turnover and fee drag -------------------------------------------------------------------

_STUDIES: dict = {}


def _study(tmp_path_factory, variant):
    """One small study per variant (cached for the module): Donchian on a perpetual venue, converted."""
    if variant not in _STUDIES:
        from sleeve_fund.data import synthetic_ohlcv
        from sleeve_fund.research.ledger import IdeaLedger
        from sleeve_fund.research.study import run_study
        from sleeve_fund.strategies.donchian import SPEC
        from sleeve_fund.venues import venue

        ledger = IdeaLedger(tmp_path_factory.mktemp("ledger") / "l.jsonl")
        params = {k: v for k, v in DONCHIAN.items() if k != "market"}
        if variant == "band":
            params["resize_band"] = 0.25
        r = run_study(SPEC, synthetic_ohlcv(days=1300, seed=3), venue("binance").instrument("BTC", "USDT"), "syn",
                      ledger, default_params=params, synthetic=True, holdout_days=100, train_days=500, test_days=300)
        _STUDIES[variant] = (r, ledger)
    return _STUDIES[variant]


def test_converted_variants_are_recorded_under_the_legacy_family_as_new_variants(tmp_path_factory):
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.trials import TrialsRegister
    from sleeve_fund.store import Store

    keys = set()
    for variant in ("fixed", "band"):
        _, ledger = _study(tmp_path_factory, variant)
        rows = [e for e in ledger.entries() if e["family"] != "benchmark"]
        assert rows and all(e["idea"] == "donchian" and e["family"] == "trend" for e in rows), rows[:2]
        assert all(e["params"].get("sizing") == "central" for e in rows), rows[:2]  # never a legacy variant's key
        assert all((e["params"].get("resize_band") == 0.25) == (variant == "band") for e in rows), rows[:2]
        keys |= {(e["idea"], str(sorted(e["params"].items()))) for e in rows}
    legacy = IdeaLedger(tmp_path_factory.mktemp("legacy") / "l.jsonl")
    for vt in (0.25, 0.40):
        legacy.record(idea="donchian", family="trend", params={"vol_target": vt}, dataset="syn", stage="sensitivity",
                      sharpe=0.1)
    assert not keys & {(e["idea"], str(sorted(e["params"].items()))) for e in legacy.entries()}  # N grows
    store = Store.in_memory()
    TrialsRegister(store).import_ledger(_study(tmp_path_factory, "fixed")[1].path)
    trials = [t for t in store.trials() if t["family"] != "benchmark"]
    assert trials and all(t["family"] == "trend" and "central" in t["settings"] for t in trials), trials[:1]


def test_turnover_and_fee_drag_are_reported_beside_the_legacy_weight_version(tmp_path_factory):
    from sleeve_fund.research.tearsheet import render

    r, ledger = _study(tmp_path_factory, "fixed")
    cmp = r.legacy_comparison
    for side in ("converted", "legacy"):
        for k in ("turnover", "fee_drag", "gross_return"):
            assert math.isfinite(cmp[side][k]) and (k == "gross_return" or cmp[side][k] >= 0), (side, k, cmp[side])
    assert cmp["converted"]["turnover"] == pytest.approx(r.turnover, rel=0.01)
    conv, leg = cmp["converted"]["gross_return"], cmp["legacy"]["gross_return"]
    want = abs(conv - leg) > (0.015 if abs(leg) < 0.05 else 0.30 * abs(leg))  # A17-3
    assert cmp["investigate"] is want, (conv, leg, cmp["investigate"])
    text = render(r, ledger).lower()
    assert "legacy" in text and text.count("turnover") >= 2 and text.count("fee drag") >= 2, text[-800:]
