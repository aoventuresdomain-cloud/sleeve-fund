"""G5: sizing from signal strength, the engine hook (Advisor spec, 15 Oct 2026 ~09:25 UK). OFF by default: a Model with
no signal_strength setting sizes exactly as before, its values never read. ON, its configured size (the smallest of its
own sizing limits) is scaled by f(strength) in [0, 1], the ceiling; every other limit applies after it, smallest wins.
Missing, NaN or out-of-range strength skips with an alert; below the floor skips with words.

Red first on 4186c10 (sleeve_fund.strength missing; signal_strength an unknown parameter)."""

from __future__ import annotations

from decimal import Decimal as D
from types import SimpleNamespace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as hs

from sleeve_fund import ramp, strength
from sleeve_fund.strategies.base import LongFlatStrategy

LINEAR = {"mapping": "linear", "floor": 0.3}
VOTES = {"mapping": "votes", "min_k": 3}
STEPS = {"mapping": "steps", "bands": [0.33, 0.66, 1.0]}


# ---- the setting ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cfg", [LINEAR, VOTES, STEPS, {"mapping": "linear"}])
def test_the_three_mappings_are_accepted(cfg):
    assert strength.check(cfg)["mapping"] == cfg["mapping"]


@pytest.mark.parametrize("cfg", [{"mapping": "curve"}, {"mapping": "linear", "floor": 1.5},
                                 {"mapping": "linear", "floor": 0.2, "gain": 2}, {"mapping": "votes", "min_k": 0},
                                 {"mapping": "steps", "bands": [0.2, 0.4, 0.6, 1.0]},
                                 {"mapping": "steps", "bands": [0.66, 0.33]}, {"mapping": "steps", "bands": [0, 1]},
                                 "linear"])
def test_no_free_form_curve_or_out_of_range_setting_is_accepted(cfg):
    with pytest.raises(ValueError):
        strength.check(cfg)


def test_off_is_none():
    assert strength.check(None) is None


# ---- f(strength) -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cfg, values, f", [
    (strength.check(LINEAR), {"strength": 1.0}, 1.0),  # s = 1: the configured size
    (strength.check(LINEAR), {"strength": 0.5}, 0.5),
    (strength.check(LINEAR), {"strength": 0.3}, 0.3),  # at the floor: taken
    (strength.check(VOTES), {"votes": (5, 8)}, 5 / 8),
    (strength.check(VOTES), {"votes": [8, 8]}, 1.0),
    (strength.check(STEPS), {"strength": 0.5}, 0.33),
    (strength.check(STEPS), {"strength": 0.66}, 0.66),
    (strength.check(STEPS), {"strength": 1}, 1.0),
])
def test_the_factor(cfg, values, f):
    st = strength.factor(cfg, values)
    assert st.f == pytest.approx(f) and not st.alert


@pytest.mark.parametrize("cfg, values", [(strength.check(LINEAR), {"strength": 0.29}),
                                         (strength.check({"mapping": "linear", "floor": 0}), {"strength": 0}),
                                         (strength.check(VOTES), {"votes": (2, 8)}),
                                         (strength.check(STEPS), {"strength": 0.2})])
def test_below_its_floor_the_entry_is_skipped_with_words_not_an_alert(cfg, values):
    st = strength.factor(cfg, values)
    assert st.f is None and not st.alert and st.words.startswith("entry skipped")


@pytest.mark.parametrize("cfg, values", [
    (strength.check(LINEAR), {}), (strength.check(LINEAR), {"strength": float("nan")}),
    (strength.check(LINEAR), {"strength": 1.01}), (strength.check(LINEAR), {"strength": -0.1}),
    (strength.check(LINEAR), {"strength": "0.5"}), (strength.check(LINEAR), {"strength": True}),
    (strength.check(STEPS), {"strength": float("inf")}),
    (strength.check(VOTES), {}), (strength.check(VOTES), {"votes": (9, 8)}), (strength.check(VOTES), {"votes": (1, 0)}),
    (strength.check(VOTES), {"votes": (0.5, 8)}), (strength.check(VOTES), {"votes": 5}),
])
def test_a_missing_nan_or_out_of_range_strength_skips_with_an_alert_never_full_size(cfg, values):
    st = strength.factor(cfg, values)
    assert st.f is None and st.alert


# ---- the engine ------------------------------------------------------------------------------------------------------
class _NoRead(dict):
    def get(self, *a, **k):
        raise AssertionError("values read with signal strength off")


def _strategy(cfg):
    events = []
    s = SimpleNamespace(_cfg=SimpleNamespace(signal_strength=strength.check(cfg)), _noted=set(), events=events,
                        settle_bars_needed=None, _decision_bars=0)
    s._note = lambda kind, msg, level="warning": events.append((kind, level, msg))
    return s


def _limits():
    # risk per trade is the Model's configured size (6% of a 100,000 fund); the rest are caps on it
    return {"3x leverage cap": D(297_000), "balanced risk profile cap": D(50_000), "risk per trade": D(6_000),
            "largest order cap": D(250_000)}


def test_off_the_limits_are_untouched_and_the_values_never_read():
    limits = _limits()
    assert LongFlatStrategy._strength_limit(_strategy(None), limits, _NoRead(), 100_000.0) is None
    assert limits == _limits()


def test_linear_0_62_scales_the_configured_size_and_journals_it_in_words():
    limits = _limits()
    got = LongFlatStrategy._strength_limit(_strategy(LINEAR), limits, {"strength": 0.62}, 100_000.0)
    assert limits[strength.LIMIT] == D(6_000) * D("0.62") == D("3720.00")
    assert got == "signal strength 0.62 (linear, floor 0.30) -> 62% of 6.0% = 3.72% of fund"
    assert min(limits, key=limits.get) == strength.LIMIT


def test_votes_5_of_8_is_5_8ths_of_the_configured_size():
    limits = _limits()
    LongFlatStrategy._strength_limit(_strategy(VOTES), limits, {"votes": (5, 8)}, 100_000.0)
    assert limits[strength.LIMIT] == D(6_000) * D(repr(5 / 8)) == D(3_750)


def test_a_cap_below_the_scaled_size_still_wins():
    limits = {**_limits(), "share of the bar's volume": D(1_000)}
    LongFlatStrategy._strength_limit(_strategy(LINEAR), limits, {"strength": 0.5}, 100_000.0)
    assert min(limits, key=limits.get) == "share of the bar's volume"  # the cap is never scaled


@pytest.mark.parametrize("values, level", [({"strength": 0.1}, "info"), ({"strength": float("nan")}, "error")])
def test_a_skipped_entry_is_journaled_and_a_bad_strength_raises_an_alert(values, level):
    s, limits = _strategy(LINEAR), _limits()
    assert LongFlatStrategy._strength_limit(s, limits, values, 100_000.0) is False
    assert s.events and s.events[0][:2] == ("entry_skipped_strength", level) and limits == _limits()


def test_strength_and_the_ramp_the_smallest_wins():
    """Strength first, then the ramp (A9): a strength-scaled 3,720 against the ramp's one minimum order of 120."""
    limits = _limits()
    LongFlatStrategy._strength_limit(_strategy(LINEAR), limits, {"strength": 0.62}, 100_000.0)
    limits[ramp.LIMIT] = ramp.cap(D("0.002"), D(0), 1, D(60_000))
    assert min(limits, key=limits.get) == ramp.LIMIT
    # and the ramp's un-ramped size for its track is the strength-scaled one, not the configured maximum
    assert min(v for k, v in limits.items() if k != ramp.LIMIT) == D("3720.00")


@settings(max_examples=300, deadline=None)
@given(s=hs.floats(0, 1, exclude_min=True), risk=hs.integers(1, 10**7), cap=hs.integers(1, 10**7), lev=hs.integers(1, 10**7),
       base=hs.floats(1, 1e7))
def test_strength_never_makes_a_position_bigger_than_its_configured_size(s, risk, cap, lev, base):
    limits = {"risk per trade": D(risk), "balanced risk profile cap": D(cap), "3x leverage cap": D(lev)}
    before = min(limits.values())
    configured = min(D(risk), D(cap))
    got = LongFlatStrategy._strength_limit(_strategy({"mapping": "linear", "floor": 0.0}), limits, {"strength": s},
                                           base)
    assert got and limits[strength.LIMIT] <= configured and min(limits.values()) <= before


def test_an_unsettled_strength_block_refuses_the_entry():
    """A10: before the Model's warm-up (settle_bars_needed) its strength isn't read: no entry, journaled."""
    s, limits = _strategy(LINEAR), _limits()
    s.settle_bars_needed, s._decision_bars = 50, 49
    assert LongFlatStrategy._strength_limit(s, limits, {"strength": 0.9}, 100_000.0) is False
    assert "haven't settled" in s.events[0][2] and limits == _limits()
    s._decision_bars, s.events[:] = 50, []
    assert LongFlatStrategy._strength_limit(s, limits, {"strength": 0.9}, 100_000.0)


# ---- F-G5-1 (CR): a weight-sized Model ------------------------------------------------------------------------------
def test_a_weight_sized_models_target_weight_is_its_configured_size_so_strength_bites():
    """CR F-G5-1: on the weight-sized path the target weight is the Model's own size; with the risk profile's cap above
    it, f must scale the target weight, not the cap (else strength silently does nothing)."""
    limits = {"free cash": D(99_000), "target weight": D(30_000), "balanced risk profile cap": D(50_000)}
    LongFlatStrategy._strength_limit(_strategy(LINEAR), limits, {"strength": 0.5}, 100_000.0)
    assert limits[strength.LIMIT] == D(15_000) and min(limits, key=limits.get) == strength.LIMIT


def test_strength_with_rebalancing_to_a_target_weight_is_refused_at_create():
    """Rebalance trades (the band's top-ups) go outside the entry sizers, where strength can't reach: refused, not a
    setting that silently does nothing (HoE F-G5-1)."""
    from nautilus_trader.model import BarType, InstrumentId

    from sleeve_fund.strategies.base import LongFlatConfig

    with pytest.raises(ValueError, match="signal strength"):
        LongFlatConfig(instrument_id=InstrumentId.from_str("BTC/USDT.BINANCE"),
                       bar_type=BarType.from_str("BTC/USDT.BINANCE-1-DAY-LAST-EXTERNAL"), assumed_taker_fee=0.001,
                       rebalance_band=0.2, signal_strength=LINEAR)


def test_the_start_check_fails_strength_on_a_rebalancing_strategy_or_a_bad_setting(tmp_path):
    from sleeve_fund.start_check import FAIL, PASS, check_strength
    from sleeve_fund.store import Store

    store = Store(f"sqlite:///{tmp_path}/sc.db")
    for name, params in (("ok", {"signal_strength": LINEAR}), ("rb", {"signal_strength": LINEAR, "rebalance_band": 0.2}),
                         ("bad", {"signal_strength": {"mapping": "curve"}}), ("off", {})):
        store.create_sleeve(name=name, strategy="buy_and_hold", instrument="BTC/USDT", bar_spec="1-DAY-LAST-EXTERNAL",
                            starting_balance=0, risk_profile="balanced", params=params)
    sec = check_strength(store)
    by = {what.split(":")[0]: st for st, what in sec.lines}
    assert by == {"ok": PASS, "rb": FAIL, "bad": FAIL} and sec.failed


def test_the_start_check_page_carries_the_strength_section_and_fails_on_it(tmp_path):
    """CR F-G5-2: the section is part of start_check.run, not only callable: a strength+rebalance Strategy fails the
    whole check, named under "Sizing by signal strength"."""
    from sleeve_fund import start_check
    from sleeve_fund.store import Store

    url = f"sqlite:///{tmp_path}/run.db"
    Store(url).create_sleeve(name="rb", strategy="buy_and_hold", instrument="BTC/USDT", bar_spec="1-DAY-LAST-EXTERNAL",
                             starting_balance=0, risk_profile="balanced",
                             params={"signal_strength": LINEAR, "rebalance_band": 0.2})
    ok, page = start_check.run(url, environ={})
    assert not ok and "Sizing by signal strength: FAIL" in page and f"rb: {strength.NO_REBALANCE}" in page, page


# ---- QD review of 8c99e9f (MUST before G5 is switched on) -------------------------------------------------------------
def _cfg(**params):
    from nautilus_trader.model import BarType, InstrumentId

    from sleeve_fund.strategies.base import LongFlatConfig

    return LongFlatConfig(instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
                          bar_type=BarType.from_str("BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL"),
                          assumed_taker_fee=0.0005, market="perp", **params)


def test_f_g5_4_strength_on_a_pair_leg_is_refused_not_silently_ignored():
    """QD F-G5-4: a leg trades only through its Pair, which sizes it; strength there would do nothing, so it's refused."""
    _cfg(signal_strength=LINEAR)  # fine on its own
    with pytest.raises(ValueError, match="signal strength"):
        _cfg(signal_strength=LINEAR, leg={"side": 1, "ratio": 1, "leverage": 1})


@pytest.mark.parametrize("strategy, params", [
    ("donchian", {"signal_strength": LINEAR}),  # spot Donchian: its config injects a 20% band
    ("trend_filter", {"signal_strength": LINEAR, "vol_target": 0.2}),  # vol target: a 25% band
])
def test_f_g5_3_the_start_check_reads_the_built_config_so_an_injected_band_fails(tmp_path, strategy, params):
    from sleeve_fund.start_check import FAIL, check_strength
    from sleeve_fund.store import Store

    store = Store(f"sqlite:///{tmp_path}/sc.db")
    store.create_sleeve(name="inj", strategy=strategy, instrument="BTC/USDT", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=0, risk_profile="balanced", params=params)
    sec = check_strength(store)
    assert sec.failed and sec.lines == [(FAIL, f"inj: {strength.NO_REBALANCE}")], sec.lines


def _noting():
    """A Strategy whose _note dedupes as the real one does (once per kind until _noted.discard)."""
    s = _strategy(LINEAR)

    def note(kind, msg, level="warning"):
        if kind not in s._noted:
            s._noted.add(kind)
            s.events.append((kind, level, msg))

    s._note = note
    return s


def test_f_g5_5_a_skip_is_written_once_per_change_not_every_bar():
    """QD F-G5-5: 1,440 one-minute bars below the floor write one event, not 1,440; an alert after an info skip is
    still written; a passed strength clears it, so the next skip is written again."""
    s = _noting()
    for _ in range(1440):
        LongFlatStrategy._strength_limit(s, _limits(), {"strength": 0.1}, 100_000.0)
    assert [e[1] for e in s.events] == ["info"]
    LongFlatStrategy._strength_limit(s, _limits(), {"strength": float("nan")}, 100_000.0)
    LongFlatStrategy._strength_limit(s, _limits(), {"strength": float("nan")}, 100_000.0)
    assert [e[1] for e in s.events] == ["info", "error"]
    assert LongFlatStrategy._strength_limit(s, _limits(), {"strength": 0.9}, 100_000.0)
    LongFlatStrategy._strength_limit(s, _limits(), {"strength": 0.1}, 100_000.0)
    assert [e[1] for e in s.events] == ["info", "error", "info"]


@pytest.mark.parametrize("values, f", [({"strength": __import__("numpy").float64(0.5)}, 0.5),
                                       ({"strength": __import__("numpy").float32(0.5)}, 0.5),
                                       ({"strength": D("0.5")}, 0.5), ({"strength": __import__("numpy").int64(1)}, 1.0)])
def test_qd_nit_numpy_and_decimal_strengths_are_read_not_skipped(values, f):
    assert strength.factor(strength.check(LINEAR), values).f == pytest.approx(f)


def test_qd_nit_numpy_votes_are_read_and_a_huge_int_skips_with_an_alert():
    import numpy as np

    assert strength.factor(strength.check(VOTES), {"votes": (np.int64(5), np.int64(8))}).f == 5 / 8
    st = strength.factor(strength.check(LINEAR), {"strength": 10**400})
    assert st.f is None and st.alert


def test_f_g5_6_a_stepped_models_later_open_is_not_scaled_again():
    """HoE F-G5-6: a stepped Model's later open is capped at Q, its first open's size, which strength already scaled;
    scaling the step again would give f squared. The step's size stands, the strength isn't read again."""
    s = _strategy(LINEAR)
    limits = {**_limits(), "the step's size": D(3_000)}  # Q from a first open at strength 0.5
    got = LongFlatStrategy._strength_limit(s, limits, {"strength": 0.5, "step_qty": 0.05}, 100_000.0)
    assert strength.LIMIT not in limits and min(limits, key=limits.get) == "the step's size"
    assert got and "first open" in got and not s.events
    # a later open with no strength of its own isn't skipped either: Q was sized at a read strength
    assert LongFlatStrategy._strength_limit(s, dict(limits), {"step_qty": 0.05}, 100_000.0) and not s.events


@pytest.mark.parametrize("params, verdict", [({"signal_strength": LINEAR, "market": "perp"}, "PASS"),
                                             ({"signal_strength": {"mapping": "curve"}, "market": "perp"}, "FAIL")])
def test_the_start_check_builds_a_multi_asset_strategy_from_the_settings_its_engine_takes(tmp_path, params, verdict):
    """HoE 10:12 UK: on a multi-asset row check_strength builds the config from store.model_params, as the engine does
    (paper.config.from_store, MA-PARAMS-1): the asset list and universe date never fail it as unknown settings, and its
    strength setting reaches the build (a bad one fails it, named)."""
    from datetime import UTC, datetime, timedelta

    from sleeve_fund.paper.config import from_store
    from sleeve_fund.start_check import check_strength
    from sleeve_fund.store import Store, model_params

    store = Store(f"sqlite:///{tmp_path}/ma.db")
    at = datetime.now(UTC) - timedelta(minutes=5)
    store.settle_flow(store.fund_flow("deposit", 100_000, actor="pm", reason="test", rebase=True, now=at), at)
    store.create_legs(name="trend-x", strategy="trend_filter", bar_spec="15-MINUTE-LAST-INTERNAL",
                      assets=["BTC/USDT", "ETH/USDT"], params=params, starting_balance=0, universe_date="2026-10-15",
                      venue="BINANCE")
    parent = next(s for s in store.sleeves(include_legs=False) if s.name == "trend-x")
    assert "assets" in parent.params and from_store(parent).params == model_params(parent.params)
    assert "signal_strength" in from_store(parent).params
    sec = check_strength(store)
    assert len(sec.lines) == 1 and sec.lines[0][0] == verdict, sec.lines
    assert sec.lines[0][1].startswith("trend-x: ") and ("sizes by signal strength (linear)" if verdict == "PASS"
                                                       else "signal_strength needs a mapping") in sec.lines[0][1]
