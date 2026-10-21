"""B-35 (a) (HoE 20:06 UK): a Model that places no stop (buy_and_hold, trend_filter) on a perpetual refused its own
defaults. The PM's rule is "always a stop, or 1x": a Strategy's own max_leverage caps its leverage below its risk
profile's, never above it, and a stopless Model on a perpetual is accepted when that effective cap is at most 1x.
Sizing, isolated margin, the liquidation price and every figure read the one capped value.

Expected values are worked by hand from the profiles (conservative 1x, balanced 2x, aggressive 3x; balanced's margin
cap 33%), never read back from the code under test."""

import math
from types import SimpleNamespace

import pytest

from sleeve_fund import risk
from sleeve_fund.risk import PROFILES
from sleeve_fund.strategies import check_perp_stop

PERP = {"market": "perp"}


def test_b35_stopless_perp_model_saves_on_its_defaults_capped_at_1x():
    from nautilus_trader.model import BarType, InstrumentId

    from sleeve_fund.strategies import REGISTRY

    for model in ("buy_and_hold", "trend_filter"):
        for profile in ("balanced", "aggressive"):
            params = {**PERP, "max_leverage": 1}
            check_perp_stop(model, params, profile)  # accepted: held at 1x by its own setting
            assert risk.leverage_cap(PROFILES[profile], params) == 1.0
            _, config_cls = REGISTRY[model]  # the save path builds the config with the stored settings
            cfg = config_cls(instrument_id=InstrumentId.from_str("BTC/USDT.BINANCE"),
                             bar_type=BarType.from_str("BTC/USDT.BINANCE-1-HOUR-LAST-EXTERNAL"),
                             assumed_taker_fee=0.00055, assumed_half_spread=0.0, **params)
            assert cfg.max_leverage == 1.0


@pytest.mark.parametrize("profile, own, cap", [("aggressive", 5, 3.0), ("aggressive", 2, 2.0), ("balanced", 1, 1.0),
                                               ("balanced", 1.5, 1.5), ("balanced", 1.25, 1.25), ("conservative", 3, 1.0),
                                               ("aggressive", None, 3.0)])
def test_b35b_the_own_cap_lowers_the_profiles_never_raises_it(profile, own, cap):
    params = {**PERP, **({} if own is None else {"max_leverage": own})}
    assert risk.leverage_cap(PROFILES[profile], params) == cap
    # sizing: the margin cap times the same leverage (balanced 33% at 1x is 33% of equity as notional)
    assert risk.position_cap(PROFILES[profile], params) == pytest.approx(PROFILES[profile].max_position_pct * cap)


@pytest.mark.parametrize("bad", ["two", "1", True, 0, -1, 0.5, 0.99, 1.555, math.nan, math.inf, [2]])
def test_b35c_a_missing_or_invalid_own_cap_reads_the_profile_and_is_refused_on_save(bad):
    from nautilus_trader.model import BarType, InstrumentId

    from sleeve_fund.strategies import REGISTRY

    params = {**PERP, "max_leverage": bad}
    assert risk.leverage_cap(PROFILES["aggressive"], params) == 3.0  # never above the profile, never a crash
    assert risk.position_cap(PROFILES["aggressive"], params) == pytest.approx(PROFILES["aggressive"].max_position_pct * 3)
    _, config_cls = REGISTRY["ping_pong"]
    with pytest.raises(ValueError, match="max_leverage .*minimum 1x; use size share to trade smaller"):
        config_cls(instrument_id=InstrumentId.from_str("BTC/USDT.BINANCE"),
                   bar_type=BarType.from_str("BTC/USDT.BINANCE-1-HOUR-LAST-EXTERNAL"),
                   assumed_taker_fee=0.00055, assumed_half_spread=0.0, **params)
    with pytest.raises(ValueError, match="perpetual places no stop"):  # a stopless Model is not let through by it
        check_perp_stop("buy_and_hold", params, "aggressive")


@pytest.mark.parametrize("own", [None, 2, 1.5])
def test_b35d_stopless_above_1x_still_refuses(own):
    params = {**PERP, **({} if own is None else {"max_leverage": own})}
    with pytest.raises(ValueError, match="perpetual places no stop"):
        check_perp_stop("buy_and_hold", params, "aggressive")


def test_b35e_spot_is_unchanged():
    params = {"max_leverage": 1}
    check_perp_stop("buy_and_hold", {}, "aggressive")  # spot: never refused, as before
    for name, p in PROFILES.items():
        assert risk.position_cap(p, params) == p.max_position_pct == risk.position_cap(p, {})


def test_b35f_two_leg_perp_leg_takes_the_lowest_of_its_leg_2x_and_the_own_cap():
    p = PROFILES["aggressive"]
    assert risk.leverage_cap(p, {**PERP, "leg": {"leverage": 3}}) == 2.0  # as before
    assert risk.leverage_cap(p, {**PERP, "leg": {"leverage": 3}, "max_leverage": 1}) == 1.0
    assert risk.leverage_cap(p, {**PERP, "leg": {"leverage": 1.5}, "max_leverage": 2}) == 1.5


@pytest.mark.parametrize("own, leg, lev", [(None, None, 3.0), (1.0, None, 1.0), (5.0, None, 3.0), (1.0, 2.0, 1.0),
                                           (None, 2.5, 2.0)])
def test_b35g_the_engines_margin_and_liquidation_read_the_same_cap(own, leg, lev):
    """LongFlatStrategy._max_leverage is the one figure for the engine's sizing, isolated margin and liquidation
    price (and through risk.leverage_cap, the gate's book and the dashboard): it must be the capped one."""
    from sleeve_fund.strategies.base import LongFlatStrategy

    strat = SimpleNamespace(runtime=SimpleNamespace(profile=PROFILES["aggressive"]), leg_leverage=leg,
                            _cfg=SimpleNamespace(max_leverage=own))
    assert LongFlatStrategy._max_leverage(strat) == lev
    params = {**PERP, **({} if own is None else {"max_leverage": own}), **({} if leg is None else {"leg": {"leverage": leg}})}
    assert risk.leverage_cap(PROFILES["aggressive"], params) == lev


def test_b35h_a_paper_runtime_sizes_on_the_capped_leverage():
    from sleeve_fund.paper.runtime import SleeveRuntime  # noqa: F401 - its cap is risk.position_cap (runtime.py)

    p = PROFILES["balanced"]
    assert risk.position_cap(p, {**PERP, "max_leverage": 1}) == pytest.approx(0.33)
    assert risk.position_cap(p, PERP) == pytest.approx(0.66)


@pytest.mark.parametrize("own, notional", [(None, "6600"), (1, "3300"), (1.5, "4950"), (5, "6600")])
def test_b35i_paper_sizes_a_perp_entry_on_the_capped_leverage(tmp_path, own, notional):
    """End to end on the paper runtime (balanced, 10,000): the margin cap is 33% of equity; at the profile's 2x the
    entry's notional is 6,600, held to 1x by its own cap 3,300, at 1.5x 4,950, and an own cap above the profile
    changes nothing."""
    from decimal import Decimal

    from test_long_short import _meta, _record

    from sleeve_fund.research.replay import replay
    from sleeve_fund.store import Store

    params = {"rise": 0.01, "dip": 0.005, "stop_loss": 0.01, **PERP, "allow_short": True,
              **({} if own is None else {"max_leverage": own})}
    _record(tmp_path / "a.jsonl.gz", _meta(10_000, params), [(5, 0.0), (20, 0.015), (10, 0.0)])
    store = Store.in_memory()
    replay(tmp_path / "a.jsonl.gz", with_fills=True, store=store)
    entry = next(o for o in store.orders("ping-pong-test")[::-1] if o["intent"] == "entry" and o["filled_qty"] > 0)
    got = Decimal(str(entry["filled_qty"])) * Decimal(str(entry["avg_px"]))
    assert abs(got - Decimal(notional)) <= Decimal("1"), got
    # the engine's margin and liquidation read the same cap: a long's liquidation at leverage L is entry x (1 - 1/L +
    # the maintenance margin, 0.5% here), nearer by its entry fee: at 2x about half the entry, at 1x (fully paid for)
    # the maintenance alone (TEC-BH102-1X, HoE and Advisor 21 Oct; until then a 1x long had none)
    liq = entry["signal"].get("liquidation_px")
    lev = {1: 1.0, 1.5: 1.5}.get(own, 2.0)
    px, qty = Decimal(str(entry["avg_px"])), Decimal(str(entry["filled_qty"]))
    fee = sum(Decimal(str(f["fee"])) for f in store.fills("ping-pong-test") if f["order_id"] == entry["order_id"])
    want = px * (1 - 1 / Decimal(str(lev)) + Decimal("0.005")) + fee / qty
    assert abs(Decimal(str(liq)) - want) <= px * Decimal("0.0002"), (liq, want)


@pytest.mark.parametrize("own, cap, lev", [(None, 0.66, 2.0), (1, 0.33, 1.0)])
def test_b35j_a_perp_study_caps_and_liquidates_at_the_own_cap(tmp_path, monkeypatch, own, cap, lev):
    """CR2 MUST 1: the study holds a perp Strategy at its own cap too: balanced's 33% margin at 2x is 66% of capital in
    notional, at its own 1x 33%; and the benchmarks' trades are liquidated at that same leverage."""

    from sleeve_fund import funding, markets
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.research import study
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.strategies.buy_and_hold import SPEC as HOLD
    from sleeve_fund.venues import venue
    from test_binance import INFO, binance_contract  # the venue's contract limits without the network

    b = venue("binance")
    monkeypatch.setattr(b, "contract", lambda pair: binance_contract(pair, get_json=lambda url: INFO))
    monkeypatch.setattr(funding, "DEFAULT_ROOT", tmp_path / "funding")
    seen = []
    real = study._benchmarks

    def spy(*a, **k):
        seen.append(a[6] if len(a) > 6 else k.get("leverage"))
        return real(*a, **k)

    monkeypatch.setattr(study, "_benchmarks", spy)
    r = study.run_study(HOLD, synthetic_ohlcv(days=200, seed=4), b.instrument("BTC", "USDT"), dataset="syn",
                        ledger=IdeaLedger(tmp_path / "l.jsonl"), synthetic=True, holdout_days=0, train_days=60,
                        test_days=60, risk_profile="balanced",
                        default_params={"market": markets.PERP, **({} if own is None else {"max_leverage": own})})
    assert any(f"capped at {cap:.0%} of capital in notional" in n for n in r.notes), r.notes
    assert seen and set(seen) == {lev}
