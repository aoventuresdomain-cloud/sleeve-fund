"""Family C, step (d) (pre-registration, section 6, rev 2; Advisor 16 Nov): the random long/short assignment. Done when:

1. a leg is priced as the engine charges it: the open of the first execution bar at or after its Monday close, half
   the spread and the taker fee each side, funding at each settlement held over (a long pays, a short receives), a
   leg kept across a change week traded once, a leg open at the end marked unpaid, a carried leg valued unpaid;
2. the draws keep the Strategy's change weeks and its carried state, re-draw the ordered pair from the 6, are seeded,
   and the share beaten is read after fees and before (fee 0, spread and funding kept);
3. the Strategy's states come from its trips, a flip on one close included;
4. on a real rank book's run the model's P&L reconciles with the engine's book, and the sheet carries the line in
   place of the random-entry verdict, with no PASS wording."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.research import pair_null as pn

MIN = pd.Timedelta(minutes=1).value
T0 = pd.Timestamp("2025-01-06", tz="UTC")  # a Monday
WK = pd.Timedelta(days=7)


def _asset(name: str, opens, taker=0.0, half=0.0, settle=None, rates=None, baseline=None) -> pn.PairAsset:
    """Minute bars from T0, one per step, each closing a minute after it opens; the close is the next open."""
    opens = np.asarray(opens, dtype=float)
    close_ns = np.array([(T0 + pd.Timedelta(minutes=i + 1)).value for i in range(len(opens))], dtype=np.int64)
    closes = np.concatenate([opens[1:], opens[-1:]])
    return pn.PairAsset(name, close_ns, opens, closes, MIN, taker, half,
                        None if settle is None else np.asarray(settle, dtype=np.int64),
                        None if rates is None else np.asarray(rates, dtype=float),
                        None if baseline is None else np.asarray(baseline, dtype=bool))


def _t(i: int) -> int:
    return (T0 + pd.Timedelta(minutes=i)).value


# ---- 1. pricing ---------------------------------------------------------------------------------------------------

def test_a_new_long_and_short_pay_spread_and_fees_each_side_and_end_marked_unpaid():
    a = _asset("A", [100, 110, 120, 130], taker=0.001, half=0.0005)
    b = _asset("B", [100, 90, 80, 70], taker=0.001, half=0.0005)
    c = _asset("C", [100, 100, 100, 100])
    p = pn._Pricer([a, b, c], [_t(1), _t(2), _t(3)], np.array([1_000.0, 1_000.0]))
    # long A opened at t1 (open 110, +half), closed at t2 (open 120, -half), fees both sides
    entry, out = 110 * 1.0005, 120 * 0.9995
    q = 1_000 / entry
    assert p.leg(0, 1, 1, 2, True) == pytest.approx(q * (out - entry) - 0.001 * q * (entry + out))
    assert p.leg(0, 1, 1, 2, False) == pytest.approx(q * (out - entry))  # pre-fee keeps the spread
    # short B opened at t2 (open 80, -half) and still open at the end, marked at the last close by t3 (70), unpaid
    entry = 80 * (1 - 0.0005)
    q = 1_000 / entry
    assert p.leg(1, -1, 2, 3, True) == pytest.approx(-q * (70 - entry) - 0.001 * q * entry)


def test_a_carried_leg_is_valued_from_the_first_change_weeks_open_without_costs():
    a = _asset("A", [100, 110, 120, 130], taker=0.001, half=0.0005)
    p = pn._Pricer([a, a, a], [_t(1), _t(2), _t(3)], np.array([1_000.0, 1_000.0]))
    q = 1_000 / 110
    assert p.leg(0, 1, 0, 1, True) == pytest.approx(q * (110 * 0.9995 - 110) - 0.001 * q * 110 * 0.9995)


def test_funding_a_long_pays_a_short_receives_over_settlements_held():
    s = [_t(1), _t(2), _t(5)]  # at t1 the leg isn't open yet; t2 held; t5 after the close
    a = _asset("A", [100] * 6, settle=s, rates=[0.01, 0.01, 0.01], baseline=[False, True, False])
    p = pn._Pricer([a, a, a], [_t(1), _t(3), _t(4)], np.array([1_000.0, 1_000.0, 1_000.0]))
    a.settle_baseline[:] = [False, False, False]  # stored rates: a long pays, a short receives
    assert p.leg(0, 1, 1, 2, True) == pytest.approx(-10 * 0.01 * 100)
    assert p.leg(0, -1, 1, 2, True) == pytest.approx(10 * 0.01 * 100)
    a.settle_baseline[:] = [False, True, False]  # t2's is the baseline (none stored): both sides pay it
    p.cache.clear()
    assert p.leg(0, 1, 1, 2, True) == p.leg(0, -1, 1, 2, True) == pytest.approx(-10 * 0.01 * 100)
    assert a.baseline_in(_t(1), _t(4)) == 1


def test_a_leg_kept_across_a_change_week_is_traded_once():
    a = _asset("A", [100, 100, 100, 100, 100], taker=0.01)
    b, c = _asset("B", [100] * 5), _asset("C", [100] * 5)
    p = pn._Pricer([a, b, c], [_t(1), _t(2), _t(3), _t(4)], np.array([1_000.0] * 3))
    kept, legs = p.book([(None, None), (0, 1), (0, 2), (0, 1)], True)
    assert kept == pytest.approx(-10.0) and legs == 4  # A long: one entry fee, open at the end; short legs free
    assert pn._runs([(None, None), (0, 1), (0, 2), (0, 1)]) == [(0, 1, 1, 4), (1, -1, 1, 2), (2, -1, 2, 3),
                                                               (1, -1, 3, 4)]


def test_no_execution_bar_after_a_change_week_is_refused():
    a = _asset("A", [100, 101])
    with pytest.raises(ValueError, match="no execution bar opens"):
        a.open_at(_t(2))


# ---- 2. the draws -------------------------------------------------------------------------------------------------

def _trend(k: int = 30):
    """Three assets over k weeks of one bar a week: A rises, C falls, B is flat."""
    n = k + 2
    bars = [np.array([100 * (1.02 ** i) for i in range(n)]), np.full(n, 100.0), np.array([100 * (0.98 ** i)
                                                                                          for i in range(n)])]
    out = []
    for name, o in zip("ABC", bars):
        close_ns = np.array([(T0 + WK * i + pd.Timedelta(minutes=1)).value for i in range(n)], dtype=np.int64)
        out.append(pn.PairAsset(name, close_ns, o, o, MIN, 0.00055, 0.0002))
    return out, [(T0 + WK * i).value for i in range(k)], (T0 + WK * k).value


def test_the_right_pair_every_week_beats_nearly_every_draw_and_the_wrong_one_almost_none():
    assets, change, end = _trend()
    k = len(change)
    best = pn.pair_null(assets, change, [(None, None)] + [(0, 2)] * k, end, [1_000.0] * k, 10_000.0)
    worst = pn.pair_null(assets, change, [(None, None)] + [(2, 0)] * k, end, [1_000.0] * k, 10_000.0)
    assert best.net_percentile > 99 and best.beats and not best.killed and best.strategy_legs == 2
    assert worst.net_percentile < 1 and not worst.beats and worst.killed
    assert best.change_weeks == k and best.mean_draw_legs > best.strategy_legs  # draws re-trade at each change week


def test_draws_are_seeded_keep_the_carried_state_and_refuse_too_few_or_not_three():
    assets, change, end = _trend(8)
    states = [(1, 2)] + [(0, 2)] * 8
    a = pn.pair_null(assets, change, states, end, [1_000.0] * 8, 10_000.0, seed=0)
    b = pn.pair_null(assets, change, states, end, [1_000.0] * 8, 10_000.0, seed=0)
    c = pn.pair_null(assets, change, states, end, [1_000.0] * 8, 10_000.0, seed=1)
    assert (a.net_percentile, a.median_net) == (b.net_percentile, b.median_net)
    assert (a.median_net, a.mean_draw_legs) != (c.median_net, c.mean_draw_legs)
    assert a.pre_percentile >= 0 and a.strategy_pre > a.strategy_net  # before fees: the fee only
    with pytest.raises(ValueError, match="at least 1,000"):
        pn.pair_null(assets, change, states, end, [1_000.0] * 8, 10_000.0, draws=999)
    with pytest.raises(ValueError, match="three assets"):
        pn.pair_null(assets[:2], change, states, end, [1_000.0] * 8, 10_000.0)
    assert len(pn.ORDERINGS) == 6 and len(set(pn.ORDERINGS)) == 6 and all(lo != sh for lo, sh in pn.ORDERINGS)
    none = pn.pair_null(assets, [], [(None, None)], end, [], 10_000.0)
    assert not none.judged and pn.null_words(none).startswith("NOT JUDGED")


# ---- 3. the Strategy's states -------------------------------------------------------------------------------------

def test_states_come_from_trips_with_a_flip_on_one_close_and_a_carried_leg():
    def m(w, minutes=1):
        return T0 + WK * w + pd.Timedelta(minutes=minutes)

    trips = {0: [{"opened": m(-2), "closed": m(1), "side": 1}, {"opened": m(1, 2), "closed": None, "side": -1}],
             1: [{"opened": m(-2), "closed": m(3), "side": -1}],
             2: [{"opened": m(1), "closed": None, "side": 1}, ]}
    change, states = pn.strategy_states(trips, T0, T0 + WK * 10)
    assert change == [(T0 + WK).value, (T0 + WK * 3).value]
    assert states == [(0, 1), (2, 1), (2, 0)]
    assert pn.week_of(pd.Timestamp("2025-01-12 23:59", tz="UTC")) == pd.Timestamp("2025-01-06", tz="UTC")


# ---- 4. on a rank book's run --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ran():
    from test_rank_legs import ran as legs_run

    return legs_run.__wrapped__()


def test_on_a_rank_books_run_the_model_reconciles_with_the_engine_and_the_sheet_says_so(ran):
    from sleeve_fund.research import basket as bk

    specs, run = ran
    params = dict(specs[0].params)
    window = ((specs[0].prices.index[30] - pd.Timedelta(days=1)).date(),
              (specs[0].prices.index[-1] - pd.Timedelta(days=1)).date())
    r = bk.rank_null(SimpleNamespace(run=run), [s.name for s in specs], specs, params, (window,), 30_000.0, [])
    assert r.change_weeks > 3 and r.strategy_legs > 0
    model, book = r.reconciled
    assert abs(model - book) <= pn.RECONCILE_PP, (model, book)
    words = pn.null_words(r)
    assert "PASS" not in words and "random long/short assignments" in words and "seed 0" in words
    study = SimpleNamespace(pair=r)
    assert bk._pair_lines(study)[0].startswith("**Random long/short assignment:** ")
    assert bk._pair_lines(SimpleNamespace(pair=None)) == []


def test_an_unreconciled_model_is_not_judged():
    r = pn.PairNull(1_000, 0, 5, 0.10, 0.12, 99.0, 99.0, 0.0, 0.0, 4, 8.0, reconciled=(0.10, 0.02))
    assert r.unreconciled and not r.judged and not r.beats and not r.killed
    assert pn.null_words(r).startswith("NOT JUDGED")
