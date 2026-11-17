"""Family C, step (b) (PM 16 Nov 21:06 UTC, backtest only): a basket's rank legs. Rank 1 goes long, the last rank short,
the middle flat, re-decided at each Monday 00:00 UTC close, with no top-up while a leg is held. Done when:

1. a basket whose definition ranks its assets (rank_book) sizes every entry off the fund, guards on or off, at a fixed
   share of at most 0.5 a leg, and nothing else can switch that on; a size scale is refused;
2. its legs follow the hand ranking: each Monday's new strongest opens long, new weakest short, the middle holds none,
   a held leg is never added to;
3. equal money at entry: each entry is its share of the fund it was sized off to the cent, those opened on one close
   sized off the same fund and together at most 1x of it; the check fails on anything else and refuses the sheet."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from test_relative_strength_rank import ORDER, _by_hand, _legs

from sleeve_fund.research import basket as bk
from sleeve_fund.research.portfolio_run import LegSpec, run_portfolio
from sleeve_fund.risk import PortfolioProfile
from sleeve_fund.strategies.definitions import is_weekly_decision

MONDAY_CLOSE = [{"left": {"time": "weekday"}, "op": "<=", "right": 0}, {"left": {"time": "hour"}, "op": "<=", "right": 0}]
BOOK = {"version": 1, "name": "rs-ls", "title": "rs", "reason": "test",
        "blocks": {"rs": {"kind": "relative_strength_rank"}},
        # reverse: a flip (strongest straight to weakest, or back) opens its new leg on the close the old one ends
        "long": {"entry": {"all": [{"left": "rs.rank", "op": "<=", "right": 1}, *MONDAY_CLOSE]},
                 "exit": {"left": "rs.rank", "op": ">=", "right": 2}, "reverse": True},
        "short": {"entry": {"all": [{"left": "rs.rank", "op": ">=", "right": 3}, *MONDAY_CLOSE]},
                  "exit": {"left": "rs.rank", "op": "<=", "right": 2}, "reverse": True}}
PARAMS = {"definition": BOOK, "market": "perp", "allow_short": True, "position_cap_pct": 0.5}
LOOSE = PortfolioProfile(gross=100, net_instrument=100, margin=100, open_risk=100, drawdown=0.99, daily_loss=0.99)


# ---- 1. sizing --------------------------------------------------------------------------------------------------

def test_only_a_ranking_definition_makes_a_rank_book_and_it_needs_a_share_of_at_most_a_half():
    assert bk.rank_book(PARAMS)
    assert not bk.rank_book({**PARAMS, "definition": {**BOOK, "blocks": {"rs": {"kind": "rsi"}}}})
    assert not bk.rank_book({"position_cap_pct": 0.5}) and not bk.rank_book(None)
    for share in (None, 0.51, 1.0, 0):
        with pytest.raises(ValueError, match="at most 0.5"):
            bk.rank_book({**PARAMS, "position_cap_pct": share} if share is not None else
                         {k: v for k, v in PARAMS.items() if k != "position_cap_pct"})
    assert bk.own_sized(bk.UNGUARDED, None) and bk.own_sized(bk.PORTFOLIO, 0.5)
    assert not bk.own_sized(bk.UNGUARDED, None, PARAMS)  # guards off, still sized off the fund
    assert bk.own_sized(bk.UNGUARDED, None, {"position_cap_pct": 0.5})


def test_a_rank_basket_runs_fund_sized_with_the_guards_off_and_refuses_a_size_scale(monkeypatch):
    seen = {}

    def fake(specs, profile, unallocated=0.0, fund_sized=False, **kw):
        seen["fund_sized"] = fund_sized
        raise RuntimeError("stop")

    monkeypatch.setattr(bk, "run_portfolio", fake)
    monkeypatch.setattr(bk, "_cut", lambda a, *x: SimpleNamespace(instrument=a.instrument, prices=None,
                                                                      exec_prices=None, warmup_prices=None))
    monkeypatch.setattr(bk, "check_assets", lambda d: None)
    monkeypatch.setattr(bk, "leg_names", lambda name, insts: [f"l{i}" for i in range(len(insts))])
    assets = [SimpleNamespace(instrument=i, exec_minutes=60, bar_minutes=1440) for i in range(3)]
    for params, want in ((PARAMS, True), ({"definition": {**BOOK, "blocks": {}}}, False)):
        basket = SimpleNamespace(name="b", strategy="rules", params=params, risk_profile=None)
        with pytest.raises(RuntimeError):
            bk._run(basket, assets, None, None, None, 30_000.0, bk.UNGUARDED, 0)
        assert seen["fund_sized"] is want
    with pytest.raises(ValueError, match="takes no size scale"):
        bk._run(SimpleNamespace(name="b", strategy="rules", params=PARAMS, risk_profile=None), assets, None, None, None, 30_000.0,
                bk.PORTFOLIO, 0, scale=0.5)


# ---- 2. the legs ------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ran():
    specs = [LegSpec(**{**s.__dict__, "params": dict(PARAMS)}) for s in _legs(days=220)]
    return specs, run_portfolio(specs, LOOSE, fund_sized=True)


def _entries(run) -> list:
    out = []
    for name, leg in run.legs.items():
        for oid, d in leg.decisions.items():
            if d.get("intent") == "entry":
                out.append((bk.entry_close(oid, d["signal"]), name, d["signal"]["side"]))
    return sorted(out)


def test_the_strongest_opens_long_the_weakest_short_and_the_middle_holds_nothing(ran):
    specs, run = ran
    closes = {s.name: pd.concat([s.warmup_prices, s.prices])["close"] for s in specs}
    mondays = [t for t in specs[0].prices.index if is_weekly_decision(t.value)]
    want, flips, prev = [], 0, None
    for m in mondays:
        sc = _by_hand(closes, m)
        order = sorted(ORDER, key=lambda a: -sc[a])
        if prev is None or prev[0] != order[0]:
            want.append((m, order[0], "long"))
        if prev is None or prev[-1] != order[-1]:
            want.append((m, order[-1], "short"))
        flips += prev is not None and (prev[0] == order[-1] or prev[-1] == order[0])
        prev = order
    assert flips >= 1  # the data holds an asset going straight from strongest to weakest, or back
    got = _entries(run)
    assert len(want) >= 8 and [(t, n, s.lower()) for t, n, s in got] == sorted(want)
    for m in mondays:  # after each Monday's decisions, exactly one long and one short are held
        held = {}
        for t, n, s in got:
            if t <= m:
                held[n] = s.lower()
        sc = _by_hand(closes, m)
        order = sorted(ORDER, key=lambda a: -sc[a])
        assert held.get(order[0]) == "long" and held.get(order[-1]) == "short"
    middle_exits = [d for leg in run.legs.values() for d in leg.decisions.values() if d.get("intent") == "rebalance"]
    assert middle_exits == []  # never topped up


def test_after_each_monday_close_only_the_strongest_and_weakest_hold_a_position(ran):
    specs, run = ran
    closes = {s.name: pd.concat([s.warmup_prices, s.prices])["close"] for s in specs}
    mondays = [t for t in specs[0].prices.index if is_weekly_decision(t.value)]
    for s in specs:
        f = run.legs[s.name].fills
        when = [bk.entry_close(oid, {}) for oid in f.index]  # every fill, by the close its order was sent on
        signed = [(1 if str(side).upper().endswith("BUY") else -1) * float(q) for side, q in zip(f["side"], f["filled_qty"])]
        for m in mondays:
            pos = sum(q for t, q in zip(when, signed) if t <= m)
            sc = _by_hand(closes, m)
            rank = sorted(ORDER, key=lambda a: -sc[a]).index(s.name) + 1
            assert (pos > 0, pos < 0) == (rank == 1, rank == 3), (s.name, m, rank, pos)


# ---- 3. equal money ---------------------------------------------------------------------------------------------

def test_every_entry_is_half_the_fund_it_was_sized_off_and_a_pair_opened_together_is_equal(ran):
    _, run = ran
    e = bk.equal_money(run, 0.5)
    assert e.passed and e.entries == len(_entries(run))
    together = {}
    for leg in run.legs.values():
        for oid, d in leg.decisions.items():
            if d.get("intent") == "entry":
                sig = d["signal"]
                assert abs(sig["budget"] - 0.5 * sig["sized_off_fund"]) <= bk.CENT_ROUNDING
                together.setdefault(bk.entry_close(oid, sig), []).append(sig)
    pairs = [v for v in together.values() if len(v) == 2]
    assert pairs and all(a["budget"] == b["budget"] and a["side"] != b["side"] for a, b in pairs)


def _tamper(run, field, value, which=0):
    import copy

    run = copy.deepcopy(run)
    entries = [d for leg in run.legs.values() for d in leg.decisions.values() if d.get("intent") == "entry"]
    entries[which]["signal"][field] = value
    return run


def test_equal_money_fails_on_an_entry_off_its_share_unequal_funds_or_over_1x(ran):
    _, run = ran
    assert bk.equal_money(_tamper(run, "budget", 1.0), 0.5).off_share == 1
    assert bk.equal_money(_tamper(run, "sized_by", "free cash"), 0.5).off_share == 1
    assert bk.equal_money(run, 0.4).off_share == bk.equal_money(run, 0.4).entries
    together = {}
    for leg in run.legs.values():
        for oid, d in leg.decisions.items():
            if d.get("intent") == "entry":
                together.setdefault(bk.entry_close(oid, d["signal"]), []).append(d)
    a, b = next(v for v in together.values() if len(v) == 2)
    old = b["signal"]["sized_off_fund"]
    b["signal"]["sized_off_fund"] = old + 100
    e = bk.equal_money(run, 0.5)
    assert e.unequal == 1 and not e.passed
    b["signal"]["sized_off_fund"] = old
    a["signal"]["budget"], b["signal"]["budget"] = a["signal"]["sized_off_fund"] * 0.6, b["signal"]["budget"] * 1.2
    e = bk.equal_money(run, 0.5)
    assert e.over_gross == 1 and not e.passed
    assert not bk.EqualMoney().passed  # nothing opened: never a pass


def test_the_sheet_carries_the_check_and_says_held_legs_drift():
    from datetime import date

    study = SimpleNamespace(equal=bk.EqualMoney(entries=12), basket=SimpleNamespace(params=PARAMS),
                            windows=((date(2024, 3, 1), date(2024, 6, 30)),), unranked=[], pair=None)
    line = bk._equal_lines(study)[0]
    assert line.startswith("**Equal money at entry:** PASS: 12 entries, each at 0.5x the fund")
    assert "drifts with its price" in line
    assert bk._equal_lines(SimpleNamespace(equal=None)) == []


def test_a_run_sized_off_each_legs_own_account_fails_the_check():
    specs = [LegSpec(**{**s.__dict__, "params": dict(PARAMS)}) for s in _legs(days=120)]
    run = run_portfolio(specs, LOOSE, fund_sized=False)
    e = bk.equal_money(run, 0.5)
    assert e.entries > 0 and e.off_share == e.entries  # nothing says what fund it was sized off: never a pass


def test_a_failed_check_refuses_the_sheet_before_any_try_is_counted(ran, monkeypatch):
    specs, run = ran
    monkeypatch.setattr(bk, "_run", lambda *a, **k: (run, [s.name for s in specs], specs))
    monkeypatch.setattr(bk, "_measure", lambda *a, **k: SimpleNamespace(net=0.0, run=run))
    monkeypatch.setattr(bk, "equal_money", lambda r, share: bk.EqualMoney(entries=4, unequal=1))
    basket = SimpleNamespace(name="b", strategy="rules", params=PARAMS, assets=(), names=())

    class NoTry:
        def __getattr__(self, name):
            raise AssertionError("no try may be counted")

    with pytest.raises(ValueError, match=r"refused: equal money at entry failed \(4 entries: 0 not at their share of "
                                         r"the fund, 1 closes unequal, 0 closes over 1x gross\), so no sheet and no try"):
        bk.basket_study(basket, ((pd.Timestamp("2024-03-01").date(), pd.Timestamp("2024-06-01").date()),),
                        pd.Timestamp("2024-07-01").date(), register=NoTry(), dataset="d", profile=bk.UNGUARDED)


def test_an_entrys_close_is_its_bars_or_else_its_order_ids_time_to_the_minute():
    assert bk.entry_close("O-20240219-000000-001-001-4", {"bar": {"close_ts": "2024-02-19T00:00:00+00:00"}}) == \
        pd.Timestamp("2024-02-19", tz="UTC")
    assert bk.entry_close("O-20240219-000003-001-001-4", {}) == pd.Timestamp("2024-02-19", tz="UTC")
    assert bk.entry_close("O-20240219-040000-001-002-9", {}) == pd.Timestamp("2024-02-19 04:00", tz="UTC")


def test_the_sheet_lists_every_unranked_monday_and_one_in_the_test_days_leaves_check_7_open():
    from datetime import date

    study = SimpleNamespace(windows=((date(2024, 3, 1), date(2024, 6, 30)),),
                            unranked=["2024-01-08", "2024-01-15", "2024-04-08", "2024-07-01"])
    line = bk._unranked_lines(study)[0]
    assert line == ("**Unranked Mondays:** 2 in the test days, legs held through each: 2024-04-08, 2024-07-01 "
                    "(2 in the warm-up). Accuracy check 7: not verified until each is explained.")
    study.unranked = ["2024-01-08"]
    assert bk._unranked_lines(study)[0].startswith("**Unranked Mondays:** none in the test days (1 in the warm-up")
    study = SimpleNamespace(equal=bk.EqualMoney(entries=2), basket=SimpleNamespace(params=PARAMS), windows=study.windows,
                            unranked=[], pair=None)
    assert bk._equal_lines(study)[2].startswith("**Unranked Mondays:** none")


def test_the_runs_unranked_mondays_are_its_warm_ups_first_four(ran):
    _, run = ran
    assert bk.unranked_dates(run) == ["2024-01-08", "2024-01-15", "2024-01-22", "2024-01-29"]


def test_a_rank_baskets_study_carries_its_runs_unranked_mondays_to_the_sheet(tmp_path):
    # CR #354: the wiring from the run to the sheet, so check 7's condition can't silently read "none"
    from test_b38_basket_study import ASSETS, CAPITAL, RESEARCH_END, WINDOWS, _register

    basket = bk.Basket("rs", "rules", dict(PARAMS), ASSETS, "2026-11-16")
    study = bk.basket_study(basket, WINDOWS, RESEARCH_END, register=_register(tmp_path), dataset="syn",
                            capital=CAPITAL, profile=bk.UNGUARDED, warm_days=40, ladder=False)
    assert study.unranked and study.unranked == bk.unranked_dates(study.base.run)
    assert study.equal is not None and study.equal.passed
    assert any(line.startswith("**Unranked Mondays:** ") for line in bk.render(study).splitlines())
