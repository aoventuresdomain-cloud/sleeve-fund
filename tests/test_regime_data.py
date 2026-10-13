"""P2-5 regime, the data side (sleeve_fund.regime; DA design data-architect/p2-5-data-design.md): the universe and the
ER series are point in time. The leak test is the spec's Done-when: rows built with every bar equal rows built with
every bar after a cut-off removed, for every day the cut-off could have seen."""

from datetime import date, timedelta
from decimal import Decimal as D

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sleeve_fund import regime
from sleeve_fund.regime import Listing, cuts, efficiency_ratio, series, state_of, universe

DAY0 = date(2024, 1, 1)


def _flat(symbols, days, value=D(100)):
    return {s: {DAY0 + timedelta(days=i): value for i in range(days)} for s in symbols}


def test_er_is_net_move_over_path_and_zero_without_movement():
    assert efficiency_ratio([D(1), D(2), D(3)]) == 1
    assert efficiency_ratio([D(1), D(2), D(1)]) == 0
    assert efficiency_ratio([D(5), D(5), D(5)]) == 0
    assert efficiency_ratio([D(10), D(12), D(11), D(14)]) == D(4) / D(6)


def test_the_universe_ranks_by_30_day_notional_one_contract_per_underlying_listed_long_enough_and_not_delisted():
    day = DAY0 + timedelta(days=100)
    notional = {"AUSDT": {DAY0 + timedelta(days=i): D(10) for i in range(120)},
                "AUSDC": {DAY0 + timedelta(days=i): D(20) for i in range(120)},  # same underlying, more notional
                "BUSDT": {DAY0 + timedelta(days=i): D(15) for i in range(120)},
                "CUSDT": {DAY0 + timedelta(days=i): D(99) for i in range(120)},  # too young
                "DUSDT": {DAY0 + timedelta(days=i): D(50) for i in range(120)},  # delisted before data_through
                "EUSDT": {DAY0 + timedelta(days=i): D(30) for i in range(80, 120)}}  # under 30 days of notional
    listings = [Listing("AUSDT", "A", DAY0), Listing("AUSDC", "A", DAY0), Listing("BUSDT", "B", DAY0),
                Listing("CUSDT", "C", day - timedelta(days=20)), Listing("DUSDT", "D", DAY0, DAY0 + timedelta(days=50)),
                Listing("EUSDT", "E", DAY0)]
    got = universe(day, listings, notional, n=5, keep_within=5, min_age_days=30, lag_days=1)
    assert [(m.symbol, m.rank, m.notional) for m in got] == [("AUSDC", 1, D(600)), ("BUSDT", 2, D(450))]


def test_a_member_stays_while_within_keep_within_and_the_lag_is_at_least_a_day():
    notional = {"A": {DAY0 + timedelta(days=i): D(10) for i in range(60)},
                "B": {DAY0 + timedelta(days=i): D(11) for i in range(60)},
                "C": {DAY0 + timedelta(days=i): D(12) for i in range(60)}}
    listings = [Listing(s, s, DAY0) for s in "ABC"]
    day = DAY0 + timedelta(days=45)
    kw = dict(n=1, keep_within=3, min_age_days=0, lag_days=1)
    assert [m.symbol for m in universe(day, listings, notional, **kw)] == ["C"]
    assert [m.symbol for m in universe(day, listings, notional, previous=frozenset({"A"}), **kw)] == ["A"]
    assert [m.symbol for m in universe(day, listings, notional, previous=frozenset({"A"}),
                                       **{**kw, "keep_within": 2})] == ["C"]
    with pytest.raises(ValueError, match="at least one day"):
        universe(day, listings, notional, **{**kw, "lag_days": 0})


@pytest.mark.parametrize("er, previous, want", [
    (D("0.20"), None, "low"), (D("0.50"), None, "mid"), (D("0.80"), None, "high"),
    (D("0.32"), "low", "low"),  # past lo but not by the band
    (D("0.34"), "low", "mid"), (D("0.28"), "mid", "mid"), (D("0.26"), "mid", "low"),
    (D("0.68"), "mid", "mid"), (D("0.70"), "mid", "high"), (D("0.62"), "high", "high"), (D("0.60"), "high", "mid"),
    (D("0.80"), "low", "high"), (D("0.10"), "high", "low"), (D("0.50"), "unknown", "mid"),
])
def test_the_state_moves_off_the_last_one_only_a_band_past_the_cut(er, previous, want):
    assert state_of(er, D("0.30"), D("0.65"), previous) == want


def test_cuts_are_the_terciles():
    assert cuts([D(i) for i in range(7)]) == (D(2), D(4))


def _world(rnd_closes, n_symbols, days):
    symbols = [f"S{i}" for i in range(n_symbols)]
    closes, notional = {}, {}
    for k, s in enumerate(symbols):
        px, c, v = D(100), {}, {}
        for i in range(days):
            px = max(D(1), px + D(rnd_closes[(k * days + i) % len(rnd_closes)]))
            c[DAY0 + timedelta(days=i)] = px
            v[DAY0 + timedelta(days=i)] = px * D(1000 + 37 * ((k * 7 + i) % 11))
        closes[s], notional[s] = c, v
    return symbols, closes, notional


KW = dict(lookback=20, n=3, keep_within=4, min_age_days=10, lag_days=1, min_members=2, min_history=10)


@settings(max_examples=25, deadline=None)
@given(st.lists(st.integers(-5, 5), min_size=10, max_size=60), st.integers(70, 110), st.integers(0, 3))
def test_no_row_reads_anything_after_its_data_through(moves, cut_day, delisted):
    days = 120
    symbols, closes, notional = _world(moves, 5, days)
    listings = [Listing(s, s, DAY0 + timedelta(days=3 * i),
                        DAY0 + timedelta(days=90) if i == delisted else None) for i, s in enumerate(symbols)]
    span = [DAY0 + timedelta(days=i) for i in range(40, days)]
    full, full_u = series(span, listings, notional, closes, **KW)
    cut = DAY0 + timedelta(days=cut_day)
    # the world as it was known at the cut-off: no bar after it, and a delisting after it not yet known
    known = lambda d: {s: {k: v for k, v in x.items() if k <= cut} for s, x in d.items()}  # noqa: E731
    then = [Listing(x.symbol, x.underlying, x.listed_at, x.delisted_at if x.delisted_at and x.delisted_at <= cut
                    else None) for x in listings]
    seen = [d for d in span if d - timedelta(days=KW["lag_days"]) <= cut]
    part, part_u = series(seen, then, known(notional), known(closes), **KW)
    assert part == full[:len(seen)]
    assert all(part_u[d] == full_u[d] for d in seen)
    assert all(r.data_through < r.date for r in full)
    assert any(r.state in ("low", "mid", "high") for r in full)


def test_a_day_with_too_few_full_windows_is_unknown_and_does_not_move_the_state():
    symbols = ["A", "B"]
    closes = {s: {DAY0 + timedelta(days=i): D(100 + (i % 3)) for i in range(60)} for s in symbols}
    del closes["A"][DAY0 + timedelta(days=50)]  # one missing close: A leaves every window holding it
    notional = _flat(symbols, 60, D(10))
    rows, _ = series([DAY0 + timedelta(days=i) for i in range(35, 60)], [Listing(s, s, DAY0) for s in symbols],
                     notional, closes, **{**KW, "min_history": 1})
    by_day = {r.date: r for r in rows}
    gap = by_day[DAY0 + timedelta(days=55)]
    assert gap.members == 1 and gap.state == "unknown" and gap.lo is None
    assert regime.STATES[-1] == "unknown"
