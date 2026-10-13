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


# --- the tables (migration 0019): write-once in the database itself, on SQLite and (CI) Postgres ---------------

import os  # noqa: E402

from sqlalchemy import text  # noqa: E402
from sqlalchemy.exc import IntegrityError  # noqa: E402

from sleeve_fund import schema  # noqa: E402
from sleeve_fund.regime import Member, RegimeRow  # noqa: E402
from sleeve_fund.store import Store, make_engine, metadata  # noqa: E402

V, CFG = "PERPV", "n20-lag1-k5-age60"


def _clean(eng):
    with eng.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS alembic_version"))
    metadata.drop_all(eng)


@pytest.fixture(params=["create_all", "0019"])
def store(request, tmp_path):
    eng = make_engine(os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path}/t.db")
    _clean(eng)
    if request.param == "0019":
        from alembic import command

        with eng.begin() as conn:
            command.upgrade(schema._config(conn), "0014")
        schema.migrate(eng, log=lambda _: None)
    yield Store(engine=eng)
    _clean(eng)


def _list(store, symbol="BTCUSDT", **kw):
    return store.record_instrument(V, symbol, **{**dict(underlying="BTC", quote="USDT", contract_type="perp",
                                                      listed_at=DAY0, source="api"), **kw})


def test_an_instrument_is_noted_once_delisted_once_and_never_rewritten(store):
    assert _list(store) == "new"
    assert _list(store) == "same"
    assert _list(store, delisted_at=DAY0 + timedelta(days=9)) == "delisted"
    assert _list(store, delisted_at=DAY0 + timedelta(days=9)) == "same"
    for change in ({"delisted_at": DAY0 + timedelta(days=10)}, {"delisted_at": None}, {"underlying": "XBT"},
                   {"listed_at": DAY0 + timedelta(days=1)}):
        with pytest.raises(ValueError, match="listed differently"):
            _list(store, **change)
    (row,) = store.instruments(V)
    assert (row["underlying"], row["listed_at"], row["delisted_at"]) == ("BTC", DAY0, DAY0 + timedelta(days=9))


def _bare_edits_refused(store, table, edits):
    for change in edits:
        with pytest.raises(IntegrityError, match="write-once"), store.engine.begin() as c:
            c.execute(text(f"UPDATE {table} SET {change}"))
    with pytest.raises(IntegrityError, match="write-once"), store.engine.begin() as c:
        c.execute(text(f"DELETE FROM {table}"))


def test_a_bare_update_or_delete_of_any_regime_table_is_refused_by_the_database(store):
    _list(store)
    _list(store, symbol="ETHUSDT", underlying="ETH")
    day = DAY0 + timedelta(days=40)
    members = [Member("BTCUSDT", "BTC", 1, D("123.45")), Member("ETHUSDT", "ETH", 2, D("67.8"))]
    assert store.record_universe(V, day, CFG, members, data_through=day - timedelta(days=1)) == "new"
    row = RegimeRow(day, 20, day, day - timedelta(days=1), D("0.41"), "mid", D("0.3"), D("0.6"), 2)
    assert store.record_regime(row, reference_venue=V, config=CFG) == "new"
    _bare_edits_refused(store, "instruments", ["underlying = 'X'", "listed_at = '2023-01-01'"])
    with store.engine.begin() as c:  # delisted_at from NULL, and seen_at, may move
        c.execute(text("UPDATE instruments SET delisted_at = '2024-03-01', seen_at = CURRENT_TIMESTAMP "
                       "WHERE symbol = 'BTCUSDT'"))
    with pytest.raises(IntegrityError, match="write-once"), store.engine.begin() as c:
        c.execute(text("UPDATE instruments SET delisted_at = '2024-03-02' WHERE symbol = 'BTCUSDT'"))
    _bare_edits_refused(store, "universe_snapshots", ["rank = 3", "notional = 1", "computed_at = CURRENT_TIMESTAMP"])
    _bare_edits_refused(store, "regime_series", ["state = 'high'", "er = 0.9", "lo = 0.1"])


def test_a_day_is_written_once_and_a_different_answer_for_it_is_refused(store):
    for s, u in (("BTCUSDT", "BTC"), ("ETHUSDT", "ETH")):
        _list(store, symbol=s, underlying=u)
    day, through = DAY0 + timedelta(days=40), DAY0 + timedelta(days=39)
    members = [Member("BTCUSDT", "BTC", 1, D("123.45")), Member("ETHUSDT", "ETH", 2, D("67.8"))]
    assert store.record_universe(V, day, CFG, members, data_through=through) == "new"
    assert store.record_universe(V, day, CFG, list(reversed(members)), data_through=through) == "same"
    with pytest.raises(ValueError, match="already written differently"):
        store.record_universe(V, day, CFG, members[:1], data_through=through)
    assert store.record_universe(V, day, "another-config", members[:1], data_through=through) == "new"
    assert [r["symbol"] for r in store.universe_on(V, day, CFG)] == ["BTCUSDT", "ETHUSDT"]

    row = RegimeRow(day, 20, day, through, D("0.41"), "mid", D("0.3"), D("0.6"), 2)
    assert store.record_regime(row, reference_venue=V, config=CFG) == "new"
    assert store.record_regime(row, reference_venue=V, config=CFG) == "same"
    with pytest.raises(ValueError, match="already written differently"):
        store.record_regime(RegimeRow(day, 20, day, through, D("0.41"), "high", D("0.3"), D("0.6"), 2),
                            reference_venue=V, config=CFG)
    got = store.regime_on(day, reference_venue=V, lookback=20, config=CFG)
    assert (got["state"], got["er"], got["members"]) == ("mid", D("0.41"), 2)
    assert store.regime_on(day + timedelta(days=1), reference_venue=V, lookback=20, config=CFG) is None


def test_the_database_refuses_a_row_that_reads_its_own_day_a_known_state_without_cuts_or_an_unlisted_member(store):
    _list(store)
    day = DAY0 + timedelta(days=40)
    with pytest.raises(IntegrityError):
        store.record_regime(RegimeRow(day, 20, day, day, D("0.4"), "mid", D("0.3"), D("0.6"), 2),
                            reference_venue=V, config=CFG)
    with pytest.raises(IntegrityError):
        store.record_regime(RegimeRow(day, 20, day, day - timedelta(days=1), D("0.4"), "mid", None, None, 2),
                            reference_venue=V, config=CFG)
    with pytest.raises(IntegrityError):
        store.record_regime(RegimeRow(day, 25, day, day - timedelta(days=1), None, "unknown", None, None, 0),
                            reference_venue=V, config=CFG)
    if store.engine.dialect.name == "sqlite":
        return  # SQLite checks foreign keys only once DA-5 turns them on; Postgres refuses the member below
    with pytest.raises(IntegrityError):  # not in instruments
        store.record_universe(V, day, CFG, [Member("DOGEUSDT", "DOGE", 1, D(1))], data_through=day - timedelta(days=1))


def test_a_pegged_underlying_never_enters_and_the_settings_are_the_advisors():
    notional = {s: {DAY0 + timedelta(days=i): D(v) for i in range(120)} for s, v in (("USDCUSDT", 999), ("BTCUSDT", 1))}
    listings = [Listing("USDCUSDT", "USDC", DAY0), Listing("BTCUSDT", "BTC", DAY0)]
    got = universe(DAY0 + timedelta(days=100), listings, notional, n=2, keep_within=2, min_age_days=0, lag_days=1)
    assert [m.symbol for m in got] == ["BTCUSDT"]
    assert regime.SETTINGS == dict(n=20, keep_within=25, min_age_days=60, lag_days=1, min_members=10, min_history=180)


# --- loading the venue's listing --------------------------------------------------------------------------------

from sleeve_fund import venues  # noqa: E402
from sleeve_fund.regime_load import sync_instruments  # noqa: E402

MS = lambda d: int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)  # noqa: E731
from datetime import datetime, timezone  # noqa: E402


def _info(*symbols):
    return {"symbols": [dict(symbol=s, baseAsset=b, quoteAsset="USDT", contractType=ct, status=st,
                             onboardDate=MS(on), deliveryDate=dd) for s, b, ct, st, on, dd in symbols]}


def test_the_listing_books_live_and_delisted_perps_once_and_refuses_a_reused_symbol(store):
    gone = DAY0 + timedelta(days=200)
    info = _info(("BTCUSDT", "BTC", "PERPETUAL", "TRADING", DAY0, 4133404800000),
                 ("OLDUSDT", "OLD", "PERPETUAL", "SETTLING", DAY0, MS(gone)),
                 ("BTCUSDT_251226", "BTC", "CURRENT_QUARTER", "TRADING", DAY0, MS(gone)))
    rows = venues.binance_listings(get_json=lambda url: info)
    assert [(r["symbol"], r["listed_at"], r["delisted_at"]) for r in rows] == [("BTCUSDT", DAY0, None),
                                                                               ("OLDUSDT", DAY0, gone)]
    assert sync_instruments(store, V, rows) == {"new": 2, "same": 0, "delisted": 0, "refused": []}
    assert sync_instruments(store, V, rows) == {"new": 0, "same": 2, "delisted": 0, "refused": []}
    reused = _info(("OLDUSDT", "OLD", "PERPETUAL", "TRADING", DAY0 + timedelta(days=300), 4133404800000))
    got = sync_instruments(store, V, venues.binance_listings(get_json=lambda url: reused))
    assert got["refused"] and got["refused"][0][0] == "OLDUSDT"
    assert {r["symbol"]: r["delisted_at"] for r in store.instruments(V)} == {"BTCUSDT": None, "OLDUSDT": gone}


def _kline(d, close, qv):
    return [MS(d), "1", "1", "1", str(close), "5", MS(d) + 86_399_999, str(qv)]


def test_daily_bars_are_the_close_and_quote_volume_per_closed_day_from_a_start():
    pages, urls = [[_kline(DAY0 + timedelta(days=i), f"{10 + i}.25", f"{100 + i}.5") for i in range(3)]], []
    got = venues.binance_daily_bars("BTCUSDT", get_json=lambda url: urls.append(url) or pages.pop(0),
                                    start=DAY0)
    assert got == {DAY0: (D("10.25"), D("100.5")), DAY0 + timedelta(days=1): (D("11.25"), D("101.5"))}
    assert f"startTime={MS(DAY0)}" in urls[0] and "symbol=BTCUSDT" in urls[0]


S3 = "http://s3.amazonaws.com/doc/2006-03-01/"


def _index(items, tag, truncated=False):
    body = "".join(f"<CommonPrefixes><Prefix>{x}</Prefix></CommonPrefixes>" if tag == "Prefix"
                   else f"<Contents><Key>{x}</Key></Contents>" for x in items)
    return (f'<ListBucketResult xmlns="{S3}"><IsTruncated>{"true" if truncated else "false"}</IsTruncated>'
            f"{body}</ListBucketResult>").encode()


def _zip(rows, header):
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        lines = (["open_time,open,high,low,close,volume,close_time,quote_volume"] if header else [])
        z.writestr("k.csv", "\n".join(lines + [",".join(map(str, r)) for r in rows]) + "\n")
    return buf.getvalue()


def test_the_archive_lists_every_perpetual_across_pages_and_reads_its_monthly_zips_with_or_without_a_header():
    base = "data/futures/um/monthly/klines/"
    pages = [_index([f"{base}AAAUSDT/", f"{base}BTCUSDT_240329/"], "Prefix", truncated=True),
             _index([f"{base}GONEUSDT/", f"{base}ODDEUR/"], "Prefix")]
    urls = []
    syms = venues.binance_archive_symbols(get_bytes=lambda url: urls.append(url) or pages.pop(0))
    assert syms == ["AAAUSDT", "GONEUSDT"]  # no quarterly contract, no unknown quote
    assert "marker=" + "data%2Ffutures%2Fum%2Fmonthly%2Fklines%2FBTCUSDT_240329%2F" in urls[1]

    jan, feb = DAY0, DAY0 + timedelta(days=31)
    files = {"index": _index([f"{base}GONEUSDT/1d/a-2024-01.zip", f"{base}GONEUSDT/1d/a-2024-01.zip.CHECKSUM",
                              f"{base}GONEUSDT/1d/a-2024-02.zip"], "Key"),
             "a-2024-01.zip": _zip([_kline(jan, "2.5", "7")], header=False),
             "a-2024-02.zip": _zip([_kline(feb, "3", "8.125")], header=True)}
    got = venues.binance_archive_daily_bars(
        "GONEUSDT", get_bytes=lambda url: files["index" if "?" in url else url.rsplit("/", 1)[1]])
    assert got == {jan: (D("2.5"), D("7")), feb: (D("3"), D("8.125"))}
    assert venues.binance_archive_listing("GONEUSDT", got) == {
        "symbol": "GONEUSDT", "underlying": "GONE", "quote": "USDT", "contract_type": "perp", "listed_at": jan,
        "delisted_at": feb, "source": "archive"}
    assert venues.binance_archive_listing("GONEUSDT", {}) is None


def test_a_daily_bar_is_booked_once_and_a_revised_one_is_refused(store):
    _list(store)
    bars = {DAY0: (D("10"), D("100")), DAY0 + timedelta(days=1): (D("11.5"), D("0"))}
    assert store.record_daily_bars(V, "BTCUSDT", bars, source="api") == {"new": 2, "same": 0}
    more = {**bars, DAY0 + timedelta(days=2): (D("12"), D("5"))}
    assert store.record_daily_bars(V, "BTCUSDT", more, source="api") == {"new": 1, "same": 2}
    with pytest.raises(ValueError, match="already held with another close"):
        store.record_daily_bars(V, "BTCUSDT", {DAY0: (D("10.01"), D("100")), DAY0 + timedelta(days=3): (D(1), D(1))},
                                source="api")
    assert store.last_daily_bar(V, "BTCUSDT") == DAY0 + timedelta(days=2)  # the refused call wrote nothing
    closes, notionals = store.daily_bars(V)
    assert closes["BTCUSDT"][DAY0 + timedelta(days=1)] == D("11.5") and notionals["BTCUSDT"][DAY0] == D("100")
    _bare_edits_refused(store, "daily_bars", ["close = 1", "notional = 2", "source = 'archive'"])
    with pytest.raises(IntegrityError):
        store.record_daily_bars(V, "BTCUSDT", {DAY0 + timedelta(days=9): (D(0), D(1))}, source="api")


# --- the backfill: listing, bars, then every day's universe and regime, written once ----------------------------

from sleeve_fund import regime_load  # noqa: E402

DAYS = 330


def _source(n_live=24, gone_day=200, days=DAYS):
    """A venue of n_live listed contracts plus one delisted (still in the listing) and one only in the archive, with
    a seeded random walk each, so the ER moves."""
    import random

    rnd = random.Random(7)
    def walk(seed_days, start=0):
        out, px = {}, D(100)
        for i in range(start, seed_days):
            px = max(D("0.01"), px * (1 + D(rnd.randint(-40, 45)) / 1000)).quantize(D("0.0001"))
            out[DAY0 + timedelta(days=i)] = (px, D(1000 + rnd.randint(0, 9000)))
        return out

    live = {f"C{i:02d}USDT": walk(days) for i in range(n_live)}
    gone = walk(gone_day + 1)
    old = walk(150)
    listings = [dict(symbol=s, underlying=s[:-4], quote="USDT", contract_type="perp", listed_at=DAY0,
                     delisted_at=None, source="api") for s in live]
    listings.append(dict(symbol="GONEUSDT", underlying="GONE", quote="USDT", contract_type="perp", listed_at=DAY0,
                         delisted_at=DAY0 + timedelta(days=gone_day), source="api"))
    api = {**live, "GONEUSDT": gone}
    calls = []

    def api_bars(sym, start=None):
        calls.append((sym, start))
        return {d: v for d, v in api[sym].items() if start is None or d >= start}

    src = {"listings": lambda: listings, "archive_symbols": lambda: ["C00USDT", "OLDUSDT"],
           "api_bars": api_bars, "archive_bars": lambda sym: {"OLDUSDT": old}[sym],
           "archive_listing": venues.binance_archive_listing}
    return src, api, calls


def test_the_backfill_loads_listed_delisted_and_archive_only_contracts_and_a_rerun_fetches_only_new_days(store):
    src, api, calls = _source()
    now = datetime(2024, 12, 1, tzinfo=timezone.utc)
    got = regime_load.load_bars(store, V, src, now, log=lambda _: None)
    assert got["refused"] == [] and got["archive_only"] == 1
    held = {r["symbol"]: r for r in store.instruments(V)}
    assert held["OLDUSDT"]["source"] == "archive" and held["OLDUSDT"]["delisted_at"] == DAY0 + timedelta(days=149)
    assert store.last_daily_bar(V, "GONEUSDT") == DAY0 + timedelta(days=200)
    calls.clear()
    assert regime_load.load_bars(store, V, src, now, log=lambda _: None)["bars_new"] == 0
    assert ("GONEUSDT", DAY0 + timedelta(days=201)) not in calls and all(s != "GONEUSDT" for s, _ in calls)
    assert ("C00USDT", DAY0 + timedelta(days=DAYS)) in calls  # from the day after the newest held


def test_the_build_writes_every_day_once_is_reproducible_and_a_later_night_only_adds_rows(store):
    src, _, _ = _source()
    regime_load.load_bars(store, V, src, log=lambda _: None)
    cfg = regime_load.config_name()
    assert cfg == "n20-k25-age60-lag1-min10-h180"
    end = DAY0 + timedelta(days=300)
    first = regime_load.build(store, V, end, lookbacks=(20,), log=lambda _: None)
    assert first["regime"]["new"] == 301 and "same" not in first["regime"]
    again = regime_load.build(store, V, end + timedelta(days=5), lookbacks=(20,), log=lambda _: None)
    assert again["regime"] == {"same": 301, "new": 5}  # the earlier days come out identical with more bars held
    late = store.regime_on(end, reference_venue=V, lookback=20, config=cfg)
    assert late["data_through"] == end - timedelta(days=1) and late["members"] >= 10
    assert late["state"] in ("low", "mid", "high")  # 180 valid days of history by then
    day = DAY0 + timedelta(days=100)
    members = store.universe_on(V, day, cfg)
    assert len(members) == 20 and "OLDUSDT" in {m["symbol"] for m in members}  # the archive-only one, in its time
    assert "OLDUSDT" not in {m["symbol"] for m in store.universe_on(V, DAY0 + timedelta(days=200), cfg)}


def test_the_cli_loads_then_builds_and_fails_on_a_refusal(store, monkeypatch, capfd):
    with pytest.raises(ValueError, match="no regime data loaders"):
        venues.regime_source("elsewhere")
    src, api, _ = _source(days=120)
    monkeypatch.setattr("sleeve_fund.store.Store", lambda: store)
    monkeypatch.setattr("sleeve_fund.venues.regime_source", lambda name: src)
    assert regime_load.main(["backfill", "--venue", V, "--end", str(DAY0 + timedelta(days=100))]) == 0
    assert store.regime_on(DAY0 + timedelta(days=100), reference_venue=V, lookback=30,
                           config=regime_load.config_name()) is not None
    api["C01USDT"] = {**api["C01USDT"], DAY0: (D("1"), D("1"))}  # the venue revised a printed day
    src["api_bars"] = lambda sym, start=None: api[sym]
    assert regime_load.main(["backfill", "--venue", V, "--end", str(DAY0 + timedelta(days=100))]) == 1
    assert "refused C01USDT" in capfd.readouterr().out
