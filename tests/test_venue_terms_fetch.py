"""FEE-LIVE storage (DA, head 3; data-architect/fee-live-storage.md): every fetch of a venue's fee rates or margin
tiers is one append-only row. The engine reads the newest good row per (venue, kind, instrument) as last-known-good;
a failed or out-of-bounds fetch is kept, with its reason, but never becomes last-known-good. CI also runs these
against Postgres via TEST_DATABASE_URL."""

import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from sleeve_fund import schema
from sleeve_fund.store import Store, make_engine, metadata

T = datetime(2026, 10, 15, 11, 0, tzinfo=timezone.utc)
BOUNDS = {"maker": (D("0"), D("0.0002")), "taker": (D("0.0002"), D("0.00055"))}  # [lowest published, VIP0]
FEE = {"symbol": "BTCUSDT", "maker": "0.0002", "taker": "0.00055"}
# The first three tiers of the venue's BTCUSDT table (vs1-tiers.md), the third left as the last: the reader opens it
TIERS = {"symbol": "BTCUSDT", "tiers": [
    {"ceiling": "300000", "mmr": "0.0033", "imr": "0.0066", "max_leverage": "150", "deduction": "0"},
    {"ceiling": "2000000", "mmr": "0.005", "imr": "0.01", "max_leverage": "100", "deduction": "510"},
    {"ceiling": "2600000", "mmr": "0.0056", "imr": "0.0111", "max_leverage": "90", "deduction": "1710"},
]}


def _clean(eng):
    with eng.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS alembic_version"))
    metadata.drop_all(eng)


@pytest.fixture
def engine(tmp_path):
    eng = make_engine(os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path}/t.db")
    _clean(eng)
    yield eng
    _clean(eng)


@pytest.fixture
def store(engine):
    s = Store(engine=engine)
    s.create_account("paper-main", "paper")
    return s


def _fee(store, at, **kw):
    args = dict(source="fee-rate", ok=True, payload=FEE, account="paper-main", at=at, bounds=BOUNDS)
    args.update(kw)
    return store.record_venue_fetch("BINANCE", "fee", "BTC/USDT", **args)


def test_the_newest_good_fetch_is_last_known_good_and_a_failed_one_never_replaces_it(store):
    first = _fee(store, T)
    newer = _fee(store, T + timedelta(hours=1), payload={**FEE, "taker": "0.0005"})
    failed = _fee(store, T + timedelta(hours=2), ok=False, reason="timeout after 10 s", payload=None)
    good = store.latest_good_venue_fetch("BINANCE", "fee", "BTC/USDT")
    assert good["id"] == newer != first and good["payload"]["taker"] == "0.0005" and good["ok"] == 1
    assert good["account"] == "paper-main" and good["source"] == "fee-rate"
    last = store.latest_venue_fetch("BINANCE", "fee", "BTC/USDT")
    assert last["id"] == failed and last["ok"] == 0 and last["reason"] == "timeout after 10 s"
    assert store.latest_good_venue_fetch("BINANCE", "fee", "ETH/USDT") is None  # never fetched
    assert store.latest_good_venue_fetch("BINANCE", "tiers", "BTC/USDT") is None  # another kind


def test_equal_fetch_times_go_to_the_later_row(store):
    _fee(store, T)
    later = _fee(store, T, payload={**FEE, "maker": "0.0001"})
    assert store.latest_good_venue_fetch("BINANCE", "fee", "BTC/USDT")["id"] == later


def test_a_fee_outside_the_published_bounds_is_kept_as_missing_with_its_reason(store):
    good = _fee(store, T)
    high = _fee(store, T + timedelta(hours=1), payload={**FEE, "taker": "0.0007"})
    row = store.latest_venue_fetch("BINANCE", "fee", "BTC/USDT")
    assert row["id"] == high and row["ok"] == 0 and "taker" in row["reason"] and "0.0007" in row["reason"]
    assert row["payload"]["taker"] == "0.0007"  # what the venue said is kept
    assert store.latest_good_venue_fetch("BINANCE", "fee", "BTC/USDT")["id"] == good
    for bad in ({**FEE, "maker": "-0.0001"}, {**FEE, "maker": "abc"}, {"symbol": "BTCUSDT", "maker": "0.0002"}):
        _fee(store, T + timedelta(hours=2), payload=bad)
        assert store.latest_venue_fetch("BINANCE", "fee", "BTC/USDT")["ok"] == 0
    assert store.latest_good_venue_fetch("BINANCE", "fee", "BTC/USDT")["id"] == good


def test_a_good_fee_needs_its_bounds_and_a_failure_needs_its_reason(store):
    with pytest.raises(ValueError, match="bounds"):
        _fee(store, T, bounds=None)
    with pytest.raises(ValueError, match="reason"):
        _fee(store, T, ok=False, reason="", payload=None)
    with pytest.raises(ValueError, match="kind"):
        store.record_venue_fetch("BINANCE", "spread", "BTC/USDT", source="x", ok=False, reason="r", at=T)
    assert store.latest_venue_fetch("BINANCE", "fee", "BTC/USDT") is None


def test_numbers_are_kept_as_the_venue_wrote_them(store):
    _fee(store, T, payload={**FEE, "taker": "0.000550"})
    assert store.latest_good_venue_fetch("BINANCE", "fee", "BTC/USDT")["payload"]["taker"] == "0.000550"
    with pytest.raises(ValueError, match="string"):
        _fee(store, T, payload={**FEE, "taker": 0.00055})


def test_tiers_the_margin_check_accepts_are_good_and_a_broken_table_is_kept_as_missing(store):
    good = store.record_venue_fetch("BINANCE", "tiers", "BTC/USDT", source="risk-limit", ok=True, payload=TIERS, at=T)
    row = store.latest_good_venue_fetch("BINANCE", "tiers", "BTC/USDT")
    assert row["id"] == good and row["account"] is None and row["payload"] == TIERS
    typo = {**TIERS, "tiers": [TIERS["tiers"][0], {**TIERS["tiers"][1], "deduction": "501"}, TIERS["tiers"][2]]}
    store.record_venue_fetch("BINANCE", "tiers", "BTC/USDT", source="risk-limit", ok=True, payload=typo,
                             at=T + timedelta(days=1))
    bad = store.latest_venue_fetch("BINANCE", "tiers", "BTC/USDT")
    assert bad["ok"] == 0 and "continuous" in bad["reason"]
    assert store.latest_good_venue_fetch("BINANCE", "tiers", "BTC/USDT")["id"] == good
    with pytest.raises(ValueError, match="public"):
        store.record_venue_fetch("BINANCE", "tiers", "BTC/USDT", source="risk-limit", ok=True, payload=TIERS, at=T,
                                 account="paper-main")


def test_the_tiers_reader_gives_the_margin_table_with_an_open_last_tier(store):
    import math

    from sleeve_fund.store import margin_tiers_of

    store.record_venue_fetch("BINANCE", "tiers", "BTC/USDT", source="risk-limit", ok=True, payload=TIERS, at=T)
    table = margin_tiers_of(store.latest_good_venue_fetch("BINANCE", "tiers", "BTC/USDT"))
    assert [t.ceiling for t in table] == [300000.0, 2000000.0, math.inf]
    assert [t.rate for t in table] == [0.0033, 0.005, 0.0056] and [t.amount for t in table] == [0.0, 510.0, 1710.0]


def test_the_database_refuses_an_edit_or_a_delete(store, engine):
    row = _fee(store, T)
    for sql in ("UPDATE venue_terms_fetch SET ok = 0, reason = 'x' WHERE id = :i",
                "UPDATE venue_terms_fetch SET fetched_at = fetched_at WHERE id = :i",
                "DELETE FROM venue_terms_fetch WHERE id = :i"):
        with pytest.raises(IntegrityError, match="append-only"), engine.begin() as c:
            c.execute(text(sql), {"i": row})
    assert store.latest_good_venue_fetch("BINANCE", "fee", "BTC/USDT")["id"] == row


@pytest.mark.parametrize("cols", [
    "venue='binance', kind='fee', ok=1, reason='', payload='{}'",  # venue upper case
    "venue='BINANCE', kind='spread', ok=1, reason='', payload='{}'",  # kind
    "venue='BINANCE', kind='fee', ok=2, reason='', payload='{}'",  # ok is 0 or 1
    "venue='BINANCE', kind='fee', ok=1, reason='x', payload='{}'",  # a good row has no reason
    "venue='BINANCE', kind='fee', ok=0, reason='', payload=NULL",  # a failed row has one
    "venue='BINANCE', kind='fee', ok=1, reason='', payload=NULL",  # a good row has its payload
    "venue='BINANCE', kind='tiers', ok=1, reason='', payload='{}', account='paper-main'",  # tiers are public
    "venue='BINANCE', kind='fee', ok=1, reason='', payload='{}', account='nobody'",  # a known account
])
def test_the_database_holds_the_shape_whatever_writes_it(store, engine, cols):
    if "nobody" in cols and engine.dialect.name == "sqlite":
        pytest.skip("the project's SQLite runs without PRAGMA foreign_keys; Postgres holds the account link")
    with pytest.raises(IntegrityError), engine.begin() as c:
        _raw_insert(c, cols)
    with engine.begin() as c:  # the control: the same statement with good values is taken, so each refusal is its own
        _raw_insert(c, "venue='BINANCE', kind='fee', ok=1, reason='', payload='{}', account='paper-main'")
        _raw_insert(c, "venue='BINANCE', kind='tiers', ok=0, reason='timeout', payload=NULL")


def _raw_insert(c, cols):
    pairs = dict(p.split("=", 1) for p in cols.split(", "))
    pairs.setdefault("account", "NULL")
    names = ", ".join(pairs) + ", instrument, source, fetched_at"
    values = ", ".join(pairs.values()) + ", 'BTC/USDT', 'fee-rate', :t"
    c.execute(text(f"INSERT INTO venue_terms_fetch ({names}) VALUES ({values})"), {"t": T})


def _shape(eng):
    i = inspect(eng)
    t = "venue_terms_fetch"
    return (sorted((c["name"], c["nullable"]) for c in i.get_columns(t)),
            sorted(c["name"] for c in i.get_check_constraints(t)),
            sorted((x["name"], tuple(x["column_names"]), bool(x["unique"])) for x in i.get_indexes(t)),
            sorted((tuple(f["constrained_columns"]), f["referred_table"]) for f in i.get_foreign_keys(t)),
            _triggers(eng))


def _triggers(eng):
    q = ("SELECT tgname FROM pg_trigger WHERE tgrelid = 'venue_terms_fetch'::regclass AND NOT tgisinternal"
         if eng.dialect.name == "postgresql" else
         "SELECT name FROM sqlite_master WHERE type = 'trigger' AND tbl_name = 'venue_terms_fetch'")
    with eng.connect() as c:
        return sorted(r[0] for r in c.execute(text(q)))


def test_0029_migrated_table_equals_create_alls_and_refuses_edits(engine, tmp_path):
    schema.migrate(engine, log=lambda _: None)
    migrated = _shape(engine)
    assert migrated[4], "the append-only triggers are part of the migration"
    with engine.begin() as c:
        c.execute(text("INSERT INTO venue_terms_fetch (venue, kind, instrument, source, fetched_at, ok, reason) "
                       "VALUES ('BINANCE', 'fee', 'BTC/USDT', 'fee-rate', :t, 0, 'timeout')"), {"t": T})
    with pytest.raises(IntegrityError, match="append-only"), engine.begin() as c:
        c.execute(text("DELETE FROM venue_terms_fetch"))
    fresh = make_engine(f"sqlite:///{tmp_path}/fresh.db") if engine.dialect.name == "sqlite" else engine
    if fresh is engine:
        with engine.begin() as c:  # the append-only triggers refuse the cleanup's DELETE: drop the table itself
            c.execute(text("DROP TABLE venue_terms_fetch"))
        _clean(engine)
    Store(engine=fresh)
    assert _shape(fresh) == migrated


def test_0029_downgrade_refuses_while_fetches_are_kept(engine):
    from alembic import command

    schema.migrate(engine, log=lambda _: None)
    with engine.begin() as c:
        c.execute(text("INSERT INTO venue_terms_fetch (venue, kind, instrument, source, fetched_at, ok, reason) "
                       "VALUES ('BINANCE', 'fee', 'BTC/USDT', 'fee-rate', :t, 0, 'timeout')"), {"t": T})
    with pytest.raises(NotImplementedError, match="never dropped"), engine.begin() as conn:
        command.downgrade(schema._config(conn), "-1")
    with engine.begin() as c:
        c.execute(text("DROP TABLE venue_terms_fetch"))


def test_the_published_vip0_stand_in_is_good_and_says_it_is_published_not_the_accounts(store):
    """Advisor HEAD3: an account with no fee endpoint trades at the venue's published VIP0 schedule, labelled and
    alerted. It is a good row whose source is PUBLISHED_FEES, so the start check and the dashboard can tell it apart
    from a rate the account reported."""
    from sleeve_fund.store import PUBLISHED_FEES

    row_id = _fee(store, T, source=PUBLISHED_FEES)
    good = store.latest_good_venue_fetch("BINANCE", "fee", "BTC/USDT")
    assert PUBLISHED_FEES == "published-vip0" and good["id"] == row_id and good["source"] == PUBLISHED_FEES
    account = _fee(store, T + timedelta(hours=1))
    assert store.latest_good_venue_fetch("BINANCE", "fee", "BTC/USDT")["id"] == account


def test_the_published_stand_in_is_never_a_live_accounts_fees(store, engine):
    """Advisor FEE-STORE (1): the published VIP0 stand-in is for paper and demo accounts. Written for a live account
    it is kept as failed, with why; and the reader never takes one for a live account however it got there, so a live
    account with only a published row has no last-known-good fees and refuses entries (FEE-LIVE)."""
    from sleeve_fund.store import PUBLISHED_FEES, venue_terms_fetch_t
    from sqlalchemy import insert

    store.create_account("live-a", "live", venue="binance")
    row_id = _fee(store, T, source=PUBLISHED_FEES, account="live-a")
    kept = store.latest_venue_fetch("BINANCE", "fee", "BTC/USDT")
    assert kept["id"] == row_id and kept["ok"] == 0 and "paper and demo accounts only" in kept["reason"]
    assert store.latest_good_venue_fetch("BINANCE", "fee", "BTC/USDT", account="live-a") is None
    with engine.begin() as c:  # one written past the store: the reader still never takes it for the live account
        c.execute(insert(venue_terms_fetch_t).values(venue="BINANCE", kind="fee", instrument="BTC/USDT",
                                                     account="live-a", source=PUBLISHED_FEES, fetched_at=T, ok=1,
                                                     reason="", payload=FEE))
    assert store.latest_good_venue_fetch("BINANCE", "fee", "BTC/USDT", account="live-a") is None
    paper = _fee(store, T, source=PUBLISHED_FEES)  # a paper account's stand-in is good
    assert store.latest_good_venue_fetch("BINANCE", "fee", "BTC/USDT", account="paper-main")["id"] == paper
    live = _fee(store, T + timedelta(hours=1), account="live-a")  # the account's own rate is
    assert store.latest_good_venue_fetch("BINANCE", "fee", "BTC/USDT", account="live-a")["id"] == live


def test_the_stand_in_ages_like_any_fetch_a_fresh_row_each_daily_attempt(store):
    """Advisor FEE-STORE (2): each daily attempt that falls back writes a fresh stand-in row at that attempt's time,
    so the 24 h and 72 h rules judge it as any fetch. Three simulated days."""
    from sleeve_fund.store import PUBLISHED_FEES

    ids = [store.record_published_fees("BINANCE", "BTC/USDT", account="paper-main", at=T + timedelta(days=d),
                                       bounds=BOUNDS) for d in range(3)]
    assert len(set(ids)) == 3
    good = store.latest_good_venue_fetch("BINANCE", "fee", "BTC/USDT", account="paper-main")
    at = good["fetched_at"] if good["fetched_at"].tzinfo else good["fetched_at"].replace(tzinfo=timezone.utc)
    assert good["id"] == ids[-1] and good["source"] == PUBLISHED_FEES and at == T + timedelta(days=2)
    assert good["payload"] == {"maker": "0.0002", "taker": "0.00055"}  # the venue profile's published VIP0, as strings
    assert (T + timedelta(days=2, hours=1)) - at == timedelta(hours=1)  # one hour old, not three days


def test_fees_are_decimal_from_the_stored_string_never_through_a_float(store):
    """Advisor FEE-STORE (3): the reader parses the stored strings to Decimal. 0.00055 through a float is not 0.00055,
    so a float anywhere on the path shows here."""
    from sleeve_fund.store import venue_fees_of

    _fee(store, T)
    fees = venue_fees_of(store.latest_good_venue_fetch("BINANCE", "fee", "BTC/USDT"))
    assert type(fees.taker) is D and type(fees.maker) is D
    assert fees.taker == D("0.00055") and fees.taker != D(0.00055)
    assert D("123456.78") * fees.taker == D("67.9012290")
