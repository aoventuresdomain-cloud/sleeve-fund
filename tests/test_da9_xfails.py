"""QA master for DA-9 (exact decimals), written against main 1709cd9 before the build: every cell is a strict xfail
that turns into an XPASS (shown as FAILED) once DA-9 lands. Strip the marks to test a DA-9 head.

Spec: data-architect/phase2-db-hardening-plan.md "DA-9" (Done when 1-3) and the six Decimal rules in
qd/day0-interface-note.md (DA, 17:48 UK). Done when 4 (golden baseline unchanged) is the Head of QA's check, not here.

Cells
  1  replay_book (journal_book) cash equals the stored cash to the cent, before and after the migration to head,
     the migrated value equals the pre-migration one, and after the migration the book is Decimal.
  2  10,000 fills sum with zero drift: the journal book and the database's own SUM are exact.
  3  schema: no Float column left on money/qty, in the code and in a migrated database; each is NUMERIC(38,18).
  4  read paths that sum money return Decimal (journal_book, funding_total, insurance_total) and the raw rows they
     sum (fills, funding, insurance, the equity mark's cash) come back as Decimal.
  5  NaN and +/-Infinity (float or Decimal) are refused at the store's money boundaries with ValueError or
     TypeError, and nothing is written.
  6  storage quantises money ROUND_HALF_EVEN at the column scale (18 places; Postgres' own NUMERIC rounding is half
     away from zero, so the store must quantise first); a quantity or price on an 18-place grid is never re-rounded.

Runs on Postgres when TEST_DATABASE_URL is set (one database per concurrent run), else on a temporary SQLite file.
HoQA 7 Oct: exactness (C2, C6, and C1's exact-figure and no-re-rounding checks) is required on Postgres only and
skips on SQLite; C1's to-the-cent and Decimal checks, C3, C4 and C5 run on both.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from sqlalchemy import Numeric, func, insert, inspect, select, text

from sleeve_fund import schema
from sleeve_fund.store import Store, fills_t, funding_t, insurance_t, make_engine, metadata

XF = pytest.mark.xfail(strict=True, raises=AssertionError, reason="DA-9 not built on 1709cd9")
# pytestmark = XF  # DA-9 built: marks lifted

PG = os.environ.get("TEST_DATABASE_URL", "").startswith(("postgres://", "postgresql"))
# HoQA ruling 7 Oct: Postgres is the journal of record, so exact storage is required there only.
SQLITE_WHY = "exact storage is required on Postgres; SQLite stores Numeric as REAL (DA design choice, NICE not MUST)"
pg_only = pytest.mark.skipif(not PG, reason=SQLITE_WHY)

PRE_DA9 = "0007"  # the last migration on 1709cd9; DA-9's migration comes after it
SCALE = 18
CENT = Decimal("0.01")
T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)

# Money and quantity columns (DA-9). Pure ratios stay float (rule: rates, shares, fractions, weights,
# volatilities): funding.rate, exit_plans.stop_frac/tp_frac/planned_r, fee_schedules.maker/taker,
# spreads.half_spread, trials.sharpe.
MONEY_QTY = {
    "sleeves": {"starting_balance"},
    "equity": {"equity", "cash", "qty", "price", "benchmark"},
    "fills": {"qty", "price", "fee"},
    "funding": {"qty", "price", "amount"},
    "insurance": {"price", "amount"},
    "demo_mirror": {"amount", "price"},
    "orders": {"qty", "filled_qty", "avg_px", "fee"},
    "exit_plans": {"risk_amount"},
}
RATIOS = {"funding.rate", "exit_plans.stop_frac", "exit_plans.tp_frac", "exit_plans.planned_r", "fee_schedules.maker",
          "fee_schedules.taker", "spreads.half_spread", "trials.sharpe"}


def _clean(eng):
    with eng.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
    metadata.drop_all(eng)


@pytest.fixture
def engine(tmp_path):
    url = os.environ.get("TEST_DATABASE_URL")
    eng = make_engine(url or f"sqlite:///{tmp_path}/da9.db")
    _clean(eng)
    yield eng
    _clean(eng)
    eng.dispose()


@pytest.fixture
def store(engine):
    schema.migrate(engine, log=lambda _: None)
    return Store(engine=engine)


def _sleeve(store: Store, name: str = "da9", start: str = "10000") -> None:
    store.create_sleeve(name=name, strategy="trend", instrument="BTC/USD", bar_spec="1-HOUR-LAST-EXTERNAL",
                        starting_balance=Decimal(start))


def _is_dec(x) -> bool:
    return isinstance(x, Decimal) and not isinstance(x, bool)


# --- a realistic journal, with its exact Decimal cash --------------------------------------------------------

def _journal(n: int = 240):
    """Round trips on an 1e-8 lot at prices on a 0.1 tick with a 0.26% taker fee (fee kept at 18 places), plus
    funding and an insurance credit. Returns (fills, funding, insurance, exact cash)."""
    fills, cash = [], Decimal("10000")
    price, qty = Decimal("61234.5"), Decimal("0.01234567")
    for i in range(n):
        side = "BUY" if i % 2 == 0 else "SELL"
        px = price + Decimal(i % 17) * Decimal("0.1") - Decimal(i % 5) * Decimal("0.3")
        fee = (qty * px * Decimal("0.0026")).quantize(Decimal(1).scaleb(-SCALE))
        sign = 1 if side == "BUY" else -1
        cash -= sign * qty * px + fee
        fills.append(dict(side=side, qty=qty, price=px, fee=fee, order_id=f"o{i}", trade_id=f"t{i}",
                          ts=T0 + timedelta(minutes=i)))
    funding = [Decimal("-0.123456789012345678"), Decimal("0.0000003"), Decimal("1.1")]
    insurance = [Decimal("0.07")]
    cash += sum(funding) + sum(insurance)
    return fills, funding, insurance, cash


def _seed(store: Store, sleeve: str = "da9"):
    fills, funding, insurance, cash = _journal()
    for f in fills:
        store.record_fill(sleeve, **f)
    for k, a in enumerate(funding):
        store.record_funding(sleeve, qty=Decimal("0.01234567"), price=Decimal("61234.5"), rate=0.0001, amount=a,
                             ts=T0 + timedelta(hours=8 * k))
    for a in insurance:
        store.record_insurance(sleeve, price=Decimal("61000"), amount=a, ts=T0 + timedelta(days=1))
    store.record_equity(sleeve, equity=cash, cash=cash, qty=Decimal(0), price=Decimal("61234.5"),
                        benchmark=Decimal("10000"), ts=T0 + timedelta(days=2))
    return cash


def _to_cent(x) -> Decimal:
    return Decimal(str(x)).quantize(CENT)


# --- cell 1 --------------------------------------------------------------------------------------------------

def test_c1_replay_cash_equals_stored_cash_to_the_cent_before_and_after_the_migration(engine):
    with engine.begin() as conn:
        from alembic import command

        command.upgrade(schema._config(conn), PRE_DA9)
    store = Store(engine=engine)
    _sleeve(store)
    exact = _seed(store)

    before = store.journal_book("da9", store.sleeve("da9").starting_balance)
    stored_before = store.last_equity("da9")["cash"]
    assert _to_cent(before["cash"]) == _to_cent(stored_before)
    if PG:  # the exact figure: Postgres only (SQLITE_WHY)
        assert _to_cent(before["cash"]) == exact.quantize(CENT)

    schema.migrate(engine, log=lambda _: None)
    store = Store(engine=engine)
    after = store.journal_book("da9", store.sleeve("da9").starting_balance)
    stored_after = store.last_equity("da9")["cash"]
    assert _is_dec(after["cash"]) and _is_dec(stored_after), (type(after["cash"]), type(stored_after))
    assert after["cash"].quantize(CENT) == stored_after.quantize(CENT)
    if PG:  # the exact figure and no re-rounding: Postgres only (SQLITE_WHY)
        assert after["cash"].quantize(CENT) == exact.quantize(CENT)
        assert after["cash"].quantize(CENT) == _to_cent(before["cash"])  # converted as stored, no re-rounding
    cols = {c["name"]: c["type"] for c in inspect(engine).get_columns("fills")}
    assert all(isinstance(cols[k], Numeric) and not _float_type(cols[k]) for k in ("qty", "price", "fee")), cols


# --- cell 2 --------------------------------------------------------------------------------------------------

N = 10_000


@pg_only
def test_c2_ten_thousand_fills_sum_with_zero_drift(store):
    _sleeve(store)
    # 0.1 x 0.3 with a 0.0001 fee: none is exact in binary, so a float sum drifts; round trips end flat.
    rows = [dict(sleeve="da9", ts=T0 + timedelta(seconds=i), side="BUY" if i % 2 == 0 else "SELL",
                 qty=Decimal("0.1"), price=Decimal("0.3"), fee=Decimal("0.0001"), order_id=f"o{i}", trade_id=f"t{i}")
            for i in range(N)]
    with store.engine.begin() as c:
        c.execute(insert(fills_t), rows)
    book = store.journal_book("da9", store.sleeve("da9").starting_balance)
    assert book["fills"] == N
    assert book["cash"] == Decimal("9999"), book["cash"]  # 10,000 less 10,000 x 0.0001, exactly
    assert book["qty"] == 0 and _is_dec(book["cash"])
    with store.engine.connect() as c:
        fee_sum = c.execute(select(func.sum(fills_t.c.fee)).where(fills_t.c.sleeve == "da9")).scalar()
        notional = c.execute(select(func.sum(fills_t.c.qty * fills_t.c.price)).where(fills_t.c.sleeve == "da9")).scalar()
    assert fee_sum == Decimal("1"), fee_sum
    assert notional == Decimal("300"), notional


# --- cell 3 --------------------------------------------------------------------------------------------------

def _float_type(t) -> bool:
    # sqlalchemy.Float subclasses Numeric; a NUMERIC column is Numeric but not Float
    from sqlalchemy import Float

    return isinstance(t, Float) or type(t).__name__.upper() in ("REAL", "DOUBLE", "DOUBLE_PRECISION", "FLOAT")


def _bad(columns_by_table) -> list[str]:
    bad = []
    for table, cols in MONEY_QTY.items():
        for name in sorted(cols):
            t = columns_by_table[table][name]
            if _float_type(t) or not isinstance(t, Numeric) or (t.precision, t.scale) != (38, SCALE):
                bad.append(f"{table}.{name}: {t!r}")
    return bad


def test_c3_no_float_column_left_on_money_or_qty_in_the_code(store):
    found = _bad({t: {c.name: c.type for c in metadata.tables[t].columns} for t in MONEY_QTY})
    assert found == [], "Float (or not NUMERIC(38,18)) money/qty columns in store.py:\n" + "\n".join(found)


def test_c3_every_float_column_left_is_a_named_pure_ratio(store):
    # a money or quantity column added later as Float (any table, any name) is caught here
    floats = sorted(f"{t.name}.{c.name}" for t in metadata.sorted_tables for c in t.columns if _float_type(c.type))
    assert [f for f in floats if f not in RATIOS] == [], floats


def test_c3_no_float_column_left_on_money_or_qty_in_a_migrated_database(store):
    insp = inspect(store.engine)
    found = _bad({t: {c["name"]: c["type"] for c in insp.get_columns(t)} for t in MONEY_QTY})
    assert found == [], "Float (or not NUMERIC(38,18)) money/qty columns after migrating:\n" + "\n".join(found)


# --- cell 4 --------------------------------------------------------------------------------------------------

def test_c4_summed_money_read_paths_return_decimal(store):
    _sleeve(store)
    _seed(store)
    book = store.journal_book("da9", store.sleeve("da9").starting_balance)
    wrong = {k: type(book[k]).__name__ for k in ("cash", "qty", "funding", "insurance", "entry_fees")
             if not _is_dec(book[k])}
    for name, v in (("funding_total", store.funding_total("da9")), ("insurance_total", store.insurance_total("da9"))):
        if not _is_dec(v):
            wrong[name] = type(v).__name__
    assert wrong == {}, wrong


def test_c4_the_rows_the_book_sums_come_back_as_decimal(store):
    _sleeve(store)
    _seed(store)
    wrong = {}
    for name, rows, keys in (("fills", store.fills("da9", limit=3), ("qty", "price", "fee")),
                             ("funding", store.funding("da9", limit=3), ("qty", "price", "amount")),
                             ("insurance", store.insurance("da9", limit=3), ("price", "amount")),
                             ("equity", [store.last_equity("da9")], ("equity", "cash", "qty", "price"))):
        for r in rows:
            for k in keys:
                if not _is_dec(r[k]):
                    wrong[f"{name}.{k}"] = type(r[k]).__name__
    assert wrong == {}, wrong


# --- cell 5 --------------------------------------------------------------------------------------------------

BAD = [("nan", float("nan")), ("inf", float("inf")), ("-inf", float("-inf")),
       ("Decimal NaN", Decimal("NaN")), ("Decimal Inf", Decimal("Infinity"))]


def _fill(**over):
    f = dict(side="BUY", qty=Decimal("0.01"), price=Decimal("60000"), fee=Decimal("1.56"), order_id="o", trade_id="t",
             ts=T0)
    return {**f, **over}


BOUNDARIES = [
    ("record_fill.qty", lambda s, v: s.record_fill("da9", **_fill(qty=v)), fills_t),
    ("record_fill.price", lambda s, v: s.record_fill("da9", **_fill(price=v)), fills_t),
    ("record_fill.fee", lambda s, v: s.record_fill("da9", **_fill(fee=v)), fills_t),
    ("record_funding.amount", lambda s, v: s.record_funding("da9", qty=Decimal("0.01"), price=Decimal("60000"),
                                                            rate=0.0001, amount=v, ts=T0), funding_t),
    ("record_insurance.amount", lambda s, v: s.record_insurance("da9", price=Decimal("60000"), amount=v, ts=T0),
     insurance_t),
    ("record_equity.cash", lambda s, v: s.record_equity("da9", equity=Decimal("10000"), cash=v, qty=Decimal(0),
                                                        price=Decimal("60000"), benchmark=Decimal("10000"), ts=T0),
     None),
]


@pytest.mark.parametrize("where,call,table", BOUNDARIES, ids=[b[0] for b in BOUNDARIES])
@pytest.mark.parametrize("label,value", BAD, ids=[b[0] for b in BAD])
def test_c5_nan_and_infinity_are_refused_at_money_boundaries(store, where, call, table, label, value):
    _sleeve(store)
    raised = None
    try:
        call(store, value)
    except Exception as e:  # noqa: BLE001 - the cell judges the kind below
        raised = e
    assert isinstance(raised, (ValueError, TypeError)), \
        f"{where} took {label}: {'stored without a refusal' if raised is None else type(raised).__name__}"
    if table is not None:
        with store.engine.connect() as c:
            assert c.execute(select(func.count()).select_from(table)).scalar() == 0


# --- cell 6 --------------------------------------------------------------------------------------------------

# (written, read back): the 19th place is a 5, where half-even and Postgres' half-away-from-zero differ
HALF_EVEN = [
    (Decimal("0.0000000000000000125"), Decimal("0.000000000000000012")),
    (Decimal("0.0000000000000000135"), Decimal("0.000000000000000014")),
    (Decimal("-0.0000000000000000125"), Decimal("-0.000000000000000012")),
    (Decimal("1234.5678901234567890125"), Decimal("1234.567890123456789012")),
]


@pg_only
@pytest.mark.parametrize("written,expected", HALF_EVEN, ids=[str(w) for w, _ in HALF_EVEN])
def test_c6_money_is_quantised_half_even_at_the_column_scale(store, written, expected):
    _sleeve(store)
    store.record_fill("da9", **_fill(fee=written))
    store.record_funding("da9", qty=Decimal("0.01"), price=Decimal("60000"), rate=0.0001, amount=written, ts=T0)
    store.record_insurance("da9", price=Decimal("60000"), amount=written, ts=T0)
    got = {"fills.fee": store.fills("da9")[0]["fee"], "funding.amount": store.funding("da9")[0]["amount"],
           "insurance.amount": store.insurance("da9")[0]["amount"]}
    assert got == dict.fromkeys(got, expected), got


GRID = [Decimal("0.123456789012345678"), Decimal("123456789.12345678"), Decimal("0.00000001"),
        Decimal("61234.123456789012345678")]


@pg_only
@pytest.mark.parametrize("value", GRID, ids=[str(v) for v in GRID])
def test_c6_quantity_and_price_on_the_grid_are_never_re_rounded(store, value):
    _sleeve(store)
    store.record_fill("da9", **_fill(qty=value, price=value))
    row = store.fills("da9")[0]
    assert (row["qty"], row["price"]) == (value, value), (row["qty"], row["price"])
    assert _is_dec(row["qty"]) and _is_dec(row["price"])
