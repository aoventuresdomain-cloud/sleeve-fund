"""PAGE-SIZE lines 1 and 2 (Advisor ruling, 20 Oct; HoE route (a), 21:15 UK): inside the fund, the Strategy page's
position size adds two figures from the entry's gate decision, never worked again:
- "Sized at N% of the room T1 allows": the decision's own size_share;
- "Risk if the stop is hit: x% of the fund (T1 limit 4%)": approved quantity x risk per unit over the book's equity
  then, and T1's cap over the same equity.
Read from the recorded row only, never worked from today's T1 (Advisor via the HoE, 21:20 UK). The four columns
(t1_used_before, t1_cap and risk_per_unit in USDT, size_share a fraction) are gate_decisions' own (DA, 0034), written
by B-24 part 2's writer. A decision from before them reads "not in this entry's record"; one with no T1 room left
(size_share NULL, the other three set) reads n/a for its room; an entry with no decision shows neither line. Expected
figures are written out here."""

import re
from decimal import Decimal

from sqlalchemy import insert
from test_fe_b20_fund_held_wording import ENTRY, QTY, _hold, _words, c  # noqa: F401
from test_fe_h4_page_dd_size import _journaled_long, _size_cells
from test_halt_ref import (
    store,  # noqa: F401 - the journal, on Postgres when TEST_DATABASE_URL is set
)

from sleeve_fund.dashboard import trading
from sleeve_fund.store import gate_decisions_t, utcnow

# The fund at 100,000; T1 4% = 4,000 of loss at the stops in BTC; 1,000 already used; 150 a unit to the stop, fees in.
# Room: (4,000 - 1,000) / 150 = 20 units. Approved 12: 60% of the room; 12 x 150 = 1,800 = 1.8% of the fund.
T1 = {"t1_used_before": Decimal(1000), "t1_cap": Decimal(4000), "risk_per_unit": Decimal(150),
      "size_share": Decimal("0.6")}
BOOK = Decimal(100000)


def _decided(store, s, approved=Decimal(12), seq=1, bar_ts=None, **t1):  # noqa: F811
    """The gate's approved decision for the entry order, as the gate writes it; `t1`: its T1 columns (0034), none for
    a row from before them."""
    with store.engine.begin() as cx:
        cx.execute(insert(gate_decisions_t), [{
            "seq": seq, "sleeve_id": s.id, "bar_ts": bar_ts or utcnow(), "intent_id": "i1", "intent": "entry", "underlying": "BTC",
            "order_id": f"{s.name}-1", "profile_version": 1, "outcome": "approved", "requested_qty": approved,
            "approved_qty": approved, "price": Decimal(str(ENTRY)), "book_equity": BOOK, "stage": "submit",
            "decided_at": utcnow(), **t1}])


def _lines(cell):
    return [ln for ln in ("room T1 allows", "if the stop is hit") if ln in cell]


def test_the_two_lines_are_the_decisions_own_t1_figures():
    d = {"approved_qty": Decimal(12), "book_equity": BOOK, **T1}
    got = trading.t1_sizing(d)
    assert got == {"room_share": 0.6, "stop_share": 0.018, "t1_limit": 0.04}, got


def test_a_decision_without_its_t1_figures_has_none_and_never_a_guess():
    base = {"approved_qty": Decimal(12), "book_equity": BOOK, **T1}
    for missing in ("t1_cap", "risk_per_unit", "book_equity"):
        assert trading.t1_sizing({k: v for k, v in base.items() if k != missing}) is None, missing
        assert trading.t1_sizing({**base, missing: None}) is None, missing
    assert trading.t1_sizing({**base, "book_equity": Decimal(0)}) is None
    assert trading.t1_sizing(None) is None


def test_fund_held_the_page_shows_both_lines_from_the_decision(c, store):  # noqa: F811
    _hold(store)
    s = _journaled_long(store, fund_equity=100_000.0)
    _decided(store, s, **T1)
    for cell in _size_cells(c, s.name):
        assert "Sized at 60% of the room T1 allows" in cell, cell
        assert "Risk if the stop is hit: 1.8% of the fund (T1 limit 4%)" in cell, cell
        assert cell.index("room T1 allows") < cell.index("if the stop is hit") < cell.index("Position value")


def test_fund_held_a_decision_journaled_before_the_t1_columns_says_so(c, store):  # noqa: F811
    _hold(store)
    s = _journaled_long(store, fund_equity=100_000.0)
    _decided(store, s)
    assert store.entry_decision(f"{s.name}-1")["approved_qty"] == Decimal(12)
    for cell in _size_cells(c, s.name):
        assert re.search(r"Room T1 allows: not in this entry's record", cell), cell
        assert "Risk if the stop is hit: not in this entry's record" in cell, cell
        assert "Position value" in cell  # the line Head 4 shows stays


def test_fund_held_an_entry_with_no_gate_decision_shows_neither_line(c, store):  # noqa: F811
    _hold(store)
    s = _journaled_long(store, fund_equity=100_000.0)
    assert store.entry_decision(f"{s.name}-1") is None
    for cell in _size_cells(c, s.name):
        assert _lines(cell) == [] and "T1" not in cell, cell


def test_the_lines_are_the_recorded_ones_never_worked_from_today_s_t1(c, store):  # noqa: F811
    """The Advisor's cell: T1's cap changed after the entry (4% to 2%) leaves that entry's lines as recorded."""
    from test_compliance_tests import _edit

    _hold(store)
    s = _journaled_long(store, fund_equity=100_000.0)
    _decided(store, s, **T1)
    before = _size_cells(c, s.name)
    c1 = store.compliance()
    store.add_compliance_version(_edit(c1, asset_stop_loss="0.02"), c1.asset_types, expected_version=c1.version,
                                 note="tighter T1", environ={})
    assert next(t["limit"] for t in store.compliance().tests if t["test"] == "asset_stop_loss") == Decimal("0.02")
    assert _size_cells(c, s.name) == before
    assert all("T1 limit 4%" in cell and "Sized at 60%" in cell for cell in before), before


def test_with_no_t1_room_left_the_room_line_reads_n_a_and_the_risk_line_stays(c, store):  # noqa: F811
    """QD2's writer: no room left (cap - used <= 0) records size_share NULL and the other three set."""
    assert trading.t1_sizing({"approved_qty": Decimal(12), "book_equity": BOOK, **T1, "size_share": None}) == {
        "room_share": None, "stop_share": 0.018, "t1_limit": 0.04}
    _hold(store)
    s = _journaled_long(store, fund_equity=100_000.0)
    _decided(store, s, **{**T1, "t1_used_before": Decimal(4000), "size_share": None})  # cap - used = 0 (DA, QD2)
    for cell in _size_cells(c, s.name):
        assert "Room T1 allows: n/a (none left when it was sized)" in cell and "Sized at" not in cell, cell
        assert "not in this entry's record" not in cell, cell
        assert "Risk if the stop is hit: 1.8% of the fund (T1 limit 4%)" in cell, cell


def test_the_limit_is_the_recorded_cap_and_one_type_throughout():
    """CR F1, F3 and F4: a cap recorded at 2% of the book reads 2%, not a fixed 4%; a column read back as a float beside
    Decimal ones never raises; t1_used_before isn't needed for either line (QD2 still journals it)."""
    d = {"approved_qty": Decimal(12), "book_equity": BOOK, "t1_cap": Decimal(2000), "risk_per_unit": 150.0,
         "size_share": 0.9}
    assert trading.t1_sizing(d) == {"room_share": 0.9, "stop_share": 0.018, "t1_limit": 0.02}


def test_the_page_reads_the_orders_latest_decision(c, store):  # noqa: F811
    """CR F2: an order with two decisions (a check re-run) shows the later one's."""
    from datetime import timedelta

    _hold(store)
    s = _journaled_long(store, fund_equity=100_000.0)
    _decided(store, s, approved=Decimal(5), seq=1, bar_ts=utcnow() - timedelta(minutes=1))
    _decided(store, s, approved=Decimal(12), seq=2)
    assert store.entry_decision(f"{s.name}-1")["approved_qty"] == Decimal(12)
