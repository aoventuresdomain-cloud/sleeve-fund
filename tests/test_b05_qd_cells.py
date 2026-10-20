"""B-05 cells (HoE, CR's mutants on 544cd42d; tests only, no code change): what fails closed stays closed, the
funding split adds up to the venue's figure with the rounding on the largest holder.
The OFF-start cell (K2) is PE4's (TEC v2). On Postgres too (TEST_DATABASE_URL)."""

from __future__ import annotations

from decimal import Decimal

from test_b05_guards import DEMO_ENV, NOW, _check
from test_b05_restart import INSTR, _Account, _long, _settled, _start, store  # noqa: F401

from sleeve_fund.paper import exec_client, exec_start, runtime


def test_an_unreadable_client_mode_is_read_as_on_and_the_journals_hold_still_refuses(store, monkeypatch):  # noqa: F811
    """CR (1), runtime.exec_target_block: the mode can't be read, so it is ON (fail closed) and the hold stands."""
    exec_client.record_connect(store, "s1", _check(account={"retCode": 0, "result": {"marginMode": "REGULAR_MARGIN"}}),
                               NOW)

    def unreadable(environ):
        raise RuntimeError("the client's settings couldn't be read")

    monkeypatch.setattr(exec_client, "exec_settings", unreadable)
    why = runtime.exec_target_block(environ={}, store=store, name="s1")
    assert why is not None and why.code == "exec_target" and "not isolated" in str(why), why


def test_an_unreadable_journal_holds_the_instrument_for_a_foreign_order(store, monkeypatch):  # noqa: F811
    """CR (2), exec_start.foreign_hold: an unread journal is a hold, and the client's holds say so."""

    def down(*a, **k):
        raise RuntimeError("journal down")

    monkeypatch.setattr(store, "sleeve_events_since", down)
    why = exec_start.foreign_hold(store, INSTR)
    assert why is not None and "couldn't be read" in why
    assert why in exec_client.holds(store, "s1")


def test_a_split_that_doesnt_divide_puts_the_rounding_on_the_largest_holder_and_adds_up_exactly():
    """CR (3), exec_start.split: thirds of 1.00000000 at the 1e-8 step leave a remainder; the tie goes by tag."""
    held = {"s1": Decimal("0.01"), "s2": Decimal("0.01"), "s3": Decimal("0.01")}
    amount = Decimal("-1.00000000")
    parts = exec_start.split(amount, Decimal("0.03"), held)
    assert sum(parts.values(), Decimal(0)) == amount
    third = Decimal("-0.33333333")
    assert parts == {"s1": third, "s2": third, "s3": amount - 2 * third}  # max by (abs, tag): s3 takes the rounding
    bigger = exec_start.split(amount, Decimal("0.04"), {"s1": Decimal("0.01"), "s2": Decimal("0.02"),
                                                         "s3": Decimal("0.01")})
    assert sum(bigger.values(), Decimal(0)) == amount and bigger["s1"] == bigger["s3"] == Decimal("-0.25")
    assert bigger["s2"] == amount - 2 * Decimal("-0.25")


def test_a_settlement_split_three_ways_books_the_venues_figure_exactly(store):  # noqa: F811
    """CR (3) through book_settlements: the three journals' funding adds up to the venue's figure to the last digit."""
    store.create_sleeve(name="s3", strategy="ping_pong", instrument=INSTR, bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, desired_state="running")
    held = {t: Decimal("0.01") for t in ("s1", "s2", "s3")}
    done = exec_start.book_settlements(store, [_settled(size="0.03", change="-1.00000000")], lambda at: held,
                                       account="demo-1", instrument=INSTR, tags=("s1", "s2", "s3"), now=NOW)
    assert set(done["booked"]) == {"S-1"} and not done["gaps"]
    assert sum(store.funding_total(t) for t in held) == Decimal("-1.00000000")
    assert store.funding_total("s3") == Decimal("-0.33333334")


def test_a_foreign_order_gone_while_the_start_check_is_still_untied_keeps_the_instruments_hold(store, monkeypatch):  # noqa: F811
    """CR F1: the foreign hold lifts only when the foreign order is gone AND a start check ties; gone alone, every
    Strategy on the instrument stays held for it, and it lifts at the first start that ties."""
    monkeypatch.setenv("EXEC_MODE", "demo")
    _long(store, "s1", 0.03, "sf-" + "a" * 32)
    venue = _Account(size="0.05", side="Buy").start()  # the account holds 0.05, the journals 0.03: untied
    try:
        venue.rest("", key="manual-1")
        _start(store, venue)
        venue.orders.pop("manual-1")
        untied = _start(store, venue)
        assert not untied["result"].clean and untied["foreign"] == []
        held = exec_start.foreign_hold(store, INSTR)
        assert held is not None and "in no journal of the fund" in held
        for name in ("s1", "s2"):
            assert held in exec_client.holds(store, name)
        assert runtime.exec_target_block(store=store, name="s2") is not None
        _long(store, "s2", 0.02, "sf-" + "b" * 32)  # now the journals hold 0.05 too
        tied = _start(store, venue)
    finally:
        venue.stop()
    assert tied["result"].clean and exec_start.foreign_hold(store, INSTR) is None
