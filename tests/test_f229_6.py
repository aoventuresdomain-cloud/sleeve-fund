"""F229-6 (HoE 13 Oct 18:12 UK): a Strategy whose quote currency is held under 8 places, on the test account its venue
names (CCY-8DP), is outside the fund's book. One predicate (supervisor.outside_fund_book) leaves it out of the fund's
equity and of both fund_holdings, and keeps its entries off the book gate; its own per-Strategy checks still run."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest

from sleeve_fund import supervisor
from sleeve_fund.paper import portfolio_gate
from sleeve_fund.portfolio.limits import Book, Holding, Intent, decide
from sleeve_fund.risk import PORTFOLIO
from sleeve_fund.store import Store


def _store(tmp_path) -> Store:
    store = Store(f"sqlite:///{tmp_path}/f.db", portfolio_gate=True)
    for name, inst, venue in (("usd-test", "BTC/USD", "KRAKEN"), ("usdt", "BTC/USDT", "BINANCE")):
        store.create_sleeve(name=name, strategy="ping_pong", instrument=inst, bar_spec="1-MINUTE-LAST-INTERNAL",
                            starting_balance=10_000, venue=venue)
    return store


def test_the_predicate_is_a_sub_8dp_quote_on_its_venues_named_test_account(tmp_path):
    store = _store(tmp_path)
    assert supervisor.outside_fund_book(store, store.sleeve("usd-test"))
    assert not supervisor.outside_fund_book(store, store.sleeve("usdt"))  # held to 8 places: the fund's
    store.create_account("desk-live", "live", venue="KRAKEN")
    store.assign_account("usd-test", "desk-live")  # not a named test account (it could never start there)
    assert not supervisor.outside_fund_book(store, store.sleeve("usd-test"))


def test_the_funds_equity_leaves_the_test_strategy_out(tmp_path):
    store = _store(tmp_path)
    assert supervisor.fund_equity(store) == Decimal(10_000)  # usdt's capital only


def test_both_fund_holdings_leave_the_test_strategys_position_out(tmp_path, monkeypatch):
    store = _store(tmp_path)
    held = {"usd-test": 0.1, "usdt": 0.0}
    monkeypatch.setattr(store, "journal_book", lambda name, bal: {"qty": held[name], "entry_px": 60_000.0,
                                                                  "fills": 1, "cash": 0.0})
    monkeypatch.setattr(store, "last_equity", lambda name: {"price": 60_000.0})
    assert portfolio_gate.fund_holdings(store, atr_pct=lambda s: 0.02) == ()
    assert supervisor.fund_holdings(store) == ()  # the cache row's stub doesn't see it either
    held.update({"usd-test": 0.0, "usdt": 0.1})
    assert len(portfolio_gate.fund_holdings(store, atr_pct=lambda s: 0.02)) == 1


def test_one_predicate_decides_all_of_them(tmp_path, monkeypatch):
    """Flip the predicate and every reader follows: equity, both holdings and the strategy's gate."""
    store = _store(tmp_path)
    monkeypatch.setattr(supervisor, "outside_fund_book", lambda store, s: True)
    assert supervisor.fund_equity(store) == 0
    monkeypatch.setattr(store, "journal_book", lambda name, bal: {"qty": 0.1, "entry_px": 60_000.0, "fills": 1,
                                                                  "cash": 0.0})
    monkeypatch.setattr(store, "last_equity", lambda name: {"price": 60_000.0})
    assert portfolio_gate.fund_holdings(store, atr_pct=lambda s: 0.02) == ()
    assert supervisor.fund_holdings(store) == ()
    assert _gate_of(store, "usdt") is None


@pytest.mark.parametrize("name, gated", [("usd-test", False), ("usdt", True)])
def test_the_test_strategys_entries_reserve_nothing_against_the_book_gate(tmp_path, name, gated):
    store = _store(tmp_path)
    assert (_gate_of(store, name) is not None) is gated


def _gate_of(store, name):
    from sleeve_fund.strategies.base import LongFlatStrategy

    s = LongFlatStrategy.__new__(LongFlatStrategy)
    s._portfolio, s._paper_gate = None, None
    s.runtime = SimpleNamespace(name=name, backtest=False, store=store)
    return LongFlatStrategy._paper_gated(s)


# --- Archived but still holding (Advisor 14 Oct 00:15 UK, cells a-e): the gate counts every Strategy whose journal
# holds a position, whatever its status, as the dashboard's book does.

T_MARK = datetime(2026, 10, 14, 9, 0, tzinfo=UTC)


def _archived_holder(tmp_path) -> Store:
    """usdt (BTC perp, balanced: 2x) bought 0.1 at 60,000 with a 1% stop, marked, then stopped and archived holding."""
    store = Store(f"sqlite:///{tmp_path}/h.db", portfolio_gate=True)
    store.create_sleeve(name="usdt", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, venue="BINANCE", params={"market": "perp"})
    store.record_order("usdt", order_id="e1", side="BUY", qty=0.1, intent="entry", reason="r",
                       signal={"stop_frac": 0.01}, ts=T_MARK)
    store.record_fill("usdt", side="BUY", qty=0.1, price=60_000.0, fee=0.0, order_id="e1", trade_id="t1", ts=T_MARK)
    store.record_equity("usdt", equity=10_000.0, cash=4_000.0, qty=0.1, price=60_000.0, benchmark=10_000.0,
                        ts=T_MARK + timedelta(minutes=1))
    store.set_desired_state("usdt", "stopped")
    store.archive("usdt")
    return store


def _atr(s):
    return 0.02


def test_a_an_archived_strategy_still_holding_counts_in_gross_net_margin_and_open_risk(tmp_path):
    store = _archived_holder(tmp_path)
    [h] = portfolio_gate.fund_holdings(store, atr_pct=_atr)
    assert (h.underlying, h.notional, h.margin) == ("BTC", Decimal(6_000), Decimal(3_000))
    assert h.risk == Decimal(60)  # to its 1% stop
    book = Book(Decimal(10_000), (h,))
    assert (book.gross(), book.net("BTC"), book.margin(), book.open_risk()) == (6_000, 6_000, 3_000, 60)
    # The interim 5% limit and the dashboard tile count it too (open_risk.account_book, book_open_risk).
    from sleeve_fund import open_risk

    assert [r["sleeve"] for r in open_risk.book_open_risk(store, _atr)] == ["usdt"]
    assert open_risk.account_book(store, "other", 0.0, _atr)[1] == pytest.approx(60.0)


def test_b_an_entry_that_breaches_gross_only_with_that_holding_is_trimmed(tmp_path):
    store = _archived_holder(tmp_path)
    others = (Holding("ETH", Decimal(4_000), Decimal(100), Decimal(10)),
              Holding("SOL", Decimal(4_000), Decimal(100), Decimal(10)))
    entry = Intent("XRP", 1, Decimal(1), Decimal(1_500), Decimal(300), Decimal(10), Decimal("0.001"), Decimal("0.001"))
    held = portfolio_gate.fund_holdings(store, atr_pct=_atr)
    without = decide(Book(Decimal(10_000), others), entry, PORTFOLIO)
    counted = decide(Book(Decimal(10_000), (*held, *others)), entry, PORTFOLIO)
    assert without.outcome == "approved"
    assert counted.outcome == "trimmed" and counted.limit_hit == "gross" and counted.approved_qty == Decimal("0.666")


def test_c_the_holding_drops_out_once_flat(tmp_path):
    store = _archived_holder(tmp_path)
    store.record_fill("usdt", side="SELL", qty=0.1, price=60_000.0, fee=0.0, order_id="x1", trade_id="t2",
                      ts=T_MARK + timedelta(minutes=2))
    assert portfolio_gate.fund_holdings(store, atr_pct=_atr) == ()
    from sleeve_fund import open_risk

    assert open_risk.book_open_risk(store, _atr) == [] and open_risk.account_book(store, "x", 0.0, _atr)[1] == 0


def test_d_the_gates_holdings_are_the_dashboard_books_holders(tmp_path):
    store = _archived_holder(tmp_path)
    store.create_sleeve(name="flat", strategy="ping_pong", instrument="ETH/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=5_000, venue="BINANCE")
    earlier = store.previous_book()  # what app.current_sleeves leaves out
    dashboard = {s.name for s in store.sleeves() if s.name not in earlier
                 and abs(store.journal_book(s.name, s.starting_balance)["qty"]) > 1e-12}
    gate = {_holder_of(store, h) for h in portfolio_gate.fund_holdings(store, atr_pct=_atr)}
    assert gate == dashboard == {"usdt"}


def test_e_the_halt_and_the_funds_equity_use_the_same_set(tmp_path):
    store = _archived_holder(tmp_path)
    assert supervisor.fund_equity(store) == Decimal(10_000)  # its capital, cash moved and position at the mark
    sup = supervisor.Supervisor(store)
    sup._halt_every_strategy("test halt", T_MARK)
    assert [c["command"] for c in store.pending_commands("usdt")] == ["flatten"]


def _holder_of(store, h):
    names = [s.name for s in store.sleeves() if s.instrument.startswith(h.underlying + "/")]
    assert len(names) == 1
    return names[0]
