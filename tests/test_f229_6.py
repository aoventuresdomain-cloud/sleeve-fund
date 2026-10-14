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
from sleeve_fund.store import Store, sleeve_archive_t


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


def _holder(tmp_path, side="BUY", db="h") -> Store:
    """usdt (BTC perp, balanced: 2x) entered 0.1 at 60,000 with a 1% stop and marked, still running."""
    store = Store(f"sqlite:///{tmp_path}/{db}.db", portfolio_gate=True)
    store.create_sleeve(name="usdt", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, venue="BINANCE", params={"market": "perp"})
    store.record_order("usdt", order_id="e1", side=side, qty=0.1, intent="entry", reason="r",
                       signal={"stop_frac": 0.01}, ts=T_MARK)
    store.record_fill("usdt", side=side, qty=0.1, price=60_000.0, fee=0.0, order_id="e1", trade_id="t1", ts=T_MARK)
    qty = 0.1 if side == "BUY" else -0.1
    store.record_equity("usdt", equity=10_000.0, cash=10_000.0 - qty * 60_000.0, qty=qty, price=60_000.0,
                        benchmark=10_000.0, ts=T_MARK + timedelta(minutes=1))
    return store


def _put_away_holding(store) -> Store:
    store.set_desired_state("usdt", "stopped")
    store.archive("usdt")
    return store


def _archived_holder(tmp_path, side="BUY") -> Store:
    """The same position, then stopped and archived still holding."""
    return _put_away_holding(_holder(tmp_path, side))


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
    store.record_equity("usdt", equity=10_000.0, cash=10_000.0, qty=0.0, price=60_000.0, benchmark=10_000.0,
                        ts=T_MARK + timedelta(minutes=3))  # its exit's process marks it flat
    assert portfolio_gate.fund_holdings(store, atr_pct=_atr) == ()
    from sleeve_fund import open_risk

    assert open_risk.book_open_risk(store, _atr) == []
    # Advisor C3: archived flat in the current book, its equity stays in the book at no risk.
    assert open_risk.account_book(store, "x", 0.0, _atr)[:2] == (10_000.0, 0)


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


@pytest.mark.parametrize("side, sign", [("BUY", 1), ("SELL", -1)])
def test_a1_an_archived_holding_counts_as_the_same_position_held_by_a_running_strategy(tmp_path, side, sign):
    """HoQA A1: long and short, the archived Strategy's figures equal the running one's."""
    from sleeve_fund import open_risk

    running = _holder(tmp_path, side, db="run")
    archived = _archived_holder(tmp_path, side)
    assert running.sleeve("usdt").status != "archived" and "usdt" in archived.archived()
    run_h, arch_h = (portfolio_gate.fund_holdings(st, atr_pct=_atr) for st in (running, archived))
    assert run_h == arch_h and len(arch_h) == 1
    book = Book(Decimal(10_000), arch_h)
    assert (book.gross(), book.net("BTC"), book.margin(), book.open_risk()) == (6_000, sign * 6_000, 3_000, 60)
    assert open_risk.book_open_risk(running, _atr) == open_risk.book_open_risk(archived, _atr)
    assert open_risk.account_book(running, "x", 0.0, _atr) == open_risk.account_book(archived, "x", 0.0, _atr)


def test_a2_an_entry_that_breaches_open_risk_only_with_that_holding_is_trimmed(tmp_path):
    """HoQA A2, the open-risk cell: 430 elsewhere plus a 50 entry is inside 5% of 10,000; the holding's 60 is not."""
    store = _archived_holder(tmp_path)
    others = (Holding("ETH", Decimal(1_000), Decimal(100), Decimal(430)),)
    entry = Intent("XRP", 1, Decimal(1), Decimal(100), Decimal(20), Decimal(50), Decimal("0.001"), Decimal("0.001"))
    held = portfolio_gate.fund_holdings(store, atr_pct=_atr)
    without = decide(Book(Decimal(10_000), others), entry, PORTFOLIO)
    counted = decide(Book(Decimal(10_000), (*held, *others)), entry, PORTFOLIO)
    assert without.outcome == "approved"
    assert counted.outcome == "trimmed" and counted.limit_hit == "open_risk" and counted.approved_qty == Decimal("0.2")


def test_a3_a_flat_archived_run_put_away_by_a_reset_adds_nothing(tmp_path):
    """HoQA A3: a reset's run is archived and flat, with its reset_runs row left behind; it adds nothing."""
    from sleeve_fund import open_risk

    store = _holder(tmp_path)
    store.record_fill("usdt", side="SELL", qty=0.1, price=60_000.0, fee=0.0, order_id="x1", trade_id="t2",
                      ts=T_MARK + timedelta(minutes=2))
    store.set_desired_state("usdt", "stopped")
    store.request_reset("usdt", "Test finished")
    run = store.split_run(store.pending_reset("usdt"))
    assert run in store.reset_runs() and run in store.archived()
    assert portfolio_gate.fund_holdings(store, atr_pct=_atr) == ()
    assert open_risk.book_open_risk(store, _atr) == [] and open_risk.account_book(store, "x", 0.0, _atr)[1] == 0


# --- F229-6-a (Advisor MUST, HoE 14 Oct 01:05 UK, HoQA cells C1-C3): the interim 5% check's readers (open_risk's
# account_equity, account_book and book_open_risk) and the Open risk tile read the same set as the gate.

def _perp(store, name, inst, venue, qty, *, archive=False):
    """A perp Strategy on the default "paper" account, holding qty at 60,000 with a 1% stop (0: flat), marked."""
    store.create_sleeve(name=name, strategy="ping_pong", instrument=inst, bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, venue=venue, params={"market": "perp"})
    if qty:
        side = "BUY" if qty > 0 else "SELL"
        store.record_order(name, order_id=f"e-{name}", side=side, qty=abs(qty), intent="entry", reason="r",
                           signal={"stop_frac": 0.01}, ts=T_MARK)
        store.record_fill(name, side=side, qty=abs(qty), price=60_000.0, fee=0.0, order_id=f"e-{name}",
                          trade_id=f"t-{name}", ts=T_MARK)
    store.record_equity(name, equity=10_000.0, cash=10_000.0 - qty * 60_000.0, qty=qty, price=60_000.0,
                        benchmark=10_000.0, ts=T_MARK + timedelta(minutes=1))
    if archive:
        store.set_desired_state(name, "stopped")
        store.archive(name)
        # archive() stamps the wall clock: pin it into this fixture's timeline, before any fixed-time clean slate.
        with store.engine.begin() as c:
            c.execute(sleeve_archive_t.update().where(sleeve_archive_t.c.sleeve == name)
                      .values(archived_at=T_MARK + timedelta(minutes=1)))


def _fund(tmp_path, db, test_qty=None):
    """Two fund Strategies on "paper" (one archived still holding, C3) and a flat one about to enter; with test_qty,
    a USD test Strategy on "paper" too, outside the fund's book."""
    store = Store(f"sqlite:///{tmp_path}/{db}.db", portfolio_gate=True)
    _perp(store, "fund", "BTC/USDT", "BINANCE", 0.1)
    _perp(store, "held", "ETH/USDT", "BINANCE", -0.1, archive=True)
    _perp(store, "entry", "SOL/USDT", "BINANCE", 0)
    if test_qty is not None:
        _perp(store, "usd-test", "BTC/USD", "KRAKEN", test_qty)
        assert supervisor.outside_fund_book(store, store.sleeve("usd-test"))
    return store


@pytest.mark.parametrize("test_qty", [0, 0.1])
def test_c1_the_open_risk_tiles_percent_equals_the_gates_without_the_test_strategy(tmp_path, test_qty):
    from sleeve_fund import open_risk
    from sleeve_fund.dashboard import trading

    store = _fund(tmp_path, "c1", test_qty)
    rows = open_risk.book_open_risk(store, _atr)
    assert [r["sleeve"] for r in rows] == ["fund", "held"]  # the archived holder counts (C3), the test one doesn't
    book = open_risk.account_equity(store, "paper")
    assert book == 30_000  # fund, held and entry: the test Strategy's equity is not in it
    gate_book, gate_risk, _ = open_risk.account_book(store, "entry", 10_000.0, _atr)
    positions = [{"sleeve": n, "perp_limit": True, "margin": 0.0} for n in ("fund", "held", "usd-test")]
    tile = trading.open_risk(positions, store)
    assert tile["book"] == gate_book == book
    assert tile["open_risk"] / tile["book"] == pytest.approx(gate_risk / gate_book)
    assert tile["left_out"] == ["usd-test"]  # named on the tile as not counted


def test_c2_the_interim_5pct_check_decides_the_same_with_or_without_the_test_strategy(tmp_path):
    from sleeve_fund import open_risk

    without = open_risk.account_book(_fund(tmp_path, "c2a"), "entry", 10_000.0, _atr)
    holding = open_risk.account_book(_fund(tmp_path, "c2b", 0.1), "entry", 10_000.0, _atr)
    flat = open_risk.account_book(_fund(tmp_path, "c2c", 0), "entry", 10_000.0, _atr)
    assert without == holding == flat
    book, risk, _ = without
    entry = 0.05 * book - risk + 1  # just over the limit
    assert open_risk.check_entry(book, risk, entry) is not None
    assert open_risk.check_entry(book, risk, entry - 2) is None


def test_c3_an_archived_holder_counts_in_all_three_readers_until_flat(tmp_path):
    from sleeve_fund import open_risk

    store = _fund(tmp_path, "c3", 0.1)
    before = open_risk.account_book(store, "entry", 10_000.0, _atr)
    assert before[1] == pytest.approx(60.0 + 60.0)  # fund's and held's 1% stops
    store.record_fill("held", side="BUY", qty=0.1, price=60_000.0, fee=0.0, order_id="x-held", trade_id="x-held",
                      ts=T_MARK + timedelta(minutes=2))
    store.record_equity("held", equity=10_000.0, cash=10_000.0, qty=0.0, price=60_000.0, benchmark=10_000.0,
                        ts=T_MARK + timedelta(minutes=3))
    assert [r["sleeve"] for r in open_risk.book_open_risk(store, _atr)] == ["fund"]
    assert open_risk.account_equity(store, "paper") == 30_000  # Advisor C3: held's equity stays in, at no risk
    assert open_risk.account_book(store, "entry", 10_000.0, _atr)[1] == pytest.approx(60.0)


def test_c3b_a_strategy_an_earlier_book_put_away_is_out_of_both(tmp_path):
    """QT3 C3b: archived flat before a clean slate, it is the earlier book's: out of the equity and the risk."""
    from sleeve_fund import open_risk

    store = _fund(tmp_path, "c3b")
    store.record_fill("held", side="BUY", qty=0.1, price=60_000.0, fee=0.0, order_id="x-held", trade_id="x-held",
                      ts=T_MARK + timedelta(minutes=2))
    store.record_equity("held", equity=10_000.0, cash=10_000.0, qty=0.0, price=60_000.0, benchmark=10_000.0,
                        ts=T_MARK + timedelta(minutes=3))
    assert open_risk.account_equity(store, "paper") == 30_000  # in the current book
    store.decide("PM", "clear", "Clean slate")  # a new book starts after held was put away
    assert "held" in store.previous_book()
    # Advisor C3b, option B: out of the holdings and the risk, its cash still in the book, as fund_equity counts it.
    assert open_risk.account_equity(store, "paper") == 30_000
    assert [r["sleeve"] for r in open_risk.book_open_risk(store, _atr)] == ["fund"]
    assert open_risk.account_book(store, "entry", 10_000.0, _atr)[:2] == (30_000.0, pytest.approx(60.0))
    # Advisor C3b, option B: the tile measures against the gate's book, the fund's equity, put-away cash included.
    from sleeve_fund.dashboard import trading

    tile = trading.open_risk([{"sleeve": "fund", "perp_limit": True, "margin": 0.0}], store)
    assert tile["book"] == float(supervisor.fund_equity(store)) == open_risk.account_equity(store, "paper") == 30_000
    assert tile["open_risk"] / tile["book"] == pytest.approx(60.0 / 30_000)  # the tile's % is the interim check's
    assert [h.notional for h in portfolio_gate.fund_holdings(store, atr_pct=_atr)] == [Decimal(6_000)]


def test_c3c_a_clean_slate_and_an_archived_holder_going_flat_after_it_fire_no_halt_or_pause(tmp_path):
    """Advisor C3b (B): the halt, the pause and their references stay on the fund's equity, so neither the slate nor
    a holder it left going flat later moves the book."""
    store = _fund(tmp_path, "c3c")
    sup = supervisor.Supervisor(store)
    sup.mark_portfolio(T_MARK + timedelta(minutes=5))
    before = supervisor.fund_equity(store)
    store.decide("PM", "clear", "Clean slate", ts=T_MARK + timedelta(minutes=6))
    sup.mark_portfolio(T_MARK + timedelta(minutes=7))
    store.record_fill("held", side="BUY", qty=0.1, price=60_000.0, fee=0.0, order_id="x-held", trade_id="x-held",
                      ts=T_MARK + timedelta(minutes=8))
    store.record_equity("held", equity=10_000.0, cash=10_000.0, qty=0.0, price=60_000.0, benchmark=10_000.0,
                        ts=T_MARK + timedelta(minutes=8))
    assert "held" in store.previous_book()  # the earlier book's now
    sup.mark_portfolio(T_MARK + timedelta(minutes=9))
    st = sup.ledger.state()
    assert supervisor.fund_equity(store) == before
    assert st.halted is None and st.paused is None and st.equity == before
