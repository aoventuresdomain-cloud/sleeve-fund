"""B7 (scripts/b7_replay_parity.py, read-only): paper's gate decisions over the B3 window against the portfolio
backtest's, and the Strategy page's figures against the journal. A journal in memory, stored bars made up here, no
server and no network."""

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal as D
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from nautilus_trader.model import Venue

from sleeve_fund.instruments import FeeSchedule, perpetual
from sleeve_fund.paper.portfolio_gate import PaperGate
from sleeve_fund.research.portfolio import fill_price
from sleeve_fund.store import Store

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_p21b_wiring_qa import portfolio_pass

NAME = "ping-pong-ls-binance"
T0 = datetime(2026, 10, 14, 9, 0, 3, tzinfo=UTC)  # the round trip's "since"
BAR = datetime(2026, 10, 14, 9, 1, tzinfo=UTC)  # paper's first decided bar
CLOSE = 60_000.0
FEES = FeeSchedule(D("0.0002"), D("0.00055"))
INST = perpetual("BTC", "USDT", FEES, Venue("BINANCE"), "BTCUSDT-PERP", price_precision=1)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b7 = _load("b7_replay_parity")


def _ns(t: datetime) -> int:
    return int(t.timestamp()) * 1_000_000_000


def _gate(store, intent_id: str, bar: datetime, qty=D("0.110"), with_bar=True, order_id="e1", fill=60_015.0) -> None:
    """One gated entry as paper sends it: the gate's check over the journal, its order row linked to the reservation
    (with the deciding bar in its signal, unless sent outside a bar decision), and its fill."""
    price = fill_price(D(str(CLOSE)), 1, 0.0002)
    intent = {"side": 1, "qty": qty, "close": D(str(CLOSE)), "price": price, "stop_frac": None, "what": "entry",
              "step": D("0.001"), "min_qty": D("0.001"), "instrument": "BTC", "leverage": 2.0, "atr_pct": 0.03,
              "taker": D(0),
              "intent_id": intent_id, **({"bar": _ns(bar)} if with_bar else {})}
    gated = PaperGate(store)(NAME, intent, _ns(bar) + 1)
    signal = {"close": CLOSE, **({"bar": _ns(bar)} if with_bar else {})}
    store.record_order(NAME, order_id=order_id, side="BUY", qty=float(qty), intent="entry", reason="start of the cycle",
                       signal=signal, ts=bar + timedelta(seconds=1), reservation=gated.checked.reservation)
    store.book_fill(NAME, side="BUY", qty=float(qty), price=fill, fee=round(float(qty) * fill * 0.00055, 8),
                    order_id=order_id, trade_id=f"t-{order_id}", ts=bar + timedelta(seconds=2))


def _journal(qty=D("0.110"), equity=50_000, mark_after=True) -> Store:
    """B3 as paper journals it: a gated entry on paper's first bar (approved in full), its fill, the flatten's fill a
    minute later, and the Strategy's last mark (after the exit, unless not mark_after)."""
    store = Store.in_memory(portfolio_gate=True)
    store.create_sleeve(name=NAME, strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000.0, params={"market": "perp", "rise": 0.01, "dip": 0.005,
                                                           "allow_short": True},
                        venue="BINANCE", risk_profile="balanced")
    portfolio_pass(store, T0, D(equity))
    _gate(store, "i1", BAR, qty)
    store.record_order(NAME, order_id="x1", side="SELL", qty=float(qty), intent="pm_flatten", reason="flatten",
                       ts=BAR + timedelta(minutes=1))
    store.book_fill(NAME, side="SELL", qty=float(qty), price=60_030.0, fee=round(float(qty) * 60_030 * 0.00055, 8),
                    order_id="x1", trade_id="t2", ts=BAR + timedelta(minutes=1, seconds=1))
    cash = store.journal_book(NAME, 10_000.0)["cash"]
    store.record_equity(NAME, equity=cash, cash=cash, qty=0.0, price=60_030.0, benchmark=10_000.0,
                        ts=BAR + timedelta(minutes=1, seconds=5 if mark_after else -30))
    return store


class _History:
    """Stored 1-minute bars stamped at their close, from `first` to `last`, a gentle walk around CLOSE."""

    def __init__(self, first: datetime, last: datetime, seed: int = 1):
        idx = pd.date_range(first, last, freq="1min", tz="UTC")
        rng = np.random.default_rng(seed)
        close = CLOSE * np.exp(np.cumsum(rng.normal(0, 0.0003, len(idx))))
        open_ = np.concatenate([[CLOSE], close[:-1]])
        self.df = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.0002,
                                "low": np.minimum(open_, close) * 0.9998, "close": close, "volume": 50.0},
                               index=pd.DatetimeIndex(idx, name="timestamp"))
        self.asked = []

    def read(self, venue, pair, minutes, start=None, end=None):
        self.asked.append((venue, pair, minutes))
        assert minutes == 1
        return self.df[(self.df.index >= start) & (self.df.index <= end)]


def _backtest(*decisions):
    return lambda specs, unallocated: SimpleNamespace(decisions=list(decisions), specs=specs, unallocated=unallocated)


def _decided(bar=BAR, outcome="approved", requested=D("0.110"), approved=D("0.110"), limit=None):
    return {"strategy": NAME, "bar_ts": bar, "outcome": outcome, "requested": requested, "approved": approved,
            "limit": limit}


HISTORY = _History(BAR - timedelta(days=16), BAR + timedelta(minutes=5))


# --- replay parity ---------------------------------------------------------------------------------------------------

def test_parity_passes_when_the_backtest_decides_as_paper_did_and_is_fed_what_paper_had(capfd):
    store = _journal()
    fed = {}

    def run(specs, unallocated):
        fed.update(spec=specs[0], unallocated=unallocated)
        return SimpleNamespace(decisions=[_decided()])

    assert b7.parity(store, NAME, T0, history=HISTORY, instrument=INST, run=run) == 0
    out = capfd.readouterr().out
    assert "decisions: paper 1, backtest 1, matched 1, mismatched 0" in out and "B7 replay parity: PASS" in out
    spec = fed["spec"]
    # From paper's first decided bar to one bar past the flatten's minute (the backtest decides a bar on the next
    # one); the warm-up before it; the Strategy's own cash and risk profile; the rest of the fund's equity as
    # unallocated cash.
    assert spec.prices.index[0] == pd.Timestamp(BAR)
    assert spec.prices.index[-1] == pd.Timestamp(BAR + timedelta(minutes=2))
    assert spec.warmup_prices.index[-1] < pd.Timestamp(BAR)
    assert spec.prices.index[0] - spec.warmup_prices.index[0] >= pd.Timedelta(days=14)
    assert (spec.capital, fed["unallocated"], spec.risk_profile) == (10_000.0, 40_000.0, "balanced")
    assert (spec.bar_minutes, spec.exec_minutes, spec.fees) == (1, 1, FEES)
    drift = next(x for x in out.splitlines() if x.startswith("fill drift"))
    assert "1 opening fill(s)" in drift and "trigger not met" in drift


def test_parity_replays_a_round_trip_inside_one_minute_one_bar_past_it(capfd):
    """B7 on cd4e589 (14 Oct 16:56 UTC): entry and flatten in one minute, so `until` truncates to the entry's own bar.
    The replay still runs to the next bar, as the backtest decides a bar only once a later one arrives."""
    store = _journal()
    fed = {}

    def run(specs, unallocated):
        fed.update(spec=specs[0])
        return SimpleNamespace(decisions=[_decided(), _decided(bar=BAR + timedelta(minutes=1))])

    until = BAR + timedelta(seconds=9)
    assert b7.parity(store, NAME, T0, until, history=HISTORY, instrument=INST, run=run) == 0
    out = capfd.readouterr().out
    # The next bar's decision is outside [start, end]: neither compared nor failed.
    assert "decisions: paper 1, backtest 1, matched 1, mismatched 0" in out and "(run to 09:02)" in out
    assert fed["spec"].prices.index[-1] == pd.Timestamp(BAR + timedelta(minutes=1))


def test_the_real_portfolio_backtest_decides_nothing_on_a_window_of_one_bar():
    """Why the replay runs one bar past: the gate pass for a bar runs at its close + BATCH_DELAY_NS, which the engine
    never reaches when that bar is its last."""
    store = _journal()
    one, why = b7.replay(store, store.sleeve(NAME), BAR, BAR, D(10_000), D(50_000), HISTORY, INST)
    assert why is None and one.decisions == []
    two, _ = b7.replay(store, store.sleeve(NAME), BAR, BAR + timedelta(minutes=1), D(10_000), D(50_000), HISTORY, INST)
    assert [d["bar_ts"] for d in two.decisions] == [BAR]


def test_parity_waits_for_the_bar_after_the_last_decided_one_then_asks_for_a_re_run(capfd, monkeypatch):
    slept = []
    monkeypatch.setattr(b7.time, "sleep", slept.append)
    clock = iter(range(0, 1000, 10))
    monkeypatch.setattr(b7.time, "monotonic", lambda: next(clock))
    history = _History(BAR - timedelta(days=16), BAR)  # the bar closing at 09:02 hasn't reached the store
    assert b7.parity(_journal(), NAME, T0, BAR + timedelta(seconds=9), history=history, instrument=INST,
                     run=lambda *a, **k: pytest.fail("ran"), wait=60.0) == 1
    out = capfd.readouterr().out
    assert "no 1-minute bar closing at 09:02:00 UTC yet" in out and "re-run B7 alone (check=replay)" in out
    assert slept and set(slept) == {15}


@pytest.mark.parametrize("backtest,what", [
    ((_decided(approved=D("0.100"), outcome="trimmed", limit="open_risk"),), "outcome, limit, approved"),
    ((_decided(requested=D("0.150"), approved=D("0.150")),), "requested, approved"),
    ((), "paper only"),
    ((_decided(), _decided(bar=BAR + timedelta(minutes=1))), "backtest only"),
], ids=["trimmed", "sized-differently", "backtest-never-entered", "backtest-entered-again"])
def test_parity_fails_on_any_decision_that_differs_and_prints_both_sides(capfd, backtest, what):
    assert b7.parity(_journal(), NAME, T0, history=HISTORY, instrument=INST, run=_backtest(*backtest)) == 1
    out = capfd.readouterr().out
    assert what in out and "B7 replay parity: FAIL (1 decision(s) differ)" in out


def test_parity_lists_an_entry_sent_outside_a_bar_decision_without_comparing_it(capfd):
    store = _journal()
    # A resumed entry after the flatten: its bar_ts is only the check time, and its order row has no signal.bar.
    portfolio_pass(store, BAR + timedelta(seconds=89), D(50_000))
    _gate(store, "i2", BAR + timedelta(seconds=90), D("0.010"), with_bar=False, order_id="e2")
    until = BAR + timedelta(minutes=2)
    assert b7.parity(store, NAME, T0, until, history=HISTORY, instrument=INST, run=_backtest(_decided())) == 0
    out = capfd.readouterr().out
    assert "decisions: paper 1, backtest 1, matched 1" in out
    assert "not decided on a bar (listed, not compared): 09:02:30 approved 0.01" in out
    assert "2 opening fill(s)" in out


def test_parity_lists_a_strategy_outside_the_fund_s_book_without_comparing_it(capfd, monkeypatch):
    # F277-QA1: a test-account Strategy outside the book (F229-6) reaches compare() as `outside`, so a backtest that
    # decided differently is listed apart, not failed.
    import sleeve_fund.supervisor as supervisor

    monkeypatch.setattr(supervisor, "outside_fund_book", lambda store, s: s.name == NAME)
    run = _backtest(_decided(requested=D("0.150"), approved=D("0.150")))
    assert b7.parity(_journal(), NAME, T0, history=HISTORY, instrument=INST, run=run) == 0
    out = capfd.readouterr().out
    assert "decisions: paper 0, backtest 0, matched 0" in out
    assert "outside the fund's book (listed, not compared): paper 1, backtest 1" in out
    monkeypatch.setattr(supervisor, "outside_fund_book", lambda store, s: False)  # in the book: the same run fails
    assert b7.parity(_journal(), NAME, T0, history=HISTORY, instrument=INST, run=run) == 1


@pytest.mark.parametrize("case,why", [
    ("no-decision", "no gate decision on a bar journaled in the window"),
    ("held-before", "the replay starts only from flat"),
    ("no-bars", "has no 1-minute bar closing at 09:01:00 UTC"),
])
def test_parity_fails_when_the_window_cant_be_replayed(capfd, case, why):
    store = _journal()
    since, history = T0, HISTORY
    if case == "no-decision":
        since = BAR + timedelta(minutes=2)
    elif case == "held-before":
        store.record_order(NAME, order_id="old", side="BUY", qty=0.01, intent="entry", reason="r",
                           ts=T0 - timedelta(hours=1))
        store.book_fill(NAME, side="BUY", qty=0.01, price=59_000.0, fee=0.3, order_id="old", trade_id="t0",
                        ts=T0 - timedelta(hours=1))
    else:
        history = _History(BAR + timedelta(minutes=1), BAR + timedelta(minutes=5))
    assert b7.parity(store, NAME, since, history=history, instrument=INST,
                     run=lambda *a, **k: pytest.fail("ran")) == 1
    assert why in capfd.readouterr().out


def test_parity_notes_other_strategies_holdings_the_replay_does_not_hold(capfd, monkeypatch):
    store = _journal()
    real = b7._figures

    def held(*a):
        return {k: {**v, "open_risk": D(250), "gross": D(5000)} for k, v in real(*a).items()}

    monkeypatch.setattr(b7, "_figures", held)
    assert b7.parity(store, NAME, T0, history=HISTORY, instrument=INST, run=_backtest(_decided())) == 0
    assert "PASS [other Strategies held gross 5000, open risk 250" in capfd.readouterr().out


def test_the_real_portfolio_backtest_enters_on_paper_s_first_bar_under_its_risk_profile():
    """No stand-in: run_portfolio over the stored bars as the script builds it. The cycle starts with a buy on the
    window's first bar, sized under the balanced profile's cap (33% margin at 2x), not all the Strategy's cash."""
    store = _journal()
    result, why = b7.replay(store, store.sleeve(NAME), BAR, BAR + timedelta(minutes=1), D(10_000), D(50_000),
                            HISTORY, INST)
    assert why is None
    (first, *_) = result.decisions
    assert (first["bar_ts"], first["outcome"], first["side"]) == (BAR, "approved", 1)
    notional = float(first["requested"]) * float(HISTORY.df.at[pd.Timestamp(BAR), "close"])
    assert 6_000 < notional <= 6_600, notional  # 66% of 10,000 less the fee buffer


# --- dashboard = journal ---------------------------------------------------------------------------------------------

def test_dashboard_passes_when_the_page_s_fees_trip_and_equity_are_the_journal_s(capfd):
    assert b7.dashboard(_journal(), NAME, T0) == 0
    out = capfd.readouterr().out
    assert "B3 trip(s): 1" in out and "B7 dashboard = journal: PASS" in out


def test_dashboard_fails_an_equity_marked_before_the_last_fill(capfd):
    assert b7.dashboard(_journal(mark_after=False), NAME, T0) == 1
    assert "a mark before the last fill" in capfd.readouterr().out


def test_dashboard_leaves_an_old_book_close_out_out_of_the_cash_it_compares_the_trips_with(capfd):
    """CR F-RN-5: the page pairs trips with Store.insurance, which leaves an old book close-out (RESET-NEG) out, so the
    cash the trips are held to leaves it out too; it is a transfer, never a trip's P&L. Written mid-trip, the worst case."""
    from sleeve_fund.store import CLOSE_OUT

    store = _journal()
    store.create_sleeve(name=f"{NAME}--20261006101219", strategy="ping_pong", instrument="BTC/USDT",
                        bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=0.0, params={"market": "perp"},
                        venue="BINANCE")
    assert store.close_out_run(f"{NAME}--20261006101219", NAME, D("82.51"), CLOSE_OUT, BAR + timedelta(seconds=30),
                               "operator", "book_reset_pm", "close-out")
    cash = store.journal_book(NAME, 10_000.0)["cash"]
    store.record_equity(NAME, equity=cash, cash=cash, qty=0.0, price=60_030.0, benchmark=10_000.0,
                        ts=BAR + timedelta(minutes=1, seconds=10))
    assert b7.dashboard(store, NAME, T0) == 0, capfd.readouterr().out
    out = capfd.readouterr().out
    assert "B7 dashboard = journal: PASS" in out
    assert "an old book close-out of -82.51 left out (a transfer, not a trip's)" in out
