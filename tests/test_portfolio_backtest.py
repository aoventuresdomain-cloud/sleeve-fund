"""P2-7 portfolio backtest, the pieces that stand without the gate's core (P2-2): venue clones, the per-close batch at
close + 1 ns in a fixed strategy order, and the one fill-cost hook. From the P2-7a spike (qd/p2-7a-spike-note.md)."""

import ast
import inspect
from decimal import Decimal as D
from types import SimpleNamespace

import pandas as pd
import pytest
from nautilus_trader.backtest import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig, LoggerConfig, StrategyConfig
from nautilus_trader.model import AccountType, BarType, Currency, Money, OmsType, OrderSide, Quantity, TraderId, Venue
from nautilus_trader.trading import Strategy

from sleeve_fund.data import bar_type_for, synthetic_ohlcv, to_bars
from sleeve_fund.instruments import FeeSchedule, perpetual, spot_pair
from sleeve_fund.research import portfolio
from sleeve_fund.research.portfolio import (
    BATCH_DELAY_NS,
    CloseBatch,
    Pending,
    clone_instrument,
    clone_venue,
    fill_price,
    strategy_order,
)

FEES = FeeSchedule(D("0.001"), D("0.002"))


# --- clones and the fill-cost hook ------------------------------------------------------------------------------

@pytest.mark.parametrize("make", [lambda: spot_pair("BTC", "USD", FEES, Venue("KRAKEN")),
                                  lambda: perpetual("BTC", "USD", FEES, Venue("KRAKEN"), symbol="PF_XBTUSD")])
def test_a_clone_is_the_same_instrument_on_its_own_venue(make):
    inst = make()
    venue = clone_venue(inst.id.venue, 2)
    c = clone_instrument(inst, venue)
    assert str(venue) == "KRAKEN_P2" and c.id.venue == venue and c.id.symbol == inst.id.symbol
    a, b = type(inst).to_dict(inst), type(c).to_dict(c)
    assert {k: v for k, v in a.items() if k != "id"} == {k: v for k, v in b.items() if k != "id"}


def test_the_expected_fill_is_the_close_plus_half_the_spread_for_the_side():
    assert fill_price(D(100), 1, 0.0005) == D("100.0500")
    assert fill_price(D(100), -1, 0.0005) == D("99.9500")
    # ACT-DRIFT (Advisor 20:05 UK): one hook, zero until the parity data says otherwise.
    assert portfolio.ACT_DRIFT_BP == 0.0
    assert fill_price(D(100), 1, 0.0005, drift_bp=2.5) == D("100.075000")
    with pytest.raises(TypeError):
        fill_price(100.0, 1, 0.0005)


def test_the_run_order_names_each_strategy_once():
    assert strategy_order(["b", "a"]) == ("b", "a")
    with pytest.raises(ValueError):
        strategy_order(["a", "a"])


def test_the_portfolio_runner_computes_no_limit_of_its_own():
    """Done when 3 (Advisor condition 2): no limit outside the limits core; nothing from risk.py beyond the profile."""
    tree = ast.parse(inspect.getsource(portfolio))
    froms = {(n.module, a.name) for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert not [f for f in froms if f[0] == "sleeve_fund.risk" and f[1] != "PortfolioProfile"], froms
    assert not [f for f in froms if f[0] in ("sleeve_fund.open_risk", "sleeve_fund.strategies")], froms


# --- the batch, on a stand-in clock --------------------------------------------------------------------------------

class _Clock:
    def __init__(self):
        self.alerts = []

    def set_time_alert_ns(self, name, ts, callback):
        self.alerts.append((name, ts, callback))


def test_a_close_is_gated_once_in_the_runs_order_whatever_order_the_intents_came_in():
    seen, sent = [], []
    batch = CloseBatch(("alpha", "beta"), gate=lambda s, i, t: seen.append((s, i, t)) or SimpleNamespace(
        approved_qty=D(0) if i == "refuse" else D(1)))
    clock = _Clock()
    for strategy, intent in (("beta", "b1"), ("alpha", "a1"), ("beta", "refuse"), ("alpha", "a2")):
        batch.post(clock, 1_000, Pending(strategy, intent, lambda i, d: sent.append(i)))
    assert [(a[0], a[1]) for a in clock.alerts] == [("portfolio-gate-1000", 1_000 + BATCH_DELAY_NS)]  # armed once
    clock.alerts[0][2](None)
    assert seen == [("alpha", "a1", 1_001), ("alpha", "a2", 1_001), ("beta", "b1", 1_001), ("beta", "refuse", 1_001)]
    assert sent == ["a1", "a2", "b1"]  # a refusal is never sent
    assert batch.passes == [(1_000, ("alpha", "beta"))] and not batch.pending
    with pytest.raises(ValueError):
        batch.post(clock, 2_000, Pending("gamma", "x", lambda i, d: None))


def test_an_intent_for_a_close_already_gated_is_refused_loudly():
    batch = CloseBatch(("alpha",), gate=lambda s, i, t: SimpleNamespace(approved_qty=D(1)))
    clock = _Clock()
    batch.post(clock, 1_000, Pending("alpha", "a1", lambda i, d: None))
    clock.alerts[0][2](None)
    with pytest.raises(ValueError, match="already been gated"):
        batch.post(clock, 1_000, Pending("alpha", "late", lambda i, d: None))
    assert len(clock.alerts) == 1 and not batch.pending  # no alert re-armed in the past, nothing queued


# --- in one engine: clones, close + 1 ns, determinism ----------------------------------------------------------------

class _LegConfig(StrategyConfig):
    # A native type: __new__ reads order_id_tag from the same kwargs, so __init__ must not forward it.
    def __init__(self, *, name: str, bar_types: tuple, order_id_tag: str):
        super().__init__()
        self.name, self.bar_types = name, bar_types


class _Leg(Strategy):
    """Wants to buy one lot on every 3rd bar and sell it on the next, on each of its instruments: through the batch."""

    def __init__(self, config):
        super().__init__(config)
        self.n, self.events = {}, []

    def on_start(self):
        for bt in self.config.bar_types:
            self.subscribe_bars(BarType.from_str(bt))

    def on_bar(self, bar):
        iid = bar.bar_type.instrument_id
        k = self.n[iid] = self.n.get(iid, 0) + 1
        self.events.append(("bar", bar.ts_event, str(iid)))
        if k % 3 in (1, 2):
            side = OrderSide.BUY if k % 3 == 1 else OrderSide.SELL
            self.batch.post(self.clock, bar.ts_event, Pending(self.config.name, (iid, side), self._send))

    def _send(self, intent, decision):
        iid, side = intent
        self.events.append(("send", self.clock.timestamp_ns(), str(iid)))
        inst = self.cache.instrument(iid)
        self.submit_order(self.order_factory.market(iid, side, Quantity(D("0.01"), inst.size_precision)))


def _run():
    engine = BacktestEngine(BacktestEngineConfig(trader_id=TraderId.from_str("PORTFOLIO-001"),
                                                 logging=LoggerConfig(bypass_logging=True)))
    usd, btc = Currency.from_str("USD"), Currency.from_str("BTC")
    spot = spot_pair("BTC", "USD", FEES, Venue("KRAKEN"))
    perp = perpetual("BTC", "USD", FEES, Venue("KRAKEN"), symbol="PF_XBTUSD")
    legs = {"alpha": [clone_instrument(spot, clone_venue("KRAKEN", 1))],
            "beta": [clone_instrument(spot, clone_venue("KRAKEN", 2)),
                     clone_instrument(perp, clone_venue("KRAKEN", 3))]}
    prices = synthetic_ohlcv(days=30, seed=1)
    batch = CloseBatch(strategy_order(legs), gate=lambda s, i, t: SimpleNamespace(approved_qty=D(1)))
    strategies = {}
    for n, (name, insts) in enumerate(legs.items()):
        for inst in insts:
            margin = isinstance(inst, type(perp))
            engine.add_venue(venue=inst.id.venue, oms_type=OmsType.NETTING,
                             account_type=AccountType.MARGIN if margin else AccountType.CASH, base_currency=None,
                             starting_balances=[Money(10_000, usd)] + ([] if margin else [Money(0, btc)]),
                             default_leverage=D(3) if margin else None)
            engine.add_instrument(inst)
        bts = tuple(str(bar_type_for(inst, 1440)) for inst in insts)
        s = _Leg(_LegConfig(name=name, bar_types=bts, order_id_tag=f"00{n + 1}"))
        s.batch = batch
        strategies[name] = s
        engine.add_strategy(s)
    for i in range(0, len(prices), 7):  # in slices, streaming, as the runner feeds a long run
        for insts in legs.values():
            for inst in insts:
                engine.add_data(to_bars(prices.iloc[i:i + 7], inst, bar_type_for(inst, 1440)))
        engine.run(streaming=True)
        engine.clear_data()
    engine.end()
    fills = engine.generate_order_fills_report()
    out = {"passes": list(batch.passes), "logs": {k: list(s.events) for k, s in strategies.items()},
           "fills": [(str(r.instrument_id), str(r.side), str(r.avg_px), int(pd.Timestamp(r.ts_last).value))
                     for r in fills.itertuples()],
           "accounts": {str(v): str(engine.cache.account_for_venue(v).id)
                        for v in (clone_venue("KRAKEN", k) for k in (1, 2, 3))},
           "closes": {int(t.value): round(float(c), 2) for t, c in prices["close"].items()}}
    engine.dispose()
    return out


@pytest.fixture(scope="module")
def runs():
    return _run(), _run()


def test_each_strategy_trades_its_own_clone_with_its_own_account(runs):
    r, _ = runs
    assert len(set(r["accounts"].values())) == 3
    venues = {f[0].split(".")[-1] for f in r["fills"]}
    assert venues == {"KRAKEN_P1", "KRAKEN_P2", "KRAKEN_P3"}  # beta's two legs: cash and margin


def test_the_gate_pass_runs_at_close_plus_one_nanosecond_after_every_bar_of_that_close(runs):
    """The set_time_alert_ns pin (P2-7a trap 1): every order goes out at exactly close + 1 ns, after every strategy has
    seen its bar for that close, and fills at that bar's close."""
    r, _ = runs
    bars = {ts for log in r["logs"].values() for kind, ts, _ in log if kind == "bar"}
    sends = [ts for log in r["logs"].values() for kind, ts, _ in log if kind == "send"]
    assert sends and all(ts - BATCH_DELAY_NS in bars for ts in sends)
    assert all(len(seen) == 2 and seen == ("alpha", "beta") for _, seen in r["passes"])
    assert all(abs(float(px) - r["closes"][ts - BATCH_DELAY_NS]) < 0.011 for _, _, px, ts in r["fills"])


def test_two_runs_are_identical(runs):
    a, b = runs
    assert a["passes"] == b["passes"] and a["fills"] == b["fills"] and a["logs"] == b["logs"]


# --- M2: the entry deferral hook in strategies.base -----------------------------------------------------------------

class _Gate:
    """A stand-in for P2-2's gate (M3): approves each entry at `share` of its quantity, keeping what it saw."""

    def __init__(self, share: D = D(1)):
        self.share, self.seen = share, []

    def __call__(self, strategy, intent, ts):
        self.seen.append((strategy, intent["side"], intent["qty"], ts))
        return SimpleNamespace(approved_qty=intent["qty"] * self.share)


def _portfolio_run(model: str, params: dict, gate: _Gate | None):
    """run_backtest of `model`, joined to a one-strategy portfolio batch with `gate` (None: an ordinary run)."""
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.strategies import REGISTRY

    cls, config_cls = REGISTRY[model]
    batch = CloseBatch(order=("a",), gate=gate) if gate is not None else None

    class Joined(cls):
        def __init__(self, config):
            super().__init__(config)
            if batch is not None:
                self.join_portfolio(batch, "a")

    REGISTRY["portfolio_probe"] = (Joined, config_cls)
    try:
        inst = spot_pair("BTC", "USD", FEES, Venue("KRAKEN"))
        result = run_backtest("portfolio_probe", synthetic_ohlcv(days=240, seed=3), inst, params=params)
    finally:
        del REGISTRY["portfolio_probe"]
    return result, batch


def _entries(result):
    return [(o, r) for o, r in result.fills.iterrows() if result.decisions[o]["intent"] == "entry"]


@pytest.mark.parametrize("params", [dict(), dict(market="perp", allow_short=True)], ids=["spot", "perp"])
def test_an_approve_all_gate_trades_exactly_as_the_ordinary_run(params):
    """M2 parity: every entry waits for the gate pass at the decision + 1 ns and fills at the same price and size."""
    plain, _ = _portfolio_run("rsi_cross", params, None)
    gate = _Gate()
    joined, batch = _portfolio_run("rsi_cross", params, gate)
    assert _entries(plain), "the model must trade for this to test anything"
    cols = ["side", "filled_qty", "avg_px"]
    assert joined.fills[cols].reset_index(drop=True).equals(plain.fills[cols].reset_index(drop=True))
    assert len(gate.seen) == len(_entries(plain))  # every entry went through the gate, and only entries
    assert all(ts == close + 1 for (close, _), (_, _, _, ts) in zip(batch.passes, gate.seen))


def test_a_gate_that_trims_sends_only_what_it_approved_and_says_so():
    gate = _Gate(D("0.5"))
    joined, _ = _portfolio_run("rsi_cross", {}, gate)
    for (oid, row), (_, _, asked, _) in zip(_entries(joined), gate.seen):
        assert D(str(row["filled_qty"])) <= asked / 2
        assert joined.decisions[oid]["signal"]["portfolio_trimmed_from"] == str(asked)


def test_a_gate_that_refuses_sends_no_entry_and_no_exit():
    joined, _ = _portfolio_run("rsi_cross", {}, _Gate(D(0)))
    assert joined.fills is None or joined.fills.empty
