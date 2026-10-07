"""EXPOSURE GATE: the adversarial invariant, written by QA BEFORE the build (PE2, in stop-safety + HC after #155).

The fix makes #155's `_entry_blocked` the ONE gate: a single state function returning (blocked, reason), checked when
an opening order is submitted AND when it fills, read by the engine, the supervisor and the dashboard. Entering a
blocked state cancels resting entries and unfilled remainders. EXITS ARE NEVER BLOCKED.

Rulings pinned (advisor-rulings.md; the latest entry wins):
- [R113] Entry choke point (Advisor 6 Oct ~20:56, to HoE): one _entry_blocked at submit + fill for all open/add paths;
  "Add" = any order increasing |position| (reversal open leg, band adds, rebalance increases; reductions always
  allowed). Partly filled resting entry on halt: cancel the remainder AT HALT TIME, keep the filled part with its
  stop, never flatten; the fill-time check is the backstop. Invariant test: every halt kind x every path, plus
  exits/stops always run. Coordinator relay of the Advisor's 20:52 reading: a fill that races the cancel, or a partial
  fill already done when the block starts, is KEPT with its stop placed and an incident written, never flattened.
- [R65] (5 Oct) an ENTRY can be skipped, an EXIT cannot; every degraded state blocks new entries and adds only.
- [R79] (18:17) a halt is cleared only by its own action (drawdown halt: resume; daily pause: next 00:00 UTC roll;
  liquidation: reset after liquidation); Stop/Start and restarts never clear it; Start refuses while halted naming
  the clearing action; Stop always works.
- [R77] (17:57) liquidated: halted through resume/restart until the PM's reset after liquidation.
- [R110] (U27, 20:41) an ordinary per-strategy Reset is refused while liquidated, pointing to the reset after
  liquidation.
- [R82] (O17, 18:18) a missing funding rate blocks new entries/adds on that perp (flat strategies too) once the
  15-minute check fails; the block lifts when the rate is stored AND the check passes.
- [R46] (P1-4, 16:45) a degraded candle blocks entries while it is the latest closed one.
- [R69] (P2-6, 14:55) wind-down: retired only once flat; [R79] Retire = Stop + Archive until P2-6.
- [R105] GAP-LIQ: a stop filled at or beyond the liquidation price books as a liquidation.

Marks. `xfail(strict=True, raises=AssertionError, reason="EXPOSURE-GATE")` where today's code fails; GAP-LIQ where the
set-up needs the gap through the stop to book as a liquidation; CHOKE-2 (deferred: backtest/study mode, demo mirror,
P2-1 resizes, rebalance) in TestChoke2 (every id contains "choke2", so `-k "not choke2"` runs the gate set alone).
P1-U35 (a raced remainder never unwatched): reason "CHOKE", in the gate set. A missing interface fails as
AssertionError("not built: ..."), never ImportError or AttributeError. Plain tests already hold.

Modes. "replay": a recorded paper session replayed through the paper engine (sleeve_fund.research.replay: the same
strategy class, SleeveRuntime in its live mode, the simulated venue; tests/test_replay.py is the proof that this IS
paper). "paper": the paper process's own control surfaces driven directly on the journal on a frozen clock
(SleeveRuntime start and tick, the supervisor's reconcile step, the dashboard's routes), followed by a replayed restart
where exposure is the question. Synthetic data only; no venue is called; no live keys.

Run from a checkout (needs tests/ on the path only for nothing; self-contained):
  TEST_DATABASE_URL=postgresql://postgres@localhost:5433/egx BACKTEST_ISOLATE=0 \
    PYTHONPATH=<checkout>:<checkout>/tests python -m pytest -q -p no:cacheprovider test_exposure_gate_xfails.py
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

os.environ.setdefault("BACKTEST_ISOLATE", "0")

from nautilus_trader.model import AggressorSide, Price, Quantity, QuoteTick, TradeId, TradeTick  # noqa: E402

from sleeve_fund import funding, markets  # noqa: E402
from sleeve_fund.paper.recorder import Recorder  # noqa: E402
from sleeve_fund.paper.runtime import SleeveRuntime  # noqa: E402
from sleeve_fund.research.replay import replay  # noqa: E402
from sleeve_fund.store import Store  # noqa: E402
from sleeve_fund.strategies import REGISTRY  # noqa: E402
from sleeve_fund.strategies.base import LongFlatConfig, LongFlatStrategy  # noqa: E402
from sleeve_fund.venues import binance_contract, venue  # noqa: E402

# The journal kind a reset after liquidation writes: sleeve_fund.store's (#164 60cb17d on), else trading's (#164
# e14acec), else the literal (#155 eb737bd). Main 9ae1f8a has #164 merged.
try:
    from sleeve_fund.store import LIQUIDATION_RESET

    HAS_164 = True  # #164's liquidated guards on Resume, Start and Reset (U25, U31, U27) are in
except ImportError:
    HAS_164 = False
    try:
        from sleeve_fund.dashboard.trading import LIQUIDATION_RESET
    except ImportError:
        LIQUIDATION_RESET = "liquidation_reset"

GATE = "EXPOSURE-GATE"
GAP_LIQ = "GAP-LIQ"
CHOKE2 = "CHOKE-2"


def xf(reason=GATE, condition=True):
    return pytest.mark.xfail(condition, strict=True, raises=AssertionError, reason=reason)


# #155 (eb737bd) is in: the runtime judges "liquidated" from the journal. Some cells already hold there and not on
# main (9ae1f8a); their marks are conditional on this. Others hold on main (#164 merged) and not on eb737bd: HAS_164.
HAS_155 = hasattr(SleeveRuntime, "_last_liquidation")
ON_MAIN = not HAS_155

NAME = "egx"
PAIR = "BTC/USDT"
PERP = {"market": "perp", "allow_short": True}
STOP = 0.02  # a real 2% stop: at 2x (balanced) the liquidation price is ~49% away, so well inside half of it
INFO = {"symbols": [  # Binance's BTCUSDT contract, offline (as tests/test_binance.py)
    {"symbol": "BTCUSDT", "pair": "BTCUSDT", "contractType": "PERPETUAL", "status": "TRADING", "baseAsset": "BTC",
     "quoteAsset": "USDT", "pricePrecision": 2, "quantityPrecision": 3,
     "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                 {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                 {"filterType": "MIN_NOTIONAL", "notional": "100"}]}]}
AUTH = ("pm", "test-pw")
SAME = {"origin": "http://testserver"}

# Reasons and the words a refusal must use to name them (assertion 4).
NAMES = {
    "drawdown_halt": r"drawdown",
    "daily_pause": r"daily[- ]loss|daily pause|00:00 utc",
    "liquidation_gap": r"liquidat",
    "liquidation": r"liquidat",
    "funding_missing": r"funding",
    "stale_data": r"degraded|stale|missing|unrefilled",
    "winding_down": r"wind(ing)?[- ]down",
    "retired": r"retired|archived",
    "stopped": r"stopped",
}
CLEARED_BY = {"drawdown_halt": "resume", "daily_pause": "00:00 utc", "liquidation": "reset after liquidation",
              "liquidation_gap": "reset after liquidation"}


def utc(s) -> pd.Timestamp:
    t = pd.Timestamp(s)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


# ---------------------------------------------------------------------------------------------------------------
# The test strategy: signal windows (a side per span of decision-bar closes) and resting stop entries (B3 breakouts
# would rest these; none ships yet, so D21 is latent). Entries go through the strategy's own _open (sizing, the
# stop-to-liquidation guard and every production check); a resting entry is journalled as _submit journals an order.
# ---------------------------------------------------------------------------------------------------------------

class EgxConfig(LongFlatConfig):
    def __init__(self, *, windows=(), rest=(), **kw):
        super().__init__(**kw)
        self.windows = [tuple(w) for w in windows]
        self.rest = [tuple(r) for r in rest]


class Egx(LongFlatStrategy):
    """Side s while open_ns <= the decision bar's close < close_ns, for each [open_ns, close_ns, s]; flat otherwise.
    rest: [place_ns, side, trigger, qty]: a resting STOP_MARKET entry placed on the first decision bar closing at or
    after place_ns."""

    gap_bars = None  # harness: the venue's own candles a paper process asks for after a stretch with no trades

    def __init__(self, config):
        super().__init__(config)
        self.windows, self._rest = config.windows, list(config.rest)

    def want_long(self, bar):
        return None

    def want_side(self, bar):
        for o, c, s in self.windows:
            if o <= bar.ts_event < c:
                return s
        return 0

    def on_start(self):
        if Egx.gap_bars is not None:  # as paper.node attaches it
            self.attach_gap_loader(lambda inst, bt, since, until: [b for b in Egx.gap_bars(inst, bt)
                                                                   if since <= b.ts_event <= until])
        super().on_start()

    def on_bar(self, bar):
        super().on_bar(bar)
        if str(bar.bar_type) != str(self._cfg.bar_type).split("@")[0]:
            return
        from nautilus_trader.model import OrderSide, TimeInForce
        due = [r for r in self._rest if r[0] <= bar.ts_event]
        self._rest = [r for r in self._rest if r[0] > bar.ts_event]
        for _, side, trigger, qty in due:
            o = self.order_factory.stop_market(instrument_id=self._cfg.instrument_id,
                                               order_side=OrderSide.BUY if side > 0 else OrderSide.SELL,
                                               quantity=self.instrument.make_qty(qty),
                                               trigger_price=self.instrument.make_price(trigger),
                                               time_in_force=TimeInForce.GTC)
            coid = str(o.client_order_id)
            self.decisions[coid] = {"intent": "entry", "reason": "QA resting stop entry (a B3 breakout)", "signal": {}}
            if self.runtime is not None:
                self.runtime.on_order(order_id=coid, side="BUY" if side > 0 else "SELL", qty=float(qty), intent="entry",
                                      reason="QA resting stop entry (a B3 breakout)", signal={},
                                      order_type="STOP_MARKET")
            self.submit_order(o)


# ---------------------------------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        from sleeve_fund.store import make_engine, metadata

        engine = make_engine(url)
        metadata.drop_all(engine)
        st = Store(engine=engine)
        yield st
        engine.dispose()  # one connection pool per test, closed after it
        return
    st = Store(f"sqlite:///{tmp_path}/egx.db")
    yield st
    st.engine.dispose()


@pytest.fixture(autouse=True)
def _offline(tmp_path, monkeypatch):
    """Binance's perp, offline: the contract from a fixed listing, the funding store in a temp dir with every
    settlement's rate there except the ones a test withholds, and the venue's own funding check answering from it."""
    monkeypatch.setattr(funding, "DEFAULT_ROOT", tmp_path / "funding")
    b = venue("binance")
    monkeypatch.setattr(b, "contract", lambda pair: binance_contract(pair, get_json=lambda url: INFO))
    monkeypatch.setitem(REGISTRY, "egx", (Egx, EgxConfig))
    Egx.gap_bars = None
    _rates({})
    monkeypatch.setattr(b, "funding_loader", lambda pair, since: [(int(t.timestamp() * 1000), r)
                                                                  for t, r in sorted(RATES.items())
                                                                  if int(t.timestamp() * 1000) >= since])
    yield
    Egx.gap_bars = None


RATES: dict = {}


def _rates(missing: set) -> None:
    """Every Binance settlement on 2 to 4 Oct 2025 has a stored, published rate of 0.01%, except `missing`."""
    import json

    RATES.clear()
    for t in pd.date_range("2025-10-02 00:00", "2025-10-05 00:00", freq="8h", tz="UTC"):
        if t not in missing:
            RATES[t] = 0.0001
    p = funding._path("BINANCE", PAIR)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"rates": [[int(t.timestamp() * 1000), r] for t, r in sorted(RATES.items())]}))
    funding._cache.clear()


@pytest.fixture
def client(store, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from sleeve_fund.dashboard import app as app_mod

    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    monkeypatch.setenv("TEARSHEET_DIR", str(tmp_path))
    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    monkeypatch.setattr(app_mod, "LEDGER", tmp_path / "idea_ledger.jsonl")
    return TestClient(app_mod.create_app(store))


# ---------------------------------------------------------------------------------------------------------------
# Sessions: a recorded paper session (a trade and a quote every second), replayed into the journal
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class Plan:
    """One run: what the strategy wants, the price, and what happens at which simulated time."""
    t0: pd.Timestamp
    minutes: int = 50
    windows: list = field(default_factory=list)  # (open, close, side) as Timestamps
    rest: list = field(default_factory=list)  # (place, side, trigger, qty)
    price: object = None  # seconds from t0 -> price
    holes: list = field(default_factory=list)  # (from s, to s): no trade or quote reaches the process
    trade_size: float = 1.0
    bar: str = "1-MINUTE-LAST-INTERNAL"
    params: dict = field(default_factory=dict)
    hooks: list = field(default_factory=list)  # (at, fn(rt, kw)) run on the first tick at or after `at`
    balance: float = 10_000.0
    tag: str = "s"


def _ns(t) -> int:
    return utc(t).value


def run(tmp_path, store, plan: Plan, monkeypatch) -> list[dict]:
    """Record the session and replay it through the paper engine into `store`; returns the orders it sent."""
    b = venue("binance")
    inst = b.instrument("BTC", "USDT")
    params = {**PERP, "stop_loss": STOP, **plan.params,
              "windows": [[_ns(o), _ns(c), s] for o, c, s in plan.windows],
              "rest": [[_ns(t), s, trig, q] for t, s, trig, q in plan.rest]}
    fees = markets.fees_for(params, b.fees, "BINANCE")
    path = f"{tmp_path}/{plan.tag}.jsonl.gz"
    rec = Recorder(path)
    rec.meta = {"balances": [f"{plan.balance:.2f} USDT"],
                "sleeve": {"name": NAME, "strategy": "egx", "instrument": PAIR, "bar_spec": plan.bar,
                           "starting_balance": 10_000, "risk_profile": "balanced", "params": params,
                           "max_notional": None, "maker_fee": str(fees.maker), "taker_fee": str(fees.taker),
                           "tick_seconds": 30}}
    rec.start(inst)
    price = plan.price or (lambda s: 60_000 + s * 0.01)
    t0 = plan.t0.value
    for i in range(plan.minutes * 60):
        if any(a <= i < z for a, z in plan.holes):
            continue
        px = round(price(i), 1)
        t = t0 + i * 1_000_000_000
        rec.quote(QuoteTick(inst.id, Price(px - 0.1, 1), Price(px + 0.1, 1), Quantity(5, 3), Quantity(5, 3), t,
                            t + 1000))
        rec.trade(TradeTick(inst.id, Price(px, 1), Quantity(plan.trade_size, 3),
                            AggressorSide.BUY if i % 2 else AggressorSide.SELL, TradeId(f"{plan.tag}{i}"), t + 2000,
                            t + 3000))
    rec.close()
    hooks = sorted(((utc(a), fn) for a, fn in plan.hooks), key=lambda h: h[0])
    real_tick = SleeveRuntime.tick

    def tick(self, **kw):
        while hooks and self.now() >= hooks[0][0]:
            hooks.pop(0)[1](self, kw)
        return real_tick(self, **kw)

    monkeypatch.setattr(SleeveRuntime, "tick", tick)
    if any(s.name == NAME for s in store.sleeves()):
        store.create_sleeve = lambda **kw: store.sleeve(kw["name"])  # a restart: the strategy is there already
    try:
        return replay(path, store=store)
    finally:
        store.__dict__.pop("create_sleeve", None)
        monkeypatch.setattr(SleeveRuntime, "tick", real_tick)


# ---------------------------------------------------------------------------------------------------------------
# Reading the journal
# ---------------------------------------------------------------------------------------------------------------

def _orders(store):
    return list(reversed(store.orders(NAME, limit=100_000)))


def _fills(store):
    return sorted(store.fills(NAME, limit=100_000), key=lambda f: (f["ts"], f["id"]))


def _events(store, kinds=None):
    ev = list(reversed(store.events(NAME, limit=100_000)))
    return [e for e in ev if kinds is None or e["kind"] in kinds]


def _signed(f) -> float:
    return f["qty"] if f["side"] == "BUY" else -f["qty"]


def _gross_after(store, at, until=None) -> list[tuple]:
    """Each fill at or after `at` that left |position| larger than it found it: (ts, side, qty, intent, before,
    after)."""
    intents = {o["order_id"]: o["intent"] for o in _orders(store)}
    net, grew = 0.0, []
    for f in _fills(store):
        before, net = net, net + _signed(f)
        if abs(net) < 1e-9:
            net = 0.0
        if f["ts"] >= at and (until is None or f["ts"] < until) and abs(net) > abs(before) + 1e-9:
            grew.append((f["ts"], f["side"], f["qty"], intents.get(f["order_id"], "?"), round(before, 6),
                         round(net, 6)))
    return grew


OPENING = ("entry", "rebalance", "add")


def _opened_after(store, at, until=None) -> list[tuple]:
    """Opening orders (an entry, add or rebalance) created at or after `at` (and before `until`)."""
    return [(o["ts"], o["side"], o["intent"], o["qty"], o["status"]) for o in _orders(store)
            if o["intent"] in OPENING and o["ts"] >= at and (until is None or o["ts"] < until)]


def _net(store) -> float:
    return round(sum(_signed(f) for f in _fills(store)), 9)


def assert_no_exposure_added(store, at, in_flight=(), until=None):
    """Assertion 1: no new opening order created from `at`, and no fill from `at` grew |position| (fills of an order
    with an intent in `in_flight`, sent before the block, excepted: the in-flight rule, test_the_filled_part_...)."""
    opened = _opened_after(store, at, until)
    assert not opened, f"opening order(s) created while blocked: {opened}"
    grew = [g for g in _gross_after(store, at, until) if g[3] not in in_flight]
    assert not grew, f"gross exposure grew while blocked: {grew}"


# ---------------------------------------------------------------------------------------------------------------
# The single gate (assumed interface: adapt the names, never the assertions)
# ---------------------------------------------------------------------------------------------------------------

def gate(store, rt=None, at=None) -> tuple[bool, str]:
    """The ONE gate's state function, (blocked, reason), as the engine, supervisor and dashboard read it. Looked for,
    in order: sleeve_fund.paper.runtime.entry_blocked(store, name, now) (a store-level function the supervisor and
    dashboard can call without a process); Store.entry_blocked(name, now=); SleeveRuntime.entry_blocked()."""
    from sleeve_fund.paper import runtime as rt_mod

    now = at or datetime.now(timezone.utc)
    fn = getattr(rt_mod, "entry_blocked", None)
    if callable(fn):
        out = fn(store, NAME, now)
    elif callable(getattr(store, "entry_blocked", None)):
        out = store.entry_blocked(NAME, now=now)
    elif rt is not None and callable(getattr(rt, "entry_blocked", None)):
        out = rt.entry_blocked()
    else:
        raise AssertionError("not built: the single exposure gate's state function (blocked, reason): "
                             "sleeve_fund.paper.runtime.entry_blocked(store, name, now)")
    assert isinstance(out, tuple) and len(out) == 2, f"the gate must return (blocked, reason), got {out!r}"
    return bool(out[0]), str(out[1] or "")


# ---------------------------------------------------------------------------------------------------------------
# Blocked reasons: how each block starts. T0 is the session start; the block starts at T0 + BLOCK minutes, except
# liquidation (journalled before the session, so in place at T0) and stale_data (the degraded bar's close).
# ---------------------------------------------------------------------------------------------------------------

STD_T0 = utc("2025-10-03 09:50")  # no Binance settlement (00/08/16 UTC) inside the session
FUND_T0 = utc("2025-10-03 07:50")  # the 08:00 settlement's rate never arrives; the 15-minute check fails at 08:15
FUND_S = utc("2025-10-03 08:00")
BLOCK, PLACE, SIGNAL, CROSS = 25, 5, 30, 35  # minutes after T0


def M(t0, m):
    return t0 + pd.Timedelta(minutes=m)


@dataclass
class Reason:
    name: str
    t0: pd.Timestamp = STD_T0
    block_at: int = BLOCK  # minutes after t0
    flattens: bool = False  # entering it closes the position (the halts do; data blocks and Stop don't)
    holds_short: bool = False  # the set-up needs a short held before the block (the gap through its stop)
    bar: str = "1-MINUTE-LAST-INTERNAL"
    until: int | None = None  # minutes after t0 the block ends by itself (stale data: the next complete bar)


REASONS = {
    "drawdown_halt": Reason("drawdown_halt", flattens=True),
    "daily_pause": Reason("daily_pause", flattens=True),
    "liquidation_gap": Reason("liquidation_gap", flattens=True, holds_short=True),
    "liquidation": Reason("liquidation", block_at=0, flattens=True),
    "funding_missing": Reason("funding_missing", t0=FUND_T0),
    "stale_data": Reason("stale_data", block_at=SIGNAL, bar="5-MINUTE-LAST-INTERNAL", until=SIGNAL + 5),
    "winding_down": Reason("winding_down"),
    "retired": Reason("retired"),
    "stopped": Reason("stopped"),
}


def _wind_down_built():
    from sleeve_fund import store as store_mod

    return "wind_down" in getattr(store_mod, "COMMANDS", ())


def _pm_stop(store):
    """The dashboard's Stop, as sleeve_command writes it (the supervisor stops the process on its next step)."""
    store.set_desired_state(NAME, "stopped")
    store.drop_pending(NAME, "lapsed: the strategy was stopped before it acted")
    store.decide("PM", "stop", "QA: stop it", NAME)


def enter(reason: Reason, plan: Plan, store, *, at=None):
    """Arrange for `reason`'s block to start at `at` (default: the reason's own time) in `plan`, with production
    guards only. Returns the block time."""
    at = at if at is not None else M(plan.t0, reason.block_at)
    base = plan.price or (lambda s: 60_000 + s * 0.01)
    k = int((at - plan.t0).total_seconds())
    if reason.name == "drawdown_halt":
        # The drawdown reference is raised so the production guard (risk.check in SleeveRuntime.tick) halts on the
        # first tick at or after `at`: the only thing the harness touches.
        # The mark is journalled too (an hour before the session), so a restart restores the same reference.
        def dd(rt, kw, t0=plan.t0):
            peak = round(kw["equity"] / 0.75, 2)
            store.record_equity(NAME, equity=peak, cash=peak, qty=0, price=kw["price"], benchmark=peak,
                                ts=(t0 - pd.Timedelta(hours=1)).to_pydatetime())
            rt.peak = peak
        plan.hooks.append((at, dd))
    elif reason.name == "daily_pause":
        # The day opened 6% higher (journalled at 00:00 UTC, so a restart restores the same baseline): the
        # production guard pauses for the day's loss on the first tick at or after `at`.
        def daily(rt, kw, t0=plan.t0):
            day_open = round(kw["equity"] / 0.94, 2)
            store.record_equity(NAME, equity=day_open, cash=day_open, qty=0, price=kw["price"], benchmark=day_open,
                                ts=t0.normalize().to_pydatetime())
            rt._day_open = day_open
        plan.hooks.append((at, daily))
    elif reason.name == "liquidation_gap":
        # A real 2% stop (well inside half the ~49% distance to liquidation at 2x), then a 60% gap up through the
        # stop AND the liquidation price, as the GAP-LIQ guards-on cases set it up.
        plan.price = lambda s, b=base: b(s) * (1.6 if s >= k else 1.0)
    elif reason.name == "liquidation":
        _liquidated_journal(store, plan.t0)
        plan.balance = store.journal_book(NAME, 10_000)["cash"]
    elif reason.name == "funding_missing":
        _rates({FUND_S})
    elif reason.name == "stale_data":
        # Two of the five minutes of the decision bar closing at `at` reach the process with no trade (an outage
        # nothing refilled): degraded, 40% missing (over 10%).
        plan.holes.append((k - 4 * 60, k - 2 * 60))
    elif reason.name == "winding_down":
        assert _wind_down_built(), "not built: wind-down (P2-6), the PM command 'wind_down' and its state"
        plan.hooks.append((at, lambda rt, kw: store.command(NAME, "wind_down", "QA: wind it down", actor="PM")))
    elif reason.name == "retired":
        def retire(rt, kw):
            _pm_stop(store)
            store.archive(NAME)  # Retire = Stop + Archive until P2-6 [R79]
        plan.hooks.append((at, retire))
    elif reason.name == "stopped":
        plan.hooks.append((at, lambda rt, kw: _pm_stop(store)))
    return at


class GapLiqMissing(AssertionError):
    """The gap through the stop and the liquidation price was not booked as a liquidation (GAP-LIQ not built)."""


def check_gap_liq(store, at):
    liq = [o for o in _orders(store) if o["intent"] == "liquidation" and o["ts"] >= at]
    if not liq:
        raise GapLiqMissing("GAP-LIQ: the gap through the stop and the liquidation price is not booked as a "
                            f"liquidation: {[(o['ts'], o['side'], o['intent']) for o in _orders(store)]}")


def _create(store, params=None, bar="1-MINUTE-LAST-INTERNAL"):
    if not any(x.name == NAME for x in store.sleeves()):
        store.create_sleeve(name=NAME, strategy="egx", instrument=PAIR, bar_spec=bar, starting_balance=10_000,
                            params={**PERP, "stop_loss": STOP, **(params or {})}, venue="BINANCE")


def _liquidated_journal(store, t0):
    """The journal as the engine leaves it after a liquidation (#155 eb737bd's words; the gap set-up needs GAP-LIQ, so
    this one stands in for the liquidated STATE, which is what the gate reads): a short of 0.10 at 60,000 (0.6x equity
    at 2x; its 2% stop inside half the ~49% to liquidation) taken by the venue at its 89,700 liquidation price two
    hours before the session, a "liquidation" event, the liquidation halt, and no reset after liquidation since."""
    try:
        from sleeve_fund.paper.runtime import WIPED_OUT
    except ImportError:
        WIPED_OUT = "Position margin lost (liquidated)"
    _create(store)
    opened, gone = (t0 - pd.Timedelta(hours=3)).to_pydatetime(), (t0 - pd.Timedelta(hours=2)).to_pydatetime()
    store.record_equity(NAME, equity=10_000, cash=10_000, qty=0, price=60_000, benchmark=10_000,
                        ts=opened - timedelta(seconds=1))
    for oid, side, px, fee, intent, ts in (("qa-short", "SELL", 60_000.0, 3.0, "entry", opened),
                                           ("qa-liq", "BUY", 89_700.0, 4.49, "liquidation", gone)):
        store.record_order(NAME, order_id=oid, side=side, qty=0.10, intent=intent, reason=f"QA {intent}", signal={},
                           order_type="MARKET", ts=ts)
        store.record_fill(NAME, side=side, qty=0.10, price=px, fee=fee, order_id=oid, trade_id=oid, ts=ts)
        store.update_order(oid, status="filled", fill_qty=0.10, fill_px=px, fee=fee)
    cash = store.journal_book(NAME, 10_000)["cash"]
    store.record_equity(NAME, equity=cash, cash=cash, qty=0, price=89_700, benchmark=10_000, ts=gone)
    text = f"{WIPED_OUT}: {10_000 - cash:,.2f}, {(10_000 - cash) / 100:.0f}% of strategy equity at entry"
    store.event(NAME, "error", "liquidation", "Liquidated: the price 89,700 reached the liquidation price 89,700",
                ts=gone)
    store.event(NAME, "error", "risk_halt", f"{text}; PM must reset it after the liquidation", ts=gone)
    store.set_status(NAME, "halted", text)


def check_entered(reason: Reason, store, at):
    """Set-up check: the block really started (passes on every head unless named)."""
    if reason.name == "liquidation_gap":
        check_gap_liq(store, at)
    elif reason.name in ("drawdown_halt", "liquidation"):
        assert store.sleeve(NAME).status == "halted", store.sleeve(NAME).status_reason
    elif reason.name == "daily_pause":
        assert store.sleeve(NAME).status == "paused", store.sleeve(NAME).status_reason
    elif reason.name in ("stopped", "retired"):
        assert store.sleeve(NAME).desired_state == "stopped"


# ---------------------------------------------------------------------------------------------------------------
# Opening paths, replayed (mode "replay")
# ---------------------------------------------------------------------------------------------------------------

def _plan_for(reason: Reason, tag="s") -> Plan:
    plan = Plan(t0=reason.t0, bar=reason.bar, tag=tag)
    if reason.holds_short:
        plan.windows.append((M(plan.t0, 2), M(plan.t0, BLOCK + 3), -1))
    return plan


def path_new_entry(reason, plan):
    plan.windows.append((M(plan.t0, SIGNAL), M(plan.t0, plan.minutes), 1))


def path_reversal(reason, plan):
    """Long, then the signal flips to short after the block: the close part runs, the opening leg must not."""
    if reason.holds_short:  # short (liquidated by the gap), then long
        plan.windows.append((M(plan.t0, SIGNAL), M(plan.t0, plan.minutes), 1))
        return
    plan.windows.append((M(plan.t0, 2), M(plan.t0, SIGNAL), 1))
    plan.windows.append((M(plan.t0, SIGNAL), M(plan.t0, plan.minutes), -1))


def _resting(reason, plan, held):
    """A resting stop entry placed before the block (after it, for a block already in place), crossed after it."""
    base = plan.price or (lambda s: 60_000 + s * 0.01)
    p0 = base(PLACE * 60)
    c = CROSS * 60
    if reason.holds_short:  # short, liquidated by the gap; a SELL stop below, crossed when the price falls back
        trig, side = round(p0 * 0.99, 1), -1
        plan.price = lambda s, b=base: (p0 * 0.97 if s >= c else b(s))
    else:
        trig, side = round(p0 * 1.01, 1), 1
        plan.price = lambda s, b=base: b(s) * (1.02 if s >= c else 1.0)
    if held:
        plan.windows.append((M(plan.t0, 2), M(plan.t0, plan.minutes), side))
    plan.rest.append((M(plan.t0, PLACE), side, trig, 0.005 if held else 0.05))


def path_resting_entry(reason, plan):
    _resting(reason, plan, held=False)


def path_resting_add(reason, plan):
    _resting(reason, plan, held=True)


def path_outage_refill(reason, plan):
    """No trade reaches the process for four minutes (+27 to +31); the strategy's signal turns long inside the hole;
    when trades come back the held candles are rebuilt from the venue's own (the gap loader) and decided on."""
    a, z = (SIGNAL - 3) * 60, (SIGNAL + 1) * 60
    plan.holes.append((a, z))
    plan.windows.append((M(plan.t0, SIGNAL - 2), M(plan.t0, plan.minutes), 1))
    price = plan.price or (lambda s: 60_000 + s * 0.01)
    t0 = plan.t0

    def venue_bars(inst, bt):
        from nautilus_trader.model import Bar

        out = []
        for m in range(SIGNAL - 3, SIGNAL + 2):
            px = round(price(m * 60), 1)
            ts = _ns(M(t0, m + 1))
            out.append(Bar(bt, inst.make_price(px), inst.make_price(px + 5), inst.make_price(px - 5),
                           inst.make_price(px), inst.make_qty(10), ts, ts))
        return out

    Egx.gap_bars = venue_bars


PATHS = {
    "new_entry": path_new_entry,
    "reversal_open_leg": path_reversal,
    "resting_entry_fill": path_resting_entry,
    "resting_add_fill": path_resting_add,
    "outage_refill": path_outage_refill,
}


def replay_cell(tmp_path, store, monkeypatch, reason_name, path_name):
    reason = REASONS[reason_name]
    plan = _plan_for(reason)
    PATHS[path_name](reason, plan)
    at = enter(reason, plan, store)
    run(tmp_path, store, plan, monkeypatch)
    check_entered(reason, store, at)
    until = M(plan.t0, reason.until).to_pydatetime() if reason.until is not None else None
    return SimpleNamespace(at=at.to_pydatetime(), until=until, plan=plan, reason=reason)


# The replay matrix: which cells exist. Stale data is a one-bar block (the degraded candle) with no resting cells
# (README: Needs Advisor); a strategy restarted liquidated holds nothing to reverse or add to, and a resting entry it
# had went with the process (the gap set-up, GAP-LIQ, carries the resting case: P1-D21).
REPLAY_CELLS = []
for _r in REASONS:
    for _p in PATHS:
        if _r == "stale_data" and _p.startswith("resting"):
            continue
        if _r == "stale_data" and _p == "outage_refill":
            continue  # the refill is what clears stale data: test_clearing_...
        if _r == "liquidation" and _p in ("reversal_open_leg", "resting_entry_fill", "resting_add_fill"):
            continue
        if _r == "liquidation_gap" and _p == "resting_add_fill":
            continue
        REPLAY_CELLS.append((_r, _p))

# (reason, path) -> mark, from the runs on main 9ae1f8a and #155 eb737bd (README). Funding: O17b's block isn't built.
# Stopped and retired: the process still running after the PM's Stop opens and fills (the stopped-process window).
XFAIL_REPLAY: dict = {
    **{(r, p): xf(GATE) for r in ("funding_missing",) for p in PATHS},  # PE2: stopped, retired pass (CHOKE)
    **{("stale_data", p): xf(GATE, condition=ON_MAIN) for p in ("new_entry", "reversal_open_leg")},  # #155 D-items
}


def _marked(cells, table):
    out = []
    for r, p in cells:
        marks = []
        if r == "liquidation_gap" and not p:  # PE2: GAP-LIQ is on this head; its clearing cell still waits on RAL
            marks.append(xf(GAP_LIQ))
        elif (r, p) in table:
            marks.append(table[(r, p)])
        elif r == "winding_down":
            marks.append(xf(GATE))
        out.append(pytest.param(r, p, id=f"{r}-{p}", marks=marks))
    return out


def _marked1(reasons, table):
    """One parameter (the reason), marked as _marked marks it."""
    return [pytest.param(p.values[0], id=p.values[0], marks=p.marks) for p in _marked([(r, "") for r in reasons],
                                                                                       {(r, ""): m for r, m in
                                                                                        table.items()})]


@pytest.mark.parametrize("reason_name,path_name", _marked(REPLAY_CELLS, XFAIL_REPLAY))
def test_replay_no_exposure_is_added_while_blocked(tmp_path, store, monkeypatch, reason_name, path_name):
    """Assertions 1 and 2 [R113, R65]: once the block starts, no opening order is created and no fill grows |position|;
    a resting entry placed before it is cancelled at the block (never fills when its trigger is crossed after it)."""
    c = replay_cell(tmp_path, store, monkeypatch, reason_name, path_name)
    assert_no_exposure_added(store, c.at, until=c.until)
    if path_name.startswith("resting"):
        rest = [o for o in _orders(store) if o["order_type"] == "STOP_MARKET"]
        live = [o for o in rest if o["status"] not in ("canceled", "rejected", "denied", "expired")]
        assert not live, f"resting entry not cancelled at the block: {[(o['status'], o['filled_qty']) for o in live]}"


# ---------------------------------------------------------------------------------------------------------------
# Restarts, the supervisor and the dashboard (mode "paper"), each followed by the restarted process replayed
# ---------------------------------------------------------------------------------------------------------------

class _Popen:
    """The paper process, for the supervisor: alive until signalled."""
    pid = 1

    def __init__(self, *a, **k):
        self.alive, self.returncode = True, None

    def poll(self):
        return None if self.alive else self.returncode

    def send_signal(self, _):
        self.alive, self.returncode = False, 0

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.alive, self.returncode = False, -9


def _supervisor(store, monkeypatch):
    from sleeve_fund import supervisor

    monkeypatch.setattr(supervisor.subprocess, "Popen", _Popen)
    sup = supervisor.Supervisor(store)
    return sup


def _post(client, data, path="command"):
    return client.post(f"/sleeves/{NAME}/{path}", data=data, auth=AUTH, headers=SAME, follow_redirects=False)


def _command_error(r):
    assert r.status_code == 303, r.status_code
    return parse_qs(urlparse(r.headers["location"]).query).get("command_error", [None])[0]


def _session_one(tmp_path, store, monkeypatch, reason_name):
    """The block starts (no opening signal in this session). Returns the block time."""
    reason = REASONS[reason_name]
    plan = _plan_for(reason, tag="s1")
    at = enter(reason, plan, store)
    run(tmp_path, store, plan, monkeypatch)
    check_entered(reason, store, at)
    return at.to_pydatetime()


def _session_two(tmp_path, store, monkeypatch, reason_name, tag="s2", hours=2, minutes=20):
    """The paper process starting again `hours` later on the same journal: its signal has been long since ten minutes
    before it started (an entry missed while it was down: restart catch-up) and stays long. Returns its start."""
    reason = REASONS[reason_name]
    t0 = reason.t0 + pd.Timedelta(hours=hours)
    base = (lambda s: 89_700 + s * 0.01) if reason_name in ("liquidation",) else None
    plan = Plan(t0=t0, minutes=minutes, tag=tag, bar=reason.bar, price=base,
                windows=[(t0 - pd.Timedelta(minutes=10), M(t0, minutes), 1)],
                balance=store.journal_book(NAME, 10_000)["cash"])
    run(tmp_path, store, plan, monkeypatch)
    return t0.to_pydatetime()


PAPER_PATHS = ("restart_catch_up", "supervisor_restart", "dashboard_resume", "dashboard_start", "dashboard_reset")
PAPER_REASONS = ("drawdown_halt", "daily_pause", "liquidation", "liquidation_gap", "funding_missing", "winding_down",
                 "retired", "stopped")
PAPER_CELLS = []
for _r in PAPER_REASONS:
    for _p in PAPER_PATHS:
        if (_r, _p) in (("drawdown_halt", "dashboard_resume"), ("stopped", "dashboard_start")):
            continue  # each is its reason's own clearing action: test_clearing_...
        if _p == "dashboard_reset" and _r in ("drawdown_halt", "daily_pause"):
            continue  # Needs Advisor: does a per-strategy Reset (a testing-time clean slate) clear a halt?
        if _r in ("stopped", "retired") and _p == "restart_catch_up":
            continue  # nothing restarts a stopped strategy's process: supervisor_restart pins that
        PAPER_CELLS.append((_r, _p))

XFAIL_PAPER: dict = {
    **{("funding_missing", p): xf(GATE) for p in PAPER_PATHS},  # O17b not built
}


@pytest.mark.parametrize("reason_name,path_name", _marked(PAPER_CELLS, XFAIL_PAPER))
def test_paper_no_exposure_is_added_after_a_restart_or_a_pm_action(tmp_path, store, client, monkeypatch, reason_name,
                                                                   path_name):
    """Assertion 1 [R113, R79, R77, R110, R82]: in each blocked state, a restart (deploy warm-up, restart catch-up of
    an entry missed while down), the supervisor's reconcile loop (desired_state running: it restarts a crashed
    process), and the dashboard's Resume, Stop then Start, and per-strategy Reset leave nothing opening: whatever the
    route answers, the process that runs next opens nothing. (Whether each refusal names its reason:
    test_each_route_refusal_names_..., kept apart so a leak here is never hidden by a wording failure.)"""
    _session_one(tmp_path, store, monkeypatch, reason_name)
    sup = _supervisor(store, monkeypatch)
    sup.step()  # the supervisor runs the strategy's process (or not, when stopped)
    if path_name == "supervisor_restart":
        proc = sup.procs[NAME]
        if proc.popen is not None:
            proc.popen.alive, proc.popen.returncode = False, 1  # it crashed
            sup.step()
            proc.next_start = datetime.min.replace(tzinfo=timezone.utc)
            sup.step()
    elif path_name == "dashboard_resume":
        _post(client, {"command": "resume", "reason": "QA: resume it"})
    elif path_name == "dashboard_start":
        _post(client, {"command": "stop", "reason": "QA: stop it"})
        sup.step()
        _post(client, {"command": "start", "reason": "QA: start it"})
        sup.step()
    elif path_name == "dashboard_reset":
        _post(client, {"reason": "QA: reset it"}, path="reset")
        sup.step()
        sup.step()
    sup.step()
    if sup.procs.get(NAME) is not None and sup.procs[NAME].alive:  # the process runs: what it does next
        t2 = _session_two(tmp_path, store, monkeypatch, reason_name)
        assert_no_exposure_added(store, t2)
    else:  # its process is not running, so nothing can open
        assert store.sleeve(NAME).desired_state != "running" or NAME in store.archived()


# ---------------------------------------------------------------------------------------------------------------
# Assertion 3: exits are never blocked
# ---------------------------------------------------------------------------------------------------------------

EXIT_AT = 32  # minutes after t0, after the block


def _exit_plan(reason_name, kind, store):
    reason = REASONS[reason_name]
    plan = _plan_for(reason, tag="x")
    t_exit = M(plan.t0, reason.block_at + 1 if reason.name == "stale_data" else EXIT_AT)
    if not reason.holds_short:
        end = reason.block_at if (reason.name == "stale_data" and kind == "signal_exit") else (
            EXIT_AT if kind == "signal_exit" else plan.minutes)
        plan.windows.append((M(plan.t0, 2), M(plan.t0, end), 1))
    if kind == "stop":
        k = int((t_exit - plan.t0).total_seconds())
        plan.price = lambda s: (60_000 + s * 0.01) * (0.97 if s >= k else 1.0)
    elif kind == "close_now":
        plan.hooks.append((t_exit, lambda rt, kw: store.command(NAME, "flatten", "QA: close it now", actor="PM")))
    return plan, t_exit


EXIT_CELLS = [(r, k) for r in ("funding_missing", "stale_data", "stopped", "winding_down")
              for k in ("stop", "signal_exit", "close_now")]
EXIT_CELLS += [(r, "on_entering_the_block") for r in ("drawdown_halt", "daily_pause", "liquidation_gap")]
XFAIL_EXIT: dict = {}
EXIT_INTENTS = {"stop": ("stop_loss",), "signal_exit": ("exit",), "close_now": ("pm_flatten",),
                "on_entering_the_block": ("risk_halt", "risk_pause", "liquidation", "stop_loss")}


@pytest.mark.parametrize("reason_name,kind", _marked(EXIT_CELLS, XFAIL_EXIT))
def test_an_exit_still_runs_while_blocked(tmp_path, store, monkeypatch, reason_name, kind):
    """Assertion 3 [R65, R113 "exits/stops always run"]: long (short for the gap) before the block; while blocked a
    stop (the price falls 3% through the 2% stop), the signal's own exit (reduce to flat), or the PM's close-now
    (flatten) still closes it; a halt's own flatten, or the liquidation, closes it as the block starts. Flat after,
    by an order of that kind, and nothing opened."""
    reason = REASONS[reason_name]
    plan, t_exit = _exit_plan(reason_name, kind, store)
    at = enter(reason, plan, store)
    run(tmp_path, store, plan, monkeypatch)
    check_entered(reason, store, at)
    closed = [o for o in _orders(store) if o["intent"] in EXIT_INTENTS[kind] and o["ts"] >= at.to_pydatetime()
              and o["filled_qty"] > 0]
    assert closed, f"no {kind} exit ran while blocked: {[(o['ts'], o['intent'], o['status']) for o in _orders(store)]}"
    assert abs(_net(store)) < 1e-9, f"still holding {_net(store)} after the {kind}"
    assert_no_exposure_added(store, at.to_pydatetime())


# ---------------------------------------------------------------------------------------------------------------
# Assertion 2 and the in-flight rule: a partly filled resting entry when the block starts
# ---------------------------------------------------------------------------------------------------------------

MAKER_BAR = "10-MINUTE-LAST-INTERNAL"


def _maker_plan(reason, monkeypatch):
    """A post-only (maker-first) entry: the only paper path that rests an entry and fills it in slices over minutes
    (switched off in production by SLEEVE_MAKER_ORDERS, and refused on a perpetual: "maker-first orders aren't
    available on a perpetual yet", so this runs on spot, long only; latent until a strategy needs maker). A spot
    target weight of 1 (capped at balanced's 33%), decided at the 10-minute close at t0+20 and resting 9 minutes; the
    tape earns it a slice a minute (BOOK_SHARE of each minute's volume through its limit, about 0.012 of the order's
    0.054 a minute). The block starts at t0+21.5 (a runtime tick), so the entry is partly filled and the rest would be
    earned minute by minute and go at market by t0+29. A halt's own flatten first earns the half minute before it
    (M9-3, the backtest's venue fills from the prints of the bar the stop fires in): a slice booked AT the block
    instant is that pre-block tape, so it counts as in flight; only fills strictly after the block are late."""
    monkeypatch.setenv("SLEEVE_MAKER_ORDERS", "1")
    monkeypatch.setitem(REGISTRY, "egx", (EgxW, EgxWConfig))
    plan = Plan(t0=reason.t0, bar=MAKER_BAR, tag="mk", trade_size=0.002,
                params={"market": "spot", "allow_short": False, "maker_wait_minutes": 9})
    plan.windows.append((M(plan.t0, 20), M(plan.t0, plan.minutes), 1.0))
    plan.price = lambda s: 60_000 + s * 0.01 - (30 if s % 20 == 0 else 0)  # a dip every 20 s trades through its bid
    return plan


def _maker_run(tmp_path, store, monkeypatch, reason_name, after=None):
    reason = REASONS[reason_name]
    plan = _maker_plan(reason, monkeypatch)
    at = enter(reason, plan, store, at=M(plan.t0, 21.5))
    if after is not None:
        after(plan, at)
    run(tmp_path, store, plan, monkeypatch)
    check_entered(reason, store, at)
    maker = [o for o in _orders(store) if o["order_type"] == "POST-ONLY LIMIT" and o["intent"] == "entry"]
    assert maker, f"setup: no post-only entry: {[(o['ts'], o['intent'], o['order_type']) for o in _orders(store)]}"
    m = maker[0]
    before = [f for f in _fills(store) if f["ts"] <= at.to_pydatetime()  # in flight: booked by the block instant
              and f["order_id"] == m["order_id"]]
    assert 0 < sum(f["qty"] for f in before) < m["qty"], ("setup: the entry is partly filled when the block starts",
                                                         [(f["ts"], f["qty"]) for f in _fills(store)], m["qty"])
    return SimpleNamespace(at=at.to_pydatetime(), maker=m, before=round(sum(_signed(f) for f in before), 9))


# The missing funding rate is a perp-only block, and maker-first is refused on a perp: no partial-fill cell for it
# until maker reaches perps (README).
PARTIAL_REASONS = ("drawdown_halt", "daily_pause", "stopped", "winding_down")
XFAIL_PARTIAL: dict = {}  # PE2: stopped passes (CHOKE)


@pytest.mark.parametrize("reason_name", _marked1(PARTIAL_REASONS, XFAIL_PARTIAL))
def test_a_partly_filled_entry_has_its_remainder_cancelled_at_the_block(tmp_path, store, monkeypatch, reason_name,
                                                                       partial="partial"):
    """Assertion 2 [R113 "cancel remainder AT HALT TIME"]: the post-only entry's unfilled remainder is cancelled when
    the block starts: its journal row is cancelled, no slice of it fills after the block, and its rest never goes at
    market; nothing else opens."""
    c = _maker_run(tmp_path, store, monkeypatch, reason_name)
    late = [f for f in _fills(store) if f["ts"] > c.at and f["order_id"] == c.maker["order_id"]]
    assert not late, f"slices of the resting entry filled after the block: {[(f['ts'], f['qty']) for f in late]}"
    row = next(o for o in _orders(store) if o["order_id"] == c.maker["order_id"])
    assert row["status"] == "canceled", (row["status"], row["filled_qty"], row["qty"])
    assert_no_exposure_added(store, c.at, in_flight=("entry",) if not late else ())  # its late slices: checked above


KEPT_REASONS = ("stopped", "winding_down")
XFAIL_KEPT: dict = {}  # PE2: stopped passes (CHOKE)


@pytest.mark.parametrize("reason_name", _marked1(KEPT_REASONS, XFAIL_KEPT))
def test_the_filled_part_is_kept_with_its_stop_and_an_incident(tmp_path, store, monkeypatch, reason_name,
                                                              kept="kept"):
    """[R113 "keep filled part with its stop, never flatten"; Advisor 20:52 via the coordinator: kept, stop placed,
    incident written]: for a block that doesn't itself flatten, the part already filled is still held right after the
    block (not flattened), an incident (an error event of kind "incident") is written at or after the block, and its
    stop protects it: the price then falls 3% through the 2% stop and the stop closes exactly what was kept. (The halts
    flatten everything by their own rule: README, Needs Advisor.)"""
    def fall(plan, at):
        k = int((M(plan.t0, 40) - plan.t0).total_seconds())
        plan.windows[:] = [(M(plan.t0, 20), M(plan.t0, plan.minutes), 1.0)]
        price = plan.price
        plan.price = lambda s: price(s) * (0.97 if s >= k else 1.0)

    c = _maker_run(tmp_path, store, monkeypatch, reason_name, after=fall)
    held = 0.0
    for f in _fills(store):
        if f["ts"] < c.at + timedelta(minutes=5):
            held += _signed(f)
    assert held > 0 and abs(held - c.before) < 1e-9, (f"the filled part was not kept as it was: {c.before} at the "
                                                      f"block, {held} five minutes later")
    inc = [e for e in _events(store, ("incident",)) if e["ts"] >= c.at]
    assert inc and inc[0]["level"] == "error", "no incident written for the kept partial fill"
    stops = [o for o in _orders(store) if o["intent"] == "stop_loss" and o["filled_qty"] > 0]
    assert stops and abs(stops[0]["filled_qty"] - c.before) < 1e-9, [(o["intent"], o["filled_qty"]) for o in stops]


# ---------------------------------------------------------------------------------------------------------------
# P1-U35 (#167 round): a raced remainder is never left unwatched, whatever the strategy's status
# ---------------------------------------------------------------------------------------------------------------

CHOKE = "CHOKE"
RACED_QTY = 0.01
U35_CELLS = (("stopped", "stopped"), ("paused", "daily_pause"), ("halted", "drawdown_halt"),
             ("liquidated", "liquidation"), ("stopped_reset_flatten_lapsed", "stopped"))


def _raced_remainder(store, reason_name, at):
    """The race, as the journal records it: a resting stop entry (BUY 0.01) placed 20 minutes before the block, whose
    fill reached the venue before the block's cancel did, so it filled a second after the block (the cancel then had
    nothing to cancel). The engine is not asked to win the race; what follows it is the question."""
    px = 89_700.0 if reason_name == "liquidation" else round(60_000 + BLOCK * 60 * 0.01, 1)
    placed, filled = at - timedelta(minutes=20), at + timedelta(seconds=1)
    store.record_order(NAME, order_id="qa-raced", side="BUY", qty=RACED_QTY, intent="entry",
                       reason="QA resting stop entry, filled racing the block's cancel", signal={},
                       order_type="STOP_MARKET", ts=placed)
    store.record_fill(NAME, side="BUY", qty=RACED_QTY, price=px, fee=round(px * RACED_QTY * 0.0005, 6),
                      order_id="qa-raced", trade_id="qa-raced", ts=filled)
    store.update_order("qa-raced", status="filled", fill_qty=RACED_QTY, fill_px=px)
    return filled


@pytest.mark.parametrize("status,reason_name", [pytest.param(st, r, id=f"u35-{st}", marks=xf(CHOKE)
                                                if st in ("halted", "paused", "liquidated") else [])  # PE2: the rest pass (U35)
                                                for st, r in U35_CELLS])
def test_a_raced_remainder_is_kept_with_a_resting_stop_and_one_incident(tmp_path, store, client, monkeypatch, status,
                                                                         reason_name):
    """P1-U35 [R113 Advisor 20:56 "fills racing cancel accepted and stopped, never unwatched"; Advisor 20:52 via the
    coordinator: kept, stop placed, incident written]: a fill that raced the cancel at a halt, a pause, a liquidation
    or the PM's Stop is KEPT (never flattened), a STOP RESTS for it (an open stop_loss order in the journal, opposite
    side, covering it) and ONE incident is opened, whatever the status, stopped included (a stopped strategy has no
    process watching it). Then the supervisor and, if it runs one, the strategy's process carry on. Last cell: a
    per-strategy Reset on the stopped strategy queues its flatten (and starts it to trade it), its process never
    reports, the PM stops it again, and the lapsed flatten must not take the stop away (nor the position)."""
    at = _session_one(tmp_path, store, monkeypatch, reason_name)
    raced = _raced_remainder(store, reason_name, at)
    sup = _supervisor(store, monkeypatch)
    sup.step()
    if sup.procs.get(NAME) is not None and sup.procs[NAME].alive:  # its process runs (still blocked): what it does
        _session_two(tmp_path, store, monkeypatch, reason_name)
    sup.step()
    if status == "stopped_reset_flatten_lapsed":
        r = _post(client, {"reason": "QA: reset it"}, path="reset")
        assert r.status_code == 303, r.status_code
        sup.step()  # holding: the reset queues a PM flatten and sets it running so the flatten can trade
        assert [c for c in store.pending_commands(NAME) if c["command"] == "flatten"], "setup: no flatten queued"
        # Ten minutes on, the process started for the flatten has not reported (the dashboard lets a Stop drop a
        # waiting flatten only then), and the PM stops it: the flatten lapses unapplied.
        from sleeve_fund.dashboard import app as app_mod

        later = datetime.now(timezone.utc) + timedelta(minutes=10)
        monkeypatch.setattr(app_mod, "utcnow", lambda: later)
        r = _post(client, {"command": "stop", "reason": "QA: stop it again"})
        assert _command_error(r) is None, f"setup: the Stop was refused: {_command_error(r)}"
        assert not store.pending_commands(NAME), "setup: the flatten did not lapse"
        # (the reset is still pending: the supervisor's next step queues its flatten again; checked as it stands)
    held = round(sum(_signed(f) for f in _fills(store) if f["ts"] >= raced - timedelta(seconds=1)), 9)
    assert abs(held - RACED_QTY) < 1e-9 and abs(_net(store) - RACED_QTY) < 1e-9, (
        f"the raced remainder was not kept: net {_net(store)}, "
        f"{[(f['ts'], f['side'], f['qty']) for f in _fills(store) if f['ts'] >= raced]}")
    from sleeve_fund.store import OPEN_ORDER_STATUSES

    stops = [o for o in _orders(store) if o["intent"] == "stop_loss" and o["side"] == "SELL"
             and o["status"] in OPEN_ORDER_STATUSES and o["qty"] - o["filled_qty"] >= RACED_QTY - 1e-9]
    assert stops, ("no stop rests for the raced remainder: "
                   f"{[(o['ts'], o['intent'], o['order_type'], o['status']) for o in _orders(store) if o['ts'] >= at]}")
    inc = [e for e in _events(store, ("incident",)) if e["ts"] >= at]
    assert len(inc) == 1 and inc[0]["level"] == "error", f"not one incident for the raced fill: {inc}"


# ---------------------------------------------------------------------------------------------------------------
# Assertion 5: after its own clearing action, opening works again
# ---------------------------------------------------------------------------------------------------------------

def _opened(store, t):
    return [o for o in _orders(store) if o["intent"] == "entry" and o["ts"] >= t and o["filled_qty"] > 0]


def _clear_drawdown_halt(tmp_path, store, monkeypatch, client):
    _session_one(tmp_path, store, monkeypatch, "drawdown_halt")
    store.command(NAME, "resume", "QA: checked, carry on", actor="PM")
    return _session_two(tmp_path, store, monkeypatch, "drawdown_halt")


def _clear_liquidation(tmp_path, store, monkeypatch, client, reason="liquidation"):
    _session_one(tmp_path, store, monkeypatch, reason)
    from sleeve_fund import store as store_mod

    if "reset_after_liquidation" in getattr(store_mod, "COMMANDS", ()):  # RAL as built (test_ral_xfails.py)
        iid = store.last_event(NAME, ("incident",))
        assert iid is not None, "no incident to reset after"
        if hasattr(store, "write_incident_note"):
            store.write_incident_note(iid["id"], author="Head of Engineering", why_stop_did_not_protect="QA gap")
        store.command(NAME, "reset_after_liquidation", "QA: incident note written", actor="PM", incident=iid["id"])
    else:  # the journal row RAL writes [R79]: the contract the gate reads
        store.event(NAME, "info", LIQUIDATION_RESET, "Reset after liquidation by PM (QA stand-in until RAL is built)")
    return _session_two(tmp_path, store, monkeypatch, reason)


def _clear_funding(tmp_path, store, monkeypatch, client):
    """Blocked at 08:20 (the rate for 08:00 missing, the check failed at 08:15); the rate is then stored and the next
    settlement's check passes; a restart after 16:00 opens."""
    reason = REASONS["funding_missing"]
    plan = _plan_for(reason, tag="f1")
    path_new_entry(reason, plan)
    at = enter(reason, plan, store)
    run(tmp_path, store, plan, monkeypatch)
    assert not _opened(store, at.to_pydatetime()), "blocked first: an entry opened with the 08:00 rate missing"
    _rates(set())  # the rate arrives, and the 16:00 one too
    return _session_two(tmp_path, store, monkeypatch, "funding_missing", hours=9)


def _clear_stale(tmp_path, store, monkeypatch, client):
    """The degraded candle at t0+30 holds the entry back; the next candle is whole again, and the entry opens on it."""
    reason = REASONS["stale_data"]
    plan = _plan_for(reason, tag="d1")
    path_new_entry(reason, plan)
    at = enter(reason, plan, store)
    run(tmp_path, store, plan, monkeypatch)
    until = M(plan.t0, reason.until).to_pydatetime()
    assert not [o for o in _opened(store, at.to_pydatetime()) if o["ts"] < until], "blocked first on the degraded bar"
    return until


def _clear_stopped(tmp_path, store, monkeypatch, client):
    _session_one(tmp_path, store, monkeypatch, "stopped")
    sup = _supervisor(store, monkeypatch)
    sup.step()
    r = _post(client, {"command": "start", "reason": "QA: start it"})
    assert _command_error(r) is None, _command_error(r)
    sup.step()
    assert sup.procs[NAME].alive
    return _session_two(tmp_path, store, monkeypatch, "stopped")


CLEARING = {"drawdown_halt": _clear_drawdown_halt, "liquidation": _clear_liquidation,
            "liquidation_gap": lambda *a: _clear_liquidation(*a, reason="liquidation_gap"),
            "funding_missing": _clear_funding, "stale_data": _clear_stale, "stopped": _clear_stopped}
XFAIL_CLEAR: dict = {
    "liquidation": xf(GATE),  # halted through the liquidation_reset row: nothing reads it until RAL
    "funding_missing": xf(GATE),  # never blocked in the first place (O17b)
    "stale_data": xf(GATE, condition=ON_MAIN),
}


@pytest.mark.parametrize("reason_name", _marked1(CLEARING, XFAIL_CLEAR))
def test_after_its_own_clearing_action_opening_works_again(tmp_path, store, client, monkeypatch, reason_name,
                                                          clear="clear"):
    """Assertion 5 [R79, R82, R46]: resume for a drawdown halt; the reset after liquidation (the liquidation_reset row
    RAL writes) for a liquidation; the rate stored and the next check passed for a missing funding rate; whole data
    for a degraded candle; Start for a stopped strategy. Then the next signal opens. (The daily pause and its 00:00 UTC
    roll: test_a_daily_pause_ends_at_the_next_0000_utc_roll_and_not_before.)"""
    t = CLEARING[reason_name](tmp_path, store, monkeypatch, client)
    assert _opened(store, t), ("nothing opened after the clearing action: "
                               f"{[(o['ts'], o['intent']) for o in _orders(store)]}")


def test_a_daily_pause_ends_at_the_next_0000_utc_roll_and_not_before(tmp_path, store, monkeypatch):
    """Assertion 5 for the daily pause [R79 "daily pause: next 00:00 UTC roll"]: paused at 10:15, it is still
    blocked 30 s before 00:00 UTC (a restart then keeps it), and opens 30 s after it, on a frozen clock."""
    at = _session_one(tmp_path, store, monkeypatch, "daily_pause")
    roll = datetime(2025, 10, 4, 0, 0, tzinfo=timezone.utc)
    equity = store.journal_book(NAME, 10_000)["cash"]

    def started(now):
        rt = SleeveRuntime(store, NAME, now=lambda: now)
        rt.on_start(0.0005)
        rt.tick(equity=equity, cash=equity, qty=0.0, price=60_000.0)
        return rt

    assert at < roll
    assert not started(roll - timedelta(seconds=30)).can_open()
    s = store.sleeve(NAME)
    assert s.paused_until == roll, f"the pause ends at {s.paused_until}, not the next 00:00 UTC"
    assert started(roll + timedelta(seconds=30)).can_open()


# ---------------------------------------------------------------------------------------------------------------
# Assertion 4: every refusal names its reason
# ---------------------------------------------------------------------------------------------------------------

GATE_REASONS = ("drawdown_halt", "daily_pause", "liquidation", "funding_missing", "stale_data", "winding_down",
                "retired", "stopped")


@pytest.mark.parametrize("reason_name", [pytest.param(r, marks=xf(GATE) if r in ("funding_missing", "winding_down") else [])
                                         for r in GATE_REASONS])  # PE2: the rest pass (CHOKE)
def test_the_gate_returns_blocked_and_names_its_reason(tmp_path, store, monkeypatch, reason_name):
    """Assertion 4, at its source: in each blocked state the ONE gate's state function returns (True, reason), the
    reason naming the state in words the PM reads (drawdown, daily loss, liquidated, funding, degraded data, wind-down,
    retired, stopped)."""
    if reason_name == "stale_data":
        reason = REASONS[reason_name]
        plan = _plan_for(reason, tag="g1")
        at = enter(reason, plan, store)
        run(tmp_path, store, plan, monkeypatch)
        at = at.to_pydatetime() + timedelta(seconds=30)
    else:
        at = _session_one(tmp_path, store, monkeypatch, reason_name) + timedelta(minutes=5)
    blocked, why = gate(store, at=at)
    assert blocked and re.search(NAMES[reason_name], why.lower()), (blocked, why)


ENTRY_BLOCK_KINDS = ("entry_blocked", "funding_entry_blocked", "degraded_bar")
XFAIL_NAMED: dict = {
    **{r: xf(GATE) for r in ("funding_missing",)},  # PE2: the rest pass (CHOKE)
    "stale_data": xf(GATE, condition=ON_MAIN),  # #155 journals "degraded_bar"
}


@pytest.mark.parametrize("reason_name", _marked1(GATE_REASONS, XFAIL_NAMED))
def test_the_engine_journals_a_refused_entry_naming_its_reason(tmp_path, store, monkeypatch, reason_name,
                                                              named="named"):
    """Assertion 4 in the engine: when the gate refuses the entry the signal wants after the block, the journal says
    so in words naming the reason (an event of kind "entry_blocked"; O17b's "funding_entry_blocked" and #155's
    "degraded_bar" count), at or after the signal."""
    c = replay_cell(tmp_path, store, monkeypatch, reason_name, "new_entry")
    signal = M(c.plan.t0, SIGNAL).to_pydatetime()
    said = [e for e in _events(store, ENTRY_BLOCK_KINDS) if e["ts"] >= min(signal, c.at)]
    assert any(re.search(NAMES[reason_name], e["message"].lower()) for e in said), (
        [(e["kind"], e["message"][:100]) for e in said] or "no refusal journalled")


ROUTE_CELLS = ([("dashboard_resume", r) for r in ("daily_pause", "liquidation", "winding_down", "retired")]
               + [("dashboard_start", r) for r in ("drawdown_halt", "daily_pause", "liquidation", "winding_down",
                                                   "retired")]
               + [("dashboard_reset", r) for r in ("liquidation",)])
XFAIL_ROUTE: dict = {
    **{c: xf(GATE) for c in ROUTE_CELLS if c[1] != "liquidation"  # accepted (wind-down: not built)
       and c not in {("dashboard_resume", "retired"), ("dashboard_start", "drawdown_halt"),
                     ("dashboard_start", "retired")}},  # PE2: these pass (CHOKE)
    **{c: xf(GATE, condition=not HAS_164) for c in ROUTE_CELLS if c[1] == "liquidation"},  # #164 refuses them
}


@pytest.mark.parametrize("route,reason_name", [pytest.param(a, b, id=f"{a}-{b}", marks=XFAIL_ROUTE.get((a, b), []))
                                               for a, b in ROUTE_CELLS])
def test_each_route_refusal_names_its_reason(tmp_path, store, client, monkeypatch, route, reason_name):
    """Assertion 4 on the dashboard [R79 "Start refuses while halted naming the clearing action", R77, R110 (U27)]:
    the route refuses in words (command_error) naming the reason or its clearing action, and changes nothing: Resume
    on a daily pause (only the 00:00 UTC roll ends it), a liquidation (only the reset after liquidation), a wind-down
    or a retired strategy; Start (after Stop) on a halt, a pause, a liquidation, a wind-down or a retired strategy;
    the ordinary Reset on a liquidation."""
    _session_one(tmp_path, store, monkeypatch, reason_name)
    sup = _supervisor(store, monkeypatch)
    sup.step()
    if route == "dashboard_start":
        if store.sleeve(NAME).desired_state == "running":
            _post(client, {"command": "stop", "reason": "QA: stop it"})
            sup.step()
        r = _post(client, {"command": "start", "reason": "QA: start it"})
    elif route == "dashboard_resume":
        r = _post(client, {"command": "resume", "reason": "QA: resume it"})
    else:
        r = _post(client, {"reason": "QA: reset it"}, path="reset")
    err = _command_error(r)
    words = NAMES[reason_name] + ("|" + re.escape(CLEARED_BY[reason_name]) if reason_name in CLEARED_BY else "")
    assert err and re.search(words, err.lower()), err or "accepted: no refusal at all"


# ---------------------------------------------------------------------------------------------------------------
# CHOKE-2 (deferred, agreed with the HoE): backtest/study mode, the demo mirror, P2-1 resizes and the rebalance.
# Must close before any G1 is judged and before the demo mirror runs for any strategy (README). Run the gate set
# alone with -k "not choke2".
# ---------------------------------------------------------------------------------------------------------------

class EgxWConfig(EgxConfig):
    pass


class EgxW(Egx):
    """A target weight w while open_ns <= the decision bar's close < close_ns, for each [open_ns, close_ns, w]
    (spot); raising it while held, past the rebalance band, is an add."""

    def target_weight(self, bar):
        for o, c, w in self.windows:
            if o <= bar.ts_event < c:
                return float(w)
        return 0.0

    want_side = LongFlatStrategy.want_side


def _hourly(t0, closes):
    idx = pd.date_range(t0, periods=len(closes), freq="1h", tz="UTC") + pd.Timedelta(hours=1)
    rows = []
    prev = closes[0]
    for c in closes:
        o, (h, lo) = prev, (max(prev, c), min(prev, c))
        rows.append((o, h, lo, c))
        prev = c
    return pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close"]).assign(volume=1e9)


BT_T0 = utc("2025-10-03 00:00")


def _backtest_cell(monkeypatch, reason_name, path_name):
    """A perp backtest on hourly bars with the paper runtime (balanced): the block starts at the 05:00 close, the
    signal turns long at 08:00, and a resting stop entry placed at 02:00 is crossed at 08:00."""
    from sleeve_fund.research.runner import run_backtest

    closes = [60_000.0 + i for i in range(24)]
    windows, rest = [], []
    block = BT_T0 + pd.Timedelta(hours=5)
    if reason_name == "liquidation_gap":
        windows.append((BT_T0 + pd.Timedelta(hours=1), BT_T0 + pd.Timedelta(hours=7), -1))
        closes = [c * (1.6 if i >= 4 else 1.0) for i, c in enumerate(closes)]
        if path_name == "resting_entry_fill":
            rest.append((BT_T0 + pd.Timedelta(hours=2), -1, 59_400.0, 0.05))
            closes = [58_200.0 if i >= 7 else c for i, c in enumerate(closes)]
    elif path_name == "resting_entry_fill":
        rest.append((BT_T0 + pd.Timedelta(hours=2), 1, 60_600.0, 0.05))
        closes = [c * (1.02 if i >= 7 else 1.0) for i, c in enumerate(closes)]
    if path_name == "new_entry":
        windows.append((BT_T0 + pd.Timedelta(hours=8), BT_T0 + pd.Timedelta(hours=24), 1))
    real = SleeveRuntime.tick

    def tick(self, **kw):
        if self.now() >= block and not getattr(self, "_qa_blocked", False):
            self._qa_blocked = True
            if reason_name == "drawdown_halt":
                self.peak = kw["equity"] / 0.75
            elif reason_name == "daily_pause":
                self._day_open = kw["equity"] / 0.94
        return real(self, **kw)

    monkeypatch.setattr(SleeveRuntime, "tick", tick)
    params = {**PERP, "stop_loss": STOP, "windows": [[_ns(o), _ns(c), s] for o, c, s in windows],
              "rest": [[_ns(t), s, trig, q] for t, s, trig, q in rest]}
    r = run_backtest("egx", _hourly(BT_T0, closes), venue("binance").instrument("BTC", "USDT"), params,
                     starting_capital=10_000, risk_profile="balanced", bar_minutes=60, half_spread=0)
    j = r.journal
    fills = sorted(j.fills_, key=lambda f: (f["ts"], f["id"]))
    orders = sorted(j.orders_.values(), key=lambda o: (o["ts"], o["id"]))
    at = block.to_pydatetime()
    if reason_name == "liquidation_gap" and not [o for o in orders if o["intent"] == "liquidation"]:
        raise GapLiqMissing(f"GAP-LIQ (backtest): {[(o['ts'], o['intent']) for o in orders]}")
    intents = {o["order_id"]: o["intent"] for o in orders}
    net, grew = 0.0, []
    for f in fills:
        before, net = net, net + _signed(f)
        if f["ts"] > at and abs(net) > abs(before) + 1e-9:
            grew.append((f["ts"], f["side"], f["qty"], intents.get(f["order_id"]) or
                         r.decisions.get(f["order_id"], {}).get("intent")))
    opened = [(o["ts"], o["intent"]) for o in orders if o["intent"] in OPENING and o["ts"] > at]
    assert not opened and not grew, f"exposure added while blocked (backtest): opened {opened}, grew {grew}"


CHOKE2_BACKTEST = [(r, p) for r in ("drawdown_halt", "daily_pause", "liquidation_gap")
                   for p in ("new_entry", "resting_entry_fill")]
XFAIL_CHOKE2: dict = {
    ("mirror", "drawdown_halt"): xf(CHOKE2),  # the catch-up buys a failed copy whatever the state now
    ("mirror", "stopped"): xf(CHOKE2),
}


class TestChoke2:
    """Every test id here contains "choke2": -k "not choke2" runs the gate set alone, -k choke2 this set alone."""

    @pytest.mark.parametrize("reason_name,path_name",
                             [pytest.param(r, p, id=f"choke2-backtest-{r}-{p}",
                                           marks=[xf(GAP_LIQ)] if r == "liquidation_gap" else
                                           XFAIL_CHOKE2.get(("backtest", r, p), []))
                              for r, p in CHOKE2_BACKTEST])
    def test_backtest_adds_no_exposure_while_blocked(self, monkeypatch, reason_name, path_name):
        """CHOKE-2, backtest/study mode [R113; P1-D21 in backtest]: the same gate in a backtest with the paper
        runtime: after a drawdown halt, a daily pause or a liquidation, no entry opens and a resting entry placed
        before it never fills."""
        _backtest_cell(monkeypatch, reason_name, path_name)

    @pytest.mark.parametrize("reason_name", [pytest.param(r, id=f"choke2-rebalance-{r}",
                                                          marks=XFAIL_CHOKE2.get(("rebalance", r), []))
                                             for r in ("drawdown_halt", "stopped")])
    def test_a_rebalance_add_is_refused_while_blocked(self, tmp_path, store, monkeypatch, reason_name):
        """CHOKE-2, the rebalance [R113 "band adds, rebalance increases"]: a spot target-weight strategy at 30%
        (band 10%) asks for 60% after the block: the increase is an add, refused; nothing grows."""
        monkeypatch.setitem(REGISTRY, "egx", (EgxW, EgxWConfig))
        reason = REASONS[reason_name]
        plan = Plan(t0=reason.t0, tag="rb", params={"market": "spot", "allow_short": False, "stop_loss": None,
                                                     "rebalance_band": 0.1})
        plan.windows += [(M(plan.t0, 2), M(plan.t0, SIGNAL), 0.3), (M(plan.t0, SIGNAL), M(plan.t0, plan.minutes), 0.6)]
        at = enter(reason, plan, store)
        orders = run(tmp_path, store, plan, monkeypatch)
        assert orders, "setup: the strategy traded"
        check_entered(reason, store, at)
        assert_no_exposure_added(store, at.to_pydatetime())

    @xf(CHOKE2)
    def test_a_p2_1_resize_is_refused_while_blocked(self):
        """CHOKE-2, P2-1 resizes [R113 "Add = any order increasing |position|"]: a centrally sized position resized
        up by its band (P2-1b, 25% band variant) while blocked is an add, refused. Assumed interface: params
        {"sizing": "central", "resize_band": 0.25} on LongFlatConfig, sizing from sleeve_fund.portfolio.sizing."""
        import inspect

        try:
            from sleeve_fund.portfolio import sizing  # noqa: F401
        except ImportError:
            raise AssertionError("not built: P2-1 central sizing (sleeve_fund.portfolio.sizing, #165)") from None
        assert "resize_band" in inspect.signature(LongFlatConfig.__init__).parameters, (
            "not built: P2-1b band resize (LongFlatConfig resize_band)")
        raise AssertionError("not built: the resize-while-blocked case is written when P2-1b lands (README)")

    @pytest.mark.parametrize("reason_name", [pytest.param(r, id=f"choke2-mirror-{r}",
                                                          marks=XFAIL_CHOKE2.get(("mirror", r), []))
                                             for r in ("drawdown_halt", "stopped")])
    def test_the_demo_mirror_adds_no_exposure_while_blocked(self, store, reason_name):
        """CHOKE-2, the demo mirror: a strategy blocked while holding 0.05 long whose copy to Bybit Demo failed. The
        mirror's catch-up (two checks in a row) would buy 0.05 on the demo account: a new opening order there while
        blocked. Assumed rule (Needs Advisor, README): while blocked the mirror only ever reduces the demo position."""
        from sleeve_fund import mirror

        _create(store, {"demo_mirror": True})
        t = datetime(2025, 10, 3, 9, 0, tzinfo=timezone.utc)
        store.record_mirror(NAME, fill_id=0, status="start", message="mirror started")
        store.record_order(NAME, order_id="qa-long", side="BUY", qty=0.05, intent="entry", reason="QA", signal={},
                           order_type="MARKET", ts=t)
        store.record_fill(NAME, side="BUY", qty=0.05, price=60_000.0, fee=1.5, order_id="qa-long", trade_id="qa-long",
                          ts=t)
        store.record_mirror(NAME, fill_id=store.last_fill_id(NAME), status="error", instrument="BTCUSDT",
                            amount=0.05, message="QA: the copy failed")
        if reason_name == "drawdown_halt":
            store.event(NAME, "error", "risk_halt", "drawdown 25.0% hit the 20% limit; flattened, PM must resume", ts=t)
            store.set_status(NAME, "halted", "drawdown 25.0% hit the 20% limit")
        else:
            _pm_stop(store)

        class Demo:
            unit = "BTC"

            def __init__(self):
                self.sent = []

            @staticmethod
            def owns(instrument):
                return instrument in mirror.BYBIT_CONTRACTS

            def position(self, symbol):
                return 0.0

            def market(self, side, symbol, qty, label):
                self.sent.append((side, symbol, qty, label))
                return "demo-1", 60_000.0

        demo = Demo()
        seen = mirror.catch_up(store, {"BYBIT": demo}, {})
        mirror.catch_up(store, {"BYBIT": demo}, seen)
        assert not demo.sent, f"the demo mirror opened on the demo account while blocked: {demo.sent}"
