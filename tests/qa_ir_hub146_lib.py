# Verbatim copy of quant-review/v2-p1/hub-146-scripts/test_hub_146_qa.py, the harness tests/test_qa_ir_l6_restart.py
# (integration-1709cd9 re-pin ir3) imports: its restart() seeds the held entry at mid. Not collected. Keep it unchanged.
"""QA PR #146 "late bars delivered for exits only; C1; P1-C10" (head ebe39c5; f06914a before the CR minor).

Independent of the PR's own tests (tests/test_timing.py, tests/test_hub_path_parity.py). A test that passes pins
the behaviour seen. A finding is a strict xfail whose reason starts with its id (P1-L..): it FAILS the run once
the behaviour is fixed, so flip it then. Imports of the code under test sit inside the helpers and test bodies,
so a renamed function fails that test, not the collection. Run from the worktree root (no Postgres needed):

  cd /tmp/claude-0/p146r && PYTHONDONTWRITEBYTECODE=1 BACKTEST_ISOLATE=0 PYTHONPATH=/tmp/claude-0/p146r \
    /tmp/claude-0/v/bin/python -m pytest -q -p no:cacheprovider -rxXs \
    --basetemp=/tmp/claude-0/p146r-scratch/pytest \
    /mnt/project-files/sleeve-fund/quant-review/v2-p1/hub-146-scripts/

How the hub path is replayed (as QA's #140 round did): Nautilus's BacktestEngine set up as build_node sets up a
hub-fed paper node (bar_execution=False: fills from trades and quotes only; CASH account for spot, MARGIN at the
venue leverage for a perp), the strategy attached to the paper runtime (SleeveRuntime, not a backtest), deciding on
`<n>-MINUTE-LAST-EXTERNAL` bars that come out of the PR's own hub_client.Decoder. The Decoder is fed the hub's
1-minute messages, each at the time it reaches the node; one trade a second (and the quote after it) reaches the
strategy except while the hub is away. The reference is research.runner.run_backtest on the same minutes.

Advisor rulings of 6 Oct 16:53 (late-146.md): NA-1 to NA-3 are pinned at the end of this file as P1-L6 to P1-L8
(HoE: in #146, with L1-L3); the parts already true on ebe39c5 are plain tests marked "Already true". NA-4 went to the
stop-safety PR: quant-review/stop-safety-xfails/test_na4_refused_start_xfails.py.

The probe strategy holds `side` (+1 long, -1 short) for the bars closing in [enter, leave) minutes after START,
and is flat otherwise, so every order is placed by design and only the stop, the target or the hub can change it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

START = 1_759_449_600_000_000_000  # 2025-10-03 00:00 UTC
S = 1_000_000_000
M = 60 * S
SPREAD = 12.0
BASE = 60_000.0


# ----------------------------------------------------------------------------------------------- the probe


def _probe_classes():
    from sleeve_fund.strategies.base import LongFlatConfig, LongFlatStrategy

    class ProbeConfig(LongFlatConfig):
        def __init__(self, *, enter: int = 5, leave: int = 60, side: int = 1, after: int = 0, w1: float | None = None,
                     w2: float | None = None, add_at: int = 10**9, **kwargs) -> None:
            super().__init__(**kwargs)
            self.enter, self.leave, self.side, self.after = enter, leave, side, after
            self.w1, self.w2, self.add_at = w1, w2, add_at

    class Probe(LongFlatStrategy):
        def __init__(self, config) -> None:
            super().__init__(config)
            self.enter, self.leave, self.side, self.after = config.enter, config.leave, config.side, config.after
            self.w1, self.w2, self.add_at = config.w1, config.w2, config.add_at
            self.accepted: list[tuple] = []  # every bar the indicators took, live or warm-up, in order

        def update_indicators(self, bar) -> None:
            self.accepted.append((int(bar.ts_event), bar.open.as_double(), bar.high.as_double(),
                                  bar.low.as_double(), bar.close.as_double(), bar.volume.as_double()))

        def _k(self, bar) -> int:
            return int((bar.ts_event - START) // M)

        def want_long(self, bar):
            return self.enter <= self._k(bar) < self.leave

        def target_weight(self, bar):
            if self.w1 is None:
                return super().target_weight(bar)
            k = self._k(bar)
            return 0.0 if not self.enter <= k < self.leave else self.w1 if k < self.add_at else self.w2

        @classmethod
        def weight_sized(cls, params):
            return params.get("w1") is not None

        def want_side(self, bar):
            k = self._k(bar)
            return self.side if self.enter <= k < self.leave else self.after if k >= self.leave else 0

    return Probe, ProbeConfig


@pytest.fixture(autouse=True)
def _probe(monkeypatch):
    from sleeve_fund.strategies import REGISTRY

    monkeypatch.setitem(REGISTRY, "probe", _probe_classes())


# Stop safety (PE2's PR, merged with #146): a stopless perp above 1x is refused at start (check_perp_stop), a new perp
# entry is refused past the 5 % open-risk limit (open_risk.LIMIT), and a restart holding a stopless position places a
# safety stop. A cell whose subject is liquidation mechanics needs a position those guards would refuse or close, so it
# lifts them with the repo's per-cell marks, saying why (as #155's QA cells do); every other cell runs guarded, its
# set-up made to comply (1x, a stop inside half the distance to liquidation, or the conservative profile).
GUARDS_OFF = "guards off: liquidation mechanics only"
LIQUIDATION_MECHANICS = (pytest.mark.no_open_risk_limit(reason=GUARDS_OFF),
                         pytest.mark.no_restart_safety_stop(reason=GUARDS_OFF))
GUARDED_STOP = 0.02  # a stop for a perp above 1x: inside half its distance to liquidation, never reached by WAVY


@pytest.fixture(autouse=True)
def _guard_marks(monkeypatch, request):
    """The guard-lift marks, applied here as the repo's tests/conftest.py does, so these files behave the same run
    from QA's shared folder (where that conftest isn't loaded); as it does, open risk reads a calm 2 % daily ATR
    unless real_daily_atr. A head without stop safety (main) has nothing to lift."""
    try:
        from sleeve_fund import open_risk
    except ImportError:
        return
    if request.node.get_closest_marker("real_daily_atr") is None:
        monkeypatch.setattr(open_risk, "history_atr_pct", lambda venue, pair, now, history=None: 0.02)
    if request.node.get_closest_marker("no_open_risk_limit") is not None:
        monkeypatch.setattr(open_risk, "LIMIT", float("inf"))
    if request.node.get_closest_marker("no_restart_safety_stop") is not None:
        from sleeve_fund.strategies.base import LongFlatStrategy

        monkeypatch.setattr(LongFlatStrategy, "_safety_stop_on_restore", lambda self, book: None)


# ----------------------------------------------------------------------------------------------- the data


def flat_prices(n_minutes: int = 40) -> np.ndarray:
    s = np.arange(n_minutes * 60)
    return BASE * (1 + 1e-7 * s)


def shape(p: np.ndarray, a: float, b: float, factor: float) -> np.ndarray:
    """Multiply the price by `factor` for seconds in [a, b) (minutes, fractional allowed)."""
    p = p.copy()
    s = np.arange(len(p))
    p[(s >= a * 60) & (s < b * 60)] *= factor
    return p


def ticks(inst, prices, gone=frozenset()):
    from nautilus_trader.model import AggressorSide, Price, Quantity, QuoteTick, TradeId, TradeTick

    out = []
    for s, px in enumerate(np.round(prices, 1)):
        if s in gone:
            continue
        t = START + s * S
        out.append(TradeTick(inst.id, Price(px, 1), Quantity(1.0, 8), AggressorSide.BUY if s % 2 else
                             AggressorSide.SELL, TradeId(str(s)), t, t + 1000))
        out.append(QuoteTick(inst.id, Price(px - SPREAD / 2, 1), Price(px + SPREAD / 2, 1), Quantity(1, 8),
                             Quantity(1, 8), t + 2000, t + 3000))
    return out


def minutes_of(prices) -> pd.DataFrame:
    """1-minute OHLC by open time (the store's convention) from the per-second trades."""
    s = pd.Series(np.round(prices, 1), index=pd.to_datetime(START + np.arange(len(prices)) * S, unit="ns", utc=True))
    m = s.resample("1min", closed="left", label="left").ohlc()
    m["volume"] = 60.0
    return m


def hub_msg(iid: str, close: int, r, refilled=False) -> dict:
    return {"t": "bar", "id": iid, "o": f"{r.open:.1f}", "h": f"{r.high:.1f}", "l": f"{r.low:.1f}",
            "c": f"{r.close:.1f}", "v": "60.00000000", "ts": close, "recv": close, "refilled": refilled}


def schedule(mins: pd.DataFrame, iid: str, lag: int = S // 2, away=None, back_at=None, refill_after_live=False,
             lost=()) -> list[tuple[int, dict]]:
    """(arrival ns, message) per hub minute. away=(a, b): minutes closing in (a, b] are refilled at back_at
    (refill_after_live: after the first live minute following them, as the hub's relay publishes a gap's live
    bar before its REST refill comes back). lost: minute closes the client never gets (recovered from the store).
    With refill_after_live the hub's own messages are sent too, as relay.on_bar and relay._fill do (PE1 17:48,
    asserts unchanged): {"t": "gap", since, until} just before the first live minute, {"t": "filled", same} right
    after the last refill. test_l2_relay_* pins that the real relay sends them."""
    out = []
    first_live = refill_at = None
    if away is not None and refill_after_live:
        first_live = min(c for c in (int(t.value) + M for t in mins.index) if c > away[1])
        refill_at = max(back_at, first_live + lag + 1)
    for ts, r in mins.iterrows():
        close = int(ts.value) + M
        if close in lost:
            continue
        if away is not None and away[0] < close <= away[1]:
            out.append((refill_at if refill_after_live else back_at, hub_msg(iid, close, r, refilled=True)))
        else:
            out.append((close + lag, hub_msg(iid, close, r)))
    if refill_at is not None:
        span = {"id": iid, "since": away[0] + M, "until": away[1]}
        out.append((first_live + lag, {"t": "gap", **span, "_order": first_live - 1}))  # before the live minute
        out.append((refill_at, {"t": "filled", **span, "_order": away[1] + 1}))  # after the last refill
    out = sorted(out, key=lambda a: (a[0], a[1].get("_order", a[1].get("ts"))))
    return [(at, {k: v for k, v in m.items() if k != "_order"}) for at, m in out]


def recover_from(mins: pd.DataFrame, available=None):
    """Decoder recover: the stored minutes closing strictly between two times; available(close, now) says
    whether the store has that minute yet (None: always)."""
    rows = [(int(ts.value) + M, r.open, r.high, r.low, r.close, r.volume) for ts, r in mins.iterrows()]

    def recover(_iid, after, before):
        return [r for r in rows if after < r[0] < before and (available is None or available(r[0]))]

    return recover


# ------------------------------------------------------------------------------------- the hub-fed paper node


class Run:
    def __init__(self, orders, fills, events, decoder, strategy_bars, store):
        self.orders, self.fills, self.events, self.decoder = orders, fills, events, decoder
        self.bars, self.store = strategy_bars, store

    def exits(self):
        return [o for o in self.orders if o["intent"] != "entry"]

    def kinds(self, kind):
        return [e for e in self.events if e["kind"] == kind]

    def sequence(self):
        """(intent, side, fill minute ceil, avg fill price) per order with a fill, in fill order."""
        by = {}
        for f in self.fills:
            q, n, ts = by.get(f["order_id"], (0.0, 0.0, f["ts"]))
            by[f["order_id"]] = (q + f["qty"], n + f["qty"] * f["price"], ts)
        intents = {o["order_id"]: (o["intent"], o["side"]) for o in self.orders}
        out = []
        for oid, (q, n, ts) in by.items():
            t = pd.Timestamp(ts).tz_convert(timezone.utc)
            out.append((*intents[oid], t if t == t.floor("1min") else t.ceil("1min"), n / q))
        return out


def paper(prices, *, enter=5, leave=60, side=1, minutes=1, perp=False, profile="aggressive", stop=0.01, tp=None,
          extra=None, lag=S // 2, gone=frozenset(), away=None, back_at=None, refill_after_live=False, lost=(), recover=None,
          held=None, heartbeat=None, history=None, warmup=0, end=None, holes=()) -> Run:
    """The hub-fed paper node replayed (see the module doc). held=(qty, entry_px, entry_ts_ns): a restart with
    that position in the journal (signed qty), its entry order journaled with this stop and target; heartbeat:
    the previous process's last heartbeat (ns); history: 1-minute Bars for the strategy's history_loader."""
    from decimal import Decimal

    from nautilus_trader.backtest import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.common import LoggerConfig, LogLevel
    from nautilus_trader.model import AccountType, BarType, Currency, Money, OmsType, TraderId

    from sleeve_fund import markets
    from sleeve_fund.instruments import ScheduleFeeModel, fill_model
    from sleeve_fund.paper.hub_client import Decoder
    from sleeve_fund.paper.runtime import SleeveRuntime
    from sleeve_fund.store import Store, utcnow
    from sleeve_fund.strategies import REGISTRY
    from sleeve_fund.venues import venue

    K = venue("KRAKEN")
    inst = K.instrument("BTC", "USD", price_precision=1)
    params = {"enter": enter, "leave": leave, "side": side, "stop_loss": stop, **(extra or {})}
    if tp:
        params["take_profit"] = tp
    if perp:
        params.update(market="perp", allow_short=True)
    fees = markets.fees_for(params, K.fees)
    store = Store.in_memory()
    store.create_sleeve(name="q146", strategy="probe", instrument="BTC/USD", bar_spec=f"{minutes}-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, risk_profile=profile, params=params)
    if held is not None:
        qty, px, ts = held
        at = datetime.fromtimestamp(ts / 1e9, tz=timezone.utc)
        sig = {"stop_frac": stop} | ({"tp_frac": tp} if tp else {})
        store.record_order("q146", order_id="O-held", side="BUY" if qty > 0 else "SELL", qty=abs(qty), intent="entry",
                           reason="held over the restart", signal=sig, ts=at)
        fee = abs(qty) * px * float(fees.taker)
        store.record_fill("q146", side="BUY" if qty > 0 else "SELL", qty=abs(qty), price=px, fee=fee,
                          order_id="O-held", trade_id="T-held", ts=at)
    if heartbeat is not None:
        from sqlalchemy import update

        from sleeve_fund.store import sleeves_t

        with store.engine.begin() as c:
            c.execute(update(sleeves_t).where(sleeves_t.c.name == "q146").values(
                heartbeat_at=datetime.fromtimestamp(heartbeat / 1e9, tz=timezone.utc)))
    runtime = SleeveRuntime(store, "q146", tick_seconds=30)
    book = runtime.book
    usd, btc = Currency.from_str("USD"), Currency.from_str("BTC")
    if perp:
        balances = [Money(book["cash"] + book["qty"] * (book["entry_px"] or 0.0), usd)]
    else:
        balances = [Money(book["cash"], usd)] + ([Money(book["qty"], btc)] if book["qty"] > 0 else [])
    engine = BacktestEngine(BacktestEngineConfig(trader_id=TraderId.from_str("QA146-001"),
                                                 logging=LoggerConfig(stdout_level=LogLevel.ERROR)))
    fee_model = ScheduleFeeModel(fees)
    reports = []
    try:
        engine.add_venue(venue=inst.id.venue, oms_type=OmsType.NETTING,
                         account_type=AccountType.MARGIN if perp else AccountType.CASH,
                         default_leverage=Decimal(markets.VENUE_LEVERAGE) if perp else None,
                         base_currency=None, starting_balances=balances, fee_model=fee_model,
                         fill_model=fill_model(), bar_execution=False)
        engine.add_instrument(inst)
        mins = minutes_of(prices)
        if holes:  # minutes the hub never had (no trades reached it): not sent, not stored
            mins = mins.drop([mins.index[k] for k in holes])
            gone = frozenset(gone) | {k * 60 + s for k in holes for s in range(60)}
        spec = f"{minutes}-MINUTE-LAST-EXTERNAL"
        dec = Decoder(spec, report=lambda level, kind, msg: reports.append({"kind": kind, "message": msg}),
                      recover=recover)
        bars = []
        for at, m in schedule(mins, str(inst.id), lag, away, back_at, refill_after_live, lost):
            if end is not None and at > end:
                continue
            got = dec(m, at)
            bars += got if isinstance(got, list) else [got] if got is not None else []
        data = ticks(inst, prices, gone)
        if end is not None:
            data = [d for d in data if d.ts_init <= end]
        engine.add_data(data + bars)
        cls, cfg_cls = REGISTRY["probe"]
        cfg = cfg_cls(instrument_id=inst.id, bar_type=BarType.from_str(f"{inst.id}-{spec}"), max_notional=None,
                      assumed_taker_fee=float(fees.taker), warmup_bars=warmup, **params)
        st = cls(cfg).attach_runtime(runtime)
        if history is not None:
            st.attach_history(history(inst))
        st.simulated_venue, st.fee_model, st.hub_fed = True, fee_model, True
        engine.add_strategy(st)
        engine.run()
        orders = sorted(store.orders("q146", limit=100_000), key=lambda o: (o["ts"], o["order_id"]))
        fills = list(reversed(store.fills("q146", limit=100_000)))
        events = list(reversed(store.events("q146", limit=100_000))) + reports
        return Run(orders, fills, events, dec, list(st.accepted), store)
    finally:
        runtime.now = utcnow
        engine.dispose()


def backtest(prices, *, enter=5, leave=60, side=1, minutes=1, perp=False, profile="aggressive", stop=0.01, tp=None,
             extra=None, holes=()):
    from sleeve_fund.instruments import BOOK_SHARE
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.venues import venue

    inst = venue("KRAKEN").instrument("BTC", "USD", price_precision=1)
    params = {"enter": enter, "leave": leave, "side": side, "stop_loss": stop, **(extra or {})}
    if tp:
        params["take_profit"] = tp
    if perp:
        params.update(market="perp", allow_short=True)
    m = minutes_of(prices)
    if holes:
        m = m.drop([m.index[k] for k in holes])
    bars = m.set_axis(m.index + pd.Timedelta(minutes=1))
    bars["volume"] = 60 / BOOK_SHARE
    kw = {}
    if minutes > 1:
        dec = bars.resample(f"{minutes}min", label="right", closed="right").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
        kw = {"exec_prices": bars, "exec_minutes": 1}
        bars = dec
    res = run_backtest("probe", bars, inst, params=params, starting_capital=10_000, risk_profile=profile,
                       bar_minutes=minutes, half_spread=SPREAD / 2 / BASE, **kw)
    j = res.journal
    orders = {o["order_id"]: o for o in j.orders_.values()} if isinstance(j.orders_, dict) else j.orders_
    seq = {}
    for f in j.fills_:
        q, n, ts = seq.get(f["order_id"], (0.0, 0.0, f["ts"]))
        seq[f["order_id"]] = (q + f["qty"], n + f["qty"] * f["price"], ts)
    out = []
    for oid, (q, n, ts) in seq.items():
        o = orders[oid]
        t = pd.Timestamp(ts).tz_convert(timezone.utc)
        out.append((o["intent"], o["side"], t if t == t.floor("1min") else t.ceil("1min"), n / q))
    return out


def minute(k: float) -> pd.Timestamp:
    return pd.Timestamp(START + int(k * M), unit="ns", tz="UTC")


def first_exit(seq):
    return next(r for r in seq if r[0] != "entry")


SETUPS = [  # (label, perp, profile, side): spot and a perp at 1x, 2x, 3x, long and short
    ("spot-long", False, "aggressive", 1),
    ("perp-1x-long", True, "conservative", 1),
    ("perp-2x-long", True, "balanced", 1),
    ("perp-3x-long", True, "aggressive", 1),
    ("perp-1x-short", True, "conservative", -1),
    ("perp-3x-short", True, "aggressive", -1),
]
SETUP_IDS = [s[0] for s in SETUPS]


def adverse(side: int, depth: float) -> float:
    """The factor that moves the price `depth` against a position on `side`."""
    return 1 - depth if side > 0 else 1 + depth


def favourable(side: int, depth: float) -> float:
    return 1 + depth if side > 0 else 1 - depth


def stored_history(prices, upto_ns: int):
    """A history_loader over the store's 1-minute bars closing at or before upto_ns (what the hub wrote)."""
    mins = minutes_of(prices)

    def make(inst):
        from nautilus_trader.model import Bar, BarType, Price, Quantity

        bt = BarType.from_str(f"{inst.id}-1-MINUTE-LAST-INTERNAL")
        rows = [Bar(bt, Price(r.open, 1), Price(r.high, 1), Price(r.low, 1), Price(r.close, 1), Quantity(60, 8),
                    int(ts.value) + M, int(ts.value) + M) for ts, r in mins.iterrows() if int(ts.value) + M <= upto_ns]

        def load(instrument, bar_type, limit):
            assert "1-MINUTE" in str(bar_type)
            return rows[-limit:]

        load.source = "QA stored minutes"
        return load

    return make


def restart(prices, down_from: float, back_at: float, *, entry_px=None, qty=0.05, side=1, heartbeat=None, **kw):
    """The process was down from minute `down_from` (its last heartbeat, unless `heartbeat` says otherwise) to
    `back_at`; the journal holds a position entered at minute 5. The new process gets trades and live hub minutes
    from back_at on, and the store's minutes up to back_at."""
    entry_px = entry_px or float(prices[5 * 60])
    gone = frozenset(range(0, int(back_at * 60)))
    lost = {START + k * M for k in range(0, int(back_at) + 1)}
    hb = START + int((heartbeat if heartbeat is not None else down_from) * M)
    kw.setdefault("history", stored_history(prices, START + int(back_at * M)))
    return paper(prices, side=side, gone=gone, lost=lost, held=(side * qty, entry_px, START + 5 * M), heartbeat=hb,
                 **kw)


# ------------------------------------------------------------------------------------------- shared checks

AWAY = (START + 6 * M, START + 12 * M)  # hub away 00:06:00-00:12:00: minutes closing 00:07-00:12 refilled ...
BACK = START + 12 * M + 5 * S  # ... at 00:12:05, those closing 00:07-00:10 over 90 s late
GONE = frozenset(range(6 * 60, 12 * 60))


def no_late_entries(run: Run) -> None:
    """No entry or addition was decided on a bar that reached the node more than 90 s after its close."""
    opening = {o["order_id"] for o in run.orders if o["intent"] in ("entry",) or
               (o["intent"] == "rebalance" and o["side"] == ("BUY" if o.get("qty", 0) >= 0 else "SELL"))}
    for t in run.store.timings("q146", limit=10_000):
        if t["order_id"] in opening and t["bar_close"] is not None:
            assert (t["bar_recv"] - t["bar_close"]).total_seconds() <= 90, t


def fill_px(run: Run, intent: str) -> float:
    return next(r[3] for r in run.sequence() if r[0] == intent)


def filled_orders(run: Run, intent: str) -> list[dict]:
    """The journal rows of `intent` that filled. Paper's stop is also journaled as a row of its own (P1-U35: intent
    stop_loss, order_type 'STOP (watched)', never filled, closed when the position closes); the market stop-loss sent
    when it fires is the row that fills, and the one these cells mean."""
    filled = {f["order_id"] for f in run.fills}
    return [o for o in run.orders if o["intent"] == intent and o["order_id"] in filled]


def price_at(prices, minute_k: float) -> float:
    return float(prices[int(minute_k * 60)])


# Advisor #146 L12 FINAL (6 Oct 19:30, + 19:38, 19:40 exact touch): phase 1 targets are market-on-touch, taker. Paper
# live books the real fill and records the level; the backtest, and paper's OUTAGE REPLAY (no real fill), book every
# target fill (touch or gap) at the level less max(half spread, 0.05 %), taker fee, never the open; the replay row
# is marked as a modelled price. (Supersedes "the target at its level" in the pins below, moved 6 Oct ~19:50.)
TP_SLIP = max(SPREAD / 2 / BASE, 0.0005)


def tp_model(level: float, side: int) -> float:
    return level * (1 - side * TP_SLIP)


def is_modelled(signal: dict | None) -> bool:
    """The replay row says its price is modelled, not a fill: a key or a text value naming it."""
    return any("model" in str(k).lower() or (isinstance(v, str) and "model" in v.lower())
               for k, v in (signal or {}).items())


# ======================================================== check 2: C1 on reconnect (hub away, node running)


_NA1_XF = pytest.mark.xfail(strict=True, reason="P1-L6 (#146, Advisor NA-1): the outage exit is booked from the "
                            "replayed missed minutes: a stop at its level or worse, on a gap at the open of the "
                            "crossing minute, a target at its level, with the market-on-return price recorded beside "
                            "the fill; ebe39c5 books market on return")
OUTAGE_CASES = [pytest.param(w, id=w) for w in ("stop", "target", "both_one_minute", "gap_through_stop")]  # PE2 (stop-safety, master 02845fb6): _NA1_XF removed, these pass


def _outage_prices(what: str, side: int, n_minutes: int):
    """stop: 2 % against 00:07:30-00:07:50, recovered; target: 3 % in favour, recovered; both: the spike then the dip
    inside the minute to 00:08 (adverse first: the stop); gap: the minute to 00:08 OPENS 3 % against (through the
    1 % stop) and the price is back to 1.5 % against from 00:09, still past the stop on return."""
    p = flat_prices(n_minutes)
    if what == "stop":
        return shape(p, 7.5, 7 + 50 / 60, adverse(side, 0.02))
    if what == "target":
        return shape(p, 7.5, 7 + 50 / 60, favourable(side, 0.03))
    if what == "both_one_minute":
        return shape(shape(p, 7 + 5 / 60, 7 + 15 / 60, favourable(side, 0.03)), 7.5, 7 + 40 / 60, adverse(side, 0.02))
    if what == "gap_through_stop":
        return shape(shape(p, 7.0, 9.0, adverse(side, 0.03)), 9.0, n_minutes, adverse(side, 0.015))
    return p


def assert_na1_price(run: Run, p, what: str, side: int, back: float) -> None:
    """Advisor NA-1 (6 Oct 16:53): the stop at its level or worse (a few bp of spread, no more); on a gap, the open
    of the minute that crossed it; the target at its level less TP_SLIP, marked modelled (L12 FINAL 19:30); the
    market-on-return price recorded beside the fill."""
    ex = first_exit(run.sequence())
    entry = fill_px(run, "entry")
    if what == "target":
        level = entry * (1 + side * 0.02)
        assert ex[0] == "take_profit" and abs(ex[3] / tp_model(level, side) - 1) < 1e-4, (ex, tp_model(level, side))
        (o,) = [o for o in run.orders if o["intent"] == "take_profit"]
        assert is_modelled(o["signal"]), o["signal"]
    elif what == "gap_through_stop":  # Advisor 20:39: a replayed stop books as the backtest: the gap open less TP_SLIP
        gap_open = price_at(p, 7.0)
        assert ex[0] == "stop_loss" and abs(ex[3] / tp_model(gap_open, side) - 1) < 1e-4, (ex, tp_model(gap_open, side))
    else:  # Advisor 20:39: the level less max(half spread, 0.05 %)
        level = entry * (1 - side * 0.01)
        assert ex[0] == "stop_loss" and abs(ex[3] / tp_model(level, side) - 1) < 1e-4, (ex, tp_model(level, side))
    (o,) = filled_orders(run, ex[0])
    assert is_modelled(o["signal"]), o["signal"]  # Advisor 20:39: every replayed exit row is marked modelled
    found = {k: v for k, v in (o["signal"] or {}).items() if "return" in k.lower()}
    assert found and all(abs(float(v) / back - 1) < 2e-3 for v in found.values()), o["signal"]


@pytest.mark.parametrize("label, perp, profile, side", SETUPS, ids=SETUP_IDS)
@pytest.mark.parametrize("what", OUTAGE_CASES)
def test_c1_reconnect_long_outage_late_refills_exit_on_the_missed_high_low(label, perp, profile, side, what):
    """Hub away 00:06-00:12 while in the position; the refills come back at 00:12:05, those closing 00:07-00:10
    over 90 s late. The exit comes on their return, not at the 00:25 signal, with no entry from a late bar, and is
    booked at the NA-1 price (assert_na1_price). Rewritten after the Advisor's NA-1 ruling: it used to accept the
    market-on-return price."""
    p = _outage_prices(what, side, 30)
    run = paper(p, side=side, perp=perp, profile=profile, tp=0.02, leave=25, gone=GONE, away=AWAY, back_at=BACK)
    seq = run.sequence()
    assert seq[0][0] == "entry" and seq[0][2] == minute(5)
    assert first_exit(seq)[0] == ("take_profit" if what == "target" else "stop_loss"), seq
    assert minute(12) <= first_exit(seq)[2] <= minute(13), seq
    no_late_entries(run)
    assert len([r for r in seq if r[0] == "entry"]) == 1  # nothing re-opened from the late or recovered bars
    assert_na1_price(run, p, what, side, back=price_at(p, 12 + 5 / 60))


@pytest.mark.parametrize("label, perp, profile, side", SETUPS, ids=SETUP_IDS)
def test_c1_reference_the_backtest_on_the_same_minutes_exits_inside_the_outage(label, perp, profile, side):
    p = shape(flat_prices(30), 7.5, 7 + 50 / 60, adverse(side, 0.02))
    bt = backtest(p, side=side, perp=perp, profile=profile, tp=0.02, leave=25)
    assert first_exit(bt)[0] == "stop_loss" and first_exit(bt)[2] == minute(8)


def _short_outage(side):
    """Hub away 00:06:20-00:07:30 (70 s, e.g. a hub container restart): the minute to 00:07 is refilled at
    00:07:30, 30 s after its close, so not late. The price dips 2 % against the position 00:06:30-00:06:50."""
    p = shape(flat_prices(30), 6.5, 6 + 50 / 60, adverse(side, 0.02))
    return p, frozenset(range(6 * 60 + 20, 7 * 60 + 30)), (START + 6 * M, START + 7 * M), START + 7 * M + 30 * S


@pytest.mark.parametrize("label, perp, profile, side", SETUPS, ids=SETUP_IDS)
def test_c1_short_outage_the_backtest_stops_out_in_the_minute_to_00_07(label, perp, profile, side):
    p, *_ = _short_outage(side)
    assert first_exit(backtest(p, side=side, perp=perp, profile=profile, leave=20))[:1] == ("stop_loss",)


# PE2 (stop-safety, master 02845fb6): passes (mark removed)
@pytest.mark.parametrize("label, perp, profile, side", SETUPS, ids=SETUP_IDS)
def test_l1_short_outage_the_refilled_minute_on_time_still_has_its_stop_checked(label, perp, profile, side):
    p, gone, away, back = _short_outage(side)
    run = paper(p, side=side, perp=perp, profile=profile, leave=20, gone=gone, away=away, back_at=back)
    ex = first_exit(run.sequence())
    assert ex[0] == "stop_loss" and ex[2] <= minute(8), run.sequence()


def _inside_15m(side):
    """15-minute bars; in the position from the bar to 00:15. Hub away 00:20:00-00:23:00 (refills at 00:23:05,
    joining the bar to 00:30, which is then sent on time); 2 % against 00:20:30-00:20:50, recovered."""
    p = shape(flat_prices(60), 20.5, 20 + 50 / 60, adverse(side, 0.02))
    return p, frozenset(range(20 * 60, 23 * 60)), (START + 20 * M, START + 23 * M), START + 23 * M + 5 * S


@pytest.mark.parametrize("label, perp, profile, side", SETUPS[:1] + SETUPS[3:4] + SETUPS[5:], ids=["spot-long",
                         "perp-3x-long", "perp-3x-short"])
def test_c1_15m_the_backtest_stops_out_in_the_minute_to_00_21(label, perp, profile, side):
    p, *_ = _inside_15m(side)
    bt = backtest(p, side=side, perp=perp, profile=profile, enter=15, leave=45, minutes=15)
    assert first_exit(bt)[0] == "stop_loss" and first_exit(bt)[2] == minute(21)


# PE2 (stop-safety, master 02845fb6): passes (mark removed)
@pytest.mark.parametrize("label, perp, profile, side", SETUPS[:1] + SETUPS[3:4] + SETUPS[5:], ids=["spot-long",
                         "perp-3x-long", "perp-3x-short"])
def test_l1_15m_a_stop_crossed_inside_an_on_time_bar_while_the_hub_was_away_runs(label, perp, profile, side):
    p, gone, away, back = _inside_15m(side)
    run = paper(p, side=side, perp=perp, profile=profile, enter=15, leave=45, minutes=15, gone=gone, away=away,
                back_at=back)
    assert run.decoder.late == 0  # the bar to 00:30 came on time
    ex = first_exit(run.sequence())
    assert ex[0] == "stop_loss" and ex[2] <= minute(31), run.sequence()


def _venue_drop(side):
    """The hub stays up but loses the venue 00:06-00:10. On its return the hub publishes the first live minute
    (to 00:11) and only then the REST refill of 00:07-00:10 (relay.on_bar: gap, refill submitted to a thread, live
    bar published). The client recovers from the store when the live minute shows the gap."""
    p = shape(flat_prices(30), 6.5, 6 + 50 / 60, adverse(side, 0.02))
    return p, frozenset(range(6 * 60, 10 * 60)), (START + 6 * M, START + 10 * M), START + 10 * M + 5 * S


@pytest.mark.parametrize("label, perp, profile, side", SETUPS[:1] + SETUPS[5:], ids=["spot-long", "perp-3x-short"])
def test_c1_refill_after_the_live_minute_is_caught_when_the_store_already_has_it(label, perp, profile, side):
    p, gone, away, back = _venue_drop(side)
    run = paper(p, side=side, perp=perp, profile=profile, leave=20, gone=gone, away=away, back_at=back,
                refill_after_live=True, recover=recover_from(minutes_of(p)))
    ex = first_exit(run.sequence())
    assert ex[0] == "stop_loss" and ex[2] <= minute(12), run.sequence()
    no_late_entries(run)


# PE2 (stop-safety, master 02845fb6): passes (mark removed)
@pytest.mark.parametrize("label, perp, profile, side", SETUPS[:1] + SETUPS[5:], ids=["spot-long", "perp-3x-short"])
def test_l2_refill_after_the_live_minute_with_the_store_not_yet_written_still_checks_the_stop(label, perp, profile,
                                                                                              side):
    p, gone, away, back = _venue_drop(side)
    run = paper(p, side=side, perp=perp, profile=profile, leave=20, gone=gone, away=away, back_at=back,
                refill_after_live=True, recover=recover_from(minutes_of(p), available=lambda close: False))
    assert [e for e in run.events if e["kind"] == "hub_gap"]  # the gap was told ...
    ex = first_exit(run.sequence())
    assert ex[0] == "stop_loss" and ex[2] <= minute(12), run.sequence()  # ... but the stop never ran


# ============================================================================== check 2: C1 on restart


RESTART_SETUPS = [SETUPS[0], SETUPS[2], SETUPS[3], SETUPS[5]]
RESTART_IDS = [s[0] for s in RESTART_SETUPS]


@pytest.mark.parametrize("label, perp, profile, side", RESTART_SETUPS, ids=RESTART_IDS)
@pytest.mark.parametrize("what", OUTAGE_CASES + [pytest.param("nothing", id="nothing")])
def test_c1_restart_checks_the_stored_minutes_since_the_last_heartbeat(label, perp, profile, side, what):
    """Process down 00:06-00:12 (last heartbeat 00:06) with the position entered at 00:05; the store has the
    minutes. On the first trade back (perp: once the restore order has put the position back) the stored highs
    and lows are checked, adverse first in one minute, and the exit is booked at the NA-1 price (assert_na1_price).
    Rewritten after the Advisor's NA-1 ruling: it used to accept the market-on-return price."""
    p = _outage_prices(what, side, 40)
    run = restart(p, 6, 12, side=side, perp=perp, profile=profile, tp=0.02, leave=30)
    seq = run.sequence()
    if what == "nothing":
        assert first_exit(seq)[0] == "exit" and first_exit(seq)[2] == minute(30) and not run.kinds("outage_exit")
        return
    assert first_exit(seq)[0] == ("take_profit" if what == "target" else "stop_loss"), seq
    assert first_exit(seq)[2] <= minute(13), seq
    assert len([r for r in seq if r[0] == "entry"]) == 1
    assert_na1_price(run, p, what, side, back=price_at(p, 12))


# PE2 (stop-safety, master 02845fb6): passes (mark removed)
@pytest.mark.parametrize("label, perp, profile, side", RESTART_SETUPS[:1] + RESTART_SETUPS[3:],
                         ids=["spot-long", "perp-3x-short"])
def test_l3_restart_after_a_hub_outage_checks_from_the_last_market_data_not_the_last_heartbeat(label, perp,
                                                                                               profile, side):
    """Hub gone from 00:06; the node heartbeats until its price watchdog stops it reporting at 00:21; the
    supervisor restarts it; the hub is back at 00:30 and the store has the minutes. The stop was crossed at 00:07."""
    p = shape(flat_prices(40), 7.5, 7 + 50 / 60, adverse(side, 0.02))
    run = restart(p, 6, 30, side=side, perp=perp, profile=profile, leave=38, heartbeat=21)
    assert first_exit(run.sequence())[0] == "stop_loss", run.sequence()


# ================================================================================= check 3: the late-bar rule

LATE_AWAY = (START + 3 * M, START + 10 * M)  # hub away 00:03-00:10; refills at 00:10:05: the bars to 00:04-00:08
LATE_BACK = START + 10 * M + 5 * S  # are over 90 s late, the one to 00:09 is 65 s late (on time)
LATE_GONE = frozenset(range(3 * 60, 10 * 60))


def late_run(stop=None, **kw) -> Run:
    return paper(flat_prices(30), gone=LATE_GONE, away=LATE_AWAY, back_at=LATE_BACK, stop=stop, **kw)


def guarded_stop(perp: bool, profile: str) -> float | None:
    """Set-up for stop safety: a perp above 1x gets GUARDED_STOP (a stopless one is refused); spot and 1x stay
    stopless. These prices never reach it, so the cell decides as it did stopless."""
    return GUARDED_STOP if perp and profile != "conservative" else None


def timing_of(run: Run, intent: str, side: str | None = None) -> dict:
    ids = {o["order_id"] for o in run.orders if o["intent"] == intent and (side is None or o["side"] == side)}
    return min((t for t in run.store.timings("q146", limit=1000) if t["order_id"] in ids), key=lambda t: t["decided"])


def hhmm(ts) -> str:
    return f"{pd.Timestamp(ts):%H:%M}"


@pytest.mark.parametrize("label, perp, profile, side", [SETUPS[0], SETUPS[3], SETUPS[4]],
                         ids=["spot-long", "perp-3x-long", "perp-1x-short"])
def test_late_bars_never_open_the_first_on_time_bar_does(label, perp, profile, side):
    run = late_run(enter=5, leave=20, side=side, perp=perp, profile=profile, stop=guarded_stop(perp, profile))
    assert run.decoder.late == 5
    no_late_entries(run)
    t = timing_of(run, "entry")
    assert hhmm(t["bar_close"]) == "00:09" and hhmm(t["bar_recv"]) == "00:10"  # the bar to 00:09, 65 s late
    skipped = run.kinds("late_entry_skipped")
    assert skipped and all("past the 90 s limit for opening" in e["message"] for e in skipped)


def test_late_bars_never_add_to_a_position():
    """Spot, weight-sized: 0.20 of the book from 00:02, 0.45 wanted from the bar to 00:06 (late): the addition
    waits for the on-time bar to 00:09."""
    run = late_run(enter=2, leave=20, extra={"w1": 0.2, "w2": 0.45, "add_at": 6, "rebalance_band": 0.1})
    t = timing_of(run, "rebalance", "BUY")
    assert hhmm(t["bar_close"]) == "00:09"
    assert any("Skipped a" in e["message"] and "addition on the 00:06 candle" in e["message"]
               for e in run.kinds("late_entry_skipped"))


def test_a_reduction_on_a_late_bar_runs():
    run = late_run(enter=2, leave=20, extra={"w1": 0.45, "w2": 0.2, "add_at": 6, "rebalance_band": 0.1})
    t = timing_of(run, "rebalance", "SELL")
    assert hhmm(t["bar_close"]) == "00:06" and (t["bar_recv"] - t["bar_close"]).total_seconds() > 90
    assert any("Exiting on the 00:06 candle" in e["message"] for e in run.kinds("late_exit"))


@pytest.mark.parametrize("label, perp, profile, side", [SETUPS[0], SETUPS[2], SETUPS[5]],
                         ids=["spot-long", "perp-2x-long", "perp-3x-short"])
def test_an_exit_on_a_late_bar_runs_and_is_said_with_its_lag(label, perp, profile, side):
    run = late_run(enter=2, leave=6, side=side, perp=perp, profile=profile, stop=guarded_stop(perp, profile))
    t = timing_of(run, "exit")
    assert hhmm(t["bar_close"]) == "00:06" and (t["bar_recv"] - t["bar_close"]).total_seconds() == 245
    (said,) = run.kinds("late_exit")
    assert "Exiting on the 00:06 candle 245 s after its close" in said["message"]


@pytest.mark.parametrize("side", [1, -1], ids=["long-to-short", "short-to-long"])
def test_a_reversal_on_a_late_bar_only_closes_the_new_side_waits_for_an_on_time_bar(side):
    run = late_run(enter=2, leave=6, side=side, perp=True, profile="aggressive", extra={"after": -side},
                   stop=GUARDED_STOP)
    assert hhmm(timing_of(run, "exit")["bar_close"]) == "00:06"  # closed on the late bar
    opened = [o for o in run.orders if o["intent"] == "entry"]
    assert [o["side"] for o in opened] == (["BUY", "SELL"] if side > 0 else ["SELL", "BUY"])
    t = timing_of(run, "entry", opened[1]["side"])
    assert hhmm(t["bar_close"]) == "00:09"  # the new side opened on the first on-time bar, not the late one
    assert any("entry after closing the" in e["message"] for e in run.kinds("late_entry_skipped"))
    no_late_entries(run)


def test_the_decoder_tells_a_run_of_late_bars_once_then_its_extent():
    run = late_run(enter=50, leave=60)
    late = run.kinds("bar_late")
    assert len(late) == 2 and "exits only" in late[1]["message"] and "5 bars" in late[1]["message"]


# PE2 (stop-safety, master 02845fb6): passes (mark removed)
def test_l4_a_run_of_late_bars_skips_its_entry_with_one_warning():
    run = late_run(enter=5, leave=20)
    assert len(run.kinds("late_entry_skipped")) == 1, [e["message"][:60] for e in run.kinds("late_entry_skipped")]


@pytest.mark.parametrize("minutes", [1, 15])
@pytest.mark.parametrize("lag, late", [(89 * S, False), (90 * S, False), (90 * S + 1, True), (91 * S, True)],
                         ids=["89s", "90s", "90s+1ns", "91s"])
def test_late_boundary_is_exactly_late_bar_seconds_in_the_decoder_and_the_strategy(minutes, lag, late):
    from sleeve_fund.paper.hub_client import LATE_BAR_SECONDS
    from sleeve_fund.strategies.base import LATE_DECISION_NS

    assert LATE_BAR_SECONDS == 90 and LATE_DECISION_NS == 90 * S
    run = paper(flat_prices(60), enter=15, leave=50, minutes=minutes, lag=lag, stop=None)
    assert (run.decoder.late > 0) is late
    assert bool([o for o in run.orders if o["intent"] == "entry"]) is not late
    assert bool(run.kinds("late_entry_skipped")) is late
    assert bool(run.kinds("bar_late")) is late


# ========================================================= check 4: order, duplicates, indicators after recovery


def _clean_bars(prices, minutes):
    return paper(prices, enter=10**6, leave=10**6 + 1, minutes=minutes, stop=None).bars


def _no_dupes_in_order(bars):
    ts = [b[0] for b in bars]
    assert ts == sorted(ts) and len(set(ts)) == len(ts)


WAVY = BASE * (1 + 0.004 * np.sin(np.arange(60 * 60) / 170) + 0.001 * np.sin(np.arange(60 * 60) / 7))


@pytest.mark.parametrize("minutes", [1, 5, 15])
@pytest.mark.parametrize("case", ["refills_before_live", "refills_after_live_store_has_them", "client_side_loss"])
def test_recovered_minutes_reach_the_strategy_oldest_first_once_and_its_bars_equal_a_clean_runs(minutes, case):
    away, back = (START + 16 * M, START + 23 * M), START + 23 * M + 5 * S
    kw = {"refills_before_live": dict(away=away, back_at=back),
          "refills_after_live_store_has_them": dict(away=away, back_at=back, refill_after_live=True,
                                                    recover=recover_from(minutes_of(WAVY))),
          "client_side_loss": dict(lost={START + k * M for k in range(17, 23)}, recover=recover_from(minutes_of(WAVY)))
          }[case]
    run = paper(WAVY, enter=10**6, leave=10**6 + 1, minutes=minutes, stop=None, **kw)
    _no_dupes_in_order(run.bars)
    assert run.bars == _clean_bars(WAVY, minutes)  # the indicators were fed exactly what a clean run's were


@pytest.mark.parametrize("minutes", [1, 15])
def test_refills_sent_again_after_a_reconnect_are_never_sent_twice(minutes):
    """The hub restarts again and re-sends the refills of 00:17-00:23 after the live minutes moved on (and once
    more mid-bar): no bar is sent twice and none changes."""
    from sleeve_fund.paper.hub_client import Decoder

    mins = minutes_of(WAVY)
    iid = "BTC/USD.KRAKEN"
    msgs = schedule(mins, iid, away=(START + 16 * M, START + 23 * M), back_at=START + 23 * M + 5 * S)
    again = [(START + 25 * M + 10 * S, m) for at, m in msgs if START + 16 * M < m["ts"] <= START + 23 * M]
    again += [(START + 31 * M + 10 * S, m) for at, m in msgs if START + 26 * M < m["ts"] <= START + 31 * M]
    d1, d2, out1, out2 = Decoder(f"{minutes}-MINUTE-LAST-EXTERNAL"), Decoder(f"{minutes}-MINUTE-LAST-EXTERNAL"), [], []
    for d, out, feed in ((d1, out1, msgs), (d2, out2, sorted(msgs + again, key=lambda a: (a[0], a[1]["ts"])))):
        for at, m in feed:
            got = d(m, at)
            out += got if isinstance(got, list) else [got] if got is not None else []
    key = [(b.ts_event, str(b.open), str(b.high), str(b.low), str(b.close), str(b.volume)) for b in out2]
    assert key == [(b.ts_event, str(b.open), str(b.high), str(b.low), str(b.close), str(b.volume)) for b in out1]
    assert len({k[0] for k in key}) == len(key)


# PE2 (stop-safety, master 02845fb6): passes (mark removed)
@pytest.mark.parametrize("minutes", [1, 15])
def test_l2_refills_after_the_live_minute_without_the_store_still_reach_the_indicators(minutes):
    run = paper(WAVY, enter=10**6, leave=10**6 + 1, minutes=minutes, stop=None, away=(START + 26 * M, START + 33 * M),
                back_at=START + 33 * M + 5 * S, refill_after_live=True,
                recover=recover_from(minutes_of(WAVY), available=lambda close: False))
    assert run.bars == _clean_bars(WAVY, minutes)


# ============================================================= check 6: parity with the backtest, hole and late


def _same_decisions(hub, bt, slack_minutes=0):
    assert [r[:2] for r in hub] == [r[:2] for r in bt], (hub, bt)
    for h, b in zip(hub, bt):
        assert abs((h[2] - b[2]).total_seconds()) <= 60 * slack_minutes, (h, b)
        assert abs(h[3] / b[3] - 1) < 0.0005, (h, b)  # within 5 bp: the paper book's spread is in its price


@pytest.mark.parametrize("label, perp, profile, side", [SETUPS[0], SETUPS[3], SETUPS[5]],
                         ids=["spot-long", "perp-3x-long", "perp-3x-short"])
def test_parity_whole_feed_hub_paper_decides_as_the_backtest(label, perp, profile, side):
    kw = dict(enter=5, leave=25, side=side, perp=perp, profile=profile, stop=guarded_stop(perp, profile))
    _same_decisions(paper(WAVY[:40 * 60], **kw).sequence(), backtest(WAVY[:40 * 60], **kw))


@pytest.mark.parametrize("label, perp, profile, side", [SETUPS[0], SETUPS[5]], ids=["spot-long", "perp-3x-short"])
def test_parity_a_hole_neither_the_hub_nor_the_store_has_gives_the_same_decisions(label, perp, profile, side):
    """The minutes to 00:05 (the entry bar) and 00:19-00:21 never existed (no trades): both decide on the next
    bar there is."""
    holes = (4, 18, 19, 20)
    kw = dict(enter=5, leave=20, side=side, perp=perp, profile=profile, stop=guarded_stop(perp, profile), holes=holes)
    run = paper(WAVY[:40 * 60], recover=recover_from(minutes_of(WAVY[:40 * 60]).drop(
        [minutes_of(WAVY[:40 * 60]).index[k] for k in holes])), **kw)
    bt = backtest(WAVY[:40 * 60], **kw)
    assert run.sequence()[0][2] == minute(6) and first_exit(run.sequence())[2] == minute(22)
    _same_decisions(run.sequence(), bt)
    assert run.kinds("hub_gap")  # the hole is told


@pytest.mark.parametrize("label, perp, profile, side", [SETUPS[0], SETUPS[5]], ids=["spot-long", "perp-3x-short"])
def test_parity_late_case_same_decisions_entries_only_on_the_first_on_time_bar(label, perp, profile, side):
    """Hub away 00:03-00:10 (refills at 00:10:05). Exit side: the exit signalled on the late bar to 00:06 runs
    when it arrives. Entry side: the backtest enters at 00:05; paper skips the late bars and enters on the bar to
    00:09 (the Advisor's late rule): the same decisions, the entry later by design."""
    kw = dict(side=side, perp=perp, profile=profile, stop=guarded_stop(perp, profile))
    p = WAVY[:30 * 60]
    out = paper(p, enter=2, leave=6, gone=LATE_GONE, away=LATE_AWAY, back_at=LATE_BACK, **kw).sequence()
    ref = backtest(p, enter=2, leave=6, **kw)
    assert [r[:2] for r in out] == [r[:2] for r in ref] and out[1][2] == minute(11) and ref[1][2] == minute(6)
    inn = paper(p, enter=5, leave=20, gone=LATE_GONE, away=LATE_AWAY, back_at=LATE_BACK, **kw).sequence()
    ref = backtest(p, enter=5, leave=20, **kw)
    assert [r[:2] for r in inn] == [r[:2] for r in ref] and ref[0][2] == minute(5) and inn[0][2] == minute(11)
    assert inn[1][2] == ref[1][2] == minute(20)


# ===================================================== check 5: P1-C10, a saved strategy on venue candles


def _supervised(tmp_path, monkeypatch, sleeves):
    from types import SimpleNamespace

    from sleeve_fund import supervisor
    from sleeve_fund.store import Store

    store = Store(f"sqlite:///{tmp_path}/c10.db")
    for name, venue_, spec, strategy, params, *profile in sleeves:  # profile: the risk profile, when set
        store.create_sleeve(name=name, strategy=strategy, instrument="BTC/USDT" if venue_ == "binance" else "BTC/USD",
                            bar_spec=spec, starting_balance=1_000, venue=venue_, params=params,
                            **({"risk_profile": profile[0]} if profile else {}))
    started = []
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **k: started.append(a[0][-1]) or
                        SimpleNamespace(pid=1, poll=lambda: None, send_signal=lambda *a: None, wait=lambda **k: 0))
    return store, supervisor.Supervisor(store), started


VENUE_CANDLES = ["1-HOUR-LAST-EXTERNAL", "4-HOUR-LAST-EXTERNAL", "1-DAY-LAST-EXTERNAL"]


@pytest.mark.parametrize("spec", VENUE_CANDLES)
def test_c10_a_saved_binance_strategy_on_venue_candles_is_refused_at_start_with_a_reason_and_no_loop(tmp_path,
                                                                                                  monkeypatch, spec):
    store, sup, started = _supervised(tmp_path, monkeypatch, [
        ("old", "binance", spec, "trend_filter", {"market": "perp"}),
        # conservative (1x): a stopless perp above 1x is refused by stop safety (check_perp_stop), not by C10
        ("hub-hourly", "binance", "1-HOUR-LAST-INTERNAL", "buy_and_hold", {"market": "perp"}, "conservative"),
        ("kraken-daily", "kraken", "1-DAY-LAST-EXTERNAL", "buy_and_hold", {}),
    ])
    for _ in range(5):
        sup.step()
    s = store.sleeve("old")
    assert "old" not in started and sorted(started) == ["hub-hourly", "kraken-daily"]
    assert s.desired_state == "stopped" and s.status == "stopped"
    size = "-".join(spec.split("-")[:2]).lower()
    assert s.status_reason == (f"not started: bar_spec: strategies on this market decide on bars built from the market "
                               f"data hub's minutes, so the venue's own {size} candles aren't available there; choose "
                               "1, 5 or 15-minute or 1-hour bars")
    kinds = [e["kind"] for e in store.events("old", limit=100)]
    assert kinds.count("start_refused") == 1 and "process_crash" not in kinds  # once, no crash loop
    assert not [e for e in store.events(None, limit=100) if e["kind"] == "supervisor_error"]


def test_c10_the_pm_starting_it_again_is_refused_again_once_per_start(tmp_path, monkeypatch):
    store, sup, started = _supervised(tmp_path, monkeypatch, [("old", "binance", "1-DAY-LAST-EXTERNAL",
                                                               "buy_and_hold", {"market": "perp"})])
    sup.step()
    store.set_desired_state("old", "running")
    sup.step()
    sup.step()
    assert started == [] and [e["kind"] for e in store.events("old", limit=100)].count("start_refused") == 2


# Stop safety (NA-4, "S6 applies to every refused start that holds a position") supersedes this pin's "not started,
# an engineer needs to close it": on the stop-safety branch the supervisor starts a stopped holder for its exits only
# (supervisor.STOPPED_HOLDING) on the next step, at any risk profile. For a venue-candle strategy that is the C10
# exits-only build (owner PE1; build_node on the hub's minutes), not in the stop-safety PR: the same strict condition
# as NA-4's venue-candle cells. Rewrite this cell's start assertions when that build lands.
def _stopped_holders_run_exits_only() -> bool:
    """Whether the supervisor under test starts a stopped holder for its exits only (stop safety; not on main)."""
    try:
        from sleeve_fund import supervisor

        return hasattr(supervisor, "STOPPED_HOLDING")
    except Exception:  # noqa: BLE001 - a head that can't say has not built it
        return False


_C10_EXITS_ONLY_XF = pytest.mark.xfail(
    condition=_stopped_holders_run_exits_only(), strict=True, raises=AssertionError,
    reason="QA P1-C10, owner PE1 (the C10 exits-only build, not in the stop-safety PR): with stop safety a refused "
    "venue-candle holder is started for its exits only on the next step (NA-4)")


@_C10_EXITS_ONLY_XF
def test_c10_refused_while_holding_a_position_says_so_never_flattens_and_leaves_the_journal_alone(tmp_path,
                                                                                                monkeypatch):
    """The CR minor (ebe39c5). It is refused even with a flatten waiting, and a PM reset can't get round it:
    nothing is started, nothing is sold, the position stays in the journal and the status says an engineer must
    close it. Nothing watches its stop or target meanwhile (no process): see Needs Advisor."""
    store, sup, started = _supervised(tmp_path, monkeypatch, [("old", "binance", "1-DAY-LAST-EXTERNAL",
                                                               "buy_and_hold", {"market": "perp"})])
    store.record_order("old", order_id="o1", side="BUY", qty=0.01, intent="entry", reason="t",
                       signal={"stop_frac": 0.02})
    store.record_fill("old", side="BUY", qty=0.01, price=100.0, fee=0.008, order_id="o1", trade_id="t1")
    before = store.journal_book("old", 1_000)
    store.command("old", "flatten", "Book kill switch: test", actor="PM")
    for _ in range(3):
        sup.step()
    store.request_reset("old", "QA: reset while refused")
    for _ in range(3):
        sup.step()
    s = store.sleeve("old")
    assert started == [] and s.desired_state == "stopped"
    assert s.status_reason.endswith("It still holds a position (0.01), which a flatten can't close while it can't "
                                    "start: an engineer needs to close it")
    assert store.journal_book("old", 1_000) == before  # never flattened, never orphaned from the journal
    assert [o["order_id"] for o in store.orders("old", limit=10)] == ["o1"]
    refused = [e for e in store.events("old", limit=100) if e["kind"] == "start_refused"]
    assert 1 <= len(refused) <= 2 and all(e["level"] == "error" for e in refused)  # an alert, not a loop


def test_c10_a_sleeve_with_no_venue_or_an_unknown_one_does_not_break_the_supervisor(tmp_path, monkeypatch):
    store, sup, started = _supervised(tmp_path, monkeypatch, [("plain", None, "1-DAY-LAST-EXTERNAL", "buy_and_hold",
                                                               {})])
    sup.step()
    assert started == ["plain"]  # the default venue (no hub) still starts on its own candles


def test_c10_the_weight_guard_fixture_change_is_setup_only():
    """test_perp_weight_guard's `_existing` moved from 1-DAY-LAST-EXTERNAL to 1-HOUR-LAST-INTERNAL: on the old
    spec the new hub check refuses first (so the weight guard's assertions could not be reached); on the new
    one the hub check passes and the weight guard still refuses with its own message."""
    from sleeve_fund.paper.config import check_hub_bar_spec
    from sleeve_fund.strategies import PERP_WEIGHT_REFUSAL, check_perp_sizing

    with pytest.raises(ValueError, match="venue's own 1-day candles"):
        check_hub_bar_spec("BINANCE", "1-DAY-LAST-EXTERNAL")
    check_hub_bar_spec("BINANCE", "1-HOUR-LAST-INTERNAL")
    with pytest.raises(ValueError, match=PERP_WEIGHT_REFUSAL[:20]):
        check_perp_sizing("donchian", {"market": "perp"})


# ===================================================== orders: the journal write is off the decision path


# PE2 (stop-safety, master 02845fb6): passes (mark removed)
def test_l5_a_failed_order_journal_write_still_stops_the_order_before_the_venue():
    from sleeve_fund.paper.queued import QueuedStore
    from sleeve_fund.store import Store

    db = Store.in_memory()
    db.create_sleeve(name="q", strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                     starting_balance=1_000)

    def fail(*a, **k):
        raise RuntimeError("database gone")

    db.record_order = fail
    qs = QueuedStore(db)
    with pytest.raises(Exception):  # inline before #146: the strategy's _submit stops here, before submit_order
        qs.record_order("q", order_id="O-1", side="BUY", qty=0.1, intent="entry", reason="t")
    qs.flush()


# ================================ Advisor rulings of 6 Oct 16:53 on NA-1, NA-2, NA-3 (HoE: in #146 with L1-L3)
#
# The outage is replayed from the venue's missed minutes, as the resting venue stops of the pre-G2 design would
# have traded them; the backtest follows the same rule. Paths: "reconnect" (hub away 00:06-00:12, refills >90 s
# late at 00:12:05, the case the PR handles) and "restart" (process down 00:06-00:12, stored minutes).

NA_SETUPS = [SETUPS[0], SETUPS[1], SETUPS[3], SETUPS[4], SETUPS[5]]  # spot, perp 1x/3x long, 1x/3x short
NA_IDS = [s[0] for s in NA_SETUPS]
ENTRY_PX = BASE * (1 + 1e-7 * 300)  # the price at 00:05, where the probe's entry fills


def _outage(path, p, *, side, perp, profile, tp=None, stop=0.01, leave=25, qty=0.05, history=None):
    if path == "reconnect":
        return paper(p, side=side, perp=perp, profile=profile, tp=tp, stop=stop, leave=leave, gone=GONE, away=AWAY,
                     back_at=BACK)
    kw = {} if history is None else {"history": history}
    run = restart(p, 6, 12, side=side, perp=perp, profile=profile, tp=tp, stop=stop, leave=leave, qty=qty, **kw)
    return run


def _entry(run) -> float:
    return fill_px(run, "entry")


def _bt_exit(p, **kw):
    return first_exit(backtest(p, **kw))


def _gap_prices(side):
    """The minute to 00:08 opens 3 % against the position (a gap through the 1 % stop), and the price comes back
    to 1.5 % against from 00:09 on: still past the stop on return, but not at the gap's price."""
    return shape(shape(flat_prices(30), 7.0, 9.0, adverse(side, 0.03)), 9.0, 30, adverse(side, 0.015))


# PE2 (stop-safety, master 02845fb6): passes (mark removed)
@pytest.mark.parametrize("label, perp, profile, side", NA_SETUPS, ids=NA_IDS)
@pytest.mark.parametrize("path", ["reconnect", "restart"])
def test_l6_na1_an_outage_stop_fills_at_its_level_like_the_backtest(path, label, perp, profile, side):
    p = shape(flat_prices(30), 7.5, 7 + 50 / 60, adverse(side, 0.02))
    run = _outage(path, p, side=side, perp=perp, profile=profile, tp=0.02)
    ex = first_exit(run.sequence())
    level = _entry(run) * (1 - side * 0.01)  # Advisor 20:39: the level less TP_SLIP, the row marked modelled
    assert ex[0] == "stop_loss" and abs(ex[3] / tp_model(level, side) - 1) < 1e-4, (ex, tp_model(level, side))
    (o,) = filled_orders(run, "stop_loss")
    assert is_modelled(o["signal"]), o["signal"]
    # The backtest-side comparison moved to test_na1_the_backtest_books_a_stop_at_its_level_less_the_slippage_floor
    # (strict xfail until D13 lands), not a loosened tolerance here.


_D13_STOP_XF = pytest.mark.xfail(strict=True, reason="NA-1 backtest side (Advisor 20:39): the backtest books a stop "
                                 "at its level (or the gap open) less max(half spread, 0.05 %), as a replayed outage "
                                 "stop; stop slippage lands with D13, which stacks on #146: not built yet")


@_D13_STOP_XF
@pytest.mark.parametrize("label, perp, profile, side", NA_SETUPS, ids=NA_IDS)
def test_na1_the_backtest_books_a_stop_at_its_level_less_the_slippage_floor(label, perp, profile, side):
    p = shape(flat_prices(30), 7.5, 7 + 50 / 60, adverse(side, 0.02))
    seq = backtest(p, side=side, perp=perp, profile=profile, tp=0.02, leave=25)
    level = seq[0][3] * (1 - side * 0.01)
    ex = first_exit(seq)
    assert ex[0] == "stop_loss" and abs(ex[3] / tp_model(level, side) - 1) < 1e-4, (ex, tp_model(level, side))


# PE2 (stop-safety, master 02845fb6): passes (mark removed)
@pytest.mark.parametrize("label, perp, profile, side", NA_SETUPS, ids=NA_IDS)
@pytest.mark.parametrize("path", ["reconnect", "restart"])
def test_l6_na1_a_gap_through_the_stop_fills_at_the_open_of_the_crossing_minute(path, label, perp, profile, side):
    p = _gap_prices(side)
    run = _outage(path, p, side=side, perp=perp, profile=profile)
    ex = first_exit(run.sequence())
    gap_open = tp_model(price_at(p, 7.0), side)  # Advisor 20:39: the open less TP_SLIP, the row marked modelled
    assert ex[0] == "stop_loss" and abs(ex[3] / gap_open - 1) < 1e-4, (ex, gap_open)
    (o,) = filled_orders(run, "stop_loss")
    assert is_modelled(o["signal"]), o["signal"]


@_D13_STOP_XF
@pytest.mark.parametrize("label, perp, profile, side", NA_SETUPS, ids=NA_IDS)
def test_na1_the_backtest_fills_a_gapped_stop_at_the_crossing_minutes_open(label, perp, profile, side):
    """Was true on ebe39c5 at the bare open; Advisor 20:39 moves it to the open less TP_SLIP (D13, strict xfail)."""
    p = _gap_prices(side)
    ex = _bt_exit(p, side=side, perp=perp, profile=profile, leave=25)
    gap = tp_model(price_at(p, 7.0), side)  # Advisor 20:39: the gap open less max(half spread, 0.05 %)
    assert ex[0] == "stop_loss" and abs(ex[3] / gap - 1) < 1e-4, (ex, gap)


_D13_ENTRY_XF = pytest.mark.xfail(strict=True, raises=AssertionError, reason="D13")  # Advisor 20:55: see below


# PE2 (stop-safety, master 02845fb6): passes (mark removed)
@pytest.mark.parametrize("label, perp, profile, side", NA_SETUPS, ids=NA_IDS)
@pytest.mark.parametrize("path", [pytest.param("reconnect", marks=_D13_ENTRY_XF), "restart"])
def test_l6_na1_an_outage_target_traded_through_fills_at_its_level_like_the_backtest(path, label, perp, profile,
                                                                                     side):
    p = shape(flat_prices(30), 7.5, 7 + 50 / 60, favourable(side, 0.03))
    run = _outage(path, p, side=side, perp=perp, profile=profile, tp=0.02)
    ex = first_exit(run.sequence())
    level = _entry(run) * (1 + side * 0.02)  # L12 FINAL 19:30: the level less TP_SLIP in the replay and the backtest
    assert ex[0] == "take_profit" and abs(ex[3] / tp_model(level, side) - 1) < 1e-4, (ex, tp_model(level, side))
    # [reconnect-*] xfail "D13" (Advisor 20:55): the backtest's entry fills at mid +- HALF spread with no floor and
    # the levels derive from the fill, so paper's target (entry at the ask) sits half a spread off the backtest's;
    # missed by ~6e-7 on 5f28360. D13 owns it; the 1e-4 tolerance stays.
    bt = _bt_exit(p, side=side, perp=perp, profile=profile, tp=0.02, leave=25)
    assert abs(ex[3] / bt[3] - 1) < 1e-4, (ex, bt)


# PE2 (stop-safety, master 02845fb6): passes (mark removed)
@pytest.mark.parametrize("path", ["reconnect", "restart"])
def test_l6_na1_the_market_on_return_price_is_recorded_beside_the_fill(path):
    p = shape(flat_prices(30), 7.5, 7 + 50 / 60, 0.98)
    run = _outage(path, p, side=1, perp=False, profile="aggressive", tp=0.02)
    (ex,) = filled_orders(run, "stop_loss")
    back = price_at(p, 12 + 5 / 60 if path == "reconnect" else 12)
    found = {k: v for k, v in (ex["signal"] or {}).items() if "return" in k.lower()}
    assert found and all(abs(float(v) / back - 1) < 2e-3 for v in found.values()), ex["signal"]


# PE2 (stop-safety, master 02845fb6): passes (mark removed)
@pytest.mark.parametrize("label, perp, profile, side", [NA_SETUPS[0], NA_SETUPS[4]], ids=[NA_IDS[0], NA_IDS[4]])
def test_l6_na1_no_backfill_falls_back_to_market_on_return_and_flags_the_trade(label, perp, profile, side):
    def unavailable(prices, upto):
        def make(inst):
            def load(instrument, bar_type, limit):
                raise LookupError("no stored history for BTC/USD")

            return load

        return make

    p = shape(flat_prices(40), 7.5, 40, adverse(side, 0.02))  # still past the stop on return
    run = restart(p, 6, 12, side=side, perp=perp, profile=profile, leave=30, history=unavailable(p, 0))
    ex = first_exit(run.sequence())
    assert ex[0] == "stop_loss" and abs(ex[3] / price_at(p, 12) - 1) < 2e-3  # market on return
    (o,) = filled_orders(run, "stop_loss")
    assert any("fallback" in k.lower() or "fallback" in str(v).lower() for k, v in (o["signal"] or {}).items()), \
        o["signal"]


# --- NA-2: both crossed in one missed minute: the stop, in paper AND the backtest


def _both_open_near_target(side):
    """The minute to 00:08: opens 1.8 % in the position's favour (nearer the 2 % target), trades 3 % in favour
    (through the target) and then 2 % against (through the 1 % stop), and closes back at the open's level; then
    flat. The minute the backtest's "nearer the open first" ordering takes the target in."""
    p = flat_prices(30).copy()
    p[420:480] = BASE * favourable(side, 0.018)
    p[425:435] = BASE * favourable(side, 0.03)
    p[445:455] = BASE * adverse(side, 0.02)
    return p


@pytest.mark.parametrize("label, perp, profile, side", NA_SETUPS, ids=NA_IDS)
@pytest.mark.parametrize("path", ["reconnect", "restart"])
def test_na2_paper_takes_the_stop_when_one_missed_minute_crossed_both_open_nearer_the_target(path, label, perp,
                                                                                           profile, side):
    """Already true on ebe39c5 (adverse first in _exit_on_breach): pinned so it stays so."""
    run = _outage(path, _both_open_near_target(side), side=side, perp=perp, profile=profile, tp=0.02)
    assert first_exit(run.sequence())[0] == "stop_loss", run.sequence()


# PE2 (stop-safety, master 02845fb6): passes (mark removed)
@pytest.mark.parametrize("label, perp, profile, side", NA_SETUPS, ids=NA_IDS)
def test_l7_na2_the_backtest_takes_the_stop_when_one_minute_crossed_both_open_nearer_the_target(label, perp, profile,
                                                                                              side):
    ex = _bt_exit(_both_open_near_target(side), side=side, perp=perp, profile=profile, tp=0.02, leave=25)
    assert ex[0] == "stop_loss", ex


# --- NA-3: liquidation, the drawdown halt and the daily pause in the outage replay

LIQ_SETUPS = [("perp-2x-long", True, "balanced", 1), ("perp-3x-long", True, "aggressive", 1),
              ("perp-3x-short", True, "aggressive", -1)]
LIQ_IDS = [s[0] for s in LIQ_SETUPS]
LIQ_DEPTH = {"balanced": 0.60, "aggressive": 0.40}  # beyond the isolated liquidation price at 2x (~-50 %) / 3x
# Liquidation mechanics: on reconnect the position is opened in the run, past what stop safety allows (the open-risk
# limit refuses it), so those cells lift the guards; the restart cells carry it from the journal and run guarded.
LIQ_PATHS = [pytest.param("reconnect", marks=LIQUIDATION_MECHANICS, id="reconnect"), "restart"]


# PE2 (stop-safety, master 02845fb6): passes (mark removed)
@pytest.mark.parametrize("label, perp, profile, side", LIQ_SETUPS, ids=LIQ_IDS)
@pytest.mark.parametrize("path", LIQ_PATHS)
def test_l8_na3_a_missed_minute_opening_beyond_liquidation_liquidates(path, label, perp, profile, side):
    p = shape(flat_prices(30), 7.0, 8.0, adverse(side, LIQ_DEPTH[profile]))  # the minute to 00:08, then back
    run = _outage(path, p, side=side, perp=perp, profile=profile, stop=0.10, qty=0.25)
    assert first_exit(run.sequence())[0] == "liquidation", run.sequence()


@pytest.mark.parametrize("label, perp, profile, side", LIQ_SETUPS, ids=LIQ_IDS)
@pytest.mark.parametrize("path", LIQ_PATHS)
def test_na3_a_minute_opening_short_of_liquidation_takes_the_reachable_stop_first(path, label, perp, profile, side):
    """Already true on ebe39c5 (the stop is checked, liquidation isn't): pinned. The minute opens flat and wicks
    beyond liquidation; the 10 % stop is reached first."""
    p = shape(flat_prices(30), 7.25, 7.5, adverse(side, LIQ_DEPTH[profile]))
    run = _outage(path, p, side=side, perp=perp, profile=profile, stop=0.10, qty=0.25)
    assert first_exit(run.sequence())[0] == "stop_loss", run.sequence()


# PE2 (stop-safety, master 02845fb6): passes (mark removed)
@pytest.mark.parametrize("label, perp, profile, side", LIQ_SETUPS, ids=LIQ_IDS)
@pytest.mark.parametrize("path", LIQ_PATHS)
def test_l8_na3_a_stopless_perp_wicked_through_liquidation_in_the_outage_is_liquidated(path, label, perp, profile,
                                                                                       side):
    p = shape(flat_prices(30), 7.25, 7.5, adverse(side, LIQ_DEPTH[profile]))
    run = _outage(path, p, side=side, perp=perp, profile=profile, stop=None, qty=0.25)
    assert first_exit(run.sequence())[0] == "liquidation", run.sequence()


# PE2 (stop-safety, master 02845fb6): passes (mark removed)
@pytest.mark.parametrize("path", LIQ_PATHS)
@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
@pytest.mark.parametrize("depth, kind", [(0.07, "risk_pause"), (0.235, "risk_halt")], ids=["daily-pause", "dd-halt"])
def test_l8_na3_the_replay_applies_the_daily_pause_and_the_drawdown_halt(path, side, depth, kind):
    """Stopless perp at 3x (aggressive: 8 % daily loss, 35 % drawdown) holding about 1.5x equity: a missed minute
    7 % against costs ~10.5 % of equity (pause); 23.5 % against costs ~35.3 % (halt), still outside the 7 % cut
    distance from liquidation. Both recover before the return."""
    p = shape(flat_prices(30), 7.25, 7.5, adverse(side, depth))
    run = _outage(path, p, side=side, perp=True, profile="aggressive", stop=None, qty=0.25)
    assert run.kinds(kind), [e["kind"] for e in run.events][-20:]
    assert first_exit(run.sequence())[0] == kind, run.sequence()


# ---- P1-L2 fix (PE1 17:48): the client holds a gap's live minute until the hub says the refill is over. The harness
# now sends the hub's {"t": "gap"} and {"t": "filled"}; these pin that the REAL relay sends them, on every path.
# A refill that fails or finds nothing must still say "filled", or the client waits on its 20 s safety net.


class _Now:
    def submit(self, fn, *a):
        fn(*a)


def _relay(monkeypatch, refill):
    import threading

    from sleeve_fund.hub import relay as relay_mod

    r = relay_mod.HubRelay.__new__(relay_mod.HubRelay)
    sent = []
    r.fanout = type("F", (), {"publish": staticmethod(sent.append)})()
    r.pairs, r.recent, r.definitions, r._lock = {"BTCUSDT-PERP.BINANCE": "BTCUSDT"}, object(), {}, threading.Lock()
    r._store, r._refill, r.sink = _Now(), _Now(), lambda msgs: None
    monkeypatch.setattr(relay_mod, "refill_bars", refill)
    return r, sent


def _raise(*a, **k):
    raise ConnectionError("qa: the venue's REST is down too")


_L2_FIX = "P1-L2 fix not pushed yet (PE1 17:48): the relay sends no {'t': 'filled'}"


# PE2 (stop-safety, master 02845fb6): passes (mark removed)
@pytest.mark.parametrize("case", ["found", "nothing_found", "refill_raises", "instrument_not_relayed"])
def test_l2_relay_says_filled_after_every_refill_found_or_not(monkeypatch, case):
    since, until = START + 7 * M, START + 10 * M
    bar = {"t": "bar", "id": "BTCUSDT-PERP.BINANCE", "o": "1", "h": "1", "l": "1", "c": "1", "v": "1",
           "ts": since, "recv": 0, "refilled": True}
    refill = {"found": lambda *a, **k: [bar], "nothing_found": lambda *a, **k: [], "refill_raises": _raise,
              "instrument_not_relayed": lambda *a, **k: [bar]}[case]
    r, sent = _relay(monkeypatch, refill)
    iid = "ETHUSDT-PERP.BINANCE" if case == "instrument_not_relayed" else "BTCUSDT-PERP.BINANCE"
    r._fill(iid, since, until)
    assert sent and sent[-1] == {"t": "filled", "id": iid, "since": since, "until": until}, sent
    assert [m for m in sent if m.get("t") == "bar"] == ([bar] if case == "found" else [])


def test_l2_relay_announces_the_gap_before_its_live_minute(monkeypatch):
    """The hold is keyed on this message: it must reach the client before the live minute that shows the gap."""
    from sleeve_fund.hub import relay as relay_mod

    r, sent = _relay(monkeypatch, lambda *a, **k: [])
    r.gaps, r._since, r._fill = relay_mod.Gaps(), {}, lambda *a: None
    monkeypatch.setattr(relay_mod.protocol, "bar_from_nautilus", lambda b, now: dict(b))
    live = {"t": "bar", "id": "BTCUSDT-PERP.BINANCE", "o": "1", "h": "1", "l": "1", "c": "1", "v": "5",
            "recv": 0, "refilled": False}
    r.on_bar({**live, "ts": START + 6 * M})
    sent.clear()
    r.on_bar({**live, "ts": START + 11 * M})
    assert [m["t"] for m in sent] == ["gap", "bar"], sent
    assert (sent[0]["since"], sent[0]["until"]) == (START + 7 * M, START + 10 * M)


# ==================== P1-U35: paper's watched stop is ONE journal row per stop, and it ends as its position does
#
# Stop safety (PE2) journals the stop paper watches in the process as an order row (intent stop_loss, order_type
# "STOP (watched)"), and a stop that fires sends its own market stop-loss, journaled as the row that fills. Pinned per
# stop (QA brief 7 Oct; ruling of the Head of QA with the HoE, 7 Oct): exactly one watched row and exactly one filled
# market stop_loss row, priced from the watched level (live: the level the market row records; outage replay: the
# modelled price, the level less TP_SLIP). When the stop FIRED the watched row ends "triggered" and links to that
# market order (a parent order or a reference field: whatever PE2 builds); when the position closed by another exit
# (target, signal, PM flatten) it ends cancelled, with its reason. On a head without P1-U35 (main 0a5cd8f) there is no
# watched row at all: the cells fail on their first assertion, "not built: watched stop row", a strict xfail there.

WATCHED = "STOP (watched)"
U35_SETUPS = [SETUPS[0], SETUPS[3], SETUPS[5]]  # spot, perp 3x long and short, each with the 1 % stop (guarded)
U35_IDS = [s[0] for s in U35_SETUPS]


def _u35_built() -> bool:
    """Whether the code under test journals paper's watched stop (the label is in the strategy's source)."""
    try:
        import inspect

        from sleeve_fund.strategies import base

        return WATCHED in inspect.getsource(base)
    except Exception:  # noqa: BLE001 - a head that can't say has not built it
        return False


_U35_XF = pytest.mark.xfail(condition=not _u35_built(), strict=True, raises=AssertionError,
                            reason="not built: watched stop row (P1-U35, the stop-safety PR): this head journals no "
                                   "'STOP (watched)' row")
_U35_TRIGGERED_XF = pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="not built: the watched stop row ends 'triggered', linked to the market "
    "stop-loss it fired (P1-U35 ruling, HoQA + HoE 7 Oct; PE2 carries the label change in the gate). ee2b414 ends it "
    "'canceled'; main has no watched row")


def _watched_rows(run: Run) -> list[dict]:
    rows = [o for o in run.orders if o.get("order_type") == WATCHED]
    assert rows, f"not built: watched stop row (P1-U35): no {WATCHED!r} row in {[o['intent'] for o in run.orders]}"
    return rows


def _u35_fired(path: str, perp: bool, profile: str, side: int) -> Run:
    """live: 2 % against from 00:10:30 on, through the 1 % stop on the hub-fed trades. reconnect: the same dip inside
    the hub outage (00:07:30-00:07:50, recovered), found by the outage replay on return (NA-1)."""
    if path == "live":
        return paper(shape(flat_prices(30), 10.5, 30, adverse(side, 0.02)), side=side, perp=perp, profile=profile,
                     leave=25, stop=0.01)
    return _outage("reconnect", shape(flat_prices(30), 7.5, 7 + 50 / 60, adverse(side, 0.02)), side=side, perp=perp,
                   profile=profile, tp=0.02)


def _the_stop_rows(run: Run) -> tuple[dict, dict]:
    (watched,) = _watched_rows(run)
    markets = [o for o in filled_orders(run, "stop_loss") if o.get("order_type") != WATCHED]
    (market,) = markets
    assert market["status"] == "filled", market
    return watched, market


def _links(a: dict, b: dict) -> bool:
    """Row a names row b's order id in a field of its own or of its signal (a parent order, a reference)."""
    fields = {k: v for k, v in a.items() if k not in ("order_id", "message", "reason", "signal")}
    fields.update({f"signal.{k}": v for k, v in (a.get("signal") or {}).items()})
    return any(v == b["order_id"] for v in fields.values())


@_U35_XF
@pytest.mark.parametrize("label, perp, profile, side", U35_SETUPS, ids=U35_IDS)
@pytest.mark.parametrize("path", ["live", "reconnect"])
def test_u35_a_fired_stop_has_one_watched_row_and_one_filled_market_stop_loss_priced_from_its_level(path, label, perp,
                                                                                                  profile, side):
    run = _u35_fired(path, perp, profile, side)
    assert first_exit(run.sequence())[0] == "stop_loss", run.sequence()
    watched, market = _the_stop_rows(run)
    level = float(watched["signal"]["stop_px"])
    entry = fill_px(run, "entry")
    assert level == pytest.approx(entry * (1 - side * 0.01), rel=1e-9), (level, entry)
    assert watched["side"] == market["side"] and watched["qty"] == pytest.approx(market["qty"], rel=1e-9)
    if path == "live":  # the market stop-loss records the level it fired at: the watched one
        sig = market["signal"] or {}
        assert sig["entry_px"] * (1 - side * sig["stop_loss"]) == pytest.approx(level, rel=1e-9), (sig, level)
    else:  # the replay's modelled price: the watched level less TP_SLIP (NA-1, Advisor 20:39)
        assert is_modelled(market["signal"]), market["signal"]
        assert abs(market["avg_px"] / tp_model(level, side) - 1) < 1e-4, (market["avg_px"], tp_model(level, side))


# PE2 (stop-safety, master 02845fb6): passes (mark removed)
@pytest.mark.parametrize("label, perp, profile, side", U35_SETUPS, ids=U35_IDS)
@pytest.mark.parametrize("path", ["live", "reconnect"])
def test_u35_a_fired_stops_watched_row_ends_triggered_and_links_to_its_market_stop_loss(path, label, perp, profile,
                                                                                       side):
    run = _u35_fired(path, perp, profile, side)
    watched, market = _the_stop_rows(run)
    assert watched["status"] == "triggered", watched
    assert _links(watched, market) or _links(market, watched), (watched, market)


def _flatten_once_in_position(monkeypatch):
    """The PM flattens once the position is open: the command reaches the runtime's next look at its commands."""
    from sleeve_fund.store import Store

    pending, sent = Store.pending_commands, []

    def with_flatten(self, sleeve):
        if not sent and self.fills(sleeve, limit=1):
            sent.append(sleeve)
            self.command(sleeve, "flatten", "QA: flatten while the stop is watched")
        return pending(self, sleeve)

    monkeypatch.setattr(Store, "pending_commands", with_flatten)


@_U35_XF
@pytest.mark.parametrize("label, perp, profile, side", [U35_SETUPS[0], U35_SETUPS[2]], ids=[U35_IDS[0], U35_IDS[2]])
@pytest.mark.parametrize("exit_by, intent", [("target", "take_profit"), ("signal", "exit"), ("flatten", "pm_flatten")])
def test_u35_a_position_closed_by_another_exit_cancels_its_watched_row_with_its_reason(exit_by, intent, label, perp,
                                                                                      profile, side, monkeypatch):
    """target: 3 % in favour from 00:10:30 through the 2 % target; signal: the probe's exit at 00:25 on flat prices;
    flatten: the PM's flatten once the position is open. The stop never fires."""
    p = shape(flat_prices(30), 10.5, 30, favourable(side, 0.03)) if exit_by == "target" else flat_prices(30)
    if exit_by == "flatten":
        _flatten_once_in_position(monkeypatch)
    run = paper(p, side=side, perp=perp, profile=profile, leave=25, stop=0.01,
                tp=0.02 if exit_by == "target" else None)
    (watched,) = _watched_rows(run)
    assert first_exit(run.sequence())[0] == intent, run.sequence()
    assert not filled_orders(run, "stop_loss"), run.sequence()
    assert watched["status"] in ("canceled", "cancelled"), watched
    assert (watched["message"] or "").strip(), watched  # with its reason
