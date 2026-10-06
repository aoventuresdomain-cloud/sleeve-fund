"""QA's hub-path parity checks from its round on #140 (P1-C2, P1-C5 and the P1-C1 repro), carried by #146: a
paper strategy fed through the hub client decides and fills as the backtest of the same minutes does; and what
an open position sees while the hub is away.

The hub path is replayed in Nautilus's BacktestEngine set up as the hub-fed paper node is (build_node, hub set):
the simulated venue fills from trades and quotes only (bar_execution=False), the strategy runs with the paper
runtime (not in backtest mode), decides on `<id>-<n>-MINUTE-LAST-EXTERNAL` bars that come out of the hub
client's Decoder, fed the hub's 1-minute messages built from the same trades, each received `lag` after its
close. The reference paths are the repository's own: research.replay (paper on its own feed, INTERNAL bars) and
research.runner.run_backtest (the backtest on the store's minutes).
"""

from __future__ import annotations

from datetime import timezone

import numpy as np
import pandas as pd
import pytest
from nautilus_trader.model import Bar

from sleeve_fund import markets
from sleeve_fund.instruments import BOOK_SHARE
from sleeve_fund.paper.hub_client import Decoder
from sleeve_fund.paper.recorder import Recorder
from sleeve_fund.research.replay import replay
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.strategies import REGISTRY
from sleeve_fund.strategies.base import LongFlatConfig, LongFlatStrategy
from sleeve_fund.venues import venue

K = venue("KRAKEN")
INST = K.instrument("BTC", "USD", price_precision=1)
START = 1_759_449_600_000_000_000  # 2025-10-03 00:00 UTC
S = 1_000_000_000
M = 60 * S
SPREAD = 12.0


class ProbeConfig(LongFlatConfig):
    def __init__(self, *, period: int = 5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.period = period


class Probe(LongFlatStrategy):
    """Long for `period` bars, flat for `period`, by the bar's close time (as tests/test_sanity.py's probe)."""

    def __init__(self, config: ProbeConfig) -> None:
        super().__init__(config)
        self.step = config.bar_type.spec.timedelta.total_seconds() * 1e9
        self.period = config.period

    def want_long(self, bar: Bar) -> bool | None:
        return (int(bar.ts_event // self.step) // self.period) % 2 == 1


@pytest.fixture(autouse=True)
def _probe(monkeypatch):
    monkeypatch.setitem(REGISTRY, "probe", (Probe, ProbeConfig))


def _fees(params):
    return markets.fees_for(params, K.fees)


def _sleeve(params, strategy, profile, spec):
    f = _fees(params)
    return {"name": "hubqa", "strategy": strategy, "instrument": "BTC/USD", "bar_spec": spec,
            "starting_balance": 10_000, "risk_profile": profile, "params": params, "max_notional": None,
            "maker_fee": str(f.maker), "taker_fee": str(f.taker), "tick_seconds": 30}


def _ticks(prices, gone=()):
    """One trade a second and the quote after it; seconds in `gone` (the hub away) carry nothing."""
    from nautilus_trader.model import AggressorSide, Price, Quantity, QuoteTick, TradeId, TradeTick

    out = []
    for s, px in enumerate(np.round(prices, 1)):
        if s in gone:
            continue
        t = START + s * S
        out.append(TradeTick(INST.id, Price(px, 1), Quantity(1.0, 8), AggressorSide.BUY if s % 2 else
                             AggressorSide.SELL, TradeId(str(s)), t, t + 1000))
        out.append(QuoteTick(INST.id, Price(px - SPREAD / 2, 1), Price(px + SPREAD / 2, 1), Quantity(1, 8),
                             Quantity(1, 8), t + 2000, t + 3000))
    return out


def _minutes(prices) -> pd.DataFrame:
    """The hub's 1-minute bars from the same trades, by open time (the store's convention)."""
    s = pd.Series(np.round(prices, 1), index=pd.to_datetime(START + np.arange(len(prices)) * S, unit="ns", utc=True))
    m = s.resample("1min", closed="left", label="left").ohlc()
    m["volume"] = 60.0
    return m


def _hub_bars(minutes: pd.DataFrame, spec: str, lag: int, back_at: int | None = None, away=(None, None)):
    """The Decoder's output for the hub's minute messages, fed in the order they reach the node: live ones `lag`
    after their close; minutes closing while the hub was away (away[0] < close <= away[1]) are refilled, in
    order, at back_at."""
    d, out, arrivals = Decoder(spec), [], []
    for ts, r in minutes.iterrows():
        close = int(ts.value) + M
        m = {"t": "bar", "id": str(INST.id), "o": f"{r.open:.1f}", "h": f"{r.high:.1f}", "l": f"{r.low:.1f}",
             "c": f"{r.close:.1f}", "v": "60.00000000", "ts": close, "recv": close, "refilled": False}
        now = close + lag
        if away[0] is not None and away[0] < close <= away[1]:
            m["refilled"], now = True, back_at
        arrivals.append((now, close, m))
    for now, _, m in sorted(arrivals, key=lambda a: (a[0], a[1])):
        b = d(m, now)
        out += b if isinstance(b, list) else [b] if b is not None else []
    return out, d


def hub_paper(prices, params, strategy="probe", profile="aggressive", minutes=1, lag=500, gone=(), away=(None, None),
              back_at=None):
    """The hub-fed paper node, replayed. Returns (orders, fills, decoder)."""
    from nautilus_trader.backtest import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.common import LoggerConfig, LogLevel
    from nautilus_trader.model import AccountType, BarType, Currency, Money, OmsType, TraderId

    from sleeve_fund.instruments import ScheduleFeeModel, fill_model
    from sleeve_fund.paper.runtime import SleeveRuntime
    from sleeve_fund.store import Store, utcnow

    spec = f"{minutes}-MINUTE-LAST-EXTERNAL"
    s = _sleeve(params, strategy, profile, f"{minutes}-MINUTE-LAST-INTERNAL")
    fees = _fees(params)
    store = Store.in_memory()
    store.create_sleeve(name=s["name"], strategy=strategy, instrument="BTC/USD", bar_spec=s["bar_spec"],
                        starting_balance=10_000, risk_profile=profile, params=params)
    runtime = SleeveRuntime(store, s["name"], tick_seconds=30)
    engine = BacktestEngine(BacktestEngineConfig(trader_id=TraderId.from_str("HUBQA-001"),
                                                 logging=LoggerConfig(stdout_level=LogLevel.ERROR)))
    fee_model = ScheduleFeeModel(fees)
    try:
        engine.add_venue(venue=INST.id.venue, oms_type=OmsType.NETTING, account_type=AccountType.CASH,
                         base_currency=None, starting_balances=[Money(10_000, Currency.from_str("USD"))],
                         fee_model=fee_model, fill_model=fill_model(), bar_execution=False)  # as build_node, hub set
        engine.add_instrument(INST)
        bars, dec = _hub_bars(_minutes(prices), spec, lag, back_at, away)
        engine.add_data(_ticks(prices, gone) + bars)
        cls, cfg_cls = REGISTRY[strategy]
        cfg = cfg_cls(instrument_id=INST.id, bar_type=BarType.from_str(f"{INST.id}-{spec}"), max_notional=None,
                      assumed_taker_fee=float(fees.taker), warmup_bars=0, **params)
        st = cls(cfg).attach_runtime(runtime)
        st.simulated_venue, st.fee_model, st.hub_fed = True, fee_model, True
        engine.add_strategy(st)
        engine.run()
        orders = list(reversed(store.orders(s["name"], limit=100_000)))
        fills = list(reversed(store.fills(s["name"], limit=100_000)))
        return orders, fills, dec, store
    finally:
        runtime.now = utcnow
        engine.dispose()


def own_feed_paper(tmp_path, prices, params, strategy="probe", profile="aggressive", minutes=1):
    from nautilus_trader.model import AggressorSide, Price, Quantity, QuoteTick, TradeId, TradeTick  # noqa: F401

    rec = Recorder(tmp_path / "p.jsonl.gz")
    rec.meta = {"balances": ["10000.00 USD"], "sleeve": _sleeve(params, strategy, profile,
                                                                f"{minutes}-MINUTE-LAST-INTERNAL")}
    rec.start(INST)
    for t in _ticks(prices):
        (rec.trade if isinstance(t, TradeTick) else rec.quote)(t)
    rec.close()
    return replay(tmp_path / "p.jsonl.gz", with_fills=True)


def backtest(prices, params, strategy="probe", profile="aggressive", minutes=1):
    m = _minutes(prices)
    bars = m.set_axis(m.index + pd.Timedelta(minutes=1))  # stamped at the close, as HistoryStore.read gives
    bars["volume"] = 60 / BOOK_SHARE
    kw = {}
    if minutes > 1:
        dec = bars.resample(f"{minutes}min", label="right", closed="right").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
        kw = {"exec_prices": bars, "exec_minutes": 1}
        bars = dec
    res = run_backtest(strategy, bars, INST, params=params, starting_capital=10_000, risk_profile=profile,
                       bar_minutes=minutes, half_spread=SPREAD / 2 / float(prices[0]), **kw)
    j = res.journal
    return list(j.orders_.values()), j.fills_


def per_order(orders, fills):
    """(side, intent, minute, qty, all-in price, fee) per order, in fill order (as tests/test_sanity.py)."""
    intents = {o["order_id"]: o["intent"] for o in orders}
    out = {}
    for f in fills:
        side, qty, notional, fee, _ = out.get(f["order_id"], (f["side"], 0.0, 0.0, 0.0, None))
        out[f["order_id"]] = (side, qty + f["qty"], notional + f["qty"] * f["price"], fee + f["fee"], f["ts"])
    rows = []
    for k, (side, qty, notional, fee, ts) in out.items():
        t = pd.Timestamp(ts).tz_convert(timezone.utc)
        minute = t if t == t.floor("1min") else t.ceil("1min")
        rows.append((side, intents[k], minute, qty, (notional + fee if side == "BUY" else notional - fee) / qty, fee))
    return rows


def _final_qty(fills):
    return sum(f["qty"] * (1 if f["side"] == "BUY" else -1) for f in fills)


TOL = {"entry": (-0.3, 0.3), "exit": (-0.3, 0.3), "stop_loss": (-2.5, 7.0), "take_profit": (-6.0, 1.0)}


def _assert_same(hub, ref, minute_slack=0):
    assert [r[:2] for r in hub] == [r[:2] for r in ref]
    for h, b in zip(hub, ref):
        assert b[3] == pytest.approx(h[3], rel=2e-3), (h, b)
        lo, hi = TOL[h[1]]
        assert lo <= (b[4] / h[4] - 1) * 1e4 <= hi, (h, b)
        late = abs((b[2] - h[2]).total_seconds())
        assert late == 0 if h[1] in ("entry", "exit") else late <= 60 * minute_slack, (h, b)


WAVE = lambda s: 60_000 * (1 + 0.01 * np.sin(s / 700) + 0.001 * np.sin(s / 11))  # noqa: E731


# --------------------------------------------------------------- check 2 + 5: same decisions and fills


@pytest.mark.parametrize("minutes", [1, 15])
def test_c2_c5_hub_paper_sends_the_same_orders_and_fills_as_own_feed_paper_and_the_backtest(tmp_path, minutes):
    prices = WAVE(np.arange(240 * 60))
    params = {"period": 3 if minutes == 15 else 7}
    o, f, dec, _ = hub_paper(prices, params, minutes=minutes)
    hub = per_order(o, f)
    po, pf = own_feed_paper(tmp_path, prices, params, minutes=minutes)
    own = per_order(po, pf)
    bo, bf = backtest(prices, params, minutes=minutes)
    bt = per_order(bo, bf)
    assert len(hub) >= 4 and dec.late == 0
    # the hub path and paper on its own feed: identical orders, minutes, sizes, prices and fees
    assert [r[:3] for r in hub] == [r[:3] for r in own]
    for h, p in zip(hub, own):
        assert h[3] == pytest.approx(p[3], rel=1e-9) and h[4] == pytest.approx(p[4], rel=1e-9)
        assert h[5] == pytest.approx(p[5], abs=0.0100001)
    # and against the backtest: same minute, size and all-in price to 0.3 bp (test_sanity.py's bar)
    _assert_same(hub, bt)
    spread_b = sum(r[3] * SPREAD / 2 for r in bt)
    assert sum(r[5] for r in bt) - spread_b == pytest.approx(sum(r[5] for r in hub), rel=0.005)
    assert _final_qty(f) == pytest.approx(_final_qty(bf), abs=1e-8)


def test_c5_stops_and_targets_through_the_hub_match_the_backtest(tmp_path):
    s = np.arange(240 * 60)
    prices = 60_000 * (1 + 0.012 * np.sin(s / 500) + 0.0015 * np.sin(s / 13))
    params = {"period": 9, "stop_loss": 0.004, "take_profit": 0.025}
    o, f, _, _ = hub_paper(prices, params)
    hub = per_order(o, f)
    bo, bf = backtest(prices, params)
    bt = per_order(bo, bf)
    assert {"entry", "stop_loss"} <= {r[1] for r in hub}
    _assert_same(hub, bt, minute_slack=1)


def test_c5_a_realistic_half_second_bar_lag_still_fills_in_the_same_minute(tmp_path):
    """Hub bars reach the node ~0.5 s after the close: the market order fills on the quote then (one second's
    move later here), in the same minute as the backtest's, within a few bp."""
    prices = WAVE(np.arange(240 * 60))
    o, f, _, _ = hub_paper(prices, {"period": 7}, lag=S // 2)
    hub = per_order(o, f)
    bt = per_order(*backtest(prices, {"period": 7}))
    assert [r[:3] for r in hub] == [r[:3] for r in bt]
    gaps = [abs(b[4] / h[4] - 1) * 1e4 for h, b in zip(hub, bt)]
    print(f"max all-in price gap at 0.5 s lag: {max(gaps):.2f} bp")
    assert max(gaps) < 5


# ------------------------------------------------------------- check 5: an open position, hub away


def _dip_prices(recover: bool):
    """20 minutes of one trade a second. The probe (period 5 on 1m) goes long on the bar closing 00:05 and flat
    on the one closing 00:10. While long, from 00:06:30 to 00:08:30 the price is 2 % lower (stop at 1 %), then
    back (recover) or still down."""
    s = np.arange(20 * 60)
    p = 60_000 * (1 + 1e-6 * s)
    dip = (s >= 6 * 60 + 30) & (s < 8 * 60 + 30)
    p[dip] *= 0.98
    if not recover:
        p[s >= 8 * 60 + 30] *= 0.98
    return p


GONE = set(range(6 * 60, 12 * 60))  # hub away 00:06:00-00:12:00: no trades, quotes or bars reach the node
AWAY = (START + 6 * M, START + 12 * M)  # minutes closing 00:07-00:12 come back as refills ...
BACK = START + 12 * M + 5 * S  # ... at 00:12:05: those closing 00:07-00:10 over 90 s late (sent, exits only)


@pytest.mark.parametrize("recover", [False, True])
def test_c5_backtest_on_the_same_minutes_stops_out_inside_the_outage(recover):
    bt = per_order(*backtest(_dip_prices(recover), {"period": 5, "stop_loss": 0.01}))
    assert [r[1] for r in bt][:2] == ["entry", "stop_loss"]
    assert bt[1][2] == pd.Timestamp(START + 7 * M, unit="ns", tz="UTC")  # the minute closing 00:07


def test_c5_hub_away_while_long_and_the_price_still_past_the_stop_the_stop_runs_on_reconnect():
    """The stop is not lost when the price is still past it. The live check waits for the missed minutes, which
    come back at 00:12:05, and the replay books the stop as the venue's resting stop would have filled it, at the
    backtest's price in the minute to 00:07 (Advisor NA-1, 6 Oct; it was sold at market on the first trade back)."""
    o, f, dec, store = hub_paper(_dip_prices(False), {"period": 5, "stop_loss": 0.01}, gone=GONE, away=AWAY,
                                 back_at=BACK)
    hub = per_order(o, f)
    bt = per_order(*backtest(_dip_prices(False), {"period": 5, "stop_loss": 0.01}))
    assert [r[1] for r in hub][:2] == ["entry", "stop_loss"]
    assert hub[1][2] == pd.Timestamp(START + 13 * M, unit="ns", tz="UTC")  # sent at 00:12:05
    assert abs(hub[1][4] / bt[1][4] - 1) < 5e-4, (hub[1], bt[1])
    assert dec.late == 4  # the bars closing 00:07-00:10 came late: exits only


def test_c5_hub_away_while_long_and_the_price_recovers_the_stop_still_runs():
    o, f, dec, store = hub_paper(_dip_prices(True), {"period": 5, "stop_loss": 0.01}, gone=GONE, away=AWAY,
                                 back_at=BACK)
    hub = per_order(o, f)
    print("hub path orders:", [(r[1], str(r[2])) for r in hub])
    assert [r[1] for r in hub][:2] == ["entry", "stop_loss"], hub
