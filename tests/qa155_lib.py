"""PR 155 QA helpers (from round 13 lib13). No venue network here (the proxy refuses Binance and Kraken), so every run is on
synthetic 1-minute OHLCV built here (a seeded GBM with volatility regimes), and Binance's contract lookup is
stubbed to its published BTCUSDT limits (0.1 tick, 0.001 lot, 5 USDT min notional).
Kept in the suite with tests/test_degraded_155_qa.py; stub_contract() is applied per test there, not at import."""
import os, sys, tempfile, json
from pathlib import Path
import numpy as np, pandas as pd


from sleeve_fund import venues  # noqa: E402

def stub_contract(monkeypatch):
    monkeypatch.setattr(venues.VENUES["BINANCE"], "contract", lambda pair: {
        "price_precision": 1, "size_precision": 3, "min_quantity": 0.001, "min_notional": 5.0})


def quiet(fn):
    fd, p = tempfile.mkstemp(); s1, s2 = os.dup(1), os.dup(2); os.dup2(fd, 1); os.dup2(fd, 2)
    try:
        return fn()
    finally:
        os.dup2(s1, 1); os.dup2(s2, 2); os.close(fd)


def synth_1m(days=30, seed=1, start="2025-01-01", p0=60_000.0, vol_day=0.03, drift_day=0.0):
    """1-minute OHLCV bars indexed by close time. Regime-switching vol, intrabar high/low from 4 sub-steps."""
    rng = np.random.default_rng(seed)
    n = days * 1440
    regime = np.repeat(rng.choice([0.6, 1.0, 1.8], size=n // 240 + 1, p=[0.4, 0.4, 0.2]), 240)[:n]
    sig = vol_day / np.sqrt(1440) * regime
    sub = rng.standard_normal((n, 4)) * (sig[:, None] / 2) + drift_day / 1440 / 4
    path = np.cumsum(sub.ravel())
    lp = np.log(p0) + path
    px = np.exp(lp).reshape(n, 4)
    opens = np.concatenate([[p0], px[:-1, -1]])
    close = px[:, -1]
    high = np.maximum(np.maximum(opens, px.max(1)), close)
    low = np.minimum(np.minimum(opens, px.min(1)), close)
    idx = pd.date_range(pd.Timestamp(start, tz="UTC") + pd.Timedelta(minutes=1), periods=n, freq="1min")
    df = pd.DataFrame({"open": opens, "high": high, "low": low, "close": close,
                       "volume": rng.uniform(5, 50, n)}, index=idx)
    return df.round({"open": 1, "high": 1, "low": 1, "close": 1})


def resample(m1, minutes):
    if minutes == 1:
        return m1
    r = m1.resample(f"{minutes}min", closed="right", label="right")
    return pd.DataFrame({"open": r["open"].first(), "high": r["high"].max(), "low": r["low"].min(),
                         "close": r["close"].last(), "volume": r["volume"].sum()}).dropna()


def binance_inst():
    return venues.VENUES["BINANCE"].instrument("BTC", "USDT", price_precision=1)


def kraken_inst():
    return venues.venue("KRAKEN").instrument("BTC", "USD", price_precision=1)


def write_funding(idx_start, idx_end, rate_fn=None, root=None):
    """Settled funding at 00/08/16 UTC for BINANCE BTC/USDT into the history dir."""
    root = Path(root or os.environ["HISTORY_DIR"])
    times = pd.date_range(idx_start.floor("8h"), idx_end.ceil("8h"), freq="8h")
    rate_fn = rate_fn or (lambda i, t: 0.0001 * (1 + (i % 5)) * (-1 if i % 7 == 3 else 1))
    rows = [[int(t.timestamp() * 1000), rate_fn(i, t)] for i, t in enumerate(times)]
    p = root / "BINANCE" / "BTC-USDT" / "funding.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"rates": rows}))
    return pd.Series([r for _, r in rows], index=times)


def ticks_from_bars(inst, m1, spread=2.0, per_bar=4):
    """Trades and quotes inside each 1-minute bar (open, then the extreme nearer the open, the other, the close)
    at seconds 1, 15, 30, 59 of the minute, so the INTERNAL 1-minute bars paper builds equal the source bars."""
    from nautilus_trader.model import AggressorSide, Price, Quantity, QuoteTick, TradeId, TradeTick
    out = []
    pp = inst.price_precision
    n = 0
    for t, r in m1.iterrows():
        end = int(t.value)  # bar close
        o, h, l, c = r.open, r.high, r.low, r.close
        seq = (o, h, l, c) if abs(h - o) < abs(o - l) else (o, l, h, c)
        for k, (sec, px) in enumerate(zip((59, 45, 30, 1), seq)):
            ts = end - sec * 1_000_000_000
            out.append(TradeTick(inst.id, Price(px, pp), Quantity(max(r.volume / 4, 0.001), 3),
                                 AggressorSide.BUY if k % 2 else AggressorSide.SELL, TradeId(str(n)), ts, ts))
            out.append(QuoteTick(inst.id, Price(px - spread / 2, pp), Price(px + spread / 2, pp), Quantity(5, 3),
                                 Quantity(5, 3), ts + 1, ts + 1))
            n += 1
    return out


def paper_session(store, name, inst, m1, *, strategy, params, warmup_bars=0, history=None, gap_loader=None,
                  profile="balanced", spread=2.0, starting=10_000.0, create=True, ticks=None):
    """One paper session on the live tick path (SleeveRuntime in live mode, trades/quotes, INTERNAL 1-minute bars),
    as sleeve_fund.paper.node builds it, over the 1-minute bars m1. Reuse `store` for a restart."""
    from decimal import Decimal
    from nautilus_trader.backtest import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.common import LoggerConfig, LogLevel
    from nautilus_trader.model import AccountType, BarType, OmsType, TraderId, Money, Currency
    from sleeve_fund import markets
    from sleeve_fund.instruments import ScheduleFeeModel, fill_model
    from sleeve_fund.paper.runtime import SleeveRuntime
    from sleeve_fund.store import utcnow
    from sleeve_fund.strategies import REGISTRY
    from sleeve_fund.paper.config import auto_warmup
    venue = str(inst.id.venue)
    if create:
        store.create_sleeve(name=name, strategy=strategy, instrument="BTC/USDT" if venue == "BINANCE" else "BTC/USD",
                            bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=starting, risk_profile=profile,
                            params=params, venue=venue if venue != "KRAKEN" else None)
    fees = markets.fees_for(params, venues.venue(venue).fees, venue)
    runtime = SleeveRuntime(store, name, tick_seconds=30)
    perp = markets.is_perp(params)
    book = runtime.book
    quote = inst.quote_currency
    if perp:
        bal = [Money(book["cash"] + book["qty"] * (book["entry_px"] or 0.0), quote)]
    else:
        bal = [Money(book["cash"], quote)] + ([Money(book["qty"], inst.base_currency)] if book["qty"] > 0 else [])
    engine = BacktestEngine(BacktestEngineConfig(trader_id=TraderId.from_str("PAPER-001"),
                                                 logging=LoggerConfig(stdout_level=LogLevel.ERROR)))
    fee_model = ScheduleFeeModel(fees)
    try:
        engine.add_venue(venue=inst.id.venue, oms_type=OmsType.NETTING,
                         account_type=AccountType.MARGIN if perp else AccountType.CASH,
                         default_leverage=markets.VENUE_LEVERAGE if perp else None, base_currency=None,
                         starting_balances=bal, fee_model=fee_model, fill_model=fill_model())
        engine.add_instrument(inst)
        engine.add_data(ticks if ticks is not None else ticks_from_bars(inst, m1, spread))
        scls, ccls = REGISTRY[strategy]
        wb = max(warmup_bars, auto_warmup(strategy, params, "1-MINUTE-LAST-INTERNAL"))
        cfg = ccls(instrument_id=inst.id, bar_type=BarType.from_str(f"{inst.id}-1-MINUTE-LAST-INTERNAL"),
                   assumed_taker_fee=float(fees.taker), assumed_half_spread=spread / 2 / float(m1.close.iloc[0]),
                   warmup_bars=wb, **params)
        s = scls(cfg).attach_runtime(runtime)
        if history is not None:
            s.attach_history(history)
        if gap_loader is not None:
            s.attach_gap_loader(gap_loader)
        s.simulated_venue = True
        s.fee_model = fee_model
        engine.add_strategy(s)
        engine.run()
        return s, runtime
    finally:
        runtime.now = utcnow
        engine.dispose()


def backtest(inst, prices, *, strategy, params, minutes=1, profile="balanced", half_spread=None, exec_prices=None,
             exec_minutes=1, starting=10_000.0):
    from sleeve_fund.research.runner import run_backtest
    kw = dict(exec_prices=exec_prices, exec_minutes=exec_minutes) if exec_prices is not None else {}
    return quiet(lambda: run_backtest(strategy, prices, inst, params=params, starting_capital=starting,
                                      risk_profile=profile, bar_minutes=minutes, half_spread=half_spread, **kw))

TREE = "PR" if "p155" in (os.environ.get("PYTHONPATH") or "") else "main"


def holed_store(root, m1_close_indexed, holes, venue="BINANCE", pair="BTC/USDT"):
    """A history store written by the hub path (append_bars, holes stay holes) from 1-minute bars indexed by
    CLOSE time, with the minutes whose OPEN times are in `holes` left out."""
    from sleeve_fund.history import HistoryStore
    hs = HistoryStore(root)
    df = m1_close_indexed.copy()
    df.index = df.index - pd.Timedelta(minutes=1)  # open times
    df = df[~df.index.isin(pd.DatetimeIndex(holes))]
    rows = [(int(t.value), r.open, r.high, r.low, r.close, r.volume) for t, r in df.iterrows()]
    for i in range(0, len(rows), 50_000):
        hs.append_bars(venue, pair, rows[i:i + 50_000], "live")
    return hs


def random_holes(m1, share, seed=5, outages=()):
    """Open times of minutes to drop: `share` of them at random, plus each (start, minutes) outage."""
    rng = np.random.default_rng(seed)
    opens = m1.index - pd.Timedelta(minutes=1)
    pick = set(opens[rng.random(len(opens)) < share])
    for start, n in outages:
        pick |= set(pd.date_range(pd.Timestamp(start, tz="UTC"), periods=n, freq="1min"))
    return sorted(pick)
