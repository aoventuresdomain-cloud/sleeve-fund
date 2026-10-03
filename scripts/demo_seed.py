"""Fill a local database with realistic-looking sleeves for dashboard work and screenshots.

Synthetic prices only, run through the real strategies, runtime and journal. Never point
this at the production database.

    python scripts/demo_seed.py sqlite:///demo.db
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

import sleeve_fund.store as store_mod
from sleeve_fund.data import synthetic_ohlcv
from sleeve_fund.instruments import spot_pair
from sleeve_fund.paper.runtime import SleeveRuntime
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.store import Store

SLEEVES = [
    # name, strategy, base, quote, start price, seed, days, profile, params, extra
    ("btc-trend-daily", "trend_filter", "BTC", "USD", 60_000, 11, 365, "balanced", {"fast": 10, "slow": 30}, {}),
    ("eth-trend-fast", "trend_filter", "ETH", "USD", 2_400, 4, 300, "aggressive",
     {"fast": 8, "slow": 25, "stop_loss": 0.08}, {}),
    ("sol-rsi-pullback", "rsi_pullback", "SOL", "USD", 140, 9, 330, "balanced",
     {"rsi_entry": 55, "vol_mult": 0.8, "atr_mult": 2.5, "ema_period": 50}, {}),
    ("sui-trend", "trend_filter", "SUI", "USD", 3.2, 21, 160, "conservative", {"fast": 5, "slow": 20},
     {"halt": True}),
]


def main(url: str) -> None:
    if not url.startswith("sqlite"):
        raise SystemExit("demo data goes in a local sqlite database only")
    store = Store(url)
    real_now = store_mod.utcnow
    for name, strat, base, quote, px, seed, days, prof, params, extra in SLEEVES:
        if any(s.name == name for s in store.sleeves()):
            continue
        start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")  # last bar closes today
        prices = synthetic_ohlcv(days=days, start=start, seed=seed, start_price=px, vol=0.03)
        store.create_sleeve(name=name, strategy=strat, instrument=f"{base}/{quote}", bar_spec="1-DAY-LAST-EXTERNAL",
                            starting_balance=10_000, params=params, risk_profile=prof)
        store.decide("PM", "create", f"demo: {strat} on {base}", name)
        rt = SleeveRuntime(store, name, tick_seconds=6 * 3600)
        store_mod.utcnow = lambda: rt.now()  # journal rows carry the simulated time
        try:
            run_backtest(strat, prices, spot_pair(base, quote, price_precision=4 if px < 10 else 2), params=params,
                         runtime=rt)
        finally:
            store_mod.utcnow = real_now
        if extra.get("halt"):
            store.set_status(name, "halted", "drawdown 10.4% hit the 10% limit")
            store.event(name, "error", "risk_halt", "drawdown 10.4% hit the 10% limit; flattened, PM must resume")
        else:
            store.set_status(name, "running")
        store.heartbeat(name)
    if not store.orders("sol-rsi-pullback", statuses=("rejected",)):
        # One rejected order so the blotter's rejected view has something to show.
        store.record_order("sol-rsi-pullback", order_id="O-DEMO-REJECT", side="BUY", qty=12.5, intent="entry",
                           reason="RSI 41.2 below 55, close 151.3 above the 50-bar EMA 148.9, volume 1.31x normal "
                                  "(needs 0.8x)", signal={"rsi": 41.2, "ema_50": 148.9, "volume_x": 1.31,
                                                          "close": 151.3, "sized_by": "balanced risk profile cap"})
        store.update_order("O-DEMO-REJECT", status="rejected", message="EOrder:Insufficient funds")
    print(f"seeded {url}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sqlite:///demo.db")
