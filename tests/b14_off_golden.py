"""B-14 OFF golden (QD for QD3, b14aa): with resting targets off, the orders and fills of a target path and a stop path,
on the paper replay and the backtest engine, as 274ecd88 (before B-14) made them. Rebuild the file only on a change
that is meant to move OFF orders: run this module from the repo root (PYTHONPATH=.:tests) and write its output to
tests/data/golden/b14_off_<base>.json (b14_off_662e8a0b.json since TEC-B14AA-REBASE; the 274ecd88 one is kept)."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from test_long_short import PERP, _meta, _record

from sleeve_fund.research.replay import replay
from sleeve_fund.store import Store

UP = [(5, 0.0), (20, 0.015), (10, 0.03), (5, 0.0), (10, -0.03), (10, 0.0)]
DOWN = [(5, 0.0), (20, 0.008), (5, 0.0), (10, -0.03), (10, 0.0)]
DROP = ("id", "order_id", "trade_id", "updated_at", "ts")


def strip(rows):
    return [{k: (str(v) if not isinstance(v, (int, float, str, bool, type(None), dict, list)) else v)
             for k, v in r.items() if k not in DROP} for r in rows]


def paper(legs):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.jsonl.gz"
        params = {"rise": 0.01, "dip": 0.005, "stop_loss": 0.01, "take_profit": 0.01, **PERP}
        _record(p, _meta(10_000, params), legs)
        store = Store.in_memory()
        replay(p, with_fills=True, store=store)
        return {"orders": strip(store.orders("ping-pong-test")[::-1]),
                "fills": strip(store.fills("ping-pong-test", limit=10_000)[::-1])}


def backtest(legs):
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.venues import venue

    px, out = 60_000.0, []
    for minutes, move in legs:
        for i in range(minutes):
            out.append(px * (1 + move * (i + 1) / minutes))
        px = out[-1]
    idx = pd.date_range("2026-10-01 00:01", periods=len(out), freq="1min", tz="UTC")
    c = np.array(out)
    bars = pd.DataFrame({"open": np.r_[c[0], c[:-1]], "high": np.maximum(c, np.r_[c[0], c[:-1]]),
                         "low": np.minimum(c, np.r_[c[0], c[:-1]]), "close": c, "volume": 1e6}, index=idx)
    inst = venue("KRAKEN").instrument("BTC", "USD", price_precision=1)
    res = run_backtest("ping_pong", bars, inst, params={"rise": 0.01, "dip": 0.005, "stop_loss": 0.01,
                                                        "take_profit": 0.01, **PERP},
                       starting_capital=10_000, risk_profile="aggressive", bar_minutes=1)
    f = res.fills.reset_index(drop=True)
    f = f[[c for c in f.columns if c not in DROP and "id" not in c.lower()]]
    return {"fills": json.loads(f.to_json(orient="records", date_format="iso")),
            "intents": sorted(str(v.get("intent")) for v in res.decisions.values()),
            "equity_last": round(float(res.equity.iloc[-1]), 8), "fees": round(float(res.fees_paid), 8)}


def off_runs() -> str:
    return json.dumps({"paper_target": paper(UP), "paper_stop": paper(DOWN),
                       "backtest_target": backtest(UP), "backtest_stop": backtest(DOWN)}, sort_keys=True, default=str)


if __name__ == "__main__":
    print(off_runs())
