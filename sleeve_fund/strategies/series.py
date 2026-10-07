"""The chart's indicator overlay (P1-3s): each indicator a strategy draws, as the strategy itself computed it on each
closed candle.

The strategy is built as a backtest builds it and fed the stored candles through its own update path (_accept, which
calls update_indicators and feeds any slower candles), never a parallel calculation, so every point is the value the
model read when it decided on that candle. It is fed closed candles only, in order, so no point exists before its
candle closed. Nothing trades: the model's decision rules never run.

Payload (v2/chart-indicators-shape.md, agreed with the Front-end Engineer 5 Oct, settled_from added 6 Oct):
    {"key", "label", "pane": "price" | "lower", "kind": "line", "group", "levels", "tf", "shown",
     "settled_from": t | None, "points": [[t, v], ...]}
shown: drawn when the chart opens (each line has its own toggle); a rule-builder strategy shows only the outputs its
rules read. Draw each line as steps held from one candle close to the next, never joined straight: a straight join
would show crosses between closes the model never saw (Independent Quant Advisor, 6 Oct 23:24).
t is the candle's close, UTC seconds; v is None only where the indicator has no value yet. Points before
settled_from are the model's warm-up: drawn as not settled (dashed, greyed), never as a normal line.
"""

from __future__ import annotations

import importlib
import math

import pandas as pd

from sleeve_fund.data import bar_type_for, to_bars
from sleeve_fund.instruments import BOOK_SHARE
from sleeve_fund.strategies import REGISTRY


def build(strategy_name: str, instrument, params: dict, bar_minutes: int):
    """The strategy as run_backtest builds it, without an engine: same config, same warm-up count."""
    strategy_cls, config_cls = REGISTRY[strategy_name]
    config = config_cls(instrument_id=instrument.id, bar_type=bar_type_for(instrument, bar_minutes),
                        assumed_taker_fee=float(instrument.taker_fee), volume_scale=BOOK_SHARE, **params)
    strategy = strategy_cls(config)
    strategy.settle_bars_needed = warmup(strategy_name, params, bar_minutes)
    return strategy


def warmup(strategy_name: str, params: dict, bar_minutes: int) -> int:
    """Candles of `bar_minutes` the model needs before its indicators read as settled, as a backtest counts them."""
    strategy_cls = REGISTRY[strategy_name][0]
    spec = getattr(importlib.import_module(strategy_cls.__module__), "SPEC", None)
    return strategy_cls.warmup_needed({**(spec.default_params if spec is not None else {}), **params}, bar_minutes)


def indicator_series(strategy_name: str, candles: pd.DataFrame, instrument, params: dict | None = None,
                     bar_minutes: int = 1440) -> list[dict]:
    """candles: closed candles of `bar_minutes`, indexed by close time, as the history store returns them."""
    from sleeve_fund.research.runner import _book_volume

    params = dict(params or {})
    strategy = build(strategy_name, instrument, params, bar_minutes)
    meta = strategy.indicator_meta()
    series = {k: {"key": k, "kind": "line", "group": None, "levels": None, "tf": None, "shown": True, **m,
                  "settled_from": None,
                  "points": []} for k, m in meta.items()}
    if not series:
        return []
    bars = to_bars(_book_volume(candles, instrument), instrument, strategy._cfg.bar_type)
    for bar in bars:
        if not strategy._accept(bar):
            continue
        t = int(bar.ts_event // 1_000_000_000)
        values = strategy.indicator_values()
        for key, s in series.items():
            v = values.get(key)
            v = None if v is None or not math.isfinite(v) else float(v)
            s["points"].append([t, v])
            if s["settled_from"] is None and v is not None and strategy.indicator_settled(key):
                s["settled_from"] = t
    return list(series.values())
