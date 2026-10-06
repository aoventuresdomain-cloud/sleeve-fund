"""The indicator library: shared blocks every strategy, the chart and research use (v2 P1-3).

Every block is built from plain settings and offers `update_raw(...)` (the inputs it needs),
`update_ohlcv(open, high, low, close, volume, ts_ns=None)` (any block, unused inputs ignored), `handle_bar(bar)`,
`value` (its main output) and `values` (every output by name), `initialized`, `warmup_bars`, `reset()`,
`confirm_lag`, and, where only the close goes in, `peek(close)` (plus `peek_values(close)` with several outputs).
`values` reads None until the block is initialized, and so does `value` on library blocks; Sma, AtrSma and Rsi keep
the `value` the hand-coded models trade, unchanged. `warmup_bars` on a block is its warm-up; on a class,
e.g. Ema.warmup_bars(period=20), it is the warm-up those settings need. Each class lists its SETTINGS for
definition checks. A value depends only on bars already passed, and no block knows its candle size: a strategy
feeds a block closed slower candles to read it on a slower timeframe. The rule evaluator, not the block, keeps
previous values and finds crosses.
"""

from __future__ import annotations

from sleeve_fund.strategies.indicators._common import PERIOD_MAX, SETTLE_LENGTHS, Setting, settle_bars
from sleeve_fund.strategies.indicators.averages import DAY_OF_MINUTE_BARS, NS_PER_DAY, Atr, Ema, Vwap, Wma
from sleeve_fund.strategies.indicators.bands import FLAT_BAND, Bollinger, Donchian, Keltner
from sleeve_fund.strategies.indicators.classic import AtrSma, Rsi, Sma
from sleeve_fund.strategies.indicators.filters import EfficiencyRatio, RelativeVolume
from sleeve_fund.strategies.indicators.oscillators import Stochastic
from sleeve_fund.strategies.indicators.swings import RsiDivergence

BLOCKS: dict[str, type] = {
    "sma": Sma,
    "ema": Ema,
    "wma": Wma,
    "vwap": Vwap,
    "rsi": Rsi,
    "bollinger": Bollinger,
    "atr": Atr,  # Wilder's (P1-I4)
    "atr_sma": AtrSma,  # the simple average the hand-coded models size stops on
    "relative_volume": RelativeVolume,
    "efficiency_ratio": EfficiencyRatio,
    "donchian": Donchian,
    "rsi_divergence": RsiDivergence,
    "stochastic": Stochastic,
    "keltner": Keltner,
}


def _block_class(kind: str) -> type:
    try:
        return BLOCKS[kind]
    except KeyError:
        raise ValueError(f"no indicator block called {kind!r}; known: {', '.join(sorted(BLOCKS))}") from None


def make_block(kind: str, **settings):
    """A block from its name in BLOCKS and its plain settings, as a definition names it. Settings are checked
    against the block's spec, and missing ones take their defaults."""
    cls = _block_class(kind)
    checked = {k: v for k, v in cls.check_settings(settings).items() if v is not None}
    return cls(**checked)


def warmup_for(specs) -> int:
    """Bars of history a strategy needs before every one of its blocks reads as settled. `specs` holds blocks,
    or (kind, settings) pairs as a definition lists them."""
    bars = 0
    for spec in specs:
        if isinstance(spec, tuple):
            kind, settings = spec
            bars = max(bars, _block_class(kind).warmup_bars(**(settings or {})))
        else:
            bars = max(bars, spec.warmup_bars)
    return bars


__all__ = [
    "BLOCKS", "DAY_OF_MINUTE_BARS", "FLAT_BAND", "NS_PER_DAY", "PERIOD_MAX", "SETTLE_LENGTHS", "Atr", "AtrSma", "Bollinger",
    "Donchian", "EfficiencyRatio", "Ema", "Keltner", "RelativeVolume", "Rsi", "RsiDivergence", "Setting", "Sma", "Stochastic",
    "Vwap", "Wma", "make_block", "settle_bars",
    "warmup_for",
]
