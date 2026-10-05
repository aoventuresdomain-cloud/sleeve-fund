from sleeve_fund.strategies.base import IdeaSpec, LongFlatConfig, LongFlatStrategy
from sleeve_fund.strategies.buy_and_hold import BuyAndHold, BuyAndHoldConfig
from sleeve_fund.strategies.dip_buy import DipBuy, DipBuyConfig
from sleeve_fund.strategies.donchian import Donchian, DonchianConfig
from sleeve_fund.strategies.ping_pong import PingPong, PingPongConfig
from sleeve_fund.strategies.rsi_bands import RsiBands, RsiBandsConfig
from sleeve_fund.strategies.rsi_cross import RsiCross, RsiCrossConfig
from sleeve_fund.strategies.rsi_pullback import RsiPullback, RsiPullbackConfig
from sleeve_fund.strategies.trend_filter import TrendFilter, TrendFilterConfig

REGISTRY = {
    "buy_and_hold": (BuyAndHold, BuyAndHoldConfig),
    "dip_buy": (DipBuy, DipBuyConfig),
    "donchian": (Donchian, DonchianConfig),
    "ping_pong": (PingPong, PingPongConfig),
    "rsi_bands": (RsiBands, RsiBandsConfig),
    "rsi_cross": (RsiCross, RsiCrossConfig),
    "rsi_pullback": (RsiPullback, RsiPullbackConfig),
    "trend_filter": (TrendFilter, TrendFilterConfig),
}

PERP_WEIGHT_REFUSAL = "sized by weight: not available on perpetuals until order sizing is rebuilt"


def check_perp_sizing(strategy: str, params: dict | None) -> None:
    """Refuse a model that sizes its position by a target weight (Donchian, trend filter with vol_target) on a
    perpetual. A perp entry ignores the weight and opens at the full cap (review round 13, E13-6), and order sizing
    is being rebuilt, so until then such a model is refused up front wherever it could start, backtest or be
    studied; every entry point calls this. Side-only models and spot are unaffected."""
    from sleeve_fund import markets

    params = params or {}
    if strategy in REGISTRY and markets.is_perp(params) and REGISTRY[strategy][0].weight_sized(params):
        raise ValueError(f"{strategy.replace('_', ' ').capitalize()} is {PERP_WEIGHT_REFUSAL}")


__all__ = [
    "PERP_WEIGHT_REFUSAL",
    "REGISTRY",
    "check_perp_sizing",
    "BuyAndHold",
    "BuyAndHoldConfig",
    "DipBuy",
    "DipBuyConfig",
    "Donchian",
    "DonchianConfig",
    "IdeaSpec",
    "LongFlatConfig",
    "LongFlatStrategy",
    "PingPong",
    "PingPongConfig",
    "RsiBands",
    "RsiBandsConfig",
    "RsiCross",
    "RsiCrossConfig",
    "RsiPullback",
    "RsiPullbackConfig",
    "TrendFilter",
    "TrendFilterConfig",
]
