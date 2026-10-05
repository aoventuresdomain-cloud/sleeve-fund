from sleeve_fund.strategies.base import IdeaSpec, LongFlatConfig, LongFlatStrategy
from sleeve_fund.strategies.buy_and_hold import BuyAndHold, BuyAndHoldConfig
from sleeve_fund.strategies.ping_pong import PingPong, PingPongConfig
from sleeve_fund.strategies.rsi_bands import RsiBands, RsiBandsConfig
from sleeve_fund.strategies.rsi_cross import RsiCross, RsiCrossConfig
from sleeve_fund.strategies.rsi_pullback import RsiPullback, RsiPullbackConfig
from sleeve_fund.strategies.trend_filter import TrendFilter, TrendFilterConfig

REGISTRY = {
    "buy_and_hold": (BuyAndHold, BuyAndHoldConfig),
    "ping_pong": (PingPong, PingPongConfig),
    "rsi_bands": (RsiBands, RsiBandsConfig),
    "rsi_cross": (RsiCross, RsiCrossConfig),
    "rsi_pullback": (RsiPullback, RsiPullbackConfig),
    "trend_filter": (TrendFilter, TrendFilterConfig),
}

__all__ = [
    "REGISTRY",
    "BuyAndHold",
    "BuyAndHoldConfig",
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
