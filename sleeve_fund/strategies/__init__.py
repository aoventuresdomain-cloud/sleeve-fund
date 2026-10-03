from sleeve_fund.strategies.base import IdeaSpec, LongFlatConfig, LongFlatStrategy
from sleeve_fund.strategies.buy_and_hold import BuyAndHold, BuyAndHoldConfig
from sleeve_fund.strategies.trend_filter import TrendFilter, TrendFilterConfig

REGISTRY = {
    "buy_and_hold": (BuyAndHold, BuyAndHoldConfig),
    "trend_filter": (TrendFilter, TrendFilterConfig),
}

__all__ = [
    "REGISTRY",
    "BuyAndHold",
    "BuyAndHoldConfig",
    "IdeaSpec",
    "LongFlatConfig",
    "LongFlatStrategy",
    "TrendFilter",
    "TrendFilterConfig",
]
