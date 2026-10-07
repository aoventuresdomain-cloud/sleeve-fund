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


STOP_KEYS = ("stop_loss", "stop_atr", "stop_swing_bars")  # the exits that place a stop at the venue on every entry
STOPLESS_MAX_LEVERAGE = 1.0  # the Independent Quant Advisor's interim minimum (6 Oct) for a perp with no placed stop


def places_stop(strategy: str, params: dict | None) -> bool:
    """Whether every entry places a stop: a stop-loss %, an ATR stop or a swing stop, set in the strategy's own
    settings or its model's defaults (dip_buy's ATR stop). A time stop or a trail checked only at bar closes
    (rsi_pullback's) is not a placed stop: a gap goes straight through it."""
    import importlib

    try:
        defaults = importlib.import_module(f"sleeve_fund.strategies.{strategy}").SPEC.default_params or {}
    except (ImportError, AttributeError):
        defaults = {}
    merged = {**defaults, **(params or {})}
    return any((merged.get(k) or 0) > 0 for k in STOP_KEYS)


def check_perp_stop(strategy: str, params: dict | None, risk_profile: str) -> None:
    """Refuse a model on a perpetual that places no stop when its risk profile allows more than 1x (the Advisor's
    interim minimum for the hand-coded models, until every model must place one). Such a position relies on the
    liquidation price alone. Spot, and any strategy that places a stop, are unaffected. Called wherever a
    strategy is created, edited or started; backtests and studies still run, so the risk can be measured."""
    from sleeve_fund import markets
    from sleeve_fund.risk import PROFILES

    profile = PROFILES.get(risk_profile)
    if profile is None or not markets.is_perp(params or {}) or places_stop(strategy, params):
        return
    if profile.max_leverage > STOPLESS_MAX_LEVERAGE:
        within = [p.name for p in PROFILES.values() if p.max_leverage <= STOPLESS_MAX_LEVERAGE]
        raise ValueError(f"{strategy.replace('_', ' ').capitalize()} on a perpetual places no stop, so it is capped at "
                         f"{STOPLESS_MAX_LEVERAGE:g}x, and the {risk_profile} risk profile allows "
                         f"{profile.max_leverage:g}x. Choose the {' or '.join(within)} profile, or set a stop-loss, an "
                         "ATR stop or a swing stop")


__all__ = [
    "PERP_WEIGHT_REFUSAL",
    "REGISTRY",
    "STOPLESS_MAX_LEVERAGE",
    "check_perp_sizing",
    "check_perp_stop",
    "places_stop",
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
