from sleeve_fund.strategies.base import IdeaSpec, LongFlatConfig, LongFlatStrategy
from sleeve_fund.strategies.buy_and_hold import BuyAndHold, BuyAndHoldConfig
from sleeve_fund.strategies.candle_step import CandleStep, CandleStepConfig
from sleeve_fund.strategies.dip_buy import DipBuy, DipBuyConfig
from sleeve_fund.strategies.donchian import Donchian, DonchianConfig
from sleeve_fund.strategies.ping_pong import PingPong, PingPongConfig
from sleeve_fund.strategies.rsi_bands import RsiBands, RsiBandsConfig
from sleeve_fund.strategies.rsi_cross import RsiCross, RsiCrossConfig
from sleeve_fund.strategies.rsi_pullback import RsiPullback, RsiPullbackConfig
from sleeve_fund.strategies.rules import Rules, RulesConfig
from sleeve_fund.strategies.trend_filter import TrendFilter, TrendFilterConfig

REGISTRY = {
    "buy_and_hold": (BuyAndHold, BuyAndHoldConfig),
    "candle_step": (CandleStep, CandleStepConfig),
    "dip_buy": (DipBuy, DipBuyConfig),
    "donchian": (Donchian, DonchianConfig),
    "ping_pong": (PingPong, PingPongConfig),
    "rsi_bands": (RsiBands, RsiBandsConfig),
    "rsi_cross": (RsiCross, RsiCrossConfig),
    "rsi_pullback": (RsiPullback, RsiPullbackConfig),
    "rules": (Rules, RulesConfig),
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


def study_only_uses(params: dict | None) -> list[str]:
    """What in these settings a study may run but paper and live may not (R-06, R-07), in words; empty when none."""
    from sleeve_fund.strategies.definitions import STUDY_ONLY_BLOCKS

    params = params or {}
    blocks = (params.get("definition") or {}).get("blocks") or {}
    found = sorted({STUDY_ONLY_BLOCKS[b["kind"]] for b in blocks.values()
                    if isinstance(b, dict) and b.get("kind") in STUDY_ONLY_BLOCKS})
    if params.get("vol_target_size") is not None:
        found.append("sizing to a target volatility (vol_target_size)")
    if params.get("blocked_entries"):
        found.append("a placebo's blocked entries (blocked_entries)")
    return found


def check_study_only(strategy: str, params: dict | None) -> None:
    """Refuse, on Start and wherever paper or live would build it, a Strategy using something only studies may run until
    the PM approves it for paper (R-06 funding rate decile, R-07 volatility-targeted sizing). Backtests and studies never
    call this."""
    found = study_only_uses(params)
    if found:
        raise ValueError(f"it uses {' and '.join(found)}, which runs in studies only until the PM approves it for paper")


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
    """Refuse a model on a perpetual that places no stop when its risk profile, lowered by its own leverage cap
    (B-35 (a)), allows more than 1x (the Advisor's
    interim minimum for the hand-coded models, until every model must place one). Such a position relies on the
    liquidation price alone. Spot, and any strategy that places a stop, are unaffected. Called wherever a
    strategy is created, edited or started; backtests and studies still run, so the risk can be measured."""
    from sleeve_fund import markets
    from sleeve_fund.risk import PROFILES, capped_leverage, own_leverage

    profile = PROFILES.get(risk_profile)
    if profile is None or not markets.is_perp(params or {}) or places_stop(strategy, params):
        return
    # B-35 (a): the Strategy's own leverage cap (max_leverage) can hold it there under a profile that allows more
    if (lev := capped_leverage(profile, params)) > STOPLESS_MAX_LEVERAGE:
        within = [p.name for p in PROFILES.values() if p.max_leverage <= STOPLESS_MAX_LEVERAGE]
        allows = (f"its own leverage cap is {lev:g}x" if own_leverage(params) is not None and lev < profile.max_leverage
                  else f"the {risk_profile} risk profile allows {lev:g}x")
        raise ValueError(f"{strategy.replace('_', ' ').capitalize()} on a perpetual places no stop, so it is capped at "
                         f"{STOPLESS_MAX_LEVERAGE:g}x, and {allows}. Choose the {' or '.join(within)} profile, set its "
                         f"leverage cap to {STOPLESS_MAX_LEVERAGE:g}x, or set a stop-loss, an ATR stop or a swing stop")


# HoE 21 Oct (TP-RESTART): an {r} target is set from the opening fill and held in memory only, so until it is journaled
# and restored on a restart, paper refuses to start one. A {level} target is worked out afresh at each close.
TARGET_RESTART_REFUSAL = "take-profit at {n}×R can't run on paper until its target survives a restart (TP-RESTART)"


def check_target_restart(strategy: str, params: dict | None) -> None:
    """Refuse to start a rules strategy whose definition takes profit at N×R, shared or on either side (TP-RESTART).
    Called where paper starts a strategy; backtests and studies still run it."""
    from sleeve_fund.strategies.definitions import side_exits

    defn = (params or {}).get("definition")
    if strategy != "rules" or not isinstance(defn, dict):
        return
    for side in ("long", "short"):
        tp = side_exits(defn.get("exits"), side).get("take_profit")
        if isinstance(tp, dict) and "r" in tp:
            raise ValueError(TARGET_RESTART_REFUSAL.format(n=f"{tp['r']:g}"))


__all__ = [
    "PERP_WEIGHT_REFUSAL",
    "REGISTRY",
    "STOPLESS_MAX_LEVERAGE",
    "TARGET_RESTART_REFUSAL",
    "check_perp_sizing",
    "check_perp_stop",
    "check_target_restart",
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
    "Rules",
    "RulesConfig",
    "TrendFilter",
    "TrendFilterConfig",
]
