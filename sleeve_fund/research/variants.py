"""The pre-registered trend-long variants (research/quant-researcher/trend-variants-preregistration.md; the PM approved
the round on 21 Oct): each one's definition file and the settings it runs with, fixed before any run. A study of a
variant runs exactly these; neighbouring settings are robustness only, never picked from. Studies only (R-06, R-07):
paper and live refuse them on Start until the PM approves (strategies.check_study_only)."""

from __future__ import annotations

from pathlib import Path

DEFINITIONS = Path(__file__).resolve().parents[2] / "configs" / "definitions"
# The variants' own definition files sit apart from the Strategy tab's (configs/definitions/*.toml): studies only.
VARIANTS = DEFINITIONS / "variants"

VOL_TARGET = {"annual": 0.40, "lookback_days": 30}  # V3, V4: 40% a year, 30 days of daily returns

# placebo: a filter on its parent's entries, judged against a placebo of the parent too (R-24)
PREREGISTERED = {
    "V1": {"parent": "qr01b_trend_multispeed_long", "definition": "qr01b_trend_multispeed_long_v1_funding",
           "params": {}, "placebo": True},
    "V2": {"parent": "qr01_trend_ema_long", "definition": "qr01_trend_ema_long_v2_funding", "params": {},
           "placebo": True},
    "V3": {"parent": "qr01b_trend_multispeed_long", "definition": "qr01b_trend_multispeed_long",
           "params": {"vol_target_size": VOL_TARGET}},
    "V4": {"parent": "qr01_trend_ema_long", "definition": "qr01_trend_ema_long", "params": {"vol_target_size": VOL_TARGET}},
    # qr01b's V3 (research/quant-researcher/qr01b-v3-adx-preregistration.md, Advisor cleared 24 Oct), long and short in
    # one definition by the PM's change of 24 Oct: each leg its parent's with a daily ADX(14) <= 25 entry filter. No
    # placebo: the fast path runs neither slower candles nor a short leg.
    "V3_ADX": {"parent": "qr01b_trend_multispeed_long", "short_parent": "qr01b_trend_multispeed_short",
               "definition": "qr01b_trend_multispeed_ls_v3_adx", "params": {}},
    # Improve-loop round 2 (Researcher 24 Oct, Advisor cleared): V3 plus one rule, shorts only below the daily SMA200.
    "V4_SHORT_SMA200": {"parent": "qr01b_trend_multispeed_long", "short_parent": "qr01b_trend_multispeed_short",
                        "definition": "qr01b_trend_multispeed_ls_v4_short_sma200", "params": {}},
    # Improve-loop round 0 (research/quant-researcher/improve-loop-plan-24oct.md; PM 24 Oct: the top 3, long and short in
    # one Strategy): each parent long exactly, plus its mirror short leg.
    "QR23_LS": {"parent": "qr23_vma_band_sma50_long", "definition": "qr23_vma_band_sma50_ls", "params": {}},
    "QR21_LS": {"parent": "qr21_trend_sma200_long", "definition": "qr21_trend_sma200_ls", "params": {}},
    # Improve-loop round 1 (Researcher 24 Oct, Advisor cleared): one change each to round 0's long and short.
    "QR23_R1": {"parent": "qr23_vma_band_sma50_ls", "definition": "qr23_vma_band_sma50_ls_r1_short_sma200", "params": {}},
    "QR21_R1": {"parent": "qr21_trend_sma200_ls", "definition": "qr21_trend_sma200_ls_r1_slope", "params": {}},
    # Round 2 (Researcher 24 Oct, Advisor cleared): qr23 round 0 plus qr21 round 1's slope rule on its 50-day average.
    "QR23_R2": {"parent": "qr23_vma_band_sma50_ls", "definition": "qr23_vma_band_sma50_ls_r2_slope", "params": {}},
}


def definition_path(name: str) -> Path:
    """A variant's definition file, or its parent's (the Strategy tab's)."""
    own = VARIANTS / f"{name}.toml"
    return own if own.is_file() else DEFINITIONS / f"{name}.toml"
