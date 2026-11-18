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
BATCH_VOL = {"annual": 0.40, "lookback_days": 30}  # idea batch S1-S4 (18 Nov), its own figure though the same

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
    # One long-only test (Researcher 25 Oct, Advisor cleared as try 28; PM chose it 25 Oct): V4 without its short leg.
    "V5_LO_ADX": {"parent": "qr01b_trend_multispeed_long", "definition": "qr01b_trend_multispeed_lo_v5_adx", "params": {}},
    # Improve-loop round 0 (research/quant-researcher/improve-loop-plan-24oct.md; PM 24 Oct: the top 3, long and short in
    # one Strategy): each parent long exactly, plus its mirror short leg.
    "QR23_LS": {"parent": "qr23_vma_band_sma50_long", "definition": "qr23_vma_band_sma50_ls", "params": {}},
    "QR21_LS": {"parent": "qr21_trend_sma200_long", "definition": "qr21_trend_sma200_ls", "params": {}},
    # Improve-loop round 1 (Researcher 24 Oct, Advisor cleared): one change each to round 0's long and short.
    "QR23_R1": {"parent": "qr23_vma_band_sma50_ls", "definition": "qr23_vma_band_sma50_ls_r1_short_sma200", "params": {}},
    "QR21_R1": {"parent": "qr21_trend_sma200_ls", "definition": "qr21_trend_sma200_ls_r1_slope", "params": {}},
    # Round 2 (Researcher 24 Oct, Advisor cleared): qr23 round 0 plus qr21 round 1's slope rule on its 50-day average.
    "QR23_R2": {"parent": "qr23_vma_band_sma50_ls", "definition": "qr23_vma_band_sma50_ls_r2_slope", "params": {}},
    # The next round (PM approved the plan 30 Oct 14:57 UTC; Advisor cleared both pre-registrations 30 Oct 15:07 UTC,
    # research/quant-researcher/next-round/): new families, so no parent and no placebo.
    "QRA0_FUNDING_FADE": {"definition": "qra0_funding_fade", "params": {}},
    "QRB0_RANGE_REVERSION_ADX": {"definition": "qrb0_range_reversion_adx", "params": {}},
    # Family H, pullback inside a trend (Advisor cleared 31 Oct 05:56 UTC as family B try 15; same folder): a new idea,
    # so no parent and no placebo.
    "QRH0_TREND_PULLBACK": {"definition": "qrh0_trend_pullback", "params": {}},
    # Family H improve items (PM reopened H 16 Nov 21:06 UTC; Advisor cleared items 1-5, 7, 8 and P1 16 Nov ~22:10 UK,
    # same folder): each one change from H0, run alone against H0 (configs/studies/family-h-improve-list.md). No placebo.
    "QRH1_TREND_PULLBACK_TS6": {"definition": "qrh1_trend_pullback_ts6", "params": {}},
    "QRH2_TREND_PULLBACK_RUNNERS": {"definition": "qrh2_trend_pullback_runners", "params": {}},
    "QRH3_TREND_PULLBACK_STOP15": {"definition": "qrh3_trend_pullback_stop15", "params": {}},
    "QRH4_TREND_PULLBACK_SMA100": {"definition": "qrh4_trend_pullback_sma100", "params": {}},
    "QRH5_TREND_PULLBACK_STOP35": {"definition": "qrh5_trend_pullback_stop35", "params": {}},
    "QRH7_TREND_PULLBACK_SHORT_ONLY": {"definition": "qrh7_trend_pullback_short_only", "params": {}},
    "QRH8_TREND_PULLBACK_RSI595": {"definition": "qrh8_trend_pullback_rsi595", "params": {}},
    "QRHP1_TREND_PULLBACK_BREAK_FILTER": {"definition": "qrhp1_trend_pullback_break_filter", "params": {}},
    # The H combination (Advisor 18 Nov 17:38, family B slot 24): H7's short side with H1's 6-candle time stop and P1's
    # short-side break filter, nothing else. It runs only once the exit-delay fix is deployed and H0 is re-run on it.
    "QRHC_TREND_PULLBACK_COMBINATION": {"definition": "qrhc_trend_pullback_combination", "params": {}},
    # Idea batch N1-N4 (Advisor cleared 17 Nov ~08:15 UK with pre-run implementation fixes, no try spent;
    # research/quant-researcher/next-round/idea-batch-17nov.md): new ideas, so no parent and no placebo.
    "QRN1_CAPITULATION_REVERSAL": {"definition": "qrn1_capitulation_reversal", "params": {}},
    "QRN2_SHORT_TREND_SLEEVE": {"definition": "qrn2_short_trend_sleeve", "params": {}},
    "QRN3_WEEKEND_FADE": {"definition": "qrn3_weekend_fade", "params": {}},
    "QRN4_WIDE_RANGE_CONTINUATION": {"definition": "qrn4_wide_range_continuation", "params": {}},
    # Family C, relative strength across BTC, ETH and SOL (PM yes 16 Nov 21:06 UTC; Advisor cleared, rev 3 frozen
    # 17 Nov; same folder): a new family, so no parent and no placebo. 0.5x of the fund a leg, 1x gross (Advisor Q2), which makes
    # it a rank book sized off the fund.
    "QRC0_RELATIVE_STRENGTH": {"definition": "qrc0_relative_strength", "params": {"position_cap_pct": 0.5}},
    # Idea batch S1-S4 (Researcher 18 Nov, research/quant-researcher/next-round/idea-batch-18nov.md; Advisor cleared
    # for the 2024 signal count as family B tries 27-30, FINAL ~10:56 UTC): each sized to 40% a year of 30-day
    # volatility on the study's own settings, nothing about sizing in the definitions. S3 and S4 each carry the one
    # fallback they pre-register, used only if the main version's count is under 60 expected test trades.
    "QRS1_SHORT_PULLBACK_DOWNTREND": {"definition": "qrs1_short_pullback_downtrend", "params": {"vol_target_size": BATCH_VOL}},
    "QRS2_LONDON_ASIA_RANGE_FADE": {"definition": "qrs2_london_asia_range_fade", "params": {"vol_target_size": BATCH_VOL}},
    "QRS3_SHOCK_FADE_4H": {"definition": "qrs3_shock_fade_4h", "params": {"vol_target_size": BATCH_VOL}},
    "QRS3_SHOCK_FADE_4H_FB175": {"definition": "qrs3_shock_fade_4h_fb175", "params": {"vol_target_size": BATCH_VOL}},
    "QRS4_VWAP_RANGE_REVERSION_4H": {"definition": "qrs4_vwap_range_reversion_4h",
                                     "params": {"vol_target_size": BATCH_VOL}},
    "QRS4_VWAP_RANGE_REVERSION_4H_FB125": {"definition": "qrs4_vwap_range_reversion_4h_fb125",
                                           "params": {"vol_target_size": BATCH_VOL}},
}


def definition_path(name: str) -> Path:
    """A variant's definition file, or its parent's (the Strategy tab's)."""
    own = VARIANTS / f"{name}.toml"
    return own if own.is_file() else DEFINITIONS / f"{name}.toml"
