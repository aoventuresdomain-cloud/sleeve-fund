"""One-page tear sheet in Markdown, led by out-of-sample results after fees."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from sleeve_fund.research.guardrails import G1_RULES, MIN_OOS_TRADES, nearby_scored
from sleeve_fund.research.ledger import IdeaLedger
from sleeve_fund.research.metrics import (
    daily_returns,
    deflated_sharpe_probability,
    sharpe_beats_probability,
    summary,
    years_covered,
)
from sleeve_fund import funding
from sleeve_fund.research.study import StudyResult

ROBUST_SHARE = 0.6
# G1's bar: at least this probability that the out-of-sample Sharpe beats the benchmark's by more
# than the best of the variants tried would by luck.
G1_CONFIDENCE = 0.95
SHARPE_CHECK = "G1 test: out-of-sample Sharpe clearly beats benchmark after fees"
JUDGED_CHECK = "Runs complete enough to judge"
NOT_JUDGED = "NOT JUDGED"
# Where N or the deflated Sharpe is shown while an idea has a run whose count failed (QA P1-T8, P1-T10).
N_UNCERTAIN = ("N uncertain: a run of this idea couldn't be counted in full, so this N and the bar it sets read low "
               "until it is re-counted.")
# A check that rests on the out-of-sample a study couldn't produce: shown, but not counted as a fail.
NOT_APPLICABLE = "N/A"
NEARBY_CHECK = "Holds at nearby settings"
RANDOM_ENTRY_CHECK = "Beats random entry times"
RANDOM_SIDE_CHECK = "Beats random long or short"
# A perpetual's out-of-sample funding at the venue's own rates: WARN is shown, not a fail; past the limits the study
# isn't judged (funding.baseline_check; Advisor, 6 Oct 2026, QA P1-O17).
FUNDING_CHECK = "Funding charged at the venue's own rates"
OOS_CHECKS = (SHARPE_CHECK, "Holds up when parameters move", "Enough out-of-sample trades to judge",
              RANDOM_ENTRY_CHECK, RANDOM_SIDE_CHECK)
# More of the test windows' trades than this left out at the edges, and the trade count is flagged.
EXCLUDED_FLAG = 0.10


def g1_verdict(checks: list[tuple[str, str, str]]) -> tuple[str, list[str]]:
    """G1 passes only when every check that can fail passes: the Sharpe test, robustness, an unused
    holdout and enough trades. A strong Sharpe on three trades, or on a holdout already looked at,
    is not evidence. A study whose runs raised errors, or whose out-of-sample the risk guard left flat,
    is not judged at all: neither a pass nor a fail. Returns the verdict and the checks that failed."""
    unjudged = [name for name, verdict, _ in checks if verdict == NOT_JUDGED]
    if unjudged:
        return NOT_JUDGED, unjudged
    failed = [name for name, verdict, _ in checks if verdict not in ("PASS", "WARN", "INFO", NOT_APPLICABLE)]
    return ("FAIL" if failed else "PASS"), failed


def _pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.1%}"


def _share(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.0%}"


def _param(v) -> str:
    """A grid value as set: 20, 0.005 or "ema", not int() of it, which crashed on a word and showed a
    0.5% threshold as 0 (round 11 minor)."""
    if isinstance(v, str):
        return v
    f = float(v)
    return str(int(f)) if f.is_integer() else f"{f:g}"


def _num(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"


def _row(label: str, s: dict, b: dict) -> str:
    return (
        f"| {label} | {_pct(s['cagr'])} | {_pct(b['cagr'])} | {_num(s['sharpe'])} | {_num(b['sharpe'])} "
        f"| {_pct(s['max_drawdown'])} | {_pct(b['max_drawdown'])} |"
    )


def _idea(r: StudyResult) -> str:
    from sleeve_fund.research.trials import legacy_idea_hash

    return legacy_idea_hash(r.spec.name)


def _counts(r: StudyResult, ledger: IdeaLedger, register) -> dict:
    """The counts a result is judged by: with the trials register, its idea family's (Advisor, 6 Oct 2026);
    without one, the idea counter's, as before."""
    return register.counts(_idea(r)) if register is not None else ledger.counts()


def _trial_spread(r: StudyResult, register) -> float | None:
    """The spread of the idea family's variant Sharpes, one per variant, for G1's best-of-N hurdle (QA P1-T5)."""
    if register is None:
        return None
    sharpes = [x for x in register.sharpes(_idea(r)) if x is not None and math.isfinite(x)]
    return float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else None


def _breakeven_words(r: StudyResult) -> str:
    from sleeve_fund.research.study import breakeven_fee

    if not r.cost_ladder:
        return "not tested: no cost ladder was run"
    return getattr(r, "breakeven", "") or breakeven_fee(r.cost_ladder)[1]


def _breakeven_cell(row) -> str:
    """A grid point's break-even fee per side, or why there is none, in a table cell."""
    fee = row.get("breakeven_fee")
    if fee is not None and not (isinstance(fee, float) and math.isnan(fee)):
        return f"{fee:.3%}"
    words = row.get("breakeven")
    if not isinstance(words, str):
        return "–"
    if words.startswith("made no trades"):
        return "no trades"
    return "loses at no fee" if words.startswith("loses") else "above the ladder" if words.startswith("still") else "–"


def _nearby(r: StudyResult) -> tuple[str, str]:
    """Holds at nearby settings, around what tuning chose, never the defaults (Independent Quant Advisor,
    6 Oct 2026, P1-G2): each walk-forward fold's choice on that fold's training grid, reporting the worst
    fold, and the whole period's choice on the full grid. Any failure fails the check."""
    params = [c for c in r.spec.param_grid if c in r.sensitivity.columns]
    chosen = getattr(r, "chosen_params", None) or r.default_params
    final = nearby_scored(r.sensitivity, chosen, params)
    words = f"whole period's choice: {final[1]}"
    unscored = ("FAIL", "no setting scored a Sharpe on this fold's training stretch (none traded), so nothing was "
                        "chosen", -math.inf)
    folds = [(f, unscored if getattr(f, "unscored", False) else nearby_scored(f.grid, f.chosen, params))
             for f in r.folds if isinstance(getattr(f, "grid", None), pd.DataFrame) and not f.grid.empty]
    if not folds:
        return final[0], words
    failed = [x for x in folds if x[1][0] == "FAIL"]
    f, (_, worst, _) = min(folds, key=lambda x: x[1][2])
    words += (f"; {len(folds) - len(failed)} of {len(folds)} folds' choices hold; worst, the fold testing to "
              f"{f.test_end:%b %Y} ({json.dumps(f.chosen)}): {worst}")
    return ("FAIL" if final[0] == "FAIL" or failed else "PASS"), words


def g1_checks(r: StudyResult, ledger: IdeaLedger, register=None) -> list[tuple[str, str, str]]:
    """register: the trials register, when the study ran against the database. Its count of variants, which
    includes single backtests and paper strategies (QA P1-T1), then sets the bar instead of the idea counter's."""
    oos = summary(r.oos_returns)
    bench = summary(r.oos_benchmark_returns)
    bench_sharpe_full = summary(daily_returns(r.full_period_benchmark.equity))["sharpe"]
    share_beating = float((r.sensitivity["sharpe"] > bench_sharpe_full).mean()) if len(r.sensitivity) else 0.0
    trips = r.oos_trades
    counts = _counts(r, ledger, register)
    beats, hurdle = sharpe_beats_probability(r.oos_returns, r.oos_benchmark_returns, counts["variants"],
                                             trial_spread=_trial_spread(r, register))
    unjudged = math.isnan(beats)
    if unjudged:
        beats = 0.0  # too short, or too few independent days, to judge
    why_not = r.not_judged
    checks = [
        (JUDGED_CHECK, NOT_JUDGED if why_not else "PASS",
         f"Not judged: {why_not}" if why_not else "no strategy errors, and out-of-sample traded"),
        (
            "Out-of-sample return vs benchmark after fees (shown, not the G1 test)",
            "INFO",
            f"CAGR {_pct(oos['cagr'])} vs {_pct(bench['cagr'])}",
        ),
        (
            SHARPE_CHECK,
            "PASS" if oos["sharpe"] > bench["sharpe"] and beats >= G1_CONFIDENCE else "FAIL",
            f"Sharpe {_num(oos['sharpe'])} vs {_num(bench['sharpe'])}; " + (
                "too few independent out-of-sample days to judge" if unjudged else
                f"{_share(beats)} likely to beat it by more than the best of {counts['variants']} variants would by "
                f"luck ({_num(hurdle)}); bar: {G1_CONFIDENCE:.0%}") + (f" {N_UNCERTAIN}" if counts.get("n_uncertain") else ""),
        ),
        _random_entry_check(r),
        (RANDOM_SIDE_CHECK, *((NOT_APPLICABLE, "long only: there is no side to draw") if r.random_side is None else
                              (NOT_APPLICABLE if r.random_side.verdict == "N/A" else r.random_side.verdict,
                               r.random_side.words))),
        (
            "Holds up when parameters move",
            "PASS" if share_beating >= ROBUST_SHARE else "FAIL",
            f"{share_beating:.0%} of {len(r.sensitivity)} grid points beat the benchmark Sharpe (bar: {ROBUST_SHARE:.0%})",
        ),
        # v2 P1-6: the whole grid can beat the benchmark while the chosen value sits on a peak; this asks
        # whether the settings right next to it still work. In-sample, so it counts under NOT JUDGED too.
        (NEARBY_CHECK, *_nearby(r)),
        ("Break-even fee (shown, not a test)", "INFO", _breakeven_words(r)),
        *_funding_rows(r),
        (
            "Holdout not used for tuning",
            "PASS",
            (f"opened once, for this read-out; holdout not judged: {r.holdout_not_judged}" if r.holdout_not_judged
             else "opened once, for this read-out") if r.holdout else r.holdout_withheld or "untouched",
        ),
        (
            "Variants tried disclosed",
            "PASS",
            f"{counts['variants']} variants across {counts['ideas']} ideas so far",
        ),
        (
            # A Sharpe built on a handful of trades is luck, not evidence, so this one fails G1. It counts
            # the trades the out-of-sample Sharpe stands on, not the in-sample ones.
            "Enough out-of-sample trades to judge",
            "PASS" if trips >= MIN_OOS_TRADES else "FAIL",
            f"{trips} closed in the {len(r.folds)} walk-forward test windows, opened there too (bar: {MIN_OOS_TRADES}); "
            + _excluded_words(r) +
            f"{len(r.round_trips)} over the full research period, in-sample; turnover {r.turnover:.1f}x a year",
        ),
    ]
    if why_not:
        # Under a NOT JUDGED headline these can't fail: they measure the out-of-sample the study didn't get,
        # and red Fail chips there read as a verdict (review round 9, N6).
        checks = [(name, NOT_APPLICABLE if verdict == "FAIL" and name in OOS_CHECKS else verdict,
                   f"not judged: {ev}" if verdict == "FAIL" and name in OOS_CHECKS else ev)
                  for name, verdict, ev in checks]
    return checks


def _funding_rows(r: StudyResult) -> list[tuple[str, str, str]]:
    """A perpetual's funding: the G1 check on out-of-sample, and, shown only, the opened holdout (not judged past the
    same limits) and the full research period."""
    rows = []
    if r.funding_baseline[1]:
        rows.append((FUNDING_CHECK, *r.funding_check))
    if r.holdout and r.holdout_funding_baseline[1]:
        verdict, words = funding.baseline_check(*r.holdout_funding_baseline, "holdout")
        rows.append(("Holdout funding at the venue's own rates (shown, not a test)", "INFO",
                     f"{'holdout not judged: ' if verdict == NOT_JUDGED else ''}{words}"))
    full = r.full_period
    if full.funding_held:
        rows.append(("Full-period funding at the venue's own rates (shown, not a test)", "INFO",
                     funding.baseline_check(full.funding_at_baseline, full.funding_held,
                                            full.funding_baseline_longest, "full-period")[1]))
    return rows


def _random_entry_check(r: StudyResult) -> tuple[str, str, str]:
    """C3: a weak benchmark is not judged, neither a pass nor a fail, and says why (Advisor, 19:19). The Sharpe
    test against buy-and-hold still has to pass on its own."""
    re_ = r.random_entry
    if re_ is None or re_.verdict == "N/A":
        return RANDOM_ENTRY_CHECK, NOT_APPLICABLE, "no out-of-sample trades to compare"
    if re_.verdict == "WEAK":
        return RANDOM_ENTRY_CHECK, NOT_APPLICABLE, f"weak, not judged (in market {re_.exposure:.0%}): {re_.words}"
    return RANDOM_ENTRY_CHECK, re_.verdict, re_.words


def _excluded_words(r: StudyResult) -> str:
    excluded = r.excluded_trades
    if not excluded:
        return ""
    share = excluded / (excluded + r.oos_trades)
    flag = f"; flagged: more than {EXCLUDED_FLAG:.0%}" if share > EXCLUDED_FLAG else ""
    return f"{excluded} left out at the windows' edges ({share:.0%}{flag}); "


def oos_gaps(r: StudyResult) -> str:
    """Why out-of-sample has test windows without a trade, in words, or '' when every window traded.
    A halted fold reads +0.0% with a Sharpe of 0.00, which looks like a result and isn't one. Halts
    before a test window and inside one are told apart: only the first leaves the whole window flat,
    so the counts can't seem to disagree (review round 9, N6)."""
    n = len(r.folds)
    idle = [f for f in r.folds if f.closed_in_window == 0]
    halted = [f for f in r.folds if f.halted]
    if not idle and not halted:
        return ""
    words = []
    if idle:
        words.append(f"**No trades out-of-sample in {len(idle)} of {n} test windows.**")
    if halted:
        who = f"the {r.risk_profile} risk profile" if r.risk_profile else "the risk guard"
        before = [f for f in halted if f.halted_before_test]
        inside = [f for f in halted if not f.halted_before_test]
        split = []
        if before:
            split.append(f"{len(before)} in the training stretch, so {'that' if len(before) == 1 else 'each'} "
                         "test window sat flat at +0.0% throughout")
        if inside:
            traded = sum(1 for f in inside if f.closed_in_window)
            split.append(f"{len(inside)} inside the test window, flat from then on"
                         + (f" ({traded} of them closed a trade first)" if traded else ""))
        words.append(
            f"{who[0].upper()}{who[1:]} halted the strategy in {len(halted)} of {n} folds: {' and '.join(split)}. "
            + "; ".join(f"The fold testing to {f.test_end:%b %Y} halted on {f.halted}" for f in halted)
            + ". A halted run stays flat, as paper does until you resume it. Each fold's run trades through its "
            "training stretch first, so the position carried into the test is realistic.")
    quiet = [f for f in idle if not f.halted]
    if quiet:
        words.append(f"In {len(quiet)} of the windows without a trade no halt was involved: the signal never "
                     "closed a trade there.")
    return " ".join(words)


def _halted_on(halted: str) -> str:
    """' (halted 12 Mar 2026)' from a fold's halt words, for the folds table; '' when it wasn't."""
    return f" (halted {halted.split(' (')[0]})" if halted else ""


def _n(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _span(minutes: int) -> str:
    return f"{minutes / 1440:g} days" if minutes >= 1440 and minutes % 1440 == 0 else (
        f"{minutes / 60:g} hours" if minutes >= 60 else f"{minutes} minutes")


def render(r: StudyResult, ledger: IdeaLedger, register=None) -> str:
    spec = r.spec
    oos = summary(r.oos_returns)
    oos_b = summary(r.oos_benchmark_returns)
    full = summary(daily_returns(r.full_period.equity))
    full_b = summary(daily_returns(r.full_period_benchmark.equity))
    counts = _counts(r, ledger, register)
    project = ({**register.counts(), "ideas_by_family": register.ideas_by_family()} if register is not None else
               counts)
    dsr = (register.deflated_sharpe(r.oos_returns, _idea(r)) if register is not None else
           deflated_sharpe_probability(r.oos_returns, counts["variants"], ledger.sharpes()))
    years_full = years_covered(r.full_period.equity)
    fee_drag = r.full_period.fees_paid / r.full_period.equity.mean() / years_full
    ts = r.trade_stats

    out: list[str] = []
    out.append(f"# Tear sheet: {spec.name}")
    out.append("")
    if r.synthetic:
        out.append("> **Synthetic data.** This run only proves the pipeline works. The numbers say nothing about the strategy.")
        out.append("")
    out.append(
        f"Tested on `{r.instrument}` at {r.bar_minutes}-minute bars" + (f" on `{r.venue}`" if r.venue else "")
    )
    out.append("")
    if r.settings:
        out.append(f"Settings: {r.settings}")
        out.append("")
    out.append(f"G1 rules: {G1_RULES}")
    out.append("")
    out.append(
        f"Dataset `{r.dataset}` · research period {r.research_start:%d %b %Y} to {r.research_end:%d %b %Y} · "
        f"holdout: last {r.holdout_days} days {'(opened)' if r.holdout else '(untouched)'} · "
        f"fees: {r.fee_note}"
    )
    out.append("")
    if r.holdout:  # read by the pipeline: a holdout not judged blocks promotion, not G1 (pipeline.promotable)
        out.append(f"Holdout: NOT JUDGED ({r.holdout_not_judged})" if r.holdout_not_judged else "Holdout: judged")
        out.append("")
    out.append("## Idea")
    out.append("")
    out.append(f"**Family:** {spec.family} · **Asset class:** {spec.asset_class} · **Benchmark:** {spec.benchmark} · "
               f"**Default risk profile:** {spec.default_risk_profile}")
    out.append("")
    out.append(f"**In plain English:** {spec.idea}")
    out.append("")
    out.append(f"**What the code does:** {spec.rules}")
    out.append("")
    if r.bar_minutes < 1440:
        out.append(f"*Windows above count bars. These are {r.bar_minutes}-minute bars, so a \"50-day\" average here is "
                   f"50 bars, {_span(50 * r.bar_minutes)}.*")
        out.append("")
    out.append(f"**Known weaknesses:** {spec.known_weaknesses or 'none recorded'}")
    out.append("")
    out.append("## G1 checks")
    out.append("")
    checks = g1_checks(r, ledger, register)
    verdict, failed = g1_verdict(checks)
    if verdict == NOT_JUDGED:
        out.append(f"**G1: {verdict}** ({r.not_judged}; this is neither a pass nor a fail"
                   + (", and the holdout stays unspent)" if r.holdout_days else ")"))
    else:
        out.append(f"**G1: {verdict}**" + (f" (failed: {', '.join(failed)})" if failed else " (every check passed)"))
    out.append("")
    out.append("| Check | Result | Evidence |")
    out.append("| --- | --- | --- |")
    for name, result, evidence in checks:
        out.append(f"| {name} | {result} | {evidence} |")
    out.append("")
    gaps = oos_gaps(r)
    if gaps:
        out.append(f"> {gaps}")
        out.append("")
    out.append("The PM's G1 rule (3 Oct 2026) is a higher out-of-sample Sharpe than buy-and-hold after fees. The build "
               f"applies it strictly: by more than luck across every variant tried, at {G1_CONFIDENCE:.0%} confidence "
               "(paired block bootstrap, blocks as long as the returns' persistence), with enough trades, an unused "
               "holdout and robustness to parameter moves. The confidence bar is the build's, pending the PM's choice. "
               f"Variants are counted from `{ledger.path.name}`, the idea ledger kept with the research data.")
    out.append("")
    out.append("## Results after fees")
    out.append("")
    out.append("| Period | Strategy CAGR | Benchmark CAGR | Strategy Sharpe | Benchmark Sharpe | Strategy max DD | Benchmark max DD |")
    out.append("| --- | --- | --- | --- | --- | --- | --- |")
    out.append(_row(f"Walk-forward out-of-sample ({len(r.folds)} folds, {oos['days']} days)", oos, oos_b))
    out.append(_row(f"Full research period, params {json.dumps(r.default_params)} (in-sample)", full, full_b))
    if r.holdout:
        out.append(_row("Holdout (not judged)" if r.holdout_not_judged else "Holdout", r.holdout, r.holdout_benchmark))
    out.append("")
    out.append(f"Out-of-sample Sortino {_num(oos['sortino'])} vs {_num(oos_b['sortino'])}; "
               f"Calmar {_num(oos['calmar'])} vs {_num(oos_b['calmar'])}; "
               f"volatility {oos['volatility']:.0%} vs {oos_b['volatility']:.0%}.")
    out.append("")
    out.append("## Trading and costs (full research period, default params)")
    out.append("")
    out.append("| Closed trades | Win rate | Net P&L | Avg win | Avg loss | Expectancy per trade | Profit factor | Best | Worst |")
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    out.append(f"| {ts['trades']} ({ts['wins']} won, {ts['losses']} lost) | {_share(ts['win_rate'])} | {ts['pnl']:,.0f} "
               f"| {_pct(ts['avg_win'])} | {_pct(ts['avg_loss'])} | {_pct(ts['expectancy'])} | {_num(ts['profit_factor'])} "
               f"| {_pct(ts['best'])} | {_pct(ts['worst'])} |")
    out.append("")
    out.append("All trade figures are after fees. Expectancy is the average return per closed trade.")
    out.append("")
    out.append(f"- Turnover: {r.turnover:.1f}x average equity a year")
    out.append(f"- Fees paid: {r.full_period.fees_paid:,.0f} on {r.full_period.starting_capital:,.0f} starting capital; "
               f"fee drag {fee_drag:.2%} of average equity a year")
    out.append(f"- Time in the market: {r.full_period.exposure.gt(0.001).mean():.0%} of bars hold a position")
    out.append("")
    if r.cost_ladder:
        chosen = getattr(r, "chosen_params", None) or r.default_params
        out.append(f"## Cost ladder (full research period, chosen settings {json.dumps(chosen)})")
        out.append("")
        out.append(f"**Break-even fee:** {_breakeven_words(r)}.")
        out.append("")
        out.append("| Fee per side | Total return | Sharpe | Round trips | Fees paid |")
        out.append("| --- | --- | --- | --- | --- |")
        for rung in r.cost_ladder:
            out.append(f"| {rung.fee:.2%} | {_pct(rung.total_return)} | {_num(rung.sharpe)} | {rung.round_trips} "
                       f"| {rung.fees_paid:,.0f} |")
        out.append("")
        out.append("The settings tuning on the whole research period picks (best in-sample Sharpe on the grid), at "
                   "each fee, charged per side on maker and taker fills alike, so for a post-only strategy it mixes "
                   "the two. Every rung also pays half the "
                   f"bid-ask spread as above plus {r.ladder_slippage:.2%} slippage on orders that take liquidity. 0.02% and "
                   "0.05% are a low-fee perpetual venue's maker and taker rates, 0.10-0.40% typical spot taker rates, "
                   "0.80% a high-fee spot venue's taker rate. The break-even interpolates log(1 + return) between "
                   "rungs, then re-runs at that fee to verify it.")
        out.append("")
    out.append("## Walk-forward folds")
    out.append("")
    out.append("| Train | Test to | Chosen params | Train Sharpe | Test trades | Test CAGR | Benchmark CAGR | Test Sharpe | Benchmark Sharpe |")
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for f in r.folds:
        out.append(
            f"| {f.train_start:%b %Y} to {f.train_end:%b %Y} | {f.test_end:%b %Y} | {json.dumps(f.chosen)} "
            f"| {_num(f.train_sharpe)} | {f.closed_in_window}{_halted_on(f.halted)} | {_pct(f.test['cagr'])} "
            f"| {_pct(f.benchmark_test['cagr'])} | {_num(f.test['sharpe'])} | {_num(f.benchmark_test['sharpe'])} |"
        )
    out.append("")
    out.append("## Parameter sensitivity (full research period, in-sample)")
    out.append("")
    cols = [c for c in r.sensitivity.columns if c in spec.param_grid]
    out.append("| " + " | ".join(cols) + " | CAGR | Sharpe | Max DD | Round trips | Break-even fee per side, on top of spread and slippage |")
    out.append("| " + " | ".join("---" for _ in cols) + " | --- | --- | --- | --- | --- |")
    for _, row in r.sensitivity.iterrows():
        out.append(
            "| " + " | ".join(_param(row[c]) for c in cols)
            + f" | {_pct(row['cagr'])} | {_num(row['sharpe'])} | {_pct(row['max_drawdown'])} | {int(row['round_trips'])} "
            f"| {_breakeven_cell(row)} |"
        )
    out.append(f"\nBenchmark over the same period: CAGR {_pct(full_b['cagr'])}, Sharpe {_num(full_b['sharpe'])}. "
               "Break-even fees in this table are interpolated between the cost ladder's rungs; the chosen "
               "settings' is re-run to verify it, as the cost ladder says.")
    out.append("")
    out.append("## Idea counter")
    out.append("")
    uncertain = f" {N_UNCERTAIN}" if counts.get("n_uncertain") else ""
    if register is not None:
        out.append(f"- {_n(counts['variants'], 'distinct variant')} of this idea tried so far, in studies, backtests and "
                   f"paper strategies ({counts['evaluations']} evaluations including walk-forward refits): the N its "
                   f"results are judged by.{uncertain}")
    out.append(f"- {_n(project['ideas'], 'idea')} and {_n(project['variants'], 'distinct variant')} tested so far "
               f"across the project ({project['evaluations']} evaluations), shown for awareness. By family: "
               + ", ".join(f"{k} {v}" for k, v in project["ideas_by_family"].items()))
    if math.isnan(dsr):
        out.append("- Deflated Sharpe: can't be computed here: out-of-sample needs at least 30 days whose returns "
                   "vary, and " + ("these test windows never traded." if r.oos_trades == 0 else
                                   f"this one has {oos['days']} days.") + uncertain)
    else:
        out.append(f"- Deflated Sharpe: {_share(dsr)} probability the out-of-sample Sharpe "
                   f"is real rather than the best of many tries (higher is better; 95% is a strong bar).{uncertain}")
    out.append("")
    out.append("## Caveats")
    out.append("")
    out.append("- Signal orders go to the venue as the bar that triggered them closes: at market, paying the taker fee "
               "and half the bid-ask spread, or post-only first when the strategy waits for a maker fill. Slippage "
               "beyond the spread is not modelled.")
    out.append("- Returns are in the quote currency, not pounds. GBP returns and UK capital gains tax are not included.")
    for note in r.notes:
        out.append(f"- {note}")
    out.append("")
    return "\n".join(out)
