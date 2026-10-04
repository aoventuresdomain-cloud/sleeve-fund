"""One-page tear sheet in Markdown, led by out-of-sample results after fees."""

from __future__ import annotations

import json
import math


from sleeve_fund.research.ledger import IdeaLedger
from sleeve_fund.research.metrics import (
    daily_returns,
    deflated_sharpe_probability,
    sharpe_beats_probability,
    summary,
    years_covered,
)
from sleeve_fund.research.study import StudyResult

MIN_ROUND_TRIPS = 10
ROBUST_SHARE = 0.6
# G1's bar: at least this probability that the out-of-sample Sharpe beats the benchmark's by more
# than the best of the variants tried would by luck.
G1_CONFIDENCE = 0.95
SHARPE_CHECK = "G1 test: out-of-sample Sharpe clearly beats benchmark after fees"


def g1_verdict(checks: list[tuple[str, str, str]]) -> tuple[str, list[str]]:
    """G1 passes only when every check that can fail passes: the Sharpe test, robustness, an unused
    holdout and enough trades. A strong Sharpe on three trades, or on a holdout already looked at,
    is not evidence. Returns the verdict and the checks that failed."""
    failed = [name for name, verdict, _ in checks if verdict not in ("PASS", "INFO")]
    return ("FAIL" if failed else "PASS"), failed


def _pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.1%}"


def _share(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.0%}"


def _num(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"


def _row(label: str, s: dict, b: dict) -> str:
    return (
        f"| {label} | {_pct(s['cagr'])} | {_pct(b['cagr'])} | {_num(s['sharpe'])} | {_num(b['sharpe'])} "
        f"| {_pct(s['max_drawdown'])} | {_pct(b['max_drawdown'])} |"
    )


def g1_checks(r: StudyResult, ledger: IdeaLedger) -> list[tuple[str, str, str]]:
    oos = summary(r.oos_returns)
    bench = summary(r.oos_benchmark_returns)
    bench_sharpe_full = summary(daily_returns(r.full_period_benchmark.equity))["sharpe"]
    share_beating = float((r.sensitivity["sharpe"] > bench_sharpe_full).mean()) if len(r.sensitivity) else 0.0
    trips = len(r.round_trips)
    counts = ledger.counts()
    beats, hurdle = sharpe_beats_probability(r.oos_returns, r.oos_benchmark_returns, counts["variants"])
    unjudged = math.isnan(beats)
    if unjudged:
        beats = 0.0  # too short, or too few independent days, to judge
    checks = [
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
                f"luck ({_num(hurdle)}); bar: {G1_CONFIDENCE:.0%}"),
        ),
        (
            "Holds up when parameters move",
            "PASS" if share_beating >= ROBUST_SHARE else "FAIL",
            f"{share_beating:.0%} of {len(r.sensitivity)} grid points beat the benchmark Sharpe (bar: {ROBUST_SHARE:.0%})",
        ),
        (
            "Holdout not used for tuning",
            "FAIL" if r.holdout_reused else "PASS",
            "holdout opened more than once" if r.holdout_reused else
            ("opened once, for this read-out" if r.holdout else "untouched"),
        ),
        (
            "Variants tried disclosed",
            "PASS",
            f"{counts['variants']} variants across {counts['ideas']} ideas so far",
        ),
        (
            # A Sharpe built on a handful of trades is luck, not evidence, so this one fails G1.
            "Enough trades to judge",
            "PASS" if trips >= MIN_ROUND_TRIPS else "FAIL",
            f"{trips} closed trades over the research period (bar: {MIN_ROUND_TRIPS}); turnover {r.turnover:.1f}x a year",
        ),
    ]
    return checks


def _n(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _span(minutes: int) -> str:
    return f"{minutes / 1440:g} days" if minutes >= 1440 and minutes % 1440 == 0 else (
        f"{minutes / 60:g} hours" if minutes >= 60 else f"{minutes} minutes")


def render(r: StudyResult, ledger: IdeaLedger) -> str:
    spec = r.spec
    oos = summary(r.oos_returns)
    oos_b = summary(r.oos_benchmark_returns)
    full = summary(daily_returns(r.full_period.equity))
    full_b = summary(daily_returns(r.full_period_benchmark.equity))
    counts = ledger.counts()
    trial_sharpes = ledger.sharpes()
    dsr = deflated_sharpe_probability(r.oos_returns, counts["variants"], trial_sharpes)
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
    out.append(
        f"Dataset `{r.dataset}` · research period {r.research_start:%d %b %Y} to {r.research_end:%d %b %Y} · "
        f"holdout: last {r.holdout_days} days {'(opened)' if r.holdout else '(untouched)'} · "
        f"fees: {r.fee_note}"
    )
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
    checks = g1_checks(r, ledger)
    verdict, failed = g1_verdict(checks)
    out.append(f"**G1: {verdict}**" + (f" (failed: {', '.join(failed)})" if failed else " (every check passed)"))
    out.append("")
    out.append("| Check | Result | Evidence |")
    out.append("| --- | --- | --- |")
    for name, result, evidence in checks:
        out.append(f"| {name} | {result} | {evidence} |")
    out.append("")
    out.append("The PM's G1 rule (3 Oct 2026) is a higher out-of-sample Sharpe than buy-and-hold after fees. The build "
               f"applies it strictly: by more than luck across every variant tried, at {G1_CONFIDENCE:.0%} confidence "
               "(paired block bootstrap, blocks as long as the returns' persistence), with enough trades, an unused "
               "holdout and robustness to parameter moves. The confidence bar is the build's, pending the PM's choice. "
               f"Variants are counted from `{ledger.path.name}`, which is kept in the repository.")
    out.append("")
    out.append("## Results after fees")
    out.append("")
    out.append("| Period | Strategy CAGR | Benchmark CAGR | Strategy Sharpe | Benchmark Sharpe | Strategy max DD | Benchmark max DD |")
    out.append("| --- | --- | --- | --- | --- | --- | --- |")
    out.append(_row(f"Walk-forward out-of-sample ({len(r.folds)} folds, {oos['days']} days)", oos, oos_b))
    out.append(_row(f"Full research period, params {json.dumps(r.default_params)} (in-sample)", full, full_b))
    if r.holdout:
        out.append(_row("Holdout", r.holdout, r.holdout_benchmark))
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
    out.append("## Walk-forward folds")
    out.append("")
    out.append("| Train | Test to | Chosen params | Train Sharpe | Test CAGR | Benchmark CAGR | Test Sharpe | Benchmark Sharpe |")
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for f in r.folds:
        out.append(
            f"| {f.train_start:%b %Y} to {f.train_end:%b %Y} | {f.test_end:%b %Y} | {json.dumps(f.chosen)} "
            f"| {_num(f.train_sharpe)} | {_pct(f.test['cagr'])} | {_pct(f.benchmark_test['cagr'])} "
            f"| {_num(f.test['sharpe'])} | {_num(f.benchmark_test['sharpe'])} |"
        )
    out.append("")
    out.append("## Parameter sensitivity (full research period, in-sample)")
    out.append("")
    cols = [c for c in r.sensitivity.columns if c in spec.param_grid]
    out.append("| " + " | ".join(cols) + " | CAGR | Sharpe | Max DD | Round trips |")
    out.append("| " + " | ".join("---" for _ in cols) + " | --- | --- | --- | --- |")
    for _, row in r.sensitivity.iterrows():
        out.append(
            "| " + " | ".join(str(int(row[c])) for c in cols)
            + f" | {_pct(row['cagr'])} | {_num(row['sharpe'])} | {_pct(row['max_drawdown'])} | {int(row['round_trips'])} |"
        )
    out.append(f"\nBenchmark over the same period: CAGR {_pct(full_b['cagr'])}, Sharpe {_num(full_b['sharpe'])}.")
    out.append("")
    out.append("## Idea counter")
    out.append("")
    out.append(f"- {_n(counts['ideas'], 'idea')} and {_n(counts['variants'], 'distinct variant')} tested so far "
               f"({counts['evaluations']} evaluations including walk-forward refits). By family: "
               + ", ".join(f"{k} {v}" for k, v in counts["ideas_by_family"].items()))
    out.append(f"- Deflated Sharpe: {_share(dsr)} probability the out-of-sample Sharpe "
               "is real rather than the best of many tries (higher is better; 95% is a strong bar).")
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
