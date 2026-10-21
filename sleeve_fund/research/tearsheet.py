"""One-page tear sheet in Markdown, led by out-of-sample results after fees."""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pandas as pd

from sleeve_fund import funding, markets
from sleeve_fund.research import analysis_pack, mintrl, periods, regime_split
from sleeve_fund.research.guardrails import CANDIDATE_CHECK, G1_RULES, MIN_OOS_TRADES, nearby_scored
from sleeve_fund.research.ledger import IdeaLedger
from sleeve_fund.research.metrics import (
    daily_returns,
    deflated_sharpe_probability,
    sharpe_beats_probability,
    summary,
    years_covered,
)
from sleeve_fund.research.pbo import PBO_CHECK, g1_row
from sleeve_fund.research.random_entry import CLOSE_PRICED
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
# Rows whose NOT JUDGED never passes G1 but never hides another check's fail (R4).
LAST_TIER = (PBO_CHECK, CANDIDATE_CHECK)
NEARBY_CHECK = "Holds at nearby settings"
RANDOM_ENTRY_CHECK = "Beats random entry times"
RANDOM_SIDE_CHECK = "Beats random long or short"
# A perpetual's out-of-sample funding at the venue's own rates: WARN is shown, not a fail; past the limits the study
# isn't judged (funding.baseline_check; Advisor, 6 Oct 2026, QA P1-O17).
FUNDING_CHECK = "Funding charged at the venue's own rates"
LIQUIDATION_CHECK = "Liquidations"
ABSORB_CHECK = "Gap absorbed by the venue past liquidation (shown, not a test)"
ABSORB_FAIL_CHECK = "Passes with liquidation gaps charged"
ABSORB_FAIL = "passes only because the venue absorbs liquidation gaps"
# A G1 check that a person's acknowledgement can clear: shown, and it stops a pass until acknowledged.
NEEDS_ACK = "ACK"
OOS_CHECKS = (SHARPE_CHECK, "Holds up when parameters move", "Enough out-of-sample trades to judge",
              RANDOM_ENTRY_CHECK, RANDOM_SIDE_CHECK)
# More of the test windows' trades than this left out at the edges, and the trade count is flagged.
EXCLUDED_FLAG = 0.10


def g1_verdict(checks: list[tuple[str, str, str]]) -> tuple[str, list[str]]:
    """G1 passes only when every check that can fail passes: the Sharpe test, robustness, an unused
    holdout and enough trades. A strong Sharpe on three trades, or on a holdout already looked at,
    is not evidence. A study whose runs raised errors, or whose out-of-sample the risk guard left flat,
    is not judged at all: neither a pass nor a fail. A PBO that couldn't be judged (too few variants, a window too
    short) never counts as a pass, but nor does it hide another check's fail (Advisor 13 Oct 16:50, R4); nor does a
    benchmark's (18:05: never a G1 candidate). Returns the
    verdict and the checks that failed, or that weren't judged."""
    unjudged = [name for name, verdict, _ in checks if verdict == NOT_JUDGED and name not in LAST_TIER]
    if unjudged:
        return NOT_JUDGED, unjudged
    failed = [name for name, verdict, _ in checks
              if verdict not in ("PASS", "WARN", "INFO", NOT_APPLICABLE, NOT_JUDGED)]
    if failed:
        return "FAIL", failed
    pbo_unjudged = [name for name, verdict, _ in checks if verdict == NOT_JUDGED]
    return (NOT_JUDGED, pbo_unjudged) if pbo_unjudged else ("PASS", [])


def _pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.1%}"


def _share(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.0%}"


def _param(v) -> str:
    """A grid value as set: 20, 0.005 or "ema", not int() of it, which crashed on a word and showed a
    0.5% threshold as 0 (round 11 minor)."""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):  # a setting with parts, e.g. signal_strength: "linear, floor 0.3" (G5)
        return ", ".join(str(x) if k == "mapping" else f"{k} {_param(x)}" for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return "/".join(_param(x) for x in v)
    f = float(v)
    return str(int(f)) if f.is_integer() else f"{f:g}"


def _num(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"


# A benchmark with no days to summarise: a perp hold whose funding isn't known over the window (RE-COST).
_NO_BENCHMARK = {"cagr": math.nan, "sharpe": math.nan, "max_drawdown": math.nan, "sortino": math.nan,
                 "calmar": math.nan, "volatility": math.nan, "days": 0}


def _bench(returns) -> dict:
    return summary(returns) if returns is not None and len(returns.dropna()) >= 2 else _NO_BENCHMARK


def _row(label: str, s: dict, b: dict | None) -> str:
    b = b or _NO_BENCHMARK
    return (
        f"| {label} | {_pct(s['cagr'])} | {_pct(b['cagr'])} | {_num(s['sharpe'])} | {_num(b['sharpe'])} "
        f"| {_pct(s['max_drawdown'])} | {_pct(b['max_drawdown'])} |"
    )


def _idea(r: StudyResult) -> str:
    from sleeve_fund.research.trials import legacy_idea_hash

    return getattr(r, "idea_hash", None) or legacy_idea_hash(r.spec.name)


def _period_label(p) -> str:
    return f"{p.label} ({p.partial})" if p.partial else p.label


def _period_tables(r: StudyResult) -> list[str]:
    """R-13: the scored window's returns by year and by month, the Strategy's against buy-and-hold's."""
    eq, hold = periods.scored(r)
    trips = [vars(t) for t in periods.trade_rows(r.full_period) if t.closed is not None]
    years, hold_years = periods.periods(eq, "Y", trips), {p.label: p for p in periods.periods(hold, "Y")}
    out = ["## Returns by year", "",
           f"Over the scored window ({_scored_words(eq)}), after fees and funding; a partial year says which part, "
           f"and its Sharpe is n/a under {periods.MIN_SHARPE_DAYS} days. Trades are counted in the year they closed.", "",
           "| Year | Strategy | Buy and hold | Difference | Strategy Sharpe | Buy-and-hold Sharpe | Strategy max DD "
           "| Buy-and-hold max DD | Trades |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for p in years:
        b = hold_years.get(p.label)
        out.append(f"| {_period_label(p)} | {_pct(p.ret)} | {_pct(b.ret if b else None)} "
                   f"| {_pct(p.ret - b.ret if b else None)} | {_num(p.sharpe)} | {_num(b.sharpe if b else None)} "
                   f"| {_pct(p.max_dd)} | {_pct(b.max_dd if b else None)} | {p.trades} |")
    out += ["", "## Returns by month", "",
            "Each month end on end, so the months compound to the window's total return; p marks a partial month.", ""]
    months, hold_months = periods.periods(eq, "M"), {p.label: p for p in periods.periods(hold, "M")}
    by = {p.label: p for p in months}
    hold_by_year = hold_years
    for title, cell, year_cell in (
            ("Strategy", lambda m: m and m.ret, lambda y: y.ret),
            ("Buy and hold", lambda m: hold_months.get(m.label).ret if m and m.label in hold_months else None,
             lambda y: hold_by_year[y.label].ret if y.label in hold_by_year else None),
            ("Difference", lambda m: m.ret - hold_months[m.label].ret if m and m.label in hold_months else None,
             lambda y: y.ret - hold_by_year[y.label].ret if y.label in hold_by_year else None)):
        out += [f"**{title}**", "", "| Year | " + " | ".join(periods.MONTHS) + " | Year |",
                "| --- |" + " --- |" * 13]
        for y in years:
            cells = []
            for i in range(1, 13):
                m = by.get(f"{y.label}-{i:02d}")
                cells.append("" if m is None else _pct(cell(m)) + (" p" if m.partial else ""))
            out.append(f"| {y.label} | " + " | ".join(cells) + f" | {_pct(year_cell(y))}{' p' if y.partial else ''} |")
        out.append("")
    return out


def _scored_words(eq) -> str:
    first, last = periods.span(eq)
    return f"the full research period, the closes of {first:%d %b %Y} to {last:%d %b %Y}"


def _every_trade(r: StudyResult, files: bool, shown: int) -> list[str]:
    """R-15: every trade of the scored window with the equity after it; the sheet lists the first `shown`, the CSV
    beside it all of them, and says so."""
    rows = periods.trade_rows(r.full_period)
    eq, _ = periods.scored(r)
    out = ["## Every trade", "",
           f"Starting capital: {float(eq.iloc[0]):,.0f}. Every trade of the scored window ({_scored_words(eq)}), after "
           "fees and funding, with the equity after it and its drawdown at the trade's close from the equity's peak so far "
           "(marked to market daily; the headline max drawdown, from the whole daily curve, can be deeper, since a "
           "trough inside a trade shows here only if it lasts to the close); funding and any insurance-fund cover "
           "are booked to the trade they fell in; P&L % is of the equity before the trade. Sizes round to the venue's lot, so a smaller "
           "account's sizes and returns can differ slightly from a larger one's. A position still open at the end is "
           "the last row, marked to market there, so the P&L column sums to the end equity less the start.", ""]
    if files:
        out += ["Beside this sheet: the [trade list (CSV)](beside:trades.csv), the [daily equity (CSV)](beside:equity.csv) "
                "and the [equity curve against buy-and-hold (SVG)](beside:equity.svg).", ""]
    if not rows:
        return out + ["No trades.", ""]
    out += ["| # | Side | Entry | Entry price | Exit | Exit price | Size | Notional | Held | Fees | Funding | P&L "
            "| P&L % | Exit reason | Equity after | Drawdown at trade close |",
            "| --- |" + " --- |" * 15]
    for t in rows[:shown]:
        out.append(f"| {t.n} | {t.side} | {t.opened:%d %b %Y %H:%M} | {t.entry_px:,.6g} "
                   f"| {'open' if t.closed is None else f'{t.closed:%d %b %Y %H:%M}'} "
                   f"| {'' if t.exit_px is None else f'{t.exit_px:,.6g}'} | {t.qty:,.6g} | {t.notional:,.2f} "
                   f"| {t.held / pd.Timedelta(days=1):.1f} days | {t.fees:,.2f} | {t.funding:,.2f} | {t.pnl:,.2f} "
                   f"| {_pct(t.pnl_pct)} | {t.reason} | {t.equity_after:,.2f} | {_pct(t.drawdown)} |")
    if len(rows) > shown:
        where = "the [trade list (CSV)](beside:trades.csv)" if files else "the trade list beside this sheet"
        out += ["", f"{shown} of {len(rows)} trades shown; every trade is in {where}."]
    return out + [""]


def _asset_class(r: StudyResult, spec) -> str:
    """R-04: what the study's runs traded, not the strategy's default: a perpetual when they traded one."""
    return "perpetual" if markets.is_perp(r.full_period.params) else spec.asset_class


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


def _double_fees_line(r: StudyResult) -> str:
    """B-22 (Advisor B22-DOUBLE-FEES), in the line the page reads (development.double_fees): the out-of-sample
    windows re-run at twice the venue's fees, passed when still above 0 net."""
    from sleeve_fund.instruments import pct

    d = getattr(r, "double_fees", None)
    if d is None:
        return "**Profitable at double fees:** not tested: no out-of-sample window chose settings to re-run"
    return (f"**Profitable at double fees:** {'PASS' if d['passed'] else 'FAIL'}, net {d['net_return']:+.1%} over the "
            f"out-of-sample period at 2x the venue's fees ({pct(d['maker'])} maker, {pct(d['taker'])} taker; "
            f"{d['oos_days']} days, {d['round_trips']} round trips; at the venue's fees {d['base_net_return']:+.1%})")


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


def g1_checks(r: StudyResult, ledger: IdeaLedger, register=None,
              liquidation_ack: str | dict | None = None) -> list[tuple[str, str, str]]:
    """register: the trials register, when the study ran against the database. Its count of variants, which
    includes single backtests and paper strategies (QA P1-T1), then sets the bar instead of the idea counter's.
    liquidation_ack: the PM's acknowledgement of the study's liquidations after gaps past correctly placed stops: a
    note for all of them, or Store.g1_acks' rows by liquidation time. Never read for any other liquidation."""
    # On a perpetual the benchmark is the perp hold, priced only on the out-of-sample days its funding is known for,
    # and the strategy is compared on those same days (Advisor, 7 Oct 2026, RE-COST).
    insufficient = getattr(r, "hold_insufficient", False)
    oos = _bench(r.compare_returns)
    bench = _bench(r.oos_benchmark_returns)
    bench_sharpe_full = summary(daily_returns(r.full_period_benchmark.equity))["sharpe"]
    share_beating = float((r.sensitivity["sharpe"] > bench_sharpe_full).mean()) if len(r.sensitivity) else 0.0
    trips = r.oos_trades
    counts = _counts(r, ledger, register)
    beats, hurdle = (math.nan, math.nan) if insufficient else sharpe_beats_probability(
        r.compare_returns, r.oos_benchmark_returns, counts["variants"], trial_spread=_trial_spread(r, register))
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
        (SHARPE_CHECK, NOT_APPLICABLE, f"no verdict: {r.hold_note}") if insufficient else (
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
        ("Holds up when parameters move", NOT_APPLICABLE,
         "no verdict: the perpetual buy and hold isn't priced over the whole research period, since a funding "
         "settlement in it has no known rate") if not getattr(r, "hold_full_period", True) else (
            "Holds up when parameters move",
            "PASS" if share_beating >= ROBUST_SHARE else "FAIL",
            f"{share_beating:.0%} of {len(r.sensitivity)} grid points beat the benchmark Sharpe (bar: {ROBUST_SHARE:.0%})",
        ),
        # v2 P1-6: the whole grid can beat the benchmark while the chosen value sits on a peak; this asks
        # whether the settings right next to it still work. In-sample, so it counts under NOT JUDGED too.
        (NEARBY_CHECK, *_nearby(r)),
        # R4: how often the best in-sample configuration is no better than average out of sample. In-sample grid,
        # so it counts under NOT JUDGED too.
        g1_row(getattr(r, "pbo", None)),
        *([(CANDIDATE_CHECK, NOT_JUDGED, "a benchmark: buy and hold is never a G1 candidate")]
          if getattr(getattr(r, "spec", None), "family", None) == "benchmark" else []),
        ("Break-even fee (shown, not a test)", "INFO", _breakeven_words(r)),
        _stop_stress_row(r, counts, register, insufficient),
        mintrl.g1_row(r.compare_returns),  # R5: shown at G1, gates G2
        *([] if not getattr(r, "first_touch", None) else [(FIRST_TOUCH_CHECK, "INFO", first_touch_words(r))]),
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
        _liquidation_check(r, liquidation_ack),
    ]
    checks += _absorb_rows(r, checks, lambda view: g1_checks(view, ledger, register, liquidation_ack))
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


STOP_STRESS_CHECK = "Stop slippage stress (shown, not a test)"


def _stop_stress_row(r: StudyResult, counts: dict, register, insufficient: bool) -> tuple[str, str, str]:
    """D13-INTRAMINUTE-JUMP (3): out-of-sample with 20 bp more on each stop exit, a jump through the stop paper may
    see and a backtest's minutes can't; a Sharpe test that would fail under it is flagged to the PM. Required on the
    sheet when the entries' median stop sits under 0.5% of the price (GAP-DOWN, Advisor 15 Oct)."""
    st = getattr(r, "stop_stress", None) or {}
    median, stops = st.get("median_stop"), st.get("stops", 0)
    need = median is not None and median < st.get("required_below", 0.005)
    head = (f"required, median stop {median:.2%} is under {st.get('required_below', 0.005):.1%}: " if need else
            f"median stop {median:.2%}: " if median is not None else "")
    if not stops:
        return STOP_STRESS_CHECK, "INFO", head + "no stop exits out of sample, so nothing to stress"
    charge = st["charge"]
    base = r.compare_returns
    stressed = base - charge.reindex(base.index).fillna(0.0)
    s0, s1 = _bench(base)["sharpe"], _bench(stressed)["sharpe"]
    words = f"{stops} stop exits out of sample at +{st.get('bp', 20):.0f} bp each: Sharpe {_num(s0)} to {_num(s1)}"
    if not insufficient:
        bench = _bench(r.oos_benchmark_returns)["sharpe"]
        spread = _trial_spread(r, register)
        b0, _ = sharpe_beats_probability(base, r.oos_benchmark_returns, counts["variants"], trial_spread=spread)
        b1, _ = sharpe_beats_probability(stressed, r.oos_benchmark_returns, counts["variants"], trial_spread=spread)
        passes = lambda s, b: not math.isnan(b) and s > bench and b >= G1_CONFIDENCE  # noqa: E731
        if passes(s0, b0) and not passes(s1, b1):
            words += "; the Sharpe test would FAIL under it: flagged to the PM"
        elif passes(s0, b0):
            words += "; the Sharpe test still passes under it"
    return STOP_STRESS_CHECK, "INFO", head + words


def _absorb_rows(r: StudyResult, checks: list, judge) -> list[tuple[str, str, str]]:
    """ABSORB-GAP (INSURE-RULE point 5; Advisor 15 Oct ~02:05 UK): every test above read the curve with the gaps the
    venue absorbed past liquidation charged, nothing re-sized. This says how much, per window, beside out-of-sample
    and the holdout uncharged, for information only; and an idea that passes uncharged but fails charged FAILs, saying
    so. judge: g1_checks on a result, to judge the uncharged view. None when nothing was absorbed."""
    gaps = [(f"out-of-sample test window {k + 1}", f.absorbed) for k, f in enumerate(r.folds)
            if getattr(f, "absorbed", None)]
    if getattr(r, "holdout_absorbed", None):
        gaps.append(("holdout", r.holdout_absorbed))
    if not gaps:
        return []
    total, count = sum(a["amount"] for _, g in gaps for a in g), sum(len(g) for _, g in gaps)
    per = "; ".join(f"{w} {sum(a['amount'] for a in g):,.2f} over {_n(len(g), 'liquidation')}" for w, g in gaps)
    words = (f"{total:,.2f} over {_n(count, 'liquidation')} ({per}); every test here reads the curve gap charged, no "
             "re-sizing. Uncharged, for information only:")
    uncharged = getattr(r, "oos_uncharged_returns", None)
    if uncharged is not None:
        words += (f" out-of-sample Sharpe {_num(summary(uncharged)['sharpe'])} "
                  f"(charged {_num(summary(r.oos_returns)['sharpe'])})")
    if getattr(r, "holdout_uncharged", None) and r.holdout:
        words += (f"{',' if uncharged is not None else ''} holdout Sharpe {_num(r.holdout_uncharged['sharpe'])} "
                  f"(charged {_num(r.holdout['sharpe'])})")
    re_free = getattr(r, "random_entry_uncharged", None)
    if re_free is not None and r.random_entry is not None:
        words += (f"; random entry beats {re_free.return_percentile:.0f}% uncharged "
                  f"(charged {r.random_entry.return_percentile:.0f}%)")
    rows = [(ABSORB_CHECK, "INFO", words)]
    verdict, failed = g1_verdict(checks)
    if uncharged is None or verdict != "FAIL":
        return rows
    compare = r.oos_compare_returns
    view = dataclasses.replace(
        r, oos_returns=uncharged, oos_compare_returns=None if compare is None else uncharged.reindex(compare.index),
        folds=[dataclasses.replace(f, absorbed=[]) for f in r.folds], oos_uncharged_returns=None, holdout_absorbed=[],
        **{k: getattr(r, f"{k}_uncharged") for k in ("random_entry", "random_side")
           if getattr(r, f"{k}_uncharged", None) is not None})
    if g1_verdict(judge(view))[0] == "PASS":
        rows.append((ABSORB_FAIL_CHECK, "FAIL", f"{ABSORB_FAIL}: it passes G1 uncharged, and charged it fails "
                                                 + ", ".join(failed)))
    return rows


def _liquidation_check(r: StudyResult, ack: str | dict | None) -> tuple[str, str, str]:
    """GAP-LIQ-CAP (Independent Quant Advisor 6 Oct 23:42, 7 Oct 00:19 (5)): any liquidation in out-of-sample or
    the holdout is a G1 finding, whatever the P&L. A gap past a stop placed within half the distance to liquidation
    stops a pass until the PM acknowledges it; with no stop, or one beyond half way, it is a FAIL no acknowledgement
    clears."""
    liqs = getattr(r, "liquidations", None) or []
    if not liqs:
        return LIQUIDATION_CHECK, "PASS", "none in out-of-sample or the holdout"
    words, waiting = [], False
    for liq in liqs:
        lost = f"{liq['x']:,.2f} lost" if liq.get("x") is not None else "its margin lost"
        if not liq["stop_ok"]:
            # An acknowledgement is never read here: it records that the PM has seen it, never a pass (DA, HoE).
            why = ("no stop" if liq["stop_why"] == "missing" else f"its stop was {liq['stop_why']}") + \
                ": a G1 FAIL the PM's acknowledgement can't clear"
        elif (note := _ack_for(ack, liq)) is not None:
            why = f"a gap past a correctly placed stop at {liq['stop_px']:,.6g}, acknowledged by the PM: {note}"
        else:
            waiting = True
            why = f"a gap past a correctly placed stop at {liq['stop_px']:,.6g}: needs the PM's acknowledgement"
        words.append(f"liquidated in the {liq['window']} on {pd.Timestamp(liq['ts']):%d %b %Y} ({lost}); {why}")
    verdict = "FAIL" if any(not liq["stop_ok"] for liq in liqs) else NEEDS_ACK if waiting else "PASS"
    return LIQUIDATION_CHECK, verdict, "; ".join(words)


def _ack_for(ack, liq: dict) -> str | None:
    """The PM's acknowledgement of one liquidation: a note covering every one, or the store's acknowledgements by
    liquidation time (Store.g1_acks: {liquidation_ts: row}); None when it isn't acknowledged."""
    if ack is None or isinstance(ack, str):
        return ack or None
    row = next((v for ts, v in ack.items() if pd.Timestamp(ts) == pd.Timestamp(liq["ts"])), None)
    if row is None:
        return None
    return (row.get("note") if isinstance(row, dict) else None) or "acknowledged"


FIRST_TOUCH_CHECK = "First-touch candles resolved by rule (shown, not a test)"


def first_touch_words(r: StudyResult) -> str:
    """R2-G1: how many out-of-sample candles a first_touch rule settled by its fixed resolution, and over the
    threshold, both resolutions' results and which one G1 judged."""
    ft = r.first_touch
    words = (f"{ft['ambiguous']} of {ft['reached']} out-of-sample level-reaching candles ({ft['share']:.1%}) were "
             f"same-minute ambiguous (threshold {ft['threshold']:.0%} pooled or in any fold, or under "
             f"{ft['min_reached']} candles)")
    if not ft["flipped"]:
        return words + ": judged as ruled"
    ruled, opposite = ft["as_ruled"], ft["opposite"]
    words += (f": re-run the opposite way. As ruled ({ft['resolved']}): Sharpe {_num(ruled['sharpe'])}, CAGR "
              f"{_pct(ruled['cagr'])}; opposite ({ft['opposite_resolved']}): Sharpe {_num(opposite['sharpe'])}, CAGR "
              f"{_pct(opposite['cagr'])}. "
              f"Judged on the worse, {'the opposite' if ft['judged_on'] == 'opposite' else 'as ruled'}")
    if "holdout_judged_on" in ft:
        words += f"; the holdout on {'the opposite' if ft['holdout_judged_on'] == 'opposite' else 'as ruled'}"
    return words


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
            + ". A halted run stays flat, as paper does until you resume it. Each test window starts flat "
            "and trades from its own first bar, so no position or halt is carried in from training.")
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


# R-12 (Advisor SCORE-WINDOW, 21 Oct 2026): this round only, QR-B1, QR-B2 and QR-B3 must pass on both windows.
OLD_WINDOW_CHECK = "Also passes on the old three-year walk-forward (this round)"
SCREENING_DISCLOSURE = ("Disclosed: the earlier lab rounds (3 Oct 2026) looked at 2020 to 2022 when these ideas were "
                        "screened.")


def old_window_check(old: StudyResult, ledger: IdeaLedger, register=None) -> tuple[str, str, str]:
    """R-12: the old three-year walk-forward's G1 verdict on the same settings, as one more G1 check: the sheet passes
    only when the continuous window and this both pass."""
    verdict, failed = g1_verdict(g1_checks(old, ledger, register))
    why = f"failed: {', '.join(failed)}" if verdict == "FAIL" else (
        f"not judged: {', '.join(failed)}" if verdict == NOT_JUDGED else "every check passed")
    return OLD_WINDOW_CHECK, verdict, f"G1 {verdict} on the walk-forward ({old.settings.split(' · ')[-1]}): {why}"


def render(r: StudyResult, ledger: IdeaLedger, register=None, liquidation_ack: str | dict | None = None,
           flat: StudyResult | None = None, old: StudyResult | None = None, files: bool = False,
           shown_trades: int = 50) -> str:
    """flat: for a strength-sized variant (G5), its flat version's study on the same windows, shown side by side
    (strength_report); without it the flat version stays. old: R-12, the same settings' old three-year walk-forward
    study beside a continuous window's: one more G1 check, so the sheet passes only when both do. files: the sheet's
    trade list, daily equity and curve sit beside it (run.write_beside), linked as beside:<kind>, which the
    dashboard points at its own route for them, so the sheet itself names no run stamp and two runs of one study
    still render the same; shown_trades: the trades listed on the sheet itself (R-15)."""
    spec = r.spec
    oos = summary(r.oos_returns)
    oos_b = _bench(r.oos_benchmark_returns)
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
    out.append(f"**Family:** {spec.family} · **Asset class:** {_asset_class(r, spec)} · **Benchmark:** {spec.benchmark} · "
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
    checks = g1_checks(r, ledger, register, liquidation_ack)
    if old is not None:
        checks.append(old_window_check(old, ledger, register))
    verdict, failed = g1_verdict(checks)
    if verdict == NOT_JUDGED:
        why = r.not_judged or "; ".join(e for name, v, e in checks if name in failed and v == NOT_JUDGED)
        out.append(f"**G1: {verdict}** ({why}; this is neither a pass nor a fail"
                   + (", and the holdout stays unspent)" if r.holdout_days else ")"))
    else:
        out.append(f"**G1: {verdict}**" + (f" (failed: {', '.join(failed)})" if failed else " (every check passed)"))
    out.append("")
    out.append("| Check | Result | Evidence |")
    out.append("| --- | --- | --- |")
    for name, result, evidence in checks:
        out.append(f"| {name} | {result} | {evidence} |")
    out.append("")
    if getattr(r, "warmup_bars", None) is not None:  # R-11
        out.append(f"> Scored continuously: {r.settings.split(' · ')[-1]}. The holdout stays locked.")
        out.append("")
    by_year = getattr(r, "random_entry_by_year", None) or {}
    if by_year:  # R-14
        out.append("> Random entry by test year (each year's trips, whole, against draws in that year): " + "; ".join(
            f"{y}: {re_.return_percentile:.0f}th percentile of {re_.draws:,} draws, {re_.trades} trades"
            if re_.trades else f"{y}: no trades" for y, re_ in by_year.items()) + ".")
        out.append(f"> {CLOSE_PRICED}.")
        out.append("")
    if old is not None:
        out.append(f"> {SCREENING_DISCLOSURE}")
        out.append("")
    for label in getattr(r, "fill_labels", None) or []:
        out.append(f"> Fills: {label}. Without 1-minute bars, what traded first inside a bar is unknown, so a stop the "
                   "bar traded through was filled at its worst price, and a stop and a target both inside one bar as the "
                   "stop.")
        out.append("")
    if getattr(r, "margin_note", ""):
        out.append(f"> {r.margin_note}. Its liquidation prices and largest position use that table. Price tick and "
                   "quantity step: the venue's listing when the server reads it, else the contract's own (shortcut: "
                   "the sheet doesn't say which).")
        out.append("")
    check = getattr(r, "spot_check", None)
    if check and "gap" in check:
        out.append(f"> 5-minute pass spot-checked on 1-minute bars: the window {check['window']} (seed {check['seed']}) "
                   f"ended {check['gap']:+,.2f} on 1-minute bars against 5-minute ones.")
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
               f"volatility {oos['volatility']:.0%} vs {'n/a' if math.isnan(oos_b['volatility']) else f"{oos_b['volatility']:.0%}"}.")
    out.append("")
    if getattr(r, "hold_note", ""):
        out.append(f"Benchmark: {r.hold_note}.")
        out.append("")
    if getattr(r, "bar_note", ""):  # BH-102 rule 3: which hold set the bar, both Sharpes beside it
        out.append(f"G1 {r.bar_note}.")
        out.append("")
    if getattr(r, "hold_liquidated", None):  # BH-102: a 1x perp hold whose funding took its margin is liquidated
        out.append("The perp hold was liquidated on " + "; ".join(r.hold_liquidated) + ", and held nothing after.")
        out.append("")
    spot = getattr(r, "spot_hold_returns", None)
    perp = getattr(r, "perp_hold_returns", None)
    if perp is not None and len(perp.dropna()) >= 2 and getattr(r, "bar_hold", "") not in ("", "perp hold"):
        # BH-102 rule 3: the bar is another hold; the perp's beside it
        p_ = summary(perp)
        out.append(f"Holding the 1x perpetual instead, out-of-sample: CAGR {_pct(p_['cagr'])}, Sharpe "
                   f"{_num(p_['sharpe'])}, max drawdown {_pct(p_['max_drawdown'])} (shown, not the G1 benchmark).")
        out.append("")
    elif spot is not None and len(spot.dropna()) >= 2:
        s_ = summary(spot)
        out.append(f"Holding spot instead, out-of-sample: CAGR {_pct(s_['cagr'])}, Sharpe {_num(s_['sharpe'])}, "
                   f"max drawdown {_pct(s_['max_drawdown'])} (shown, not the G1 benchmark).")
        out.append("")
    schedule = getattr(getattr(r, "full_period", None), "funding_schedule", "")
    if schedule:
        out.append(f"Funding schedule: {schedule}.")
        out.append("")
    stress = getattr(r, "funding_stress", None)
    if stress:
        (s0, s1), (h0, h1) = stress["strategy"], stress["hold"]
        out.append(f"Funding stress, full research period: at {stress['rate']:.2%} a settlement instead of the assumed "
                   f"{stress['assumed']:.2%}, the strategy returns {_pct(s1)} (from {_pct(s0)}) and the perpetual buy "
                   f"and hold {_pct(h1)} (from {_pct(h0)}). The assumed rate is light in strong uptrends.")
        out.append("")
    out.extend(_period_tables(r))
    out.append(f"## Trading and costs (full research period, {r.research_start:%d %b %Y} to {r.research_end:%d %b %Y}, "
               "default params)")  # R-16: the span the table's trips and funding cover
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
    out.append(f"- Time in the market: {r.full_period.exposure.abs().gt(0.001).mean():.0%} of bars hold a position")
    out.append(f"- Stops clamped: {_n(getattr(r.full_period, 'stops_clamped', 0), 'trade')}, held at half the way to "
               "the liquidation price when the level loosened past it or funding moved it nearer")
    refused = getattr(r.full_period, "entries_refused_liquidation", 0)
    if refused:  # BH-102: a run the rule refuses throughout would otherwise show zero trades and say nothing
        out.append(f"- Entries refused: {refused} entr{'y' if refused == 1 else 'ies'}, the stop more than its share "
                   "of the way to the liquidation price (a fully paid 1x long is exempt)")
    out.append("")
    out.append(_double_fees_line(r))
    out.append("")
    out += analysis_pack.section(r)  # R-17/R-18/R-19: risk, trades and costs, before the cost ladder
    out.extend(_every_trade(r, files, shown_trades))
    if r.cost_ladder:
        chosen = getattr(r, "chosen_params", None) or r.default_params
        out.append(f"## Cost ladder (full research period, chosen settings {json.dumps(chosen)})")
        out.append("")
        out.append(f"**Break-even fee:** {_breakeven_words(r)}.")
        out.append("")
        out.append("| Fee per side | Total return | Sharpe | Round trips | Fees paid | Out-of-sample timing "
                   "| Random entry (median) |")
        out.append("| --- | --- | --- | --- | --- | --- | --- |")
        for rung in r.cost_ladder:
            timing, rand = getattr(rung, "oos_timing_return", None), getattr(rung, "random_return", None)
            out.append(f"| {rung.fee:.2%} | {_pct(rung.total_return)} | {_num(rung.sharpe)} | {rung.round_trips} "
                       f"| {rung.fees_paid:,.0f} | {'—' if timing is None else _pct(timing)} "
                       f"| {'—' if rand is None else _pct(rand)} |")
        out.append("")
        out.append("The settings tuning on the whole research period picks (best in-sample Sharpe on the grid), at "
                   "each fee, charged per side on maker and taker fills alike, so for a post-only strategy it mixes "
                   "the two. Every rung also pays half the "
                   f"bid-ask spread as above plus {r.ladder_slippage:.2%} slippage on orders that take liquidity. 0.02% and "
                   "0.05% are a low-fee perpetual venue's maker and taker rates, 0.10-0.40% typical spot taker rates, "
                   "0.80% a high-fee spot venue's taker rate. The break-even interpolates log(1 + return) between "
                   "rungs, then re-runs at that fee to verify it. The last two columns are the random-entry benchmark "
                   "at the rung's cost: the strategy's out-of-sample trips priced on the bars' closes, and the median "
                   "of the same number of random trips held as long.")
        out.append("")
    if r.by_regime:
        out += regime_split.render(r.by_regime)
    out.append("## Walk-forward folds")
    out.append("")
    out.append("| Train | Test to | Chosen params | Train Sharpe | Test trades | Test CAGR | Benchmark CAGR | Test Sharpe | Benchmark Sharpe |")
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for f in r.folds:
        out.append(
            f"| {f.train_start:%b %Y} to {f.train_end:%b %Y} | {f.test_end:%b %Y} | {json.dumps(f.chosen)} "
            f"| {_num(f.train_sharpe)} | {f.closed_in_window}{_halted_on(f.halted)} | {_pct(f.test['cagr'])} "
            f"| {_pct((f.benchmark_test or _NO_BENCHMARK)['cagr'])} | {_num(f.test['sharpe'])} "
            f"| {_num((f.benchmark_test or _NO_BENCHMARK)['sharpe'])} |"
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
    if getattr(r, "ablation", None) is not None:
        from sleeve_fund.research.ablation import section

        out.extend(section(r.ablation))
    out.append("## Idea counter")
    out.append("")
    uncertain = f" {N_UNCERTAIN}" if counts.get("n_uncertain") else ""
    if register is not None:
        out.append(f"- {_n(counts['variants'], 'distinct variant')} of this idea tried so far, in studies, backtests and "
                   f"paper strategies ({counts['evaluations']} evaluations including walk-forward refits): the N its "
                   f"results are judged by.{uncertain}")
        from sleeve_fund.research.candidates import BUDGET_PER_IDEA, configurations

        out.append(f"- Budget: {len(configurations(register, _idea(r)))} of {BUDGET_PER_IDEA} configurations used "
                   "for this idea family (the fund summary shows it beside the ranking, R-20).")
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
    from sleeve_fund.research.strength_report import section as strength_section

    out += strength_section(r, flat)
    from sleeve_fund.research.equal_exposure import section as equal_exposure_section

    out += equal_exposure_section(getattr(r, "equal_exposure", None))
    from sleeve_fund.research.placebo import section as placebo_section

    out += placebo_section(getattr(r, "placebo", None))
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
