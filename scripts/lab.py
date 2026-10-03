"""Strategy lab runner. Research only: reads public price history, places no orders.

    python scripts/lab.py data                      # fetch or extend minute history
    python scripts/lab.py classifier [--quick]      # test 1: does the day-type classifier work?

Results go to research/results/ as Markdown and JSON, and every evaluation is appended to the
idea ledger so later scores are judged against the number of tries.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sleeve_fund.lab import classifier_test as ct  # noqa: E402
from sleeve_fund.lab import data, sim  # noqa: E402
from sleeve_fund.lab import mtf_trend as mt  # noqa: E402
from sleeve_fund.lab import pairs as pr  # noqa: E402
from sleeve_fund.lab import sweep as sw  # noqa: E402
from sleeve_fund.lab import vwap_day as vd  # noqa: E402
from sleeve_fund.research.ledger import IdeaLedger  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "minute"
RESULTS = ROOT / "research" / "results"
LEDGER = ROOT / "research" / "idea_ledger.jsonl"
BASKET = list(data.SYMBOLS)


def cmd_data(args) -> None:
    for inst in args.instruments:
        t = time.time()
        p = data.update(inst, DATA, log=lambda m: None)
        df = data.load(inst, DATA)
        print(f"{inst}: {data.span(df)} ({time.time() - t:.0f}s) -> {p.name}")


def _fmt(x, pct=False, d=2):
    if x is None or (isinstance(x, float) and x != x):
        return "n/a"
    return f"{x:.0%}" if pct else f"{x:.{d}f}"


# Classifier threshold sets. "doc" is the strategy doc's rules; "relaxed" is the one revision tried after
# the first run called trend on only 2-6% of sessions (counted in the idea ledger like any other try).
CLASSIFIER_VARIANTS = {
    "doc": {},
    "relaxed": {"side_share": 0.7, "trend_slope": 0.10, "relvol_min": 1.0},
}


def cmd_classifier(args) -> None:
    bars = [5] if args.quick else [5, 15]
    anchors = ["utc", "us_open"]
    decide = [60] if args.quick else [60, 90]
    rows, spans = [], {}
    for f in RESULTS.glob("classifier_sessions_*"):  # superseded per-session dumps (they included the holdout)
        f.unlink()
    for inst in args.instruments:
        m = data.load(inst, DATA)
        _, end = dev_window(m)
        m = m[m.index < end]  # the holdout year stays untouched
        spans[inst] = data.span(m)
        for bar in bars:
            b = data.resample(m, bar)
            for anchor in anchors:
                for d in decide:
                    if d % bar:
                        continue
                    for vname, th in CLASSIFIER_VARIANTS.items():
                        so = ct.session_outcomes(b, anchor=anchor, decide_at=d, **th)
                        for horizon in ("4h", "session"):
                            s = ct.summarise(so, horizon)
                            rows.append({"instrument": inst, "variant": vname, "bar": bar, "anchor": anchor,
                                         "decide_at": d, "horizon": horizon, **s})
                        print(f"{inst} {vname} {bar}m {anchor} @{d}: {rows[-1]['verdict']}", flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    (RESULTS / "classifier.json").write_text(json.dumps({"run": stamp, "spans": spans, "rows": rows}, default=str,
                                                        indent=1))
    lines = [f"# Classifier test ({stamp})", "",
             "Does the day-type classifier tell trending sessions from balanced ones, using only bars closed by the "
             "decision time? Continuation is the move after the decision in the trend's direction, in daily ATR; "
             "the 95% range is a bootstrap. Move is the absolute move. t compares trend and balance moves. "
             "Development history only (the most recent year is held back). Variants: "
             + "; ".join(f"{k} {v or 'as the doc'}" for k, v in CLASSIFIER_VARIANTS.items()) + ".", "",
             "## History used", "", "| Instrument | From | To | Years | Missing minutes |", "|---|---|---|---|---|"]
    for k, v in spans.items():
        lines.append(f"| {k} | {v.get('start')} | {v.get('end')} | {v.get('years')} | {v.get('missing_pct')}% |")
    lines += ["", "## Results", "",
              "| Instrument | Variant | Bar | Session | Decide | Horizon | Sessions | Trend share | Balance share "
              "| Continuation (95% range) | Hit | Move trend | Move balance | t | Verdict |", "|" + "---|" * 15]
    for r in rows:
        c = r.get("trend_continuation_atr") or (float("nan"),) * 3
        sh = r.get("share", {})
        lines.append(
            f"| {r['instrument']} | {r['variant']} | {r['bar']}m | {r['anchor']} | {r['decide_at']}m | {r['horizon']} | {r['sessions']} "
            f"| {_fmt(sh.get('trend_up', 0) + sh.get('trend_down', 0), True)} | {_fmt(sh.get('balance'), True)} "
            f"| {_fmt(c[0])} ({_fmt(c[1])} to {_fmt(c[2])}) | {_fmt(r.get('trend_continuation_hit'), True)} "
            f"| {_fmt(r.get('move_trend_atr'))} | {_fmt(r.get('move_balance_atr'))} | {_fmt(r.get('move_t_trend_vs_balance'), d=1)} "
            f"| {r['verdict']} |")
    (RESULTS / "classifier.md").write_text("\n".join(lines) + "\n")
    ledger = IdeaLedger(LEDGER)
    for r in rows:
        c = r.get("trend_continuation_atr") or (0,)
        ledger.record(idea="day_type_classifier", family="classifier",
                      params={"variant": r["variant"], **CLASSIFIER_VARIANTS[r["variant"]], "bar": r["bar"], "anchor": r["anchor"], "decide_at": r["decide_at"], "horizon": r["horizon"]},
                      dataset=f"binance_1m:{r['instrument']}", stage="diagnostic",
                      sharpe=float(c[0]) if c[0] == c[0] else 0.0)
    print(f"wrote {RESULTS / 'classifier.md'}")


HOLDOUT_DAYS = 365
STAGES = [("1 week", 7), ("1 month", 30), ("3 months", 91), ("6 months", 182), ("1 year", 365),
          ("2 years", 730), ("all development history", None)]
COST_LEVELS = ["kraken_pro_taker", "high_volume_taker", "institutional"]


def dev_window(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Development period: everything except the most recent year, which stays untouched
    (the holdout) until a strategy is final."""
    end = df.index[-1].floor("D") - pd.Timedelta(days=HOLDOUT_DAYS)
    return df.index[0], end


def _stage_rows(name: str, res_by_inst: dict, starts: dict, end_by_inst: dict, days: int | None) -> dict:
    """Stats for one variant over one look-back window, per instrument and for the equal-weight
    basket. A result is either a trades table or, for always-invested benchmarks, daily returns."""
    per, daily_all, trades = {}, [], {}
    for inst, res in res_by_inst.items():
        end = end_by_inst[inst]
        start = starts[inst] if days is None else max(starts[inst], end - pd.Timedelta(days=days))
        if isinstance(res, pd.Series):
            d = res[(res.index >= start.floor("D")) & (res.index < end)]
            sub = sim.trades_frame([], inst)
        else:
            et = pd.to_datetime(res["entry_time"]) if len(res) else pd.Series(dtype="datetime64[ns, UTC]")
            sub = res[(et >= start) & (et < end)] if len(res) else res
            d = sim.daily_returns(sub, start, end - pd.Timedelta(minutes=1))
        per[inst] = sim.stats(sub, d)
        trades[inst] = sub
        daily_all.append(d.rename(inst))
    basket = pd.concat(daily_all, axis=1).fillna(0).mean(axis=1)
    pooled = pd.concat([t for t in trades.values() if len(t)]) if any(len(t) for t in trades.values()) else sim.trades_frame([])
    is_series = all(isinstance(r, pd.Series) for r in res_by_inst.values())
    return {"per": per, "basket": sim.stats(pooled, basket), "always_in": is_series,
            "check": sim.basket_check(trades) if not is_series else {"passes": False}}


def write_report(*, slug: str, title: str, note: str, idea: str, family: str, params: dict, results: dict,
                 starts: dict, ends: dict, spans: dict) -> None:
    """Staged report: every variant over every look-back window, per mode and cost level."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    insts = list(starts)
    variants = sorted({k[0] for k in results}, key=lambda v: (v.startswith("bench"), v))
    L = [f"# {title}", "", f"Run {stamp}. Development history only: the most recent {HOLDOUT_DAYS} days are held "
         "back untouched. Short windows are smoke tests, not verdicts. Expectancy is the average net return per trade "
         "in basis points of equity. Sharpe uses daily returns of an equal-weight basket (per instrument in the "
         "right-hand columns). Costs are per side. " + note, "",
         "## History", "", "| Instrument | From | To (holdout starts) | Years |", "|---|---|---|---|"]
    for k, v in spans.items():
        L.append(f"| {k} | {v.get('start')} | {v.get('end')} | {v.get('years')} |")
    jsn = {"run": stamp, "params": params, "stages": {}}
    for mode_name, long_only in (("long and short", False), ("long only (spot)", True)):
        for cost in COST_LEVELS:
            if not any(k[1] == long_only and k[2] == cost for k in results):
                continue
            L += ["", f"## {mode_name}, {cost.replace('_', ' ')} costs ({sim.COSTS[cost]['fee']:g} bp fee + "
                      f"{sim.COSTS[cost]['slip']:g} bp slippage per side)", "",
                  "| Variant | Window | Trades | Expectancy bp | Win rate | Sharpe | Total return | Max DD | "
                  + " | ".join(f"{i.removesuffix('/USD')} Sharpe" for i in insts) + " | Cross-asset pass |",
                  "|" + "---|" * (9 + len(insts))]
            for v in variants:
                res = results.get((v, long_only, cost))
                if not res:
                    continue
                for stage, days in STAGES:
                    r = _stage_rows(v, res, starts, ends, days)
                    b = r["basket"]
                    jsn["stages"].setdefault(f"{v}|{mode_name}|{cost}", {})[stage] = r
                    L.append(f"| {v} | {stage} | {'always in' if r['always_in'] else b['trades']} | "
                             f"{_bp(b['expectancy'])} | {_rate(b['win_rate'])} | {b['sharpe']:.2f} | "
                             f"{_pct(b['total_return'])} | {_pct(-b['max_dd'])} | "
                             + " | ".join(f"{r['per'][i]['sharpe']:.2f}" for i in insts)
                             + f" | {'n/a' if r['always_in'] else ('yes' if r['check']['passes'] else 'no')} |")
    out = RESULTS / slug
    out.with_suffix(".md").write_text("\n".join(L) + "\n")
    out.with_suffix(".json").write_text(json.dumps(jsn, default=str))
    ledger = IdeaLedger(LEDGER)
    for (v, long_only, cost), res in results.items():
        if cost != "kraken_pro_taker":
            continue
        for inst, r in res.items():
            st = _stage_rows(v, {inst: r}, starts, ends, None)["per"][inst]
            ledger.record(idea=f"{idea}:{v}", family="benchmark" if v.startswith("bench") else family,
                          params={**params, "long_only": long_only}, dataset=f"binance_1m:{inst}", stage="screen",
                          sharpe=st["sharpe"])
    print(f"wrote {out}.md")


def cmd_vwap_day(args) -> None:
    """Strategies 1 and 2 with benchmarks, staged from 1 week to all development history."""
    bar = args.bar
    variants = {
        "1 trend pullback": vd.Params(s2=False),
        "2 band fade": vd.Params(s1=False),
        "1+2 together": vd.Params(),
        "bench: 1 without classifier": vd.Params(s2=False, use_classifier=False),
        "bench: 2 on every day": vd.Params(s1=False, use_classifier=False),
    }
    for anchor in args.anchors:
        results = {}  # (variant, mode, cost) -> {inst: trades}
        starts, ends, spans = {}, {}, {}
        for inst in args.instruments:
            m = data.load(inst, DATA)
            lo, hi = dev_window(m)
            m = m[m.index < hi]
            spans[inst] = data.span(m)
            b = data.resample(m, bar)
            starts[inst] = b.index[0] + pd.Timedelta(days=40)  # indicator warm-up
            ends[inst] = hi
            for vname, p0 in variants.items():
                p = vd.Params(**{**p0.as_dict(), "anchor": anchor})
                prepared = vd.prepare(b, p)
                for long_only in (False, True):
                    for cost in COST_LEVELS:
                        book = vd.run(b, p, cost=cost, long_only=long_only, start=starts[inst], prepared=prepared)
                        tf = sim.trades_frame(book.trades, inst)
                        results.setdefault((vname, long_only, cost), {})[inst] = tf
                print(f"{anchor} {inst} {vname}: {len(results[(vname, True, COST_LEVELS[0])][inst])} long-only trades",
                      flush=True)
            for long_only in (False, True):
                for cost in COST_LEVELS:
                    tf = vd.intraday_momentum(b, anchor=anchor, cost=cost, long_only=long_only, start=starts[inst])
                    tf["instrument"] = inst
                    results.setdefault(("bench: intraday momentum", long_only, cost), {})[inst] = tf
        write_report(slug=f"vwap_day_{bar}m_{anchor}", title=f"Strategies 1 and 2: staged screen ({bar}-minute bars, {anchor} sessions)",
                     note="Risk 0.5% of equity per trade, leverage up to 4x (1x long-only).", idea="vwap_day",
                     family="intraday_vwap", params={"anchor": anchor, "bar": bar}, results=results, starts=starts,
                     ends=ends, spans=spans)


def cmd_mtf(args) -> None:
    """Strategy 4 against its benchmarks: the plain crossover (always in) and buy-and-hold."""
    bar = args.bar if args.bar != 5 else 15
    p = mt.Params(htf_min=args.htf)
    results, starts, ends, spans = {}, {}, {}, {}
    for inst in args.instruments:
        m = data.load(inst, DATA)
        lo, hi = dev_window(m)
        m = m[m.index < hi]
        spans[inst] = data.span(m)
        b = data.resample(m, bar)
        starts[inst] = b.index[0] + pd.Timedelta(days=60)  # higher-timeframe warm-up
        ends[inst] = hi
        prepared = mt.prepare(b, p)
        for long_only in (False, True):
            for cost in COST_LEVELS:
                book = mt.run(b, p, cost=cost, long_only=long_only, start=starts[inst], prepared=prepared)
                results.setdefault(("4 crossover + anchored VWAP entry", long_only, cost), {})[inst] = sim.trades_frame(book.trades, inst)
                results.setdefault(("bench: plain crossover, always in", long_only, cost), {})[inst] = \
                    mt.crossover_benchmark(b, p, cost=cost, long_only=long_only, start=starts[inst])
                results.setdefault(("bench: buy and hold", long_only, cost), {})[inst] = mt.buy_and_hold(b, starts[inst])
        print(f"{inst}: {len(results[('4 crossover + anchored VWAP entry', False, COST_LEVELS[0])][inst])} trades", flush=True)
    write_report(slug=f"mtf_{bar}m_{args.htf}m", title=f"Strategy 4: staged screen ({bar}-minute entries, {args.htf // 60}-hour direction)",
                 note="Risk 0.75% of equity per entry, leverage up to 3x (1x long-only). Not yet included: the "
                      "swing-low anchor alternative and the add-on entry.", idea="mtf_avwap", family="trend",
                 params={"bar": bar, **p.as_dict()}, results=results, starts=starts, ends=ends, spans=spans)


def cmd_sweep(args) -> None:
    """Strategy 3 on core levels, against trading the breakout of the same levels."""
    bar = args.bar if args.bar != 5 else 15
    variants = {"3 sweep reversal": sw.Params(), "3 sweep reversal, no trend filter": sw.Params(trend_filter=False),
                "bench: breakout of the same levels": sw.Params(opposite=True, trend_filter=False)}
    results, starts, ends, spans = {}, {}, {}, {}
    for inst in args.instruments:
        m = data.load(inst, DATA)
        lo, hi = dev_window(m)
        m = m[m.index < hi]
        spans[inst] = data.span(m)
        b = data.resample(m, bar)
        starts[inst] = b.index[0] + pd.Timedelta(days=40)
        ends[inst] = hi
        for vname, p in variants.items():
            prepared = sw.prepare(b, p)
            for long_only in (False, True):
                for cost in COST_LEVELS:
                    book = sw.run(b, p, cost=cost, long_only=long_only, start=starts[inst], prepared=prepared)
                    results.setdefault((vname, long_only, cost), {})[inst] = sim.trades_frame(book.trades, inst)
        print(f"{inst}: {len(results[('3 sweep reversal', False, COST_LEVELS[0])][inst])} trades", flush=True)
    write_report(slug=f"sweep_{bar}m", title=f"Strategy 3: staged screen ({bar}-minute bars, core levels)",
                 note="Levels: prior session high and low, Asia session (00:00 to 08:00 UTC) high and low. Risk 0.5% "
                      "per trade, leverage up to 4x (1x long-only).", idea="sweep", family="intraday_reversal",
                 params={"bar": bar, **sw.Params().as_dict()}, results=results, starts=starts, ends=ends, spans=spans)


def cmd_pairs(args) -> None:
    """Strategy 5 on every pair of the basket (pairs need shorting: no long-only version)."""
    hourly = {}
    for inst in args.instruments:
        m = data.load(inst, DATA)
        lo, hi = dev_window(m)
        hourly[inst] = (data.resample(m[m.index < hi], 60), hi)
    variants = {"5 regime-switching pairs": pr.Params(), "5 revert regime only": pr.Params(allow_trend=False),
                "bench: static hedge, no regime switch": pr.Params(regime_switch=False)}
    results, starts, ends, spans = {}, {}, {}, {}
    for a, b in pr.pair_names(args.instruments):
        name = f"{a.split('/')[0]}/{b.split('/')[0]}"
        ha, hb = hourly[a][0], hourly[b][0]
        common = ha.index.intersection(hb.index)
        if len(common) < 24 * 400:
            print(f"{name}: not enough shared history", flush=True)
            continue
        starts[name] = common[0] + pd.Timedelta(days=200)  # 180-day selection window plus warm-up
        ends[name] = min(hourly[a][1], hourly[b][1])
        spans[name] = {"start": str(common[0].date()), "end": str(common[-1].date()),
                       "years": round((common[-1] - common[0]).days / 365.25, 2)}
        for vname, p in variants.items():
            df = pr.prepare(ha.loc[common], hb.loc[common], p)
            for cost in COST_LEVELS:
                results.setdefault((vname, False, cost), {})[name] = pr.run(df, p, cost=cost, start=starts[name], name=name)
        print(f"{name}: {len(results[('5 regime-switching pairs', False, COST_LEVELS[0])][name])} trades", flush=True)
    write_report(slug="pairs_1h", title="Strategy 5: staged screen (hourly bars, every pair in the basket)",
                 note="Each trade risks 0.75% of equity to the z = 3.5 stop; gross leverage up to 3x; the short leg "
                      "pays 5% a year. Exits act on hourly closes. Trend-regime entries are the EMA crossover itself, "
                      "not yet the pullback to the spread's anchored VWAP.", idea="pairs", family="relative_value",
                 params=pr.Params().as_dict(), results=results, starts=starts, ends=ends, spans=spans)


def _pct(x, d=1):
    return "n/a" if x is None or x != x else f"{x * 100:+.{d}f}%"


def _rate(x):
    return "n/a" if x is None or x != x else f"{x:.0%}"


def _bp(x):
    return "n/a" if x is None or x != x else f"{x * 1e4:+.0f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["data", "classifier", "vwap_day", "mtf", "sweep", "pairs"])
    ap.add_argument("--htf", type=int, default=240)
    ap.add_argument("--bar", type=int, default=5)
    ap.add_argument("--anchors", nargs="*", default=["utc", "us_open"])
    ap.add_argument("--instruments", nargs="*", default=BASKET)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    {"data": cmd_data, "classifier": cmd_classifier, "vwap_day": cmd_vwap_day, "mtf": cmd_mtf, "sweep": cmd_sweep, "pairs": cmd_pairs}[args.command](args)


if __name__ == "__main__":
    main()
