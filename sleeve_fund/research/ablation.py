"""Ablation report (v2 P1-6-ABL): one rule-builder definition run with each of its blocks switched off in turn, to
show which pieces earn their place. Diagnostic only: nothing here is judged, ranked or promoted.

Spec: /mnt/project-files/sleeve-fund/qd/ablation-spec.md, approved by the Independent Quant Advisor (13 Oct 2026),
on PE1's plan and the Advisor's 7 Oct 18:20/18:25 UK rulings.

Units, one variant each, in path order: every condition inside a side's entry rule (a {left, op, right} or a
first_touch), each side's signal exit, a setup's cancel, and each of the definition's time_stop, stop, trail and
exit_at_level. Costs, blocks, sizing and risk caps are never units.

Removing a condition gives it its parent's neutral value (Advisor 18:20 Q1): inside `any` the member is dropped; in
every other place (all, only_when, event/confirm, setup/trigger) it becomes always true. Removing an exit, or a
cancel, removes it. A removal that leaves a side's entry always true is "not ablatable": entering without the signal
is what the random-entry benchmark already measures (Advisor 13 Oct Q1). One that empties an `any` means that side
never enters: reported, not run, no trials row.

A variant with no level exit left (stop, trail or exit_at_level, which rest at the venue in a backtest) is stopless,
so it runs at 1x, and the base also runs at 1x beside it so the difference reflects the exit, not the leverage
(Advisor 18:20 Q3, 13 Oct Q2).

Each variant that runs is one trials row, kind 'ablation', under the parent's idea and family, its own definition
hash, and the parent's hash and the removed path in its settings. Ablation rows are not in N, the 20-per-idea budget
or PBO (Advisor 18:25 Q8): TrialsRegister counts kind 'single' only. That exclusion is safe only because nothing
built here can be promoted (R6)."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from sleeve_fund.strategies.definitions import (
    LEVEL_EXITS,
    check_definition,
    definition_hash,
)

# A condition that always holds: what a removed condition becomes outside an `any`.
ALWAYS = {"left": 0.0, "op": ">=", "right": 0.0}
EXITS = ("time_stop", *LEVEL_EXITS)
NOT_JUDGED = "not judged"
RANDOM_ENTRY = "not ablatable; see random-entry benchmark"
# What a row's deltas compare: the figures a removed filter can move (Advisor 13 Oct Q3). Time in market is there
# because dropping a filter often lifts return only by adding exposure.
FIGURES = ("net_return", "sharpe", "max_drawdown", "trades", "time_in_market")


@dataclass(frozen=True)
class Score:
    """One evaluation's out-of-sample figures."""

    net_return: float
    sharpe: float | None
    max_drawdown: float
    trades: int
    time_in_market: float  # share of the window's bars holding a position
    data_start: object = None
    data_end: object = None


@dataclass
class Row:
    path: str  # the removed unit; "" for the base
    words: str
    status: str  # "ran", "not ablatable", "no trades"
    note: str = ""
    one_x: bool = False  # run stopless at 1x, compared with the base at 1x
    definition_hash: str = ""
    score: Score | None = None
    delta: dict = field(default_factory=dict)
    trial_id: str | None = None


@dataclass
class AblationReport:
    name: str
    parent_hash: str
    base: Row
    base_1x: Row | None
    rows: list[Row]

    @property
    def nothing_to_ablate(self) -> bool:
        return not any(r.status == "ran" for r in self.rows)


def _leaves(rule, path: str, out: list) -> None:
    """Every condition in an entry rule, by the path check_definition names it with."""
    if "op" in rule or "first_touch" in rule:
        out.append(path)
    elif "all" in rule or "any" in rule:
        for i, r in enumerate(rule["all"] if "all" in rule else rule["any"]):
            _leaves(r, f"{path}[{i}]", out)
    elif "only_when" in rule:
        _leaves(rule["only_when"], f"{path}.only_when", out)
        _leaves(rule["then"], f"{path}.then", out)
    elif "event" in rule:
        _leaves(rule["event"], f"{path}.event", out)
        _leaves(rule["confirm"], f"{path}.confirm", out)
    elif "setup" in rule:
        _leaves(rule["setup"], f"{path}.setup", out)
        _leaves(rule["trigger"], f"{path}.trigger", out)
        if "cancel" in rule:
            out.append(f"{path}.cancel")  # one unit: its neutral value is never cancelling


def units(defn: dict) -> list[str]:
    """The definition's ablation units, in path order: long, then short, then the exits."""
    out: list[str] = []
    for side in ("long", "short"):
        if side in defn:
            _leaves(defn[side]["entry"], f"{side}.entry", out)
            if "exit" in defn[side]:
                out.append(f"{side}.exit")
    out.extend(f"exits.{k}" for k in EXITS if k in (defn.get("exits") or {}))
    return out


def _steps(path: str) -> list:
    """'long.entry[1].only_when' -> ['long', 'entry', 1, 'only_when']."""
    steps: list = []
    for part in path.split("."):
        name, *idx = part.replace("]", "").split("[")
        steps.append(name)
        steps.extend(int(i) for i in idx)
    return steps


NEVER = object()  # a dropped `any` member while the tree is pruned: false, the neutral value inside an `any`


def _prune(rule):
    """The rule with every NEVER taken out: dropped from an `any` (an `any` left empty is NEVER), and NEVER anywhere
    else making its parent NEVER, as a condition that can't hold does."""
    if rule is NEVER:
        return NEVER
    if "all" in rule or "any" in rule:
        key = "all" if "all" in rule else "any"
        items = [_prune(r) for r in rule[key]]
        if key == "all":
            return NEVER if any(r is NEVER for r in items) else {"all": items}
        items = [r for r in items if r is not NEVER]
        return {"any": items} if items else NEVER
    parts = {"only_when": ("only_when", "then"), "event": ("event", "confirm"), "setup": ("setup", "trigger")}
    for k, keys in parts.items():
        if k in rule:
            pruned = {**rule, **{x: _prune(rule[x]) for x in keys}}
            return NEVER if any(pruned[x] is NEVER for x in keys) else pruned
    return rule


def _always(rule) -> bool:
    """Whether a rule holds on every candle."""
    if rule == ALWAYS:
        return True
    if "all" in rule:
        return all(_always(r) for r in rule["all"])
    if "any" in rule:
        return any(_always(r) for r in rule["any"])
    if "only_when" in rule:
        return _always(rule["only_when"]) and _always(rule["then"])
    if "event" in rule:
        return _always(rule["event"]) and _always(rule["confirm"])
    if "setup" in rule:
        return "cancel" not in rule and _always(rule["setup"]) and _always(rule["trigger"])
    return False


def remove(defn: dict, path: str) -> tuple[str, dict | None, str]:
    """(status, variant, note) for the definition with the unit at `path` switched off. status: "ran" (the variant
    to run), "not ablatable" or "no trades" (no variant)."""
    steps = _steps(path)
    variant = copy.deepcopy(defn)
    if steps[0] == "exits":
        del variant["exits"][steps[1]]
        if not variant["exits"]:
            del variant["exits"]
    else:
        parent = variant
        for s in steps[:-1]:
            parent = parent[s] if not isinstance(s, int) else parent["all" if "all" in parent else "any"][s]
        last = steps[-1]
        if last in ("exit", "cancel"):
            del parent[last]
        elif isinstance(last, int) and "any" in parent:
            parent["any"][last] = NEVER
            entry = _prune(variant[steps[0]]["entry"])
            if entry is NEVER:
                # Nothing is left that could hold, so the side never enters (Advisor 18:20 Q1, FOLLOW-UP edge).
                if len(steps) == 3:
                    return "not ablatable", None, f"it is the only member of the {steps[0]} entry's any"
                return "no trades", None, f"removing it empties an any, so the {steps[0]} side never enters"
            variant[steps[0]]["entry"] = entry
        elif isinstance(last, int):
            parent["all"][last] = copy.deepcopy(ALWAYS)
        else:
            parent[last] = copy.deepcopy(ALWAYS)
        if _always(variant[steps[0]]["entry"]):
            return "not ablatable", None, RANDOM_ENTRY
    try:
        check_definition(variant)
    except ValueError as e:
        return "not ablatable", None, f"without it the definition is refused: {e}"
    return "ran", variant, ""


def stopless(defn: dict) -> bool:
    """No level exit resting at the venue: the variant relies on its liquidation price alone."""
    return not any(k in (defn.get("exits") or {}) for k in LEVEL_EXITS)


def describe(defn: dict, path: str) -> str:
    """The removed unit in words, for the report."""
    steps = _steps(path)
    if steps[0] == "exits":
        return f"{steps[1].replace('_', ' ')} exit"
    if steps[-1] == "exit":
        return f"{steps[0]} signal exit"
    node = defn
    for s in steps:
        node = node[s] if not isinstance(s, int) else node["all" if "all" in node else "any"][s]
    if steps[-1] == "cancel":
        return f"{steps[0]} setup cancel"
    if "first_touch" in node:
        ft = node["first_touch"]
        return f"{steps[0]} entry: first touch of {_operand(ft['reach'])} before {_operand(ft['before'])}"
    return f"{steps[0]} entry: {_operand(node['left'])} {node['op'].replace('_', ' ')} {_operand(node['right'])}"


def _operand(x) -> str:
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return f"{x:g}"
    if isinstance(x, str):
        return x
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def _delta(score: Score, against: Score) -> dict:
    out = {}
    for k in FIGURES:
        a, b = getattr(score, k), getattr(against, k)
        out[k] = None if a is None or b is None else a - b
    return out


Evaluate = Callable[[dict, bool], Score]


def ablate(defn: dict, evaluate: Evaluate, *, register=None, dataset: str = "", idea_hash: str = "",
           family: str = "", venue: str | None = None) -> AblationReport:
    """Run the base and every ablatable variant through evaluate(definition, one_x) and report each against the
    base, in path order. register: a TrialsRegister to record each variant that ran in, as kind 'ablation' under
    idea_hash and family (the parent's); the base is the parent's own row and is not recorded again."""
    check_definition(defn)
    parent = definition_hash(defn, venue)
    name = defn.get("name") or parent
    plan = [(p, *remove(defn, p)) for p in units(defn)]
    base = Row("", "base (every block on)", "ran", definition_hash=parent, score=evaluate(defn, False))
    base_1x = None
    if any(status == "ran" and stopless(v) for _, status, v, _ in plan):
        base_1x = Row("", "base at 1x", "ran", note="beside the stopless variants", one_x=True,
                      definition_hash=parent, score=evaluate(defn, True))
    rows = []
    for path, status, variant, note in plan:
        row = Row(path, describe(defn, path), status, note=note)
        if status == "ran":
            row.one_x = stopless(variant)
            row.definition_hash = definition_hash(variant, venue)
            row.score = evaluate(variant, row.one_x)
            row.delta = _delta(row.score, (base_1x if row.one_x else base).score)
            if row.one_x:
                row.note = "stopless, at 1x; compared with the base at 1x"
        rows.append(row)
    report = AblationReport(name, parent, base, base_1x, rows)
    if register is not None:
        record(report, register, dataset=dataset, idea_hash=idea_hash, family=family)
    return report


def record(report: AblationReport, register, *, dataset: str, idea_hash: str, family: str) -> list[str]:
    """Write each variant that ran as a trials row, kind 'ablation'. Returns their ids."""
    from sleeve_fund.research.trials import trial_row

    if not idea_hash or not family:
        raise ValueError("an ablation row is recorded under its parent's idea and family; both are needed")
    out = []
    for r in report.rows:
        if r.status != "ran":
            continue
        s = r.score
        row = {**trial_row(definition_hash=r.definition_hash, idea_hash=idea_hash, name=report.name, family=family,
                           settings={"parent_definition_hash": report.parent_hash, "removed_path": r.path,
                                     "one_x": r.one_x},
                           dataset=dataset, stage="out_of_sample", source="study", sharpe=s.sharpe, trades=s.trades,
                           oos_trades=s.trades, data_start=s.data_start, data_end=s.data_end),
               "kind": "ablation"}
        register.store.add_trials([row])
        r.trial_id = row["id"]
        out.append(row["id"])
    return out


def _pts(x) -> str:
    """A change in a percentage, in percentage points."""
    return "n/a" if x is None else f"{x * 100:+.2f} pts"


def _num(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:+.2f}"


def _cells(s: Score | None, d: dict | None) -> list[str]:
    if s is None:
        return ["", "", "", "", ""]
    d = d or {}
    pair = [(f"{s.net_return:.2%}", _pts(d.get("net_return"))),
            ("n/a" if s.sharpe is None else f"{s.sharpe:.2f}", _num(d.get("sharpe"))),
            (f"{s.max_drawdown:.2%}", _pts(d.get("max_drawdown"))),
            (f"{s.trades:,}", "" if d.get("trades") is None else f"{d['trades']:+,}"),
            (f"{s.time_in_market:.0%}", _pts(d.get("time_in_market")))]
    return [v if not d else f"{v} ({c})" for v, c in pair]


def section(report: AblationReport) -> list[str]:
    """The tear sheet's ablation section, as markdown lines. Each figure carries its change against the base (or
    the base at 1x) in brackets. Nothing in it is a verdict."""
    out = ["## Ablation (not judged)", "",
           ("Each block switched off in turn, settings otherwise unchanged, out of sample. This shows what each "
            "piece contributes; it is not a test, and no variant here can be promoted."), ""]
    if report.nothing_to_ablate:
        out += ["Nothing to ablate: no block can be switched off on its own.", ""]
    out += ["| Removed | Net return | Sharpe | Max drawdown | Trades | Time in market | Note |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in [report.base, *([report.base_1x] if report.base_1x else []), *report.rows]:
        label = r.words if not r.path else f"{r.words} (`{r.path}`)"
        if r.status != "ran":
            out.append(f"| {label} | | | | | | {r.status}: {r.note} |")
        else:
            out.append(f"| {label} | " + " | ".join(_cells(r.score, r.delta)) + f" | {r.note} |")
    out.append("")
    return out


def _utc(x) -> pd.Timestamp:
    """A time as UTC; a store's naive timestamps are UTC."""
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def backtest_evaluator(prices: pd.DataFrame, instrument, windows, *, bar_minutes: int, holdout_start,
                       risk_profile: str | None = None, exec_prices: pd.DataFrame | None = None, half_spread=None,
                       starting_capital: float = 10_000.0, params: dict | None = None, locks=None,
                       idea_hash: str | None = None) -> Evaluate:
    """evaluate(definition, one_x) over the out-of-sample windows [(start, end), ...]: each window runs on the bars
    up to its end, trading from its start (so the blocks warm up on the bars before it), as a study's test windows
    do. one_x runs with no risk profile, where the engine's leverage is 1x, for the stopless variants and the base
    beside them.

    An ablation never reads the held-back period, so it can't spend or stale the holdout the edge test is judged on
    (CR pre-review, HoE 13 Oct): holdout_start is where the study's holdout begins (None only when the data keeps
    none back), and no bar from it on is read; a window reaching it is refused. With locks and idea_hash, a window
    overlapping a period this idea has already opened on the underlying is refused too. Nothing here opens a lock."""
    from sleeve_fund.research.metrics import (
        daily_returns,
        fills_to_rows,
        summary,
        trades,
    )
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.strategies.definitions import to_params

    if not windows:
        raise ValueError("an ablation needs at least one out-of-sample window")
    held = None if holdout_start is None else _utc(holdout_start)
    locked = None
    if locks is not None and idea_hash:
        from sleeve_fund.research.holdout import underlying_of

        row = locks.lock(idea_hash, underlying_of(str(instrument.id.symbol)))
        if row is not None and row.get("period_start") is not None and row.get("period_end") is not None:
            locked = (_utc(row["period_start"]), _utc(row["period_end"]))
    for start, end in windows:
        start, end = _utc(start), _utc(end)
        if held is not None and end >= held:
            raise ValueError(f"an ablation never reads the held-back period: the window to {end:%d %b %Y} reaches "
                             f"the holdout from {held:%d %b %Y}")
        if locked is not None and start <= locked[1] and end >= locked[0]:
            raise ValueError(f"an ablation never reads the held-back period: the window {start:%d %b %Y} to "
                             f"{end:%d %b %Y} overlaps this idea's opened holdout, {locked[0]:%d %b %Y} to "
                             f"{locked[1]:%d %b %Y}")
    if held is not None:
        prices = prices[prices.index < held]
        exec_prices = None if exec_prices is None else exec_prices[exec_prices.index < held]

    def evaluate(defn: dict, one_x: bool) -> Score:
        rets, held, bars, n = [], 0, 0, 0
        for start, end in windows:
            start, end = pd.Timestamp(start), pd.Timestamp(end)
            df = prices[prices.index <= end]
            fine = None if exec_prices is None else exec_prices[exec_prices.index <= end]
            p = {**(params or {}), **to_params(defn, str(instrument.id.venue)), "trade_from": start.value}
            res = run_backtest("rules", df, instrument, p, starting_capital=starting_capital, bar_minutes=bar_minutes,
                               risk_profile=None if one_x else risk_profile, exec_prices=fine,
                               half_spread=half_spread)
            eq = res.equity[res.equity.index >= start]
            rets.append(daily_returns(eq))
            ex = res.exposure[res.exposure.index >= start]
            held, bars = held + int((ex.abs() > 0).sum()), bars + len(ex)
            n += sum(1 for t in trades(fills_to_rows(res.fills), res.shorts) if pd.Timestamp(t["opened"]) >= start)
        r = pd.concat(rets).sort_index()
        s = summary(r) if len(r.dropna()) >= 2 else {"total_return": 0.0, "sharpe": None, "max_drawdown": 0.0}
        return Score(net_return=float((1 + r.fillna(0)).prod() - 1), sharpe=s["sharpe"],
                     max_drawdown=float(s["max_drawdown"]), trades=n, time_in_market=held / bars if bars else 0.0,
                     data_start=pd.Timestamp(windows[0][0]).to_pydatetime(),
                     data_end=pd.Timestamp(windows[-1][1]).to_pydatetime())

    return evaluate
