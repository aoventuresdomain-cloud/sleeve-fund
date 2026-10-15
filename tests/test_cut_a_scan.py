"""CUT-A decision-path scan (CUT-A-REPLAY, HoE 15 Oct; written for RESUME-LEG line 1): inside the fund no decision
reads a Strategy's own equity or drawdown, nor a leg's (Advisor SIZE-FUND Q4, RESUME-LEG). The scan reads the multi-asset and two-leg runtime's
source: every read of a Strategy's own equity figures (its peak, day's open, last equity, the pair's summed reference)
and every use of its own brakes (risk.check, the profile's max_drawdown and daily_loss) must either sit in a branch
that runs only outside the fund (a `fund_held` test) or be one of the readers ALLOWED names, by function, with its
exact count: display, journal text and bookkeeping that keeps the figures, none deciding anything inside the fund.
A new read anywhere, or one more in an allowed reader, turns it red until it is guarded or argued onto the list."""
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pytest

import sleeve_fund

ROOT = Path(sleeve_fund.__file__).parent
FILES = ("paper/runtime.py", "paper/node.py", "strategies/base.py", "strategies/pair.py", "two_legs.py")
OWN = {"peak", "_day_open", "_last_equity", "reference_equity", "max_drawdown", "daily_loss"}
OWN_CALLS = {"last_equity", "pair_state"}

# (file, function): (unguarded reads, why none decides anything inside the fund)
ALLOWED = {
    ("paper/runtime.py", "tick"): (8, ("keeps the peak and day's open marked (the pair's too) and words the "
                                           "drawdown journal rows; the brakes are in the branch after `elif "
                                           "self.fund_held`")),
    ("paper/runtime.py", "_pair_first_mark"): (2, "bookkeeping: restores the pair's drawdown reference"),
    ("paper/runtime.py", "_pair_rebase_after_ral"): (1, "journal: the re-base row's old peak"),
    ("paper/runtime.py", "_reset_after_liquidation"): (1, "journal: the reset row's old peak"),
    ("strategies/base.py", "_margin_lost"): (2, "journal: the liquidation halt's share, of the fund inside it"),
    ("strategies/base.py", "_on_tick"): (1, "journal: the peak written beside a risk exit the tick decided"),
    ("strategies/base.py", "_risk_level"): (4, "backtest only, where the single Strategy is the whole fund"),
    ("strategies/pair.py", "_retry_refusal"): (1, "the pair's legging state, no equity"),
    ("strategies/pair.py", "_write"): (1, "the pair's legging state, no equity"),
    ("strategies/pair.py", "journal"): (2, "the pair's legging state, no equity"),
}


def _mentions_fund_held(node) -> bool:
    return "fund_held" in ast.unparse(node)


def _outside_only(test, branch: str) -> bool:
    """True when the `branch` ("body" or "orelse") of an `if test` runs only outside the fund."""
    if not _mentions_fund_held(test):
        return False
    negated = (isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)) or (
        isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And) and any(
            isinstance(v, ast.UnaryOp) and isinstance(v.op, ast.Not) and _mentions_fund_held(v) for v in test.values))
    return negated if branch == "body" else not negated and not isinstance(test, ast.BoolOp)


def _read(node) -> str | None:
    if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) and node.attr in OWN:
        return node.attr
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr in OWN_CALLS:
            return f"{node.func.attr}()"
        if node.func.attr == "check" and isinstance(node.func.value, ast.Name) and node.func.value.id == "risk":
            return "risk.check"
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value == "reference_equity":
        return "['reference_equity']"
    return None


def unguarded(source: str) -> Counter:
    """Each function's reads of a Strategy's own equity or brakes that no `fund_held` test keeps outside the fund."""
    found: Counter = Counter()

    def walk(node, func, guarded):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func = node.name
        if not guarded and _read(node):
            found[func] += 1
        if isinstance(node, ast.If):
            walk(node.test, func, guarded)
            for branch in ("body", "orelse"):
                for child in getattr(node, branch):
                    walk(child, func, guarded or _outside_only(node.test, branch))
            return
        for child in ast.iter_child_nodes(node):
            walk(child, func, guarded)

    walk(ast.parse(source), "<module>", False)
    return found


def _scan(file: str, source: str | None = None) -> dict:
    counts = unguarded(source if source is not None else (ROOT / file).read_text())
    return {(file, f): n for f, n in counts.items()}


def test_no_decision_path_reads_a_strategys_own_equity_or_drawdown_inside_the_fund():
    found = {k: v for f in FILES for k, v in _scan(f).items()}
    extra = {k: v for k, v in found.items() if k not in ALLOWED or v > ALLOWED[k][0]}
    assert extra == {}, f"unguarded own-equity reads (guard them with fund_held, or argue them onto ALLOWED): {extra}"
    stale = {k: (found.get(k, 0), n) for k, (n, _) in ALLOWED.items() if found.get(k, 0) < n}
    assert stale == {}, f"ALLOWED counts above what is read now (lower them): {stale}"


@pytest.mark.parametrize("file, anchor, reader", [
    # A new brake on the tick, ahead of the fund_held branch: the drawdown halt by another name.
    ("paper/runtime.py", "        flatten = False\n        self.flatten_why = None\n",
     "        if equity < self.peak * 0.8:\n            flatten = True\n"),
    # A leg sizing off its own last equity, in a new helper.
    ("strategies/base.py", "    def _guard_breached(self, price: float) -> bool:\n",
     "    def _budget(self):\n        return self.runtime._last_equity * 0.3\n\n"),
    # The replay's own levels without their guard (the leak this scan found, 15 Oct).
    ("strategies/base.py", ' and not getattr(self.runtime, "fund_held", False):\n                p, day_open',
     ":\n                p, day_open"),
])
def test_a_deliberate_reader_turns_the_scan_red(file, anchor, reader):
    source = (ROOT / file).read_text()
    assert source.count(anchor) == 1, anchor
    seeded = source.replace(anchor, reader + anchor if not anchor.startswith(" and not") else reader)
    found = _scan(file, seeded)
    assert any(k not in ALLOWED or v > ALLOWED[k][0] for k, v in found.items()), found
