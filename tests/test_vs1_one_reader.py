"""VS-1 Done when 2 (one liquidation formula): margin.py is the one reader of a maintenance rate. Everywhere else a
Strategy's maintenance figure (one stated rate, or the venue's tiers for the instrument) is only passed on, to a
margin or markets function or a type check, never multiplied, compared or turned into a number of its own; so no
reader can price a liquidation at one flat rate where the venue's tiers apply."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

import sleeve_fund

ROOT = Path(sleeve_fund.__file__).parent
FIELD = "maintenance_margin"
OWNERS = {"margin.py"}  # the one reader: tiers, tier_at and the solves


def _parents(tree):
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node


def misreads(source: str) -> list[int]:
    """Lines reading a maintenance figure as a number: any load of `.maintenance_margin` that isn't an argument of a
    call (passed on), nor a field set where the figure is declared."""
    tree = ast.parse(source)
    _parents(tree)
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and node.attr == FIELD and isinstance(node.ctx, ast.Load)):
            continue
        up = node.parent
        if isinstance(up, ast.keyword):
            up = up.parent
        if isinstance(up, ast.Call) and node is not up.func:
            continue  # passed on whole
        out.append(node.lineno)
    return out


def _files():
    return sorted(p for p in ROOT.rglob("*.py") if p.name not in OWNERS and "migrations" not in p.parts)


def test_no_reader_outside_margin_prices_a_liquidation_at_one_rate_of_its_own():
    found = {str(p.relative_to(ROOT)): lines for p in _files() if (lines := misreads(p.read_text()))}
    assert found == {}, f"a maintenance figure read as a number outside margin.py (pass it to margin): {found}"


@pytest.mark.parametrize("reader", [
    "liq = entry * (1 - 1 / lev + t.maintenance_margin)",  # the flat-rate long formula, inline
    "rate = cfg.perp.maintenance_margin",  # taken out to use later
    "if terms.maintenance_margin > 0.004:\n    pass",
])
def test_a_deliberate_reader_turns_it_red(reader):
    assert misreads(reader) == [1]
