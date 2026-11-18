"""No definition's `reason` carries a result figure (CR M1 on #368, HoE 18 Nov). The repository is public and a reason
is outside a definition's hash, so a figure there would publish a result without changing what was tested. A reason
may state the rule's own settings (1.5 ATR, a 1% band, 2 x volume, 1x); it may not state a P&L, a cost in bps, a trade
count or a Sharpe ratio. Figures in the examples below are made up."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from sleeve_fund.strategies.definitions import load_definition_file

ROOT = Path(__file__).resolve().parents[1]
DEFINITIONS = sorted((ROOT / "configs" / "definitions").rglob("*.toml"))

# A result figure: a signed number (-1,234, +5.6%), a thousands figure, a count of trades or tries, a figure in basis
# points, or a Sharpe ratio. A rule setting is unsigned and has no thousands separator, so it never matches.
RESULT_FIGURE = re.compile(
    r"(?<![\w.])[-+−]\d"
    r"|\b\d{1,3}(?:,\d{3})+\b"
    r"|\b\d[\d,]*\s+(?:trades|tries)\b"
    r"|\d\s*bps?\b"
    r"|\bsharpe\b[^.;]{0,15}\d",
    re.IGNORECASE,
)


def figures(reason: str) -> list[str]:
    return [m.group(0) for m in RESULT_FIGURE.finditer(reason)]


@pytest.mark.parametrize("reason", [
    "an approximate check found the entries it keeps lost -1,234 over 123 trades",
    "the costs run to 9.9 bps a side",
    "it made +4.2% after fees",
    "a Sharpe of 0.7 out of sample",
    "it kept 77 trades of the book",
    "the book's P&L came to 4,321",
])
def test_the_pattern_catches_a_result_figure(reason):
    assert figures(reason)


@pytest.mark.parametrize("reason", [
    "entries at RSI(2) below 5 / above 95; a 2.5 ATR(14) stop; out after 10 candles. Held at 1x.",
    "long when the close is more than 1% above its 50-day average; (1, 50, 1%) is the paper's standard",
    "on at least 1.5 x normal volume within 20 candles",
    "the top tenth and at least 0.03% per 8 hours",
])
def test_the_pattern_leaves_a_rule_setting_alone(reason):
    assert not figures(reason)


def test_there_are_definitions_to_read():
    assert len(DEFINITIONS) > 50


@pytest.mark.parametrize("path", DEFINITIONS, ids=lambda p: p.stem)
def test_no_definition_reason_carries_a_result_figure(path):
    reason = load_definition_file(path).get("reason", "")
    assert not figures(reason), f"{path.name}: its reason carries {figures(reason)}"
