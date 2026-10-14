"""The Researcher's definitions in configs/definitions/ (QR-B1..B9, 14 Oct): each passes the builder's checks at the
candle it decides on, as its first line names it, and builds. Data only: nothing here seeds or starts a Strategy."""

import re
from pathlib import Path

import pytest

from sleeve_fund.strategies.definitions import (
    Compiled,
    check_definition,
    load_definition_file,
    to_params,
)

DEFINITIONS = Path(__file__).resolve().parent.parent / "configs" / "definitions"
CANDLE_MINUTES = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}
RESEARCH = sorted(DEFINITIONS.glob("qr*.toml"))


def _decision_minutes(path: Path) -> int:
    m = re.search(r"Decision candle: (\w+)", path.read_text().splitlines()[0])
    assert m and m.group(1) in CANDLE_MINUTES, f"{path.name}: its first line names no decision candle"
    return CANDLE_MINUTES[m.group(1)]


def test_the_research_definitions_are_all_here():
    assert len(RESEARCH) >= 14, [p.name for p in RESEARCH]


@pytest.mark.parametrize("path", RESEARCH, ids=lambda p: p.stem)
def test_a_research_definition_passes_the_builder_checks_at_its_candle_and_builds(path):
    defn = load_definition_file(path)
    checked = check_definition(defn, bar_minutes=_decision_minutes(path))
    Compiled(checked)
    params = to_params(defn)
    assert params["definition"]["name"] == path.stem.replace("_", "-")
    # a long and a short leg are separate definitions while `exits` is shared by both sides (gap G1)
    assert ("long" in defn) != ("short" in defn), f"{path.name} carries both sides"
