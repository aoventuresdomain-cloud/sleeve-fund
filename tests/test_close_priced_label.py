"""HoE 21 Oct (Deploy 2): RANDOM-ENTRY fills the headline and cost-ladder draws as the engine fills a rule definition's
exits; the per-year (R-14), equal-exposure (R-07) and placebo (R-24) blocks still call the close-priced benchmark, so
each says so until they are wired to the filled draws (a board follow-up after Deploy 2)."""

import math

from sleeve_fund.research.equal_exposure import EqualExposure, section as equal_exposure_section
from sleeve_fund.research.placebo import PlaceboResult, section as placebo_section
from sleeve_fund.research.random_entry import CLOSE_PRICED
from sleeve_fund.research.tearsheet import render
from test_continuous_window import _study

LABEL = "random entry: close-priced (headline and cost ladder use filled draws)."


def test_the_label_is_the_rulings_words():
    assert f"{CLOSE_PRICED}." == LABEL


def test_the_equal_exposure_block_says_its_random_entry_is_close_priced():
    e = EqualExposure(math.nan, {}, 0.0, 0.0, 0.0, {}, {}, math.nan, math.nan, math.nan)
    assert LABEL in equal_exposure_section(e)


def test_the_placebo_block_says_its_random_entry_is_close_priced():
    assert LABEL in placebo_section(PlaceboResult(why_not="test"))


def test_the_per_year_block_says_its_random_entry_is_close_priced(tmp_path):
    r, ledger = _study(tmp_path)
    assert r.random_entry_by_year
    lines = render(r, ledger).splitlines()
    at = next(i for i, x in enumerate(lines) if x.startswith("> Random entry by test year"))
    assert lines[at + 1] == f"> {LABEL}"


def _block(sheet: str, heading: str) -> str:
    """A sheet section from its heading to the next one."""
    lines = sheet.splitlines()
    at = next(i for i, x in enumerate(lines) if x.startswith(heading))
    end = next((i for i in range(at + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return "\n".join(lines[at:end])


def test_the_close_priced_blocks_carry_no_verdict_and_only_the_headline_decides(tmp_path):
    """Advisor (via the HoE, 21 Oct): while they are close-priced, the per-year, equal-exposure and placebo blocks show
    their percentiles with no PASS/FAIL word of their own, and none counts towards edge found: the checks table is
    the same whatever they say."""
    import dataclasses
    import re

    import numpy as np

    from sleeve_fund.research.random_entry import RandomEntryResult

    r, ledger = _study(tmp_path)
    plain = render(r, ledger)
    winning = PlaceboResult(signals=100, blocked=40, masked=50, variant_return=0.5, full_return=0.5, draws=100,
                            returns=np.zeros(100), draw_blocked=np.full(100, 40))
    losing = dataclasses.replace(winning, variant_return=-0.5)
    assert winning.passed and not losing.passed
    year = next(iter(r.random_entry_by_year))
    sheets = []
    for placebo, pct in ((winning, 99.0), (losing, 1.0)):
        r.placebo = placebo
        r.random_entry_by_year = {year: RandomEntryResult(pct, pct, 0.1, 0.0, 5, 0.3, 1000, False, False)}
        sheets.append(render(r, ledger))
    word = re.compile(r"\b(PASS|FAIL|YES|NO)\b")
    for sheet in sheets:
        assert not word.search(_block(sheet, "## The filter against a placebo"))
        assert not word.search(next(x for x in sheet.splitlines() if x.startswith("> Random entry by test year")))
    table = [[x for x in s.splitlines() if x.startswith("| ")] for s in (plain, *sheets)]
    checks = [[x for x in t if "Random" in x or "PASS" in x or "FAIL" in x] for t in table]
    assert checks[0] == checks[1] == checks[2]  # the verdicts don't move with them
