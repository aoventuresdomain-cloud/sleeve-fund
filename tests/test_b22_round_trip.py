"""B-22 round trip (HoE): what a real study writes on its tear sheet is what the page reads and shows. The study side
(research.study, tearsheet) writes the 'Profitable at double fees' line; the page side (dashboard.development) parses
it. One real run at twice the venue's fees, rendered to a sheet on disk, read back as the page reads it."""

import dataclasses

import pytest

from sleeve_fund.dashboard import development as dev
from sleeve_fund.data import synthetic_ohlcv
from sleeve_fund.research.ledger import IdeaLedger
from sleeve_fund.research.study import run_study
from sleeve_fund.research.tearsheet import render
from sleeve_fund.strategies.trend_filter import SPEC


@pytest.fixture(scope="module")
def ran(tmp_path_factory, instrument):
    ledger = IdeaLedger(tmp_path_factory.mktemp("b22rt") / "l.jsonl")
    r = run_study(SPEC, synthetic_ohlcv(days=1900, seed=3), instrument, ledger=ledger, dataset="syn", synthetic=True,
                  holdout_days=200, train_days=730, test_days=365, use_holdout=True)
    return r, ledger


def test_b22j_the_page_shows_the_verdict_and_net_the_study_wrote(ran, tmp_path):
    r, ledger = ran
    d = r.double_fees
    assert d is not None
    sheet = tmp_path / "trend_filter_syn_20261020-180000.md"
    sheet.write_text(render(r, ledger), encoding="utf-8")
    shown = dev.read_sheet(sheet)["double_fees"]
    assert shown == {"verdict": "PASS" if d["net_return"] > 0 else "FAIL", "net": f"{d['net_return'] * 100:+.1f}%"}


def test_b22j_a_study_with_no_window_to_re_run_shows_not_tested(ran, tmp_path):
    r, ledger = ran
    sheet = tmp_path / "trend_filter_syn_20261020-180100.md"
    sheet.write_text(render(dataclasses.replace(r, double_fees=None), ledger), encoding="utf-8")
    assert dev.read_sheet(sheet)["double_fees"] == {"verdict": "NOT TESTED", "net": None}
