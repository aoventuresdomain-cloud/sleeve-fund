"""The multi-day batch's count list (Researcher 18 Nov, idea-batch-18nov-b.md). M1 to M3 are the Researcher's files
byte for byte, counted on 2024 only with the edge pre-checks, no try. Figures in fixtures are synthetic."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pytest

from sleeve_fund.research import rescore, rounds
from sleeve_fund.research.variants import BATCH_VOL, PREREGISTERED, definition_path
from sleeve_fund.strategies.definitions import (
    check_definition,
    load_definition_file,
    to_params,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_signal_count import PAIRS, _bars, _History

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "configs" / "studies"
M_COUNT = STUDIES / "qrm-batch-count-list.md"
M_FROZEN = {"qrm1_tsmom_20d": "fe3df2f265b8f7858d877c885146942989d8ab030a25f461b4fab97f174517b1",
            "qrm2_calm_regime_breakout": "b20a747b0f71088bec43ad7e806d8fe59384820ba8d663d66ee5902bc369d775",
            "qrm3_crowd_squeeze_breakout": "278602bfea87106354cb8820f4d8efcc0dface967b09da11add28202fc4960c7"}


@pytest.mark.parametrize("name", M_FROZEN)
def test_each_m_definition_is_the_researchers_file_and_checks_on_4h(name):
    assert hashlib.sha256(definition_path(name).read_bytes()).hexdigest() == M_FROZEN[name]
    defn = load_definition_file(definition_path(name))
    check_definition(defn, bar_minutes=240)
    assert "sizing" not in defn
    assert PREREGISTERED[name.upper()] == {"definition": name, "params": {"vol_target_size": BATCH_VOL}}


def test_the_syntax_the_hoe_asked_after_reads_as_written():
    """A 20-day lag of the daily close (M1), a one-day lag of the daily channel's midpoint (M3) and the funding
    percentile on 4h (M3, studies only)."""
    m1 = load_definition_file(definition_path("qrm1_tsmom_20d"))["blocks"]
    assert any(b["kind"] == "lag" and b.get("timeframe") == "1d" and b["period"] == 20 and b["input"] == "close"
               for b in m1.values())
    m3 = load_definition_file(definition_path("qrm3_crowd_squeeze_breakout"))
    assert any(b["kind"] == "funding_percentile" for b in m3["blocks"].values())
    from sleeve_fund.strategies import study_only_uses

    assert study_only_uses(to_params(m3))  # refused by paper and live


def test_m2s_regime_pair_is_the_simple_average_atr_so_it_decides_inside_the_train_year():
    """A Wilder daily ATR(50) needs 1,000 daily candles to settle, longer than the data before the test window: the
    Researcher's pre-run fix makes the regime pair simple averages, which settle in their period."""
    blocks = load_definition_file(definition_path("qrm2_calm_regime_breakout"))["blocks"]
    assert blocks["atr5"]["kind"] == blocks["atr50"]["kind"] == "atr_sma"
    assert blocks["atr50"]["period"] == 50 and blocks["atr50"]["timeframe"] == "1d"


def test_the_m_count_list_records_no_try_and_counts_on_2024_only(capfd):
    text = M_COUNT.read_text()
    names = ["qrm1_tsmom_20d", "qrm2_calm_regime_breakout", "qrm3_crowd_squeeze_breakout"]
    assert rescore.rows_of(text) == [(i, n, 240) for i, n in enumerate(names, 1)]
    assert rescore.confirming_of(text) == {"fund_guards": False, "size_scale": 1.0}
    assert rounds.round_of(text, [1, 2, 3]) is None
    history = _History({p: _bars(seed=i) for i, p in enumerate(PAIRS.split(","))})

    def no_run(*a, **k):
        raise AssertionError("a signal count runs no study")

    assert rescore.main([str(M_COUNT), "--venue", "binance", "--pair", PAIRS, "--count-signals"], run=no_run,
                        journal=lambda: None, history=history) == 0
    out = capfd.readouterr().out
    assert "FAILED" not in out, out
    for i, name in enumerate(names, 1):
        assert f"row {i:>2} {name}: train window 2024-01-01 to 2024-12-31 (366 days)" in out
        assert f"row {i:>2} {name}: pre-check pooled edge before fees at least 25 bps a trade: " in out
        assert f"row {i:>2} {name}: sizing to 40% a year over 30 days" in out
    m1 = next(line for line in out.splitlines() if line.startswith("row  1 ") and "in all" in line)
    assert int(re.search(r"; ([\d,]+) in all", m1).group(1).replace(",", "")) > 0
    assert "dry run: 3 checked, 0 run, train-window signals counted (no try)" in out
