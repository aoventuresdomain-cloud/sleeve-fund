"""H improve row H1 + vol_target (Advisor 18 Nov 16:41) and the multi-day batch's count list (Researcher 18 Nov,
idea-batch-18nov-b.md). H1's definition runs unchanged, sized to the batch's volatility target on the study list, as a
new variant of its idea (a new basket key, never a re-run of H1's try), printed as family B try 30 off a stated prior
of 24. M1 and M3 are the Researcher's files byte for byte, counted on 2024 only with the gross-edge pre-checks, no
try. Figures in fixtures are synthetic."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sleeve_fund.research import basket as bk
from sleeve_fund.research import rescore, rounds
from sleeve_fund.research.pinned import parse
from sleeve_fund.research.variants import BATCH_VOL, PREREGISTERED, definition_path
from sleeve_fund.strategies.definitions import (
    check_definition,
    definition_hash,
    load_definition_file,
    to_params,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_signal_count import PAIRS, _bars, _History

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "configs" / "studies"
H1_VOL = STUDIES / "h1-vol-list.md"
M_COUNT = STUDIES / "qrm-batch-count-list.md"
M_FROZEN = {"qrm1_tsmom_20d": "fe3df2f265b8f7858d877c885146942989d8ab030a25f461b4fab97f174517b1",
            "qrm3_crowd_squeeze_breakout": "278602bfea87106354cb8820f4d8efcc0dface967b09da11add28202fc4960c7"}


def _counted_like_17nov(text: str) -> bool:
    return rescore.confirming_of(text) == rescore.confirming_of((STUDIES / "idea-batch-17nov-list.md").read_text())


def test_h1_vol_is_h1s_definition_unchanged_with_the_batch_volatility_target():
    variant = PREREGISTERED["QRH1_TREND_PULLBACK_TS6_VOL"]
    assert variant == {"definition": "qrh1_trend_pullback_ts6", "params": {"vol_target_size": BATCH_VOL}}
    pinned = parse(H1_VOL.read_text())
    plain = rescore.request("qrh1_trend_pullback_ts6", 240, "binance", PAIRS, pinned)
    sized = rescore.request("qrh1_trend_pullback_ts6_vol", 240, "binance", PAIRS, pinned)
    assert {k: v for k, v in sized.params.items() if k != "vol_target_size"} == plain.params
    assert sized.params["vol_target_size"] == BATCH_VOL and "vol_target_size" not in plain.params
    assert sized.strategy == plain.strategy and tuple(sized.pairs) == tuple(plain.pairs)
    sized.validate()


def test_h1_vol_is_a_new_variant_in_the_register_never_a_re_run_of_h1s_try():
    """A basket try is keyed by basket_key, which hashes its params: the sizing makes a new key, while the definition
    (and so its idea, and the family it counts in) is H1's."""
    pinned = parse(H1_VOL.read_text())
    keys = {}
    for name in ("qrh1_trend_pullback_ts6", "qrh1_trend_pullback_ts6_vol"):
        req = rescore.request(name, 240, "binance", PAIRS, pinned)
        b = SimpleNamespace(strategy=req.strategy, params=req.params, names=["BTC", "ETH", "SOL"],
                            assets=(SimpleNamespace(bar_minutes=240),))
        keys[name] = bk.basket_key(b)
        assert definition_hash(req.params["definition"]) == definition_hash(
            load_definition_file(definition_path("qrh1_trend_pullback_ts6")))
    assert keys["qrh1_trend_pullback_ts6"] != keys["qrh1_trend_pullback_ts6_vol"]


def test_the_h1_vol_list_is_a_counted_family_b_try_30_on_the_rounds_dates():
    text = H1_VOL.read_text()
    rows = rescore.rows_of(text)
    assert rows == [(1, "qrh1_trend_pullback_ts6_vol", 240)]
    assert parse(text) == parse((STUDIES / "next-round-pinned-dates-list.md").read_text())
    assert _counted_like_17nov(text) and "Size scale" not in text
    r = rounds.round_of(text, [1])
    assert r.name == "h1-vol" and r.families == {"B": (1,)} and r.prior == {"B": 24}
    # five H0-idea rows confirmed in the register (H0-H4): try 30; if H5 or H8 has run too, 31 or 32 (Advisor)
    line = rounds.try_line(r, 1, 5)
    assert "family B try 30 (24 stated before the round, 5 in the trials register before this one)" in line
    assert "family B try 31 " in rounds.try_line(r, 1, 6)
    assert "Prior 24 = slot 30 - 1 - 5 confirmed H0-idea rows (H0-H4); H5/H8 unconfirmed" in " ".join(text.split())


def test_the_h1_vol_list_dry_runs(capfd):
    assert rescore.main([str(H1_VOL), "--pair", PAIRS, "--dry-run"], run=lambda *a, **k: None,
                        journal=lambda: None) == 0
    assert "dry run: 1 checked, 0 run" in capfd.readouterr().out


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


def test_m2_is_not_listed_until_its_atr_settles_inside_the_train_year():
    assert "QRM2_CALM_REGIME_BREAKOUT" not in PREREGISTERED
    assert "qrm2" not in "".join(n for _, n, _ in rescore.rows_of(M_COUNT.read_text()))


def test_the_m_count_list_records_no_try_and_counts_on_2024_only(capfd):
    text = M_COUNT.read_text()
    assert rescore.rows_of(text) == [(1, "qrm1_tsmom_20d", 240), (2, "qrm3_crowd_squeeze_breakout", 240)]
    assert rescore.confirming_of(text) == {"fund_guards": False, "size_scale": 1.0}
    assert rounds.round_of(text, [1, 2]) is None
    history = _History({p: _bars(seed=i) for i, p in enumerate(PAIRS.split(","))})

    def no_run(*a, **k):
        raise AssertionError("a signal count runs no study and opens no journal")

    assert rescore.main([str(M_COUNT), "--venue", "binance", "--pair", PAIRS, "--count-signals"], run=no_run,
                        journal=no_run, history=history) == 0
    out = capfd.readouterr().out
    assert "FAILED" not in out, out
    for i, name in enumerate(["qrm1_tsmom_20d", "qrm3_crowd_squeeze_breakout"], 1):
        assert f"row {i:>2} {name}: train window 2024-01-01 to 2024-12-31 (366 days)" in out
        assert f"row {i:>2} {name}: pre-check pooled gross edge at least 15 bps a trade: " in out
        assert f"row {i:>2} {name}: sizing to 40% a year over 30 days" in out
    m1 = next(line for line in out.splitlines() if line.startswith("row  1 ") and "in all" in line)
    assert int(re.search(r"; ([\d,]+) in all", m1).group(1).replace(",", "")) > 0
    assert "dry run: 2 checked, 0 run, train-window signals counted (no try)" in out
