"""B-22 (Advisor B22-DOUBLE-FEES): profitable at double fees. The study re-runs each out-of-sample window, with the
settings that window chose, at exactly twice the venue's maker and taker fees, spread, slippage and funding as they
were; never the holdout. Pass: net above 0 over the out-of-sample period. Stored on the result for the page, and said
on the tear sheet in the line the page reads.

Expected values are worked from the runs the study made, captured as they ran, never read back from its own figures."""

import re
from decimal import Decimal

import pandas as pd
import pytest

from sleeve_fund.data import synthetic_ohlcv
from sleeve_fund.research import study as study_mod
from sleeve_fund.research.ledger import IdeaLedger
from sleeve_fund.research.study import run_study
from sleeve_fund.research.tearsheet import render
from sleeve_fund.strategies.trend_filter import SPEC

# The line the page parses (FE2's development.double_fees, _DOUBLE_FEES)
LINE = re.compile(r"^\*\*Profitable at double fees:\*\* (PASS|FAIL), net ([+-]\d+\.\d)% over the out-of-sample "
                  r"period at 2x the venue's fees \(", re.M)
KW = dict(dataset="syn", synthetic=True, holdout_days=200, train_days=730, test_days=365)


@pytest.fixture(scope="module")
def ran(tmp_path_factory, instrument):
    """One study, every engine run it makes captured (its fees, spread and window), with the holdout opened."""
    calls, real = [], study_mod.run_backtest

    def spy(name, df, inst, params, **k):
        res = real(name, df, inst, params, **k)
        calls.append({"df": df, "params": params, "fees": k.get("fees"), "half_spread": k.get("half_spread"),
                      "trade_from": params.get("trade_from"), "res": res})
        return res

    mp = pytest.MonkeyPatch()
    mp.setattr(study_mod, "run_backtest", spy)
    try:
        prices = synthetic_ohlcv(days=1900, seed=3)
        ledger = IdeaLedger(tmp_path_factory.mktemp("b22") / "l.jsonl")
        r = run_study(SPEC, prices, instrument, ledger=ledger, use_holdout=True, **KW)
    finally:
        mp.undo()
    return r, calls, render(r, ledger)


def _windows(r, calls):
    """The out-of-sample window runs (trading from a test window's first bar; the holdout's ends past the research
    period), by window start: 1x and 2x."""
    out = {}
    for c in calls:
        if c["trade_from"] is None or c["df"].index[-1] > r.research_end:
            continue
        out.setdefault(c["trade_from"], []).append(c)
    return out


def test_b22a_the_result_is_stored_and_passes_only_on_net_above_zero(ran):
    r, _, _ = ran
    d = r.double_fees
    assert d is not None and set(d) >= {"passed", "net_return", "base_net_return", "maker", "taker", "oos_start",
                                        "oos_end", "oos_days", "round_trips", "base_round_trips", "trades_differ"}
    assert d["passed"] is (d["net_return"] > 0)
    assert d["maker"] == 2 * r.fee_basis.maker and d["taker"] == 2 * r.fee_basis.taker  # exactly twice, both


def test_b22b_each_window_is_re_run_at_twice_its_fees_with_the_same_spread_and_settings(ran):
    r, calls, _ = ran
    windows = _windows(r, calls)
    chose = [f for f in r.folds if not f.unscored]
    assert chose and len(windows) == len(chose)
    for runs in windows.values():
        base = [c for c in runs if c["fees"] is None]
        twice = [c for c in runs if c["fees"] is not None]
        assert len(base) >= 1 and len(twice) == 1, [c["fees"] for c in runs]
        b, t = base[0], twice[0]
        assert t["params"] == b["params"] and t["df"].index.equals(b["df"].index)  # same settings, same window
        assert t["half_spread"] == b["half_spread"]  # spread and slippage as they were
        paid = r.fee_basis
        assert (t["fees"].maker, t["fees"].taker) == (paid.maker * 2, paid.taker * 2)


def test_b22c_a_real_re_run_the_double_fee_trades_differ_from_the_base_ones(ran):
    """Fee-sensitive: the strategy sizes from its cash, so paying more leaves less to buy with and every later
    trade's size moves. A subtraction of fees from the 1x trades would leave them as they were."""
    r, calls, _ = ran
    differ = []
    for runs in _windows(r, calls).values():
        b = next(c for c in runs if c["fees"] is None)["res"].fills
        t = next(c for c in runs if c["fees"] is not None)["res"].fills
        assert len(b) and len(t)
        differ.append(not (len(b) == len(t) and (b["filled_qty"].to_numpy() == t["filled_qty"].to_numpy()).all()))
    assert any(differ) and r.double_fees["trades_differ"] is True


def test_b22d_its_net_is_the_out_of_sample_period_compounded_never_the_holdout(ran):
    r, _, _ = ran
    d = r.double_fees
    assert r.holdout is not None  # the holdout was opened in this study, and is not in the figure
    twice = r.double_fee_returns
    assert twice.index.equals(r.oos_returns.index)
    assert twice.index.max() <= r.research_end
    assert (d["oos_start"], d["oos_end"], d["oos_days"]) == (twice.index[0], twice.index[-1], len(twice))
    assert d["net_return"] == pytest.approx(float((1 + twice).prod() - 1), abs=1e-12)
    assert d["base_net_return"] == pytest.approx(float((1 + r.oos_returns).prod() - 1), abs=1e-12)
    assert d["net_return"] < d["base_net_return"]  # it traded, so twice the fees cost more


def test_b22e_the_tear_sheet_says_it_in_the_line_the_page_reads(ran):
    r, _, sheet = ran
    m = LINE.findall(sheet)
    assert len(m) == 1, [ln for ln in sheet.splitlines() if "double fees" in ln]
    verdict, net = m[0]
    assert verdict == ("PASS" if r.double_fees["passed"] else "FAIL")
    assert float(net) == pytest.approx(round(r.double_fees["net_return"] * 100, 1))


def test_b22f_the_stored_fields_survive_the_result_as_plain_values(ran):
    """What FE reads: plain numbers, words and timestamps, no engine objects."""
    r, _, _ = ran
    for k, v in r.double_fees.items():
        assert isinstance(v, (bool, int, float, str, Decimal, pd.Timestamp)), (k, type(v))


def test_b22g_double_fees_are_twice_the_schedule():
    from sleeve_fund.instruments import FeeSchedule

    assert study_mod.doubled(FeeSchedule(Decimal("0.0002"), Decimal("0.00055"))) == FeeSchedule(Decimal("0.0004"),
                                                                                                 Decimal("0.0011"))


@pytest.mark.parametrize("daily, passed", [([0.01, -0.02], False), ([0.0, 0.0], False), ([0.01, 0.0], True)])
def test_b22h_it_passes_only_above_zero_net(daily, passed, monkeypatch):
    """Hand figures: +1% then -2% is -1.02% net; flat is 0, not above it; +1% is +1%."""
    from types import SimpleNamespace

    monkeypatch.setattr(study_mod, "round_trips", lambda fills, shorts: [0.1])

    from sleeve_fund.instruments import FeeSchedule

    days = pd.date_range("2024-01-01", periods=2, tz="UTC")
    fills = pd.DataFrame({"filled_qty": [1.0], "avg_px": [100.0]})
    run = SimpleNamespace(fills=fills, shorts=None)
    d = study_mod._double_fees(pd.Series([0.02, 0.0], index=days), pd.Series(daily, index=days), [(run, run)],
                               FeeSchedule(Decimal("0.0004"), Decimal("0.0011")))
    want = (1 + daily[0]) * (1 + daily[1]) - 1
    assert d["passed"] is passed and d["net_return"] == pytest.approx(want, abs=1e-15)
    assert d["base_net_return"] == pytest.approx(0.02) and d["trades_differ"] is False
    assert study_mod._double_fees(pd.Series(dtype=float), pd.Series(dtype=float), [], FeeSchedule(
        Decimal(0), Decimal(0))) is None


def test_b22i_the_fields_are_stored_beside_the_tear_sheet_for_the_page(ran, tmp_path):
    """The stored result FE reads as fields: <sheet>.results.json, {"double_fees": {...}}, fees as decimal strings and
    times in ISO 8601, every field the study keeps."""
    import json

    from sleeve_fund.research.run import results_path, write_results

    r, _, _ = ran
    sheet = tmp_path / "trend_filter_syn_20261020-170000.md"
    path = write_results(r, sheet)
    assert path == results_path(sheet) == tmp_path / "trend_filter_syn_20261020-170000.results.json"
    got = json.loads(path.read_text())["double_fees"]
    d = r.double_fees
    assert set(got) == set(d)
    assert got["passed"] is d["passed"] and got["net_return"] == d["net_return"]
    assert Decimal(got["maker"]) == d["maker"] and Decimal(got["taker"]) == d["taker"]
    assert pd.Timestamp(got["oos_start"]) == d["oos_start"] and pd.Timestamp(got["oos_end"]) == d["oos_end"]
    assert (got["oos_days"], got["round_trips"], got["trades_differ"]) == (d["oos_days"], d["round_trips"],
                                                                          d["trades_differ"])


def test_b22k_a_window_that_never_traded_is_the_same_trades_at_any_fee():
    """A window whose settings never traded has no fills at either fee (an empty frame, no columns): same trades, no
    round trips, and the study goes on (it raised KeyError: filled_qty)."""
    from types import SimpleNamespace

    from sleeve_fund.instruments import FeeSchedule

    days = pd.date_range("2024-01-01", periods=2, tz="UTC")
    flat = SimpleNamespace(fills=pd.DataFrame(), shorts=None)
    d = study_mod._double_fees(pd.Series([0.0, 0.0], index=days), pd.Series([0.0, 0.0], index=days), [(flat, flat)],
                               FeeSchedule(Decimal("0.0004"), Decimal("0.0011")))
    assert d["trades_differ"] is False and d["round_trips"] == d["base_round_trips"] == 0 and d["passed"] is False


def test_b22l_the_re_runs_add_nothing_to_g1s_first_touch_count(instrument, monkeypatch, tmp_path):
    """The first-touch share G1 reads is counted from each window's own run, once: the double-fee re-runs count
    nothing (they are re-scored, never re-judged)."""
    counted, real = [], study_mod.run_backtest

    def spy(*a, first_touch_count_from=None, **kw):
        if first_touch_count_from is not None:
            counted.append(first_touch_count_from)
        return real(*a, first_touch_count_from=first_touch_count_from, **kw)

    monkeypatch.setattr(study_mod, "run_backtest", spy)
    r = run_study(SPEC, synthetic_ohlcv(days=1200, seed=3), instrument, ledger=IdeaLedger(tmp_path / "l.jsonl"),
                  dataset="syn", synthetic=True, holdout_days=0, train_days=365, test_days=365)
    assert r.double_fees is not None and len(counted) == len(r.folds) == len(set(counted))


def test_b22m_judged_on_the_opposite_resolution_the_double_fee_run_is_flipped_too(instrument, monkeypatch, tmp_path):
    """Advisor (TEC-B22-FLIPS): when G1 judges the opposite resolution, the 2x run is flipped as well, and each
    window's flipped pair differs only in its fees, 1x then exactly 2x: same bars, settings, spread and window."""
    from test_r2_g1 import _stats

    flipped, real = [], study_mod.run_backtest

    def fake(name, df, inst, params, *, first_touch_flip=False, **kw):
        res = real(name, df, inst, params, first_touch_flip=first_touch_flip, **kw)
        if name != "buy_and_hold":
            res.first_touch = _stats(51, flip=first_touch_flip)
            if first_touch_flip and len(res.equity):
                res.equity = res.equity * pd.Series(0.9995, index=res.equity.index).cumprod()
                flipped.append({"df": df, "params": params, "fees": kw.get("fees"),
                                "half_spread": kw.get("half_spread")})
        return res

    monkeypatch.setattr(study_mod, "run_backtest", fake)
    r = run_study(SPEC, synthetic_ohlcv(days=1200, seed=3), instrument, ledger=IdeaLedger(tmp_path / "l.jsonl"),
                  dataset="syn", synthetic=True, holdout_days=0, train_days=365, test_days=365)
    assert r.first_touch["judged_on"] == "opposite"
    windows = [f for f in r.folds if not f.unscored]
    assert windows and len(flipped) == 2 * len(windows)
    pairs = {}
    for c in flipped:
        pairs.setdefault(c["params"]["trade_from"], []).append(c)
    assert len(pairs) == len(windows)
    paid = r.fee_basis
    for one, two in pairs.values():
        assert one["fees"] is None and (two["fees"].maker, two["fees"].taker) == (paid.maker * 2, paid.taker * 2)
        assert one["params"] == two["params"] and one["df"].index.equals(two["df"].index)
        assert one["half_spread"] == two["half_spread"]
