"""RE-COST (#181, QD) QA pins, ported from QA Tester 2's master. Black-box where the code allows: the oracles are
the engine's own journal and plain arithmetic, not the build's helpers.

Source: /mnt/project-files/sleeve-fund/quant-review/re-cost-xfails/test_re_cost_xfails.py
Findings: RE-COST DW1-DW4 and HOLD1-HOLD3 (#181), all fixed on main, so these cells carry no xfail mark.

Done-when (HoQA 01:39 + Advisor 01:04/7 Oct + HoE 01:40):
  DW1 the benchmark and the run use one fee basis: a perp study's benchmark cost per side = the perp taker fee + half spread
  DW2 a perp study pays the same fee in the run and in the benchmark's strategy_return (recomputed from the journal)
  DW3 the fee note shows the fee actually paid and the break-even fee
  DW4 the cost ladder shows the random-entry return at each rung's own fee (+ spread + slippage)
  HOLD1 the headline buy and hold on a perp is a 1x long perp: its fees and its funding (not the spot hold)
  HOLD2 spot stays as a secondary line, labelled "holding spot instead"
  HOLD3 the perp line covers only the span with funding history, labelled; never zero-filled
"""

import copy
import json
import shutil
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.data import synthetic_ohlcv
from sleeve_fund.research import random_entry as RE
from sleeve_fund.research import study as S
from sleeve_fund.research.ledger import IdeaLedger
from sleeve_fund.research.study import run_study
from sleeve_fund.research.tearsheet import render
from sleeve_fund.strategies.trend_filter import SPEC

PERP_TAKER = 0.0005  # the simulated low-fee perp market's (markets.LOW_FEE_PERP)
PERP_FUNDING = 0.0001  # per settlement, 3 a day
SPREAD = 0.0003


def _need(obj, name, what):
    assert hasattr(obj, name), f"not built: {what}"
    return getattr(obj, name)


def _spec(market):
    if market == "spot":
        return SPEC
    return replace(SPEC, param_grid={k: v[:2] for k, v in SPEC.param_grid.items()} | {"market": [market]},
                   default_params={**SPEC.default_params, "market": market})


# A study runs for about 90 s, and most cells run the same one, so a run is shared by the cells whose study inputs are
# all equal (HoQA 7 Oct 00:06 UK; the PR lists who shares). The key is every input: the spec's market, the prices
# (their hash), the instrument, the spread, the windows and any other argument; a run with its fee spy (run_backtest
# patched) is never shared. Each cell gets its own deep copy of the result and of what the Spy recorded, and its own
# copy of the run's idea ledger at tmp_path / "l", so nothing one cell does reaches another. DW4's spy cell keeps its
# own run (UNCACHED).
_RUNS: dict[tuple, tuple] = {}
UNCACHED = ("test_dw4_each_ladder_rung_carries_the_random_entry_return_at_its_own_cost",)
_CELL = {"name": None, "spy": None}


@pytest.fixture(autouse=True)
def _cell(request):
    _CELL["name"], _CELL["spy"] = request.node.originalname, None
    yield
    _CELL["name"], _CELL["spy"] = None, None


def _shared(key, tmp_path, run):
    """run() with tmp_path's ledger, or a copy of the run an earlier cell made with the same key."""
    spy = _CELL["spy"]
    if _CELL["name"] in UNCACHED or getattr(S.run_backtest, "__name__", "") == "<lambda>":
        return run()
    hit = _RUNS.get(key)
    if hit is None or (spy is not None and hit[1] is None):
        result = run()
        _RUNS[key] = (copy.deepcopy(result), copy.deepcopy((spy.calls, spy.draw_holds)) if spy else None,
                      tmp_path / "l")
        return result
    result, recorded, ledger = hit
    if spy is not None:
        spy.calls, spy.draw_holds = copy.deepcopy(recorded)
    if ledger.exists() and not (tmp_path / "l").exists():
        tmp_path.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if ledger.is_dir() else shutil.copy2)(ledger, tmp_path / "l")
    return copy.deepcopy(result)


def _prices_key(prices):
    return (len(prices), str(prices.index[0]), str(prices.index[-1]), int(pd.util.hash_pandas_object(prices).sum()))


class Spy:
    """Records every random_entry call (closes, trades, windows, cost) and the draws' holds, then calls the real one."""

    def __init__(self, monkeypatch):
        self.calls, self.draw_holds = [], []
        _CELL["spy"] = self
        real_re, real_draw = S.random_entry, RE._random_entries

        def re_(closes, trades, windows, cost, *a, **k):
            self.draw_holds.append([])
            out = real_re(closes, trades, windows, cost, *a, **k)
            self.calls.append({"closes": np.asarray(closes, dtype=float), "trades": list(trades), "windows": list(windows),
                               "cost": cost, "kw": k, "res": out, "draws": self.draw_holds[-1]})
            return out

        def draw(rng, start, end, holds):
            order, entries = real_draw(rng, start, end, holds)
            if self.draw_holds:
                self.draw_holds[-1].append((start, end, sorted(int(h) for h in holds)))
            return order, entries

        monkeypatch.setattr(S, "random_entry", re_)
        monkeypatch.setattr(RE, "_random_entries", draw)


def _study(tmp_path, instrument, market="perp", prices=None, spread=SPREAD, seed=3, days=1100, **kw):
    prices = synthetic_ohlcv(days=days, seed=seed) if prices is None else prices
    key = ("study", market, _prices_key(prices), repr(instrument), spread, 0, 365, 365, repr(sorted(kw.items())))
    return _shared(key, tmp_path, lambda: run_study(
        _spec(market), prices, instrument, dataset="syn", ledger=IdeaLedger(tmp_path / "l"), synthetic=True,
        holdout_days=0, train_days=365, test_days=365, half_spread=spread, **kw))


# ---------------------------------------------------------------------------------------------------------- DW1
@pytest.mark.parametrize("market, taker", [("perp", PERP_TAKER)], ids=["perp"])
def test_dw1_the_benchmark_costs_the_fee_the_run_pays_plus_half_the_spread(tmp_path, instrument, monkeypatch, market, taker):
    """RE-COST DW1 (#181): the benchmark used to charge the instrument's spot taker fee on a perp study, while the run
    paid the perp market's 0.05%. Source: quant-review/re-cost-xfails/test_re_cost_xfails.py."""
    spy = Spy(monkeypatch)
    _study(tmp_path, instrument, market)
    assert spy.calls, "set-up: the study reached its benchmark"
    want = (float(instrument.taker_fee) if taker is None else taker) + SPREAD
    assert abs(spy.calls[0]["cost"] - want) < 1e-12, (market, spy.calls[0]["cost"], want)


# ---------------------------------------------------------------------------------------------------------- DW2
def _runs_fee_rates(monkeypatch):
    """Σ fees_paid / Σ(filled_qty x avg_px) of every run the study makes on the perp market with its own fees."""
    runs = []
    real = S.run_backtest
    monkeypatch.setattr(S, "run_backtest", lambda *a, **k: runs.append((a, k, r := real(*a, **k))) or r)
    return runs


def _rate(res):
    q = [float(x) for x in res.fills["filled_qty"]]
    px = [float(x) for x in res.fills["avg_px"]]
    return float(res.fees_paid) / sum(a * b for a, b in zip(q, px))


def test_dw2_the_runs_fee_rate_is_the_benchmarks_cost_with_no_spread(tmp_path, instrument, monkeypatch):
    """RE-COST DW2 (#181): the benchmark's cost per side must be the fee the run's own fills pay.
    Source: quant-review/re-cost-xfails/test_re_cost_xfails.py."""
    spy = Spy(monkeypatch)
    runs = _runs_fee_rates(monkeypatch)
    _study(tmp_path, instrument, "perp", spread=0.0)
    paid = [res for a, k, res in runs if (a[3] or {}).get("market") == "perp" and k.get("fees") is None
            and not res.fills.empty]
    assert paid, "set-up: perp runs that traded"
    for res in paid:
        assert abs(_rate(res) - PERP_TAKER) < 3e-5, ("a run pays the perp taker fee", _rate(res))
    assert abs(spy.calls[0]["cost"] - _rate(paid[0])) < 3e-5, ("and so must the benchmark", spy.calls[0]["cost"])


def test_dw2_the_benchmarks_strategy_return_recomputes_from_the_perp_fee(tmp_path, instrument, monkeypatch):
    """RE-COST DW2 (#181): the benchmark's strategy_return was charged the wrong fee, so it did not recompute.
    Source: quant-review/re-cost-xfails/test_re_cost_xfails.py."""
    spy = Spy(monkeypatch)
    _study(tmp_path, instrument, "perp")
    c = spy.calls[0]
    per_trade = [t.side * (c["closes"][t.exit] / c["closes"][t.entry] - 1) - 2 * (PERP_TAKER + SPREAD) for t in c["trades"]]
    assert per_trade, "set-up: trades reached the benchmark"
    want = float(np.prod(1 + np.array(per_trade)) - 1)
    assert abs(c["res"].strategy_return - want) < 1e-9, (c["res"].strategy_return, want)


# ---------------------------------------------------------------------------------------------------------- DW3
def test_dw3_the_fee_note_shows_the_fee_paid_and_the_break_even_fee(tmp_path, instrument):
    """RE-COST DW3 (#181): the fee note showed the instrument's spot fee, not the one the runs paid, and no break-even.
    Source: quant-review/re-cost-xfails/test_re_cost_xfails.py."""
    r = _study(tmp_path, instrument, "perp")
    note = r.fee_note
    assert f"{PERP_TAKER:.2%} taker" in note, note
    assert f"{float(instrument.taker_fee):.2%} taker" not in note, ("the spot fee is not what was paid", note)
    assert "break-even" in note.lower(), note
    assert str(r.breakeven) in note, ("the break-even fee itself, not just the word", r.breakeven, note)


# ---------------------------------------------------------------------------------------------------------- DW4
def test_dw4_each_ladder_rung_carries_the_random_entry_return_at_its_own_cost(tmp_path, instrument, monkeypatch):
    """RE-COST DW4 (#181): the cost ladder showed no random-entry return at each rung's own fee.
    Source: quant-review/re-cost-xfails/test_re_cost_xfails.py."""
    spy = Spy(monkeypatch)
    r = _study(tmp_path, instrument, "perp")
    assert r.cost_ladder, "set-up: a ladder"
    head = spy.calls[0]
    slip = r.ladder_slippage
    for rung in r.cost_ladder:
        got = _need(rung, "random_return", "LadderRung.random_return (the random-entry median at the rung's fee)")
        assert got is not None, ("not built: no random-entry return for", rung.fee)
        cost = rung.fee + SPREAD + slip
        again = RE.random_entry(head["closes"], head["trades"], head["windows"], cost, **head["kw"])
        assert got == pytest.approx(again.median_random_return, abs=1e-12), (rung.fee, got, again.median_random_return)
        mine = _need(rung, "oos_timing_return", "LadderRung.oos_timing_return (the strategy's trips at the rung's fee)")
        assert mine == pytest.approx(again.strategy_return, abs=1e-12), (rung.fee, mine, again.strategy_return)
    fees = [rung.fee for rung in r.cost_ladder]
    rand = [rung.random_return for rung in r.cost_ladder]
    assert fees == sorted(fees) and rand == sorted(rand, reverse=True) and rand[0] > rand[-1], (fees, rand)


def test_dw4_the_rendered_ladder_shows_the_random_entry_return(tmp_path, instrument):
    """RE-COST DW4 (#181): the rendered ladder did not show the random-entry return.
    Source: quant-review/re-cost-xfails/test_re_cost_xfails.py."""
    r = _study(tmp_path, instrument, "perp")
    text = render(r, IdeaLedger(tmp_path / "l")).lower()
    assert "random" in text.split("cost ladder", 1)[-1] if "cost ladder" in text else False, "not built: no ladder text"


# --------------------------------------------------------------------------------------------------- perp-hold lines
def test_hold1_the_headline_hold_on_a_perp_study_pays_perp_funding(tmp_path, instrument):
    """The perp hold's out-of-sample return over a fold is the price move less the funding a long pays (0.01% x 3 a day)
    less the 0.05% opening fee, to 5% on the growth factor; the spot hold has no funding.

    RE-COST HOLD1 (#181): the headline buy and hold on a perp study was the spot hold: no funding, the spot venue's
    fee. Source: quant-review/re-cost-xfails/test_re_cost_xfails.py.
    """
    prices = synthetic_ohlcv(days=1100, seed=3)
    r = _study(tmp_path, instrument, "perp", prices=prices)
    bench = r.oos_benchmark_returns
    assert len(bench) > 300, "set-up: a benchmark over the out-of-sample days"
    days = len(bench)
    perp_hold = float((1 + bench).prod() - 1)
    day_close = prices["close"].resample("1D").last()
    daily = day_close.pct_change().reindex(bench.index)
    assert daily.notna().all(), "set-up: a price move for every benchmark day"
    move = float((1 + daily).prod() - 1)
    # funding is charged on the position's CURRENT value, so as a share of equity it is 0.03% a day on every day
    perp_like = float((1 + daily - 3 * PERP_FUNDING).prod() * (1 - PERP_TAKER) - 1)
    spot_like = (1 + move) * (1 - float(instrument.taker_fee)) - 1  # what the spot hold gives
    assert abs(perp_hold - perp_like) < abs(perp_hold - spot_like), {"perp_hold": perp_hold, "perp_like": perp_like,
                                                                    "spot_like": spot_like, "move": move}
    # the engine sizes the hold slightly under full equity, so allow 5% on the growth factor (spot would be 19% off)
    assert abs((1 + perp_hold) / (1 + perp_like) - 1) < 0.05, {"perp_hold": perp_hold, "perp_like": perp_like}


def test_hold2_spot_is_a_secondary_line_labelled_holding_spot_instead(tmp_path, instrument):
    """RE-COST HOLD2 (#181): the spot hold was not shown beside the perp hold, or was not labelled.
    Source: quant-review/re-cost-xfails/test_re_cost_xfails.py."""
    r = _study(tmp_path, instrument, "perp")
    spot = _need(r, "spot_hold_returns", "StudyResult.spot_hold_returns (the out-of-sample spot hold)")
    assert spot is not None and len(spot) == len(r.oos_returns), "not built: a spot line over the same days"
    text = render(r, IdeaLedger(tmp_path / "l"))
    assert "holding spot instead" in text.lower(), "not built: the label"
    # the headline numbers must stay the perp hold, not be replaced by the spot one
    assert not np.allclose(r.oos_benchmark_returns.to_numpy(), spot.to_numpy())


def _native(tmp_path, monkeypatch, kept_from, days=1100):
    """The venue's own perpetual with its settled funding (every 8 hours, 0.01%) kept from `kept_from` on."""
    from sleeve_fund import funding
    from sleeve_fund.venues import venue

    prices = synthetic_ohlcv(days=days, seed=3)
    idx = prices.index.tz_localize("UTC") if prices.index.tz is None else prices.index
    due = pd.date_range(idx[0].ceil("8h"), idx[-1], freq="8h")
    d = tmp_path / "BINANCE" / "BTC-USDT"
    d.mkdir(parents=True)
    (d / "funding.json").write_text(json.dumps({"rates": [[int(t.timestamp() * 1000), 0.0001] for t in due
                                                          if t >= kept_from]}))
    monkeypatch.setattr(funding, "DEFAULT_ROOT", tmp_path)
    funding._cache.clear()
    return prices, venue("BINANCE").instrument("BTC", "USDT")


def _native_study(tmp_path, monkeypatch, kept_from):
    prices, inst = _native(tmp_path, monkeypatch, kept_from)
    key = ("native", _prices_key(prices), repr(inst), str(kept_from), 0, 365, 365)
    r = _shared(key, tmp_path, lambda: run_study(
        _spec("perp"), prices, inst, dataset="syn", ledger=IdeaLedger(tmp_path / "l"), synthetic=True,
        holdout_days=0, train_days=365, test_days=365))
    return r, prices


def test_hold3_a_window_with_a_missing_settlement_is_left_out_never_zero_filled(tmp_path, monkeypatch):
    """RE-COST HOLD3 (#181): a window with funding missing was zero-filled or priced; it must be left out, and
    labelled. Source: quant-review/re-cost-xfails/test_re_cost_xfails.py."""
    # Funding known only from a month INTO the last fold's window: that fold has a missing settlement, earlier folds too.
    full, prices = _native_study(tmp_path / "a", monkeypatch, pd.Timestamp("1900-01-01", tz="UTC"))
    last = full.folds[-1]
    kept = pd.Timestamp(last.test_end) - pd.Timedelta(days=200)
    kept = kept.tz_localize("UTC") if kept.tzinfo is None else kept
    part, _ = _native_study(tmp_path / "b", monkeypatch, kept)
    benchmark = part.oos_benchmark_returns
    assert len(benchmark) < len(part.oos_returns), "the uncovered days must leave the comparison"
    assert benchmark.index.isin(part.oos_returns.index).all()
    zero = int((benchmark == 0.0).sum())
    assert zero == int((full.oos_benchmark_returns == 0.0).sum()), ("a zero-filled day appeared", zero)
    assert all(f.benchmark_test is None for f in part.folds[:-1]), "a fold whose window lacks funding has no hold line"
    text = render(part, IdeaLedger(tmp_path / "b" / "l"))
    assert "out-of-sample days" in text, "not built: the covered span is labelled"
    assert str(len(benchmark)) in text and str(len(part.oos_returns)) in text, "the label names covered and total days"


def test_hold3_under_half_the_days_covered_is_no_verdict_not_a_pass(tmp_path, monkeypatch):
    """RE-COST HOLD3 (#181): with under half the out-of-sample days covered the hold must give no verdict.
    Source: quant-review/re-cost-xfails/test_re_cost_xfails.py."""
    full, _ = _native_study(tmp_path / "a", monkeypatch, pd.Timestamp("1900-01-01", tz="UTC"))
    last = full.folds[-1]
    kept = pd.Timestamp(last.test_end) - pd.Timedelta(days=100)
    kept = kept.tz_localize("UTC") if kept.tzinfo is None else kept
    part, _ = _native_study(tmp_path / "b", monkeypatch, kept)
    insufficient = _need(part, "hold_insufficient", "StudyResult.hold_insufficient")
    assert insufficient is True, "not built: under 50% coverage"
    from sleeve_fund.research.tearsheet import SHARPE_CHECK, g1_checks

    checks = {n: (v, ev) for n, v, ev in g1_checks(part, IdeaLedger(tmp_path / "b" / "l"))}
    assert checks[SHARPE_CHECK][0] == "N/A", checks[SHARPE_CHECK]
