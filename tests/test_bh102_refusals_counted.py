"""BH-102 (HoE, from HoQA's bisect, 21 Oct): research makes the stop-share refusal visible. A backtest with no runtime
refuses an entry whose stop is past its share of the way to the liquidation price as paper does, but only paper
journals it, so a run could show zero trades and say nothing. The result counts the refused entry signals, as it
counts stops clamped, and the sheet says so. A 1x short keeps the rule (a fully paid 1x long is exempt, Advisor (A))."""
from __future__ import annotations

import dataclasses

import pandas as pd
import test_hub_146_qa as qa
from test_hub_146_qa import _probe, flat_prices  # noqa: F401 - _probe registers the probe
from test_r04_short_leg_sheet import SHORT, _line, perp_venue  # noqa: F401 - the fixture

from sleeve_fund.instruments import BOOK_SHARE
from sleeve_fund.research import tearsheet
from sleeve_fund.research.ledger import IdeaLedger
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.research.study import run_study
from sleeve_fund.strategies.definitions import to_params
from sleeve_fund.strategies.rules import SPEC
from sleeve_fund.venues import venue


def _probe_run(side: int, stop: float):
    m = qa.minutes_of(flat_prices(40))
    bars = m.set_axis(m.index + pd.Timedelta(minutes=1))
    bars["volume"] = 60 / BOOK_SHARE
    inst = venue("KRAKEN").instrument("BTC", "USD", price_precision=1)
    params = {"enter": 5, "leave": 35, "side": side, "stop_loss": stop, "market": "perp", "allow_short": True}
    return run_backtest("probe", bars, inst, params=params, starting_capital=10_000, risk_profile="conservative",
                        bar_minutes=1, half_spread=0)


def test_a_1x_short_refused_at_a_wide_stop_is_counted():
    """A 1x short's liquidation is about 99% up; with no runtime the share is half, so a 50% stop (the largest taken) is past it. Nothing
    trades, and the result says why. The same stop on a 1x long trades and counts nothing."""
    short = _probe_run(-1, 0.50)
    assert short.fills is None or len(short.fills) == 0
    assert short.entries_refused_liquidation > 0
    long_ = _probe_run(1, 0.50)
    assert len(long_.fills) > 0 and long_.entries_refused_liquidation == 0


def test_the_sheet_says_how_many_entries_the_rule_refused(tmp_path, perp_venue):
    """The R-04 short leg on the perpetual with a 50% stop: past half the way to its liquidation about 99% up."""
    inst, bars = perp_venue
    params = {**to_params(SHORT), "market": "perp", "allow_short": True, "stop_loss": 0.50}
    spec = dataclasses.replace(SPEC, param_grid={k: [v] for k, v in params.items()})
    ledger = IdeaLedger(tmp_path / "l.jsonl")
    r = run_study(spec, bars, inst, dataset="bh102", ledger=ledger, default_params=params, synthetic=True,
                  holdout_days=0, train_days=730, test_days=365)
    sheet = tearsheet.render(r, ledger)
    n = r.full_period.entries_refused_liquidation
    assert n > 0, "set-up: the short leg's entries are refused"
    line = _line(sheet, "- Entries refused:")
    assert line.startswith(f"- Entries refused: {n} entr"), line
    assert "liquidation price" in line
