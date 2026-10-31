"""The next round (PM approved the plan 30 Oct 14:57 UTC; Advisor cleared both pre-registrations 30 Oct 15:07 UTC):
family A's funding fade and family B's range reversion as their pre-registrations write them, and the round's list on
the sealed dates (data from 2024-01-01, research to 2026-04-30, holdout 2026-05-01 to 2026-10-30, locked)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from sleeve_fund.research import rescore
from sleeve_fund.research.pinned import parse
from sleeve_fund.research.study import continuous_warmup
from sleeve_fund.research.variants import PREREGISTERED, definition_path
from sleeve_fund.strategies import study_only_uses
from sleeve_fund.strategies.definitions import check_definition, load_definition_file, to_params

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "configs" / "studies" / "next-round-pinned-dates-list.md"


def _defn(key: str) -> dict:
    return load_definition_file(definition_path(PREREGISTERED[key]["definition"]))


def test_family_a_fades_an_extreme_rank_gated_on_the_rate_and_is_studies_only():
    d = _defn("QRA0_FUNDING_FADE")
    check_definition(d, bar_minutes=240)
    assert d["blocks"]["fr"] == {"kind": "funding_percentile", "window_days": 90}
    assert d["long"]["entry"] == {"all": [{"left": "fr.pct", "op": "<=", "right": 10},
                                          {"left": "fr.rate", "op": "<", "right": 0}]}
    assert d["short"]["entry"] == {"all": [{"left": "fr.pct", "op": ">=", "right": 90},
                                           {"left": "fr.rate", "op": ">=", "right": 0.0003}]}
    assert d["long"]["exit"] == {"left": "fr.pct", "op": ">=", "right": 50}
    assert d["short"]["exit"] == {"left": "fr.pct", "op": "<=", "right": 50}
    for side, op in (("long", "sub"), ("short", "add")):
        assert d["exits"][side] == {"stop": {"level": {op: ["close", {"mul": [3.0, "atr"]}]}},
                                    "time_stop": {"minutes": 10080}}
    params = {**to_params(d, "BINANCE"), "market": "perp", "allow_short": True}
    assert study_only_uses(params)  # paper and live refuse it
    assert continuous_warmup("rules", params, 240) * 240 / 1440 < 91  # the 90-day window, inside 2024


def test_family_b_is_its_pre_registrations_toml_with_a_141_day_warm_up():
    d = _defn("QRB0_RANGE_REVERSION_ADX")
    check_definition(d, bar_minutes=60)
    assert d["blocks"] == {"bb": {"kind": "bollinger", "period": 20, "k": 2.0}, "atr": {"kind": "atr", "period": 14},
                           "adx": {"kind": "adx", "period": 14, "timeframe": "1d"}}
    assert "exit" not in d["long"] and "exit" not in d["short"]  # no signal exit: time and stop only
    params = {**to_params(d, "BINANCE"), "market": "perp", "allow_short": True}
    assert continuous_warmup("rules", params, 60) * 60 / 1440 == 141


def test_the_list_is_three_basket_rows_on_the_sealed_dates_with_the_holdout_locked():
    text = LIST.read_text()
    p = parse(text)
    assert (p.data_start, p.research_end, p.holdout_start, p.holdout_end) == (
        date(2024, 1, 1), date(2026, 4, 30), date(2026, 5, 1), date(2026, 10, 30))
    assert rescore.rows_of(text) == [(1, "qra0_funding_fade", 240), (2, "qrb0_range_reversion_adx", 60),
                                     (3, "qrh0_trend_pullback", 240)]  # H0 added 31 Oct
    for n, name, minutes in rescore.rows_of(text):
        req = rescore.request(name, minutes, "binance", "BTC/USDT,ETH/USDT,SOL/USDT", p)
        assert tuple(req.pairs) == ("BTC/USDT", "ETH/USDT", "SOL/USDT") and req.params["market"] == "perp"
        assert (req.use_holdout, req.train_days, req.test_days, req.holdout_days) == (False, 366, 485, 183)
        req.validate()  # row 2's 1-hour basket too: the floor is 1-hour (PM 30 Oct)


def test_the_list_is_round_next_0_with_a_budget_of_6_and_bs_13_prior_tries():
    from sleeve_fund.research import rounds

    text = LIST.read_text()
    r = rounds.round_of(text, [n for n, _, _ in rescore.rows_of(text)])
    assert (r.name, r.budget, r.families, r.prior) == ("next-0", 6, {"A": (1,), "B": (2, 3)}, {"B": 13})
    assert "family A try 1 " in rounds.try_line(r, 1, 0) and "family B try 14 " in rounds.try_line(r, 2, 0)


def test_the_list_dry_runs(capfd):
    assert rescore.main([str(LIST), "--pair", "BTC/USDT,ETH/USDT,SOL/USDT", "--dry-run"], run=lambda *a, **k: None,
                        journal=lambda: None) == 0
    assert "dry run: 3 checked, 0 run" in capfd.readouterr().out
