"""P1-6-ABL: the ablation report (research.ablation), on the spec the Independent Quant Advisor approved on 13 Oct
2026 (qd/ablation-spec.md) and PE1's test cells A1-A10."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.dashboard.results import ablation_rows
from sleeve_fund.research import ablation
from sleeve_fund.research.ablation import (
    ALWAYS,
    RANDOM_ENTRY,
    Score,
    ablate,
    remove,
    section,
    units,
)
from sleeve_fund.research.trials import TrialsRegister, model_run_row
from sleeve_fund.store import Store
from sleeve_fund.strategies.definitions import definition_hash, load_definition_file

ROOT = Path(__file__).resolve().parent.parent
REASON = "A test case, written before it is run."
RSI = {"left": "rsi", "op": "<", "right": 30}
TREND = {"left": "close", "op": ">", "right": "sma"}
QUIET = {"left": "rsi", "op": ">", "right": 10}
DIP = {"left": "close", "op": "<", "right": {"prev": "close"}}
RICH = {
    "version": 1, "reason": REASON,
    "blocks": {"rsi": {"kind": "rsi"}, "sma": {"kind": "sma", "period": 50},
               "dc": {"kind": "donchian", "period": 20}},
    "long": {"entry": {"all": [RSI, {"any": [TREND, DIP]}]}, "exit": {"left": "rsi", "op": ">=", "right": 55}},
    "exits": {"time_stop": {"bars": 10}, "stop": {"level": "dc.lower"}},
}


def _score(defn: dict, one_x: bool) -> Score:
    """A stand-in evaluation: a different, repeatable score for each definition and leverage."""
    h = int(definition_hash(defn), 16)
    return Score(net_return=(h % 1000) / 10_000 - (0.01 if one_x else 0), sharpe=(h % 300) / 100,
                 max_drawdown=-(h % 200) / 1000, trades=h % 90, time_in_market=(h % 100) / 100)


class _Rows:
    """A trials store holding rows in memory, as Store.add_trials / trials give them."""

    def __init__(self):
        self.rows = []

    def add_trials(self, rows):
        self.rows.extend({"status": "ok", "kind": "single", **r} for r in rows)
        return len(rows)

    def trials(self, idea_hash=None):
        return [r for r in self.rows if idea_hash is None or r["idea_hash"] == idea_hash]


def _register(store=None):
    reg = TrialsRegister(store or _Rows())
    row = model_run_row(strategy="rules", params={"definition": "parent"}, dataset="d", source="backtest",
                        setup={"risk_profile": None}, sharpe=0.8)
    reg.store.add_trials([row])
    return reg, row["idea_hash"]


def test_a1_units_are_every_entry_condition_signal_exit_and_exit_in_path_order():
    assert units(RICH) == ["long.entry[0]", "long.entry[1][0]", "long.entry[1][1]", "long.exit",
                           "exits.time_stop", "exits.stop"]
    rsi_bands = load_definition_file(ROOT / "configs" / "definitions" / "rsi_bands.toml")
    report = ablate(rsi_bands, _score)
    assert [(r.path, r.status) for r in report.rows] == [
        ("long.entry", "not ablatable"), ("long.exit", "ran"), ("short.entry", "not ablatable"), ("short.exit", "ran")]


def test_a1_a_condition_in_all_becomes_always_true_and_an_exit_is_removed():
    status, v, _ = remove(RICH, "long.entry[0]")
    assert status == "ran" and v["long"]["entry"]["all"][0] == ALWAYS and v["long"]["entry"]["all"][1] == RICH[
        "long"]["entry"]["all"][1]
    assert "exit" not in remove(RICH, "long.exit")[1]["long"]
    assert remove(RICH, "exits.time_stop")[1]["exits"] == {"stop": {"level": "dc.lower"}}
    assert RICH["exits"]["time_stop"] == {"bars": 10}  # the parent is never changed


def test_a2_the_same_inputs_give_the_same_report():
    one, two = ablate(RICH, _score), ablate(copy.deepcopy(RICH), _score)
    assert section(one) == section(two)
    assert [r.__dict__ for r in one.rows] == [r.__dict__ for r in two.rows]


def test_a3_a_single_condition_definition_has_nothing_to_ablate_and_adds_no_rows():
    one = {"version": 1, "reason": REASON, "blocks": {"rsi": {"kind": "rsi"}}, "long": {"entry": RSI}}
    reg, idea = _register()
    before = len(reg.store.rows)
    report = ablate(one, _score, register=reg, dataset="d", idea_hash=idea, family="rule-builder")
    assert report.nothing_to_ablate and [r.status for r in report.rows] == ["not ablatable"]
    assert len(reg.store.rows) == before
    assert "Nothing to ablate" in "\n".join(section(report))


@pytest.mark.parametrize("real", [False, True])
def test_a4_each_variant_is_an_ablation_row_and_n_is_unchanged_by_running_or_rerunning(real, tmp_path):
    reg, idea = _register(Store(f"sqlite:///{tmp_path}/t.db") if real else None)
    counts, sharpes = reg.counts(idea), reg.sharpes(idea)
    report = ablate(RICH, _score, register=reg, dataset="d", idea_hash=idea, family="rule-builder")
    ran = [r for r in report.rows if r.status == "ran"]
    rows = [t for t in reg.store.trials(idea) if t.get("kind") == "ablation"]
    assert len(rows) == len(ran) == 6
    for r, t in zip(ran, rows):
        assert t["id"] == r.trial_id and t["definition_hash"] == r.definition_hash != report.parent_hash
        assert t["family"] == "rule-builder" and t["idea_hash"] == idea and t["stage"] == "out_of_sample"
        assert f'"parent_definition_hash": "{report.parent_hash}"' in t["settings"]
        assert f'"removed_path": "{r.path}"' in t["settings"]
    assert reg.counts(idea) == counts and reg.sharpes(idea) == sharpes
    ablate(RICH, _score, register=reg, dataset="d", idea_hash=idea, family="rule-builder")  # A4b: a re-run
    assert len([t for t in reg.store.trials(idea) if t.get("kind") == "ablation"]) == 12
    assert reg.counts(idea) == counts and reg.sharpes(idea) == sharpes
    assert reg.counts() == reg.counts(idea)  # nor project-wide


def test_a4_an_ablation_row_needs_its_parents_idea_and_family():
    with pytest.raises(ValueError, match="parent's idea and family"):
        ablate(RICH, _score, register=_register()[0], dataset="d", idea_hash="", family="rule-builder")


def test_a5_removing_an_any_member_drops_it_and_the_other_still_gates():
    status, v, _ = remove(RICH, "long.entry[1][0]")
    assert status == "ran" and v["long"]["entry"] == {"all": [RSI, {"any": [DIP]}]}


def test_a6_removing_a_sides_sole_entry_condition_is_not_ablatable():
    for defn, path in [({**RICH, "long": {"entry": RSI}}, "long.entry"),
                       ({**RICH, "long": {"entry": {"all": [RSI]}}}, "long.entry[0]"),
                       ({**RICH, "long": {"entry": {"only_when": RSI, "then": {"all": [ALWAYS]}}}},
                        "long.entry.only_when")]:
        assert remove(defn, path) == ("not ablatable", None, RANDOM_ENTRY)


def test_a7_removing_the_stop_runs_stopless_at_1x_beside_the_base_at_1x():
    calls = []

    def score(defn, one_x):
        calls.append((definition_hash(defn), one_x))
        return _score(defn, one_x)

    report = ablate(RICH, score)
    stop = next(r for r in report.rows if r.path == "exits.stop")
    assert stop.one_x and "stopless, at 1x, no halts" in stop.note
    assert report.base_1x is not None and report.base_1x.one_x and (report.parent_hash, True) in calls
    assert stop.delta["net_return"] == pytest.approx(stop.score.net_return - report.base_1x.score.net_return)
    others = [r for r in report.rows if r.status == "ran" and r.path != "exits.stop"]
    assert not any(r.one_x for r in others)  # each still rests the stop, so sized as the base is
    assert all(r.delta["sharpe"] == pytest.approx(r.score.sharpe - report.base.score.sharpe) for r in others)
    text = "\n".join(section(report))
    assert "| base at 1x, no halts |" in text and "stopless, at 1x, no halts; compared with the base at 1x, no halts" in text


def test_a7_no_stop_removed_means_no_base_at_1x():
    report = ablate({**RICH, "exits": {"time_stop": {"bars": 10}, "stop": {"level": "dc.lower"},
                                       "trail": {"level": "dc.lower", "ratchet": True}}}, _score)
    assert report.base_1x is None  # the trail still rests a level when the stop goes, and the stop when the trail does


def test_a8_an_any_emptied_inside_an_all_is_no_trades_and_a_whole_entry_any_not_ablatable():
    nested = {**RICH, "long": {"entry": {"all": [RSI, {"any": [TREND]}]}}}
    status, v, note = remove(nested, "long.entry[1][0]")
    assert (status, v) == ("no trades", None) and "never enters" in note
    reg, idea = _register()
    before = len(reg.store.rows)
    report = ablate(nested, _score, register=reg, dataset="d", idea_hash=idea, family="rule-builder")
    assert [r.status for r in report.rows if r.path == "long.entry[1][0]"] == ["no trades"]
    assert len(reg.store.rows) == before + sum(r.status == "ran" for r in report.rows)
    whole = {**RICH, "long": {"entry": {"any": [RSI]}}}
    assert remove(whole, "long.entry[0]")[0] == "not ablatable"


def test_a8_an_any_emptied_inside_an_any_is_dropped_and_the_rest_still_enters():
    defn = {**RICH, "long": {"entry": {"any": [RSI, {"any": [TREND]}]}}}
    assert remove(defn, "long.entry[1][0]")[:2] == ("ran", {**defn, "long": {"entry": {"any": [RSI]}}})


def test_a_setup_cancel_is_one_unit_and_its_removal_drops_it():
    setup = {"setup": QUIET, "trigger": RSI, "cancel": TREND, "expire_after": {"bars": 4, "timeframe": "1d"}}
    defn = {**RICH, "long": {"entry": setup}}
    assert units(defn)[:3] == ["long.entry.setup", "long.entry.trigger", "long.entry.cancel"]
    assert "cancel" not in remove(defn, "long.entry.cancel")[1]["long"]["entry"]
    assert remove(defn, "long.entry.setup")[0] == "ran"


def test_the_section_is_labelled_not_judged_and_the_results_page_reads_it_back():
    report = ablate(RICH, _score)
    text = "\n".join(["# Tear sheet: x", "", *section(report), "## Idea counter", "", "- 1 distinct variant"])
    assert "## Ablation (not judged)" in text
    for word in ("PASS", "FAIL", "promote ", "best"):
        assert word not in "\n".join(section(report))
    rows = ablation_rows(text)
    assert len(rows) == 2 + len(report.rows)  # the base, the base at 1x, one per unit
    assert rows[0][0] == "base (every block on)" and rows[2][0].endswith("(long.entry[0])")
    assert ablation_rows("# Tear sheet: x\n\n## Idea counter\n") is None


def test_on_real_backtests_the_report_runs_and_is_repeatable(prices, instrument):
    days = prices.index
    windows = [(days[400], days[599]), (days[600], days[799])]
    evaluate = ablation.backtest_evaluator(prices, instrument, windows, bar_minutes=1440, half_spread=0,
                                           holdout_start=days[800], risk_profile="balanced")
    defn = REAL
    one, two = ablate(defn, evaluate), ablate(defn, evaluate)
    assert section(one) == section(two)
    assert one.base.score.trades > 0 and 0 < one.base.score.time_in_market < 1
    assert one.base.score.data_start == pd.Timestamp(days[0]).to_pydatetime()  # the first bar read, warm-up included
    assert [r.path for r in one.rows if r.status == "ran"] == ["long.entry[0]", "long.entry[1]", "long.exit", "exits.stop"]
    assert one.base_1x is not None


REAL = {"version": 1, "reason": REASON, "blocks": {"rsi": {"kind": "rsi"}, "dc": {"kind": "donchian", "period": 20}},
        "long": {"entry": {"all": [{"left": "rsi", "op": "<", "right": 45}, {"left": "close", "op": ">", "right": 0}]},
                 "exit": {"left": "rsi", "op": ">=", "right": 55}},
        "exits": {"stop": {"level": "dc.lower"}}}


def test_the_1x_pair_keeps_the_profiles_caps_and_drops_only_its_halts(prices, instrument, monkeypatch):
    """F235-1 (HoE and Advisor, 13 Oct): one_x runs under the same profile at 1x with no halts. A profile tight enough
    to halt the normal base still caps the 1x run's position, and that run never halts or pauses."""
    from sleeve_fund import risk
    from sleeve_fund.research import runner

    monkeypatch.setitem(risk.PROFILES, "tight", risk.RiskProfile("tight", max_drawdown=0.01, daily_loss=0.005,
                                                                 max_position_pct=0.33, max_open_positions=3,
                                                                 max_leverage=2.0))
    seen, real_run = [], runner.run_backtest

    def spy(*a, **k):
        res = real_run(*a, **k)
        seen.append(res)
        return res

    monkeypatch.setattr(runner, "run_backtest", spy)
    days = prices.index
    stopless = {k: v for k, v in REAL.items() if k != "exits"}

    def run(one_x, profile):
        seen.clear()
        ablation.backtest_evaluator(prices, instrument, [(days[400], days[799])], bar_minutes=1440, half_spread=0,
                                    holdout_start=days[800], risk_profile=profile)(stopless, one_x)
        return seen[0]

    halts = ("risk_halt", "risk_pause")
    assert any(e["kind"] in halts for e in run(False, "tight").risk_events)  # the normal base halts
    one_x = run(True, "tight")
    assert not any(e["kind"] in halts for e in one_x.risk_events)
    assert 0 < one_x.exposure.max() < 0.4  # sized at the 33% cap (the marks drift with the price after entry)
    with pytest.raises(ValueError, match="an ablation needs a risk profile"):
        run(True, None)  # F235-6: never uncapped

    # CR F235-5: a 1x short on a perp. The switch is explicit, so no halt or pause level is priced at all: the run
    # never halts, every fill is at a finite price, and the short stays within the cap.
    short = {"version": 1, "reason": REASON, "blocks": {"rsi": {"kind": "rsi"}},
             "short": {"entry": {"left": "rsi", "op": ">", "right": 55}, "exit": {"left": "rsi", "op": "<=", "right": 45}}}

    def run_short(one_x):
        seen.clear()
        ablation.backtest_evaluator(prices, instrument, [(days[400], days[799])], bar_minutes=1440, half_spread=0,
                                    holdout_start=days[800], risk_profile="tight",
                                    params={"market": "perp", "allow_short": True})(short, one_x)
        return seen[0]

    assert any(e["kind"] in halts for e in run_short(False).risk_events)
    one_x_short = run_short(True)
    assert not any(e["kind"] in halts for e in one_x_short.risk_events)
    assert len(one_x_short.fills) and np.isfinite(one_x_short.fills["avg_px"].astype(float)).all()
    opened = one_x_short.exposure[one_x_short.exposure != 0].iloc[0]
    assert -0.331 < opened < 0  # it went short, sized at the 33% cap, at 1x


def test_an_ablation_window_reaching_the_holdout_is_refused(prices, instrument):
    days = prices.index
    with pytest.raises(ValueError, match="never reads the held-back period"):
        ablation.backtest_evaluator(prices, instrument, [(days[600], days[800])], bar_minutes=1440,
                                    holdout_start=days[800], risk_profile="balanced")


def test_an_ablation_reads_no_bar_of_the_holdout(prices, instrument):
    """The held-back bars changed beyond recognition change nothing: they are never read."""
    days = prices.index
    windows = [(days[600], days[799])]
    changed = prices.copy()
    changed.loc[changed.index >= days[800], ["open", "high", "low", "close"]] *= 3
    one = ablation.backtest_evaluator(prices, instrument, windows, bar_minutes=1440, half_spread=0,
                                      holdout_start=days[800], risk_profile="balanced")(REAL, False)
    two = ablation.backtest_evaluator(changed, instrument, windows, bar_minutes=1440, half_spread=0,
                                      holdout_start=days[800], risk_profile="balanced")(REAL, False)
    assert one == two


def test_an_ablation_overlapping_an_opened_holdout_is_refused_and_opens_no_lock(prices, instrument, tmp_path):
    from sleeve_fund.research.holdout import HoldoutLocks

    days = prices.index
    locks = HoldoutLocks(Store(f"sqlite:///{tmp_path}/t.db"))
    assert locks.open("idea", "BTC", days[500], days[599])
    before = locks.store.holdout_locks("idea")
    with pytest.raises(ValueError, match="overlaps this idea's opened holdout"):
        ablation.backtest_evaluator(prices, instrument, [(days[550], days[650])], bar_minutes=1440,
                                    holdout_start=None, locks=locks, idea_hash="idea", risk_profile="balanced")
    ablation.backtest_evaluator(prices, instrument, [(days[600], days[699])], bar_minutes=1440,
                                holdout_start=None, locks=locks, idea_hash="idea", risk_profile="balanced")  # after it
    assert locks.store.holdout_locks("idea") == before
    assert locks.store.holdout_locks() == before


def test_f235_5_halts_off_drops_only_the_drawdown_halt_and_the_daily_pause():
    from sleeve_fund import risk

    p = risk.profile("balanced")
    assert risk.check(p, 50.0, 100.0, 100.0).action == "halt"  # 50% down: the halt
    assert risk.check(p, 50.0, 100.0, 100.0, halts=False) is None
    assert risk.check(p, 0.0, 100.0, 100.0, halts=False).action == "halt"  # a book that can't be valued still halts


@pytest.mark.parametrize("engine", ["sqlite", "postgres"])
@pytest.mark.parametrize("profile", [None, "no_such_profile"])
def test_f235_6_an_ablation_with_no_risk_profile_is_refused_before_anything_runs(prices, instrument, tmp_path,
                                                                                  monkeypatch, engine, profile):
    """F235-6: no ablation row runs uncapped. None, or a name with no profile, is refused when the evaluator is built,
    before a backtest runs or a lock is read, on either engine."""
    import os

    from sleeve_fund.research import runner
    from sleeve_fund.research.holdout import HoldoutLocks

    url = os.environ.get("TEST_DATABASE_URL") if engine == "postgres" else f"sqlite:///{tmp_path}/t.db"
    if not url:
        pytest.skip("Postgres only: TEST_DATABASE_URL")
    locks = HoldoutLocks(Store(url))
    monkeypatch.setattr(runner, "run_backtest", lambda *a, **k: pytest.fail("a backtest ran"))
    monkeypatch.setattr(locks, "lock", lambda *a, **k: pytest.fail("a lock was read"))
    days = prices.index
    with pytest.raises(ValueError, match="needs a risk profile|unknown risk profile"):
        ablation.backtest_evaluator(prices, instrument, [(days[400], days[599])], bar_minutes=1440, half_spread=0,
                                    holdout_start=days[800], locks=locks, idea_hash="idea", risk_profile=profile)
    with pytest.raises(TypeError, match="risk_profile"):
        ablation.backtest_evaluator(prices, instrument, [(days[400], days[599])], bar_minutes=1440,
                                    holdout_start=days[800])  # left out: no default to fall back on
