"""R6: the optimiser's one door (research.candidates), PE1's cells R6-1..R6-8."""

import ast
import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sleeve_fund.research import candidates
from sleeve_fund.research.ablation import ablate, remove
from sleeve_fund.research.candidates import (
    BUDGET_PER_IDEA,
    OPTIMISER_PACKAGE,
    PROTECTED_MODULES,
    Candidate,
    PmOverride,
    idea_hash,
    validate_candidate,
)
from sleeve_fund.research.trials import TrialsRegister, trial_row
from sleeve_fund.store import Store
from sleeve_fund.strategies.definitions import definition_hash

ROOT = Path(__file__).resolve().parent.parent
REASON = "A test case, written before it is run."
PARENT = {
    "version": 1, "name": "dip", "reason": REASON,
    "blocks": {"rsi": {"kind": "rsi", "period": 14}, "dc": {"kind": "donchian", "period": 20}},
    "long": {"entry": {"all": [{"left": "rsi", "op": "<", "right": 30}, {"left": "close", "op": ">", "right": 0}]},
             "exit": {"left": "rsi", "op": ">=", "right": 55}},
    "exits": {"time_stop": {"bars": 10}, "stop": {"level": "dc.lower"}},
}


class _Rows:
    def __init__(self):
        self.rows = []

    def add_trials(self, rows):
        self.rows.extend({"status": "ok", "kind": "single", **r} for r in rows)
        return len(rows)

    def trials(self, idea_hash=None):
        return [r for r in self.rows if idea_hash is None or r["idea_hash"] == idea_hash]


def _register(store=None):
    return TrialsRegister(store or _Rows())


def _with(**changes):
    d = copy.deepcopy(PARENT)
    d.update(changes)
    return d


def _tried(register, defn):
    register.store.add_trials([trial_row(definition_hash=definition_hash(defn), idea_hash=idea_hash(defn),
                                         name="dip", family="rule-builder", settings={}, dataset="d",
                                         stage="in_sample", source="optimiser", sharpe=0.5)])


@pytest.mark.parametrize("key, value", [("costs", {"breakout_slippage_bp": 0}), ("version", 2),
                                        ("reason", "Reworded after seeing the result."), ("name", "other")])
def test_r6_1_a_candidate_changing_anything_outside_the_strategy_is_refused_naming_it(key, value):
    with pytest.raises(ValueError, match=f"it changed {key}"):
        validate_candidate(PARENT, _with(**{key: value}), register=_register())


def test_r6_2_an_unknown_top_level_key_is_refused_before_anything_runs():
    with pytest.raises(ValueError, match="it changed leverage"):
        validate_candidate(PARENT, _with(leverage=3), register=_register())
    bad = _with(long={"entry": {"left": "rsi", "op": "<<", "right": 30}})
    with pytest.raises(ValueError, match="no comparison"):  # the checker's refusal, before any backtest
        validate_candidate(PARENT, bad, register=_register())


def test_r6_3_a_settings_change_is_a_candidate_of_the_same_idea_and_only_data():
    cand = _with(exits={"time_stop": {"bars": 12}, "stop": {"level": "dc.lower"}},
                 blocks={"rsi": {"kind": "rsi", "period": 10}, "dc": {"kind": "donchian", "period": 20}})
    out = validate_candidate(PARENT, cand, register=_register())
    assert isinstance(out, Candidate) and out.status == "candidate"
    assert out.idea_hash == idea_hash(PARENT) and out.parent_hash == definition_hash(PARENT)
    assert out.definition_hash == definition_hash(cand) != out.parent_hash


def test_r6_4_the_budget_is_counted_by_the_register_and_past_it_needs_the_pms_override():
    reg = _register()
    for bars in range(1, BUDGET_PER_IDEA + 1):
        _tried(reg, _with(exits={"time_stop": {"bars": bars}, "stop": {"level": "dc.lower"}}))
    _tried(reg, _with(exits={"time_stop": {"bars": 1}, "stop": {"level": "dc.lower"}}))  # a re-run: still 20
    next_one = _with(exits={"time_stop": {"bars": 99}, "stop": {"level": "dc.lower"}})
    with pytest.raises(ValueError, match=f"tried {BUDGET_PER_IDEA} configurations"):
        validate_candidate(PARENT, next_one, register=reg)
    again = _with(exits={"time_stop": {"bars": 3}, "stop": {"level": "dc.lower"}})
    assert validate_candidate(PARENT, again, register=reg)  # one already tried is no new configuration
    ok = PmOverride(by="pm", reason="Wider stop range asked for.", at=datetime(2026, 10, 13, tzinfo=UTC))
    assert validate_candidate(PARENT, next_one, register=reg, override=ok).override == ok
    for by, why in [("optimiser", "more"), ("pm", " ")]:
        with pytest.raises(ValueError, match="the PM's, with the reason"):
            PmOverride(by=by, reason=why, at=ok.at)


def test_r6_4_ablation_rows_use_no_budget(tmp_path):
    reg = _register(Store(f"sqlite:///{tmp_path}/t.db"))
    for bars in range(1, BUDGET_PER_IDEA):
        _tried(reg, _with(exits={"time_stop": {"bars": bars}, "stop": {"level": "dc.lower"}}))

    def score(defn, one_x):
        from sleeve_fund.research.ablation import Score
        return Score(0.0, 0.0, 0.0, 1, 0.5)

    ablate(PARENT, score, register=reg, dataset="d", idea_hash=idea_hash(PARENT), family="rule-builder")
    assert len(candidates.configurations(reg, idea_hash(PARENT))) == BUDGET_PER_IDEA - 1
    assert validate_candidate(PARENT, _with(exits={"time_stop": {"bars": 50}, "stop": {"level": "dc.lower"}}),
                              register=reg)


def test_r6_7_a_restructured_candidate_is_refused_unless_flagged_a_new_idea():
    cand = _with(long={"entry": {"left": "rsi", "op": "<", "right": 30}, "exit": PARENT["long"]["exit"]})
    assert idea_hash(cand) != idea_hash(PARENT)
    with pytest.raises(ValueError, match="flagged as a new idea"):
        validate_candidate(PARENT, cand, register=_register())
    out = validate_candidate(PARENT, cand, register=_register(), new_idea=True)
    assert out.new_idea and out.idea_hash == idea_hash(cand)


def test_r6_7_numbers_alone_never_make_a_new_idea():
    assert idea_hash(_with(long={**PARENT["long"], "exit": {"left": "rsi", "op": ">=", "right": 61.5}})) == \
        idea_hash(PARENT)
    assert idea_hash(_with(long={**PARENT["long"], "exit": {"left": "rsi", "op": ">", "right": 55}})) != \
        idea_hash(PARENT)


@pytest.mark.parametrize("real", [False, True])
def test_r6_8_an_ablation_variant_or_one_built_from_it_is_refused(real, tmp_path):
    reg = _register(Store(f"sqlite:///{tmp_path}/t.db") if real else None)

    def score(defn, one_x):
        from sleeve_fund.research.ablation import Score
        return Score(0.0, 0.0, 0.0, 1, 0.5)

    ablate(PARENT, score, register=reg, dataset="d", idea_hash=idea_hash(PARENT), family="rule-builder")
    variant = remove(PARENT, "long.entry[1]")[1]
    with pytest.raises(ValueError, match="is an ablation variant"):
        validate_candidate(PARENT, variant, register=reg)
    tweaked = copy.deepcopy(variant)
    tweaked["exits"]["time_stop"] = {"bars": 11}
    with pytest.raises(ValueError, match="built from an ablation variant"):
        validate_candidate(variant, tweaked, register=reg)
    stopless = remove(PARENT, "exits.stop")[1]  # a removed exit: a different idea, refused all the same
    with pytest.raises(ValueError, match="is an ablation variant"):
        validate_candidate(PARENT, stopless, register=reg, new_idea=True)


def test_r6_8_an_ablation_row_without_its_lineage_refuses_every_candidate():
    reg = _register()
    reg.store.add_trials([{**trial_row(definition_hash="x" * 16, idea_hash="i", name="dip", family="rule-builder",
                                       settings={"one_x": False}, dataset="d", stage="out_of_sample", source="study",
                                       sharpe=None), "kind": "ablation"}])
    with pytest.raises(ValueError, match="lineage can't be checked"):
        validate_candidate(PARENT, PARENT, register=reg)


# --- R6-5: the optimiser package never reaches what it must not touch ----------------------------------------------

def _protected(name: str) -> bool:
    return any(name == p or name.startswith(p + ".") for p in PROTECTED_MODULES)


def protected_uses(source: str) -> list[str]:
    """What a module does that an optimiser may not: import a protected module, open a file for writing, or write to
    the database directly. Each as a short line."""
    out = []
    for node in ast.walk(ast.parse(source)):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module] if _protected(node.module) else [f"{node.module}.{a.name}" for a in node.names]
        out += [f"imports {n}" for n in names if _protected(n)]
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name in ("add_trials", "add_holdout_lock", "execute", "write_text", "write_bytes"):
                out.append(f"calls {name}")
            if name == "open":
                modes = [a.value for a in node.args[1:2] if isinstance(a, ast.Constant)]
                modes += [k.value.value for k in node.keywords if k.arg == "mode" and isinstance(k.value, ast.Constant)]
                if any(set(str(m)) & set("wax+") for m in modes):
                    out.append("opens a file for writing")
    return sorted(set(out))


def test_r6_5_the_optimiser_package_touches_nothing_protected():
    assert OPTIMISER_PACKAGE == "sleeve_fund/research/optimiser"  # the agreed path, until an optimiser exists
    found = {str(p.relative_to(ROOT)): protected_uses(p.read_text(encoding="utf-8"))
             for p in sorted((ROOT / OPTIMISER_PACKAGE).rglob("*.py"))}
    assert {k: v for k, v in found.items() if v} == {}


def test_r6_5_the_scan_catches_a_module_that_reaches_a_protected_one():
    bad = ("import sleeve_fund.risk\nfrom sleeve_fund.research import trials\nfrom sleeve_fund.paper.runtime import x\n"
           "def go(store, p):\n    store.add_trials([])\n    open(p, 'w')\n    open(p, mode='a')\n")
    assert protected_uses(bad) == ["calls add_trials", "imports sleeve_fund.paper.runtime",
                                   "imports sleeve_fund.research.trials", "imports sleeve_fund.risk",
                                   "opens a file for writing"]
    ok = "from sleeve_fund.research.candidates import validate_candidate\nopen('x').read()\n"
    assert protected_uses(ok) == []


def test_r6_5_every_protected_module_exists():
    for name in PROTECTED_MODULES:
        path = ROOT / name.replace(".", "/")
        assert path.with_suffix(".py").exists() or (path / "__init__.py").exists(), name


def test_r6_6_no_code_path_from_a_candidate_to_paper():
    """Nothing outside this guard and its tests names a Candidate: a Strategy reaches paper only when the PM adds its
    definition. Nor does the guard import anything that could start one."""
    users = [str(p.relative_to(ROOT)) for p in (ROOT / "sleeve_fund").rglob("*.py")
             if "Candidate" in p.read_text(encoding="utf-8") and p.name != "candidates.py"]
    assert users == []
    assert protected_uses((ROOT / "sleeve_fund/research/candidates.py").read_text(encoding="utf-8")) == []
    assert json.dumps(PARENT)  # a candidate is plain data
