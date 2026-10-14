"""CR F236-3: a rule-builder study counts under its own definition's structural idea (candidates.idea_hash), never the
shared "rules" family, so its trial count, holdout lock and PBO family are its own; a parent and a retune of it share
one; a hand-coded model keeps its name's family. HoQA F236-4: a caller's idea_hash other than the definition's own is
refused, and a rule-builder pre-registration lands in its definition's family."""

import copy

import pytest
from test_candidates import PARENT
from test_r2_g1_oos_population import DEFN, SPEC, _study

from sleeve_fund.research.candidates import idea_hash
from sleeve_fund.research.study import idea_of, run_study
from sleeve_fund.strategies.definitions import to_params


def test_a_rules_studys_family_is_its_definitions_structural_idea():
    assert idea_of(to_params(PARENT), None) == idea_hash(PARENT)


def test_a_retune_shares_the_family_and_a_new_structure_does_not():
    retune = copy.deepcopy(PARENT)
    retune["blocks"]["rsi"]["period"] = 21
    other = copy.deepcopy(PARENT)
    other["long"]["entry"]["all"] = other["long"]["entry"]["all"][:1]  # a condition removed: a new idea
    assert idea_of(to_params(retune), None) == idea_of(to_params(PARENT), None)
    assert idea_of(to_params(other), None) != idea_of(to_params(PARENT), None)


def test_a_callers_own_idea_is_taken_and_a_hand_coded_model_keeps_its_names_family():
    assert idea_of(to_params(PARENT), idea_hash(PARENT)) == idea_hash(PARENT)
    retune = copy.deepcopy(PARENT)
    retune["blocks"]["rsi"]["period"] = 21
    assert idea_of(to_params(retune), idea_hash(PARENT)) == idea_hash(PARENT)  # a retune runs under its parent's idea
    assert idea_of({"fast": 10}, None) is None and idea_of(None, None) is None
    assert idea_of({"fast": 10}, "given") == "given"


def test_any_other_idea_for_a_definition_is_refused():
    other = copy.deepcopy(PARENT)
    other["long"]["entry"]["all"] = other["long"]["entry"]["all"][:1]
    for wrong in ("given", idea_hash(other)):
        with pytest.raises(ValueError, match="isn't this definition's own idea"):
            idea_of(to_params(PARENT), wrong)


def test_run_study_refuses_a_foreign_idea_before_counting_anything(tmp_path):
    import dataclasses

    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.trials import TrialsRegister
    from sleeve_fund.store import Store
    from sleeve_fund.venues import venue

    store = Store(f"sqlite:///{tmp_path}/t.db")
    params = to_params(PARENT)
    spec = dataclasses.replace(SPEC, param_grid={k: [v] for k, v in params.items()})
    with pytest.raises(ValueError, match="isn't this definition's own idea"):
        run_study(spec, synthetic_ohlcv(days=200, seed=1), venue("KRAKEN").instrument("BTC", "USD"), dataset="syn",
                  ledger=IdeaLedger(tmp_path / "l.jsonl"), default_params=params, holdout_days=0, train_days=30,
                  test_days=30, register=TrialsRegister(store), idea_hash="given")
    assert store.trials(None) == []


def test_a_rules_preregistration_lands_in_its_definitions_family(tmp_path):
    from sleeve_fund.research.trials import TrialsRegister, legacy_idea_hash
    from sleeve_fund.store import Store

    store = Store(f"sqlite:///{tmp_path}/t.db")
    register = TrialsRegister(store)
    register.preregister(strategy="rules", family="rules", settings=to_params(PARENT), setup={}, dataset="syn")
    register.preregister(strategy="ma_cross", family="trend", settings={"fast": 50}, setup={}, dataset="syn")
    assert [t["idea_hash"] for t in store.trials(idea_hash(PARENT))] == [idea_hash(PARENT)]
    assert len(store.trials(legacy_idea_hash("ma_cross"))) == 1 and store.trials(legacy_idea_hash("rules")) == []


def test_run_study_records_the_definitions_family_on_its_result():
    r, _ = _study([(15, 15), (14, 14), (14, 14)])
    assert r.idea_hash == idea_hash(DEFN)
