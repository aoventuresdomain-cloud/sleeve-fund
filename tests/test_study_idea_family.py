"""CR F236-3: a rule-builder study counts under its own definition's structural idea (candidates.idea_hash), never the
shared "rules" family, so its trial count, holdout lock and PBO family are its own; a parent and a retune of it share
one; a caller's idea_hash still wins; a hand-coded model keeps its name's family."""

import copy

from test_candidates import PARENT
from test_r2_g1_oos_population import DEFN, _study

from sleeve_fund.research.candidates import idea_hash
from sleeve_fund.research.study import idea_of
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


def test_a_callers_idea_hash_wins_and_a_hand_coded_model_keeps_its_names_family():
    assert idea_of(to_params(PARENT), "given") == "given"
    assert idea_of({"fast": 10}, None) is None and idea_of(None, None) is None


def test_run_study_records_the_definitions_family_on_its_result():
    r, _ = _study([(15, 15), (14, 14), (14, 14)])
    assert r.idea_hash == idea_hash(DEFN)
