"""The trials register (v2 P1-6): every variant counted, the idea counter folded in, Sharpe deflated by the count."""

import json

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.research import trials as trials_mod
from sleeve_fund.research.ledger import IdeaLedger
from sleeve_fund.research.metrics import deflated_sharpe_probability
from sleeve_fund.research.trials import LEGACY_CODE, TrialsRegister, code_version, content_hash
from sleeve_fund.store import Store


@pytest.fixture
def reg(tmp_path):
    return TrialsRegister(Store(f"sqlite:///{tmp_path}/t.db"))


def _trial(reg, definition="d1", idea="i1", dataset="ds", family="trend", sharpe=0.5, stage="out_of_sample"):
    return reg.record(definition_hash=content_hash(definition), idea_hash=content_hash(idea), name="rsi_trend",
                      family=family, settings={"rsi": 14}, dataset=dataset, stage=stage, source="study",
                      sharpe=sharpe, trades=120, oos_trades=110)


def test_content_hash_ignores_key_order():
    assert content_hash({"a": 1, "b": [1, 2]}) == content_hash({"b": [1, 2], "a": 1})
    assert content_hash({"a": 1}) != content_hash({"a": 2})


def test_code_version_follows_the_indicator_source(tmp_path, monkeypatch):
    assert len(code_version()) == 40 and int(code_version(), 16) >= 0
    (tmp_path / "x.py").write_text("A = 1\n")
    monkeypatch.setattr(trials_mod, "INDICATORS", tmp_path)
    code_version.cache_clear()
    try:
        first = code_version()
        (tmp_path / "x.py").write_text("A = 2\n")
        code_version.cache_clear()
        assert code_version() != first
    finally:
        code_version.cache_clear()


def test_counts_variants_ideas_and_evaluations(reg):
    _trial(reg)
    _trial(reg)  # the same variant run again: one more evaluation, not another variant
    _trial(reg, definition="d2")  # another setting of the same idea
    _trial(reg, definition="d2", dataset="other")  # the same setting on other data
    _trial(reg, definition="d3", idea="i2")
    _trial(reg, definition="bh", idea="bh", family="benchmark")  # benchmarks are never counted
    assert reg.counts() == {"ideas": 2, "variants": 4, "evaluations": 5}
    assert sorted(reg.sharpes()) == [0.5] * 4  # one per variant, its latest (Advisor, 6 Oct 2026, P1-T2)


def test_changed_indicator_code_is_a_new_variant(reg, monkeypatch):
    _trial(reg)
    monkeypatch.setattr(trials_mod, "code_version", lambda: "f" * 40)
    _trial(reg)
    assert reg.counts()["variants"] == 2


def test_rows_are_stored_as_given_and_never_nan(reg):
    _trial(reg, sharpe=float("nan"))
    row = reg.store.trials()[0]
    assert row["sharpe"] is None and row["code_version"] == code_version()
    assert json.loads(row["settings"]) == {"rsi": 14} and row["oos_trades"] == 110


@pytest.mark.parametrize("bad", [{"stage": "sensitivity"}, {"source": "guess"}, {"sharpe": float("inf")},
                                 {"status": "pending"}])
def test_bad_trials_are_refused(reg, bad):
    row = {"id": "a" * 16, "definition_hash": "d", "idea_hash": "i", "code_version": "c", "definition_name": "n",
           "family": "f", "settings": "{}", "dataset": "ds", "stage": "holdout", "source": "study", "sharpe": 1.0}
    with pytest.raises(ValueError):
        reg.store.add_trials([dict(row, **bad)])


def test_the_idea_counter_is_imported_once_and_left_in_place(reg, tmp_path):
    path = tmp_path / "idea_ledger.jsonl"
    ledger = IdeaLedger(path)
    ledger.record(idea="a", family="trend", params={"x": 1}, dataset="d", stage="sensitivity", sharpe=0.1)
    ledger.record(idea="a", family="trend", params={"x": 2}, dataset="d", stage="wf_train", sharpe=0.2)
    ledger.record(idea="buy_and_hold", family="benchmark", params={}, dataset="d", stage="x", sharpe=0.3)
    before = path.read_text()
    assert reg.import_ledger(path) == 3
    assert reg.import_ledger(path) == 0  # running it again changes nothing
    assert path.read_text() == before
    rows = reg.store.trials()
    assert {r["source"] for r in rows} == {"ledger_import"} and {r["code_version"] for r in rows} == {LEGACY_CODE}
    assert reg.counts() == {"ideas": 1, "variants": 2, "evaluations": 2}
    ledger.record(idea="a", family="trend", params={"x": 3}, dataset="d", stage="holdout", sharpe=0.4)
    assert reg.import_ledger(path) == 1  # only the new line
    assert reg.counts()["variants"] == 3


def test_no_ledger_file_imports_nothing(reg, tmp_path):
    assert reg.import_ledger(tmp_path / "missing.jsonl") == 0


def test_deflated_sharpe_uses_the_register_count_and_falls_as_variants_grow(reg):
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(0.001, 0.01, 400))
    _trial(reg)
    one = reg.deflated_sharpe(returns, content_hash("i1"))
    assert one == pytest.approx(deflated_sharpe_probability(returns, 1))
    for i in range(50):
        _trial(reg, definition=f"v{i}")
    many = reg.deflated_sharpe(returns, content_hash("i1"))
    assert many == pytest.approx(deflated_sharpe_probability(returns, 51))
    assert many < one


def test_the_dashboard_folds_the_idea_counter_in_at_start(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    from sleeve_fund.dashboard import app as app_mod

    path = tmp_path / "idea_ledger.jsonl"
    IdeaLedger(path).record(idea="a", family="trend", params={}, dataset="d", stage="holdout", sharpe=0.1)
    monkeypatch.setattr(app_mod, "LEDGER", path)
    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path / "sheets")
    store = Store(f"sqlite:///{tmp_path}/t.db")
    app_mod.create_app(store)
    app_mod.create_app(store)  # a restart imports nothing twice
    assert len(store.trials()) == 1
