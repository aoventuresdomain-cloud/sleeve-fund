"""QA P1-T8 (QA Tester 2, final 6 Oct 17:36), copied from quant-review/v2-p1/trials-154-scripts/test_trials_154_qa.py:
the T8 cases. The rest of that file is already in test_trials_wiring.py."""
import pytest

# --- P1-T8 fix interface (HoE design, 6 Oct 17:30). Rename here only; the assertions below use these names. ---
T8_STATUS_FIELD = "status"            # trials column; a crash in the metrics step writes status = "failed"
T8_FAILED = "failed"
T8_UNCERTAIN_TEXT = "N uncertain"      # shown wherever N or the deflated Sharpe appears, and in the refusals
T8_COUNTS_FLAG = "n_uncertain"         # TrialsRegister.counts(idea)[T8_COUNTS_FLAG] -> bool
T8_BACKTEST_METRICS = ("sleeve_fund.dashboard.app", "_backtest_trial_metrics")  # the metrics step of a backtest's trial
T8_STRATEGY_METRICS = ("sleeve_fund.dashboard.app", "_strategy_trial_metrics")  # the metrics step of a strategy's trial
T8_SAVE_ERROR_TEXT = "was not saved"   # what the user sees when the save rolls back (renamed: the page escapes an apostrophe)

from sleeve_fund.data import synthetic_ohlcv


def _store(tmp_path):
    from sleeve_fund.store import Store

    return Store(f"sqlite:///{tmp_path / 'j.db'}")




def test_t8_a_strategy_whose_count_failed_still_blocks_a_holdout_on_seen_data(tmp_path, monkeypatch):
    """Rewritten 17:45 for the T8 interface (QD 17:40): `_counted` is gone; a strategy is created while its metrics
    step raises. Same assertion as before: the year it was chosen on is seen data, so that holdout is refused."""
    import pandas as pd

    from sleeve_fund.research.holdout import HoldoutLocks
    from sleeve_fund.research.trials import legacy_idea_hash

    c, store = _t8_failed_strategy(tmp_path, monkeypatch)
    now = pd.Timestamp.now(tz="UTC")
    refusal = HoldoutLocks(store).refusal(legacy_idea_hash("trend_filter"), "BTC", now - pd.Timedelta(days=365),
                                          now - pd.Timedelta(days=1))
    assert refusal != ""


# ===================== P1-T8 fix: pre-build strict xfails (HoQA 17:30; Advisor 17:06) =====================
_T8 = "QA P1-T8 fix not built yet (HoE design 17:30)"


def _t8_client(tmp_path, monkeypatch):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path.cwd() / "tests"))
    import test_dashboard as td  # _new, AUTH, SAME

    return _plain_client(tmp_path, monkeypatch), td


def _plain_client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from sleeve_fund.dashboard import app as app_mod
    from sleeve_fund.store import Store

    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    monkeypatch.setenv("TEARSHEET_DIR", str(tmp_path))
    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    monkeypatch.setattr(app_mod, "LEDGER", tmp_path / "idea_ledger.jsonl")
    store = Store(f"sqlite:///{tmp_path}/t.db")
    return TestClient(app_mod.create_app(store)), store


def _fail_trial_inserts(store):
    """A database failure on the trial insert, at the database itself, whatever code path writes it."""
    from sqlalchemy import text

    with store.engine.begin() as c:
        c.execute(text("CREATE TRIGGER qa_fail_trials BEFORE INSERT ON trials "
                       "BEGIN SELECT RAISE(ABORT, 'qa: trial insert fails'); END"))


def _backtest(c, monkeypatch, fast=5):
    from sleeve_fund.dashboard import preview
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.venues import KRAKEN

    preview._history.clear()
    monkeypatch.setattr(KRAKEN, "daily_history", lambda pair: synthetic_ohlcv(days=400, seed=3, vol=0.03))
    q = ("/backtest?run=1&instrument=ETH/USD&strategy=trend_filter&p_trend_filter__fast=%d"
         "&p_trend_filter__slow=20&starting_balance=5000&period=365" % fast)
    return c.get(q, auth=("pm", "test-pw"))


def _trial_rows(store, source):
    return [t for t in store.trials() if t["source"] == source]


# (a) a database failure on the trial insert rolls back the whole save, and the user sees an error
def test_t8a_a_failed_trial_insert_rolls_back_the_backtest(tmp_path, monkeypatch):
    (c, store), _ = _t8_client(tmp_path, monkeypatch)
    before = len(store.backtests())
    _fail_trial_inserts(store)
    page = _backtest(c, monkeypatch).text
    assert len(store.backtests()) == before, "a backtest was saved without its trial row"
    assert "Every trade" not in page and T8_SAVE_ERROR_TEXT in page


def test_t8a_a_failed_trial_insert_rolls_back_a_new_strategy(tmp_path, monkeypatch):
    (c, store), td = _t8_client(tmp_path, monkeypatch)
    _fail_trial_inserts(store)
    r = td._new(c)
    # store.sleeve raises KeyError for a missing name (QD 17:40)
    assert not [x for x in store.sleeves() if x.name == "btc-test"], "a strategy was made without its trial row"
    assert "error" in r.headers.get("location", "") or r.status_code >= 400


def test_t8a_a_failed_trial_insert_rolls_back_a_settings_edit(tmp_path, monkeypatch):
    (c, store), td = _t8_client(tmp_path, monkeypatch)
    assert td._new(c).status_code == 303
    rows = len(_trial_rows(store, "strategy"))
    _fail_trial_inserts(store)
    r = c.post("/sleeves/btc-test/settings", data={"risk_profile": "conservative", "reason": "tighter"},
               auth=td.AUTH, headers=td.SAME, follow_redirects=False)
    assert store.sleeve("btc-test").risk_profile == "balanced", "the edit was kept without its trial row"
    assert len(_trial_rows(store, "strategy")) == rows
    assert "settings_error" in r.headers.get("location", "")


# (b) a crash in the metrics step keeps the save, with a status='failed' trial row in the same transaction
def test_t8b_a_metrics_crash_keeps_the_backtest_with_a_failed_row_that_counts(tmp_path, monkeypatch):
    import importlib

    from sleeve_fund.research.trials import TrialsRegister, legacy_idea_hash

    (c, store), _ = _t8_client(tmp_path, monkeypatch)
    mod, fn = T8_BACKTEST_METRICS
    monkeypatch.setattr(importlib.import_module(mod), fn, lambda *a, **k: (_ for _ in ()).throw(KeyError("from")))
    before = len(store.backtests())
    assert "Every trade" in _backtest(c, monkeypatch).text
    assert len(store.backtests()) == before + 1
    rows = _trial_rows(store, "backtest")
    assert len(rows) == 1 and rows[0][T8_STATUS_FIELD] == T8_FAILED and rows[0]["backtest_id"]
    assert TrialsRegister(store).counts(legacy_idea_hash("trend_filter"))["variants"] == 1  # counts towards N


def test_t8b_a_metrics_crash_keeps_the_strategy_with_a_failed_row_dated_open_start_to_now(tmp_path, monkeypatch):
    import importlib

    import pandas as pd

    from sleeve_fund.research.trials import OPEN_START, TrialsRegister, legacy_idea_hash

    (c, store), td = _t8_client(tmp_path, monkeypatch)
    mod, fn = T8_STRATEGY_METRICS
    monkeypatch.setattr(importlib.import_module(mod), fn, lambda *a, **k: (_ for _ in ()).throw(KeyError("fee")))
    t0 = pd.Timestamp.now(tz="UTC")
    r = td._new(c)
    assert r.status_code == 303 and store.sleeve("btc-test") is not None
    rows = _trial_rows(store, "strategy")
    assert len(rows) == 1 and rows[0][T8_STATUS_FIELD] == T8_FAILED
    assert pd.Timestamp(rows[0]["data_start"]) == pd.Timestamp(OPEN_START)
    assert pd.Timestamp(rows[0]["data_end"]) >= t0
    assert TrialsRegister(store).counts(legacy_idea_hash("trend_filter"))["variants"] == 1


# (c) while an idea has a failed row: G1 not judged (N uncertain), holdout refused, "N uncertain" shown; a re-count clears all
def _t8_failed_strategy(tmp_path, monkeypatch):
    import importlib

    (c, store), td = _t8_client(tmp_path, monkeypatch)
    mod, fn = T8_STRATEGY_METRICS
    real = getattr(importlib.import_module(mod), fn)
    monkeypatch.setattr(importlib.import_module(mod), fn, lambda *a, **k: (_ for _ in ()).throw(KeyError("fee")))
    assert td._new(c).status_code == 303
    monkeypatch.setattr(importlib.import_module(mod), fn, real)  # the cause is gone; a re-count can now succeed
    return c, store


def _t8_study(tmp_path, store):
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.study import run_study
    from sleeve_fund.research.trials import TrialsRegister
    from sleeve_fund.strategies.trend_filter import SPEC
    from sleeve_fund.venues import venue

    reg, ledger = TrialsRegister(store), IdeaLedger(tmp_path / "l.jsonl")
    r = run_study(SPEC, synthetic_ohlcv(days=1200, seed=3), venue("KRAKEN").instrument("BTC", "USD"), dataset="syn",
                  ledger=ledger, synthetic=True, holdout_days=0, train_days=365, test_days=180, register=reg)
    return r, ledger, reg


def _t8_signals(tmp_path, c, store):
    import pandas as pd

    from sleeve_fund.research.holdout import HoldoutLocks
    from sleeve_fund.research.tearsheet import NOT_JUDGED, g1_checks, g1_verdict, render
    from sleeve_fund.research.trials import legacy_idea_hash

    idea = legacy_idea_hash("trend_filter")
    r, ledger, reg = _t8_study(tmp_path, store)
    checks = g1_checks(r, ledger, reg)
    later = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)
    refusal = HoldoutLocks(store).refusal(idea, "BTC", later, later + pd.Timedelta(days=120))
    return {
        "flag": reg.counts(idea).get(T8_COUNTS_FLAG),
        "g1": g1_verdict(checks)[0] == NOT_JUDGED and any(T8_UNCERTAIN_TEXT in ev for _, _, ev in checks),
        "holdout": T8_UNCERTAIN_TEXT in refusal,
        "tearsheet": T8_UNCERTAIN_TEXT in render(r, ledger, reg),
        "research": T8_UNCERTAIN_TEXT in c.get("/research", auth=("pm", "test-pw")).text,
    }


def test_t8c_a_failed_row_makes_n_uncertain_everywhere(tmp_path, monkeypatch):
    c, store = _t8_failed_strategy(tmp_path, monkeypatch)
    assert _t8_signals(tmp_path, c, store) == {"flag": True, "g1": True, "holdout": True, "tearsheet": True,
                                               "research": True}
# The re-count case (all three clear) is DA-13 follow-up: test_da13_recount_xfails.py (HoE scope, 17:31).


# --- 36e5a97 re-test: unannounced probe F (Advisor 17:06: "N uncertain" wherever N or the deflated Sharpe appears) ---
def test_t10_the_tear_sheets_deflated_sharpe_and_n_lines_say_n_uncertain(tmp_path):
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.study import run_study
    from sleeve_fund.research.tearsheet import render
    from sleeve_fund.research.trials import TrialsRegister, failed_row
    from sleeve_fund.strategies.trend_filter import SPEC
    from sleeve_fund.venues import venue

    store = _store(tmp_path)
    store.add_trials([failed_row(strategy="trend_filter", params={"fast": 5}, source="backtest", error="qa")])
    reg, ledger = TrialsRegister(store), IdeaLedger(tmp_path / "l.jsonl")
    r = run_study(SPEC, synthetic_ohlcv(days=1200, seed=3), venue("KRAKEN").instrument("BTC", "USD"), dataset="syn",
                  ledger=ledger, synthetic=True, holdout_days=0, train_days=365, test_days=180, register=reg)
    lines = render(r, ledger, reg).splitlines()
    dsr = [l for l in lines if "Deflated Sharpe" in l]
    n = [l for l in lines if "results are judged by" in l]
    assert dsr and all(T8_UNCERTAIN_TEXT in l for l in dsr), dsr
    assert n and all(T8_UNCERTAIN_TEXT in l for l in n), n
