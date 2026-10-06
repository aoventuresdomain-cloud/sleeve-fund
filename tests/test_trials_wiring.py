"""Every backtest, new or re-set paper strategy counts in the trials register, and the tear sheet and the
Research page read the register's N (QA P1-T1)."""

import pytest

from sleeve_fund.research.trials import TrialsRegister
from sleeve_fund.store import Store
from sleeve_fund.venues import KRAKEN

from test_dashboard import AUTH, SAME, _new, client  # noqa: F401  (the fixture)


def test_a_single_backtest_increments_the_variant_count(client, monkeypatch):
    from sleeve_fund.dashboard import preview
    from sleeve_fund.data import synthetic_ohlcv

    c, store = client
    preview._history.clear()
    monkeypatch.setattr(KRAKEN, "daily_history", lambda pair: synthetic_ohlcv(days=400, seed=3, vol=0.03))
    register = TrialsRegister(store)
    before = register.counts()["variants"]
    q = ("/backtest?run=1&instrument=ETH/USD&strategy=trend_filter&p_trend_filter__fast=5"
         "&p_trend_filter__slow=20&starting_balance=5000&period=365")
    assert "Every trade" in c.get(q, auth=AUTH).text
    rows = [t for t in store.trials() if t["source"] == "backtest"]
    assert register.counts()["variants"] == before + 1 and len(rows) == 1
    t = rows[0]
    assert t["definition_name"] == "trend_filter" and t["stage"] == "in_sample" and t["sharpe"] is not None
    assert t["data_start"] is not None and t["data_end"] > t["data_start"] and t["backtest_id"]
    # Other settings are another variant; the same ones again are another evaluation of the same variant.
    c.get(q.replace("fast=5", "fast=6"), auth=AUTH)
    assert register.counts()["variants"] == before + 2


def test_a_new_strategy_and_a_settings_change_are_counted(client):
    c, store = client
    assert _new(c).status_code == 303
    register = TrialsRegister(store)
    made = [t for t in store.trials() if t["source"] == "strategy"]
    assert len(made) == 1 and made[0]["sharpe"] is None and made[0]["dataset"].endswith("-60m")
    r = c.post("/sleeves/btc-test/settings", data={"risk_profile": "conservative", "reason": "tighter"},
               auth=AUTH, headers=SAME, follow_redirects=False)
    assert r.status_code == 303 and "settings_error" not in r.headers["location"]
    assert len([t for t in store.trials() if t["source"] == "strategy"]) == 2
    assert register.ideas_by_family() == {"trend": 1}
    # Another risk profile is another variant (Advisor, 6 Oct 2026: the profile is in the key).
    assert register.counts()["variants"] == 2


def test_a_counting_failure_is_an_error_event_and_loses_neither_the_backtest_nor_the_strategy(client, monkeypatch):
    """Code Reviewer on #154: counting ran before the backtest was saved and after the strategy was committed, so
    a failure there lost the PM's result or reported an error for a strategy already made. Now each is kept and the
    missed count is an error event."""
    from sleeve_fund.dashboard import app as appmod, preview
    from sleeve_fund.data import synthetic_ohlcv

    def broken(*args, **kwargs):
        raise KeyError("from")

    c, store = client
    monkeypatch.setattr(appmod, "_count_backtest", broken)
    monkeypatch.setattr(appmod, "_count_strategy", broken)
    preview._history.clear()
    monkeypatch.setattr(KRAKEN, "daily_history", lambda pair: synthetic_ohlcv(days=400, seed=3, vol=0.03))
    before = len(store.backtests())
    q = ("/backtest?run=1&instrument=ETH/USD&strategy=trend_filter&p_trend_filter__fast=5"
         "&p_trend_filter__slow=20&starting_balance=5000&period=365")
    assert "Every trade" in c.get(q, auth=AUTH).text
    assert len(store.backtests()) == before + 1  # saved, though not counted
    r = _new(c)
    assert r.status_code == 303 and "error" not in r.headers["location"] and store.sleeve("btc-test") is not None
    failed = [e for e in store.events(min_level="error") if e["kind"] == "trials_count_failed" and e["sleeve"] is None]
    assert len(failed) == 1 and "backtest" in failed[0]["message"] and "KeyError" in failed[0]["message"]
    made = [e for e in store.events("btc-test", min_level="error") if e["kind"] == "trials_count_failed"]
    assert len(made) == 1 and "strategy btc-test" in made[0]["message"]
    assert not [t for t in store.trials() if t["source"] in ("backtest", "strategy")]


def test_the_research_page_shows_the_registers_count(client):
    c, store = client
    TrialsRegister(store).record(definition_hash="d", idea_hash="i", name="x", family="trend", settings={},
                                 dataset="ds", stage="in_sample", source="backtest", sharpe=0.5)
    for i in range(2):
        TrialsRegister(store).record(definition_hash=f"e{i}", idea_hash="j", name="y", family="trend", settings={},
                                     dataset="ds", stage="in_sample", source="backtest", sharpe=0.5)
    page = c.get("/research", auth=AUTH).text
    assert 'Ideas tested</div><div class="v lead">2<' in page and 'Variants tried</div><div class="v lead">3<' in page


def test_overlapping_imports_lose_no_rows_and_raise_nothing(tmp_path):
    """QA m3: a second start-up importing the same counter at once must not fail the dashboard."""
    import json

    store = Store(f"sqlite:///{tmp_path}/t.db")
    path = tmp_path / "idea_ledger.jsonl"
    path.write_text("\n".join(json.dumps({"ts": f"2026-01-01T00:00:{i:02d}+00:00", "idea": "a", "family": "trend",
                                          "params": {"n": i}, "dataset": "ds", "stage": "wf_train", "sharpe": 0.1})
                              for i in range(20)) + "\n")
    register = TrialsRegister(store)
    real = store._insert_trials
    calls = {"n": 0}

    def racing(rows):
        calls["n"] += 1
        if calls["n"] == 1:  # another process gets there between the check and the insert
            real(rows[:5])
        return real(rows)

    store._insert_trials = racing
    assert register.import_ledger(path) == 15 and len(store.trials()) == 20


def test_the_tear_sheet_judges_by_the_ideas_own_family_and_shows_the_project_total(tmp_path, instrument):
    """Advisor, 6 Oct 2026: N is everything tried under the result's own idea; the project-wide total is shown
    for awareness and never gates."""
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.study import run_study
    from sleeve_fund.research.tearsheet import render
    from sleeve_fund.research.trials import legacy_idea_hash
    from sleeve_fund.strategies.trend_filter import SPEC

    store = Store(f"sqlite:///{tmp_path}/t.db")
    register = TrialsRegister(store)
    idea = legacy_idea_hash(SPEC.name)
    for i in range(40):  # forty backtests of other settings of this idea before the study
        register.record(definition_hash=f"d{i}", idea_hash=idea, name=SPEC.name, family="trend",
                        settings={"fast": i}, dataset="other", stage="in_sample", source="backtest", sharpe=0.2)
    for i in range(25):  # and another idea's, which must not raise this one's bar
        register.record(definition_hash=f"o{i}", idea_hash="other", name="x", family="trend", settings={},
                        dataset="other", stage="in_sample", source="backtest", sharpe=0.2)
    ledger = IdeaLedger(tmp_path / "l.jsonl")
    r = run_study(SPEC, synthetic_ohlcv(days=1200, seed=3), instrument, dataset="syn", ledger=ledger, synthetic=True,
                  holdout_days=0, train_days=365, test_days=180, register=register)
    sheet = render(r, ledger, register)
    mine, everyone = register.counts(idea)["variants"], register.counts()["variants"]
    assert mine >= 41 and everyone == mine + 25
    assert f"{mine} distinct variants of this idea" in sheet and f"{everyone} distinct variants tested so far" in sheet


def test_the_variant_key_holds_the_setup_but_not_the_fee_ladder(tmp_path):
    """Advisor, 6 Oct 2026: risk profile, walk-forward windows and the selection fee make a new variant."""
    from sleeve_fund.research.trials import legacy_definition_hash, run_setup

    base = dict(risk_profile="balanced", fee=0.0026, windows=(730, 365, 200))
    key = legacy_definition_hash("trend_filter", {"fast": 5}, run_setup(**base))
    for change in (dict(risk_profile="conservative"), dict(fee=0.004), dict(windows=(365, 180, 200)),
                   dict(windows=None)):
        assert legacy_definition_hash("trend_filter", {"fast": 5}, run_setup(**{**base, **change})) != key
    assert legacy_definition_hash("trend_filter", {"fast": 5}, run_setup(**{**base, "fee": 0.0026000000001})) == key


def test_the_study_does_not_count_its_fee_ladder_or_benchmark(tmp_path, instrument):
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.study import run_study
    from sleeve_fund.strategies.trend_filter import SPEC

    store = Store(f"sqlite:///{tmp_path}/t.db")
    register = TrialsRegister(store)
    r = run_study(SPEC, synthetic_ohlcv(days=1200, seed=3), instrument, dataset="syn",
                  ledger=IdeaLedger(tmp_path / "l.jsonl"), synthetic=True, holdout_days=0, train_days=365,
                  test_days=180, register=register)
    names = {t["definition_name"] for t in store.trials()}
    assert names == {SPEC.name}  # no buy-and-hold benchmark rows
    assert register.counts()["variants"] == len(r.sensitivity)  # one per grid point; ladder rungs add none


def test_an_unchanged_clone_adds_no_variant_and_only_engineering_runs_are_left_out(client):
    c, store = client
    register = TrialsRegister(store)
    assert _new(c).status_code == 303
    assert _new(c, name="btc-clone", **{"from": "clone", "source": "btc-test"}).status_code == 303
    assert register.counts()["variants"] == 1 and register.counts()["evaluations"] == 2
    register.record(definition_hash="fixture", idea_hash="i", name="x", family="trend", settings={}, dataset="ds",
                    stage="in_sample", source="engineering", sharpe=3.0)
    assert register.counts()["variants"] == 1  # marked as engineering: not counted


def test_spread_is_the_variants_trial_sharpes_floored_at_the_no_skill_error(tmp_path):
    """Advisor, 6 Oct 2026: one Sharpe per variant, its latest; spread = their std or 1/sqrt(T), the larger."""
    import math

    import numpy as np
    import pandas as pd

    from sleeve_fund.research.metrics import PERIODS_PER_YEAR, deflated_sharpe_probability

    store = Store(f"sqlite:///{tmp_path}/t.db")
    register = TrialsRegister(store)
    for i in range(30):
        for sharpe in (5.0, float(i) / 10):  # an old evaluation, then the latest
            register.record(definition_hash=f"d{i}", idea_hash="i", name="x", family="trend", settings={},
                            dataset="ds", stage="in_sample", source="backtest", sharpe=sharpe)
    assert sorted(register.sharpes("i")) == pytest.approx([i / 10 for i in range(30)])
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.001, 0.01, 400))
    assert register.deflated_sharpe(returns, "i") == pytest.approx(
        deflated_sharpe_probability(returns, 30, [i / 10 for i in range(30)]))
    # Trial Sharpes all alike: their spread is 0, and the floor 1/sqrt(T) stands.
    flat = [1.0] * 30
    assert deflated_sharpe_probability(returns, 30, flat) == pytest.approx(deflated_sharpe_probability(returns, 30))
    wide = [float(x) for x in rng.normal(0, 3 * math.sqrt(PERIODS_PER_YEAR / 399), 30)]
    assert deflated_sharpe_probability(returns, 30, wide) < deflated_sharpe_probability(returns, 30)


# QA round on #154 (quant-review/v2-p1/trials-154.md): QA Tester 2's strict xfails, now passing.
from sleeve_fund.data import synthetic_ohlcv  # noqa: E402
from sleeve_fund.data import synthetic_ohlcv


# QA P1-T3: the CLI study on a real data file (--data) never writes the trials register
def test_cli_study_on_a_data_file_is_counted(tmp_path, monkeypatch):
    from sleeve_fund.__main__ import main
    from sleeve_fund.store import Store

    db = f"sqlite:///{tmp_path / 'j.db'}"
    monkeypatch.setenv("DATABASE_URL", db)
    store = Store(db)
    before = len(store.trials())
    import sleeve_fund.__main__ as cli
    monkeypatch.setattr(cli, "load_kraken_ohlcvt", lambda path: synthetic_ohlcv(days=1900, seed=3))
    csv = tmp_path / "XBTUSD_1440.csv"
    csv.write_text("stand-in: the loader is replaced with 1,900 synthetic days\n")
    assert main(["--ledger", str(tmp_path / "l.jsonl"), "study", "trend_filter", "--data", str(csv),
                 "--holdout-days", "200", "--train-days", "730", "--test-days", "365",
                 "--out", str(tmp_path / "sheet.md")]) == 0
    assert len(store.trials()) > before  # Advisor 14:26: every run counts, whatever path started it


def _store(tmp_path):
    from sleeve_fund.store import Store

    return Store(f"sqlite:///{tmp_path / 'j.db'}")


# QA P1-T4 (Advisor 14:47): a paper strategy row must store data_end = creation/edit time, not be undated
def test_a_paper_strategy_is_dated_to_its_creation(tmp_path):
    import pandas as pd

    from sleeve_fund.research.holdout import HoldoutLocks
    from sleeve_fund.research.trials import legacy_idea_hash, record_model_run, run_setup

    store = _store(tmp_path)
    record_model_run(store, strategy="trend_filter", params={"a": 1}, dataset="d", source="strategy",
                     setup=run_setup(risk_profile="balanced", fee=0.001))
    row = store.trials(legacy_idea_hash("trend_filter"))[-1]
    assert row["data_end"] is not None
    now = pd.Timestamp.now(tz="UTC")
    # A 30-day holdout wholly after the creation: unseen, so no undated 90-day/20-trade rule applies.
    assert HoldoutLocks(store).refusal(legacy_idea_hash("trend_filter"), "BTC", now + pd.Timedelta(days=1),
                                       now + pd.Timedelta(days=31)) == ""


# QA P1-T6 (Advisor 14:47): the backtest period (preset name or exact custom dates) is in the variant key
def test_backtest_period_preset_is_in_the_variant_key(tmp_path):
    from sleeve_fund.dashboard import app as appmod
    from sleeve_fund.research.trials import TrialsRegister, legacy_idea_hash

    store = _store(tmp_path)
    monkey = {"strategy": "trend_filter", "pair": "BTC/USD", "venue": "kraken", "params": {"fast": 50, "slow": 200},
              "minutes": 1440, "risk_profile": "balanced"}
    result = {"from": "2024-01-01", "to": "2025-01-01", "fee_schedule": {"taker": 0.004}, "spread": {"half": 0.0001},
              "strategy": {"sharpe": 1.0}, "trades": {"trades": 10}}
    appmod._count_backtest(store, {**monkey, "days": 365}, result, "r1")
    appmod._count_backtest(store, {**monkey, "days": None}, {**result, "from": "2018-01-01"}, "r2")
    assert TrialsRegister(store).counts(legacy_idea_hash("trend_filter"))["variants"] == 2  # last year vs all history


# QA P1-T5 (Advisor 14:47): G1's best-of-N hurdle must use max(cross-variant spread, bootstrap spread), as the DSR does
def test_g1_hurdle_rises_with_the_variants_sharpe_spread(tmp_path):
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.study import run_study
    from sleeve_fund.research.tearsheet import SHARPE_CHECK, g1_checks
    from sleeve_fund.research.trials import TrialsRegister, legacy_idea_hash
    from sleeve_fund.strategies.trend_filter import SPEC
    from sleeve_fund.venues import venue

    inst = venue("KRAKEN").instrument("BTC", "USD")
    hurdles = []
    for tag, spread in (("narrow", 0.0), ("wide", 3.0)):
        store = _store(tmp_path / tag) if (tmp_path / tag).mkdir() is None else None
        reg = TrialsRegister(store)
        idea = legacy_idea_hash(SPEC.name)
        for i in range(40):  # 40 variants of this idea; Sharpes all equal, or spread from -3 to +3
            reg.record(definition_hash=f"d{i}", idea_hash=idea, name=SPEC.name, family="trend", settings={"fast": i},
                       dataset="other", stage="in_sample", source="backtest", sharpe=0.5 + spread * (i / 39 * 2 - 1))
        ledger = IdeaLedger(tmp_path / tag / "l.jsonl")
        r = run_study(SPEC, synthetic_ohlcv(days=1200, seed=3), inst, dataset="syn", ledger=ledger, synthetic=True,
                      holdout_days=0, train_days=365, test_days=180, register=reg)
        hurdles.append(next(c for c in g1_checks(r, ledger, reg) if c[0] == SHARPE_CHECK)[2])
    assert hurdles[0] != hurdles[1], hurdles[0]  # today identical: the trial spread never reaches G1


def test_a_data_file_holdout_in_the_old_counter_is_locked_by_its_underlying():
    # QA m-145a: the CLI names a data file's dataset by its stem, in the venue's own codes.
    from sleeve_fund.research.holdout import underlying_of_dataset

    assert underlying_of_dataset("XBTUSD_1440") == "BTC" and underlying_of_dataset("ETHUSD") == "ETH"
    assert underlying_of_dataset("kraken-btcusd-store-60m") == "BTC" and underlying_of_dataset("synthetic") is None
