"""Holdout locks (v2 P1-7, C4): one look per idea and underlying, and none over days the idea already read."""

import pandas as pd
import pytest

from sleeve_fund.research.holdout import COUNTER_RETIRED, HoldoutLocks, underlying_of, underlying_of_dataset
from sleeve_fund.research.ledger import IdeaLedger
from sleeve_fund.research.trials import TrialsRegister, legacy_idea_hash
from sleeve_fund.store import Store

IDEA = legacy_idea_hash("rsi_trend")
T = pd.Timestamp


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path}/t.db")


def _trial(store, start, end, stage="in_sample"):
    TrialsRegister(store).record(definition_hash="d", idea_hash=IDEA, name="rsi_trend", family="trend",
                                 settings={}, dataset="ds", stage=stage, source="study", sharpe=0.5,
                                 data_start=start, data_end=end)


def test_underlying_is_the_base_asset_upper_case():
    assert underlying_of("btc/USDT") == "BTC"
    assert underlying_of_dataset("binance-btcusdt-store-60m") == "BTC"
    assert underlying_of_dataset("kraken-ethusd-store") == "ETH"
    assert underlying_of_dataset("syn") is None


def test_opens_once_per_idea_and_underlying_at_any_venue(store):
    locks = HoldoutLocks(store)
    start, end = T("2025-01-01", tz="UTC"), T("2025-12-31", tz="UTC")
    _trial(store, T("2022-01-01", tz="UTC"), T("2024-12-31", tz="UTC"))
    assert locks.refusal(IDEA, "BTC", start, end) == ""
    assert locks.open(IDEA, "btc", start, end)
    locks.link_trial(IDEA, "BTC", "t1")
    locks.link_trial(IDEA, "BTC", "t2")  # the first link stands
    assert store.holdout_locks(IDEA)[0]["trial_id"] == "t1"
    assert not locks.open(IDEA, "BTC", start, end)  # the database refuses the second opening
    assert "was opened on" in locks.refusal(IDEA, "BTC", start, end)
    assert locks.refusal(IDEA, "ETH", start, end) == ""  # another underlying has its own holdout
    assert [r["underlying"] for r in store.holdout_locks()] == ["BTC"]


def test_refused_over_days_the_idea_already_read(store):
    _trial(store, T("2022-01-01", tz="UTC"), T("2025-03-01", tz="UTC"))
    words = HoldoutLocks(store).refusal(IDEA, "BTC", T("2025-01-01", tz="UTC"), T("2025-12-31", tz="UTC"))
    assert "read data inside the held-back period (up to 01 Mar 2025)" in words


def test_a_trial_with_unknown_dates_blocks_a_holdout_before_the_counter_retired(store):
    _trial(store, None, None)
    words = HoldoutLocks(store).refusal(IDEA, "BTC", T("2025-01-01"), T("2025-12-31"))
    assert "kept no dates" in words and "only days after 06 Oct 2026 are unseen" in words


def test_undated_ideas_open_a_holdout_wholly_after_the_counter_retired_once_it_is_long_enough(store):
    """Advisor, C4 option b: days after the counter retired are unseen by undated runs, but the holdout must
    last 90 days before its one look."""
    _trial(store, None, None)
    locks, start = HoldoutLocks(store), COUNTER_RETIRED + pd.Timedelta(days=1)
    assert "10 more days of data" in locks.refusal(IDEA, "BTC", start, start + pd.Timedelta(days=80))
    assert locks.refusal(IDEA, "BTC", start, start + pd.Timedelta(days=90)) == ""
    assert locks.undated(IDEA)


def test_retuning_an_undated_idea_moves_its_holdout_start_forward(store):
    """Advisor, C4 option b, condition 2: days after the counter retired that a new study tuned on are seen."""
    _trial(store, None, None)
    start = COUNTER_RETIRED + pd.Timedelta(days=1)
    _trial(store, start, start + pd.Timedelta(days=30))
    words = HoldoutLocks(store).refusal(IDEA, "BTC", start, start + pd.Timedelta(days=120))
    assert "read data inside the held-back period" in words
    later = start + pd.Timedelta(days=31)
    assert HoldoutLocks(store).refusal(IDEA, "BTC", later, later + pd.Timedelta(days=90)) == ""


def test_holdout_trials_themselves_do_not_block(store):
    _trial(store, T("2025-01-01", tz="UTC"), T("2025-12-31", tz="UTC"), stage="holdout")
    assert HoldoutLocks(store).refusal(IDEA, "BTC", T("2025-01-01"), T("2025-12-31")) == ""


def test_the_idea_counters_openings_are_imported_once(store, tmp_path):
    path = tmp_path / "idea_ledger.jsonl"
    ledger = IdeaLedger(path)
    ledger.record(idea="rsi_trend", family="trend", params={}, dataset="binance-btcusdt-store-60m", stage="holdout",
                  sharpe=0.2)
    ledger.record(idea="rsi_trend", family="trend", params={}, dataset="kraken-btcusd-store", stage="holdout",
                  sharpe=0.1)  # the same underlying elsewhere: already locked
    ledger.record(idea="rsi_trend", family="trend", params={}, dataset="syn", stage="holdout", sharpe=0.1)
    ledger.record(idea="rsi_trend", family="trend", params={}, dataset="kraken-ethusd-store", stage="wf_train",
                  sharpe=0.1)
    locks = HoldoutLocks(store)
    assert locks.import_ledger(path) == 1
    assert locks.import_ledger(path) == 0
    row = store.holdout_locks()[0]
    assert (row["underlying"], row["source"], row["period_start"]) == ("BTC", "ledger_import", None)
    assert locks.refusal(IDEA, "BTC", T("2026-01-01"), T("2026-12-31")).startswith("this idea's holdout on BTC")


def test_bad_source_is_refused(store):
    with pytest.raises(ValueError):
        store.add_holdout_lock({"id": "x", "idea_hash": IDEA, "underlying": "BTC", "source": "guess"})


def test_a_study_records_its_trials_with_dates_and_opens_the_holdout_once(store, tmp_path, instrument):
    import numpy as np

    from sleeve_fund.research.study import run_study
    from sleeve_fund.strategies.trend_filter import SPEC

    idx = pd.date_range("2024-01-01", periods=150, freq="1D", tz="UTC")
    c = 100 * np.exp(np.cumsum(np.random.default_rng(5).normal(0, 0.003 * 38, len(idx))))
    o = np.r_[c[0], c[:-1]]
    bars = pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c, "volume": 1e6},
                        index=idx)
    path = tmp_path / "l.jsonl"
    register, locks = TrialsRegister(store), HoldoutLocks(store)

    def study(dataset):
        return run_study(SPEC, bars, instrument, dataset=dataset, ledger=IdeaLedger(path), synthetic=True,
                         holdout_days=30, train_days=60, test_days=30, default_params={"fast": 20, "slow": 100},
                         use_holdout=True, register=register, locks=locks)

    first = study("kraken-btcusd-store")
    assert first.holdout is not None
    trials = store.trials()
    assert trials and all(t["data_start"] is not None for t in trials)
    research = [t for t in trials if t["stage"] == "in_sample"]
    assert max(pd.Timestamp(t["data_end"]) for t in research).tz_localize(None) < idx[-30].tz_localize(None)
    assert [t["stage"] for t in trials].count("holdout") == 1
    lock = store.holdout_locks()[0]
    assert lock["underlying"] == underlying_of(f"{instrument.base_currency}/x") and lock["trial_id"]
    # The counter lines the study wrote are the same rows: folding the counter in adds nothing.
    assert register.import_ledger(path) == 0
    # Another venue's dataset name can't open the same days again.
    second = study("binance-btcusdt-store-60m")
    assert second.holdout is None and "was opened on" in second.holdout_withheld


def test_a_lock_taken_mid_study_withholds_the_holdout(store, tmp_path, instrument):
    """The lock is claimed before the look: a second study that took it after this one's check still wins, and
    this one never sees the held-back days (Code Reviewer, #145)."""
    import numpy as np

    from sleeve_fund.research.study import run_study
    from sleeve_fund.strategies.trend_filter import SPEC

    idx = pd.date_range("2024-01-01", periods=150, freq="1D", tz="UTC")
    c = 100 * np.exp(np.cumsum(np.random.default_rng(5).normal(0, 0.003 * 38, len(idx))))
    o = np.r_[c[0], c[:-1]]
    bars = pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c, "volume": 1e6},
                        index=idx)

    class RacedLocks(HoldoutLocks):
        def refusal(self, idea_hash, underlying, start, end):
            answer = super().refusal(idea_hash, underlying, start, end)
            assert HoldoutLocks(self.store).open(idea_hash, underlying, start, end)  # the other study gets there
            return answer

    result = run_study(SPEC, bars, instrument, dataset="kraken-btcusd-store", ledger=IdeaLedger(tmp_path / "l.jsonl"),
                       synthetic=True, holdout_days=30, train_days=60, test_days=30,
                       default_params={"fast": 20, "slow": 100}, use_holdout=True,
                       register=TrialsRegister(store), locks=RacedLocks(store))
    assert result.holdout is None and "while this one ran" in result.holdout_withheld
    assert [t["stage"] for t in store.trials()].count("holdout") == 0
    assert store.holdout_locks()[0]["trial_id"] is None


def test_an_undated_idea_keeps_its_holdout_closed_until_it_holds_enough_trades(store, tmp_path, instrument):
    """Advisor, C4 option b: only the trade count is read before the claim, and the lock is not spent."""
    import numpy as np

    from sleeve_fund.research.study import run_study
    from sleeve_fund.strategies.trend_filter import SPEC

    TrialsRegister(store).record(definition_hash="d", idea_hash=legacy_idea_hash(SPEC.name), name=SPEC.name,
                                 family="trend", settings={}, dataset="ds", stage="in_sample", source="study",
                                 sharpe=0.5)
    idx = pd.date_range("2026-03-01", periods=400, freq="1D", tz="UTC")
    c = 100 * np.exp(np.cumsum(np.random.default_rng(5).normal(0, 0.003 * 38, len(idx))))
    o = np.r_[c[0], c[:-1]]
    bars = pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c, "volume": 1e6},
                        index=idx)
    result = run_study(SPEC, bars, instrument, dataset="kraken-btcusd-store", ledger=IdeaLedger(tmp_path / "l.jsonl"),
                       synthetic=True, holdout_days=120, train_days=120, test_days=60,
                       default_params={"fast": 20, "slow": 100}, use_holdout=True,
                       register=TrialsRegister(store), locks=HoldoutLocks(store))
    assert result.holdout is None and "trades, and 20 are needed" in result.holdout_withheld
    assert store.holdout_locks() == []
