"""R4: the probability of backtest overfitting (research.pbo), PE1's cells R4-1..R4-8."""

import itertools

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.data import synthetic_ohlcv
from sleeve_fund.research.ledger import IdeaLedger
from sleeve_fund.research.pbo import (
    NA,
    NOT_JUDGED,
    PARTIAL,
    PBO_CHECK,
    TOO_PARTIAL,
    TOO_FEW,
    _ranks,
    g1_row,
    pbo,
    verdict_of,
)
from sleeve_fund.research.study import grid, run_study
from sleeve_fund.research.tearsheet import g1_verdict, render
from sleeve_fund.strategies.trend_filter import SPEC

DAYS = pd.date_range("2024-01-01", periods=400, freq="D", tz="UTC")


def _reference(x: np.ndarray, blocks: int) -> float:
    """CSCV the long way: every split, Sharpes from scratch, ranks by counting."""
    parts = np.array_split(np.arange(len(x)), blocks)
    n, below, total = x.shape[1], 0, 0
    for half in itertools.combinations(range(blocks), blocks // 2):
        ins = np.concatenate([parts[b] for b in half])
        out = np.concatenate([parts[b] for b in range(blocks) if b not in half])

        def sharpes(rows):
            m, s = x[rows].mean(axis=0), x[rows].std(axis=0, ddof=1)
            return [m[j] / s[j] if s[j] > 1e-15 else 0.0 for j in range(n)]

        a, b = sharpes(ins), sharpes(out)
        best = max(range(n), key=lambda j: (a[j], -j))
        rank = sum(1 for v in b if v < b[best]) + (sum(1 for v in b if v == b[best]) + 1) / 2
        w = rank / (n + 1)
        below += np.log(w / (1 - w)) <= 0
        total += 1
    return below / total


def _frame(x):
    return pd.DataFrame(x, index=DAYS[: len(x)], columns=[f"v{i}" for i in range(x.shape[1])])


def test_r4_1_it_matches_cscv_computed_the_long_way():
    x = np.random.default_rng(7).normal(0.001, 0.01, size=(160, 10))
    x[:, 3] += np.linspace(-0.01, 0.01, 160)  # one configuration whose edge flips half-way
    got = pbo(_frame(x), blocks=8)
    assert got.value == pytest.approx(_reference(x, 8), abs=1e-9)
    assert got.n_variants == 10 and got.n_combos == 70


def test_r4_1_when_the_in_sample_best_is_the_worst_out_of_sample_it_is_overfit():
    """Configuration j earns about +j a day in the first half of the days and -j in the second: whichever half of the
    blocks holds more early days, its best is the other half's worst."""
    rng = np.random.default_rng(2)
    j = np.arange(10, dtype=float) * 0.001
    x = np.vstack([j + rng.normal(0, 0.002, size=(80, 10)), -j + rng.normal(0, 0.002, size=(80, 10))])
    got = pbo(_frame(x), blocks=8)
    assert got.value == pytest.approx(_reference(x, 8), abs=1e-9)
    assert got.value > 0.5 and got.verdict == "FAIL"  # the splits with two early blocks a side are a coin toss


def test_r4_2_pure_noise_is_overfit_about_half_the_time():
    values = [pbo(_frame(np.random.default_rng(seed).normal(0, 0.01, size=(320, 20)))).value for seed in range(8)]
    assert 0.4 < np.mean(values) < 0.6


def test_r4_3_one_dominant_configuration_is_not_overfit():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 0.01, size=(320, 12))
    x[:, 4] = rng.normal(0.01, 0.002, size=320)  # better than the rest on every block
    got = pbo(_frame(x))
    assert got.value == 0.0 and got.verdict == "PASS" and got.n_combos == 12_870


def test_r4_4_too_few_variants_or_days_are_not_judged_and_never_pass_g1():
    """Advisor 13 Oct 16:50: each block needs 20 days, and neither shortfall counts as a pass."""
    x = np.random.default_rng(1).normal(0, 0.01, size=(320, 9))
    got = pbo(_frame(x))
    assert (got.value, got.verdict) == (None, TOO_FEW) and "fewer than the 10" in got.words
    short = pbo(_frame(np.random.default_rng(1).normal(0, 0.01, size=(319, 12))))
    assert (short.value, short.verdict) == (None, NA) and "319 days" in short.words
    assert pbo(_frame(np.random.default_rng(1).normal(0, 0.01, size=(320, 12)))).value is not None
    for r in (got, short):
        assert g1_row(r)[1] == NOT_JUDGED and g1_row(r)[2].startswith("PBO not judged: ")
    assert g1_row(short)[2].startswith("PBO not judged: window too short")
    assert g1_verdict([("Something else", "PASS", ""), g1_row(short)]) == (NOT_JUDGED, [PBO_CHECK])


def test_f238_1_a_study_with_no_pbo_result_is_not_judged_so_g1_fails_closed():
    assert g1_row(None) == (PBO_CHECK, NOT_JUDGED, "PBO not computed for this study")  # a result from before R4
    assert g1_verdict([("Something else", "PASS", ""), g1_row(None)]) == (NOT_JUDGED, [PBO_CHECK])


def test_f238_3_more_than_a_fifth_of_the_family_not_re_run_is_not_judged_a_fifth_still_is():
    f = _frame(np.random.default_rng(8).normal(0, 0.01, size=(320, 12)))
    three = tuple(f"trial t{i}: ValueError: bad" for i in range(3))  # 3 of 15 = 20%: judged, labelled partial
    four = tuple(f"trial t{i}: ValueError: bad" for i in range(4))  # 4 of 16 = 25%: not judged
    judged, not_judged = pbo(f, partial=three), pbo(f, partial=four)
    assert judged.value == pbo(f).value and g1_row(judged)[2].startswith(PARTIAL)
    assert (not_judged.value, not_judged.verdict) == (None, TOO_PARTIAL) and "4 of the 16" in not_judged.words
    assert g1_row(not_judged)[1] == NOT_JUDGED


def test_r4_5_thresholds_at_their_boundaries():
    assert [verdict_of(v) for v in (0.0, 0.2499, 0.25, 0.5, 0.5001, 1.0)] == [
        "PASS", "PASS", "WARN", "WARN", "FAIL", "FAIL"]


def test_r4_6_a_fail_fails_g1_and_a_warn_does_not():
    ok = ("Something else", "PASS", "")
    assert g1_verdict([ok, (PBO_CHECK, "FAIL", "x")]) == ("FAIL", [PBO_CHECK])
    assert g1_verdict([ok, (PBO_CHECK, "WARN", "x")]) == ("PASS", [])
    assert g1_verdict([ok, (PBO_CHECK, "N/A", "x")]) == ("PASS", [])


def test_r4_a_pbo_not_judged_never_passes_g1_but_never_hides_a_fail():
    short = g1_row(pbo(_frame(np.random.default_rng(1).normal(0, 0.01, size=(319, 12)))))
    ok, bad, errors = ("Something else", "PASS", ""), ("Enough trades", "FAIL", "3 trades"), ("Runs judged", NOT_JUDGED, "x")
    assert g1_verdict([ok, short]) == (NOT_JUDGED, [PBO_CHECK])
    assert g1_verdict([ok, bad, short]) == ("FAIL", ["Enough trades"])
    assert g1_verdict([errors, bad, short]) == (NOT_JUDGED, ["Runs judged"])  # unjudgeable runs still come first


def test_r4_the_dashboard_reads_the_pbo_row_in_the_same_order():
    from sleeve_fund.dashboard.pipeline import _g1
    from sleeve_fund.research.guardrails import G1_RULES

    def sheet(*rows):
        body = "\n".join(f"| {n} | {v} | {e} |" for n, v, e in [("G1 test", "PASS", "Sharpe 1.2"), *rows])
        return f"{body}\nG1 rules: {G1_RULES}\n"

    assert _g1(sheet((PBO_CHECK, "WARN", "30% of splits")))[0] == "PASS"
    assert _g1(sheet((PBO_CHECK, NOT_JUDGED, "PBO not judged: window too short")))[:2] == (
        NOT_JUDGED, "PBO not judged: window too short")
    assert _g1(sheet(("Enough trades", "FAIL", "3"), (PBO_CHECK, NOT_JUDGED, "PBO not judged")))[0] == "FAIL"


def test_r4_8_ties_take_the_average_rank():
    assert _ranks(np.array([[0.5, 0.1, 0.5, 0.9]])).tolist() == [[2.5, 1.0, 2.5, 4.0]]
    # Ten different flat series (not duplicates), every Sharpe 0: the pick ranks 5.5 of 10, the median, so overfit.
    flat = np.tile(np.arange(10) * 1e-4, (160, 1))
    assert pbo(_frame(flat), blocks=8).value == 1.0


def test_exact_duplicate_series_are_one_configuration_and_are_counted():
    """Advisor 13 Oct 16:50: a family setting that trades exactly like a grid point is not another variant."""
    f = _frame(np.random.default_rng(6).normal(0, 0.01, size=(320, 10)))
    doubled = pd.concat([f, f[["v2", "v7"]].set_axis(["again 2", "again 7"], axis=1)], axis=1)
    got = pbo(doubled)
    assert (got.n_variants, got.dropped) == (10, 2) and got.value == pbo(f).value
    assert "2 identical series dropped" in got.words
    nine = pbo(pd.concat([f.iloc[:, :9], f[["v0"]].set_axis(["again 0"], axis=1)], axis=1))
    assert (nine.verdict, nine.n_variants, nine.dropped) == (TOO_FEW, 9, 1)  # duplicates don't make up the ten


def test_a_counted_setting_that_couldnt_be_re_run_labels_the_result_partial():
    f = _frame(np.random.default_rng(8).normal(0, 0.01, size=(320, 10)))
    got = pbo(f, partial=("trial ab12: ValueError: bad setting",))
    assert (got.value, got.verdict) == (pbo(f).value, pbo(f).verdict)
    assert got.partial == ("trial ab12: ValueError: bad setting",)
    _, verdict, words = g1_row(got)
    assert verdict == got.verdict and words.startswith(f"{PARTIAL}: ") and "trial ab12: ValueError" in words
    assert "1 counted setting couldn't be re-run" in words and not g1_row(pbo(f))[2].startswith(PARTIAL)


def test_the_same_returns_give_the_same_value_and_gaps_are_dropped_together():
    x = np.random.default_rng(3).normal(0, 0.01, size=(320, 10))
    f = _frame(x)
    assert pbo(f) == pbo(f.copy())
    gappy = f.copy()
    gappy.iloc[0, 3] = np.nan
    assert pbo(gappy) == pbo(f.iloc[1:])


def test_r4_7_the_study_feeds_every_configuration_tried_and_shows_it(tmp_path, instrument):
    prices = synthetic_ohlcv(days=1900, seed=3)
    ledger = IdeaLedger(tmp_path / "l.jsonl")
    r = run_study(SPEC, prices, instrument, dataset="syn", ledger=ledger, synthetic=True, holdout_days=200,
                  train_days=730, test_days=365)
    assert r.pbo.n_variants == len(grid(SPEC.param_grid))  # every grid point, the discarded ones too
    row = next(c for c in render(r, ledger).splitlines() if c.startswith(f"| {PBO_CHECK} |"))
    assert r.pbo.words in row or r.pbo.verdict in row


def test_r4_the_study_feeds_the_idea_familys_counted_settings_on_the_research_period_only(tmp_path, instrument,
                                                                                            monkeypatch):
    """Advisor 13 Oct 16:50: every counted setting of the idea family joins the grid, each re-run on the study's
    research period and fee model; the holdout is never read, ablation rows stay out, a grid point already run isn't
    run again, and one that can't be re-run is listed."""
    from sleeve_fund.research import study
    from sleeve_fund.research.trials import TrialsRegister, legacy_idea_hash, trial_row
    from sleeve_fund.store import Store

    store = Store(f"sqlite:///{tmp_path}/t.db")
    register, idea = TrialsRegister(store), legacy_idea_hash(SPEC.name)

    def row(i, settings, kind="single"):
        return {**trial_row(definition_hash=f"d{i}", idea_hash=idea, name=SPEC.name, family="trend",
                            settings=settings, dataset="other", stage="in_sample", source="backtest", sharpe=0.1),
                "kind": kind}

    store.add_trials([row(1, {"fast": 30, "slow": 120}), row(2, {"fast": 30, "slow": 160}),
                      row(3, {"fast": 20, "slow": 100}),  # a grid point: already a column
                      row(4, {"fast": 30, "slow": 120}),  # the same setting counted twice: one re-run
                      row(5, {"fast": 10, "slow": 300}, kind="ablation"),  # never fed in
                      row(6, {"fast": 31, "slow": 120})])  # its re-run fails below
    bad = next(t["id"] for t in store.trials(idea) if '"fast": 31' in t["settings"])
    seen, real = [], study.run_backtest

    def spy(name, df, instrument, params, **kw):
        seen.append((params.get("fast"), params.get("slow"), df.index[-1]))
        if params.get("fast") == 31:
            raise ValueError("bad setting")
        return real(name, df, instrument, params, **kw)

    monkeypatch.setattr(study, "run_backtest", spy)
    prices = synthetic_ohlcv(days=1900, seed=3)
    r = run_study(SPEC, prices, instrument, dataset="syn", ledger=IdeaLedger(tmp_path / "l.jsonl"), synthetic=True,
                  holdout_days=200, train_days=730, test_days=365, register=register)
    family = [s for s in seen if (s[0], s[1]) in {(30, 120), (30, 160), (31, 120), (10, 300)}]
    assert sorted((f, s) for f, s, _ in family) == [(30, 120), (30, 160), (31, 120)]  # once each, no ablation
    holdout_start = prices.index[-1] - pd.Timedelta(days=199)
    assert all(end < holdout_start for _, _, end in family)
    assert r.pbo.n_variants + r.pbo.dropped == len(grid(SPEC.param_grid)) + 2
    assert len(r.pbo.partial) == 1 and r.pbo.partial[0].startswith(f"trial {bad}: ValueError: bad setting")
    assert g1_row(r.pbo)[2].startswith(PARTIAL) and r.error_count == 0  # its failure isn't the study's


def _register(tmp_path, monkeypatch=None):
    """A register on its own database. With monkeypatch, the 18:05 ruling an hour ago, so rows written now are after
    it whatever the wall clock says."""
    from sleeve_fund.research import trials
    from sleeve_fund.research.trials import TrialsRegister
    from sleeve_fund.store import Store

    if monkeypatch is not None:
        monkeypatch.setattr(trials, "NO_SELECTION_RULING", pd.Timestamp.now(tz="UTC").to_pydatetime()
                            - pd.Timedelta(hours=1))

    store = Store(f"sqlite:///{tmp_path}/t.db")
    return store, TrialsRegister(store)


SETUP = {"risk_profile": None, "windows": [730, 365, 0], "fee": 0.008}


def _run_row(register, settings, sharpe=0.4, **kw):
    from sleeve_fund.research.trials import legacy_definition_hash, legacy_idea_hash

    return register.record(definition_hash=legacy_definition_hash(SPEC.name, settings, SETUP),
                           idea_hash=legacy_idea_hash(SPEC.name), name=SPEC.name, family="trend", settings=settings,
                           dataset="syn", stage="in_sample", source="study", sharpe=sharpe,
                           data_start=DAYS[0].to_pydatetime(), data_end=DAYS[-1].to_pydatetime(), **kw)


def test_r4_one_pre_registered_setting_has_no_selection_and_g1_rests_on_the_rest(tmp_path, instrument):
    """Advisor 13 Oct 18:05: registered before its first run, the family's only setting: PBO is "not applicable: no
    selection", shown N/A, and G1 is decided by the other rows."""
    import dataclasses

    from sleeve_fund.research.pbo import NO_SELECTION
    from sleeve_fund.research.trials import legacy_idea_hash

    store, register = _register(tmp_path)
    one = dataclasses.replace(SPEC, param_grid={"fast": [50], "slow": [200]})
    reg = register.preregister(strategy=SPEC.name, family=SPEC.family, settings={"fast": 50, "slow": 200},
                               setup=SETUP, dataset="syn")
    assert register.no_selection(legacy_idea_hash(SPEC.name)) == "one pre-registered setting, tried as registered"
    r = run_study(one, synthetic_ohlcv(days=1200, seed=3), instrument, dataset="syn",
                  ledger=IdeaLedger(tmp_path / "l.jsonl"), synthetic=True, holdout_days=0, train_days=730,
                  test_days=365, register=register)
    assert len(store.trials(legacy_idea_hash(SPEC.name))) > 1 and r.pbo.verdict == NO_SELECTION
    row = g1_row(r.pbo)
    assert row[1] == "N/A" and row[2].startswith(NO_SELECTION)
    assert g1_verdict([("Something else", "PASS", ""), row]) == ("PASS", [])
    first = store.trials(legacy_idea_hash(SPEC.name))[0]
    assert first["id"] == reg and first["sharpe"] is None  # written before any run


def test_r4_a_registration_reads_no_bars_so_never_blocks_a_holdout(tmp_path):
    """DA 13 Oct: a registration written inside a later holdout's window refuses nothing; it read no bars."""
    from sleeve_fund.research.holdout import HoldoutLocks
    from sleeve_fund.research.trials import NO_BARS, legacy_idea_hash

    store, register = _register(tmp_path)
    register.preregister(strategy=SPEC.name, family="trend", settings={"fast": 50, "slow": 200}, setup=SETUP,
                         dataset="syn")
    row = store.trials(legacy_idea_hash(SPEC.name))[0]
    made = pd.Timestamp(row["created_at"])
    made = made.tz_localize("UTC") if made.tzinfo is None else made
    assert pd.Timestamp(row["data_start"]).value == pd.Timestamp(NO_BARS).value  # not when it was written
    locks = HoldoutLocks(store)
    assert locks.refusal(legacy_idea_hash(SPEC.name), "BTC", made - pd.Timedelta(days=30),
                         made + pd.Timedelta(days=30)) == ""


def test_r4_a_second_setting_ends_the_exemption_for_the_whole_family(tmp_path, monkeypatch):
    from sleeve_fund.research.trials import legacy_idea_hash

    _, register = _register(tmp_path, monkeypatch)
    idea = legacy_idea_hash(SPEC.name)
    register.preregister(strategy=SPEC.name, family="trend", settings={"fast": 50, "slow": 200}, setup=SETUP,
                         dataset="syn")
    _run_row(register, {"fast": 50, "slow": 200})
    assert register.no_selection(idea)
    _run_row(register, {"fast": 20, "slow": 200})  # anyone, anywhere, tries another
    assert register.no_selection(idea) is None
    assert pbo(_frame(np.random.default_rng(1).normal(0, 0.01, size=(320, 2)))).verdict == TOO_FEW  # normal PBO


@pytest.fixture
def r4_db(tmp_path, monkeypatch):
    """A register on its own database, Postgres when TEST_DATABASE_URL is set (the write-once triggers differ by
    engine), with the 18:05 ruling an hour ago as in _register."""
    import os

    from sqlalchemy import text

    from sleeve_fund.research import trials
    from sleeve_fund.research.trials import TrialsRegister
    from sleeve_fund.store import Store, make_engine, metadata

    monkeypatch.setattr(trials, "NO_SELECTION_RULING", pd.Timestamp.now(tz="UTC").to_pydatetime()
                        - pd.Timedelta(hours=1))
    eng = make_engine(os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path}/t.db")

    def clean():
        with eng.begin() as c:
            c.execute(text("DROP TABLE IF EXISTS alembic_version"))
        metadata.drop_all(eng)

    clean()
    store = Store(engine=eng)
    yield store, TrialsRegister(store)
    clean()


def test_r4_a_counted_rows_settings_and_date_cannot_be_edited(r4_db):
    """TRIALS-WORM: the edits the exemption guards against are refused by the database itself (a check violation on
    Postgres, an ABORT on SQLite), so the register keeps both its settings and its order."""
    from sqlalchemy import update
    from sqlalchemy.exc import IntegrityError

    from sleeve_fund.store import trials_t

    store, register = r4_db
    reg = register.preregister(strategy=SPEC.name, family="trend", settings={"fast": 50, "slow": 200}, setup=SETUP,
                               dataset="syn")
    run = _run_row(register, {"fast": 50, "slow": 200})
    later = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)
    for edit in (update(trials_t).where(trials_t.c.id == run).values(settings='{"fast": 40, "slow": 200}'),
                 update(trials_t).where(trials_t.c.id == reg).values(created_at=later.to_pydatetime())):
        with pytest.raises(IntegrityError, match="write-once"), store.engine.begin() as c:
            c.execute(edit)


def test_r4_an_edited_trial_row_loses_the_exemption(r4_db):
    """A row whose stored settings differ from the setting its definition hash was made from (written so by
    hand, or edited where the triggers were lifted) ends the exemption for the whole family. Built by insert, so the
    write-once triggers stay on."""
    from sleeve_fund.research.trials import legacy_definition_hash, legacy_idea_hash

    store, register = r4_db
    idea = legacy_idea_hash(SPEC.name)
    register.preregister(strategy=SPEC.name, family="trend", settings={"fast": 50, "slow": 200}, setup=SETUP,
                         dataset="syn")
    _run_row(register, {"fast": 50, "slow": 200})
    assert register.no_selection(idea)
    register.record(definition_hash=legacy_definition_hash(SPEC.name, {"fast": 50, "slow": 200}, SETUP),
                    idea_hash=idea, name=SPEC.name, family="trend", settings={"fast": 40, "slow": 200},
                    dataset="syn", stage="in_sample", source="study", sharpe=0.4,
                    data_start=DAYS[0].to_pydatetime(), data_end=DAYS[-1].to_pydatetime())
    assert register.no_selection(idea) is None


def test_r4_a_registration_dated_after_the_run_has_no_exemption(r4_db):
    """The registration must come before the family's first run: one written after it (the order an edited
    created_at would fake) is not a pre-registration of that run."""
    from sleeve_fund.research.trials import legacy_idea_hash

    store, register = r4_db
    idea = legacy_idea_hash(SPEC.name)
    _run_row(register, {"fast": 50, "slow": 200})
    register.preregister(strategy=SPEC.name, family="trend", settings={"fast": 50, "slow": 200}, setup=SETUP,
                         dataset="syn")
    assert register.no_selection(idea) is None


def test_r4_a_hand_coded_model_counted_before_the_ruling_needs_one_setting_only(tmp_path, monkeypatch):
    from sleeve_fund.research import trials
    from sleeve_fund.research.trials import legacy_idea_hash, trial_row

    store, register = _register(tmp_path, monkeypatch)
    idea = legacy_idea_hash(SPEC.name)
    _run_row(register, {"fast": 50, "slow": 200})
    assert register.no_selection(idea) is None  # first counted after the ruling, with no registration
    old = {**trial_row(definition_hash="d0", idea_hash=idea, name=SPEC.name, family="trend",
                       settings={"fast": 50, "slow": 200}, dataset="old", stage="in_sample", source="backtest",
                       sharpe=0.3), "created_at": trials.NO_SELECTION_RULING - pd.Timedelta(days=3)}
    store.add_trials([old])
    assert register.no_selection(idea) == "a hand-coded Model with one setting since its first counted run"


def test_r4_buy_and_hold_is_never_a_g1_candidate(tmp_path, instrument):
    from sleeve_fund.research.guardrails import CANDIDATE_CHECK
    from sleeve_fund.research.tearsheet import g1_checks
    from sleeve_fund.strategies.buy_and_hold import SPEC as HOLD

    ledger = IdeaLedger(tmp_path / "l.jsonl")
    r = run_study(HOLD, synthetic_ohlcv(days=1200, seed=3), instrument, dataset="syn", ledger=ledger, synthetic=True,
                  holdout_days=0, train_days=730, test_days=365)
    checks = g1_checks(r, ledger)
    row = next(c for c in checks if c[0] == CANDIDATE_CHECK)
    assert row[1] == NOT_JUDGED and "never a G1 candidate" in row[2]
    passing = [(n, "PASS" if n not in (CANDIDATE_CHECK, PBO_CHECK) else v, e) for n, v, e in checks]
    assert g1_verdict(passing)[0] == NOT_JUDGED  # every other row passing still isn't a pass
    assert g1_verdict([("Enough trades", "FAIL", "3"), row]) == ("FAIL", ["Enough trades"])  # nor hides a fail


def test_f238_2_pbo_reads_the_studys_own_idea_family_and_the_model_names_only_without_one():
    from sleeve_fund.research.study import _overfit
    from sleeve_fund.research.trials import legacy_idea_hash

    class Register:
        def __init__(self):
            self.asked = []

        def no_selection(self, idea):
            self.asked.append(idea)
            return ""

        def _counted(self, idea):
            self.asked.append(idea)
            return []

    for given, want in (("structural-abc", "structural-abc"), (None, legacy_idea_hash(SPEC.name))):
        reg = Register()
        _overfit({}, [], reg, SPEC, pd.DataFrame(), bt=None, idea_hash=given)
        assert reg.asked == [want, want]
