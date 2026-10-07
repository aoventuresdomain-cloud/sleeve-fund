"""The shared paper scenarios (tests/paper_scenarios.py) do what they say, change nothing they don't, and match Head
of QA's own set-up where its file is present."""

import os
from datetime import datetime, timedelta, timezone

import pytest

from paper_scenarios import GAP_PX, NAME, frozen_clock, liquidate_in_paper


@pytest.fixture
def store(tmp_path):
    from sleeve_fund.store import Store

    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        from sleeve_fund.store import make_engine, metadata

        engine = make_engine(url)
        metadata.drop_all(engine)
        return Store(engine=engine)
    return Store(f"sqlite:///{tmp_path}/t.db")


def test_liquidate_in_paper_closes_past_bankruptcy_halts_and_reports_y_at_entry(tmp_path, store):
    import copy

    from sleeve_fund import risk

    objects, profiles = dict(risk.PROFILES), copy.deepcopy(risk.PROFILES)
    f = liquidate_in_paper(tmp_path, store)
    assert store.sleeve(NAME).status == "halted" and store.journal_book(NAME, 10_000)["qty"] == 0
    assert f.close["avg_px"] > f.entry["avg_px"] * 1.5 and 0 < f.rem < 10_000
    assert 9_000 < f.at_entry < 11_000 and f.x > 0 and f.y_at_entry == pytest.approx(100 * f.x / f.at_entry)
    assert risk.PROFILES == profiles  # nothing it touched is left changed
    assert all(risk.PROFILES[k] is v for k, v in objects.items())  # the same profile objects


def test_no_guard_is_lifted_unless_named_and_known(tmp_path, store):
    with pytest.raises(ValueError, match="no such guard"):
        liquidate_in_paper(tmp_path, store, guards_off=("stopless_cap",))


def test_matches_head_of_qas_own_liquidation(tmp_path, store):
    """Guard: where QA's ral-xfails file is in tests/, its own _liquidate gives the same figures on this head."""
    from sleeve_fund.store import Store

    qa = pytest.importorskip("test_ral_xfails")
    (tmp_path / "mine").mkdir()
    (tmp_path / "qa").mkdir()
    mine = liquidate_in_paper(tmp_path / "mine", store)
    theirs = qa._liquidate(tmp_path / "qa", Store(f"sqlite:///{tmp_path}/qa.db"))
    for k in ("rem", "peak", "before", "at_entry", "entry_fee"):
        assert getattr(mine, k) == pytest.approx(getattr(theirs, k)), k
    for k in ("orders", "reason"):
        assert getattr(mine, k) == getattr(theirs, k), k


def test_frozen_clock_freezes_every_binding_of_utcnow_and_moves(monkeypatch):
    """Every loaded sleeve_fund module attribute bound to store.utcnow, whatever its name (journal and riskops, which
    halt clearing and the 00:00 pause read, research.runner's _utcnow), returns the frozen time, and follows the
    setter."""
    import importlib
    import sys

    from sleeve_fund import store

    for mod in ("sleeve_fund.dashboard.app", "sleeve_fund.supervisor", "sleeve_fund.paper.runtime",
                "sleeve_fund.paper.journal", "sleeve_fund.dashboard.riskops", "sleeve_fund.research.runner"):
        importlib.import_module(mod)
    real = store.utcnow
    bound = [(m, a) for n, m in list(sys.modules.items()) if m is not None and n.startswith("sleeve_fund")
             for a, v in vars(m).items() if v is real]
    assert len(bound) > 5 and any(a == "_utcnow" for _, a in bound), bound
    at = datetime(2025, 10, 3, 23, 59, tzinfo=timezone.utc)
    move = frozen_clock(monkeypatch, at)
    assert all(getattr(m, a)() == at for m, a in bound), [(m.__name__, a) for m, a in bound]
    move(at + timedelta(minutes=2))
    assert all(getattr(m, a)() == at + timedelta(minutes=2) for m, a in bound)
    with pytest.raises(ValueError):
        frozen_clock(monkeypatch, datetime(2025, 10, 3))


def test_frozen_clock_refuses_a_named_module_without_utcnow(monkeypatch):
    at = datetime(2025, 10, 3, 23, 59, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="no utcnow"):
        frozen_clock(monkeypatch, at, "sleeve_fund.markets")
    frozen_clock(monkeypatch, at, "sleeve_fund.supervisor")
    from sleeve_fund import supervisor

    assert supervisor.utcnow() == at


def test_the_gap_price_is_where_the_recording_leaves_it():
    assert GAP_PX == pytest.approx(60_900 * 1.6)
