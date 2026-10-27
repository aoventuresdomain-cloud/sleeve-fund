"""The basket's confirming runs (Advisor 27 Oct, fast lane, research only). Done when:

1. an asset list can run with the fund guards off: no fund limit trims, pauses or halts it, so its joint random
   draws are judged; the default is guarded, and paper and live never take the switch;
2. an asset list can run guarded at a fixed size scale k: every leg's account holds k of its even share, so each
   entry is k x the leg's current equity, compounding as the unscaled leg does, and the book is (1 - k) x capital
   + k x the unscaled book (Advisor check 2a); its sheet headed fitness only;
3. the unguarded run counts once, as any basket; the scaled run records no trials row, and its sheet says so;
   a pooled sheet says its legs are never rebalanced (Advisor check 2b);
4. a study list sets both by two lines of its common settings, refused on a single-asset row or an opened holdout."""

from __future__ import annotations

import re
from dataclasses import fields, replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from test_basket_4h import DEFINITION, PAIRS, _History, _req

from sleeve_fund.research import basket as bk
from sleeve_fund.research import rescore, trials
from sleeve_fund.research import run as run_mod
from sleeve_fund.risk import PORTFOLIO
from sleeve_fund.store import Store

ROOT = Path(__file__).resolve().parents[1]
LIMITS = ("gross", "net_instrument", "margin", "open_risk", "drawdown", "daily_loss")


# ---- 1. guarded by default; paper and live never unguarded ----------------------------------------------------

def test_a_study_and_a_basket_are_guarded_unless_switched_off():
    req = run_mod.StudyRequest(strategy="rules", pair="BTC/USDT")
    assert req.fund_guards is True and req.size_scale is None
    assert all(getattr(PORTFOLIO, k) is not None for k in LIMITS)  # the fund's own profile has every limit on
    assert all(getattr(bk.UNGUARDED, k) is None for k in LIMITS)
    import inspect

    sig = inspect.signature(bk.basket_study)
    assert sig.parameters["profile"].default is PORTFOLIO and sig.parameters["counted"].default is True
    assert sig.parameters["scale"].default is None


def test_paper_and_live_code_never_reads_the_switch_or_the_unguarded_profile():
    """Only the research study path names them: the runtime's limits are its own profile."""
    allowed = {"sleeve_fund/research/basket.py", "sleeve_fund/research/basket_run.py",
               "sleeve_fund/research/run.py", "sleeve_fund/research/rescore.py"}
    hits = set()
    for path in (ROOT / "sleeve_fund").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if re.search(r"\bUNGUARDED\b|\bfund_guards\b|\bsize_scale\b", text):
            hits.add(str(path.relative_to(ROOT)))
    assert hits == allowed


def test_the_switch_and_the_scale_are_an_asset_lists_only_and_keep_the_holdout_locked():
    single = run_mod.StudyRequest(strategy="rules", pair="BTC/USD", fund_guards=False)
    with pytest.raises(ValueError, match="asset list's settings only"):
        single.validate()
    with pytest.raises(ValueError, match="asset list's settings only"):
        run_mod.StudyRequest(strategy="rules", pair="BTC/USD", size_scale=0.5).validate()
    with pytest.raises(ValueError, match="holdout locked"):
        replace(_req(), fund_guards=False, use_holdout=True).validate()
    for bad in (0, 0.0, -0.1, 1.5, True, "0.5"):
        with pytest.raises(ValueError, match="size scale"):
            replace(_req(), size_scale=bad).validate()
    with pytest.raises(ValueError, match="on or off"):
        replace(_req(), fund_guards="off").validate()
    replace(_req(), fund_guards=False).validate()
    replace(_req(), size_scale=0.55).validate()


def test_a_scaled_entry_is_k_of_the_legs_current_equity_and_the_book_is_linear_in_k():
    """At k = 0.5, unguarded on both: every entry is half the unscaled run's, after the leg's gains and losses too
    (so k scales the leg's current equity, never the starting fund), and the book is 0.5 x capital + 0.5 x the
    unscaled book at every mark."""
    from qa155_lib import resample

    from sleeve_fund.venues import venue

    h = _History()
    assets = [bk.BasketAsset(venue("KRAKEN").instrument(p.split("/")[0], "USD", price_precision=1),
                             resample(h.minute[p], 240), h.minute[p], bar_minutes=240, exec_minutes=1) for p in PAIRS]
    basket = bk.Basket("k", "rules", {"definition": DEFINITION}, tuple(assets), "2025-01-02")
    first, last = date(2025, 2, 1), date(2025, 3, 2)
    full, _, _ = bk._run(basket, assets, first, last, None, 10_000.0, bk.UNGUARDED, 180)
    half, _, specs = bk._run(basket, assets, first, last, None, 10_000.0, bk.UNGUARDED, 180, scale=0.5)
    assert [s.capital for s in specs] == [2_500.0, 2_500.0] and half.unallocated == 5_000.0

    def entries(run):
        return [(d["strategy"], d["ts"], float(d["requested"]))
                for d in run.decisions if d.get("kind", "entry") == "entry"]

    a, b = entries(full), entries(half)
    assert len(a) > 4 and [x[:2] for x in a] == [x[:2] for x in b]
    ratios = [y[2] / x[2] for x, y in zip(a, b)]
    assert max(abs(r - 0.5) for r in ratios) < 2e-3  # lot rounding apart
    assert ((half.book - (5_000.0 + 0.5 * full.book)).abs() / full.book).max() < 2e-3


# ---- 4. the study list's lines ----------------------------------------------------------------------------------

def test_a_study_lists_two_lines_set_the_confirming_run():
    assert rescore.confirming_of("- **Holdout:** locked\n") == {}
    got = rescore.confirming_of("- **Fund guards:** off\n- **Size scale:** 0.55\n")
    assert got == {"fund_guards": False, "size_scale": 0.55}
    assert rescore.confirming_of("- **Fund guards:** on\n") == {"fund_guards": True}
    assert rescore.confirming_of("- **Size scale:** 0.55 or so\n") == {}  # never half-read


def test_a_lists_confirming_lines_reach_the_request():
    name = next(p.stem for p in sorted(rescore.DEFINITIONS.glob("*.toml")))
    req = rescore.request(name, 240, "kraken", "BTC/USD,ETH/USD", confirming={"fund_guards": False})
    assert req.fund_guards is False and req.size_scale is None and req.pairs == ("BTC/USD", "ETH/USD")
    assert rescore.request(name, 240, "kraken", "BTC/USD").fund_guards is True
    assert {f.name for f in fields(req)} >= {"fund_guards", "size_scale"}


# ---- 1-3. the runs ------------------------------------------------------------------------------------------------

def _ran(tmp, **settings):
    store = Store(f"sqlite:///{tmp}/t.db")
    studies, runs = [], []
    mp = pytest.MonkeyPatch()
    real_render, real_run = bk.render, bk.run_portfolio

    def spy_render(study, opened=None):
        studies.append(study)
        return real_render(study, opened)

    def spy_run(legs, profile, *a, **k):
        runs.append((list(legs), profile))
        return real_run(legs, profile, *a, **k)

    mp.setattr(bk, "render", spy_render)
    mp.setattr(bk, "run_portfolio", spy_run)
    try:
        out = run_mod.run_store_study(replace(_req(), **settings), store=store, out_dir=tmp / "sheets",
                                      history=_History())
    finally:
        mp.undo()
    return out.read_text(encoding="utf-8"), store, studies[0], runs


@pytest.fixture(scope="module")
def unguarded(tmp_path_factory):
    return _ran(tmp_path_factory.mktemp("off"), fund_guards=False)


@pytest.fixture(scope="module")
def scaled(tmp_path_factory):
    return _ran(tmp_path_factory.mktemp("k"), size_scale=0.3)


def _shares(run) -> list[float]:
    """Each sent entry's requested notional over the fund's equity at its last mark before it."""
    book = run.book
    out = []
    for d in run.decisions:
        if d.get("kind", "entry") != "entry" or float(d["requested"]) <= 0:
            continue
        equity = float(book[book.index <= pd.Timestamp(d["ts"])].iloc[-1])
        out.append(float(d["requested"]) * float(d["price"]) / equity)
    return out


def test_with_the_guards_off_nothing_is_trimmed_paused_or_halted_and_the_draws_are_judged(unguarded):
    sheet, _, study, runs = unguarded
    assert runs and all(profile is bk.UNGUARDED for _, profile in runs)  # every run of the study, ladder included
    assert study.unguarded and study.gate_touched == 0.0
    run = study.base.run
    assert run.halt is None and run.pauses == [] and not run.refusals
    assert study.pooled.why_not == "" and study.pooled.verdict in ("PASS", "FAIL")
    assert "**Run:** fund guards OFF, research only" in sheet
    assert 0.4 < min(_shares(run)) and max(_shares(run)) < 0.6  # each leg at its own account's free cash: half


def test_at_a_size_scale_every_entry_is_k_of_the_fund_and_the_sheet_is_fitness_only(scaled, unguarded):
    sheet, _, study, runs = scaled
    assert runs and all(profile is PORTFOLIO for _, profile in runs)  # guarded, as paper
    assert all(leg.capital == 10_000.0 * 0.3 / len(legs) for legs, _ in runs for leg in legs)
    shares = _shares(study.base.run)
    # at most 0.3 of each leg's half, less as the leg has lost (the k = 0.5 test above pins the scaling itself)
    assert shares and 0.1 < min(shares) and max(shares) < 0.153
    assert "**Basket verdict:** FITNESS ONLY" in sheet and "fitness only" in sheet
    assert "position_cap_pct" not in study.basket.params  # the scale stays out of the basket and its key
    assert study.key == unguarded[2].key == bk.basket_key(study.basket)


def test_the_unguarded_run_counts_once_and_the_scaled_run_never(unguarded, scaled):
    sheet, store, study, _ = unguarded
    rows = trials.TrialsRegister(store)._counted(trials.legacy_idea_hash("rules"))
    assert study.counted and study.tries == 1 and len(rows) == 1 and rows[0]["definition_hash"] == study.key
    assert "**Tries:** 1 " in sheet
    sheet, store, study, _ = scaled
    assert not study.counted and study.tries == 0
    assert trials.TrialsRegister(store)._counted(trials.legacy_idea_hash("rules")) == []
    assert "**Tries:** not a new try" in sheet


def test_a_pooled_sheet_says_its_legs_are_never_rebalanced(unguarded, scaled):
    for sheet, *_ in (unguarded, scaled):
        assert bk.LEGS_NOT_REBALANCED in sheet
