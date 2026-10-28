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


def test_guards_off_on_a_perp_each_leg_sizes_off_its_own_share_never_the_whole_fund(tmp_path):
    """27 Oct: on margin, a leg sized off the fund asked the whole fund's size on half its cash (2x gross here, 3x on
    three assets; SOL lost its whole account), and the pooled trips no longer added up to the book."""
    _, _, study, runs = _ran(tmp_path, fund_guards=False, market="perp")
    assert runs and all(profile is bk.UNGUARDED for _, profile in runs)
    shares = _shares(study.base.run)
    assert shares and max(shares) < 0.6  # half the fund each, less as a leg has lost
    trips, book = study.pooled.reconciled
    assert not study.pooled.unreconciled and abs(trips - book) < 1e-6
    assert 0.9 < bk.gross_peak(study.base.run) <= bk.GROSS_CAP  # fully invested, never more than the fund
    grown = bk.fill_peaks(study.base.run)
    assert grown and all(0.9 < x <= bk.GROSS_CAP for x, _ in grown.values())  # each fill leaves a leg at 1x of its own
    assert set(study.held) == set(study.basket.names)
    assert all(0.9 < x0 <= x + 0.01 and 0 <= c < 0.05 for x, _, x0, c in study.held.values())
    assert "**Highest held exposure per leg (x own account):**" in bk.render(study)


def test_a_leg_sized_off_the_whole_fund_with_the_guards_off_stops_the_run(tmp_path, monkeypatch):
    """Advisor 27 Oct (b): no leg opens above its own account; the old sizing (each leg off the fund, about 2x its
    own account on two perps) is refused, never a sheet."""
    real = bk.run_portfolio
    monkeypatch.setattr(bk, "run_portfolio", lambda *a, **k: real(*a, **{**k, "fund_sized": True}))
    with pytest.raises(ValueError, match=r"a fill left a leg at [12]\.\d\dx its own account's equity"):
        _ran(tmp_path, fund_guards=False, market="perp")


def test_a_leg_held_past_1x_by_funding_and_fees_is_drift_not_sizing_and_runs(tmp_path, monkeypatch):
    """28 Oct: on the real run all three legs were long for weeks and read about 1.02x of their accounts as funding
    came off their cash. That is the run, not a sizing fault, so it is not refused (the sheet shows it); a fill that
    grows a leg above 1x is."""
    real = bk.run_portfolio

    def drifted(*a, **k):
        run = real(*a, **k)
        for r in run.legs.values():  # every held mark after the first of each position reads 3% higher
            e = r.exposure
            held = (e.abs() > 0) & (e.shift(1, fill_value=0.0).abs() > 0)
            r.exposure.loc[held] = e[held] * 1.03
        return run

    monkeypatch.setattr(bk, "run_portfolio", drifted)
    _, _, study, _ = _ran(tmp_path, fund_guards=False, market="perp")
    assert bk.gross_peak(study.base.run) > bk.GROSS_CAP >= max(x for x, _ in bk.fill_peaks(study.base.run).values())
    assert max(x for x, *_ in study.held.values()) > bk.GROSS_CAP  # shown on the sheet, never a stop


def test_an_add_sized_off_the_whole_fund_stops_the_run_like_an_open(tmp_path, monkeypatch):
    """Advisor 28 Oct (1): every fill that grows a leg is checked, opens and adds alike."""
    real = bk.run_portfolio

    def added(*a, **k):
        run = real(*a, **k)
        r = next(iter(run.legs.values()))
        f = r.fills.sort_values("ts_last", kind="stable")
        first = f.iloc[[0]].copy()  # a second buy on the open's own fill: the leg doubles, as an add off the fund would
        first.index = [str(first.index[0]) + "-add"]
        r.fills = pd.concat([f, first])
        return run

    monkeypatch.setattr(bk, "run_portfolio", added)
    with pytest.raises(ValueError, match=r"a fill left a leg at [12]\.\d\dx its own account's equity"):
        _ran(tmp_path, fund_guards=False, market="perp")


def test_pooled_trips_that_dont_add_up_to_the_book_stop_the_sheet_and_record_no_try(tmp_path, monkeypatch):
    """Advisor 27 Oct (a): with the guards off, a pooled reconcile more than 0.5 points out refuses the whole sheet."""
    real = bk.pooled_legs

    def off(*a, **k):
        legs, (trips, book), why = real(*a, **k)
        return legs, (trips + 0.01, book), why

    monkeypatch.setattr(bk, "pooled_legs", off)
    with pytest.raises(ValueError, match="the trades and the book disagree, so no sheet and no try"):
        _ran(tmp_path, fund_guards=False)
    assert trials.TrialsRegister(Store(f"sqlite:///{tmp_path}/t.db"))._counted(trials.legacy_idea_hash("rules")) == []


def test_at_a_size_scale_every_entry_is_k_of_the_fund_and_the_sheet_is_fitness_only(scaled, unguarded):
    sheet, _, study, runs = scaled
    # guarded, as paper; one run with the guards off counts the legs' unguarded entries for the sheet (Advisor 28 Oct)
    assert runs and [profile is PORTFOLIO for _, profile in runs].count(False) == 1
    assert all(profile is PORTFOLIO or profile is bk.UNGUARDED for _, profile in runs)
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


def test_each_run_shows_its_drawdown_and_the_scaled_run_its_largest_share(unguarded, scaled):
    sheet, _, study, _ = unguarded
    dd = bk.worst_drawdown(study.base.returns)
    assert 0 < dd < 1 and f"**Pooled worst drawdown:** {dd:.2%}" in sheet
    sheet, _, study, _ = scaled
    share = bk.largest_share(study.base.run, study.windows)
    assert f"**Worst drawdown in fund terms:** {bk.worst_drawdown(study.base.returns):.2%}" in sheet
    assert f"**Largest Strategy share of the book:** {share:.1%}" in sheet
    assert 0 < share < 0.4  # 0.3 of the fund at the start, drifting with the Strategy's P&L
    assert bk.worst_drawdown(pd.Series([0.1, -0.5, 0.2])) == pytest.approx(0.5)
    assert bk.worst_drawdown(pd.Series([-0.1, 0.05])) == pytest.approx(0.1)  # from the start, before any gain


def test_a_pooled_sheet_says_its_legs_are_never_rebalanced(unguarded, scaled):
    for sheet, *_ in (unguarded, scaled):
        assert bk.LEGS_NOT_REBALANCED in sheet




def _asks(*rows):
    """Decisions for intents(): (leg, hours after the first bar, side, requested, approved), price 100."""
    from types import SimpleNamespace

    t0 = pd.Timestamp("2025-02-03", tz="UTC")
    return SimpleNamespace(decisions=[
        {"strategy": leg, "ts": t0 + pd.Timedelta(hours=h, seconds=1), "bar_ts": t0 + pd.Timedelta(hours=h),
         "side": side, "price": 100.0, "requested": req, "approved": ok, "limit": None if ok >= req else "pause"}
        for leg, h, side, req, ok in rows])


WINDOW = [(date(2025, 2, 1), date(2025, 3, 2))]
FOUR = {"a": pd.Timedelta(hours=4), "b": pd.Timedelta(hours=4)}


def test_an_entry_refused_on_every_bar_while_its_signal_stands_is_one_intent():
    """Advisor 28 Oct: the halt's 20 refusals, every 4 hours, are one idea, not 20 on both sides of the share."""
    run = _asks(*[("a", 4 * k, "buy", 1.0, 0.0) for k in range(20)], ("b", 0, "buy", 1.0, 1.0))
    found = bk.intents(run, WINDOW, FOUR)
    assert [(i.leg, i.outcome) for i in found] == [("a", "refused"), ("b", "taken")]
    assert bk.gate_touched(run, WINDOW, FOUR) == pytest.approx((100.0, 200.0))  # 50%, not 2,000 of 2,100
    # a bar with no ask ends it: the signal dropped, so the next ask is a new intent
    assert len(bk.intents(_asks(("a", 0, "buy", 1.0, 0.0), ("a", 8, "buy", 1.0, 0.0)), WINDOW, FOUR)) == 2
    assert len(bk.intents(_asks(("a", 0, "buy", 1.0, 0.0), ("a", 4, "sell", 1.0, 0.0)), WINDOW, FOUR)) == 2


def test_an_entry_filled_on_a_later_bar_is_delayed_and_counts_whole_in_the_share():
    """Refused on its first bar (a day pause), filled in full two bars later: another trade than the one the random
    draws copy, so all of its notional is touched, though none of its size was cut."""
    run = _asks(("a", 0, "buy", 1.0, 0.0), ("a", 4, "buy", 1.0, 0.0), ("a", 8, "buy", 1.0, 1.0))
    (i,) = bk.intents(run, WINDOW, FOUR)
    assert i.outcome == "delayed" and (i.filled_at - i.first) / FOUR["a"] == 2 and i.limit == "pause"
    assert bk.gate_touched(run, WINDOW, FOUR) == pytest.approx((100.0, 100.0))


def test_a_trim_then_a_top_up_is_one_intent_touched_by_the_shortfall_left():
    """Trimmed to 0.6 of its ask, topped up by 0.3 on the next bar: one intent of 100 asked, 10 still unfilled; the
    top-up's own ask never counts again."""
    run = _asks(("a", 0, "buy", 1.0, 0.6), ("a", 4, "buy", 0.4, 0.3))
    (i,) = bk.intents(run, WINDOW, FOUR)
    assert i.outcome == "trimmed" and i.filled == pytest.approx(90.0)
    assert bk.gate_touched(run, WINDOW, FOUR) == pytest.approx((10.0, 100.0))
    # topped up in full, nothing is left unfilled and the next ask starts a new intent
    run = _asks(("a", 0, "buy", 1.0, 0.6), ("a", 4, "buy", 0.4, 0.4), ("a", 8, "buy", 0.5, 0.5))
    assert [i.outcome for i in bk.intents(run, WINDOW, FOUR)] == ["taken", "taken"]


def test_a_guarded_scaled_sheet_shows_each_assets_intents_beside_its_unguarded_entries(scaled):
    from qa155_lib import resample

    from sleeve_fund.venues import venue

    h = _History()
    assets = [bk.BasketAsset(venue("KRAKEN").instrument(p.split("/")[0], "USD", price_precision=1),
                             resample(h.minute[p], 240), h.minute[p], bar_minutes=240, exec_minutes=1) for p in PAIRS]
    basket = bk.Basket("k", "rules", {"definition": DEFINITION}, tuple(assets), "2025-01-02")
    first, last = date(2025, 2, 1), date(2025, 3, 2)
    run, names, specs = bk._run(basket, assets, first, last, None, 10_000.0, PORTFOLIO, 180, scale=0.5)
    found = bk.intents(run, [(first, last)], {n: bk._bar(s) for n, s in zip(names, specs)})
    asks = [d for d in run.decisions if first <= bk._decided_on(d) <= last]
    assert len(asks) == 40 and len(found) == 22  # the halt's 20 repeat refusals are 2 intents, one per leg
    cut, asked = bk.gate_touched(run, [(first, last)], {n: bk._bar(s) for n, s in zip(names, specs)})
    assert cut / asked < 0.25  # every ask counted: 48%
    sheet, _, study, _ = scaled
    assert study.unguarded_entries == {"BTC": 10, "ETH": 11}
    assert "**Compliance tests, each intent once**" in sheet and "| ETH | 11 | 11 | 10 | 0 | 1 |" in sheet
    assert "Touched by: T3 Whole fund" in sheet


def test_a_sheet_the_tests_never_touched_shows_no_compliance_table(unguarded):
    sheet, _, study, _ = unguarded
    assert study.intents and all(i.outcome == "taken" for i in study.intents) and not study.unguarded_entries
    assert "Compliance tests, each intent once" not in sheet
