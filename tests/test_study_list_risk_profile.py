"""A study list's risk settings (HoE 16 Nov, V5's paper benchmark). Done when:

1. "Risk profile: <name>" and "Leverage cap: <x>" in a list's common settings reach the request: the profile on the
   StudyRequest, the cap in the params as a Strategy's own max_leverage; a bad or half-read line is refused;
2. a list without them runs exactly as before: every list on main reads no risk settings, and its requests are equal
   to the ones built without the new argument (risk profile none, no max_leverage);
3. an asset list run under a profile runs every leg under it (as paper), says so on its sheet, and is never a try."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from test_basket_guards import _ran

from sleeve_fund.research import basket as bk
from sleeve_fund.research import rescore, trials
from sleeve_fund.risk import OWN_LEVERAGE, PORTFOLIO

ROOT = Path(__file__).resolve().parents[1]
V5_LIST = ROOT / "configs" / "studies" / "qr01b-v5-basket-guarded-btc-eth-list.md"
V5_2X_LIST = ROOT / "configs" / "studies" / "qr01b-v5-basket-guarded-btc-eth-2x-list.md"
# Every list but the ones that name a risk profile on purpose (V5's paper benchmark, HoE 16 Nov), pinned below.
PROFILED = {V5_2X_LIST}
LISTS = sorted(set((ROOT / "configs" / "studies").glob("*.md")) - PROFILED)
RISK_LINES = "- **Risk profile:** aggressive\n- **Leverage cap:** 2x\n"


# ---- 1. the lines -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, want", [
    ("- **Holdout:** locked\n", {}),
    ("- **Risk:** risk profile \"none\" (research: 1x, uncapped and unguarded)\n", {}),  # the lists' prose line
    ("- **Risk profile:** aggressive\n", {"risk_profile": "aggressive"}),
    ("- **Risk profile:** Balanced\n", {"risk_profile": "balanced"}),
    (RISK_LINES, {"risk_profile": "aggressive", OWN_LEVERAGE: 2.0}),
    ("- **Risk profile:** aggressive\n- **Leverage cap:** 1.5\n", {"risk_profile": "aggressive", OWN_LEVERAGE: 1.5}),
])
def test_a_lists_risk_lines_read(text, want):
    assert rescore.risk_of(text) == want


@pytest.mark.parametrize("text, match", [
    ("- **Risk profile:** reckless\n", "unknown risk profile"),
    ("- **Risk profile:** aggressive or so\n", "names no profile"),  # never half-read
    ("- **Leverage cap:** 2x\n", "names no risk profile"),
    ("- **Risk profile:** aggressive\n- **Leverage cap:** 4x\n", "above the aggressive profile's own"),
    ("- **Risk profile:** conservative\n- **Leverage cap:** 2x\n", "above the conservative profile's own"),
    ("- **Risk profile:** aggressive\n- **Leverage cap:** 0.5x\n", "minimum 1x"),
    ("- **Risk profile:** aggressive\n- **Leverage cap:** 1.255x\n", "2 decimal places"),
    ("- **Risk profile:** aggressive\n- **Leverage cap:** two\n", "number of times"),
])
def test_a_bad_risk_line_is_refused(text, match):
    with pytest.raises(ValueError, match=match):
        rescore.risk_of(text)


def test_the_lines_reach_the_request_as_its_profile_and_its_own_leverage_cap():
    req = rescore.request("v5_lo_adx", 240, "kraken", "BTC/USD,ETH/USD",
                          risk={"risk_profile": "aggressive", OWN_LEVERAGE: 2.0})
    plain = rescore.request("v5_lo_adx", 240, "kraken", "BTC/USD,ETH/USD")
    assert req.risk_profile == "aggressive" and req.params[OWN_LEVERAGE] == 2.0
    assert {k: v for k, v in req.params.items() if k != OWN_LEVERAGE} == plain.params
    assert replace(req, risk_profile=None, params=plain.params) == plain


def test_a_risk_profile_is_an_asset_lists_setting_only():
    with pytest.raises(ValueError, match="asset list's settings only"):
        rescore.request("v5_lo_adx", 240, "kraken", "BTC/USD", risk={"risk_profile": "aggressive"})


# ---- 2. a list without them runs as before --------------------------------------------------------------------------

@pytest.mark.parametrize("path", LISTS, ids=lambda p: p.stem)
def test_every_list_on_main_reads_no_risk_settings_and_builds_the_same_requests(path):
    text = path.read_text(encoding="utf-8")
    assert rescore.risk_of(text) == {}
    try:
        rows, pinned = rescore.rows_of(text), rescore.parse_pinned(text)
    except ValueError:
        pytest.skip("a list the runner refuses before any row is built")
    pair = "BTC/USDT,ETH/USDT" if "basket" in path.stem or "asset" in text.lower() else "BTC/USDT"
    for _, name, minutes in rows[:3]:
        try:
            before = rescore.request(name, minutes, "binance", pair, pinned)
        except ValueError:
            continue  # a row the runner refuses on main is refused the same way: not this change's to pin
        after = rescore.request(name, minutes, "binance", pair, pinned, risk=rescore.risk_of(text))
        assert after == before and after.risk_profile is None and OWN_LEVERAGE not in after.params


def test_the_v5_list_dry_runs_as_before(capfd):
    assert rescore.main([str(V5_LIST), "--venue", "binance", "--pair", "BTC/USDT,ETH/USDT", "--dry-run"]) == 0
    out = capfd.readouterr().out
    assert "risk profile" not in out and "leverage" not in out
    assert "not new tries: fund guards on, size scale 0.346" in out


def test_the_v5_list_with_the_two_lines_dry_runs_at_the_profile_and_cap_and_never_a_try(tmp_path, capfd):
    text = V5_LIST.read_text(encoding="utf-8").replace("- **Fund guards:** on\n", "- **Fund guards:** on\n" + RISK_LINES)
    assert RISK_LINES in text
    listed = tmp_path / "v5-aggressive-2x.md"
    listed.write_text(text, encoding="utf-8")
    assert rescore.main([str(listed), "--venue", "binance", "--pair", "BTC/USDT,ETH/USDT", "--dry-run"]) == 0
    out = capfd.readouterr().out
    assert "not new tries: every leg under the aggressive risk profile, as paper, leverage cap 2x" in out
    assert "dry run: 1 checked, 0 run" in out


def test_the_committed_v5_2x_list_differs_from_the_reference_only_in_its_risk_lines_and_dry_runs_at_them(capfd):
    """V5's paper benchmark (HoE 16 Nov): the 30 Oct reference list with the two risk lines, nothing else changed in
    what runs, and never a try."""
    ref, two = V5_LIST.read_text(encoding="utf-8"), V5_2X_LIST.read_text(encoding="utf-8")
    assert rescore.risk_of(two) == {"risk_profile": "aggressive", OWN_LEVERAGE: 2.0}
    assert rescore.rows_of(two) == rescore.rows_of(ref) and rescore.parse_pinned(two) == rescore.parse_pinned(ref)
    for line in ("- **Fund guards:** on", "- **Size scale:** 0.346", "Fees per side:** 0% and 0.055%"):
        assert line in two and line in ref
    assert rescore.main([str(V5_2X_LIST), "--venue", "binance", "--pair", "BTC/USDT,ETH/USDT", "--dry-run"]) == 0
    out = capfd.readouterr().out
    assert "not new tries: every leg under the aggressive risk profile, as paper, leverage cap 2x" in out
    assert "dry run: 1 checked, 0 run" in out


# ---- 3. the run ---------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def at_profile(tmp_path_factory):
    from test_basket_4h import _req

    tmp = tmp_path_factory.mktemp("risk")
    params = {**_req().params, OWN_LEVERAGE: 2.0}
    return _ran(tmp, size_scale=0.3, risk_profile="aggressive", params=params)


def test_every_leg_runs_under_the_profile_with_its_cap_as_paper(at_profile):
    sheet, _, _, runs = at_profile
    assert runs and all(leg.risk_profile == "aggressive" and leg.params[OWN_LEVERAGE] == 2.0
                        for legs, _ in runs for leg in legs)
    assert all(profile is PORTFOLIO or profile is bk.UNGUARDED for _, profile in runs)  # the fund guards as before
    assert "every leg under the aggressive risk profile, as paper, margined at up to 2x" in sheet
    assert "**Basket verdict:** FITNESS ONLY" in sheet


def test_a_run_under_a_profile_is_never_a_try(at_profile):
    sheet, store, study, _ = at_profile
    assert not study.counted and study.tries == 0
    assert trials.TrialsRegister(store)._counted(trials.legacy_idea_hash("rules")) == []
    assert "**Tries:** not a new try" in sheet


def test_without_a_profile_the_legs_run_as_before(tmp_path):
    _, _, study, runs = _ran(tmp_path, size_scale=0.3)
    assert study.basket.risk_profile is None and all(leg.risk_profile is None for legs, _ in runs for leg in legs)
    assert bk._risk_words(study.basket) == ""


def test_the_profile_stays_out_of_the_baskets_key():
    from test_basket_4h import DEFINITION

    a = bk.Basket("k", "rules", {"definition": DEFINITION}, (), "2025-01-02")
    assert bk.basket_key(replace(a, risk_profile="aggressive")) == bk.basket_key(a)



# ---- 4. a profile alone (CR 16 Nov): never a try, fitness only, and never on an unguarded run ----------------------

@pytest.fixture(scope="module")
def profile_only(tmp_path_factory):
    from test_basket_4h import _req

    tmp = tmp_path_factory.mktemp("profile-only")
    return _ran(tmp, risk_profile="aggressive", params={**_req().params, OWN_LEVERAGE: 2.0})  # guards on, no scale


def test_a_profile_alone_is_never_a_try_and_its_sheet_says_fitness_only(profile_only):
    sheet, store, study, runs = profile_only
    assert not study.counted and study.tries == 0
    assert trials.TrialsRegister(store)._counted(trials.legacy_idea_hash("rules")) == []
    assert "**Basket verdict:** FITNESS ONLY: a guarded run under the aggressive risk profile" in sheet
    assert "**Run:** fund guards on, as paper: fitness only" in sheet
    assert "**Tries:** not a new try: a benchmark under paper's risk caps, never counted" in sheet
    assert "already counted" not in sheet
    assert runs and all(profile is PORTFOLIO for _, profile in runs)


def test_a_risk_profile_is_refused_on_an_unguarded_run(tmp_path, capfd):
    from test_basket_4h import _req

    with pytest.raises(ValueError, match="fund guards on"):
        replace(_req(), fund_guards=False, risk_profile="aggressive").validate()
    text = V5_LIST.read_text(encoding="utf-8").replace("- **Fund guards:** on", "- **Fund guards:** off")
    listed = tmp_path / "v5-unguarded-aggressive.md"
    listed.write_text(text + RISK_LINES, encoding="utf-8")
    assert rescore.main([str(listed), "--venue", "binance", "--pair", "BTC/USDT,ETH/USDT", "--dry-run"]) == 2
    assert "refused: a risk profile runs with the fund guards on" in capfd.readouterr().out
    replace(_req(), fund_guards=False).validate()  # the unguarded try without a profile, as before
