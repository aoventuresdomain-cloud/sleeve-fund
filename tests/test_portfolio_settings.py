"""Portfolio Settings (Bridge Drop A; Advisor BRIDGE-3 and BRIDGE-A-FE, 14 Oct 2026): the five compliance tests with
their use now, edited through a review step into a new version shown in history; the whole-fund test and the loss
brakes can't be switched off or set outside their bounds, on the form and on the server; the asset-group map with
unmapped assets first and flagged; edits apply to new trades only."""

from __future__ import annotations

import re
from decimal import Decimal as D
from html import unescape

import pytest
from test_dashboard import (  # noqa: F401 - the fixtures
    AUTH,
    SAME,
    _strategy_accounts,
    client,
)

from sleeve_fund import compliance_rules as rules


@pytest.fixture
def ps(client):  # noqa: F811
    """The DA's real versioned tables (v1 seeded by the store: BTC and ETH large, SOL and XRP mid), with two assets
    the map leaves out."""
    c, store = client
    for u in ("ADA", "DOGE"):
        store.instrument_id("binance_usdm", f"{u}USDT", "perp", u)
    return c, store


def _n(store) -> int:
    return store.compliance().version


def _form(**over) -> dict:
    f = {"expected_version": "1", "asset_stop_loss.limit": "4", "type_stop_loss.limit": "6",
         "fund_stop_loss.limit": "10", "asset_stop_loss.enabled": "on", "type_stop_loss.enabled": "on", "fund_gross.limit": "1.5", "asset_net.limit": "0.5", "fund_margin.limit": "50",
         "fund_day_loss.limit": "3", "fund_drawdown.limit": "15", "fund_gross.enabled": "on", "asset_net.enabled": "on",
         "fund_margin.enabled": "on", "group.BTC": "large", "group.ETH": "large", "group.SOL": "mid", "group.XRP": "mid"}
    f.update(over)
    return {k: v for k, v in f.items() if v is not None}


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", html)))


def test_the_page_shows_the_five_tests_their_limits_and_the_fixed_lines(ps):
    c, _ = ps
    r = c.get("/portfolio-settings", auth=AUTH)
    assert r.status_code == 200
    t = _text(r.text)
    for words in ("T1 Per asset", "T2 Per asset group", "T3 Whole fund", "T4 Size limits", "T5 Loss brakes",
                  "Edits apply to new trades only.", "can be changed but not switched off",
                  "If the tests can't be read, no new trade goes out.",
                  "An asset with no group can't take a new trade while the per-group test is on."):
        assert words in t
    assert "1.5x" in t and "15%" in t and "50%" in t
    assert t.count("Locked on") == 3  # T3 and both loss brakes
    assert t.count("No group · ideas refused") == 2  # ADA and DOGE: not in the v1 map
    assert "sleeve" not in t.lower()


def test_an_edit_is_reviewed_old_to_new_then_saved_as_a_new_version_in_history(ps):
    c, store = ps
    r = c.post("/portfolio-settings/review", data=_form(**{"asset_stop_loss.limit": "3",
                                                           "type_stop_loss.enabled": None, "group.ADA": "small"}),
               auth=AUTH, headers=SAME)
    assert r.status_code == 200
    t = _text(r.text)
    assert "T1 Per asset: 4% → 3%" in t and "T2 Per asset group: on → off" in t and "ADA group: none → small" in t
    assert _n(store) == 1  # the review writes nothing
    carried = dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)">', r.text))
    r = c.post("/portfolio-settings/apply", data={**carried, "reason_pick": "Tighter while testing"}, auth=AUTH,
               headers=SAME, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/portfolio-settings?saved=2"
    cur = store.compliance()
    new = {t["test"]: t for t in cur.tests}
    assert new["asset_stop_loss"]["limit"] == D("0.03") and not new["type_stop_loss"]["enabled"]
    assert cur.asset_types == {"ADA": "small", "BTC": "large", "ETH": "large", "SOL": "mid", "XRP": "mid"}
    assert cur.note == "Tighter while testing" and cur.created_by == "pm"
    page = _text(c.get("/portfolio-settings?saved=2", auth=AUTH).text)
    assert "Saved as version 2" in page and "Tighter while testing" in page and "T1 Per asset: 4% → 3%" in page
    assert "Large" in page


def test_locked_tests_and_bounds_are_refused_on_the_form_and_on_the_server(ps):
    c, store = ps
    r = c.post("/portfolio-settings/review", data=_form(**{"fund_stop_loss.limit": "12", "fund_margin.limit": "90"}),
               auth=AUTH, headers=SAME)
    t = _text(r.text)
    assert "Not saved." in t and "T3 Whole fund must be between 1% and 10%." in t and "T4 Margin must be between 10% and 80%." in t
    # The form can't switch a locked test off (no checkbox); a hand-made post can't either.
    assert rules.validate({**rules.defaults(), "fund_drawdown": {"limit": D("0.15"), "enabled": False}}) == [
        "T5 Drawdown halt can be changed, but it can't be switched off."]
    r = c.post("/portfolio-settings/apply", data={**_form(**{"fund_day_loss.limit": "6"}), "reason_pick": "Other",
                                                  "reason_note": "testing the bound"}, auth=AUTH, headers=SAME)
    assert r.status_code == 400 and "T5 Daily loss pause must be between 1% and 5%." in r.text
    assert _n(store) == 1


def test_a_stale_form_is_refused_rather_than_overwrite_a_newer_version(ps):
    c, store = ps
    ok = c.post("/portfolio-settings/apply", data={**_form(), "reason_pick": "New asset or group"}, auth=AUTH,
                headers=SAME, follow_redirects=False)
    assert ok.status_code == 303
    again = c.post("/portfolio-settings/apply", data={**_form(**{"fund_gross.limit": "1.2"}),
                                                       "reason_pick": "Tighter while testing"}, auth=AUTH, headers=SAME)
    assert again.status_code == 400 and "changed since this page was opened" in again.text
    assert _n(store) == 2


def test_a_reason_is_required(ps):
    c, store = ps
    r = c.post("/portfolio-settings/apply", data=_form(), auth=AUTH, headers=SAME)
    assert r.status_code == 400 and _n(store) == 1


def test_a_narrower_test_above_a_wider_one_is_a_warning_not_a_refusal(ps):
    c, _ = ps
    r = c.post("/portfolio-settings/review", data=_form(**{"asset_stop_loss.limit": "4", "asset_stop_loss.enabled": "on",
                                                           "type_stop_loss.limit": "3", "type_stop_loss.enabled": "on"}),
               auth=AUTH, headers=SAME)
    t = _text(r.text)
    assert "Per asset (4%) is above per asset group (3%); the per asset group test will bind first." in t
    assert "Apply from the next trade" in t


def test_the_rules_module_round_trips_the_pms_units():
    from sleeve_fund.dashboard import compliance

    tests = rules.defaults()
    fields = dict(compliance.to_form(tests, {"BTC": "mid"}))
    back, groups, errors = compliance.from_form(fields)
    assert not errors and groups == {"BTC": "mid"}
    assert {k: (t["limit"], t["enabled"]) for k, t in back.items()} == {
        k: (t["limit"], t["enabled"] or rules.BY_KEY[k].locked) for k, t in tests.items()}


def test_switching_the_group_test_off_names_the_unmapped_assets_it_stops_refusing(ps):
    c, _ = ps
    r = c.post("/portfolio-settings/review", data=_form(**{"type_stop_loss.enabled": None}),
               auth=AUTH, headers=SAME)
    t = _text(r.text)
    assert ("Assets with no group (ADA, DOGE) will no longer be refused. They still count under per asset, whole fund "
            "and size limits.") in t
    page = _text(c.get("/portfolio-settings", auth=AUTH).text)
    assert "Version 1 · in force since" in page and "set by migration 0027" in page and "in force" in page
    assert "14 Oct 22:00 UK" in page and "UTC" not in page.split("Edit tests and groups")[1].split("const ")[0]  # v1 at 21:00 UTC, in UK time (BST)
    assert "until the next day begins" in page
    assert "Refuse or trim" in page and "Pause new trades" in page and "Flatten and halt" in page


def test_each_row_carries_the_share_of_its_limit_used_as_a_number():
    from types import SimpleNamespace
    from unittest import mock

    from sleeve_fund.dashboard import compliance

    v1 = SimpleNamespace(version=1, tests=[{"test": k, "limit": t["limit"], "enabled": True}
                                           for k, t in rules.defaults().items()], asset_types={"BTC": "large"})
    store = SimpleNamespace(compliance=lambda version=None: v1, instruments=list, compliance_versions=lambda n: [])
    use = {"source": "gate", "fund_gross": {"used": D("0.75"), "what": ""}}
    with mock.patch.object(compliance, "use_now", return_value=use):
        rows = {r["key"]: r for g in compliance.page(store)["tests"] for r in g["rows"]}
    assert rows["fund_gross"]["fill"] == 50.0 and rows["asset_net"]["fill"] is None
