"""Portfolio Settings (Bridge Drop A; Advisor BRIDGE-3 and BRIDGE-A-FE, 14 Oct 2026): the five compliance tests with
their use now, edited through a review step into a new version shown in history; the whole-fund test and the loss
brakes can't be switched off or set outside their bounds, on the form and on the server; the asset-group map with
unmapped assets counted as the strictest group; edits apply to new trades only."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from decimal import Decimal as D
from html import unescape
from types import SimpleNamespace

import pytest
from test_dashboard import (  # noqa: F401 - the fixtures
    AUTH,
    SAME,
    _strategy_accounts,
    client,
)

from sleeve_fund import compliance_rules as rules


class Versions:
    """Stand-in for the DA's versioned compliance tables (data-architect/bridge-a-data-spec.md §3) until they land:
    the same reader and writer names and shapes, whole versions only, compare-and-set on the version."""

    def __init__(self, assets=("BTC", "SOL")):
        self.assets = list(assets)
        self.versions = [{"version": 1, "created_at": datetime(2026, 10, 14, 20, 0, tzinfo=UTC),
                          "created_by": "migration 0025", "note": None,
                          "tests": [{"test": k, "limit": t["limit"], "enabled": t["enabled"]}
                                    for k, t in rules.defaults().items()], "asset_types": {}}]

    def compliance(self, version=None):
        v = self.versions[-1] if version is None else self.versions[version - 1]
        return SimpleNamespace(version=v["version"], tests=v["tests"], asset_types=v["asset_types"])

    def compliance_versions(self, limit=30):
        return [{k: v[k] for k in ("version", "created_at", "created_by", "note")} for v in self.versions[::-1][:limit]]

    def instruments(self):
        return [{"underlying": a} for a in self.assets]

    def add_compliance_version(self, tests, asset_types, *, created_by, note, expected_version):
        if expected_version != self.versions[-1]["version"]:
            raise ValueError("The tests changed since you opened this page; review again.")
        errors = rules.validate({t["test"]: t for t in tests})
        if errors:
            raise ValueError(" ".join(errors))
        n = len(self.versions) + 1
        self.versions.append({"version": n, "created_at": datetime(2026, 10, 14, 21, n, tzinfo=UTC),
                              "created_by": created_by, "note": note, "tests": tests, "asset_types": dict(asset_types)})
        return n


@pytest.fixture
def ps(client):  # noqa: F811
    c, store = client
    v = Versions()
    for name in ("compliance", "compliance_versions", "instruments", "add_compliance_version"):
        setattr(store, name, getattr(v, name))
    return c, v


def _form(**over) -> dict:
    f = {"expected_version": "1", "asset_stop_loss.limit": "2", "type_stop_loss.limit": "3",
         "fund_stop_loss.limit": "5", "fund_gross.limit": "1.5", "asset_net.limit": "0.5", "fund_margin.limit": "50",
         "fund_day_loss.limit": "3", "fund_drawdown.limit": "15", "fund_gross.enabled": "on", "asset_net.enabled": "on",
         "fund_margin.enabled": "on", "group.BTC": "large"}
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
                  "An asset with no group counts as small, the strictest group."):
        assert words in t
    assert "1.5x" in t and "15%" in t and "50%" in t
    assert t.count("Always on") == 3  # T3 and both loss brakes
    assert "Small (no group set)" in t  # BTC and SOL unmapped at version 1
    assert "sleeve" not in t.lower()


def test_an_edit_is_reviewed_old_to_new_then_saved_as_a_new_version_in_history(ps):
    c, v = ps
    r = c.post("/portfolio-settings/review", data=_form(**{"asset_stop_loss.limit": "1.5",
                                                           "asset_stop_loss.enabled": "on"}), auth=AUTH, headers=SAME)
    assert r.status_code == 200
    t = _text(r.text)
    assert "T1 Per asset: 2% → 1.5%" in t and "T1 Per asset: off → on" in t and "BTC group: none → large" in t
    assert len(v.versions) == 1  # the review writes nothing
    carried = dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)">', r.text))
    r = c.post("/portfolio-settings/apply", data={**carried, "reason_pick": "Tighter while testing"}, auth=AUTH,
               headers=SAME, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/portfolio-settings?saved=2"
    new = {t["test"]: t for t in v.versions[-1]["tests"]}
    assert new["asset_stop_loss"]["limit"] == D("0.015") and new["asset_stop_loss"]["enabled"]
    assert v.versions[-1]["asset_types"] == {"BTC": "large"} and v.versions[-1]["note"] == "Tighter while testing"
    page = _text(c.get("/portfolio-settings?saved=2", auth=AUTH).text)
    assert "Saved as version 2" in page and "Tighter while testing" in page and "T1 Per asset: 2% → 1.5%" in page
    assert "Large" in page


def test_locked_tests_and_bounds_are_refused_on_the_form_and_on_the_server(ps):
    c, v = ps
    r = c.post("/portfolio-settings/review", data=_form(**{"fund_stop_loss.limit": "12", "fund_margin.limit": "90"}),
               auth=AUTH, headers=SAME)
    t = _text(r.text)
    assert "Not saved." in t and "T3 Whole fund must be between 1% and 10%." in t and "T4 Margin must be between 10% and 80%." in t
    # The form can't switch a locked test off (no checkbox); a hand-made post can't either.
    assert rules.validate({**rules.defaults(), "fund_drawdown": {"limit": D("0.15"), "enabled": False}}) == [
        "T5 Drawdown halt can be changed but not switched off."]
    r = c.post("/portfolio-settings/apply", data={**_form(**{"fund_day_loss.limit": "6"}), "reason_pick": "Other",
                                                  "reason_note": "testing the bound"}, auth=AUTH, headers=SAME)
    assert r.status_code == 400 and "T5 Daily loss pause must be between 1% and 5%." in r.text
    assert len(v.versions) == 1


def test_a_stale_form_is_refused_rather_than_overwrite_a_newer_version(ps):
    c, v = ps
    ok = c.post("/portfolio-settings/apply", data={**_form(), "reason_pick": "New asset or group"}, auth=AUTH,
                headers=SAME, follow_redirects=False)
    assert ok.status_code == 303
    again = c.post("/portfolio-settings/apply", data={**_form(**{"fund_gross.limit": "1.2"}),
                                                       "reason_pick": "Tighter while testing"}, auth=AUTH, headers=SAME)
    assert again.status_code == 400 and "changed since you opened this page" in again.text
    assert len(v.versions) == 2


def test_a_reason_is_required(ps):
    c, v = ps
    r = c.post("/portfolio-settings/apply", data=_form(), auth=AUTH, headers=SAME)
    assert r.status_code == 400 and len(v.versions) == 1


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
