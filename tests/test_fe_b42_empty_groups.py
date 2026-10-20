"""B-42 (HoQA F-V3-EMPTYGROUPS): Portfolio Settings saved an empty asset-group map with T2 (per asset group) on, and T2
then judged nothing without a word. Review and apply refuse it, saying why; with T2 off an empty map still saves.
On Postgres too (TEST_DATABASE_URL)."""

from __future__ import annotations

from test_dashboard import AUTH, SAME
from test_fe_b31_locked_half_page import (
    client,  # noqa: F401 - the app on the shared store
)
from test_halt_ref import (
    store,  # noqa: F401 - the journal, on Postgres when TEST_DATABASE_URL is set
)
from test_portfolio_settings import _form, _n, _text

WHY = "T2 Per asset group is on but no asset has a group, so it would judge nothing"
NO_GROUPS = {"group.BTC": None, "group.ETH": None, "group.SOL": None, "group.XRP": None}


def test_review_and_apply_refuse_an_empty_group_map_while_t2_is_on(client):  # noqa: F811
    c, store = client  # noqa: F811
    review = _text(c.post("/portfolio-settings/review", data=_form(**NO_GROUPS), auth=AUTH, headers=SAME).text)
    assert "Not saved." in review and WHY in review and "Apply these changes" not in review, review[:600]
    r = c.post("/portfolio-settings/apply", data={**_form(**NO_GROUPS), "reason_pick": "Other", "reason_note": "testing the fund"},
               auth=AUTH, headers=SAME, follow_redirects=False)
    assert r.status_code == 400 and WHY in r.text, r.text
    assert _n(store) == 1 and store.compliance().asset_types


def test_with_t2_off_an_empty_group_map_still_saves(client):  # noqa: F811
    c, store = client  # noqa: F811
    off = _form(**NO_GROUPS, **{"type_stop_loss.enabled": None})
    review = _text(c.post("/portfolio-settings/review", data=off, auth=AUTH, headers=SAME).text)
    assert WHY not in review and "Apply these changes" in review
    r = c.post("/portfolio-settings/apply", data={**off, "confirm_off": "on", "reason_pick": "Other",
                                                  "reason_note": "testing with no groups"}, auth=AUTH, headers=SAME,
               follow_redirects=False)
    assert r.status_code == 303, r.text
    assert _n(store) == 2 and not store.compliance().asset_types
