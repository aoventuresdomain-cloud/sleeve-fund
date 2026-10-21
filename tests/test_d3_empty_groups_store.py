"""B-42 follow-up (CR, HoE D3): the store refuses an empty asset-group map while T2 (per asset group) is on, as the form
does, so no caller (a migration, a script, research settings) can write a T2 that judges nothing. With T2 off an empty
map still saves. On Postgres too (TEST_DATABASE_URL)."""

from __future__ import annotations

import pytest
from test_compliance_tests import _edit, engine, store  # noqa: F401

from sleeve_fund import compliance_rules as rules


def test_the_store_refuses_an_empty_group_map_while_t2_is_on(store):  # noqa: F811
    c = store.compliance()
    assert any(t["test"] == "type_stop_loss" and t["enabled"] for t in c.tests)
    with pytest.raises(ValueError, match="T2 Per asset group is on but no asset has a group"):
        store.add_compliance_version(_edit(c), {}, expected_version=c.version, note="no groups", environ={})
    assert store.compliance().version == c.version and store.compliance().asset_types


def test_with_t2_off_the_store_saves_an_empty_group_map(store):  # noqa: F811
    c = store.compliance()
    v = store.add_compliance_version(_edit(c, type_stop_loss=False), {}, expected_version=c.version, note="no groups",
                                     environ={})
    assert v == c.version + 1 and not store.compliance().asset_types


def test_the_form_and_the_store_say_the_same_words():
    assert rules.EMPTY_GROUPS.startswith(rules.name("type_stop_loss") + " is on but no asset has a group")
