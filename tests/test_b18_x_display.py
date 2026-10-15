"""B-18 (HoE 15 Oct 17:39 UK): the T4 size limits ("x" unit) show 2 decimal places on the page, whatever the value's own
digits. Display only: stored limits and the gate's maths never see it, and the edit form still round-trips its value."""

from decimal import Decimal

import pytest

from sleeve_fund import compliance_rules as rules


@pytest.mark.parametrize("key, value, shown", [
    ("fund_gross", Decimal("0.0312345678"), "0.03x"),
    ("fund_gross", Decimal("1.5"), "1.50x"),
    ("asset_net", 0.5, "0.50x"),
    ("asset_net", Decimal(1), "1.00x"),
    ("fund_gross", Decimal(3), "3.00x"),
    ("fund_gross", 0.125, "0.13x"),
    ("asset_net", Decimal("0.004999"), "0.00x"),
    ("fund_gross", Decimal("1.23456E+1"), "12.35x"),
])
def test_every_x_value_shows_two_decimal_places(key, value, shown):
    assert rules.show(key, value) == shown


def test_percent_limits_are_unchanged():
    assert rules.show("asset_stop_loss", Decimal("0.04")) == "4%"
    assert rules.show("fund_margin", Decimal("0.5")) == "50%"
    assert rules.show("asset_stop_loss", Decimal("0.0025")) == "0.25%"


def test_display_only_the_rule_bounds_and_the_forms_value_round_trip():
    gross = rules.BY_KEY["fund_gross"]
    assert (gross.default, gross.low, gross.high) == (Decimal("1.5"), Decimal("0.1"), Decimal(3))  # stored untouched
    # The form's input is every stored digit (rules.edit, B-18 edge), never the rounded display value.
    assert rules.edit("fund_gross", Decimal("0.125")) == "0.125" and rules.show("fund_gross", Decimal("0.125")) == "0.13x"
    assert rules.edit("asset_net", 0.5) == "0.5" and rules.edit("asset_stop_loss", Decimal("0.04125")) == "4.125"
    assert rules.edit("fund_margin", Decimal(1)) == "100"
    assert Decimal(rules.show("fund_gross", gross.default)[:-1]) == gross.default
    assert Decimal(rules.show("asset_net", Decimal("0.5"))[:-1]) == Decimal("0.5")
