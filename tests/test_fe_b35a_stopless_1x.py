"""B-35 (a) form half (QT3's test_b35_stopless_perp_model_saves_on_its_defaults_capped_at_1x, HoE ruling 20 Oct): a
model that places no stop, made on the BTC/USDT perpetual through the real form with every default untouched, saves
under the risk profile the PM picked, held at 1x by its own leverage cap (risk.OWN_LEVERAGE, QD3's risk half), written
as the number 1. The form says so in words beside the risk profile. Below 1x is refused in words. On Postgres too
(TEST_DATABASE_URL)."""

import pytest
from test_browser import _open, browser  # noqa: F401 - the browser fixture
from test_fe_b20_browser import held_site  # noqa: F401 - a fund-held store, served
from test_fe_b35_form_defaults import WHERE, _choose, _submit

from sleeve_fund import markets, risk
from sleeve_fund.dashboard.app import _form_params


@pytest.mark.parametrize("model", ["buy_and_hold", "trend_filter"])
def test_a_stopless_perp_model_saves_on_its_defaults_at_1x_under_the_picked_profile(held_site, browser, model):  # noqa: F811
    base, store = held_site
    page, errors = _open(browser, f"{base}/sleeves/new?{WHERE['perp']}")
    _choose(page, model, "perp")
    assert page.input_value("#risk_profile") == "balanced"
    note = page.locator("#stopless-note")
    assert note.get_attribute("hidden") is None and "No stop, so capped at 1x" in note.text_content(), note.text_content()
    name = page.input_value("input[name=name]")
    _submit(page)
    assert [x for x in page.inner_text("main").splitlines() if "Not saved" in x] == []
    s = store.sleeve(name)
    assert markets.is_perp(s.params) and s.risk_profile == "balanced"  # the profile as picked, 2x
    assert s.params[risk.OWN_LEVERAGE] == 1 and type(s.params[risk.OWN_LEVERAGE]) in (int, float)
    assert risk.leverage_cap(risk.profile(s.risk_profile), s.params) == 1.0
    assert errors == []
    page.context.close()


def test_the_form_writes_the_number_1_and_refuses_below_1x():
    """QD3 / HoE 20:35 UK: the config refuses the string "1"; the form writes a number. A Strategy's own leverage
    below 1x is refused in words, and a model that places a stop, or spot, gets no cap of its own."""
    form = {"market": "perp"}
    params = _form_params(form, "buy_and_hold")
    assert params[risk.OWN_LEVERAGE] == 1 and not isinstance(params[risk.OWN_LEVERAGE], (str, bool))
    assert risk.OWN_LEVERAGE not in _form_params({}, "buy_and_hold")  # spot
    assert risk.OWN_LEVERAGE not in _form_params({"market": "perp", "stop_loss_pct": "2"}, "buy_and_hold")
    with pytest.raises(ValueError, match="minimum 1x"):
        risk.check_own_leverage(_form_params({"market": "perp", "p_buy_and_hold__max_leverage": "0.5"},
                                             "buy_and_hold")[risk.OWN_LEVERAGE])


def test_the_strategy_page_says_its_own_cap_holds_it_at_1x(held_site, browser):  # noqa: F811
    """CR2's FOLLOW-UP via QD3: the Settings row read the profile's "up to 2x leverage" for a Strategy its own cap
    holds at 1x. It says both."""
    import re

    base, _ = held_site
    page, errors = _open(browser, f"{base}/sleeves/new?{WHERE['perp']}")
    _choose(page, "buy_and_hold", "perp")
    name = page.input_value("input[name=name]")
    _submit(page)
    page.goto(f"{base}/sleeves/{name}")
    row = page.locator("tr", has=page.locator("th", has_text=re.compile(r"^Risk profile$"))).first
    assert "up to 2x leverage, capped at 1x (no stop when saved)" in row.inner_text(), row.inner_text()
    assert errors == []
    page.context.close()
