"""B-35 (HoE 15:24 UK, ruling (a) 15:44 UK): the PM's first click. Every model the New Strategy form offers saves on
its untouched defaults, with its suggested name, through the real form on a fund-held store, on spot and on the
perpetual. A model that places no stop is held to 1x on a perpetual (strategies.check_perp_stop, the PM's rule
"always a stop, or 1x"), so with no stop set the form picks a profile within 1x and says so, rather than leaving the
2x default for the save to refuse. Donchian's default lookbacks ("20,55,100") are a list, not one number. Two candle
chips of the same length (1h, and the venue's own 1h) suggest different names, so the second is not a duplicate."""

import pytest
from test_browser import _open, _step, browser  # noqa: F401 - the browser fixture
from test_fe_b20_browser import held_site  # noqa: F401 - a fund-held store, served

from sleeve_fund.dashboard.app import _strategy_choices
from sleeve_fund.risk import PROFILES
from sleeve_fund.strategies import STOPLESS_MAX_LEVERAGE, places_stop

MODELS = [c["name"] for c in _strategy_choices(tests=True)]


# The two kinds the PM's form offers on its own: a spot venue's spot, a perpetual venue's perpetual (the form
# follows the venue, venueField).
WHERE = {"spot": "instrument=BTC/USD&venue=kraken", "perp": "instrument=BTC/USDT&venue=binance"}


def _choose(page, model, market):
    assert page.input_value("#market") == market
    _step(page, "Model")
    page.select_option("#strategy", model)


def _submit(page):
    with page.expect_navigation():
        page.evaluate("""() => { const f = document.getElementById("sleeve-form");
            f.querySelector("input[name=reason_pick]").checked = true; f.requestSubmit(); }""")
    page.wait_for_load_state("networkidle")


@pytest.mark.parametrize("market", ["spot", "perp"])
@pytest.mark.parametrize("model", MODELS)
def test_b35_every_model_saves_on_its_untouched_defaults(held_site, browser, model, market):  # noqa: F811
    base, store = held_site
    page, errors = _open(browser, f"{base}/sleeves/new?{WHERE[market]}")
    refusal = next(c["perp_refused"] for c in _strategy_choices(tests=True) if c["name"] == model)
    if market == "perp" and refusal:  # sized by weight: offered greyed with the reason, never a refused save
        option = page.locator(f"#strategy option[value={model}]")
        assert option.is_disabled() and "not available on perpetuals" in option.inner_text(), option.inner_text()
        assert errors == []
        page.context.close()
        return
    _choose(page, model, market)
    name = page.input_value("input[name=name]")
    _submit(page)
    refused = [line for line in page.inner_text("main").splitlines() if "Not saved" in line]
    assert refused == [], (model, market, refused)
    s = store.sleeve(name)
    assert s.starting_balance == 0
    perp = s.params.get("market") == "perp"
    if perp and not places_stop(s.strategy, s.params):
        assert PROFILES[s.risk_profile].max_leverage <= STOPLESS_MAX_LEVERAGE, (model, s.risk_profile)
    assert errors == []
    page.context.close()


def test_b35_a_stopless_perp_says_why_its_held_to_1x_and_a_stop_frees_the_profile(held_site, browser):  # noqa: F811
    base, _ = held_site
    page, errors = _open(browser, f"{base}/sleeves/new?instrument=BTC/USDT&venue=binance")
    _choose(page, "trend_filter", "perp")
    _step(page, "Capital and risk")
    assert PROFILES[page.input_value("#risk_profile")].max_leverage <= STOPLESS_MAX_LEVERAGE
    note = page.locator("#stopless-note")
    assert note.is_visible() and "places no stop" in note.inner_text() and "1x" in note.inner_text()
    _step(page, "Orders and exits")
    page.select_option("#stop_kind", "pct")
    page.fill("#stop_loss_pct", "2")
    _step(page, "Capital and risk")
    assert not note.is_visible()
    page.select_option("#risk_profile", "balanced")
    assert page.input_value("#risk_profile") == "balanced"  # with a stop the PM's choice stands
    page.context.close()
    page, errors = _open(browser, f"{base}/sleeves/new?{WHERE['spot']}")
    _choose(page, "trend_filter", "spot")
    _step(page, "Capital and risk")
    assert not page.locator("#stopless-note").is_visible()  # spot has no leverage to hold
    assert errors == []
    page.context.close()


def test_b35_the_two_1h_chips_suggest_different_names(held_site, browser):  # noqa: F811
    base, _ = held_site
    # On a spot venue: a hub-fed one offers only the 1h built from trades (no offered chip is refused, below).
    page, errors = _open(browser, f"{base}/sleeves/new?{WHERE['spot']}")
    names = []
    for spec in ("1-HOUR-LAST-INTERNAL", "1-HOUR-LAST-EXTERNAL"):
        page.check(f"input[name=bar_spec][value='{spec}']", force=True)
        names.append(page.input_value("input[name=name]"))
    assert names[0] != names[1], names
    assert all(n.startswith("trend-filter-btcusd-1h") for n in names), names
    assert errors == []
    page.context.close()


def test_b35_the_backtest_form_holds_a_stopless_perp_to_1x_as_paper_does(held_site, browser):  # noqa: F811
    """Advisor B35-STOPLESS-DEFAULT's condition: research runs the Strategy at the leverage paper will, 1x."""
    base, _ = held_site
    page, errors = _open(browser, f"{base}/backtest?{WHERE['perp']}&strategy=trend_filter")
    assert PROFILES[page.input_value("#risk_profile")].max_leverage <= STOPLESS_MAX_LEVERAGE
    assert page.locator("#stopless-note").is_visible()
    assert errors == []
    page.context.close()


@pytest.mark.parametrize("market", ["spot", "perp"])
def test_b35_no_offered_candle_chip_is_refused(held_site, browser, market):  # noqa: F811
    """QT3's sweep (HoE 15:32 UK): on a perpetual the form offered "1h venue", 4h and 1d, which the save refused. Every
    chip the form offers saves, with ping pong's untouched defaults, on that chip."""
    base, store = held_site
    page, _ = _open(browser, f"{base}/sleeves/new?{WHERE[market]}")
    chips = page.eval_on_selector_all("input[name=bar_spec]:not([disabled])", "els => els.map(e => e.value)")
    page.context.close()
    assert chips
    refused = {}
    for spec in chips:
        page, errors = _open(browser, f"{base}/sleeves/new?{WHERE[market]}")
        _choose(page, "ping_pong", market)
        page.evaluate("""spec => { const r = document.querySelector(`input[name=bar_spec][value='${spec}']`);
            r.checked = true; r.dispatchEvent(new Event("change", {bubbles: true})); }""", spec)
        name = page.input_value("input[name=name]")
        _submit(page)
        lines = [line for line in page.inner_text("main").splitlines() if "Not saved" in line]
        if lines:
            refused[spec] = lines
        else:
            assert store.sleeve(name).bar_spec == spec
        assert errors == [], (spec, errors)
        page.context.close()
    assert refused == {}, refused
