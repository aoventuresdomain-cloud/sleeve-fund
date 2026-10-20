"""B-35 regression (QT3's s5 sweep on dfdc2a8, HoQA MUST, HoE top priority): the model the PM picks is the model saved,
or the form says in words why it can't be, never another one. Picking Donchian on the BTC/USDT perpetual moved the pick
to buy and hold without a word, so a minute or hour chip saved and started a model the PM never chose. On spot every
Donchian chip but the daily one was offered, then refused. One source for both: a model refused on the market stays
picked and says why, with nothing saved; the chips offered are the venue's (app.venue_bars) that the model can use
(app.MODEL_BAR)."""

import pytest
from test_browser import _open, _step, browser  # noqa: F401 - the browser fixture
from test_fe_b20_browser import held_site  # noqa: F401 - a fund-held store, served

from sleeve_fund.dashboard.app import MODEL_BAR, _strategy_choices, venue_bars
from sleeve_fund.strategies import PERP_WEIGHT_REFUSAL

WHERE = {"spot": ("instrument=BTC/USD&venue=kraken", "kraken"), "perp": ("instrument=BTC/USDT&venue=binance", "binance")}
REFUSED_ON_PERP = [c["name"] for c in _strategy_choices(tests=True) if c["perp_refused"]]


def _chips(page):
    return page.eval_on_selector_all("input[name=bar_spec]", "els => els.map(e => [e.value, !e.disabled])")


def _try_save(page, spec):
    """Picks `spec` (if offered) and submits as the PM would; returns whether the page moved on. The name the form
    suggested for that chip is left on page.picked_name."""
    page.evaluate("""spec => { const r = document.querySelector(`input[name=bar_spec][value='${spec}']`);
        if (r && !r.disabled) { r.checked = true; r.dispatchEvent(new Event("change", {bubbles: true})); } }""", spec)
    from playwright.sync_api import TimeoutError as Timeout

    page.picked_name = page.input_value("input[name=name]")
    try:
        with page.expect_navigation(timeout=4000):
            page.evaluate("""() => { const f = document.getElementById("sleeve-form");
                f.querySelector("input[name=reason_pick]").checked = true; f.requestSubmit(); }""")
    except Timeout:
        return False
    page.wait_for_load_state("networkidle")
    return True


def _saved(store):
    return {s.name: s.strategy for s in store.sleeves()}


@pytest.mark.parametrize("model", REFUSED_ON_PERP)
def test_a_model_refused_on_a_perpetual_stays_picked_says_why_and_saves_nothing(held_site, browser, model):  # noqa: F811
    base, store = held_site
    before = _saved(store)
    page, errors = _open(browser, f"{base}/sleeves/new?{WHERE['perp'][0]}&strategy={model}")
    _step(page, "Model")
    assert page.input_value("#strategy") == model  # never moved to another model
    note = page.locator("#model-refused")
    assert note.is_visible() and PERP_WEIGHT_REFUSAL.split(":")[0] in note.inner_text(), note.inner_text()
    for spec, offered in _chips(page):
        if offered:
            moved = _try_save(page, spec)
            assert not moved and _saved(store) == before, (model, spec, _saved(store))
            assert page.input_value("#strategy") == model
    assert errors == []
    page.context.close()


def test_the_server_refuses_donchian_on_a_perpetual_in_words_and_saves_nothing(held_site, browser):  # noqa: F811
    """The server is the last word: whatever a form sends, the saved model is the one sent, or none."""
    base, store = held_site
    before = _saved(store)
    page, _ = _open(browser, f"{base}/sleeves/new?{WHERE['perp'][0]}&strategy=donchian")
    page.evaluate("""() => { const s = document.getElementById("sleeve-form").elements.strategy;
        [...s.options].forEach((o) => { o.disabled = false; }); s.value = "donchian";
        const r = document.querySelector("input[name=bar_spec][value='1-HOUR-LAST-INTERNAL']");
        r.disabled = false; r.checked = true; }""")  # what the old script sent, whatever this one does
    with page.expect_navigation():
        page.evaluate("""() => { const f = document.getElementById("sleeve-form");
            f.querySelector("input[name=reason_pick]").checked = true; HTMLFormElement.prototype.submit.call(f); }""")
    words = page.inner_text("main")
    assert "Not saved" in words and "not available on perpetuals" in words, words[:600]
    assert _saved(store) == before
    page.context.close()


def test_donchian_on_spot_offers_only_the_daily_chip_and_saves_as_donchian(held_site, browser):  # noqa: F811
    base, store = held_site
    page, errors = _open(browser, f"{base}/sleeves/new?{WHERE['spot'][0]}&strategy=donchian")
    _step(page, "Model")
    offered = [spec for spec, on in _chips(page) if on]
    assert offered == [MODEL_BAR["donchian"]], offered
    assert MODEL_BAR["donchian"] in venue_bars(WHERE["spot"][1])
    title = page.locator("input[name=bar_spec][value='1-HOUR-LAST-INTERNAL']").locator("xpath=ancestor::label").get_attribute("title")
    assert "decides on daily candles" in title, title
    name = page.input_value("input[name=name]")
    assert _try_save(page, MODEL_BAR["donchian"])
    assert store.sleeve(name).strategy == "donchian" and store.sleeve(name).bar_spec == MODEL_BAR["donchian"]
    assert errors == []
    page.context.close()


def test_moving_off_donchian_offers_every_venue_chip_again(held_site, browser):  # noqa: F811
    base, _ = held_site
    page, errors = _open(browser, f"{base}/sleeves/new?{WHERE['spot'][0]}&strategy=donchian")
    _step(page, "Model")
    page.select_option("#strategy", "ping_pong")
    assert sorted(spec for spec, on in _chips(page) if on) == sorted(venue_bars(WHERE["spot"][1]))
    assert not page.locator("#model-refused").is_visible()
    assert errors == []
    page.context.close()


@pytest.mark.parametrize("market", ["spot", "perp"])
def test_every_model_on_its_default_chip_saves_as_itself_or_says_why(held_site, browser, market):  # noqa: F811
    """saved == picked, for every model the form offers, on its first chip offered."""
    base, store = held_site
    swapped = {}
    for c in _strategy_choices(tests=True):
        page, errors = _open(browser, f"{base}/sleeves/new?{WHERE[market][0]}&strategy={c['name']}")
        _step(page, "Model")
        picked = page.input_value("#strategy")
        assert picked == c["name"], (c["name"], picked)
        spec = next((s for s, on in _chips(page) if on), None)
        moved = _try_save(page, spec) if spec else False
        if moved and "Not saved" not in page.inner_text("main"):
            got = store.sleeve(page.picked_name).strategy
            if got != picked:
                swapped[picked] = got
        elif not moved:
            assert page.locator("#model-refused").is_visible(), (c["name"], market)
        assert errors == [], (c["name"], errors)
        page.context.close()
    assert swapped == {}, swapped
