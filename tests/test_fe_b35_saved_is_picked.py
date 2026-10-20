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



def _perp_entry(page):
    """Opens the instrument list, types BTC/USDT and returns its perpetual entry: (greyed, text)."""
    page.fill("#instrument", "")
    page.click("#instrument")
    page.type("#instrument", "BTC/USDT")
    opt = page.locator("#instrument-list [role=option]", has_text="BTC/USDT").filter(has_text="perpetual").first
    opt.wait_for(state="visible", timeout=5000)
    return opt.get_attribute("aria-disabled") == "true", opt.inner_text(), opt


def test_donchian_never_lands_on_a_perpetual_with_no_chip_to_pick(held_site, browser):  # noqa: F811
    """QT3 on merge 9 (s5-donchian.tsv: BTC/USDT perpetual, "FAIL no candle chip offered"): Donchian picked, then the
    perpetual picked from the instrument list, left the PM on a model with every chip greyed, a dead end. With such a
    model picked, the list greys that market with the reason in words and a click on it changes nothing; a model that
    can trade it picks it as before."""
    base, store = held_site
    before = _saved(store)
    page, errors = _open(browser, f"{base}/sleeves/new?{WHERE['spot'][0]}&strategy=donchian")
    greyed, text, opt = _perp_entry(page)
    assert greyed and "Not for this model" in text and PERP_WEIGHT_REFUSAL.split(":")[1].strip() in text, text
    opt.click(force=True)
    assert page.input_value("[data-pl-venue]") == "kraken" and page.input_value("#strategy") == "donchian"
    assert any(offered for _, offered in _chips(page))  # its daily chip is still there to pick
    page.evaluate("""() => { const s = document.getElementById("strategy"); s.value = "buy_and_hold";
        s.dispatchEvent(new Event("change", {bubbles: true})); }""")
    greyed, text, opt = _perp_entry(page)
    assert not greyed, text
    opt.click()
    assert page.input_value("[data-pl-venue]") == "binance"
    assert _saved(store) == before and errors == []
    page.context.close()


def _entry(page, typed, kind):
    page.fill("#instrument", "")
    page.click("#instrument")
    page.type("#instrument", typed)
    opt = page.locator("#instrument-list [role=option]", has_text=typed).filter(has_text=kind).first
    opt.wait_for(state="visible", timeout=5000)
    return opt.get_attribute("aria-disabled") == "true", opt.inner_text()


def test_the_instrument_list_greys_by_the_markets_own_candles_and_only_perps_by_weight(held_site, browser):  # noqa: F811
    """CR's FOLLOW-UP 1 on 6b1324fe: each of plRefuse's two reasons pinned. A model that decides on one candle length
    greys a market whose candles (the venue's own list) don't include it, and not one whose do; a model sized by
    weight greys only the perpetuals."""
    base, _ = held_site
    page, errors = _open(browser, f"{base}/sleeves/new?{WHERE['spot'][0]}&strategy=donchian")
    greyed, text = _entry(page, "BTC/USD", "spot")
    assert not greyed, text  # its daily candles are this market's, and spot takes a model sized by weight
    page.evaluate("""() => { const s = document.getElementById("strategy"); s.value = "buy_and_hold";
        const o = s.selectedOptions[0]; o.dataset.bar = "1-HOUR-LAST-EXTERNAL"; o.dataset.barWords = "It decides on the venue's hourly candles";
        s.dispatchEvent(new Event("change", {bubbles: true})); }""")  # a model held to a candle only spot has
    greyed, text = _entry(page, "BTC/USDT", "perpetual")
    assert greyed and "hourly candles, which this market doesn't offer" in text, text
    greyed, text = _entry(page, "BTC/USD", "spot")
    assert not greyed, text
    assert errors == []
    page.context.close()


def test_clearing_a_model_refusal_keeps_another_checks_block(held_site, browser):  # noqa: F811
    """CR's follow-up on 3934040: guard() clears only the block it set. A submit another check holds off (data-blocked)
    stays off when the picked model stops being refused."""
    base, _ = held_site
    page, errors = _open(browser, f"{base}/sleeves/new?{WHERE['perp'][0]}&strategy=donchian")
    _step(page, "Model")
    assert page.locator("#model-refused").is_visible()
    page.evaluate("""() => document.querySelectorAll("#sleeve-form button[data-label]").forEach((b) => {
        b.setAttribute("data-blocked", ""); b.disabled = true; })""")  # another check holds it off too
    page.select_option("#strategy", "ping_pong")
    assert not page.locator("#model-refused").is_visible()
    state = page.eval_on_selector_all("#sleeve-form button[data-label]",
                                      "bs => bs.map(b => [b.hasAttribute('data-blocked'), b.disabled])")
    assert state and all(s == [True, True] for s in state), state
    assert errors == []
    page.context.close()
