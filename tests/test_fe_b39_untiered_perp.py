"""B-39 (HoE, QT3's sweep on 5814bee3): the New Strategy form offered an ETH/USDT perpetual, whose venue tiers aren't
kept, and in QT3's sweep it saved and the node then crashed at start with NoMaintenanceTable (VS-1). On the venue's real
tables the form is refused at save in the direct POST's words (the strategy's own build, base.py, the one check both
paths go through); the sweep saved because the test harness gives every pair an interim table in-process
(conftest._vs1_interim_tables) while its node ran without it. The pick-list now offers a perpetual only where its tiers
are kept (app.offered, beside app.venue_bars for the candle chips). The supervisor's start refusal is PE4's
(fe-plans/b39/)."""

import pytest
from test_browser import _open, _step, browser  # noqa: F401 - the browser fixture
from test_fe_b20_browser import held_site  # noqa: F401 - a fund-held store, served

VS1 = "has no maintenance table kept"


def _submit(page):
    with page.expect_navigation():
        page.evaluate("""() => { const f = document.getElementById("sleeve-form");
            f.querySelector("input[name=reason_pick]").checked = true; f.requestSubmit(); }""")
    page.wait_for_load_state("networkidle")


@pytest.mark.no_maintenance_tables
def test_b39_the_form_refuses_an_untiered_perp_at_save(held_site, browser):  # noqa: F811
    base, store = held_site
    page, errors = _open(browser, f"{base}/sleeves/new?instrument=ETH/USDT&venue=binance")
    _step(page, "Model")
    page.select_option("#strategy", "dip_buy")
    _submit(page)
    text = page.inner_text("main")
    assert "Not saved" in text and VS1 in text, text[:600]  # the direct POST's words
    assert all(s.instrument != "ETH/USDT" for s in store.sleeves()), [s.name for s in store.sleeves()]
    assert errors == []
    page.context.close()


@pytest.mark.no_maintenance_tables
def test_b39_the_pick_list_offers_a_perp_only_where_its_tiers_are_kept():
    """The venue seed and the full listing both go through app.offered: BTC/USDT perpetual (its tiers kept) is
    offered, ETH/USDT perpetual is not; spot pairs are offered as before."""
    from sleeve_fund.dashboard import app as app_mod
    from sleeve_fund.store import Store

    perp = next(v for v in app_mod.venue_choices() if v["perpetual"])
    assert "BTC/USDT" in perp["pairs"] and "ETH/USDT" not in perp["pairs"], perp["pairs"]
    spot = next(v for v in app_mod.venue_choices() if v["key"] == "kraken")
    assert "ETH/USD" in spot["pairs"]
    listing = {"binance": ["BTC/USDT", "ETH/USDT", "SOL/USDT"], "kraken": ["BTC/USD", "ETH/USD"]}
    store = Store("sqlite://")
    opts = app_mod.instrument_options(store, lambda venue=None: listing.get(venue, []))
    perps = {o["value"] for o in opts if o["group"] == "Perpetuals"}
    assert perps == {"BTC/USDT"}, perps
    assert {"BTC/USD", "ETH/USD"} <= {o["value"] for o in opts if o["group"] == "Spot"}
