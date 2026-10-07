"""Every page in a real browser: no script errors, and the forms' scripts actually run.

The Python tests can't see a broken script; a page whose JavaScript throws still renders, but its
summary, wizard and live updates silently stop. Skipped where no Chromium is installed."""

import os
import socket
import threading
import time

import pytest

playwright = pytest.importorskip("playwright.sync_api")

PASSWORD = "browser-pw"


def _chromium(p):
    for kwargs in ({}, {"executable_path": "/opt/pw-browsers/chromium"}):
        if kwargs and not os.path.exists(kwargs["executable_path"]):
            continue
        try:
            return p.chromium.launch(**kwargs)
        except Exception:  # noqa: BLE001 - try the next way, then skip
            continue
    if os.environ.get("CI"):  # CI installs Chromium for these; never let them skip there unnoticed
        pytest.fail("Chromium did not launch")
    pytest.skip("no Chromium to run the browser tests")


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    import httpx
    import uvicorn

    from sleeve_fund.dashboard import app as app_mod
    from sleeve_fund.dashboard import preview
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.store import Store
    from sleeve_fund.venues import KRAKEN

    tmp = tmp_path_factory.mktemp("browser")
    mp = pytest.MonkeyPatch()
    mp.setenv("DASHBOARD_PASSWORD", PASSWORD)
    mp.setenv("TEARSHEET_DIR", str(tmp))
    mp.setattr(app_mod, "TEARSHEETS", tmp)
    mp.setattr(KRAKEN, "daily_history", lambda pair: synthetic_ohlcv(days=400, seed=3, vol=0.03))
    from sleeve_fund.dashboard import charts

    listed = {"kraken": ["BTC/USD", "ETH/USD", "ADA/USD"], "binance": ["BTC/USDT", "ETH/USDT"]}
    mp.setattr(charts, "instruments", lambda get_json=None, venue=None: listed[venue])  # no venue calls
    preview._history.clear()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app_mod.create_app(Store(f"sqlite:///{tmp}/b.db")), port=port,
                                           log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    form = {"name": "eth-trend", "strategy": "trend_filter", "instrument": "ETH/USD", "bar_spec": "1-DAY-LAST-EXTERNAL",
            "starting_balance": "5000", "risk_profile": "balanced", "warmup_bars": "0",
            "p_trend_filter__fast": "10", "p_trend_filter__slow": "30", "reason": "browser test"}
    r = httpx.post(f"{base}/sleeves/new", data=form, auth=("pm", PASSWORD), headers={"origin": base})
    assert r.status_code in (200, 303), r.text[:300]
    # Two marks today: a book under two days old, so the book chart must open on 1D with its points showing.
    marks = Store(f"sqlite:///{tmp}/b.db")
    marks.record_equity("eth-trend", equity=5_000, cash=5_000, qty=0, price=2_500, benchmark=5_000)
    marks.record_equity("eth-trend", equity=5_012.5, cash=5_012.5, qty=0, price=2_510, benchmark=5_010)
    yield base
    server.should_exit = True
    thread.join(timeout=5)
    mp.undo()


@pytest.fixture(scope="module")
def browser():
    with playwright.sync_playwright() as p:
        b = _chromium(p)
        yield b
        b.close()


def _open(browser, url):
    ctx = browser.new_context(http_credentials={"username": "pm", "password": PASSWORD})
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(f"script error: {e}"))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(url)
    page.wait_for_load_state("networkidle")
    return page, errors


PAGES = ["/", "/trades", "/orders", "/alerts", "/risk", "/ops", "/research", "/strategies/trend_filter",
         "/decisions", "/reports", "/accounts", "/settings", "/sleeves/eth-trend", "/backtest",
         "/backtest?run=1&instrument=ETH/USD&strategy=trend_filter&p_trend_filter__fast=5&p_trend_filter__slow=20"]


@pytest.mark.parametrize("path", PAGES)
def test_page_runs_without_script_errors(site, browser, path):
    page, errors = _open(browser, site + path)
    assert "Couldn't run it" not in page.content()
    assert errors == []
    page.context.close()


def test_book_chart_opens_young_books_on_1d_and_switches_to_percent(site, browser):
    page, errors = _open(browser, site + "/")
    page.wait_for_selector("#curve-h ~ * .chart-legend, .chart-legend", timeout=5000)
    assert page.get_attribute('[data-days="1"]', "aria-pressed") == "true"
    assert page.locator("#eq canvas").count() >= 1  # drawn by Lightweight Charts, not Chart.js
    legend = page.inner_text(".chart-legend")
    assert "Book" in legend and "5,012.50" in legend
    page.click('[data-unit="pct"]')
    assert page.get_attribute('[data-unit="pct"]', "aria-pressed") == "true"
    assert "%" in page.inner_text(".chart-legend").split("Drawdown")[0]
    assert errors == []
    page.context.close()


@pytest.mark.usefixtures("maker_on")
@pytest.mark.parametrize("query", ["", "?instrument=ETH/USD&strategy=trend_filter&from=backtest"
                                       "&bar_spec=1-DAY-LAST-EXTERNAL&warmup_bars=40&execution=maker"])
def test_new_strategy_form_scripts_run(site, browser, query):
    page, errors = _open(browser, f"{site}/sleeves/new{query}")
    assert errors == []
    items = page.locator("#summary li")
    assert items.count() >= 4  # the plain-English summary was built
    assert page.locator(".wiz-nav").count() == 1  # and the step-by-step wizard took over
    if "maker" in query:
        assert "post-only limit" in page.inner_text("#summary")
    page.context.close()


@pytest.mark.usefixtures("maker_on")
def test_order_type_shows_the_wait_only_for_maker_orders(site, browser):
    page, errors = _open(browser, f"{site}/backtest")
    assert page.is_hidden("#maker_wait_minutes")
    page.select_option("#execution", "maker")
    assert page.is_visible("#maker_wait_minutes")
    assert errors == []
    page.context.close()


@pytest.mark.parametrize("path", PAGES)
def test_page_fits_a_phone_without_sideways_scrolling(site, browser, path):
    ctx = browser.new_context(http_credentials={"username": "pm", "password": PASSWORD},
                              viewport={"width": 390, "height": 844})
    page = ctx.new_page()
    page.goto(site + path)
    page.wait_for_load_state("networkidle")
    wide = page.evaluate("""() => {
        const vw = document.documentElement.clientWidth;
        const out = [];
        for (const el of document.querySelectorAll('body *')) {
            const r = el.getBoundingClientRect();
            if (r.width && r.right > vw + 1 && !el.closest('.scroll-x, .table-wrap')) out.push(
                `${el.tagName.toLowerCase()}.${[...el.classList].join('.')}#${el.id} right=${Math.round(r.right)}`);
        }
        return {sw: document.documentElement.scrollWidth, vw, out: out.slice(0, 8)};
    }""")
    ctx.close()
    assert wide["sw"] <= wide["vw"], wide


def test_backtest_result_comes_before_its_settings_on_a_phone(site, browser):
    ctx = browser.new_context(http_credentials={"username": "pm", "password": PASSWORD},
                              viewport={"width": 390, "height": 844})
    page = ctx.new_page()
    page.goto(site + PAGES[-1])
    page.wait_for_load_state("networkidle")
    head, form = page.locator(".bt-head").bounding_box(), page.locator("#bt-form").bounding_box()
    if os.environ.get("SCREENSHOT_DIR"):
        page.screenshot(path=os.path.join(os.environ["SCREENSHOT_DIR"], "backtest-390.png"), full_page=False)
    ctx.close()
    assert head["y"] < form["y"]


@pytest.mark.parametrize("width", [390, 768, 1440])
def test_a_long_short_backtests_trades_say_long_or_short_at_every_width(site, browser, width):
    """Round 12, M12-U5: the side was only under Size, hidden at 1440 px and on phone cards, then only on the
    entry cell, also hidden on phone cards. Every trade of a long/short run shows its side at every width."""
    ctx = browser.new_context(http_credentials={"username": "pm", "password": PASSWORD},
                              viewport={"width": width, "height": 900})
    page = ctx.new_page()
    page.goto(site + "/backtest?run=1&instrument=ETH/USD&strategy=rsi_bands&market=perp&allow_short=1"
              "&risk_profile=conservative")
    page.wait_for_load_state("networkidle")
    cells = page.locator("section[aria-labelledby=bt-tr-h] tbody tr:not(.detail) td:first-child")
    sides = [cells.nth(i).inner_text() if cells.nth(i).is_visible() else "" for i in range(cells.count())]
    ctx.close()
    assert sides and all(s.rstrip().endswith(("long", "short")) for s in sides), sides[:5]
    assert any(s.rstrip().endswith("short") for s in sides)


def test_a_new_strategy_can_be_set_up_without_typing(site, browser):
    """UI v2, item 9: the instrument, candle length, name and reason are all picked; nothing is typed."""
    page, errors = _open(browser, f"{site}/sleeves/new")
    page.click("#instrument")
    page.wait_for_selector("#instrument-list .opt >> text=ETH/USDT perpetual")
    assert page.locator("#instrument-list .grp").all_text_contents()[:2] == ["Perpetuals", "Spot"]
    page.keyboard.press("ArrowDown")
    page.keyboard.press("Enter")  # the keyboard picks too
    assert page.input_value("#instrument") == "ETH/USDT" and page.input_value("input[name=venue]") == "binance"
    assert page.input_value("#market") == "perp"  # a perpetual venue trades its perpetual
    page.click("#instrument")
    page.click("#instrument-list .opt >> text=ADA/USD spot")
    assert page.input_value("#instrument") == "ADA/USD" and page.input_value("input[name=venue]") == "kraken"
    assert page.input_value("#market") == "spot"
    page.click(".chip-pick span:text-is('4h')")
    assert page.input_value("#name") == "trend-filter-adausd-4h"
    page.click(".wiz-nav button:has-text('Name and reason')")
    page.click("fieldset.reasons label.opt:has-text('New test')")
    page.evaluate("document.getElementById('sleeve-form').requestSubmit()")
    page.wait_for_url("**/sleeves/trend-filter-adausd-4h")
    assert errors == []
    page.context.close()


def test_a_reason_dialog_is_searchable_and_needs_no_typing(site, browser):
    page, errors = _open(browser, f"{site}/sleeves/eth-trend")
    opener = page.locator("[data-open=dlg-start], [data-open=dlg-pause]").first
    which = opener.get_attribute("data-open")
    word, reason = ("review", "Restart after review") if which == "dlg-start" else ("outage", "Data or venue problem")
    opener.click()
    dlg = page.locator(f"#{which}")
    confirm = dlg.locator("[data-needs-reason]").first
    assert confirm.is_disabled()
    dlg.locator("[data-reason-search]").fill(word)
    shown = [t.split("\n")[0] for t in dlg.locator("label.opt:visible").all_inner_texts()]
    assert shown == [reason, "+ Write your own reason"]
    dlg.locator("[data-reason-search]").fill("")
    dlg.locator(f"label.opt:has-text('{reason}')").click()  # a click is all it takes
    assert confirm.is_enabled() and f"Logged as: {reason}" in dlg.inner_text()
    assert errors == []
    page.context.close()


def test_development_columns_end_on_the_same_line(site, browser):
    """UI v2, item 8: at 1440 x 900 the model list and the study are the same height; the list scrolls."""
    ctx = browser.new_context(http_credentials={"username": "pm", "password": PASSWORD},
                              viewport={"width": 1440, "height": 900})
    page = ctx.new_page()
    page.goto(f"{site}/research")
    page.wait_for_load_state("networkidle")
    col, study = page.locator(".plan-col").bounding_box(), page.locator("#study").bounding_box()
    if os.environ.get("SCREENSHOT_DIR"):
        page.screenshot(path=os.path.join(os.environ["SCREENSHOT_DIR"], "development-1440.png"), full_page=True)
    ctx.close()
    assert abs((col["y"] + col["height"]) - (study["y"] + study["height"])) <= 2


def test_records_and_risk_charts_are_lightweight_charts(site, browser):
    # QA U10: no hand-drawn SVG line charts left; Records' return curve is drawn by Lightweight Charts.
    page, errors = _open(browser, site + "/records")
    assert page.locator("svg.rec-curve").count() == 0
    if page.locator(".lw-line.rec-curve").count():
        page.wait_for_selector(".lw-line.rec-curve canvas", timeout=5000)
    page.context.close()
    risk, risk_errors = _open(browser, site + "/risk")
    assert risk.locator(".rh-chart svg").count() == 0 and risk.locator(".lw-line.rh-dd").count() == 1
    # QA U9: no hint under the price chart; the attribution sits once in the side rail.
    assert "Tap a buy or sell arrow" not in risk.content() and risk.locator(".rail-foot a[href*=tradingview]").count() == 1
    assert errors == [] and risk_errors == []
    risk.context.close()


def test_a_models_sentence_fills_placeholders_that_carry_a_format(site, browser):
    # QA U11: "{long_entry:g}" was left raw because the picker's pattern skipped a format spec.
    page, errors = _open(browser, site + "/backtest?strategy=rsi_bands")
    desc = page.locator('.params[data-strategy="rsi_bands"] .desc')
    assert desc.count() == 1
    text = desc.inner_text()
    assert "{" not in text and "}" not in text, text
    assert errors == []
    page.context.close()


def test_risk_limits_table_fits_its_panel_on_a_desktop(site, browser):
    # QA U3: the nine-column Limits table fits the panel at 1440 wide, Risk to stop included, with no sideways scroll.
    ctx = browser.new_context(http_credentials={"username": "pm", "password": PASSWORD}, viewport={"width": 1440, "height": 900})
    page = ctx.new_page()
    page.goto(site + "/risk#limits")
    page.wait_for_load_state("networkidle")
    page.click('a[data-tab="limits"]')
    size = page.evaluate("() => { const s = document.querySelector('#lim-h').closest('section');"
                         " return [s.scrollWidth, s.clientWidth, s.querySelectorAll('thead th').length]; }")
    assert size[2] == 9 and size[0] <= size[1], size
    ctx.close()


def _fixture_candles(n=40, day=86400, start=1_760_000_000 - 1_760_000_000 % 86400 - 40 * 86400):
    """Daily candles and the strategy's own recorded indicators in the agreed shape (v2/chart-indicators-shape.md):
    t is the bar's close, settled_from marks where warm-up ends."""
    times = [start + i * day for i in range(n)]
    candles = [{"time": t, "open": 100 + i, "high": 102 + i, "low": 99 + i, "close": 101 + i} for i, t in enumerate(times)]
    ema = [[t + day, 100.5 + i] for i, t in enumerate(times)]
    rsi = [[t + day, 40 + (i % 20)] for i, t in enumerate(times)]
    return {"interval": 1440, "source": "venue", "candles": candles, "volume": [{"time": t, "value": 1.0} for t in times],
            "markers": [], "notes": {}, "lines": [], "intervals": ["1d"], "chosen": "1d", "pair": "ETH/USD",
            "home": "ETH/USD", "pairs": [],
            "indicators": [
                {"key": "ema", "label": "EMA(10)", "pane": "price", "kind": "line", "group": None,
                 "settled_from": times[10] + day, "points": ema},
                {"key": "rsi", "label": "RSI(14)", "pane": "lower", "kind": "line", "levels": [30, 70],
                 "settled_from": times[14] + day, "points": rsi},
                {"key": "div", "label": "RSI divergence", "pane": "price", "kind": "marker",
                 "points": [[times[20] + day, "bull"]]}]}


def test_the_strategys_recorded_indicators_are_drawn_with_warm_up_marked_and_nothing_recomputed(site, browser):
    import json

    fixture = _fixture_candles()
    ctx = browser.new_context(http_credentials={"username": "pm", "password": PASSWORD})
    ctx.route("**/api/sleeves/eth-trend/candles*", lambda route: route.fulfill(
        status=200, content_type="application/json", body=json.dumps(fixture)))
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(f"script error: {e}"))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(site + "/sleeves/eth-trend")
    page.wait_for_load_state("networkidle")
    page.wait_for_selector(".pc-legend span", timeout=5000)
    legend = page.inner_text(".pc-legend")
    assert "EMA(10)" in legend
    assert page.locator(".pc-sub").count() == 1 and "RSI(14)" in page.inner_text(".pc-sub-legend")
    note = page.inner_text(".pc-strat-note")
    assert "Shaded: the model's indicators were still warming up; any trade here is marked unsettled" in note
    assert errors == []
    ctx.close()


def test_the_overlay_names_the_candle_size_it_was_recorded_on_and_groups_share_a_colour(site, browser):
    import json

    fixture = _fixture_candles()
    for i in fixture["indicators"]:  # recorded on 4-hour candles, on a daily chart
        if i["kind"] == "line":
            i["points"] = [[p[0] - 86400 + 14400 * (k % 6), p[1]] for k, p in enumerate(i["points"])]
    fixture["indicators"].append({"key": "bb.upper", "label": "Bollinger upper", "pane": "price", "kind": "line",
                                  "group": "bb", "settled_from": None, "points": [[p[0], 130.0] for p in fixture["indicators"][0]["points"]]})
    ctx = browser.new_context(http_credentials={"username": "pm", "password": PASSWORD})
    ctx.route("**/api/sleeves/eth-trend/candles*", lambda route: route.fulfill(
        status=200, content_type="application/json", body=json.dumps(fixture)))
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(f"script error: {e}"))
    page.goto(site + "/sleeves/eth-trend")
    page.wait_for_load_state("networkidle")
    page.wait_for_selector(".pc-legend span", timeout=5000)
    assert "minute candles, so it shows only on that interval" in page.inner_text(".pc-strat-note")
    assert errors == []
    ctx.close()


def test_a_settings_change_rebuilds_the_overlay_labels_and_levels_and_warm_up_joins_the_line(site, browser):
    """CR on #173: the same keys with a new label and guide levels (an RSI period or entry level changed) redraw
    the legend and the strip; the dashed warm-up reaches the first settled point."""
    import json

    first = _fixture_candles()
    second = json.loads(json.dumps(first))
    for i in second["indicators"]:
        if i["key"] == "rsi":
            i["label"], i["levels"] = "RSI(7)", [25, 75]
    served = [first]
    ctx = browser.new_context(http_credentials={"username": "pm", "password": PASSWORD})
    ctx.route("**/api/sleeves/eth-trend/candles*", lambda route: route.fulfill(
        status=200, content_type="application/json", body=json.dumps(served[0])))
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(f"script error: {e}"))
    page.goto(site + "/sleeves/eth-trend")
    page.wait_for_load_state("networkidle")
    page.wait_for_selector(".pc-sub-legend", timeout=5000)
    assert "RSI(14)" in page.inner_text(".pc-sub-legend")
    served[0] = second
    page.click(".pc-intervals button")  # a live re-render with the new settings, no page reload
    page.wait_for_function("document.querySelector('.pc-sub-legend').innerText.includes('RSI(7)')", timeout=5000)
    assert "RSI(7)" in page.inner_text(".pc-sub-legend") and errors == []
    ctx.close()


def test_the_platforms_own_sentence_shows_when_it_has_no_indicators_to_draw(site, browser):
    import json

    fixture = _fixture_candles()
    fixture["indicators"], fixture["indicators_note"] = [], "The strategy's indicators are drawn on its own 1h candles."
    ctx = browser.new_context(http_credentials={"username": "pm", "password": PASSWORD})
    ctx.route("**/api/sleeves/eth-trend/candles*", lambda route: route.fulfill(
        status=200, content_type="application/json", body=json.dumps(fixture)))
    page = ctx.new_page()
    page.goto(site + "/sleeves/eth-trend")
    page.wait_for_load_state("networkidle")
    page.wait_for_selector(".pc-legend", timeout=5000)
    assert "drawn on its own 1h candles" in page.inner_text(".pc-strat-note")
    ctx.close()


def test_recorded_lines_are_steps_with_a_toggle_and_unshown_ones_wait_until_asked_for(site, browser):
    """Advisor rules via QD: values hold from close to close (never joined by straight segments), only the lines the
    strategy's rules read are drawn at first, the others are offered, and each drawn line can be hidden again."""
    import json

    fixture = _fixture_candles()
    fixture["indicators"].append({"key": "atr", "label": "ATR(14)", "pane": "lower", "kind": "line", "shown": False,
                                  "settled_from": None, "points": [[p[0], 2.0] for p in fixture["indicators"][0]["points"]]})
    ctx = browser.new_context(http_credentials={"username": "pm", "password": PASSWORD})
    ctx.route("**/api/sleeves/eth-trend/candles*", lambda route: route.fulfill(
        status=200, content_type="application/json", body=json.dumps(fixture)))
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(f"script error: {e}"))
    page.goto(site + "/sleeves/eth-trend")
    page.wait_for_load_state("networkidle")
    page.wait_for_selector(".pc-strat-more button", timeout=5000)
    assert page.locator(".pc-sub").count() == 1 and "ATR(14)" not in page.inner_text(".pc-sub-legend")
    assert "Also recorded, not drawn" in page.inner_text(".pc-strat-more")
    page.click(".pc-strat-more button")
    page.wait_for_function("document.querySelectorAll('.pc-sub').length === 2", timeout=5000)
    legend = page.locator(".pc-legend span", has_text="EMA(10)")
    legend.click()
    assert page.locator(".pc-legend span", has_text="EMA(10)").evaluate("e => e.style.opacity") == "0.4"
    assert errors == []
    ctx.close()


def test_recorded_decisions_are_drawn_once_each_on_closed_candles_and_never_ahead(site, browser):
    import json
    import time

    fixture = _fixture_candles()
    day = 86400
    t = [c["time"] for c in fixture["candles"]]
    future = (int(time.time()) // day + 5) * day
    fixture["decisions"] = [
        {"kind": "fill", "side": "buy", "t": t[12] + 3600, "signal_t": t[11] + day, "price": 112.0, "reason": "ema cross", "code": ""},
        {"kind": "missed", "side": "buy", "t": t[20] + day, "signal_t": t[20] + day, "price": None, "reason": "Halted: daily loss pause", "code": "blocked"},
        {"kind": "missed", "side": "sell", "t": t[25] + day, "signal_t": t[25] + day, "price": None, "reason": "Stale data: no candle for 3 minutes", "code": "stale_data"},
        {"kind": "missed", "side": "buy", "t": future, "signal_t": future, "price": None, "reason": "ahead of its candle", "code": "blocked"},
        {"kind": "missed", "side": "buy", "t": 5, "signal_t": 5, "price": None, "reason": "no such candle", "code": "blocked"},
    ]
    ctx = browser.new_context(http_credentials={"username": "pm", "password": PASSWORD})
    ctx.route("**/api/sleeves/eth-trend/candles*", lambda route: route.fulfill(
        status=200, content_type="application/json", body=json.dumps(fixture)))
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(f"script error: {e}"))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(site + "/sleeves/eth-trend")
    page.wait_for_load_state("networkidle")
    page.wait_for_selector(".pc-legend span", timeout=5000)
    drawn = json.loads(page.get_attribute(".pc-canvas", "data-decisions"))
    assert drawn == {"fills": 1, "missed": 2}
    assert errors == []
    ctx.close()


def test_the_warm_up_shading_follows_the_models_own_lines_not_ones_the_pm_adds(site, browser):
    import json

    fixture = _fixture_candles()
    t = [c["time"] for c in fixture["candles"]]
    fixture["indicators"][0]["shown"] = True
    fixture["indicators"][1]["shown"] = False
    fixture["indicators"][1]["settled_from"] = t[30] + 86400  # a slow line the PM has not turned on
    ctx = browser.new_context(http_credentials={"username": "pm", "password": PASSWORD})
    ctx.route("**/api/sleeves/eth-trend/candles*", lambda route: route.fulfill(
        status=200, content_type="application/json", body=json.dumps(fixture)))
    page = ctx.new_page()
    page.goto(site + "/sleeves/eth-trend")
    page.wait_for_load_state("networkidle")
    page.wait_for_selector(".pc-legend span", timeout=5000)
    assert page.get_attribute(".pc-canvas", "data-warm") == "10"
    ctx.close()


def test_decisions_that_do_not_line_up_with_this_candle_size_say_so(site, browser):
    import json

    fixture = _fixture_candles()
    fixture["decisions"] = [{"kind": "missed", "side": "buy", "t": 0, "signal_t": fixture["candles"][5]["time"] + 14400,
                             "price": None, "reason": "Halted", "code": "blocked"}]
    ctx = browser.new_context(http_credentials={"username": "pm", "password": PASSWORD})
    ctx.route("**/api/sleeves/eth-trend/candles*", lambda route: route.fulfill(
        status=200, content_type="application/json", body=json.dumps(fixture)))
    page = ctx.new_page()
    page.goto(site + "/sleeves/eth-trend")
    page.wait_for_load_state("networkidle")
    page.wait_for_selector(".pc-legend span", timeout=5000)
    assert "show only on that interval" in page.inner_text(".pc-strat-note")
    assert page.get_attribute(".pc-canvas", "data-decisions") == '{"fills":0,"missed":0}'
    ctx.close()


def _results_sheet(trades: int) -> str:
    from sleeve_fund.research.guardrails import G1_RULES

    verdict = "PASS" if trades >= 100 else "FAIL"
    return f"""# Tear sheet: rsi_pullback

Tested on `BTC/USDT` at 60-minute bars on `binance`

Settings: balanced risk profile

G1 rules: {G1_RULES}

Dataset `store-real` · research period 01 Jan 2025 to 01 Oct 2026 · holdout: last 90 days (untouched) · fees: 0.05% taker

## G1 checks

**G1: {'PASS' if trades >= 100 else 'FAIL'}** (x)

| Check | Result | Evidence |
| --- | --- | --- |
| G1 test: out-of-sample Sharpe clearly beats benchmark after fees | PASS | Sharpe 1.2 vs 0.4 |
| Holds at nearby settings | PASS | all 4 neighbours of the chosen settings still beat buy and hold |
| Break-even fee (shown, not a test) | INFO | about 0.12% per side |
| Enough out-of-sample trades to judge | {verdict} | {trades} closed in the 5 walk-forward test windows, opened there too (bar: 100); 300 over the full research period |

**Break-even fee:** about 0.12% per side.

## Idea counter

- 12 distinct variants of this idea tried so far, in studies, backtests and paper strategies (40 evaluations including walk-forward refits): the N its results are judged by.
- 3 ideas and 20 distinct variants tested so far across the project (60 evaluations), shown for awareness. By family: trend 2
- Deflated Sharpe: 97% probability the out-of-sample Sharpe is real rather than the best of many tries (higher is better; 95% is a strong bar).
"""


def test_results_page_shows_the_guardrail_figures_and_says_not_judged_under_100_trades(site, browser):
    """P1-6: the Results page reads the study's tear sheet. A study under 100 out-of-sample trades says not judged."""
    from sleeve_fund.dashboard import app as app_mod

    (app_mod.TEARSHEETS / "rsi_pullback_20261007-100000.md").write_text(_results_sheet(84), encoding="utf-8")
    (app_mod.TEARSHEETS / "rsi_pullback_20261007-110000.md").write_text(_results_sheet(150), encoding="utf-8")
    page, errors = _open(browser, site + "/results/rsi_pullback_20261007-100000")
    text = page.locator("main").inner_text()
    assert "12 variants of this idea" in text and "Deflated Sharpe 97%" in text, text
    assert "0.12%" in text and "Holds" in text
    assert "84 of 100 needed" in text and "Not judged. 84 trades is fewer than the 100 needed" in text
    assert "Not run yet" in page.locator("#ablation").inner_text()
    assert "UK" in text and "UTC" not in text
    # Dark mode readable: the page sits on the desk's dark panel, text well clear of it.
    bg, fg = page.evaluate("""() => { const p = getComputedStyle(document.querySelector('#nearby'));
        return [p.backgroundColor, getComputedStyle(document.querySelector('#nearby p')).color]; }""")
    def lum(c):
        parts = c[c.index("(") + 1:-1].split(",")[:3]
        return sum(int(x) * w for x, w in zip(parts, (0.2126, 0.7152, 0.0722)))

    assert lum(fg) - lum(bg) > 90, (bg, fg)
    assert errors == []
    page.context.close()
    page, errors = _open(browser, site + "/results/rsi_pullback_20261007-110000")
    text = page.locator("main").inner_text()
    assert "150 of 100 needed" in text and "Not judged. 150" not in text and errors == []
    page.context.close()


def test_a_liquidated_strategy_says_what_clears_it_and_its_reset_shows_refusals(site, browser):
    """P1-RAL screen: the page names why it is halted and the one thing that clears it; the dialog shows the loss, asks
    for the incident note, refuses a reset without it in words, and sends the reset once the note is written."""
    import httpx

    from sleeve_fund.dashboard import app as app_mod
    from sleeve_fund.store import Store

    form = {"name": "liq-trend", "strategy": "trend_filter", "instrument": "ETH/USD", "bar_spec": "1-DAY-LAST-EXTERNAL",
            "starting_balance": "5000", "risk_profile": "balanced", "warmup_bars": "0",
            "p_trend_filter__fast": "10", "p_trend_filter__slow": "30", "reason": "browser test"}
    r = httpx.post(f"{site}/sleeves/new", data=form, auth=("pm", PASSWORD), headers={"origin": site})
    assert r.status_code in (200, 303), r.text[:300]
    store = Store(f"sqlite:///{app_mod.TEARSHEETS}/b.db")
    store.set_desired_state("liq-trend", "running")
    store.event("liq-trend", "error", "liquidation", "Liquidated: the price 2,000 gapped through 2,100")
    store.set_status("liq-trend", "halted",
                     "Position margin lost (liquidated): 3,328.70, 112.4% of strategy equity at entry (includes adds)")
    store.event("liq-trend", "error", "incident", "Incident: liquidated; 1,671.30 left")

    page, errors = _open(browser, f"{site}/sleeves/liq-trend")
    assert "Only a reset after liquidation clears it" in page.inner_text("[data-live=banners]")
    assert page.locator("button[data-open=dlg-resume]").count() == 0
    page.click("button[data-open=dlg-ral]")
    dialog = page.locator("#dlg-ral")
    assert dialog.is_visible()
    assert "3,328.70" in dialog.inner_text() and "112.4% of its equity when the position was opened" in dialog.inner_text()
    dialog.locator("form").nth(1).locator("button[name=command]").click()  # reset before the note
    page.wait_for_load_state()
    assert "Not done:" in page.inner_text("main") and "no note yet" in page.inner_text("main")
    page.click("button[data-open=dlg-ral]")
    dialog.locator("input[name=author]").fill("PM")
    dialog.locator("textarea[name=why_stop_did_not_protect]").fill("The price gapped past the half-liquidation stop")
    dialog.locator("form").nth(0).locator("button").click()
    page.wait_for_load_state()
    page.click("button[data-open=dlg-ral]")
    assert "Written by PM" in dialog.inner_text()
    dialog.locator("form").nth(1).locator("button[name=command]").click()
    page.wait_for_load_state()
    assert any(c["command"] == "reset_after_liquidation" for c in store.pending_commands("liq-trend"))
    assert "Not done:" not in page.inner_text("main") and errors == []
    page.context.close()
