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
            "starting_balance": "5000", "risk_profile": "balanced", "warmup_bars": "0", "execution": "maker",
            "maker_wait_minutes": "15", "p_trend_filter__fast": "10", "p_trend_filter__slow": "30", "reason": "browser test"}
    r = httpx.post(f"{base}/sleeves/new", data=form, auth=("pm", PASSWORD), headers={"origin": base})
    assert r.status_code in (200, 303), r.text[:300]
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
