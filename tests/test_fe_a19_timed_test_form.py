"""A-19 (PM's go, 15 Oct; HoE Done-when): the PM starts the candle-step timed test himself from the New Strategy form.
The form offers it in a "Test" group of its own, apart from the Strategies, and only family "test" unlisted models go
there; choosing it fills in the PM's settings (perp, shorts on, 1-minute candles, balanced, stop and target 0.1%, size
share 60% shown as a %, run until stopped, the demo copy unticked). Bad size share or run-for-hours input is refused in
words with the form sent back filled in. Creating it never starts it, and it is a normal Strategy in the fund. The
created row is read back from the journal, on Postgres too (TEST_DATABASE_URL)."""

import re
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient
from test_halt_ref import (
    store,  # noqa: F401 - the journal, on Postgres when TEST_DATABASE_URL is set
)

AUTH = ("pm", "pw")
SAME = {"origin": "http://testserver"}
FORM = {"name": "timed-test", "strategy": "candle_step", "instrument": "BTC/USD", "bar_spec": "1-MINUTE-LAST-INTERNAL",
        "starting_balance": "10000", "risk_profile": "balanced", "warmup_bars": "0", "market": "perp",
        "allow_short": "1", "stop_loss_pct": "0.1", "take_profit_pct": "0.1", "p_candle_step__size_share": "60",
        "reason": "the PM's timed test"}


@pytest.fixture
def c(store, tmp_path, monkeypatch):  # noqa: F811
    from sleeve_fund.dashboard import app as app_mod
    from sleeve_fund.dashboard import book

    monkeypatch.setenv("DASHBOARD_PASSWORD", "pw")
    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    monkeypatch.setattr(app_mod, "LEDGER", tmp_path / "idea_ledger.jsonl")
    monkeypatch.setattr(book, "is_test_account", lambda store, sleeve: False)
    return TestClient(app_mod.create_app(store))


def _post(c, **over):
    return c.post("/sleeves/new", data={**FORM, **over}, auth=AUTH, headers=SAME, follow_redirects=False)


def _groups(html: str) -> dict:
    sel = re.search(r'<select name="strategy".*?</select>', html, re.DOTALL).group(0)
    return {g: re.findall(r'<option value="([a-z_]+)"', body)
            for g, body in re.findall(r'<optgroup label="([^"]+)">(.*?)</optgroup>', sel, re.DOTALL)}


def test_the_form_offers_the_timed_test_in_its_own_test_group_and_no_other_unlisted_model(c):
    groups = _groups(c.get("/sleeves/new", auth=AUTH).text)
    assert groups["Test"] == ["candle_step"]
    assert "candle_step" not in groups["Strategies"] and "trend_filter" in groups["Strategies"]
    assert "rules" not in groups["Strategies"] + groups["Test"]  # unlisted, not a test model: stays off the form


def test_choosing_it_opens_the_form_with_the_pms_settings(c):
    html = c.get("/sleeves/new?strategy=candle_step", auth=AUTH).text
    assert re.search(r'<option value="perp"\s+selected', html)
    assert re.search(r'id="allow_short" name="allow_short" value="1" checked', html)
    assert re.search(r'name="bar_spec" value="1-MINUTE-LAST-INTERNAL" checked', html)
    assert re.search(r'<option value="balanced"\s+selected', html)
    assert 'name="stop_loss_pct" type="number" min="0.1" max="50" step="0.01" placeholder="e.g. 5" value="0.1"' in html
    assert 'name="take_profit_pct" type="number" min="0.1" max="1000" step="0.01" placeholder="e.g. 15" value="0.1"' in html
    assert re.search(r'name="p_candle_step__size_share" value="60"', html)
    assert re.search(r'name="p_candle_step__run_for_hours" value="" placeholder="Until stopped"', html)
    assert re.search(r'id="demo_mirror" name="demo_mirror" value="1" >', html)  # shown, not ticked
    assert re.search(r"data-prefill=", html)


def test_created_it_is_stopped_with_the_settings_as_stored(c, store):  # noqa: F811
    r = _post(c)
    assert r.status_code == 303 and r.headers["location"] == "/sleeves/timed-test", r.headers.get("location")
    s = store.sleeve("timed-test")
    assert s.desired_state == "stopped"
    assert (s.strategy, s.bar_spec, s.risk_profile) == ("candle_step", "1-MINUTE-LAST-INTERNAL", "balanced")
    p = s.params
    assert p["size_share"] == pytest.approx(0.6) and p["market"] == "perp" and p["allow_short"] is True
    assert p["stop_loss"] == pytest.approx(0.001) and p["take_profit"] == pytest.approx(0.001)
    assert "run_for_hours" not in p and "demo_mirror" not in p  # until stopped; no demo copy


def test_a_run_window_is_kept_in_hours(c, store):  # noqa: F811
    assert _post(c, p_candle_step__run_for_hours="6").status_code == 303
    assert float(store.sleeve("timed-test").params["run_for_hours"]) == 6


@pytest.mark.parametrize(("field", "value", "words"), [
    ("p_candle_step__size_share", "0", "size share: a % of fund equity above 0 and at most 100"),
    ("p_candle_step__size_share", "101", "size share: a % of fund equity above 0 and at most 100"),
    ("p_candle_step__size_share", "-5", "size share: a % of fund equity above 0 and at most 100"),
    ("p_candle_step__run_for_hours", "0", "run for hours: more than 0, or leave it empty to run until stopped"),
    ("p_candle_step__run_for_hours", "-2", "run for hours: more than 0, or leave it empty to run until stopped"),
    # F-A19-2: float() takes these, and none is a size or a number of hours
    ("p_candle_step__size_share", "nan", "size share: a % of fund equity above 0 and at most 100"),
    ("p_candle_step__size_share", "1e-9", "size share: a % of fund equity above 0 and at most 100"),  # stores as 0
    ("p_candle_step__run_for_hours", "inf", "run for hours: more than 0, or leave it empty to run until stopped"),
    ("p_candle_step__run_for_hours", "Infinity", "run for hours: more than 0, or leave it empty to run until stopped"),
    ("p_candle_step__run_for_hours", "nan", "run for hours: more than 0, or leave it empty to run until stopped"),
])
def test_bad_input_is_refused_in_words_and_the_form_comes_back_filled_in(c, store, field, value, words):  # noqa: F811
    r = _post(c, **{field: value})
    assert r.status_code == 303 and r.headers["location"].startswith("/sleeves/new?")
    q = parse_qs(urlparse(r.headers["location"]).query)
    assert q["error"] == [words] and q[field] == [value] and q["name"] == ["timed-test"]
    assert not any(s.name == "timed-test" for s in store.sleeves())
    page = c.get(r.headers["location"], auth=AUTH).text
    assert f"Not saved: {words}" in page and f'name="{field}" value="{value}"' in page


def test_clone_with_changes_brings_the_size_share_back_as_a_percent(c, store):  # noqa: F811
    assert _post(c).status_code == 303
    html = c.get("/sleeves/timed-test", auth=AUTH).text
    link = re.search(r'href="/sleeves/new\?([^"]+)"', html).group(1).replace("&amp;", "&")
    q = parse_qs(link)
    assert q["strategy"] == ["candle_step"] and q["p_candle_step__size_share"] == ["60"]
    again = c.get(f"/sleeves/new?{link}", auth=AUTH).text
    assert re.search(r'name="p_candle_step__size_share" value="60"', again)


def test_clone_with_changes_does_not_fill_in_what_the_source_left_unset(c, store):  # noqa: F811
    """F-A19-1: the PM's settings fill a fresh pick only; a clone of a timed test run with shorts off and no target
    must come back with shorts off and no target, not with the defaults laid over it."""
    form = {k: v for k, v in FORM.items() if k != "allow_short"}
    r = c.post("/sleeves/new", data={**form, "take_profit_pct": ""}, auth=AUTH, headers=SAME, follow_redirects=False)
    assert r.status_code == 303
    p = store.sleeve("timed-test").params
    assert not p.get("allow_short") and not p.get("take_profit")
    html = c.get("/sleeves/timed-test", auth=AUTH).text
    link = re.search(r'href="/sleeves/new\?([^"]+)"', html).group(1).replace("&amp;", "&")
    again = c.get(f"/sleeves/new?{link}", auth=AUTH).text
    assert re.search(r'id="allow_short" name="allow_short" value="1" >', again)
    assert re.search(r'name="take_profit_pct" type="number"[^>]*value=""', again)
    assert re.search(r'name="p_candle_step__size_share" value="60"', again)


def test_the_button_says_a_test_model_is_created_stopped(c):
    test = c.get("/sleeves/new?strategy=candle_step", auth=AUTH).text
    other = c.get("/sleeves/new?strategy=trend_filter", auth=AUTH).text
    assert re.search(r'data-test-label="Create \(stopped\)">Create \(stopped\)</button>', test)
    assert re.search(r'data-test-label="Create \(stopped\)">Start strategy in paper mode</button>', other)



# HoE MUST (15 Oct, 18:53 UK): on the server the fund holds the cash (the re-base deposit applied at the 04:59 reset),
# so a Strategy is made with 0 (CUT-A (a)) and its own drawdown halt doesn't act (CUT-A (d)). The form must send 0 and
# say so, or every create there is refused.
HELD_WORDS = "The fund holds the capital. This Strategy sizes from fund equity."


def _hold(store):  # noqa: F811
    at = datetime.now(UTC) - timedelta(minutes=5)
    store.settle_flow(store.fund_flow("deposit", 100_000, actor="pm", reason="the PM's reset", rebase=True, now=at), at)


def _capital_value(html: str) -> str | None:
    m = re.search(r'<input[^>]*name="starting_balance"[^>]*>', html)
    return re.search(r'value="([^"]*)"', m.group(0)).group(1) if m else None


def test_fund_held_the_timed_test_from_the_form_is_made_stopped_with_0(c, store):  # noqa: F811
    _hold(store)
    html = c.get("/sleeves/new?strategy=candle_step", auth=AUTH).text
    assert _capital_value(html) == "0"
    r = _post(c, starting_balance=_capital_value(html))
    assert r.status_code == 303 and r.headers["location"] == "/sleeves/timed-test", r.headers.get("location")
    s = store.sleeve("timed-test")
    assert s.desired_state == "stopped" and s.starting_balance == 0


def test_fund_held_an_ordinary_strategy_from_the_form_is_made_too(c, store):  # noqa: F811
    _hold(store)
    html = c.get("/sleeves/new?strategy=trend_filter", auth=AUTH).text
    form = {"name": "btc-trend", "strategy": "trend_filter", "instrument": "BTC/USD", "bar_spec": "1-HOUR-LAST-INTERNAL",
            "starting_balance": _capital_value(html), "risk_profile": "balanced", "warmup_bars": "0",
            "p_trend_filter__fast": "10", "p_trend_filter__slow": "30", "reason": "a Strategy in the fund"}
    r = c.post("/sleeves/new", data=form, auth=AUTH, headers=SAME, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/sleeves/btc-trend", r.headers.get("location")
    assert store.sleeve("btc-trend").starting_balance == 0


def test_fund_held_the_form_asks_no_capital_and_names_no_own_drawdown_halt(c, store):  # noqa: F811
    _hold(store)
    html = c.get("/sleeves/new", auth=AUTH).text
    assert re.search(r'<input type="hidden" id="starting_balance" name="starting_balance" value="0">', html)
    assert 'type="number" min="100"' not in html and HELD_WORDS in html
    labels = re.findall(r'<option value="(?:conservative|balanced|aggressive)"[^>]*>([^<]+)</option>', html)
    assert labels and not any("halt at" in w for w in labels)
    assert "Balanced: leverage up to 2x" in labels


def test_not_held_the_form_is_as_it_was(c):
    html = c.get("/sleeves/new", auth=AUTH).text
    assert re.search(r'id="starting_balance" name="starting_balance" type="number" min="100" step="100" value="10000"'
                     r" required", html)
    assert HELD_WORDS not in html
    labels = re.findall(r'<option value="(?:conservative|balanced|aggressive)"[^>]*>([^<]+)</option>', html)
    assert labels and all("halt at" in w for w in labels)
