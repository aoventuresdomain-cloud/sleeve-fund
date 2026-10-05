"""Research → Development (combined build spec, part B): plan statuses, recommended settings, the venue chosen
inside each study, the Results and History tabs, and the result page's verdict banner and cost ladder."""

import json
import re

import pytest
from fastapi.testclient import TestClient

from sleeve_fund.dashboard import development as dev
from sleeve_fund.store import Store

AUTH = ("pm", "test-pw")
SAME = {"origin": "http://testserver"}


def _sheet(strategy="rsi_cross", g1="FAIL", venue="BINANCE", pair="BTC/USDT", minutes=15, ladder=(6.1, 1.2, -4.2, -9.5, -61.0),
           taker="0.05%", means=None):
    rungs = "\n".join(f"| {fee} | {ret:+.1f}% | 0.10 | 2480 | 100 |"
                      for fee, ret in zip(("0.00%", "0.02%", "0.05%", "0.10%", "0.80%"), ladder))
    top = next((i for i, r in enumerate(ladder) if r <= 0), None)
    if ladder[0] <= 0:
        be = "loses money even at 0.00% fees"
    elif top is None:
        be = "still makes money at 0.80% per side, the top of the ladder"
    else:
        be = "stops making money at about 0.031% per side (between 0.02% and 0.05%)"
    sharpe_row = {"PASS": "PASS", "FAIL": "FAIL", "NOT JUDGED": "N/A"}[g1]
    judged = "NOT JUDGED" if g1 == "NOT JUDGED" else "PASS"
    return f"""# Tear sheet: {strategy}

Tested on `{pair}` at {minutes}-minute bars on `{venue}`

Settings: balanced risk profile · exits: the signal only · walk-forward 730 days training, 365 days testing

Dataset `{venue.lower()}-{pair.replace('/', '').lower()}-store-{minutes}m` · research period 01 Oct 2021 to 05 Oct 2025 · holdout: last 365 days (untouched) · fees: Venue: 0.02% maker, {taker} taker, published schedule

## G1 checks

| Check | Result | Evidence |
| --- | --- | --- |
| Runs complete enough to judge | {judged} | {'Not judged: the risk guard halted it' if g1 == 'NOT JUDGED' else 'no strategy errors'} |
| G1 test: out-of-sample Sharpe clearly beats benchmark after fees | {sharpe_row} | Sharpe 0.41 vs 0.88 |
| Holds up when parameters move | {sharpe_row} | 0% of 4 grid points |
| Enough out-of-sample trades to judge | PASS | 1284 closed in the 1 walk-forward test windows (bar: 10) |

## Results after fees

| Period | Strategy CAGR | Benchmark CAGR | Strategy Sharpe | Benchmark Sharpe | Strategy max DD | Benchmark max DD |
| --- | --- | --- | --- | --- | --- | --- |
| Walk-forward out-of-sample (1 folds, 365 days) | -4.2% | +38.0% | 0.41 | 0.88 | -14.0% | -31.0% |

## Cost ladder (full research period, default params)

**Break-even fee:** {be}.

| Fee per side | Total return | Sharpe | Round trips | Fees paid |
| --- | --- | --- | --- | --- |
{rungs}

The same strategy and settings at each fee. Every rung also pays half the bid-ask spread as above plus 0.02% slippage on orders that take liquidity.
""" + (f"\n## What it means\n\n{means}\n" if means else "")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    monkeypatch.setenv("TEARSHEET_DIR", str(tmp_path))
    from sleeve_fund import history
    from sleeve_fund.dashboard import app as app_mod

    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    monkeypatch.setattr(app_mod, "LEDGER", tmp_path / "idea_ledger.jsonl")
    monkeypatch.setattr(history, "DEFAULT_ROOT", tmp_path / "hist")
    store = Store(f"sqlite:///{tmp_path}/t.db")
    return TestClient(app_mod.create_app(store)), store, tmp_path


def _coverage(root, venue, pair, first, last):
    d = root / "hist" / venue / pair.replace("/", "-")
    d.mkdir(parents=True)
    (d / "coverage.json").write_text(json.dumps({"first": first, "last": last, "cursor": "x"}))


def test_plan_statuses_follow_the_newest_real_verdict():
    """Passed G1, Killed (a real-data FAIL), Not judged; untested, Ready when research gave it settings."""
    assert dev.status({"g1": "PASS"}, False)[:2] == ("passed", "Passed G1")
    assert dev.status({"g1": "FAIL"}, True)[:2] == ("killed", "Killed")
    assert dev.status({"g1": "NOT JUDGED"}, True)[:2] == ("unjudged", "Not judged")
    assert dev.status({"g1": None}, True)[:2] == ("ready", "Ready")
    assert dev.status({"g1": None}, False)[:2] == ("untested", "Not tested")


def test_the_development_tab_lists_every_model_with_its_status(client):
    c, _, tmp = client
    (tmp / "rsi_bands_binance-btcusdt-store-15m_20261005-100000.md").write_text(_sheet("rsi_bands", "FAIL"))
    (tmp / "trend_filter_kraken-btcusd-store-1440m_20261005-110000.md").write_text(
        _sheet("trend_filter", "PASS", venue="KRAKEN", pair="BTC/USD", minutes=1440, ladder=(9, 8, 7, 6, 2)))
    page = c.get("/research", auth=AUTH).text
    by_model = {m: st for st, m in re.findall(r'<li data-status="(\w+)"><a class="plan-card" href="[^"]*" data-plan="(\w+)"', page)}
    assert by_model == {"rsi_cross": "ready", "trend_filter": "passed", "rsi_bands": "killed",
                                  "buy_and_hold": "untested", "ping_pong": "untested", "rsi_pullback": "untested",
                                  "dip_buy": "untested", "donchian": "untested"}
    # Ready first (and picked), then what passed; the meta line says where the last verdict came from.
    assert page.index('data-plan="rsi_cross"') < page.index('data-plan="trend_filter"') < page.index('data-plan="rsi_bands"')
    assert "Kill on BTC/USDT, Binance USD-M perpetuals · break-even 0.03%" in page
    assert "Pass on BTC/USD, Kraken spot" in page and "not tested yet" in page
    for word, n in (("Ready", 1), ("Passed G1", 1), ("Killed", 1)):
        assert re.search(rf'>{word} <span class="n">{n}</span></button>', page)
    assert '<button type="button" data-filter="all" aria-pressed="true">All <span class="n">6</span>' in page
    # No page-level venue switch that reloads; the menu and the tab both say Development.
    assert "data-reload" not in page and 'data-tab="development">Development' in page
    assert "<span>Development</span>" in page


def test_rsi_cross_comes_with_research_recommended_settings(client):
    """Run 1 of the strategy sprint: 15-minute candles, learn 730 days, test 365, a sealed 365-day holdout, a stop
    2 ATR (14 bars) below entry, balanced, on Binance BTC/USDT; other models fall back to the standing defaults."""
    c, _, _ = client
    p = dev.plan("rsi_cross", None)
    assert p["recommended"] and p["values"] == {"venue": "binance", "instrument": "BTC/USDT", "minutes": 15,
                                                "train_days": 730, "test_days": 365, "holdout_days": 365,
                                                "use_holdout": False, "risk_profile": "balanced", "stop_atr": 2,
                                                "atr_bars": 14}
    assert dev.how_sentence(p["values"]) == "15-minute candles. Learn on 2 years, test on the next year. Latest year sealed."
    assert dev.exits_sentence(p["values"]) == ("Stop 2 average true ranges below entry (14 bars). No target. "
                                               "Balanced profile.")
    page = c.get("/research?strategy=rsi_cross", auth=AUTH).text
    assert "15-minute candles. Learn on 2 years, test on the next year. Latest year sealed." in page
    assert "Recommended by research" in page and "<b>Answers:</b> Does buying after a sharp 15-minute dip" in page
    assert '<option value="15" selected>' in page and 'value="730"' in page and 'name="stop_atr"' in page
    assert '<input type="radio" name="venue" value="binance" checked>' in page and 'value="BTC/USDT"' in page
    assert "Runs 4 variants and the cost ladder at 0.05% (Binance USD-M perpetuals taker fee). Long only." in page
    other = c.get("/research?strategy=trend_filter", auth=AUTH).text
    assert "Daily candles. Learn on 1 year, test on the next 180 days. Latest year sealed." in other
    assert dev.GENERIC_WHY.replace("'", "&#39;") in other and '<input type="radio" name="venue" value="kraken" checked>' in other


def test_a_form_sent_back_keeps_what_was_sent_and_only_for_its_model():
    p = dev.plan("rsi_cross", None)
    assert dev.form_values(p, {"strategy": "trend_filter", "minutes": "60"}) == p["values"]
    back = dev.form_values(p, {"strategy": "rsi_cross", "minutes": "60", "train_days": "400", "test_days": "90",
                               "holdout_days": "0", "venue": "kraken", "instrument": "ETH/USD"})
    assert back["minutes"] == "60" and back["venue"] == "kraken" and "stop_atr" not in back and not back["use_holdout"]
    picked = dev.form_values(p, {"strategy": "rsi_cross", "venue": "kraken"})
    assert picked["venue"] == "kraken" and picked["minutes"] == 15


def test_every_venue_s_history_is_on_the_page_for_the_study_to_switch_in_place(client):
    c, store, tmp = client
    _coverage(tmp, "BINANCE", "BTC/USDT", "2021-10-01T00:00:00+00:00", "2099-01-01T00:00:00+00:00")
    _coverage(tmp, "KRAKEN", "ETH/USD", "2016-01-01T00:00:00+00:00", "2020-01-01T00:00:00+00:00")
    page = c.get("/research", auth=AUTH).text
    data = json.loads(re.search(r'<script type="application/json" id="study-data">(.*?)</script>', page, re.S).group(1))
    assert data["venues"]["binance"]["held"]["BTC/USDT"] == {"text": "77.3 years stored · current", "tone": "running"}
    assert data["venues"]["kraken"]["held"]["ETH/USD"]["text"] == "Catching up, from 01 Jan 2016"
    assert data["plans"]["rsi_cross"]["values"]["minutes"] == 15 and data["plans"]["trend_filter"]["variants"] == 11
    assert '<span class="chip running" id="hist-chip">77.3 years stored · current</span>' in page
    assert '<datalist id="pairs-binance"><option value="BTC/USDT">' in page and '<datalist id="pairs-kraken">' in page
    # History tab: both venues, with the venue named; the collect form picks the venue.
    assert re.search(r'ETH/USD</td>\s*<td[^>]*>Kraken spot</td>', page)
    assert re.search(r'BTC/USDT</td>\s*<td[^>]*>Binance USD-M perpetuals</td>', page)
    assert '<select name="venue" aria-label="Venue" id="collect-venue">' in page


def test_get_research_history_never_errors(client):
    """Sprint known issue: GET /research/history errored. It returns to the page's History tab, and an
    unreadable coverage file is left out rather than taking the page down."""
    c, _, tmp = client
    _coverage(tmp, "KRAKEN", "ETH/USD", "2016-01-01T00:00:00+00:00", "2020-01-01T00:00:00+00:00")
    bad = tmp / "hist" / "KRAKEN" / "BTC-USD"
    bad.mkdir(parents=True)
    (bad / "coverage.json").write_text("{not json")
    _coverage(tmp, "BINANCE", "SOL/USDT", "2023-01-01T00:00:00", "2099-01-01T00:00:00")  # no zone: read as UTC
    r = c.get("/research/history?venue=binance", auth=AUTH)
    assert r.status_code == 200 and str(r.url).endswith("/research?venue=binance#history")
    assert "ETH/USD</td>" in r.text and "SOL/USDT</td>" in r.text and "BTC/USD</td>" not in r.text
    assert c.get("/research/history", auth=AUTH).status_code == 200


def test_a_collect_from_a_study_answers_in_that_study(client, monkeypatch):
    from sleeve_fund.venues import venue

    c, store, _ = client
    monkeypatch.setattr(venue("binance"), "check_listed", lambda pair: None)
    r = c.post("/research/history", data={"instrument": "DOGE/USDT", "venue": "binance", "strategy": "trend_filter",
                                          "from": "study"}, auth=AUTH, headers=SAME)
    assert "Asked the collector for DOGE/USDT" in r.text
    assert 'data-plan="trend_filter" aria-current="true"' in r.text and "data-open" in r.text
    assert 'data-default="development"' in r.text
    plain = c.post("/research/history", data={"instrument": "PEPE/USDT", "venue": "binance"}, auth=AUTH, headers=SAME)
    assert 'data-default="history"' in plain.text and [h["instrument"] for h in store.history_requests("BINANCE")] == [
        "DOGE/USDT", "PEPE/USDT"]


def test_results_tab_gives_each_study_its_verdict_and_break_even(client):
    c, _, tmp = client
    (tmp / "rsi_cross_binance-btcusdt-store-15m_20261005-140000.md").write_text(_sheet())
    (tmp / "rsi_cross_binance-ethusdt-store-15m_20261005-150000.md").write_text(
        _sheet(g1="NOT JUDGED", pair="ETH/USDT", ladder=(-1, -2, -3, -4, -9)))
    page = c.get("/research", auth=AUTH).text
    results = page[page.index('data-panel="results"'):page.index('data-panel="history"')]
    rows = re.findall(r'<a href="/research/(\w[\w-]*)">.*?<span class="chip (\w+)"[^>]*>([\w ]+)</span>.*?'
                      r'<td class="num" data-m="hide">([^<]+)</td>', results, re.S)
    assert [(r[0][-15:], r[2], r[3]) for r in rows] == [("20261005-150000", "Not judged", "None"),
                                                         ("20261005-140000", "Kill", "0.03%")]  # newest first
    assert "BTC/USDT · 15-minute · Binance USD-M perpetuals" in results
    assert 'href="/research/rsi_cross_binance-btcusdt-store-15m_20261005-140000/download"' in results


def test_the_result_page_leads_with_the_verdict_and_the_break_even_fee(client):
    c, _, tmp = client
    (tmp / "kill.md").write_text(_sheet(means="Fees eat the small moves it finds."))
    page = c.get("/research/kill", auth=AUTH).text
    assert '<span class="big">Kill</span>' in page
    assert ("Break-even fee is 0.03% per side. Binance USD-M perpetuals charges 0.05%, so fees eat the edge. "
            "G1 failed on: G1 test: out-of-sample Sharpe clearly beats benchmark after fees; holds up when parameters "
            "move.") in page
    # Four figures: break-even vs the kill line, return out of sample, Sharpe vs buy-and-hold, trades a day.
    assert "kill line 0.05%, the venue's fee" in page and ">-4.2%</div>" in page
    assert "0.41 <span class=\"faint\">/</span> 0.88" in page and "1,284" in page and "about 3.5 a day" in page
    # The cost ladder: a bar per rung, the venue's own fee starred, the slippage said; the full sheet below.
    at = page.index('id="ladder-h"')
    ladder = page[at:page.index("</svg>", at)]
    assert ladder.count("<rect ") == 5 and "0.05% ★" in page and "0.02% slippage" in page
    assert "What it means" in page and "Fees eat the small moves it finds." in page
    assert 'href="/research/kill/download"' in page and "Tear sheet: rsi_cross" in page

    (tmp / "passes.md").write_text(_sheet(g1="PASS", ladder=(9, 8, 7, 6, 2)))
    ok = c.get("/research/passes", auth=AUTH).text
    assert '<span class="big">Pass</span>' in ok and "still makes money at 0.80% per side" in ok
    assert "What it means" not in ok  # not invented when the sheet has none
    (tmp / "kraken.md").write_text(_sheet(g1="PASS", venue="KRAKEN", pair="BTC/USD", taker="0.40%", ladder=(9, 8, 7, 6, -2)))
    kr = c.get("/research/kraken", auth=AUTH).text
    assert "Kraken spot charges 0.40%, so fees eat the edge" in kr and 'class="marker"' in kr  # 0.40% sits between rungs
    (tmp / "unjudged.md").write_text(_sheet(g1="NOT JUDGED", ladder=(-1, -2, -3, -4, -9)))
    nj = c.get("/research/unjudged", auth=AUTH).text
    assert '<span class="big">Not judged</span>' in nj and "It loses money even with no fees" in nj
    assert "Not judged: the risk guard halted it." in nj and "Not judged: Not judged" not in nj
    # An old sheet with none of it still opens, and says so.
    (tmp / "trend_filter_x.md").write_text("# Tear sheet\n\n| a | b |\n| - | - |\n| 1 | 2 |\n")
    old = c.get("/research/trend_filter_x", auth=AUTH)
    assert old.status_code == 200 and "no cost ladder" in old.text and '<span class="big">Not judged</span>' in old.text


def test_a_real_tear_sheet_reads_back(tmp_path, instrument):
    """The result page reads the tear sheet as tearsheet.render writes it: no new maths, nothing stored twice."""
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.study import COST_LADDER, run_study
    from sleeve_fund.research.tearsheet import render
    from sleeve_fund.strategies.trend_filter import SPEC

    r = run_study(SPEC, synthetic_ohlcv(days=1500, seed=3), instrument, dataset="kraken-btcusd-store",
                  ledger=IdeaLedger(tmp_path / "l.jsonl"), holdout_days=100, train_days=730, test_days=365)
    path = tmp_path / "trend_filter_kraken-btcusd-store_20261005-120000.md"
    path.write_text(render(r, IdeaLedger(tmp_path / "l.jsonl")))
    s = dev.read_sheet(path)
    assert [x["fee"] for x in s["rungs"]] == list(COST_LADDER) and s["fee"] == float(instrument.taker_fee)
    assert s["venue_label"] == "Kraken spot" and s["minutes"] == 1440 and s["oos_trades"] == r.oos_trades
    assert s["breakeven_kind"] in ("at", "none", "above") and s["oos_days"] > 0 and s["slippage"] == 0.0002
    assert s["when"].strftime("%Y%m%d-%H%M%S") == "20261005-120000"
    assert dev.banner(s) and dev.ladder_chart(s["rungs"], s["fee"])["bars"]
