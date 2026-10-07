"""Phase 2 risk readouts (FE v2; Advisor 7 Oct 18:50, 18:55 and 19:00 UK). Open risk on the dashboard is what the 5%
open-risk limit counts, read through open_risk.book_open_risk, so the tile equals the gate: a perpetual with no stop,
a close-checked trail or a stop the price has gone through counts at notional x max(10%, 3 daily ATR), named as
estimated, never 0. A stop's distance towards liquidation reads amber past half way. A stop booked as a liquidation
reads in the Advisor's words, from the engine's stop_relabelled flag. conftest gives every pair a 2% daily ATR, so a
stopless perpetual counts at 10% of its notional."""

import re

import pytest
from fastapi.testclient import TestClient

from sleeve_fund import open_risk
from sleeve_fund.dashboard import trading
from sleeve_fund.store import Store

PERP = {"market": "perp", "allow_short": True}
AUTH = ("pm", "test-pw")


def _hold(store, name, qty, price=60_000.0, entry=60_000.0, strategy="ping_pong", params=None, stop_frac=None):
    """A strategy holding qty at entry, marked at price, with the entry's stop journaled as stop_frac (None: none)."""
    store.create_sleeve(name=name, strategy=strategy, instrument="BTC/USD", bar_spec="1-HOUR-LAST-INTERNAL",
                        starting_balance=10_000, params=PERP if params is None else params,
                        risk_profile="conservative")
    side = "BUY" if qty > 0 else "SELL"
    store.record_order(name, order_id=f"E-{name}", side=side, qty=abs(qty), intent="entry", reason="test",
                       signal={} if stop_frac is None else {"stop_frac": stop_frac})
    store.record_fill(name, side=side, qty=abs(qty), price=entry, fee=0.0, order_id=f"E-{name}", trade_id=f"t-{name}")
    store.record_equity(name, equity=10_000, cash=10_000 - qty * entry, qty=qty, price=price, benchmark=10_000)


@pytest.fixture
def store():
    return Store.in_memory()


def test_the_helper_gives_each_open_perp_its_basis_and_the_limits_figure(store):
    _hold(store, "stopped", 0.1, price=61_000, stop_frac=0.02)  # stop 58,800: 0.1 x 2,200
    _hold(store, "nostop", 0.1)  # 6,000 notional x 10%
    _hold(store, "through", 0.1, price=58_000, stop_frac=0.02)  # mark under its stop: stopless, never 0
    _hold(store, "short", -0.1, price=59_000, stop_frac=0.02)  # stop 61,200 above: 0.1 x 2,200
    _hold(store, "spot", 0.1, params={})  # spot is outside the limit
    _hold(store, "flat", 0.0)
    rows = {r["sleeve"]: r for r in open_risk.book_open_risk(store, lambda s: 0.02)}
    assert set(rows) == {"stopped", "nostop", "through", "short"}
    assert rows["stopped"] == {"sleeve": "stopped", "risk": pytest.approx(220.0), "basis": "stop", "atr_pct": None}
    assert rows["nostop"]["basis"] == "stopless" and rows["nostop"]["risk"] == pytest.approx(600.0)
    assert rows["through"]["basis"] == "gapped" and rows["through"]["risk"] == pytest.approx(580.0)
    assert rows["short"]["basis"] == "stop" and rows["short"]["risk"] == pytest.approx(220.0)


def test_the_helper_never_raises_and_asks_the_atr_only_where_it_needs_it(store):
    _hold(store, "stopped", 0.1, price=61_000, stop_frac=0.02)
    _hold(store, "nostop", 0.1)
    asked = []

    def atr(s):
        asked.append(s.name)
        return None

    rows = {r["sleeve"]: r for r in open_risk.book_open_risk(store, atr)}
    assert asked == ["nostop"]
    assert rows["nostop"]["risk"] is None and rows["nostop"]["atr_pct"] is None
    assert rows["stopped"]["risk"] == pytest.approx(220.0)


def test_the_helper_leaves_out_archived_strategies_and_filters_by_account(store):
    _hold(store, "a", 0.1)
    _hold(store, "b", 0.1)
    store.archived = lambda: {"b": None}  # archived while still holding (Store.archive refuses a holder)
    assert [r["sleeve"] for r in open_risk.book_open_risk(store, lambda s: 0.02)] == ["a"]
    account = store.account_of("a")
    assert [r["sleeve"] for r in open_risk.book_open_risk(store, lambda s: 0.02, account)] == ["a"]
    assert open_risk.book_open_risk(store, lambda s: 0.02, "no-such-account") == []


@pytest.mark.parametrize("stop_frac", [None, 0.02])
def test_a_trailing_model_takes_the_engines_basis(store, stop_frac):
    """Advisor 18:55 / 19:00 UK (PE2 pin G14): a close-checked trail with no resting stop (default rsi_pullback
    journals no stop_frac) counts stopless; a trail with a resting hard stop counts to that stop."""
    _hold(store, "rp", 0.1, price=61_000, strategy="rsi_pullback", params={**PERP, "atr_mult": 2.5},
          stop_frac=stop_frac)
    (row,) = open_risk.book_open_risk(store, lambda s: 0.02)
    if stop_frac is None:
        assert row["basis"] == "stopless" and row["risk"] == pytest.approx(610.0)
    else:
        assert row["basis"] == "stop" and row["risk"] == pytest.approx(220.0)


def test_the_helper_adds_up_to_what_the_gate_counts(store):
    """One formula: account_book's open risk for the other strategies equals the helper's rows added up."""
    _hold(store, "stopped", 0.1, price=61_000, stop_frac=0.02)
    _hold(store, "nostop", 0.1)
    _hold(store, "through", 0.1, price=58_000, stop_frac=0.02)
    _, gate, _ = open_risk.account_book(store, "an-entry-elsewhere", 0.0, lambda s: 0.02)
    assert gate == pytest.approx(sum(r["risk"] for r in open_risk.book_open_risk(store, lambda s: 0.02)))


@pytest.mark.parametrize("price, stop, liq, share", [
    (60_000, 57_000, 54_000, 0.5),  # long, half way
    (60_000, 58_800, 54_000, 0.2),
    (60_000, 54_000, 54_000, 1.0),  # at liquidation
    (60_000, 53_000, 54_000, 7 / 6),  # past it: the venue liquidates first
    (60_000, 63_000, 66_000, 0.5),  # short, half way
    (60_000, 61_000, 54_000, None),  # the price has gone through the stop
    (60_000, None, 54_000, None),
    (60_000, 57_000, None, None),
])
def test_stop_to_liquidation_is_the_share_of_the_way_from_the_mark(price, stop, liq, share):
    got = trading.stop_to_liquidation(price, stop, liq)
    assert got == (pytest.approx(share) if share is not None else None)


def test_a_stop_booked_as_a_liquidation_reads_in_the_advisors_words():
    order = {"intent": "liquidation", "signal": {"stop_relabelled": True, "stop_px": 58_800.0, "stop_fill_px": 53_900}}
    assert trading.stop_past_liquidation(order) == "Liquidated: stop at 58,800 would have filled past liquidation"
    assert trading.stop_past_liquidation({"intent": "liquidation", "signal": {}}) is None  # an ordinary one
    assert trading.stop_past_liquidation({"intent": "stop_loss", "signal": {"stop_relabelled": True}}) is None
    assert trading.stop_past_liquidation(None) is None
    assert trading.order_view({"status": "filled", "intent": "liquidation", "avg_px": 1.0, "filled_qty": 1.0,
                               "signal": order["signal"]})["relabelled"].startswith("Liquidated: stop at 58,800")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    monkeypatch.setenv("TEARSHEET_DIR", str(tmp_path))
    from sleeve_fund.dashboard import app as app_mod

    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    monkeypatch.setattr(app_mod, "LEDGER", tmp_path / "idea_ledger.jsonl")
    store = Store(f"sqlite:///{tmp_path}/t.db")
    return TestClient(app_mod.create_app(store)), store


def _tiles(c):
    out = {}
    for url in ("/", "/risk", "/trades"):
        html = c.get(url, auth=AUTH).text
        m = re.search(r'<div class="k">Open risk</div><div class="v([^"]*)">([^<]*)', html)
        out[url] = (m.group(1), m.group(2).strip()) if m else None
    return out


def test_the_tile_counts_a_stopless_perp_at_the_limits_estimate(client):
    """Gap A: the stopless perp is in the sum, named as estimated, and the figure equals the gate's."""
    c, store = client
    _hold(store, "stopped", 0.1, price=61_000, stop_frac=0.02)
    _hold(store, "nostop", 0.1)
    gate = sum(r["risk"] for r in open_risk.book_open_risk(store, lambda s: 0.02))
    assert gate == pytest.approx(820.0)
    for url, got in _tiles(c).items():
        assert got is not None, url
        assert "warn" in got[0] and got[1] == "820.00 · 1 estimated", (url, got)
    risk = c.get("/risk", auth=AUTH).text
    assert "nostop: no stop, counted at the larger of 10% and 3 daily ATRs" in risk
    assert "600.00 est." in risk


def test_a_perp_past_its_stop_never_reads_0(client):
    """Gap B: the price has gone through the stop with the position still open: stopless, never 0."""
    c, store = client
    _hold(store, "through", 0.1, price=58_000, stop_frac=0.02)
    for url, got in _tiles(c).items():
        assert got[1] == "580.00 · 1 estimated" and "warn" in got[0], (url, got)
    risk = c.get("/risk", auth=AUTH).text
    assert "580.00 est. · past stop" in risk
    assert "through: the price has gone through its stop and the position is still open" in risk


def test_a_perp_whose_daily_atr_isnt_known_is_left_out_and_named(client, monkeypatch):
    c, store = client
    monkeypatch.setattr(open_risk, "history_atr_pct", lambda venue, pair, now, history=None: None)
    _hold(store, "nostop", 0.1)
    for url, got in _tiles(c).items():
        assert got[1] == "0.00+ · 1 not counted" and "warn" in got[0], (url, got)
    assert "nostop: not counted, its daily ATR isn&#39;t known yet" in c.get("/risk", auth=AUTH).text


def test_a_stop_past_half_way_to_liquidation_reads_amber(client, monkeypatch):
    c, store = client
    monkeypatch.setattr(trading, "position_margin", lambda x: (1_000.0, 54_000.0))
    _hold(store, "far", 0.1, stop_frac=0.08)  # stop 55,200: 80% of the way from 60,000 to 54,000
    page = c.get("/sleeves/far", auth=AUTH).text
    assert re.search(r'<span class="warn" title="From the current price: past the half-way rule[^"]*">stop 80% of the way to liquidation', page)
    assert "stop 80% of the way to liquidation" in c.get("/risk", auth=AUTH).text


def test_the_hover_says_what_the_figure_is_made_of(client):
    """CR 7 Oct: with spot held too, the figure is the 5% limit's plus spot risk to stop, and the hover says so."""
    c, store = client
    _hold(store, "perp", 0.1, price=61_000, stop_frac=0.02)
    assert "The 5% limit&#39;s figure" in c.get("/risk", auth=AUTH).text
    _hold(store, "spot", 0.1, price=61_000, params={"stop_loss": 0.02}, stop_frac=0.02)
    for url in ("/", "/risk", "/trades"):
        assert "The 5% limit&#39;s figure plus spot risk to stop" in c.get(url, auth=AUTH).text, url
