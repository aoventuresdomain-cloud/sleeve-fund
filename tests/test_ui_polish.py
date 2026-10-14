"""UI polish (during testing, fe-plans/ui-polish-plan.md): pins for each item as it lands."""

import re
from decimal import Decimal as D
from pathlib import Path

import pandas as pd
import pytest

from sleeve_fund.portfolio.limits import Holding

DASH = Path(__file__).resolve().parents[1] / "sleeve_fund" / "dashboard"


def test_up6_no_collect_leftovers_in_the_templates_or_the_css():
    """UP-6: the Collect button went in UI v2 (item 10); a study asks the collector itself. Its form styles and any
    "Collect" control are gone. The history request route stays: a study's own request shares its checks."""
    css = (DASH / "static" / "app.css").read_text()
    assert ".collect-" not in css
    for page in (DASH / "templates").glob("*.html"):
        text = page.read_text()
        assert "collect-" not in text, page.name
        assert not re.search(r">\s*Collect\b", text), page.name


# --- UP-1: the Risk & health header's total size, as the entry gate reads it -----------------------------------------

BOOK = {"exposure_pct": 0.24, "net_pct": -0.16}  # the strategies' own marks, used only before the fund is marked


def _gated_store(tmp_path, monkeypatch, notionals, equity=D(10_000)):
    """A journal whose fund the supervisor has marked at `equity`, holding `notionals` (underlying -> signed notional)
    as the gate's fund_holdings reads them."""
    from sleeve_fund.gate_ledger import DbLedger
    from sleeve_fund.paper import portfolio_gate
    from sleeve_fund.portfolio.gate import PortfolioState
    from sleeve_fund.store import Store, utcnow

    store = Store(f"sqlite:///{tmp_path}/g.db", portfolio_gate=True)
    now = utcnow()
    DbLedger(store, lambda: None).set_state(PortfolioState(mark_ts=now, equity=equity, hwm=equity, day=now.date(),
                                                           day_start_equity=equity))
    held = tuple(Holding(u, D(n), abs(D(n)), abs(D(n)) * D("0.02")) for u, n in notionals.items())
    monkeypatch.setattr(portfolio_gate, "fund_holdings", lambda store, now=None, atr_pct=None: held)
    return store, now


def test_up1_total_size_and_net_are_the_figures_the_entry_gate_reads(tmp_path, monkeypatch):
    from sleeve_fund.dashboard.riskops import gate_size
    from sleeve_fund.paper.portfolio_gate import paper_gate

    store, now = _gated_store(tmp_path, monkeypatch, {"BTC": 9_000, "ETH": -3_000})
    store.create_sleeve(name="s1", strategy="trend_filter", instrument="BTC/USD", bar_spec="1-HOUR-LAST-INTERNAL",
                        starting_balance=10_000, params={})
    size = gate_size(store, BOOK)
    gated = paper_gate(store)("s1", {"side": 1, "qty": D("0.001"), "close": D(60_000), "price": D(60_000),
                                     "stop_frac": 0.02, "what": "entry", "step": D("0.001"), "min_qty": D("0.001"),
                                     "instrument": "BTC", "leverage": None, "atr_pct": None}, pd.Timestamp(now).value)
    gate = gated.decision.figures["gross"]
    assert size["source"] == "gate" and size["total"] == pytest.approx(gate["before"]) == pytest.approx(1.2)
    assert size["limit"] == pytest.approx(gate["cap"]) == 1.5
    assert size["net"] == pytest.approx(0.6)  # long 9,000, short 3,000, on a 10,000 book


@pytest.mark.parametrize(("long", "tone"), [(11_900, ""), (12_000, "warn"), (14_999, "warn"), (15_000, "loss")])
def test_up1_total_size_is_amber_from_80_percent_of_the_limit_and_red_at_it(tmp_path, monkeypatch, long, tone):
    from sleeve_fund.dashboard.riskops import gate_size

    store, _ = _gated_store(tmp_path, monkeypatch, {"BTC": long})
    assert gate_size(store, BOOK)["tone"] == tone


def test_up1_before_the_fund_is_marked_the_header_says_it_reads_the_strategies_marks(tmp_path):
    from sleeve_fund.dashboard.riskops import gate_size
    from sleeve_fund.store import Store

    size = gate_size(Store(f"sqlite:///{tmp_path}/n.db", portfolio_gate=True), BOOK)
    assert (size["source"], size["total"], size["net"], size["tone"]) == ("marks", 0.24, -0.16, "")


def test_up1_a_gate_book_that_cant_be_read_says_so_not_that_the_fund_has_no_mark(tmp_path, monkeypatch):
    from sleeve_fund.dashboard.riskops import gate_size
    from sleeve_fund.paper import portfolio_gate

    store, _ = _gated_store(tmp_path, monkeypatch, {"BTC": 9_000})

    def down(*_a, **_k):
        raise OSError("database down")

    monkeypatch.setattr(portfolio_gate, "PaperGate", down)
    size = gate_size(store, BOOK)
    assert (size["source"], size["total"], size["net"]) == ("unreadable", 0.24, -0.16)


def test_up1_the_risk_header_shows_net_and_total_size_against_its_limit(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from sleeve_fund.dashboard import app as app_mod

    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    store, _ = _gated_store(tmp_path, monkeypatch, {"BTC": 9_000, "ETH": -3_000})
    page = TestClient(app_mod.create_app(store)).get("/risk", auth=("pm", "test-pw")).text
    assert 'Net 60% long · <span class="warn"' in page
    assert "Total size (long + short) 1.20x of book, limit 1.5x" in page and "every position and resting entry" in page
    head = page.split('id="strats-h"')[1].split("</div>")[0]
    assert "Gross" not in head and "gross" not in head


# --- UP-2: the Position tab's side and leverage, spot and perp ---------------------------------------------------------


def _page(tmp_path, monkeypatch, name, params, fills, mark):
    from fastapi.testclient import TestClient

    from sleeve_fund.dashboard import app as app_mod
    from sleeve_fund.store import Store

    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    store = Store(f"sqlite:///{tmp_path}/p.db")
    store.create_sleeve(name=name, strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, risk_profile="balanced",
                        params={"rise": 0.01, "dip": 0.005, "stop_loss": 0.02, **params})
    cash = 10_000.0
    for n, (side, qty, px) in enumerate(fills):
        store.record_fill(name, side=side, qty=qty, price=px, fee=0.0, order_id=f"o{n}", trade_id=f"t{n}")
        cash -= (qty if side == "BUY" else -qty) * px
    qty = sum(q if s == "BUY" else -q for s, q, _ in fills)
    store.record_equity(name, equity=cash + qty * mark, cash=cash, qty=qty, price=mark, benchmark=10_000)
    return TestClient(app_mod.create_app(store)).get(f"/sleeves/{name}", auth=("pm", "test-pw")).text


def test_up2_a_perp_reads_its_leverage_in_the_cap_format_and_the_drifted_one_as_effective(tmp_path, monkeypatch):
    """Long 0.05 at 60,000 at the balanced 2x cap; the price is now 62,100, so notional now over the same margin is
    2.07x. The Position tab reads "2×" and "effective 2.07×", never "2×" twice; the table's chip is the cap."""
    html = _page(tmp_path, monkeypatch, "lev-drift", {"market": "perp", "allow_short": True},
                 [("BUY", 0.05, 60_000.0)], 62_100.0)
    lev = html.split(">Leverage</dt><dd>")[1].split("</dd>")[0]
    assert lev == '2× <span class="faint">· effective 2.07×</span>'
    assert lev.count("2×") == 1 and "2.00×" not in lev
    assert '<span class="lev">2×</span>' in html  # the positions table: the cap
    assert "above entry cap because the price moved" not in html  # the note is unchanged: not triggered here


def test_up2_a_spot_position_shows_its_side_leverage_and_exposure(tmp_path, monkeypatch):
    html = _page(tmp_path, monkeypatch, "spot-pos", {}, [("BUY", 0.05, 60_000.0)], 60_000.0)
    size = html.split("<h3>Size</h3>")[1].split("</dl>")[0]
    assert "<dt>Side</dt><dd>Long</dd>" in size
    assert '<dt title="Spot: no leverage">Leverage</dt><dd>1×</dd>' in size
    assert ">Exposure</dt><dd>0.30× equity" in size


# --- UP-3: P&L by strategy, biggest contributor first ------------------------------------------------------------------


def test_up3_a_bigger_loss_is_listed_above_a_smaller_gain_and_still_reads_as_a_loss(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from sleeve_fund.dashboard import app as app_mod
    from sleeve_fund.store import Store

    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    store = Store(f"sqlite:///{tmp_path}/m.db")
    for name, equity in (("small-gain", 10_100.0), ("big-loss", 9_700.0)):
        # On the Strategy venue, in the fund's book (CCY-8DP-UI: BTC/USD at the default venue is a test account).
        store.create_sleeve(name=name, strategy="trend_filter", instrument="BTC/USDT", bar_spec="1-HOUR-LAST-INTERNAL",
                            starting_balance=10_000, params={}, venue="BINANCE")
        store.record_equity(name, equity=equity, cash=equity, qty=0, price=60_000, benchmark=10_000)
    html = TestClient(app_mod.create_app(store)).get("/", auth=("pm", "test-pw")).text
    today = html.split('data-period-rows="day"')[1].split("</tbody>")[0]
    rows = today.split("<tr")[1:]
    assert "big-loss" in rows[0] and "small-gain" in rows[1]  # by size: -300 before +100
    assert '<i class="loss" style="right:50%' in rows[0] and "−300.00" in rows[0]  # still a loss, bar to the left
    assert '<i class="gain" style="left:50%' in rows[1] and "+100.00" in rows[1]
    assert "Book" in rows[-1] and "−200.00" in rows[-1]  # the rows still add up to Book
