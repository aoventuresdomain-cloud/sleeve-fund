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
