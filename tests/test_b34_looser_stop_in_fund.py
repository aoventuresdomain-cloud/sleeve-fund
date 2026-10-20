"""B-34 row 11 (QD capital audit, 15 Oct 2026; HoE: fix first, it blocks a PM action): inside the fund a Strategy's
own equity is its P&L from 0, so the settings save compared a looser stop with a per-Strategy drawdown halt measured
on that P&L. After any loss the "room" was negative and every looser stop was refused, however small. In the fund the
halt that would close the position is the fund's drawdown halt (T5), on the fund's last mark.

Expected figures are worked by hand: fund 100,000 marked once, so its reference is 100,000 and the T5 halt (15%,
seeded) leaves 15,000."""

from __future__ import annotations

from datetime import timedelta
from urllib.parse import unquote_plus

import pytest
from fastapi.testclient import TestClient

from sleeve_fund.store import Store, utcnow

AUTH = ("pm", "test-pw")
SAME = {"origin": "http://testserver"}
NAME = "btc-fund"


@pytest.fixture(params=[True], ids=["marked"])
def fund(request, tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    monkeypatch.setenv("TEARSHEET_DIR", str(tmp_path))
    from sleeve_fund.dashboard import app as app_mod
    from sleeve_fund.supervisor import Supervisor

    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    monkeypatch.setattr(app_mod, "LEDGER", tmp_path / "idea_ledger.jsonl")
    store = Store(f"sqlite:///{tmp_path}/fund.db", portfolio_gate=True)
    before = utcnow() - timedelta(hours=1)
    store.settle_flow(store.fund_flow("deposit", 100_000, actor="pm", reason="B-34: the fund's capital", rebase=True,
                                      now=before), before)
    store.create_sleeve(name=NAME, strategy="trend_filter", instrument="BTC/USD", bar_spec="1-HOUR-LAST-EXTERNAL",
                        starting_balance=0, risk_profile="balanced", params={"stop_loss": 0.03})
    if store.joining(NAME):
        store.land_move(NAME)
    if request.param:
        Supervisor(store).mark_portfolio(utcnow())
    return TestClient(app_mod.create_app(store)), store


def _hold(store, qty: float, loss: float):
    """An open long of `qty` at 50,000 with a 3% stop, and the Strategy's own mark at a `loss` (P&L from 0)."""
    from sleeve_fund.paper.runtime import SleeveRuntime

    rt = SleeveRuntime(store, NAME)
    rt.on_order(order_id="E-1", side="BUY", qty=qty, intent="entry", reason="Signal to be long",
                signal={"stop_frac": 0.03, "risk_amount": round(qty * 50_000 * 0.03, 2),
                        "stop_cfg": {"stop_loss": 0.03}})
    rt.on_fill(side="BUY", qty=qty, price=50_000.0, fee=24.0, order_id="E-1", trade_id="T-1")
    store.record_equity(NAME, equity=-loss, cash=-qty * 50_000 - 24.0, qty=qty, price=50_000.0, benchmark=0)


def _save(c, **over):
    form = {"risk_profile": "balanced", "instrument": "BTC/USD", "reason": "looser stop", **over}
    r = c.post(f"/sleeves/{NAME}/settings", data=form, auth=AUTH, headers=SAME, follow_redirects=False)
    return unquote_plus(r.headers["location"])


def test_after_a_loss_a_small_looser_stop_saves_with_the_pms_confirmation(fund):
    c, store = fund
    _hold(store, 0.06, loss=24.0)  # a 24 fee loss: on main the room was -24 and this was refused
    loc = _save(c, stop_loss_pct="5")
    assert "looser than the 3.0%" in loc and "drawdown halt" not in loc, loc  # asks to confirm; never refused
    loc = _save(c, stop_loss_pct="5", confirm_looser="1")
    assert "saved=settings" in loc, loc
    assert store.sleeve(NAME).params["stop_loss"] == 0.05


def test_a_stop_that_would_lose_more_than_the_fund_has_before_its_halt_is_refused_in_its_words(fund):
    """2 BTC at 50,000 with a 20% stop: 2 x 10,000 = 20,000 from here, plus the fee to close, over the 15,000 left
    before the fund's 15% halt."""
    c, store = fund
    _hold(store, 2.0, loss=24.0)
    loc = _save(c, stop_loss_pct="20", confirm_looser="1")
    assert "15,000.00 left before the fund's drawdown halt (15%, T5)" in loc, loc
    assert store.sleeve(NAME).params["stop_loss"] == 0.03
    # 2 BTC with a 7% stop: 7,000 plus the fee, inside the 15,000: saves
    assert "saved=settings" in _save(c, stop_loss_pct="7", confirm_looser="1")


@pytest.mark.parametrize("fund", [False], ids=["never_marked"], indirect=True)
def test_a_looser_stop_on_a_fund_never_marked_is_refused_until_it_can_be_checked(fund):
    c, store = fund
    _hold(store, 0.06, loss=24.0)
    loc = _save(c, stop_loss_pct="5", confirm_looser="1")
    assert "hasn't been marked yet" in loc and store.sleeve(NAME).params["stop_loss"] == 0.03, loc
    assert "saved=settings" in _save(c, stop_loss_pct="2")  # tighter: never asks the fund
