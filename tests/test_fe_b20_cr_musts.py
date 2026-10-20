"""CR's MUSTs on fe-dfdc2a8 (B-20). MUST 1: the Risk page's "would halt the fund" reads the gate's own reference
(PortfolioState.reference(): the high-water mark from every mark, or the PM's Resume re-base), not the daily curve's
last drawdown, which misses an intraday peak and is 0 before a daily close; unknown when it can't be read, never "would
not halt". MUST 2: each fund-held figure CR's mutants survived gets a cell. On Postgres too (TEST_DATABASE_URL)."""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from test_fe_b20_fund_held_wording import _hold, _mark, _summaries
from test_halt_ref import (
    store,  # noqa: F401 - the journal, on Postgres when TEST_DATABASE_URL is set
)

from sleeve_fund import supervisor

AUTH = ("pm", "pw")
PERP = {"market": "perp", "allow_short": True}
ENTRY, FEE = 60_000.0, 3.3


@pytest.fixture
def c(store, tmp_path, monkeypatch):  # noqa: F811
    from sleeve_fund.dashboard import app as app_mod

    monkeypatch.setenv("DASHBOARD_PASSWORD", "pw")
    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    monkeypatch.setattr(app_mod, "LEDGER", tmp_path / "idea_ledger.jsonl")
    return TestClient(app_mod.create_app(store))


def _long(store, name, qty, mark, ts=None):  # noqa: F811
    """A fund-held 2x perp long (made with 0, a 2% stop) of `qty` at 60,000 marked at `mark`, as the engine journals."""
    store.create_sleeve(name=name, strategy="ping_pong", instrument="BTC/USDT", venue="binance",
                        bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=0, desired_state="running",
                        params={"rise": 0.01, "dip": 0.005, "stop_loss": 0.02, **PERP})
    store.record_order(name, order_id=f"{name}-1", side="BUY", qty=qty, intent="entry", reason="dipped 0.5%")
    store.update_order(f"{name}-1", fill_qty=qty, fill_px=ENTRY, fee=FEE)
    store.record_fill(name, side="BUY", qty=qty, price=ENTRY, fee=FEE, order_id=f"{name}-1", trade_id=f"{name}-T")
    cash = -qty * ENTRY - FEE
    store.record_equity(name, equity=cash + qty * mark, cash=cash, qty=qty, price=mark, benchmark=0, ts=ts)
    return store.sleeve(name)


def _views(store):  # noqa: F811
    from sleeve_fund.dashboard import book as bookm
    from sleeve_fund.dashboard import riskops

    summaries, frames = _summaries(store)
    book = bookm.book_view(store, summaries, frames)
    return riskops.risk_view(store, summaries, book), book, summaries, frames


def _row(view, shock):
    return next(sc for sc in view["scenarios"] if sc["shock"] == shock)


def test_must1_the_funds_halt_is_measured_from_the_gates_high_water_mark(c, store):  # noqa: F811
    """CR's case: the fund marked at 105,000, now about 99,900, no daily close below it. A 20% fall of a 1 BTC long
    loses 11,800: 11.8% from today's equity (no halt on the daily curve) but 19.2% from the high-water mark, past the
    15% brake, as the gate reads it. A 10% fall stays under it."""
    from sleeve_fund.portfolio import compliance

    _hold(store)
    _long(store, "big", 1.0, 59_000.0)
    view, book, _, _ = _views(store)
    _mark(store, round(book["equity"], 2), hwm=105_000)
    view, book, _, _ = _views(store)
    t5b = float(compliance.load(store).drawdown)
    for shock, halts in ((-0.20, True), (-0.10, False)):
        loss = _row(view, shock)["loss"]
        assert (1 - (book["equity"] - loss) / 105_000 >= t5b) is halts
        assert _row(view, shock)["fund_halt"] is halts, (shock, loss, book["equity"])
    assert not view["fund_halt_unknown"]
    page = c.get("/risk", auth=AUTH).text
    assert "would halt <span" in page and "the fund" in page


def test_must1_unmarked_the_funds_halt_is_unknown_never_would_not_halt(c, store):  # noqa: F811
    _hold(store)
    _long(store, "big", 1.0, 59_000.0)
    view, _, _, _ = _views(store)
    assert view["fund_halt_unknown"] and all(sc["fund_halt"] is None for sc in view["scenarios"])
    page = c.get("/risk", auth=AUTH).text
    assert "Whether the fund would halt is unknown" in page and "The fund would not halt" not in page
    assert "The fund: unknown" in page


def test_must2b_a_strategys_own_drawdown_never_breaches_inside_the_fund(store):  # noqa: F811
    """A Strategy made with 0 that is up 597 (its equity), whose 20% fall would take it far below its own peak: only
    the fund's brake acts, so no row names it."""
    _hold(store)
    _long(store, "up", 0.1, 66_000.0)
    view, _, summaries, _ = _views(store)
    assert summaries[0]["equity"] > 0
    assert not any(sc["breaches"] for sc in view["scenarios"])
    assert not any(s["breach"] for s in view["rows"][0]["shocks"])


def test_must2cd_todays_move_is_over_the_funds_equity_and_uses_no_day_pause(store):  # noqa: F811
    from sleeve_fund.dashboard.metrics import fund_equity_or_none

    _hold(store)
    _long(store, "down", 0.1, 59_000.0)
    view, _, summaries, _ = _views(store)
    x = summaries[0]
    assert x["day_ret"] < 0
    assert x["day_ret"] == pytest.approx(x["day_pnl"] / fund_equity_or_none(store), rel=1e-12)  # (d)
    assert view["rows"][0]["day_used"] == 0.0  # (c): its own day pause doesn't act inside the fund


def test_must2ef_the_benchmark_and_the_cash_row_count_the_funds_cash(store):  # noqa: F811
    from sleeve_fund.dashboard import book as bookm

    _hold(store)
    _long(store, "down", 0.1, 59_000.0)
    _, book, summaries, _ = _views(store)
    base = bookm.fund_cash(store)
    assert base == pytest.approx(100_000)
    assert book["bench_ret"] == pytest.approx((sum(x["benchmark"] for x in summaries) + base) / book["start"] - 1)
    assert book["bench_ret"] == pytest.approx(0.0, abs=1e-12)  # (e): not -100%
    cash = next(r for r in book["allocation"] if r["name"] == "Cash")
    assert cash["value"] == pytest.approx(summaries[0]["cash"] + base)  # (f)


def test_must2g_the_book_curve_measures_its_drawdown_from_the_funds_capital(store):  # noqa: F811
    """The first daily close is already 500 down: 0.5% below the 100,000 the fund started with, not 0%."""
    from sleeve_fund.dashboard import book as bookm

    _hold(store)
    ts = datetime.now(UTC) - timedelta(days=2)
    store.create_sleeve(name="one", strategy="ping_pong", instrument="BTC/USDT", venue="binance",
                        bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=0, params={"rise": 0.01, "dip": 0.005})
    store.record_equity("one", equity=-500, cash=-500, qty=0, price=60_000, benchmark=0, ts=ts)
    summaries, frames = _summaries(store)
    curve = bookm.book_curve(summaries, frames, bookm.fund_cash(store))
    assert curve["equity"].iloc[0] == pytest.approx(99_500)
    assert curve["drawdown"].iloc[0] == pytest.approx(0.005)


def test_must2h_the_withdrawal_previews_margin_reads_the_saved_limit(store, monkeypatch):  # noqa: F811
    from dataclasses import replace
    from decimal import Decimal
    from types import SimpleNamespace

    from sleeve_fund.dashboard import withdrawal
    from sleeve_fund.paper import portfolio_gate
    from sleeve_fund.portfolio import compliance

    _hold(store)
    saved = replace(compliance.load(store), gross=3.0, net_instrument=1.0, margin=0.8)
    compliance.save(store, saved, changed_by="pm", note="T4 at its maxima")
    ledger = SimpleNamespace(state=lambda: SimpleNamespace(equity=Decimal(100_000)), positions=lambda: ({}, ()),
                             reservations=lambda: ())
    monkeypatch.setattr(portfolio_gate, "PaperGate", lambda st: SimpleNamespace(ledger=ledger))
    limits = {x["key"]: x["limit"] for x in withdrawal.limits_in_use(store)["limits"]}
    assert limits["margin"] == pytest.approx(0.8), limits
    assert float(supervisor.fund_book(store).capital) == pytest.approx(100_000)
