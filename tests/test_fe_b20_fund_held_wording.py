"""B-20 (HoE board, with QD's audit qd-v3 §5): once the fund holds the cash (the re-base deposit applied,
supervisor.bridge_start) every Strategy is made with 0 (CUT-A (a)), its own drawdown halt and day pause don't act (the
fund's brakes T5 do, CUT-A (d)), and its marks are its P&L from 0. So no page states a per-Strategy halt, room, drawdown
or capital inside a fund-held book, exposure and today's move are over the fund's equity, and a perp's margin and
liquidation are the engine's (base.py `cash + self._funded()`), not worked from a P&L cash. A store from before the
bridge reads as it did. On Postgres too (TEST_DATABASE_URL)."""

import html as htmlmod
import re
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from test_halt_ref import (
    store,  # noqa: F401 - the journal, on Postgres when TEST_DATABASE_URL is set
)

from sleeve_fund import margin, markets, risk, supervisor

AUTH = ("pm", "pw")
PERP = {"market": "perp", "allow_short": True}
QTY, ENTRY, MARK, FEE = 0.1, 60_000.0, 59_000.0, 3.3
# The old rule's words (fe-v3 §4 and the B-20 board line): a Strategy's own halt level, room, capital or drawdown.
OLD_RULE = (r"Room to halt", r"Room before halt", r"halts? at \d", r"Starting capital", r"No strategy would halt",
            r"Strategies that would halt", r"of its capital", r"Worst drawdown", r"before halt")


@pytest.fixture
def c(store, tmp_path, monkeypatch):  # noqa: F811
    from sleeve_fund.dashboard import app as app_mod

    monkeypatch.setenv("DASHBOARD_PASSWORD", "pw")
    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    monkeypatch.setattr(app_mod, "LEDGER", tmp_path / "idea_ledger.jsonl")
    return TestClient(app_mod.create_app(store))


def _hold(store, amount=100_000):  # noqa: F811
    at = datetime.now(UTC) - timedelta(minutes=5)
    store.settle_flow(store.fund_flow("deposit", amount, actor="pm", reason="the PM's reset", rebase=True, now=at), at)


def _long(store, name="held-long", balance=0):  # noqa: F811
    """A perp long of 0.1 at 60,000 marked at 59,000, made with `balance` (0 inside the fund). Its mark is as the
    engine journals it: spot-style cash (the notional and fee out), with the fund's virtual funding netted out."""
    store.create_sleeve(name=name, strategy="ping_pong", instrument="BTC/USDT", venue="binance",
                        bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=balance,
                        params={"rise": 0.01, "dip": 0.005, **PERP})
    store.record_order(name, order_id=f"{name}-1", side="BUY", qty=QTY, intent="entry", reason="dipped 0.5%")
    store.update_order(f"{name}-1", fill_qty=QTY, fill_px=ENTRY, fee=FEE)
    store.record_fill(name, side="BUY", qty=QTY, price=ENTRY, fee=FEE, order_id=f"{name}-1", trade_id=f"{name}-T")
    cash = balance - QTY * ENTRY - FEE
    store.record_equity(name, equity=cash + QTY * MARK, cash=cash, qty=QTY, price=MARK, benchmark=balance)
    return store.sleeve(name)


def _mark(store, equity=100_000, hwm=None):  # noqa: F811
    """The supervisor's marks of the fund as the gate keeps them (gate.mark_book): one at the high-water mark `hwm`
    (if any) a minute ago, then `equity` now."""
    from sleeve_fund.paper.portfolio_gate import PaperGate
    from sleeve_fund.portfolio.gate import mark_book
    from sleeve_fund.risk import PORTFOLIO

    ledger, now = PaperGate(store).ledger, datetime.now(UTC)
    if hwm is not None:
        mark_book(ledger, str(hwm), now - timedelta(minutes=1), PORTFOLIO)
    mark_book(ledger, str(equity), now, PORTFOLIO)


def _engine_liquidation(store, s):  # noqa: F811
    """The engine's liquidation for the position, worked independently of the dashboard: base.py's
    isolated_liquidation(cash + self._funded()), funded = the fund's book x VIRTUAL_HEADROOM (paper/node.py)."""
    from sleeve_fund.paper.node import VIRTUAL_HEADROOM

    t, tiered = markets.held_terms(s.params, s.venue, s.instrument)
    assert tiered, "the test pair needs a kept tier table"
    funded = float(supervisor.fund_book(store).total) * VIRTUAL_HEADROOM
    lev = risk.leverage_cap(risk.profile(s.risk_profile), s.params)
    cash = -QTY * ENTRY - FEE
    return margin.isolated_liquidation(cash + funded, QTY, ENTRY, lev, t.maintenance_margin), QTY * ENTRY / lev


def _words(c, path):
    r = c.get(path, auth=AUTH)
    assert r.status_code == 200, path
    return " ".join(htmlmod.unescape(re.sub(r"<[^>]+>", " ", r.text)).split()), htmlmod.unescape(r.text)


def _summaries(store):  # noqa: F811
    from sleeve_fund.dashboard import book as bookm
    from sleeve_fund.dashboard.metrics import sleeve_summary

    frames = {s.name: bookm.daily(store, s.name) for s in store.sleeves()}
    return [bookm.sleeve_extras(store, sleeve_summary(store, s), frames[s.name]) for s in store.sleeves()], frames


def test_fund_held_a_perps_margin_and_liquidation_are_the_engines(c, store):  # noqa: F811
    """The finding: from the mark's cash alone (P&L from 0) the margin read 0 and the liquidation sat above a long's
    entry, as if it were already liquidated."""
    from sleeve_fund.dashboard import trading

    _hold(store)
    s = _long(store)
    liq, posted = _engine_liquidation(store, s)
    assert liq is not None and liq < ENTRY
    summaries, _ = _summaries(store)
    row = trading.book_positions(store, summaries)["rows"][0]
    assert row["margin"] == pytest.approx(posted)
    assert row["liq_px"] == pytest.approx(liq)
    # the wiring: the Strategy page shows the same, and no page throws
    words, _ = _words(c, f"/sleeves/{s.name}")
    shown = re.search(r"Liquidation ([\d,]+(?:\.\d+)?)", words)
    assert shown and float(shown.group(1).replace(",", "")) == pytest.approx(liq, rel=1e-4), words[:300]
    for path in ("/", "/risk"):
        _words(c, path)


def test_fund_held_stress_counts_the_position_not_its_pnl(store):  # noqa: F811
    """At stake is what the position can lose (its margin and the fee, on isolated margin), not the Strategy's P&L,
    which inside the fund is about 0."""
    from sleeve_fund.dashboard import book as bookm
    from sleeve_fund.dashboard import riskops

    _hold(store)
    s = _long(store)
    summaries, frames = _summaries(store)
    book = bookm.book_view(store, summaries, frames)
    view = riskops.risk_view(store, summaries, book)
    down20 = view["down20"]
    raw = QTY * MARK * 0.20
    assert down20["loss"] == pytest.approx(raw)  # 1,180: inside what the position can lose
    from sleeve_fund.venues import venue as venue_profile

    liq, posted = _engine_liquidation(store, s)
    taker = float(markets.fees_for(s.params, venue_profile(s.venue).fees, s.venue).taker)
    # from the mark: its margin and the liquidation fee (markets.gap_loss_cap), less what it has already lost
    at_stake = posted + taker * QTY * liq + QTY * (MARK - ENTRY)
    down50 = next(sc for sc in view["scenarios"] if sc["shock"] == -0.50)
    assert down50["loss"] == pytest.approx(at_stake)  # capped at what the position can lose, never at its P&L
    assert not any(sc["breaches"] for sc in view["scenarios"])  # its own halt doesn't act inside the fund


def test_fund_held_no_page_states_a_strategys_own_halt_room_or_capital(c, store):  # noqa: F811
    _hold(store)
    s = _long(store)
    _mark(store, 99_896.7)  # the gate's mark, which the fund's halt line reads its high-water mark from
    for path in ("/", "/risk", "/setup", f"/sleeves/{s.name}", "/sleeves/new"):
        words, _ = _words(c, path)
        hits = [w for w in OLD_RULE if re.search(w, words)]
        assert hits == [], (path, hits)
    home, _ = _words(c, "/")
    assert "Fund's brakes" in home
    riskp, _ = _words(c, "/risk")
    assert "The fund's brakes" in riskp and "The fund would not halt at ±20%" in riskp
    setup, _ = _words(c, "/setup")
    assert "leverage up to 2x" in setup and "The fund's brakes pause and halt every Strategy" in setup
    page, _ = _words(c, f"/sleeves/{s.name}")
    assert "Held by the fund" in page and "the fund's brakes pause and halt it" in page
    assert "× fund equity" in page
    form, _ = _words(c, "/sleeves/new")
    assert "Risk per trade, % of fund equity" in form


def test_fund_held_exposure_and_today_are_over_the_funds_equity(store):  # noqa: F811
    _hold(store)
    _long(store)
    summaries, _ = _summaries(store)
    x = summaries[0]
    fund = float(supervisor.fund_equity(store))
    assert x["in_fund"] and x["room"] is None and x["dd_used"] == 0 and x["drawdown"] == 0
    assert x["exposure"] == pytest.approx(QTY * MARK / fund)
    assert abs(x["day_ret"]) < 0.01  # a 103.30 loss over a fund of about 99,900, not over a P&L of -103.30


def test_fund_held_the_risk_pages_halt_line_is_the_funds_own_brake(c, store):  # noqa: F811
    from sleeve_fund.portfolio import compliance

    _hold(store)
    _long(store)
    _, page = _words(c, "/risk")
    t5b = compliance.load(store).drawdown
    assert 'data-halt-name="fund"' in page
    assert f'"halt": {float(t5b)}' in page or f'"halt":{float(t5b)}' in page


def test_the_size_limit_shown_is_the_saved_version_not_the_seeded_constant(c, store):  # noqa: F811
    from dataclasses import replace

    from sleeve_fund.portfolio import compliance

    _hold(store)
    _long(store)
    compliance.save(store, replace(compliance.load(store), gross=3.0), changed_by="pm", note="B-20 cell")
    words, _ = _words(c, "/risk")
    assert "limit 3.0x" in words


def test_fund_held_a_strategys_chart_has_no_drawdown_of_its_own(c, store):  # noqa: F811
    _hold(store)
    s = _long(store)
    store.record_equity(s.name, equity=50, cash=50, qty=0, price=MARK, benchmark=0)
    store.record_equity(s.name, equity=-20, cash=-20, qty=0, price=MARK, benchmark=0)
    d = c.get(f"/api/sleeves/{s.name}/equity", auth=AUTH).json()
    assert d["in_fund"] is True and max(d["drawdown"]) == 0 and d["worst"] == 0


def test_fund_held_correlation_is_a_number_not_blank(store):  # noqa: F811
    from sleeve_fund.dashboard import book as bookm

    _hold(store)
    now = datetime.now(UTC)
    for n, sign in (("a", 1), ("b", -1)):
        store.create_sleeve(name=n, strategy="ping_pong", instrument="BTC/USDT", venue="binance",
                            bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=0, params={"rise": 0.01, "dip": 0.005})
        pnl = 0.0
        for d in range(bookm.MIN_DAYS_FOR_CORRELATION + 5, 0, -1):
            pnl += sign * (10 if d % 2 else -7)
            store.record_equity(n, equity=pnl, cash=pnl, qty=0, price=60_000, benchmark=0, ts=now - timedelta(days=d))
    summaries, frames = _summaries(store)
    corr = bookm.correlation(summaries, frames)
    assert corr["matrix"][0][1] == pytest.approx(-1.0)


def test_fund_held_reports_read_as_contribution_to_the_fund(c, store):  # noqa: F811
    _hold(store)
    s = _long(store)
    words, _ = _words(c, f"/reports?sleeve={s.name}")
    assert "over the fund's capital" in words


def test_not_held_a_strategy_with_its_own_capital_reads_as_before(c, store):  # noqa: F811
    s = _long(store, "own-long", balance=10_000)
    words, _ = _words(c, "/")
    assert "Room to halt" in words and "Fund's brakes" not in words
    page, _ = _words(c, f"/sleeves/{s.name}")
    assert "Starting capital" in page and "halt at" in page
    from sleeve_fund.dashboard import trading

    summaries, _ = _summaries(store)
    row = trading.book_positions(store, summaries)["rows"][0]
    lev = risk.leverage_cap(risk.profile(s.risk_profile), s.params)
    t, _ = markets.held_terms(s.params, s.venue, s.instrument)
    assert row["liq_px"] == pytest.approx(margin.isolated_liquidation(10_000 - QTY * ENTRY - FEE, QTY, ENTRY, lev,
                                                                      t.maintenance_margin))


def test_the_withdrawal_preview_and_the_risk_page_read_the_saved_size_limits(c, store, monkeypatch):  # noqa: F811
    """QD via the HoE: with the PM's T4-at-maxima workaround the gate applies 3x gross and 1x net, while the withdrawal
    preview (withdrawal.py) and the Risk size line (riskops) read the module's seeded 1.5x and 0.5x."""
    from dataclasses import replace
    from decimal import Decimal
    from types import SimpleNamespace

    from sleeve_fund.dashboard import withdrawal
    from sleeve_fund.paper import portfolio_gate
    from sleeve_fund.portfolio import compliance

    _hold(store)
    _long(store)
    compliance.save(store, replace(compliance.load(store), gross=3.0, net_instrument=1.0), changed_by="pm",
                    note="T4 at its maxima")
    ledger = SimpleNamespace(state=lambda: SimpleNamespace(equity=Decimal(100_000)), positions=lambda: ({}, ()),
                             reservations=lambda: ())
    monkeypatch.setattr(portfolio_gate, "PaperGate", lambda st: SimpleNamespace(ledger=ledger))
    limits = {x["key"]: x["limit"] for x in withdrawal.limits_in_use(store)["limits"]}
    assert limits["gross"] == pytest.approx(3.0) and limits["net"] == pytest.approx(1.0), limits
    words, _ = _words(c, "/risk")
    assert "limit 3.0x" in words and "1.5x" not in words
