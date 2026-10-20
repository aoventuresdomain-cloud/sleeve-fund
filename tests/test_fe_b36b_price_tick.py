"""B-36b (HoE ruling (A), Head 5): the dashboard puts a perp's liquidation on the venue's price tick, as the engine
does, through DA's shared reader markets.price_tick(venue, pair) (the tick kept beside the tiers; None when not kept).
The cells stub the reader: with a tick, the Portfolio row's liquidation and the stress cap read the ticked
price (margin.to_tick, toward the entry); with none, both read as before. On Postgres too (TEST_DATABASE_URL)."""

import pytest
from test_fe_b20_fund_held_wording import _engine_liquidation, _hold, _summaries
from test_fe_b36_stress_cap import ENTRY, QTY, _long
from test_halt_ref import (
    store,  # noqa: F401 - the journal, on Postgres when TEST_DATABASE_URL is set
)

from sleeve_fund import margin, markets
from sleeve_fund.venues import venue as venue_profile

TICK = 7.0  # coarse on purpose, so the ticked price is never the raw one by chance


def _stub(monkeypatch, tick):
    seen = []
    monkeypatch.setattr(markets, "price_tick", lambda venue, pair: seen.append((venue, pair)) or tick)
    return seen


def _row(store):  # noqa: F811
    from sleeve_fund.dashboard import trading

    summaries, _ = _summaries(store)
    return trading.book_positions(store, summaries)["rows"][0], summaries[0]


@pytest.mark.parametrize("mark", [ENTRY * 0.98, ENTRY * 1.05])
def test_the_portfolio_rows_liquidation_is_on_the_venues_tick(store, monkeypatch, mark):  # noqa: F811
    _hold(store)
    s = _long(store, "a", mark)
    raw, posted = _engine_liquidation(store, s)
    seen = _stub(monkeypatch, TICK)
    row, _ = _row(store)
    want = margin.to_tick(raw, QTY, TICK)
    assert want != pytest.approx(raw, abs=1e-9) and want % TICK == pytest.approx(0, abs=1e-6)
    assert want > raw  # a long's moves up, toward the entry
    assert row["liq_px"] == pytest.approx(want)
    assert row["margin"] == pytest.approx(posted)  # the tick moves the price, not the margin
    assert (s.venue, s.instrument) in seen


def test_the_stress_cap_books_the_fee_at_the_ticked_price(store, monkeypatch):  # noqa: F811
    from sleeve_fund.dashboard.riskops import most_it_can_lose

    _hold(store)
    mark = ENTRY * 0.80
    s = _long(store, "lose", mark)
    raw, posted = _engine_liquidation(store, s)
    _stub(monkeypatch, TICK)
    _, x = _row(store)
    liq = margin.to_tick(raw, QTY, TICK)
    taker = float(markets.fees_for(s.params, venue_profile(s.venue).fees, s.venue).taker)
    want = posted + QTY * (mark - ENTRY) + taker * QTY * liq
    assert abs(taker * QTY * (liq - raw)) > 1e-5  # the tick changes the fee by more than the tolerance below
    assert most_it_can_lose(x, 1e12) == pytest.approx(want, rel=0, abs=1e-6)


def test_no_kept_tick_reads_as_before(store, monkeypatch):  # noqa: F811
    _hold(store)
    s = _long(store, "a", ENTRY * 0.98)
    raw, _ = _engine_liquidation(store, s)
    _stub(monkeypatch, None)
    row, _ = _row(store)
    assert row["liq_px"] == pytest.approx(raw)


def test_a_reader_that_fails_shows_never_the_old_figure(store, monkeypatch):  # noqa: F811
    """HoE (Head 5 merge rule): a money figure that silently falls back is what B-36b fixes, so a tick reader that
    fails is an error the page shows, never the unticked price."""
    def boom(venue, pair):
        raise RuntimeError("tick store unreadable")

    _hold(store)
    _long(store, "a", ENTRY * 0.98)
    monkeypatch.setattr(markets, "price_tick", boom)
    with pytest.raises(RuntimeError, match="tick store unreadable"):
        _row(store)
