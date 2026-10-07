"""v2 P2-2: one open-risk formula from a position to the book (cell G5), and trailing stops (cells G12-G14, Advisor
18:55 and 19:00 UK): a trail enforced as a resting or every-tick stop counts at its current level, a candle-close trail
with nothing resting counts at the stopless measure, and a resting hard stop bounds a trailing position."""

from decimal import Decimal as D

import pytest

from sleeve_fund.open_risk import position_risk
from sleeve_fund.portfolio import Position, holding_for
from sleeve_fund.portfolio.book import backtest_book, book_equity
from sleeve_fund.portfolio.holding import ALIASES, underlying


def _long(qty="2", stop=None, lev=5.0, instrument="BTC/USDT"):
    return Position("s1", underlying(instrument), D(qty), None if stop is None else D(stop), lev)


@pytest.mark.parametrize("side, stop, atr", [
    (1, "95", None),  # stopped: to its stop from the mark
    (-1, "105", None),  # a short's stop above the mark
    (1, None, 0.02),  # stopless: the 10% floor
    (1, None, 0.05),  # stopless: 3 daily ATRs
    (1, "101", 0.02),  # the price has gone through its stop: stopless, never 0
])
def test_g5_the_books_holding_risk_is_the_interim_checks_formula(side, stop, atr):
    pos = Position("s1", "BTC", side * D("2"), None if stop is None else D(stop), 5.0)
    h = holding_for(pos, D("100"), atr)
    interim = position_risk(side * 2.0, 100.0, None if stop is None else float(stop), atr)
    assert float(h.risk) == pytest.approx(interim, rel=1e-12) and isinstance(h.risk, D)  # exact vs float
    assert (h.notional, h.margin) == (D(side * 200), D("40"))


def test_g12_a_trail_moved_into_profit_counts_at_its_current_stop():
    """Case (a): entered at 100, the trail now rests at 105 with the mark at 110: 2 x (110 - 105) = 10 at risk."""
    assert holding_for(_long(stop="105"), D("110"), None).risk == D("10")


def test_g13_a_candle_close_trail_with_nothing_resting_counts_at_the_stopless_measure():
    """Case (b), rsi_pullback on main: no stop rests, so the position counts notional x max(10%, 3 daily ATR),
    whatever level its trail has reached; it never needs that level, so journalling it can't change the figure."""
    assert holding_for(_long(), D("110"), 0.02).risk == D("22.0")  # 220 x 10%
    assert holding_for(_long(), D("110"), 0.05).risk == D("33.0")  # 220 x 15%
    with pytest.raises(ValueError, match="daily ATR"):
        holding_for(_long(), D("110"), None)  # unmeasured: the gate's check fails closed on it


def test_g14_a_resting_hard_stop_bounds_a_trailing_position_and_without_one_it_is_stopless():
    """Advisor 19:00 UK: a candle-close trail plus a resting hard stop counts to the hard stop; with no resting stop,
    stopless. The dashboard tile's figure equals the gate's in both: FE's open_risk.book_open_risk reuses this path
    and pins it in its own tests (agreed 19:00 UK)."""
    assert holding_for(_long(stop="95"), D("110"), 0.02).risk == D("30")  # 2 x (110 - 95), the trail ignored
    assert holding_for(_long(), D("110"), 0.02).risk == D("22.0")


def test_spot_counts_its_risk_to_stop_and_its_full_notional_as_margin():
    """Advisor GATE-SPOT (20:20 UK): the book-wide 5% counts spot positions' risk to their stop too."""
    h = holding_for(_long(stop="95", lev=None), D("100"), None)
    assert (h.margin, h.risk) == (D("200"), D("10"))


def test_net_counts_by_underlying_with_a_venues_own_code_mapped():
    assert holding_for(_long(instrument="XBT/USD", stop="95"), D("100"), None).underlying == "BTC"


def test_the_book_is_the_whole_fund_and_a_backtests_is_over_its_share():
    assert book_equity([D("3000"), D("3000")], D("4000")) == D("10000")
    book, label = backtest_book(D("2000"), 0.2)
    assert book == D("10000") and "others flat" in label
    with pytest.raises(TypeError):
        book_equity([3_000.0], D("4000"))


def test_the_alias_map_matches_the_research_one():
    from sleeve_fund.research.holdout import _ALIASES

    assert ALIASES == _ALIASES


def test_importing_the_gate_core_loads_no_engine_or_store():
    """The P2-2 core stays pure like the sizing core (Q199-1): importing it loads no engine, database or paper code."""
    import subprocess
    import sys

    code = ("import sys, sleeve_fund.portfolio, sleeve_fund.portfolio.gate, sleeve_fund.portfolio.book; "
            "print([m for m in sys.modules if m.startswith(('nautilus', 'sqlalchemy', 'sleeve_fund.store', "
            "'sleeve_fund.strategies', 'sleeve_fund.paper', 'sleeve_fund.research'))])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]"
