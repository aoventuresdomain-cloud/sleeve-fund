"""B-24 (Advisor B24-LEVERAGE, 20 Oct 2026): every gate decision carries the leverage figures of the position it
leaves, computed once at the order's expected fill and never re-derived by a page:

  margin_leverage       notional / the position's isolated margin (the Strategy's 1x/2x/3x cap; spot 1)
  share_of_fund         the Strategy's resulting position / the fund's equity
  gross_after           the fund's gross after the order / equity: T4 gross's own figure
  net_underlying_after  the fund's signed net in the asset after it / equity: T4 net's, spot and perp summed

No new limit: T4 is the fund's, the perp cap the Strategy's (risk.leverage_cap, checked before the order as today).
Figures here are worked by hand from the inputs, never read back from the code under test."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal as D

import pytest

from sleeve_fund.portfolio.compliance import SEEDED
from sleeve_fund.portfolio.limits import Book, Holding, Intent, decide
from sleeve_fund.research.portfolio import gate_intent

E = D(100_000)
PX = D(50_000)
STEP = D("0.001")


def _perp(qty, side=1, held=0, lev=3, underlying="BTC") -> Intent:
    """An order at 50,000 on a perp at `lev`x: margin 50,000 / lev a unit; risk 500 a unit (a 1% stop)."""
    return Intent(underlying, side, D(qty), PX, PX / lev, D(500), STEP, STEP, D(held), D(lev))


def test_a_flat_to_a_3x_perp_at_a_tenth_of_the_fund_journals_3x_and_a_tenth_and_t4_gross_says_the_same():
    """Advisor cell a: 0.2 BTC at 50,000 is 10,000, a tenth of 100,000, posted at 3x."""
    d = decide(Book(E), _perp("0.2"), SEEDED)
    assert d.outcome == "approved", d
    assert d.leverage == {"margin_leverage": D(3), "share_of_fund": D("0.1"), "gross_after": D("0.1"),
                          "net_underlying_after": D("0.1")}
    assert float(d.leverage["gross_after"]) == d.figures["gross"]["after"]  # T4's own figure, the same sum
    assert float(d.leverage["net_underlying_after"]) == d.figures["net_instrument"]["after"]


def test_t4_off_still_journals_the_figures():
    """The figures don't depend on a test being on: with T4 off there is no gross row, yet gross_after is there."""
    off = replace(SEEDED, gross=None, net_instrument=None, margin=None)
    d = decide(Book(E), _perp("0.2"), off)
    assert "gross" not in d.figures and d.leverage["gross_after"] == D("0.1")


def test_b_an_add_past_t4_gross_is_held_to_the_room_and_journals_the_combined_position():
    """Advisor cell b, as T4 acts today (it trims to the room, Advisor 2): the fund holds 140,000 of BTC, this
    Strategy 40,000 of it (0.8 BTC); an add of 0.4 (20,000) would take gross to 1.6x of a 1.5x limit, so 0.2 goes
    (10,000), leaving the Strategy 1.0 BTC = 50,000, half the fund, and the fund at 1.5x."""
    book = Book(E, (Holding("BTC", D(140_000), D(140_000) / 3, D(1_400)),))
    loose = replace(SEEDED, net_instrument=2.0, margin=0.8, open_risk=0.10, asset_open_risk=None,
                    group_open_risk=None)
    d = decide(book, _perp("0.4", held="0.8"), loose)
    assert (d.outcome, d.approved_qty, d.limit_hit) == ("trimmed", D("0.2"), "gross"), d
    assert d.leverage == {"margin_leverage": D(3), "share_of_fund": D("0.5"), "gross_after": D("1.5"),
                          "net_underlying_after": D("1.5")}
    under = decide(book, _perp("0.1", held="0.8"), loose)  # under the limit: the whole add, the combined figures
    assert (under.outcome, under.leverage["share_of_fund"], under.leverage["gross_after"]) == (
        "approved", D("0.45"), D("1.45"))


def test_e_a_mark_alone_over_the_limit_forces_nothing_and_the_next_add_is_refused():
    """Advisor cell e: marks took the fund to 1.6x with no order; nothing is sold by the gate (it only judges what
    raises a figure); the next add has no room and is refused, its figures the position as it stands."""
    book = Book(E, (Holding("BTC", D(160_000), D(160_000) / 3, D(1_600)),))
    loose = replace(SEEDED, net_instrument=2.0, margin=0.8, open_risk=0.10, asset_open_risk=None,
                    group_open_risk=None)
    d = decide(book, _perp("0.1", held="3.2"), loose)
    assert (d.outcome, d.approved_qty, d.limit_hit) == ("rejected", D(0), "gross"), d
    assert d.leverage["gross_after"] == D("1.6") and d.leverage["share_of_fund"] == D("1.6")


@pytest.mark.parametrize("intent", [{"reduce_only": True, "what": "entry"}, {"what": "exit"}, {"what": "stop_loss"},
                                    {"what": "risk_halt"}, {"what": "leg_unwind"}])
def test_f_a_reduce_or_exit_is_never_judged_however_far_over_the_fund_is(tmp_path, intent):
    """Advisor cells e and f: reduce-only orders and exits never reach the gate (HoQA F208-1), so no leverage figure
    can hold one back."""
    from sleeve_fund.paper.portfolio_gate import PaperGate
    from sleeve_fund.store import Store

    gate = PaperGate(Store(f"sqlite:///{tmp_path}/j.db"))
    assert gate("s1", intent, 1) is None


@pytest.mark.parametrize("equity", [D(0), D(-1)])
def test_h_a_fund_at_or_below_zero_has_no_figures_and_opens_nothing(equity):
    """Advisor cell h: n/a, every opening refused (the exits never reach the gate, above)."""
    d = decide(Book(equity), _perp("0.2"), SEEDED)
    assert d.outcome == "rejected" and d.leverage == {}


def test_i_spot_and_perp_on_one_asset_net_signed_and_spot_is_1x():
    """Advisor cell i: the fund holds 20,000 of BTC spot and is 30,000 short on the BTC perp: net -10,000. A spot buy
    of 0.2 (10,000) leaves net 0 and gross 60,000; spot posts its whole notional, so 1x."""
    book = Book(E, (Holding("BTC", D(20_000), D(20_000), D(200)), Holding("BTC", D(-30_000), D(10_000), D(300))))
    d = decide(book, Intent("BTC", 1, D("0.2"), PX, PX, D(500), STEP, STEP, D("0.4")), SEEDED)
    assert d.outcome == "approved", d
    assert d.leverage == {"margin_leverage": D(1), "share_of_fund": D("0.3"), "gross_after": D("0.6"),
                          "net_underlying_after": D(0)}


def test_c_d_a_flips_open_leg_is_judged_on_the_book_after_its_close():
    """Advisor cells c and d. The engine sends a flip as two orders and the open only once the close has filled
    (base.LongFlatStrategy._flip, on_order_filled), so the open is judged on a book with the long gone and nothing
    held. Net per asset 0.5x: a 0.6 short (30,000, 0.3x) goes whole; a 1.2 short (60,000) is held to the 0.5x room,
    1.0; the close itself is an exit and never judged (cell f)."""
    after_close = Book(E)
    d = decide(after_close, _perp("0.6", side=-1), SEEDED)
    assert d.outcome == "approved" and d.leverage["net_underlying_after"] == D("-0.3"), d
    d = decide(after_close, _perp("1.2", side=-1), replace(SEEDED, gross=3.0, margin=0.8, open_risk=0.10,
                                                           asset_open_risk=None, group_open_risk=None))
    assert (d.outcome, d.approved_qty, d.limit_hit) == ("trimmed", D("1.0"), "net_instrument"), d
    assert d.leverage["net_underlying_after"] == D("-0.5") and d.leverage["share_of_fund"] == D("0.5")


def test_the_engines_intent_carries_what_is_held_into_the_gates_figures():
    """gate_intent reads the engine's "held" (base._gate_intent: signed on a perp); an intent from before it holds 0."""
    base = {"side": 1, "qty": D("0.1"), "price": PX, "stop_frac": 0.01, "step": STEP, "min_qty": STEP,
            "instrument": "BTC/USDT", "leverage": 2, "taker": D(0)}
    assert gate_intent({**base, "held": D("-0.3")}).held == D("-0.3")
    assert gate_intent(base).held == 0


def test_g_margin_leverage_is_the_strategys_cap_by_construction_never_above_it():
    """Advisor cell g: a position is margined at risk.leverage_cap (the profile's, or a two-leg perp leg's own when
    lower, never above 2x), so no order can carry 3.5x on a 3x cap: a leg asking 3.5x is margined at 2x, and the
    figure journaled is that cap."""
    from sleeve_fund import risk

    p = replace(risk.profile("aggressive"), max_leverage=3.0)
    assert risk.leverage_cap(p, {"leg": {"leverage": 3.5}}) == 2.0
    assert risk.leverage_cap(p, {}) == 3.0
    intent = gate_intent({"side": 1, "qty": D("0.1"), "price": PX, "stop_frac": 0.01, "step": STEP, "min_qty": STEP,
                          "instrument": "BTC/USDT", "leverage": risk.leverage_cap(p, {}), "taker": D(0)})
    assert decide(Book(E), intent, SEEDED).leverage["margin_leverage"] == D(3)


def test_a_two_leg_entry_journals_each_legs_figures_with_the_fund_counting_both():
    """A spot buy of 0.2 BTC and a 2x perp short of 0.2 BTC (10,000 each): gross 0.2x, net 0 in BTC, each leg a tenth
    of the fund; the spot leg at 1x, the perp at 2x."""
    from sleeve_fund.portfolio.limits import decide_pair

    spot = Intent("BTC", 1, D("0.2"), PX, PX, D(500), STEP, STEP)
    perp = _perp("0.2", side=-1, lev=2)
    a, b = decide_pair(Book(E), spot, perp, SEEDED)
    assert (a.outcome, b.outcome) == ("approved", "approved"), (a, b)
    assert a.leverage == {"margin_leverage": D(1), "share_of_fund": D("0.1"), "gross_after": D("0.2"),
                          "net_underlying_after": D(0)}
    assert b.leverage == {"margin_leverage": D(2), "share_of_fund": D("0.1"), "gross_after": D("0.2"),
                          "net_underlying_after": D(0)}


def test_k_a_fill_marks_its_own_figures_at_the_fill_price_with_the_decisions_margin_leverage():
    """Part 2: the fund holds 140,000 of BTC; 0.2 BTC long decided at 50,000 (3x) with 0.8 held fills 0.15 so far at
    50,400 average. At the fill: gross (140,000 + 0.15 x 50,400 = 147,560) / 100,000 = 1.4756; net the same; the
    Strategy's 0.95 at 50,400 = 47,880, 0.4788 of the fund; margin leverage the decision's 3. A short adding to a
    short: net 140,000 - 7,560 = 132,440, share |-0.8 - 0.15| x 50,400."""
    from sleeve_fund.portfolio.limits import fill_mark

    got = fill_mark(E, D(140_000), D(140_000), D("0.8"), 1, D("0.15"), D(50_400), D(3))
    assert got == {"margin_leverage": D(3), "share_of_fund": D("0.4788"), "gross_after": D("1.4756"),
                   "net_underlying_after": D("1.4756")}
    short = fill_mark(E, D(140_000), D(140_000), D("-0.8"), -1, D("0.15"), D(50_400), D(2))
    assert short == {"margin_leverage": D(2), "share_of_fund": D("0.4788"), "gross_after": D("1.4756"),
                     "net_underlying_after": D("1.3244")}
    assert fill_mark(D(0), D(0), D(0), D(0), 1, D("0.2"), PX, D(3)) == {}


@pytest.mark.parametrize("gross, net, words", [
    ("1.5", "-0.5", None),  # landing exactly on T4 is within it, as the check's
    ("1.5001", "0.4", "gross 1.5001x over its 1.5x"),
    ("0.9", "-0.6", "net in its underlying 0.6000x over its 0.5x"),
    ("1.6", "0.7", "gross 1.6000x over its 1.5x and net in its underlying 0.7000x over its 0.5x"),
])
def test_j_a_fill_that_slips_the_fund_past_t4_is_alerted_in_words_and_refuses_nothing(gross, net, words):
    """Advisor cell j: slippage can land a fill past T4 (gross 1.5x, net 0.5x here); it is alerted, never undone."""
    from sleeve_fund.portfolio.limits import fill_breach

    t4 = replace(SEEDED, gross=1.5, net_instrument=0.5)
    got = fill_breach({"gross_after": D(gross), "net_underlying_after": D(net)}, t4)
    assert (got is None) if words is None else (words in got and "slippage past T4" in got), got
    assert fill_breach({}, t4) is None
    assert fill_breach({"gross_after": D(9), "net_underlying_after": D(9)}, replace(t4, gross=None,
                                                                                    net_instrument=None)) is None
