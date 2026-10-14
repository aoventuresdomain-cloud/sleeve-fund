"""F-P28-UI1 (Independent Quant Advisor, 14 Oct 10:05 UK): the Risk page judges a two-leg Strategy as one, with its
summed guard. One row with its legs nested; drawdown from the pair's high-water mark and today's loss from its day
start; cap use gross (both legs' notional, no hedge credit) on the row and each leg's own under it; a shock is an
instant gap whose loss is the legs' signed sum, a perp leg losing at most its margin and the liquidation fee, and a leg
liquidated in the gap is named as leaving the other unhedged. The book's scenario totals are unchanged."""

from __future__ import annotations

import pytest

from tests.test_two_legs_dashboard import AUTH, carry  # noqa: F401  (the fixture)


def _view(store):
    from sleeve_fund.dashboard import riskops
    from sleeve_fund.dashboard.app import FloatView
    from sleeve_fund.dashboard.book import book_view, daily, sleeve_extras
    from sleeve_fund.dashboard.metrics import sleeve_summary

    st = FloatView(store)
    summaries = [sleeve_extras(st, sleeve_summary(st, s), daily(st, s.name)) for s in st.sleeves()]
    frames = {x["sleeve"].name: daily(st, x["sleeve"].name) for x in summaries}
    book = book_view(st, summaries, frames)
    return riskops.risk_view(st, summaries, book), summaries


def _shock(view, row, move):
    from sleeve_fund.dashboard.riskops import SHOCKS

    return row["shocks"][SHOCKS.index(move)], view["scenarios"][SHOCKS.index(move)]


def test_one_row_for_the_pair_with_its_legs_nested(carry):  # noqa: F811
    _, store = carry
    view, _ = _view(store)
    names = [r["x"]["sleeve"].name for r in view["rows"]]
    assert names == ["carry", "solo"]
    pair = view["rows"][0]
    assert [leg["x"]["sleeve"].name for leg in pair["legs"]] == ["carry", "carry~2"]
    assert pair["x"]["equity"] == pytest.approx(10_001.0)
    # Gross: 0.05 x 61,000 on each leg over the pair's 10,001, against the Strategy's cap; each leg keeps its own.
    assert pair["cap_used"] == pytest.approx((2 * 3_050 / 10_001) / pair["legs"][0]["x"]["cap"])
    assert pair["legs"][1]["cap_used"] == pytest.approx(3_050 / 4_953 / pair["legs"][1]["x"]["cap"])


def test_a_hedged_pair_is_not_halted_by_a_rally_that_would_halt_its_short_leg_alone(carry):  # noqa: F811
    _, store = carry
    view, _ = _view(store)
    pair = view["rows"][0]
    short = pair["legs"][1]["shocks"]
    shock, scenario = _shock(view, pair, 0.50)
    assert short[-1]["breach"]  # the short leg on its own would read as breaching
    assert not shock["breach"] and "carry" not in scenario["breaches"] and "carry~2" not in scenario["breaches"]
    long_loss, short_loss = (leg["shocks"][-1]["loss"] for leg in pair["legs"])
    assert shock["loss"] == pytest.approx(long_loss + short_loss)  # the signed sum
    assert long_loss < 0 < short_loss


def test_a_leg_liquidated_in_the_gap_counts_its_margin_and_leaves_the_other_unhedged(carry, monkeypatch):  # noqa: F811
    from sleeve_fund.dashboard import riskops

    _, store = carry
    real = riskops.most_it_can_lose
    # The perp leg's margin and liquidation fee: 500, so a +20% rally (a 610 move on its 3,050 short) liquidates it.
    monkeypatch.setattr(riskops, "most_it_can_lose",
                        lambda x: 500.0 if x["sleeve"].name == "carry~2" else real(x))
    view, _ = _view(store)
    pair = view["rows"][0]
    shock, scenario = _shock(view, pair, 0.20)
    assert shock["unhedged"] == [2] and scenario["unhedged"] == ["carry"]
    assert shock["loss"] == pytest.approx(-610.0 + 500.0)  # the long leg's move in full, the perp's margin and fee
    assert not _shock(view, pair, 0.10)[0]["unhedged"]  # 305 is within its 500
    client, _ = carry
    page = client.get("/risk", auth=AUTH).text
    assert "carry: leg liquidated in this move, the other leg left unhedged" in page
    assert "+20%: carry would lose a leg to liquidation, the other leg left unhedged" in page


def test_the_pairs_drawdown_and_days_loss_are_the_summed_guards(carry):  # noqa: F811
    from sleeve_fund.store import utcnow

    _, store = carry
    assert store.set_pair_state("carry", expected=None, state="flat")
    store.mark_pair("carry", 10_500, utcnow().date())
    view, _ = _view(store)
    pair = view["rows"][0]
    assert pair["x"]["drawdown"] == pytest.approx(1 - 10_001 / 10_500)
    assert pair["x"]["day_ret"] == pytest.approx(10_001 / 10_500 - 1)


def test_the_books_scenario_totals_still_count_every_leg(carry):  # noqa: F811
    _, store = carry
    view, _ = _view(store)
    for i, sc in enumerate(view["scenarios"]):
        legs = [r for r in view["rows"] for r in (r.get("legs") or [r])]
        assert sc["loss"] == pytest.approx(sum(r["shocks"][i]["loss"] for r in legs))


def test_the_risk_page_shows_the_pair_once_with_its_legs_under_it(carry):  # noqa: F811
    client, _ = carry
    page = client.get("/risk", auth=AUTH).text
    limits = page.split('id="lim-h"')[1].split("</table>")[0]
    assert limits.count('<a href="/sleeves/carry">carry</a>') == 1
    assert "Two legs, flat: limits on both legs together" in limits
    assert limits.count('class="leg-row"') == 2
    assert '<a href="/sleeves/carry~2">Leg 2</a> <span class="faint">BTC/USDT perp</span>' in limits
    strats = page.split('id="strats-h"')[1].split("</section>")[0]
    assert strats.count('href="/sleeves/carry"') == 1 and "carry~2" not in strats


def test_the_risk_tables_summed_figures_are_the_gates_to_the_cent(carry):  # noqa: F811
    """X-D2-5 (HoQA): the pair's row on the Risk page reads the very figures its summed guard judges: both legs'
    equity, the summed high-water mark, today's summed start and the room to the halt line, to the cent."""
    from datetime import timedelta

    from sleeve_fund.paper.runtime import SleeveRuntime
    from sleeve_fund.store import utcnow

    client, store = carry
    clock = [utcnow() + timedelta(seconds=1)]
    rt = SleeveRuntime(store, "carry", now=lambda: clock[0])
    rt.paired = True  # as the Pair sets it
    rt.on_start(0.0005)
    # Leg 2 marks itself; the guard on leg 1 judges both legs summed (ruling E).
    for own, other in ((5_048.0, 4_953.0), (5_311.37, 4_942.21), (5_087.13, 4_900.66)):
        clock[0] += timedelta(seconds=1)
        store.record_equity("carry~2", equity=other, cash=4_998.0, qty=-0.05, price=61_000, benchmark=5_000,
                            ts=clock[0])
        rt.tick(equity=own, cash=1_998.0, qty=0.05, price=61_000.0, other_equity=other)
    summed, peak, day_open = 5_087.13 + 4_900.66, rt.peak, rt._day_open
    assert (round(peak, 2), round(day_open, 2)) == (10_253.58, 10_001.00)  # the gate's own figures
    max_dd = rt.profile.max_drawdown
    view, _ = _view(store)
    pair = view["rows"][0]
    assert round(pair["x"]["equity"], 2) == round(summed, 2)
    assert round(pair["x"]["peak"], 2) == round(peak, 2)
    assert round(pair["x"]["equity"] / (1 + pair["x"]["day_ret"]), 2) == round(day_open, 2)
    assert round(pair["headroom"], 2) == round(summed - peak * (1 - max_dd), 2)
    limits = client.get("/risk", auth=AUTH).text.split('id="lim-h"')[1].split("</table>")[0]
    assert f"{summed - peak * (1 - max_dd):,.2f}" in limits
