"""The candle-step test Strategy on the engine (pe1-plans/candle-step-design.md; Advisor CANDLE-STEP and MUSTs; PM 22:38
UK): its cycle as fills, through the same order path every Strategy uses. Red first: the plan is in
test_candle_step_plan.py; these pin what the engine does with it."""

from decimal import Decimal as D

import pytest

from sleeve_fund.research.runner import run_backtest
from sleeve_fund.strategies.candle_step import step_at
from test_backtest import _path

PARAMS = {"market": "perp", "allow_short": True, "size_share": 0.6, "stop_loss": 0.001}


def _from_step_0(prices, n, px=100.0):
    """Flat closes from the first daily bar that is step 0 of a cycle, with the bars before it at the same price, so
    nothing reaches the 0.1% stop and the step clock alone trades."""
    first = next(i for i, t in enumerate(prices.index) if step_at(t.to_pydatetime(), 1440)[1] == 0)
    return first, [px] * (first + n)


def _fills(res):
    fills = res.fills.sort_values("ts_last")
    return [(row["side"], D(str(row["filled_qty"])), res.decisions[o]["intent"]) for o, row in fills.iterrows()]


def test_one_cycle_buys_halves_flips_in_two_orders_halves_and_reverses(prices, instrument):
    first, closes = _from_step_0(prices, 7)
    res = run_backtest("candle_step", _path(prices, closes), instrument, PARAMS, half_spread=0)
    fills = _fills(res)
    q = fills[0][1]
    assert q == D(60)  # 0.6 of the 10,000 at 100, the step's own limit
    assert fills == [("BUY", q, "entry"), ("SELL", q / 2, "exit"), ("SELL", q / 2, "exit"), ("SELL", q, "entry"),
                     ("BUY", q / 2, "exit"), ("BUY", q / 2, "exit"), ("BUY", q, "entry")]


def test_a_flip_never_holds_more_than_one_side_at_once(prices, instrument):
    first, closes = _from_step_0(prices, 7)
    res = run_backtest("candle_step", _path(prices, closes), instrument, PARAMS, half_spread=0)
    # Never Q long and Q short at once (a double-size order reads ~1.2): the close fills first. Q is fixed at step 0, so
    # the fees paid since then lift it a hair above 0.6 of the smaller equity.
    assert res.exposure.abs().max() <= 0.6 * 1.01


def test_a_stop_inside_the_cycle_makes_the_rest_of_it_a_no_op(prices, instrument):
    first, closes = _from_step_0(prices, 12)
    closes[first + 1:first + 6] = [99.7] * 5  # through the 0.1% stop on the bar after the buy, then flat
    closes[first + 6:] = [99.7] * (len(closes) - first - 6)
    res = run_backtest("candle_step", _path(prices, closes), instrument, PARAMS, half_spread=0)
    intents = [i for _, _, i in _fills(res)]
    assert intents[:2] == ["entry", "stop_loss"]
    assert intents[2] == "entry"  # nothing until the next cycle's step 0, which buys again
    buys = res.fills.sort_values("ts_last")
    assert step_at(buys["ts_last"].iloc[2].to_pydatetime(), 1440)[1] == 0


@pytest.mark.parametrize("bad", [0, -0.1, 1.5])
def test_size_share_outside_its_range_is_refused(bad, prices, instrument):
    with pytest.raises(ValueError, match="size_share"):
        run_backtest("candle_step", _path(prices, [100.0] * 8), instrument, {**PARAMS, "size_share": bad})


def test_a_stop_after_a_reduce_sells_only_what_is_left(prices, instrument):
    first, closes = _from_step_0(prices, 6)  # one cycle: steps 0-5
    closes[first + 2:] = [99.7] * (len(closes) - first - 2)  # through the stop after the half sold, before the flip
    res = run_backtest("candle_step", _path(prices, closes), instrument, PARAMS, half_spread=0)
    fills = _fills(res)
    q = fills[0][1]
    # The resting stop shrank to Q/2 with the reduce (the venue keeps a reduce-only order within the position; paper's
    # _sync_watched_stop does the same): it never sells more than is held, so it never opens a short.
    assert fills[:3] == [("BUY", q, "entry"), ("SELL", q / 2, "exit"), ("SELL", q / 2, "stop_loss")]
    assert len(fills) == 3  # steps 2-5 are no-ops once the stop has ended the cycle
    stop = res.fills[[res.decisions[o]["intent"] == "stop_loss" for o in res.fills.index]]
    assert D(str(stop["quantity"].iloc[0])) == q / 2  # the order itself, not only its reduce-only fill


def test_past_the_run_window_it_closes_and_opens_nothing(prices, instrument):
    first, closes = _from_step_0(prices, 12)
    hours = (first + 3) * 24  # from the first bar it sees: the window ends on step 3, short Q
    res = run_backtest("candle_step", _path(prices, closes), instrument, {**PARAMS, "run_for_hours": hours},
                       half_spread=0)
    fills = _fills(res)
    q = fills[0][1]
    assert fills == [("BUY", q, "entry"), ("SELL", q / 2, "exit"), ("SELL", q / 2, "exit"), ("SELL", q, "entry"),
                     ("BUY", q, "exit")]


def test_after_a_top_up_the_stop_covers_the_whole_position(prices, instrument):
    """Advisor (HoE 14 Oct ~23:38 UK): Q carries into the next cycle and a long short of it is topped up; once the top-up
    fills, the stop covers the FULL position. Step 4's open is trimmed to Q/2 by the bar's volume cap (a thin bar), so
    the next step 0 tops up; then the price goes through the stop, which sells everything."""
    first, closes = _from_step_0(prices, 8)
    closes[first + 7:] = [99.7] * (len(closes) - first - 7)  # through the 0.1% stop on the bar after the top-up
    path = _path(prices, closes)
    path.iloc[first + 4, path.columns.get_loc("volume")] = 3_000  # step 4: 1% of it is 30, half of Q
    res = run_backtest("candle_step", path, instrument, {**PARAMS, "max_participation": 0.01}, half_spread=0)
    fills = _fills(res)
    q = fills[0][1]
    assert fills[6:] == [("BUY", q / 2, "entry"), ("BUY", q / 2, "entry"), ("SELL", q, "stop_loss")], fills
    assert res.exposure.iloc[-1] == 0  # nothing left naked


def test_a_top_up_goes_through_the_compliance_tests_and_is_refused_on_a_full_book():
    """Advisor: a top-up is new risk, so it is gated like any entry. Net per asset at 0.45 of the book trims every open
    of Q (0.6) to about 0.45; the next step 0 asks to top up to Q and the same test refuses it, naming itself."""
    import pandas as pd
    from nautilus_trader.model import Venue

    from sleeve_fund.instruments import spot_pair
    from sleeve_fund.research.portfolio_run import LegSpec, run_portfolio
    from sleeve_fund.risk import PortfolioProfile
    from test_portfolio_run import FEES, _daily

    idx = pd.date_range("2024-01-01 01:00", periods=24 * 40, freq="h", tz="UTC")
    h = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1e6}, index=idx)
    d = _daily(h)
    w, window = d.iloc[:20], d.iloc[20:]
    spec = LegSpec("cs", "candle_step", spot_pair("BTC", "USD", FEES, Venue("KRAKEN")), window,
                   h[h.index > w.index[-1]], w, params={k: v for k, v in PARAMS.items() if k != "stop_loss"})
    run = run_portfolio([spec], PortfolioProfile(gross=100, net_instrument=0.45, margin=100, open_risk=100,
                                                 drawdown=0.99, daily_loss=0.99))
    opens = [x for x in run.decisions if x["requested"] == D(60)]
    assert opens and all(x["outcome"] == "trimmed" and x["limit"] == "net_instrument" for x in opens)
    top_ups = [x for x in run.decisions if x["requested"] < D(30)]
    assert top_ups and all(x["outcome"] == "rejected" and x["approved"] == 0 and x["limit"] == "net_instrument"
                           for x in top_ups)
    assert all(x["side"] == 1 for x in top_ups)  # only ever a long carried into step 0


def _at_the_limit(**limits):
    """Candle-step alone on a 100,000 book, flat prices, no fees or spread, Q asked at 0.6 of the book against
    compliance limits of 0.5: every open is trimmed."""
    import pandas as pd
    from nautilus_trader.model import Venue

    from sleeve_fund.fees import FeeSchedule
    from sleeve_fund.instruments import spot_pair
    from sleeve_fund.research.portfolio_run import LegSpec, run_portfolio
    from sleeve_fund.risk import PortfolioProfile
    from test_portfolio_run import _daily

    idx = pd.date_range("2024-01-01 01:00", periods=24 * 30, freq="h", tz="UTC")
    h = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1e9}, index=idx)
    d = _daily(h)
    w, window = d.iloc[:20], d.iloc[20:]
    free = FeeSchedule(D(0), D(0))
    spec = LegSpec("cs", "candle_step", spot_pair("BTC", "USD", free, Venue("KRAKEN")), window,
                   h[h.index > w.index[-1]], w, params={k: v for k, v in PARAMS.items() if k != "stop_loss"},
                   capital=100_000.0, half_spread=0.0, fees=free)
    profile = {"gross": 100, "net_instrument": 0.5, "margin": 0.5, "open_risk": 100, "drawdown": 0.99,
               "daily_loss": 0.99, **limits}
    return run_portfolio([spec], PortfolioProfile(**profile))


def test_at_1x_net_and_margin_both_bind_and_the_trim_names_net():
    """Design (g): at 1x a position's margin is its notional, so net per asset and margin bind at the same 50,000 on a
    100,000 book; on the tie the trim names net, first in limits.LIMITS order. Margin alone binds at the same size."""
    first = _at_the_limit().decisions[0]
    assert (first["outcome"], first["approved"], first["limit"]) == ("trimmed", D(500), "net_instrument")
    alone = _at_the_limit(net_instrument=100).decisions[0]
    assert (alone["outcome"], alone["approved"], alone["limit"]) == ("trimmed", D(500), "margin")


def test_must_2_a_flip_at_the_size_limit_closes_in_full_and_ends_short_exactly_at_the_limit():
    """Advisor MUST 2: the flip's close is never gated or refused, so it sells all that is held; the open is then
    gated on a flat book and trimmed to exactly the limit, 0.5 of the book the gate read (100,000 less the simulated
    funding paid so far)."""
    run = _at_the_limit()
    fills = run.legs["cs"].fills.sort_values("ts_last")
    sides = [(r["side"], D(str(r["filled_qty"]))) for _, r in fills.iterrows()]
    # B-30 (TEC, HoE to approve): Q is the 500 sent after the trim, not the 600 asked, so "sell half" sells 250 and
    # the flip's close the other 250 (was 200 then 300: 40% of the trimmed entry, the PM's 15 Oct defect).
    assert sides[:4] == [("BUY", D(500)), ("SELL", D(250)), ("SELL", D(250)), ("SELL", sides[3][1])]
    flip = run.decisions[1]
    assert (flip["side"], flip["outcome"], flip["limit"]) == (-1, "trimmed", "net_instrument")
    assert "(now 0.00x)" in flip["reason"]  # gated after the close filled: the book was flat
    import pandas as pd

    book = D(str(run.book.asof(pd.Timestamp(flip["ts"]))))
    assert flip["approved"] * flip["price"] == book / 2 and sides[3][1] == flip["approved"]


def test_its_0_1pct_target_is_kept_below_costs_and_fires_when_touched(prices, instrument):
    """PM 22:38 UK "run as i specified" (design, PM ruling): the test Strategy's 0.1% take-profit is placed though it
    is below the round trip's costs (keep_sub_cost_target, QD's C8 flag, on in this Strategy only), and it fires when
    touched: the bar after the buy reaches +0.2%, so the target sells the whole position before step 1's reduce."""
    first, closes = _from_step_0(prices, 3)
    closes[first + 1:] = [100.2] * (len(closes) - first - 1)
    res = run_backtest("candle_step", _path(prices, closes), instrument, {**PARAMS, "take_profit": 0.001},
                       half_spread=0)
    fills = _fills(res)
    q = fills[0][1]
    assert fills[:2] == [("BUY", q, "entry"), ("SELL", q, "take_profit")], fills


def test_only_the_test_strategy_keeps_a_target_below_costs_by_default():
    """The flag is off by default for every Strategy (QD C8; its cells pin that another Strategy drops such a target
    with its reason code), and only the test Strategy's defaults turn it on."""
    from sleeve_fund.strategies import REGISTRY
    import inspect

    from sleeve_fund.strategies.base import LongFlatConfig

    assert inspect.signature(LongFlatConfig.__init__).parameters["keep_sub_cost_target"].default is False
    on = [name for name, (cls, _) in REGISTRY.items()
          if getattr(getattr(cls, "SPEC", None), "default_params", {}).get("keep_sub_cost_target")]
    from sleeve_fund.strategies.candle_step import SPEC

    assert SPEC.default_params["keep_sub_cost_target"] is True and on in ([], ["candle_step"])
