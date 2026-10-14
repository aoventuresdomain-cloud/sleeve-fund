"""P2-8 two legs in the engine (strategies.pair, research.two_leg_run): a carry's two legs run as two strategies in one
engine, each on its own account with its own fees, funding and journal, coupled by a Pair. An entry opens both legs
(the perp first), every order of the trade carries one parent_trade_id, an exit closes both, and a second leg that
won't fill is sent once more and then its first leg is unwound with a warning alert (Advisor 13 Oct 23:55 UK, C)."""

import os

import pandas as pd
import pytest

os.environ.setdefault("BACKTEST_ISOLATE", "0")

from nautilus_trader.model import OrderSide  # noqa: E402

from sleeve_fund import markets  # noqa: E402
from sleeve_fund.data import synthetic_ohlcv  # noqa: E402
from sleeve_fund.instruments import FeeSchedule  # noqa: E402
from sleeve_fund.research.two_leg_run import LegRun, run_two_legs  # noqa: E402
from sleeve_fund.strategies import REGISTRY  # noqa: E402
from sleeve_fund.strategies.base import LongFlatConfig, LongFlatStrategy  # noqa: E402
from sleeve_fund.strategies.pair import BROKEN, LegFollower  # noqa: E402
from sleeve_fund.venues import venue  # noqa: E402

SPOT = {"instrument": "BTC/USDT", "params": {}, "venue": "KRAKEN", "side": 1}
PERP = {"instrument": "BTC/USDT", "params": {"market": "perp"}, "venue": "KRAKEN", "side": -1}
BARS = synthetic_ohlcv(days=40, seed=2, vol=0.02, start_price=60_000)
INST = venue("kraken").instrument("BTC", "USDT")


class WindowConfig(LongFlatConfig):
    def __init__(self, *, windows=(), **kw) -> None:
        super().__init__(**kw)
        self.windows = [tuple(w) for w in windows]


class Window(LongFlatStrategy):
    """Long while open <= the decision bar's close < close, for each [open_ns, close_ns]; flat otherwise."""

    def __init__(self, config: WindowConfig) -> None:
        super().__init__(config)
        self.windows = config.windows

    def want_long(self, bar):
        return any(o <= bar.ts_event < c for o, c in self.windows)

    def explain(self, bar, target):
        return ("In the window" if target else "Out of the window"), {}


@pytest.fixture(autouse=True)
def _window(monkeypatch):
    monkeypatch.setitem(REGISTRY, "window", (Window, WindowConfig))


def _ns(day: int) -> int:
    return int(BARS.index[day].value)


def _carry(*spans, **kw):
    params = {"windows": [[_ns(o), _ns(c)] for o, c in spans]}
    return run_two_legs("window", LegRun(SPOT, INST, BARS), LegRun(PERP, INST), 10_000, params=params,
                        risk_profile="balanced", **kw)


def _orders(r, leg):
    return [o for o in getattr(r, leg).journal.orders_.values()]


def test_an_entry_opens_both_legs_perp_first_under_one_trade_id_and_an_exit_closes_both():
    r = _carry((2, 10))
    assert r.trades == 1 and r.shares == (5000, 5000)
    lead, follow = _orders(r, "lead"), _orders(r, "follow")
    assert [o["intent"] for o in lead] == ["entry", "exit"] and [o["intent"] for o in follow] == ["entry", "exit"]
    ids = {o["parent_trade_id"] for o in lead + follow}
    assert len(ids) == 1 and None not in ids
    assert [(o["side"], o["qty"]) for o in lead] == [("BUY", lead[0]["qty"]), ("SELL", lead[0]["qty"])]
    assert [o["side"] for o in follow] == ["SELL", "BUY"] and follow[0]["qty"] == lead[0]["qty"]  # 1:1, one price
    steps = [what for _, what, _ in r.pair_log]
    assert steps[:4] == ["entry", "sent", "sent", "hedged"] and steps[-1] == "flat"
    assert r.pair_log[1][2].startswith("follow")  # the perp, the less liquid leg, goes first
    # Both legs' entries fill on the bar after the decision, and both exits on one bar.
    lf, ff = r.lead.fills, r.follow.fills
    assert list(lf["ts_last"]) == list(ff["ts_last"])


def test_each_leg_pays_its_own_venue_fees_and_only_the_perp_leg_has_funding():
    r = _carry((2, 30))
    assert r.lead.funding == [] and r.follow.funding
    assert r.funding_paid == pytest.approx(sum(f["amount"] for f in r.follow.funding))
    spot_taker = float(INST.taker_fee)
    perp_taker = float(markets.fees_for(PERP["params"], FeeSchedule(INST.maker_fee, INST.taker_fee), "KRAKEN").taker)
    assert perp_taker != spot_taker
    for res, rate in ((r.lead, spot_taker), (r.follow, perp_taker)):
        f = res.fills
        notional = (f["filled_qty"].astype(float) * f["avg_px"].astype(float)).sum()
        assert res.fees_paid / notional == pytest.approx(rate, rel=1e-3)
    assert r.equity.iloc[-1] == pytest.approx(r.lead.equity.iloc[-1] + r.follow.equity.iloc[-1])
    assert r.equity.iloc[0] == pytest.approx(10_000)


def test_a_new_window_opens_a_new_trade_with_a_new_id():
    r = _carry((2, 6), (12, 16))
    assert r.trades == 2
    ids = [o["parent_trade_id"] for o in _orders(r, "lead")]
    assert len(set(ids)) == 2 and ids[0] == ids[1] and ids[2] == ids[3]


def test_a_second_leg_the_venue_refuses_is_sent_once_more_then_the_first_is_unwound(monkeypatch):
    """The spot leg (second out) is refused at the venue each time: one retry, then the perp leg already filled is
    unwound at market (leg_unwind) with a warning alert, and both legs end flat."""
    real = LongFlatStrategy.submit_order

    def refuse_spot_entries(self, order, *a, **k):
        coid = str(order.client_order_id)
        if not isinstance(self, LegFollower) and (self.decisions.get(coid) or {}).get("intent") == "entry":
            order = self.order_factory.market(instrument_id=order.instrument_id, order_side=OrderSide.SELL,
                                              quantity=order.quantity, reduce_only=True)  # refused: nothing to reduce
            self.decisions[str(order.client_order_id)] = self.decisions[coid]
        return real(self, order, *a, **k)

    monkeypatch.setattr(LongFlatStrategy, "submit_order", refuse_spot_entries)
    r = _carry((2, 10))
    steps = [what for _, what, _ in r.pair_log]
    assert steps.count("sent") == 3  # the perp, the spot, the spot once more
    assert BROKEN in steps and steps[-1] == "flat"
    follow = _orders(r, "follow")
    assert [o["intent"] for o in follow] == ["entry", "leg_unwind"]
    alerts = [e for e in r.follow.journal.events_ if e["kind"] == BROKEN]
    assert len(alerts) == 1 and alerts[0]["level"] == "warning"
    assert float(r.follow.fills["filled_qty"].astype(float).iloc[0]) == float(r.follow.fills["filled_qty"].astype(float).iloc[1])


def test_a_stop_resting_on_one_leg_closes_the_other_leg_with_it():
    """The first leg's own stop fills at the venue (no sell-all of its own): the perp leg is closed with it at once,
    and the pair is flat; the stop's side waits for the signal to reset, so the pair isn't reopened in the window."""
    params = {"windows": [[_ns(2), _ns(38)]], "stop_loss": 0.02}
    r = run_two_legs("window", LegRun(SPOT, INST, BARS), LegRun(PERP, INST), 10_000, params=params,
                     risk_profile="balanced")
    lead, follow = _orders(r, "lead"), _orders(r, "follow")
    stopped = [o for o in lead if o["intent"] == "stop_loss" and o["filled_qty"]]
    assert stopped, "the fixture's 2% stop should be hit inside the window"
    closed = next(o for o in follow if o["intent"] == "stop_loss")
    assert closed["parent_trade_id"] == stopped[0]["parent_trade_id"] and closed["side"] == "BUY"
    assert closed["filled_qty"] == pytest.approx(follow[0]["filled_qty"])
    assert r.pair_log[-1][1] == "flat"


def _rising(days=30, start=60_000.0, daily=0.03, jump: tuple[int, float] | None = None):
    """Daily bars climbing `daily` a day; `jump` (day, factor) moves every close from that day on by the factor."""
    idx = pd.date_range("2024-01-02", periods=days, freq="D", tz="UTC")
    close = pd.Series([start * (1 + daily) ** i * (jump[1] if jump and i >= jump[0] else 1) for i in range(days)],
                      index=idx)
    open_ = close.shift(1).fillna(start)
    return pd.DataFrame({"open": open_, "high": pd.concat([open_, close], axis=1).max(axis=1) * 1.001,
                         "low": pd.concat([open_, close], axis=1).min(axis=1) * 0.999, "close": close,
                         "volume": 1_000_000.0})


def test_a_liquidated_perp_leg_closes_the_spot_leg_at_once_with_an_alert():
    """The price gaps 60% up in one bar, through the short perp leg's liquidation price at 2x, past the 50% clamp in
    the same bar: the liquidation goes first, the spot leg is closed at once (leg_unwind) with a warning alert, and
    the pair ends flat."""
    bars = _rising(daily=0.0, jump=(6, 1.6))
    params = {"windows": [[int(bars.index[1].value), int(bars.index[-1].value)]]}
    r = run_two_legs("window", LegRun(SPOT, INST, bars), LegRun({**PERP, "leverage": 2}, INST), 10_000,
                     params=params, risk_profile="aggressive")
    follow, lead = _orders(r, "follow"), _orders(r, "lead")
    cut = [o for o in follow if o["intent"] in ("liquidation", "liquidation_cut", "risk_halt")]
    assert cut, [o["intent"] for o in follow]
    unwound = [o for o in lead if o["intent"] == "leg_unwind"]
    assert unwound and unwound[0]["parent_trade_id"] == cut[0]["parent_trade_id"]
    assert any(e["kind"] == BROKEN for e in r.lead.journal.events_)
    assert [o["intent"] for o in follow].count("entry") == 1  # not reopened while the signal holds (as after a stop)


def _halts(res):
    return [e for e in res.risk_events if "halt" in str(e).lower() or "pause" in str(e).lower()]


def test_the_daily_pause_judges_both_legs_summed_so_a_hedged_carry_isnt_paused_by_one_leg():
    """Ruling E: the price jumps 20% in a day. The short perp leg alone loses past the 3% daily-loss limit, but the
    spot leg gains as much, so the pair isn't paused and both legs stay open to the window's end. (Under the
    conservative profile a leg holds 20% of its balance: ruling B's clamp, at a third up for a 1x short, would close
    the pair before one leg alone could reach the 10% drawdown limit, so the daily pause is the guard shown.)"""
    bars = _rising(days=20, daily=0.0, jump=(8, 1.2))
    params = {"windows": [[int(bars.index[1].value), int(bars.index[-1].value) + 1]]}
    r = run_two_legs("window", LegRun(SPOT, INST, bars), LegRun(PERP, INST), 10_000, params=params,
                     risk_profile="conservative")
    follow = r.follow.equity
    assert (follow.shift(1) - follow).max() / follow.max() > 0.03  # alone, it would have paused
    assert _halts(r.lead) == [] and _halts(r.follow) == []
    assert [o["intent"] for o in _orders(r, "follow")] == ["entry"]  # never closed
    assert (r.equity.shift(1) - r.equity).max() / r.equity.max() < 0.03


def test_a_pair_whose_summed_book_draws_down_past_the_limit_halts_and_flattens_both_legs():
    """Two long legs (spot and perp) on a falling price: the summed book passes the 10% drawdown limit, and both
    legs halt and close on that mark."""
    bars = _rising(days=40, daily=-0.03)
    params = {"windows": [[int(bars.index[1].value), int(bars.index[-1].value) + 1]]}
    r = run_two_legs("window", LegRun(SPOT, INST, bars), LegRun({**PERP, "side": 1}, INST), 10_000, params=params,
                     risk_profile="conservative")
    assert _halts(r.lead) and _halts(r.follow)
    lead_close = next(o for o in _orders(r, "lead") if o["intent"] == "risk_halt")
    follow_close = next(o for o in _orders(r, "follow") if o["intent"] == "risk_halt")
    assert lead_close["ts"] == follow_close["ts"] and lead_close["parent_trade_id"] == follow_close["parent_trade_id"]
    assert r.pair_log[-1][1] == "flat"


def test_the_perp_leg_is_margined_at_its_own_leverage_not_the_risk_profiles():
    """Under the balanced profile (2x) the carry's perp leg holds its own 1x by default: its liquidation price is
    about twice the entry for a short, not one and a half times; at 2x it is about one and a half."""
    for lev, near in ((1, 2.0), (2, 1.5)):
        r = _carry((2, 10)) if lev == 1 else run_two_legs(
            "window", LegRun(SPOT, INST, BARS), LegRun({**PERP, "leverage": lev}, INST), 10_000,
            params={"windows": [[_ns(2), _ns(10)]]}, risk_profile="balanced")
        entry = next(o for o in _orders(r, "follow") if o["intent"] == "entry")
        sig = entry["signal"]
        assert sig["liquidation_px"] / sig["close"] == pytest.approx(near, rel=0.02), (lev, sig)


def test_a_hedge_that_drifts_past_10_percent_is_brought_back_to_the_ratio_once_a_bar():
    """Ruling D: a pair whose second leg trades its own price. That price climbs 2% a day while the first leg's holds
    still, so the short leg's notional grows past the ratio; at the first close it has drifted more than 10% it is cut
    back (a rebalance, never gated), at most once a bar, under the trade's id."""
    flat = _rising(days=14, daily=0.0)  # 1.02^13 = 1.29: short of where ruling B's clamp closes the pair (1.33)
    climbing = _rising(days=14, start=3_000.0, daily=0.02)
    eth = venue("kraken").instrument("ETH", "USDT")
    eth_perp = {"instrument": "ETH/USDT", "params": {"market": "perp"}, "venue": "KRAKEN", "side": -1}
    params = {"windows": [[int(flat.index[1].value), int(flat.index[-1].value) + 1]]}
    r = run_two_legs("window", LegRun(SPOT, INST, flat), LegRun(eth_perp, eth, climbing), 10_000, params=params,
                     risk_profile="conservative")
    follow = _orders(r, "follow")
    cuts = [o for o in follow if o["intent"] == "rebalance"]
    assert cuts and all(o["side"] == "BUY" for o in cuts)  # a short leg grown too big is bought back
    assert {o["parent_trade_id"] for o in follow} == {follow[0]["parent_trade_id"]}
    days = [pd.Timestamp(o["ts"]).floor("D") for o in cuts]
    assert len(days) == len(set(days))  # at most once a bar
    steps = [d for _, what, d in r.pair_log if what == "rehedge"]
    assert steps and all("drifted 1" in d or "drifted" in d for d in steps)
    assert all(float(d.split("drifted ")[1].split("%")[0]) > 10 for d in steps)


def test_the_perp_legs_leverage_never_exceeds_the_risk_profiles():
    """Advisor 05:35 UK: effective leverage is min(the leg's, 2x, the profile's): a 2x leg under the conservative
    profile (1x) is margined at 1x, its liquidation about twice the entry for a short."""
    r = run_two_legs("window", LegRun(SPOT, INST, BARS), LegRun({**PERP, "leverage": 2}, INST), 10_000,
                     params={"windows": [[_ns(2), _ns(10)]]}, risk_profile="conservative")
    sig = next(o for o in _orders(r, "follow") if o["intent"] == "entry")["signal"]
    assert sig["liquidation_px"] / sig["close"] == pytest.approx(2.0, rel=0.02)


class _Eager(LegFollower):
    """A follower that, against the rules, has a signal of its own."""

    def want_long(self, bar):
        return True

    def want_side(self, bar):
        return 1  # a long perp: its config allows no short


def test_the_follower_never_opens_on_its_own_signal(monkeypatch):
    """HoE pin: a second leg whose model would open is refused; only its Pair sends its orders."""
    import sleeve_fund.research.two_leg_run as tlr

    monkeypatch.setattr(tlr, "LegFollower", _Eager)
    asked = []
    entry = tlr.Pair.entry
    monkeypatch.setattr(tlr.Pair, "entry", lambda self, leg, *a, **kw: (asked.append(leg is self.follow),
                                                                         entry(self, leg, *a, **kw))[1])
    r = _carry((30, 31))  # the lead's window opens late: until then nothing but the follower's own signal
    assert True in asked  # the follower did ask
    follow = _orders(r, "follow")
    first_lead = min(o["ts"] for o in _orders(r, "lead"))
    assert all(o["ts"] >= first_lead for o in follow)
    assert {o["parent_trade_id"] for o in follow} == {_orders(r, "lead")[0]["parent_trade_id"]}


class _Gone:
    """A coordinator that never attached to its legs."""

    def __init__(self, *a, **kw) -> None:
        self.log, self.trades = [], 0


def test_with_no_coordinator_neither_leg_opens_anything(monkeypatch):
    """Advisor M4: a leg of a two-leg Strategy with no Pair opens nothing, the lead included; no order on either."""
    import sleeve_fund.research.two_leg_run as tlr

    monkeypatch.setattr(tlr, "Pair", _Gone)
    monkeypatch.setattr(tlr, "LegFollower", _Eager)
    r = _carry((2, 10))
    assert _orders(r, "lead") == [] and _orders(r, "follow") == []


def test_a_perp_leg_past_half_its_entry_distance_to_liquidation_closes_the_pair_within_the_bar():
    """Ruling B (amended 14 Oct 05:35 UK), the G-2LEG clamp cell: no venue allows a top-up, so once the short perp
    leg's price comes nearer its liquidation price than half the distance it was entered at, both legs are closed
    with ungated leg_unwind orders on that bar, under the trade's id, with an alert and fees on both legs. It is not a
    broken pair: the model may open again, freshly sized, under a new id."""
    from sleeve_fund.strategies.pair import CLAMP, CLAMPED

    bars = _rising(days=20, daily=0.03)
    params = {"windows": [[int(bars.index[1].value), int(bars.index[-1].value) + 1]]}
    r = run_two_legs("window", LegRun(SPOT, INST, bars), LegRun(PERP, INST), 10_000, params=params,
                     risk_profile="balanced")
    lead, follow = _orders(r, "lead"), _orders(r, "follow")
    entry = follow[0]
    liq, at = entry["signal"]["liquidation_px"], float(entry["avg_px"])
    entered = (liq - at) / at
    first = next(ts for ts, high in bars["high"].items() if (liq - high) / high < CLAMP * entered)
    cut_f = next(o for o in follow if o["intent"] == "leg_unwind")
    cut_l = next(o for o in lead if o["intent"] == "leg_unwind")
    assert cut_f["parent_trade_id"] == cut_l["parent_trade_id"] == entry["parent_trade_id"]
    assert pd.Timestamp(cut_f["ts"]) == pd.Timestamp(cut_l["ts"]) <= first + pd.Timedelta(days=1)
    assert cut_f["side"] == "BUY" and cut_l["side"] == "SELL"
    assert cut_f["filled_qty"] == pytest.approx(entry["filled_qty"]) and cut_l["filled_qty"] == pytest.approx(lead[0]["filled_qty"])
    assert any(e["kind"] == CLAMPED for e in r.lead.journal.events_)
    assert not any(e["kind"] == BROKEN for e in r.lead.journal.events_)
    for res in (r.lead, r.follow):  # every fill, the unwind's included, paid its leg's fee
        assert all(sum(float(str(m).split()[0]) for m in (c if isinstance(c, (list, tuple)) else [c])) > 0
                   for c in res.fills["commissions"])
    again = [o for o in follow if o["intent"] == "entry"][1:]
    assert again and again[0]["parent_trade_id"] != entry["parent_trade_id"]
