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


def _rising(days=30, start=60_000.0, daily=0.03):
    idx = pd.date_range("2024-01-02", periods=days, freq="D", tz="UTC")
    close = pd.Series([start * (1 + daily) ** i for i in range(days)], index=idx)
    open_ = close.shift(1).fillna(start)
    return pd.DataFrame({"open": open_, "high": close * 1.001, "low": open_ * 0.999, "close": close,
                         "volume": 1_000_000.0})


def test_a_liquidated_perp_leg_closes_the_spot_leg_at_once_with_an_alert():
    """The short perp leg at 2x is cut before liquidation as the price climbs: the spot leg is closed at once
    (leg_unwind), with a warning alert, and the pair ends flat."""
    bars = _rising()
    params = {"windows": [[int(bars.index[1].value), int(bars.index[-1].value)]]}
    r = run_two_legs("window", LegRun(SPOT, INST, bars), LegRun({**PERP, "leverage": 2}, INST), 10_000,
                     params=params, risk_profile="aggressive")
    follow, lead = _orders(r, "follow"), _orders(r, "lead")
    cut = [o for o in follow if o["intent"] in ("liquidation", "liquidation_cut", "risk_halt")]
    assert cut, [o["intent"] for o in follow]
    unwound = [o for o in lead if o["intent"] == "leg_unwind"]
    assert unwound and unwound[0]["parent_trade_id"] == cut[0]["parent_trade_id"]
    assert any(e["kind"] == BROKEN for e in r.lead.journal.events_)
