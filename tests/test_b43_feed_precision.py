"""B43-PRECISION (Advisor, 20 Oct): every feed the engine can be given carries its instrument's own price precision.
B-43 builds a BTC perp at the kept tick 0.1 (1 place), not the 2 places a feed or a history may carry; a feed at another
precision than its instrument's never fills (test_multi_asset_replay's fixture showed it). Each path is loaded here as
the engine gets it, and every price in it is checked against the instrument it is fed to:
  backtest, study tool, CLI, preview and lab: venues.instrument() from history decimals, bars by data.to_bars
  replay: the recording's own instrument (replay._instrument) and rows (replay._data), and its warm-up bars (_history)
  paper: the hub's definition (hub_client.instrument_from) and the hub's messages decoded (hub_client.Decoder)
The D3 load-time FAIL for a mismatch is separate (B43-PRECISION part 3 (a))."""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from sleeve_fund.venues import venue

REPLAYS = sorted((Path(__file__).parent / "data" / "replay").glob("*.jsonl.gz"))
# Closes at 2 places, as stored history and the replay fixtures keep them: finer than BTC's kept tick
CLOSES = [60_000.12, 60_010.37, 59_990.55, 60_005.01]


def _prices(data) -> list:
    out = []
    for d in data:
        for name in ("open", "high", "low", "close", "bid_price", "ask_price", "price"):
            if hasattr(d, name):
                out.append(getattr(d, name))
    return out


def _all_at(instrument, data) -> None:
    prices = _prices(data)
    assert prices, "set-up: a feed with prices"
    assert {p.precision for p in prices} == {instrument.price_precision}, (instrument.id, instrument.price_precision)


@pytest.mark.parametrize("base", ["BTC", "ETH", "SOL"])
def test_backtest_study_and_lab_bars_carry_the_instruments_precision(base):
    from sleeve_fund.data import bar_type_for, to_bars
    from sleeve_fund.instruments import history_price_decimals

    idx = pd.date_range("2026-10-20", periods=len(CLOSES), freq="1min", tz="UTC")
    df = pd.DataFrame({"open": CLOSES, "high": CLOSES, "low": CLOSES, "close": CLOSES, "volume": [1.0] * 4}, index=idx)
    inst = venue("BINANCE").instrument(base, "USDT", price_precision=history_price_decimals(df["close"]))
    if base == "BTC":
        assert inst.price_precision == 1  # set-up: B-43's kept 0.1, coarser than the history's 2 places
    _all_at(inst, to_bars(df, inst, bar_type_for(inst, 1)))


@pytest.mark.parametrize("path", REPLAYS, ids=lambda p: p.name)
def test_a_recordings_feed_and_warm_up_carry_its_recorded_instruments_precision(path):
    from sleeve_fund.fees import FeeSchedule
    from sleeve_fund.paper.recorder import read
    from sleeve_fund.research import replay

    h, rows = read(path)
    inst = replay._instrument(h, FeeSchedule(Decimal("0.0002"), Decimal("0.0005")))
    _all_at(inst, replay._data(inst, rows))
    from nautilus_trader.model import BarType

    from sleeve_fund.data import bar_type_for

    last = max(r["e"] for r in rows) + 1
    warm = replay._history(path, last)(inst, BarType.from_str(str(bar_type_for(inst, 1))), 10)
    _all_at(inst, warm)


@pytest.mark.parametrize("base", ["BTC", "ETH"])
def test_the_paper_feed_is_decoded_at_the_hubs_definition(base):
    """Paper's instrument is the hub's definition from the market-data venue, never venues.instrument(); the hub sends
    the venue's ticks as strings at that definition's precision, and the decoder keeps them so."""
    from nautilus_trader.model import AggressorSide, Price, Quantity, QuoteTick, TradeId, TradeTick

    from sleeve_fund.hub import protocol
    from sleeve_fund.paper.hub_client import Decoder, instrument_from

    sent = venue("BINANCE").instrument(base, "USDT")  # what the venue's adapter would define, here offline
    inst = instrument_from(type(sent).to_dict(sent))
    p = inst.price_precision
    q = QuoteTick(inst.id, Price(60_000.15, p), Price(60_000.25, p), Quantity(1, 3), Quantity(1, 3), 1, 1)
    t = TradeTick(inst.id, Price(60_000.2, p), Quantity(1, 3), AggressorSide.BUY, TradeId("1"), 2, 2)
    decode = Decoder()
    _all_at(inst, [decode(protocol.quote(q, 1), 1), decode(protocol.trade(t, 2), 2)])
