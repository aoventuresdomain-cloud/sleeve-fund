"""Review round 9: instrument precision.

B9-1: venues list 8 lot decimals for XRP and ADA, which the engine keeps at 6. An 8-decimal buy was held
as 6, so a full exit left 1e-7 in the journal and reconcile halted the strategy after its first round trip.
B9-2: studies built every instrument at 2 price decimals, so a sub-dollar instrument traded on prices
rounded to the cent while the backtest page used 6.
"""

from decimal import Decimal

import pandas as pd
import pytest
from nautilus_trader.model import CurrencyPair, Money, Quantity

from sleeve_fund.instruments import lot_decimals, price_decimals
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.research.study import _halt_words, _halted_before
from sleeve_fund.venues import venue
from test_backtest import _path
from test_research import _stored_minutes


def _as_listed(base: str, price_precision: int = 6) -> CurrencyPair:
    """The pair as a venue adapter lists it in paper: 8 lot decimals whatever the currency's precision."""
    i = venue("KRAKEN").instrument(base, "USD", price_precision=price_precision)
    return CurrencyPair(
        instrument_id=i.id, raw_symbol=i.raw_symbol, base_currency=i.base_currency, quote_currency=i.quote_currency,
        price_precision=price_precision, size_precision=8, price_increment=i.price_increment,
        size_increment=Quantity(1e-8, 8), min_quantity=Quantity(1e-8, 8), min_notional=Money(1, i.quote_currency),
        margin_init=Decimal(0), margin_maint=Decimal(0), maker_fee=i.maker_fee, taker_fee=i.taker_fee,
        ts_event=0, ts_init=0,
    )


def test_sizes_never_carry_more_decimals_than_the_currency():
    assert venue("KRAKEN").instrument("XRP", "USD").size_precision == 6
    assert venue("KRAKEN").instrument("ADA", "USD").size_precision == 6
    assert venue("KRAKEN").instrument("BTC", "USD").size_precision == 8
    assert lot_decimals(_as_listed("XRP")) == 6
    assert lot_decimals(_as_listed("BTC", 1)) == 8


@pytest.mark.parametrize("base", ["XRP", "ADA"])
def test_a_full_exit_on_a_six_decimal_currency_leaves_nothing_and_never_halts(prices, base):
    # Up, then down through the slow average and up again: two entries and an exit in between.
    closes = [0.5] * 30 + [0.5 * 1.01**i for i in range(1, 40)] + [0.5 * 1.01**39 * 0.98**i for i in range(1, 40)]
    closes += [closes[-1] * 1.01**i for i in range(1, 40)]
    res = run_backtest("trend_filter", _path(prices, closes), _as_listed(base), params={"fast": 3, "slow": 8},
                       starting_capital=10_000, risk_profile="aggressive")
    sides = list(res.fills["side"])
    assert sides[:3] == ["BUY", "SELL", "BUY"]
    assert not [e for e in res.risk_events if e["kind"] == "reconcile_mismatch"]
    buys = res.fills[res.fills["side"] == "BUY"]["filled_qty"].map(Decimal)
    assert all(q == q.quantize(Decimal("1e-6")) for q in buys)
    assert sum(1 for e in res.journal.events_ if e["kind"] == "reconcile") > 0
    bought, sold = (res.fills[res.fills["side"] == s]["filled_qty"].map(Decimal).iloc[0] for s in ("BUY", "SELL"))
    assert bought == sold


def test_a_reconcile_halt_counts_as_a_halt_in_a_study():
    t = pd.Timestamp("2025-03-01", tz="UTC")
    events = [{"kind": "reconcile_mismatch", "ts": t, "message": "engine cash 1.00 vs journal 1.00; ..."}]
    assert _halted_before(events, t + pd.Timedelta(days=1))
    assert not _halted_before(events, t - pd.Timedelta(days=1))
    words = _halt_words(events, t - pd.Timedelta(days=1), t + pd.Timedelta(days=5))
    assert "reconcile mismatch" in words and "in the test window" in words


def test_a_store_study_prices_a_sub_dollar_instrument_as_the_backtest_page_does(tmp_path, monkeypatch):
    import sleeve_fund.research.study as study
    from sleeve_fund.history import HistoryStore
    from sleeve_fund.research.run import StudyRequest, run_store_study

    m = _stored_minutes(130)
    m[["open", "high", "low", "close"]] *= 0.4 / m["close"].median()
    hist = HistoryStore(tmp_path / "hist")
    hist.append("KRAKEN", "DOGE/USD", m, cursor="x")
    seen = {}

    def capture(spec, prices, instrument, **kw):
        seen["instrument"], seen["median"] = instrument, float(prices["close"].median())
        raise RuntimeError("captured")

    monkeypatch.setattr(study, "run_study", capture)
    with pytest.raises(RuntimeError, match="captured"):
        run_store_study(StudyRequest(strategy="buy_and_hold", pair="DOGE/USD", minutes=240, holdout_days=0,
                                     train_days=60, test_days=30), ledger_path=tmp_path / "l.jsonl",
                        out_dir=tmp_path / "ts", history=hist)
    inst = seen["instrument"]
    assert inst.price_precision == 6 == price_decimals(seen["median"])
    assert price_decimals(0.4) == 6 and price_decimals(2.5) == 4 and price_decimals(60_000) == 2
