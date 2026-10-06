"""The interim open-risk limit on a perpetual (Independent Quant Advisor, 6 Oct; sleeve_fund.open_risk): paper refuses an
entry that would take the book's open risk over 5% of the book. A stopless position counts at notional x max(10%,
3 x daily ATR); a stopped one from the current mark to its stop. A single-strategy backtest doesn't gate on it; it
counts how often the limit would have bound."""

import pandas as pd
import pytest

from sleeve_fund import open_risk
from sleeve_fund.research.replay import replay
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.store import Store
from test_long_short import PERP, _record


def test_a_stopped_position_risks_from_the_mark_to_its_stop():
    assert open_risk.position_risk(2.0, 110.0, stop=100.0) == pytest.approx(20.0)  # long: from the mark, not entry
    assert open_risk.position_risk(-2.0, 100.0, stop=104.0) == pytest.approx(8.0)  # short: the price rising to it


def test_a_position_the_price_has_gone_through_the_stop_of_counts_as_stopless_never_0():
    """QA P1-S9: still open past its stop (a gap, its exit not filled yet), it risks what a stopless one does."""
    assert open_risk.position_risk(2.0, 110.0, stop=112.0, atr_pct=0.01) == pytest.approx(22.0)
    assert open_risk.position_risk(-2.0, 100.0, stop=95.0, atr_pct=0.05) == pytest.approx(30.0)
    assert open_risk.gapped(2.0, 110.0, 112.0) and open_risk.gapped(-2.0, 100.0, 95.0)
    assert not open_risk.gapped(2.0, 110.0, 100.0) and not open_risk.gapped(0.0, 110.0, 112.0)
    with pytest.raises(ValueError, match="daily ATR isn't known"):
        open_risk.position_risk(2.0, 110.0, stop=112.0)


@pytest.mark.parametrize("atr_pct, move", [(0.01, 0.10), (0.05, 0.15)])
def test_a_stopless_position_counts_at_the_larger_of_10_percent_and_3_daily_atrs(atr_pct, move):
    assert open_risk.position_risk(-0.5, 4_000.0, atr_pct=atr_pct) == pytest.approx(2_000 * move)
    with pytest.raises(ValueError, match="daily ATR isn't known"):
        open_risk.position_risk(0.5, 4_000.0)


def _days(n, width=4.0, start="2025-01-02"):
    return pd.DataFrame({"high": [100 + width / 2] * n, "low": [100 - width / 2] * n, "close": [100.0] * n},
                        index=pd.date_range(start, periods=n, freq="D", tz="UTC"))


def test_the_daily_atr_is_the_librarys_wilder_atr_14_over_the_last_42_days():
    assert open_risk.daily_atr_pct(_days(13)) is None
    assert open_risk.daily_atr_pct(_days(14)) == pytest.approx(0.04)
    days = _days(15)
    days.loc[days.index[14], "high"] = 116.0  # an 18-wide day: (4 x 13 + 18) / 14
    assert open_risk.daily_atr_pct(days) == pytest.approx((4 * 13 + 18) / 14 / 100)
    wide = pd.concat([_days(30, width=40.0), _days(42, start="2025-02-01")])
    assert open_risk.daily_atr_pct(wide) == pytest.approx(0.04)  # only the last 42 days count


def test_a_backtest_reads_the_same_daily_atr_paper_does(prices):
    """QA P1-S5: each day's value is daily_atr_pct over the whole days before it, the window paper reads."""
    lookup = open_risk.daily_atr_lookup(prices)
    daily = prices[["high", "low", "close"]].groupby((prices.index - pd.Timedelta(1, "ns")).floor("D")).agg(
        {"high": "max", "low": "min", "close": "last"})
    assert lookup
    for day in list(lookup)[:3] + list(lookup)[-3:]:
        before = daily[daily.index < pd.Timestamp(day, tz="UTC")]
        assert lookup[day] == pytest.approx(open_risk.daily_atr_pct(before))


def _minutes(days, width):
    now = pd.Timestamp.now(tz="UTC").floor("1D")
    idx = pd.date_range(now - pd.Timedelta(days=days), periods=days * 1440, freq="1min", tz="UTC")
    return pd.DataFrame({"open": 100.0, "high": 100 + width / 2, "low": 100 - width / 2, "close": 100.0, "volume": 1.0},
                        index=idx)


@pytest.mark.real_daily_atr
def test_paper_reads_the_daily_atr_from_the_history_store_and_doesnt_cache_a_miss(tmp_path, monkeypatch):
    """QA P1-S3: the real read, past the 10% floor (a 6% daily ATR counts at 18%); a store that can't give 14 days
    returns None, and the next entry asks again rather than reading the miss from the cache."""
    from sleeve_fund.history import HistoryStore

    monkeypatch.setattr(open_risk, "_ATR_CACHE", {})
    hs = HistoryStore(tmp_path / "hist")
    now = pd.Timestamp.now(tz="UTC").to_pydatetime()
    assert open_risk.history_atr_pct("BINANCE", "BTC/USDT", now, history=hs) is None
    assert open_risk._ATR_CACHE == {}
    hs.append("BINANCE", "BTC/USDT", _minutes(20, width=6.0), cursor="x")
    atr = open_risk.history_atr_pct("BINANCE", "BTC/USDT", now, history=hs)
    assert atr == pytest.approx(0.06)
    assert open_risk.position_risk(1.0, 100.0, atr_pct=atr) == pytest.approx(18.0)  # 3 ATRs, over the 10% floor


def test_another_strategy_past_its_stop_counts_as_stopless_and_is_named(monkeypatch):
    store = Store.in_memory()
    store.create_sleeve(name="other", strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params=PERP, risk_profile="conservative")
    store.record_equity("other", equity=10_000, cash=4_000, qty=0.1, price=60_000, benchmark=10_000)
    monkeypatch.setattr(open_risk, "_journal_stop", lambda store, s, qty: 61_000.0)  # a long's stop above the mark
    book, risk, through = open_risk.account_book(store, "pp", 10_000, lambda s: 0.01)
    assert book == 20_000 and risk == pytest.approx(600.0) and through == ["other"]


def test_the_limit_is_5_percent_of_the_book_counting_the_entry():
    assert open_risk.check_entry(20_000, 600.0, 400.0) is None
    assert "over 5% of the book" in open_risk.check_entry(20_000, 600.0, 400.01)


def _meta():
    return {"balances": ["10000.00 USD"],
            "sleeve": {"name": "pp", "strategy": "ping_pong", "instrument": "BTC/USD",
                       "bar_spec": "1-MINUTE-LAST-INTERNAL", "starting_balance": 10_000, "risk_profile": "conservative",
                       "params": {"rise": 0.01, "dip": 0.005, **PERP}, "maker_fee": "0.0002", "taker_fee": "0.0005",
                       "tick_seconds": 30}}


@pytest.mark.parametrize("other_qty, refused", [(0.0, False), (0.05, False), (0.15, True)])
def test_paper_refuses_an_entry_that_takes_the_accounts_open_risk_over_5_percent(tmp_path, other_qty, refused):
    """ping_pong at 1x opens about 2,000 notional (200 at risk). Another stopless strategy on the account holding
    0.15 at 60,000 (9,000 notional, 900 at risk) takes the book of 20,000 past 1,000, so the entry is refused."""
    store = Store.in_memory()
    store.create_sleeve(name="other", strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params=PERP, risk_profile="conservative")
    store.record_equity("other", equity=10_000, cash=10_000 - other_qty * 60_000, qty=other_qty, price=60_000,
                        benchmark=10_000)
    path = tmp_path / "pp.jsonl.gz"
    _record(path, _meta(), [(5, 0.0), (20, 0.015), (20, -0.012), (20, 0.015)])
    orders, _ = replay(path, with_fills=True, store=store)
    entries = [o for o in orders if o["intent"] == "entry"]
    notes = [e["message"] for e in store.events("pp", limit=500) if e["kind"] == "entry_refused_open_risk"]
    if refused:
        assert entries == [] and notes and "over 5% of the book (1,000.00)" in notes[0]
    else:
        assert entries and notes == []


@pytest.mark.real_daily_atr
def test_paper_refuses_a_stopless_entry_whose_risk_cant_be_measured(tmp_path, monkeypatch):
    monkeypatch.setattr(open_risk, "history_atr_pct", lambda venue, pair, now, history=None: None)
    store = Store.in_memory()
    path = tmp_path / "pp.jsonl.gz"
    _record(path, _meta(), [(5, 0.0), (20, 0.015), (20, -0.012), (20, 0.015)])
    orders, _ = replay(path, with_fills=True, store=store)
    assert [o for o in orders if o["intent"] == "entry"] == []
    assert any("can't be measured" in e["message"] for e in store.events("pp", limit=500))


def test_a_backtest_counts_the_entries_the_limit_would_refuse_and_trades_them_all(prices, instrument, monkeypatch):
    """At 2x a stopless ping_pong opens about 6,600 notional: 660 at risk or more, over 5% of its own 10,000."""
    gated = run_backtest("ping_pong", prices.iloc[:200], instrument, PERP, risk_profile="balanced", half_spread=0)
    monkeypatch.setattr(open_risk, "LIMIT", float("inf"))
    ungated = run_backtest("ping_pong", prices.iloc[:200], instrument, PERP, risk_profile="balanced", half_spread=0)
    assert gated.open_risk_binds > 0 and ungated.open_risk_binds == 0
    cols = ["side", "filled_qty", "avg_px", "ts_last"]
    assert gated.fills[cols].equals(ungated.fills[cols]) and gated.equity.equals(ungated.equity)


def test_the_setups_the_paper_mechanics_tests_lift_the_limit_for_are_refused_with_it_on(tmp_path):
    """The two liquidation tests marked no_open_risk_limit run a stopless perp above 1x in paper. With the limit on,
    such a strategy opens nothing: at 2x ping_pong's 6,600 notional counts at 660, over 5% of its 10,000 book."""
    from test_long_short import _meta as balanced_meta

    path = tmp_path / "pp.jsonl.gz"
    _record(path, balanced_meta(10_000, {"rise": 0.01, "dip": 0.005, **PERP}),
            [(5, 0.0), (20, 0.015), (20, -0.012), (20, 0.015)])
    store = Store.in_memory()
    orders, _ = replay(path, with_fills=True, store=store)
    assert [o for o in orders if o["intent"] == "entry"] == []
    assert any(e["kind"] == "entry_refused_open_risk" for e in store.events("ping-pong-test", limit=500))
