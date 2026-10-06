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


def test_a_stopped_position_risks_from_the_mark_to_its_stop_and_nothing_once_the_stop_is_in_profit():
    assert open_risk.position_risk(2.0, 110.0, stop=100.0) == pytest.approx(20.0)  # long: from the mark, not entry
    assert open_risk.position_risk(-2.0, 100.0, stop=104.0) == pytest.approx(8.0)  # short: the price rising to it
    assert open_risk.position_risk(2.0, 110.0, stop=112.0) == 0.0  # a long's stop above the mark: in profit
    assert open_risk.position_risk(-2.0, 100.0, stop=95.0) == 0.0


@pytest.mark.parametrize("atr_pct, move", [(0.01, 0.10), (0.05, 0.15)])
def test_a_stopless_position_counts_at_the_larger_of_10_percent_and_3_daily_atrs(atr_pct, move):
    assert open_risk.position_risk(-0.5, 4_000.0, atr_pct=atr_pct) == pytest.approx(2_000 * move)
    with pytest.raises(ValueError, match="daily ATR isn't known"):
        open_risk.position_risk(0.5, 4_000.0)


def test_the_daily_atr_is_wilders_over_14_days():
    days = pd.DataFrame({"high": [102.0] * 20, "low": [98.0] * 20, "close": [100.0] * 20},
                        index=pd.date_range("2025-01-02", periods=20, freq="D", tz="UTC"))
    atr = open_risk.wilder_atr_pct(days)
    assert atr.iloc[:13].isna().all() and atr.iloc[13:].tolist() == pytest.approx([0.04] * 7)
    days.loc[days.index[14], "high"] = 116.0  # a 18-wide day: (4 x 13 + 18) / 14
    assert open_risk.wilder_atr_pct(days).iloc[14] == pytest.approx((4 * 13 + 18) / 14 / 100)


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
    """The tests marked no_open_risk_limit run a stopless perp above 1x in paper. With the limit on, such a strategy
    opens nothing: at 2x ping_pong's 6,600 notional counts at 660, over 5% of its 10,000 book."""
    from test_long_short import _meta as balanced_meta

    path = tmp_path / "pp.jsonl.gz"
    _record(path, balanced_meta(10_000, {"rise": 0.01, "dip": 0.005, **PERP}),
            [(5, 0.0), (20, 0.015), (20, -0.012), (20, 0.015)])
    store = Store.in_memory()
    orders, _ = replay(path, with_fills=True, store=store)
    assert [o for o in orders if o["intent"] == "entry"] == []
    assert any(e["kind"] == "entry_refused_open_risk" for e in store.events("ping-pong-test", limit=500))
