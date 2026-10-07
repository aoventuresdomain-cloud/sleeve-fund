"""Binance USD-M perpetuals as a paper and research venue (PM, 5 Oct 2026): public data only, no account."""

import json

import pandas as pd
import pytest

from sleeve_fund import funding, markets
from sleeve_fund.venues import binance_contract, binance_funding, binance_minutes, binance_ohlc, venue

INFO = {"symbols": [
    {"symbol": "BTCUSDT", "pair": "BTCUSDT", "contractType": "PERPETUAL", "status": "TRADING", "baseAsset": "BTC",
     "quoteAsset": "USDT", "pricePrecision": 2, "quantityPrecision": 3,
     "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                 {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                 {"filterType": "MIN_NOTIONAL", "notional": "100"}]},
    {"symbol": "BTCUSDT_260327", "contractType": "CURRENT_QUARTER", "status": "TRADING", "baseAsset": "BTC",
     "quoteAsset": "USDT", "filters": []},
    {"symbol": "OLDUSDT", "contractType": "PERPETUAL", "status": "SETTLING", "baseAsset": "OLD", "quoteAsset": "USDT",
     "filters": []},
]}


def _kline(t_ms, o, h, lo, c, v):
    return [t_ms, str(o), str(h), str(lo), str(c), str(v), t_ms + 59_999, "0", 10, "0", "0", "0"]


def test_the_venue_lists_perpetuals_under_its_own_symbol_with_its_fees_and_contract_limits():
    b = venue("binance")
    assert b.perpetual and b.symbol_of("BTC/USDT") == "BTCUSDT-PERP"
    assert (float(b.fees.maker), float(b.fees.taker)) == (0.0002, 0.0005)
    assert b.list_instruments(get_json=lambda url: INFO) == ["BTC/USDT"]  # quarterlies and settling ones left out
    assert binance_contract("BTC/USDT", get_json=lambda url: INFO) == {
        "price_precision": 1, "size_precision": 3, "min_quantity": 0.001, "min_notional": 100.0}
    with pytest.raises(ValueError, match="does not list"):
        binance_contract("ZZZ/USDT", get_json=lambda url: INFO)
    with pytest.raises(ValueError, match="not trading"):
        binance_contract("OLD/USDT", get_json=lambda url: INFO)


def test_a_backtest_instrument_is_the_perpetual_paper_receives(monkeypatch):
    b = venue("binance")
    monkeypatch.setattr(b, "contract", lambda pair: binance_contract(pair, get_json=lambda url: INFO))
    inst = b.instrument("BTC", "USDT")
    assert str(inst.id) == "BTCUSDT-PERP.BINANCE" and type(inst).__name__ == "CryptoPerpetual"
    assert (inst.price_precision, inst.size_precision, float(inst.min_notional)) == (1, 3, 100.0)
    assert float(inst.taker_fee) == 0.0005


def test_minute_pages_resume_on_the_last_minute_and_stop_at_the_forming_one():
    urls = []

    def page(url):
        urls.append(url)
        return [_kline(1_700_000_000_000 + i * 60_000, 100, 101, 99, 100.5, 2) for i in range(1500)]

    bars, cursor, done = binance_minutes("BTC/USDT", "1700000000000", get_json=page)
    assert len(bars) == 1500 and not done and "startTime=1700000000000" in urls[0] and "interval=1m" in urls[0]
    assert cursor == str(1_700_000_000_000 + 1499 * 60_000)  # the last minute again: it may have been forming
    assert bars.index[0] == pd.Timestamp(1_700_000_000_000, unit="ms", tz="UTC")
    short = lambda url: [_kline(1_700_000_000_000, 100, 101, 99, 100.5, 2)]  # noqa: E731
    assert binance_minutes("BTC/USDT", "1700000000000", get_json=short)[2]
    with pytest.raises(ValueError, match="Invalid symbol"):
        binance_ohlc("ZZZ/USDT", 15, get_json=lambda url: {"code": -1121, "msg": "Invalid symbol."})


def test_funding_is_kept_per_instrument_and_topped_up_from_the_last_rate(tmp_path):
    calls = []
    t0 = 1_700_000_000_000

    def loader(pair, start):
        calls.append(start)
        return [(t0 + i * 8 * 3_600_000, 0.0001 * (i + 1)) for i in range(3) if t0 + i * 8 * 3_600_000 >= start]

    s = funding.refresh("BINANCE", "BTC/USDT", root=tmp_path, loader=loader)
    assert list(s) == pytest.approx([0.0001, 0.0002, 0.0003]) and calls == [0]
    funding.refresh("BINANCE", "BTC/USDT", root=tmp_path, loader=loader)
    assert calls[-1] == t0 + 2 * 8 * 3_600_000 + 1  # only what is newer than the last kept
    at = pd.Timestamp(t0 + 8 * 3_600_000 + 400, unit="ms", tz="UTC")  # stamped a moment after the hour
    assert funding.rate_at(s, at) == pytest.approx(0.0002)
    assert funding.rate_at(s, at + pd.Timedelta(hours=4)) is None
    rows = [{"symbol": "BTCUSDT", "fundingTime": t0, "fundingRate": "-0.00012", "markPrice": "1"}]
    assert binance_funding("BTC/USDT", t0, get_json=lambda url: rows) == [(t0, -0.00012)]


def test_a_strategy_on_the_venue_must_trade_its_perpetual():
    from sleeve_fund.paper.config import SleeveConfig

    base = dict(name="x", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                starting_balance=1000, venue="BINANCE")
    with pytest.raises(ValueError, match="perpetuals only"):
        SleeveConfig(**base, params={"rise": 0.01, "dip": 0.005})
    cfg = SleeveConfig(**base, params={"rise": 0.01, "dip": 0.005, "market": "perp", "allow_short": True})
    assert cfg.instrument_id == "BTCUSDT-PERP.BINANCE" and cfg.bar_type.startswith("BTCUSDT-PERP.BINANCE-1-MINUTE")
    assert float(cfg.fees.taker) == 0.0005
    t = markets.terms(cfg.params, "BINANCE")
    assert t.funding_venue == "BINANCE" and t.label == "Binance USD-M perpetuals"
    with pytest.raises(ValueError, match="choose the perp market"):
        markets.terms({"market": "perp-venue-fees"}, "BINANCE")


def test_a_backtest_charges_the_rates_the_venue_settled(tmp_path, monkeypatch):
    """Funding at each 8-hour settlement is the venue's own rate, not the flat 0.01% baseline; a settlement the
    records lack is charged the baseline, and the run says so once."""
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.research.runner import run_backtest

    monkeypatch.setattr(funding, "DEFAULT_ROOT", tmp_path)
    b = venue("binance")
    monkeypatch.setattr(b, "contract", lambda pair: binance_contract(pair, get_json=lambda url: INFO))
    bars = synthetic_ohlcv(days=40, seed=2, vol=0.02, start_price=60_000)
    times = pd.date_range(bars.index[0].floor("D"), bars.index[-1], freq="8h")
    kept = [[int(t.timestamp() * 1000), 0.0003] for t in times[: len(times) // 2]]  # the second half is missing
    path = funding._path("BINANCE", "BTC/USDT")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"rates": kept}))
    inst = b.instrument("BTC", "USDT")
    r = run_backtest("ping_pong", bars, inst, {"rise": 0.01, "dip": 0.005, "market": "perp", "allow_short": True},
                     starting_capital=10_000, risk_profile="balanced")
    paid = [e["message"] for e in r.journal.events_ if e["kind"] == "funding"]
    assert any("(0.0300%)" in m for m in paid) and any("(0.0100%)" in m for m in paid)
    fallback = [e for e in r.journal.events_ if e["kind"] == "funding_fallback"]
    assert len(fallback) == 1 and "0.0100% baseline" in fallback[0]["message"]
    # A missing rate never credits: whichever side is held pays the baseline (Advisor, 6 Oct 2026; QA P1-O17).
    baseline = {ts for ts, b, held in r.funding_marks if b and held}
    assert baseline and any(not b for _, b, held in r.funding_marks if held)
    held = [f for f in r.journal.funding_ if f["ts"] in baseline]
    assert any(f["qty"] < 0 for f in held) and all(f["amount"] < 0 for f in held)  # a short pays it too
    assert any(f["amount"] > 0 for f in r.funding if f["ts"] not in baseline)  # a short does receive a real rate


@pytest.fixture
def offline_binance(tmp_path, monkeypatch):
    """Binance's contract limits and funding records without the network (CI runs where Binance is blocked)."""
    b = venue("binance")
    monkeypatch.setattr(b, "contract", lambda pair: binance_contract(pair, get_json=lambda url: INFO))
    monkeypatch.setattr(funding, "DEFAULT_ROOT", tmp_path / "funding")
    return b


def test_a_study_on_the_venue_trades_its_perpetual_long_only(tmp_path, offline_binance):
    """A perpetual venue has no spot, so a G1 study there trades the perpetual, and its tear sheet says so."""
    from test_research import _stored_minutes

    from sleeve_fund.history import HistoryStore
    from sleeve_fund.research.run import StudyRequest, run_store_study

    hist = HistoryStore(tmp_path / "hist")
    hist.append("BINANCE", "BTC/USDT", _stored_minutes(130), cursor="x")
    req = StudyRequest(strategy="buy_and_hold", pair="BTC/USDT", venue="binance", minutes=240,
                       train_days=60, test_days=30, holdout_days=30)
    sheet = run_store_study(req, ledger_path=tmp_path / "l.jsonl", out_dir=tmp_path / "ts", history=hist)
    text = sheet.read_text()
    assert sheet.name.startswith("buy_and_hold_binance-btcusdt-store-240m_")
    assert "lists perpetuals only, so every run traded the perpetual, long only" in text
    assert "exits: the signal only" in text and "Exits on top of the signal" not in text
    with pytest.raises(ValueError, match="no stored history for ETH/USDT on this venue"):
        run_store_study(StudyRequest(strategy="buy_and_hold", pair="ETH/USDT", venue="binance"), history=hist)


def test_a_strategy_keeps_its_venue_and_a_backtest_prune_takes_it_away():
    from sleeve_fund.paper.config import from_store
    from sleeve_fund.store import Store

    store = Store("sqlite://")
    store.create_sleeve(name="bn", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=1000, params={"market": "perp"}, venue="binance")
    store.create_sleeve(name="kr", strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=1000)
    assert store.sleeve("bn").venue == "BINANCE" and store.sleeve("kr").venue is None
    assert {s.name: s.venue for s in store.sleeves()} == {"bn": "BINANCE", "kr": None}
    assert from_store(store.sleeve("bn")).venue == "BINANCE" and from_store(store.sleeve("kr")).venue == "KRAKEN"


def test_a_perp_study_holds_its_benchmark_at_the_perp_cap_as_the_backtest_page_does(tmp_path, offline_binance):
    """On a perpetual the balanced profile puts 33% of equity up as margin at 2x, a notional of 66%. The G1
    benchmark is held at that, as the backtest page's is (min(cap, 1)), not at the 33% margin share (review
    round 13, E13-1)."""
    from sleeve_fund import risk
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.study import run_study
    from sleeve_fund.strategies.buy_and_hold import SPEC as HOLD

    cap = min(risk.position_cap(risk.profile("balanced"), {"market": markets.PERP}), 1.0)
    assert cap == pytest.approx(0.66)
    r = run_study(HOLD, synthetic_ohlcv(days=200, seed=4), offline_binance.instrument("BTC", "USDT"), dataset="syn",
                  ledger=IdeaLedger(tmp_path / "l.jsonl"), synthetic=True, holdout_days=0, train_days=60,
                  test_days=60, risk_profile="balanced")
    bench = r.full_period_benchmark.exposure
    assert abs(bench[bench > 0].iloc[0] - cap) < 0.0015  # sized at the cap on entry, within 0.15 points
    assert any("capped at 66% of capital in notional" in n and "held at the same exposure" in n for n in r.notes)
