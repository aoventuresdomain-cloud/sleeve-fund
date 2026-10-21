"""RESEARCH-SPOT-FEES (PM 17:56 UK 21 Oct, via the HoE; the Advisor sets the rate): a research-only spot market,
"spot-lowfee", so the leaders can be compared on the perpetual and on spot. It trades the perp venue's stored prices
in a cash account at a low-fee spot schedule: no margin, no funding, long or flat only. The sheet names the fee as
research only; paper, Start and the Strategy form refuse the market. Expected values are worked out here."""
from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from o17_harness import (  # noqa: F401 - fixtures
    _no_swallowed_strategy_errors, _o17win, backtest, binance, hourly_bars, utc, win, write_rates)
from sleeve_fund import markets
from sleeve_fund.instruments import FeeSchedule

SPOT_LOW = {"market": "spot-lowfee"}
TAKER = 0.001  # the Advisor's taker rate for the research spot schedule (21 Oct)


def _spot(spans) -> dict:
    return {**win(*spans), "market": "spot-lowfee", "allow_short": False}


def test_the_market_is_spot_at_its_own_fees_with_no_terms():
    assert markets.market_of(SPOT_LOW) == markets.SPOT_LOWFEE
    assert not markets.is_perp(SPOT_LOW) and markets.terms(SPOT_LOW, "BINANCE", "BTC/USDT") is None
    venue = FeeSchedule(Decimal("0.0002"), Decimal("0.00055"))
    assert markets.fees_for(SPOT_LOW, venue, "BINANCE") == markets.SPOT_LOWFEE_FEES
    assert float(markets.SPOT_LOWFEE_FEES.taker) == TAKER
    assert markets.SPOT_LOWFEE not in markets.MARKETS  # never offered as a choice


def test_a_held_long_on_the_perp_venues_prices_pays_the_spot_fee_and_no_funding(binance):
    """Buy and hold across eight 8-hourly settlements of 0.3% each: a perp would pay about 2.4%; spot-lowfee pays
    none, holds the coin in a cash account (no margin), and its entry pays the research spot taker rate on its value."""
    from sleeve_fund.research.runner import run_backtest

    write_rates({t: 0.003 for t in pd.date_range("2025-10-03 00:00", "2025-10-06 00:00", freq="8h", tz="UTC")})
    perp = run_backtest("buy_and_hold", hourly_bars("2025-10-03 00:00", "2025-10-06 00:00"),
                        binance.instrument("BTC", "USDT"), {"market": "perp"}, starting_capital=10_000, bar_minutes=60,
                        half_spread=0)
    assert perp.funding, "set-up: the perp hold pays the stored funding"
    r = run_backtest("buy_and_hold", hourly_bars("2025-10-03 00:00", "2025-10-06 00:00"),
                     binance.instrument("BTC", "USDT"), SPOT_LOW, starting_capital=10_000, bar_minutes=60,
                     half_spread=0)
    assert not r.funding, r.funding
    assert len(r.fills) == 1
    fill = r.fills.iloc[0]
    qty, px = float(fill["filled_qty"]), float(fill["avg_px"])
    fee = float(str(fill["commissions"][0]).split()[0])
    assert fee == pytest.approx(qty * px * TAKER, abs=0.01)  # the venue books it to the cent
    last = float(hourly_bars("2025-10-03 00:00", "2025-10-06 00:00")["close"].iloc[-1])
    assert float(r.equity.iloc[-1]) == pytest.approx(10_000 - qty * px - fee + qty * last, abs=0.01)


def test_a_short_is_refused(binance):
    with pytest.raises(ValueError, match="short positions need a perpetual"):
        backtest(binance, hourly_bars("2025-10-03 00:00", "2025-10-04 00:00"),
                 {**_spot([("2025-10-03 01:00", "2025-10-03 20:00", -1)]), "allow_short": True})


@pytest.mark.parametrize("market", ["spot-lowfee", "perp-funding-stress"])
@pytest.mark.parametrize("venue", ["KRAKEN", "BINANCE"])
def test_paper_start_and_the_form_refuse_a_research_only_market(market, venue):
    """check_venue is the one gate paper's config (so Start) and the Strategy form both call."""
    with pytest.raises(ValueError, match="research only"):
        markets.check_venue({"market": market}, venue)


def test_a_paper_config_refuses_it():
    from sleeve_fund.paper.config import SleeveConfig

    with pytest.raises(ValueError, match="research only"):
        SleeveConfig(name="s", strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                     starting_balance=0.0, params={"market": "spot-lowfee"}, venue="KRAKEN")


def _spot_study(tmp_path_factory, base: str):
    """A trend study on the perp venue's prices, asked for on spot-lowfee; funding stored at 0.3% a settlement, so a
    perp run would pay plenty."""
    import json

    from sleeve_fund import funding
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.research import tearsheet
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.study import run_study
    from sleeve_fund.strategies.trend_filter import SPEC
    from sleeve_fund.venues import binance_contract, venue
    from test_binance import INFO

    mp = pytest.MonkeyPatch()
    d = tmp_path_factory.mktemp("spot")
    b = venue("binance")
    mp.setattr(b, "contract", lambda pair: binance_contract(pair, get_json=lambda url: INFO))
    mp.setattr(funding, "DEFAULT_ROOT", d / "funding")
    prices = synthetic_ohlcv(days=800, seed=4)
    times = pd.date_range(prices.index[0].floor("D") - pd.Timedelta("1D"), prices.index[-1] + pd.Timedelta("1D"),
                          freq="8h")
    p = funding._path("BINANCE", f"{base}/USDT")
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"rates": [[int(t.timestamp() * 1000), 0.003] for t in times]}))
    funding._cache.clear()
    ledger = IdeaLedger(d / "l.jsonl")
    r = run_study(SPEC, prices, b.instrument(base, "USDT"), dataset="syn", ledger=ledger, synthetic=True,
                  holdout_days=0, train_days=300, test_days=200,
                  default_params={"fast": 20, "slow": 50, "market": "spot-lowfee"})
    sheet = tearsheet.render(r, ledger)
    yield r, sheet
    mp.undo()
    funding._cache.clear()


@pytest.fixture(scope="module")
def spot_study(tmp_path_factory):
    yield from _spot_study(tmp_path_factory, "BTC")


@pytest.fixture(scope="module")
def spot_study_eth(tmp_path_factory):
    """ETH keeps no margin tiers: research perp runs assume BTC's (R-28), so the sheet's perp line must too (CR M1)."""
    yield from _spot_study(tmp_path_factory, "ETH")


def test_a_study_asked_for_on_spot_lowfee_trades_spot_and_holds_spot(spot_study):
    r, _ = spot_study
    assert r.full_period.params["market"] == "spot-lowfee"
    assert not r.full_period.funding  # no funding on any run
    assert r.hold_market == "spot-lowfee" and not r.hold_insufficient
    assert r.full_period.fills is not None and len(r.full_period.fills)  # set-up: it traded


def test_the_sheet_names_the_fee_as_research_only(spot_study):
    _, sheet = spot_study
    assert "0.10% taker on every other order" in sheet and "(research only: a low-fee spot schedule" in sheet
    assert "traded the perpetual" not in sheet and "no margin and no funding" in sheet
    assert "**Asset class:** spot" in sheet or "Asset class:** spot" in sheet


def test_the_sheet_carries_the_advisors_label_and_prices_caveat(spot_study):
    _, sheet = spot_study
    assert "Assumed spot fee 0.10% (research only); paper needs the account's own rate." in sheet
    assert "Spot priced on perpetual prices, so the gap between the two prices is ignored" in sheet


def test_the_sheet_shows_the_entry_tier_fee_and_the_perp_beside_it(spot_study):
    r, sheet = spot_study
    assert "Spot at our current entry-tier fee (0.40% maker, 0.80% taker)" in sheet
    tier = r.spot_entry_tier
    assert tier.total_return < r.cost_ladder[3].total_return  # 0.80% a side costs more than the 0.10% rung
    spot, perp = r.spot_vs_perp["spot"], r.spot_vs_perp["perp"]
    assert spot["funding_paid"] == 0 and perp["funding_paid"] > 0  # 0.3% a settlement stored: the perp pays it
    assert r.double_fees is not None  # set-up: the double-fee test ran; it runs at twice the spot fee (Advisor: 0.20%)
    assert (r.double_fees["maker"], r.double_fees["taker"]) == (Decimal("0.002"), Decimal("0.002"))
    assert "| The perpetual, with its funding |" in sheet and "| Spot at the assumed research fee, no funding |" in sheet


def test_the_hold_bar_is_the_price_held_paying_the_spot_fee_in_and_out(spot_study):
    """Each window's hold: bought at the close before it and sold at its last, each side paying the 0.10% fee plus
    half the spread, no funding: (1 - c)^2 * last / first - 1 over the window."""
    from sleeve_fund.venues import venue

    r, _ = spot_study
    assert "spot fee and half the spread once in and once out, with no funding" in r.hold_note
    c = 0.001 + venue("binance").assumed_half_spread
    fold = r.folds[0]
    from sleeve_fund.data import synthetic_ohlcv

    closes = synthetic_ohlcv(days=800, seed=4)["close"]
    utc_ = lambda t: pd.Timestamp(t).tz_localize("UTC") if pd.Timestamp(t).tzinfo is None else pd.Timestamp(t)  # noqa: E731
    before = closes[closes.index <= utc_(fold.train_end)].iloc[-1]  # the close before the window's first bar
    at_last = closes[closes.index <= utc_(fold.test_end)].iloc[-1]
    want = (1 - c) ** 2 * at_last / before - 1
    assert fold.benchmark_test["total_return"] == pytest.approx(want, rel=1e-6)


def test_nothing_on_spot_lowfee_has_a_liquidation_price(spot_study):
    r, _ = spot_study
    sigs = [d for d in r.full_period.decisions.values()]
    assert sigs and not any(d.get("intent") == "liquidation" for d in sigs)
    assert not any(d.get("signal", {}).get("liquidation_px") for d in sigs)


def test_the_same_trades_on_perp_and_spot_differ_only_by_fees_and_funding(binance):
    """Buy and hold on the perp venue's prices, at one fee schedule and with no funding stored: the perpetual at 1x and
    spot-lowfee buy the same quantity at the same time and end at the same value; only funding (none here) and the
    fee schedule can part them."""
    from sleeve_fund.research.runner import run_backtest

    write_rates({t: 0.0 for t in pd.date_range("2025-10-03 00:00", "2025-10-06 00:00", freq="8h", tz="UTC")})
    bars = hourly_bars("2025-10-03 00:00", "2025-10-06 00:00")
    fees = markets.SPOT_LOWFEE_FEES
    runs = [run_backtest("buy_and_hold", bars, binance.instrument("BTC", "USDT"), {"market": m},
                         starting_capital=10_000, bar_minutes=60, half_spread=0, fees=fees) for m in ("perp", "spot-lowfee")]
    (p_fill,), (s_fill,) = (r.fills.itertuples() for r in runs)
    assert (p_fill.ts_last, float(p_fill.filled_qty), float(p_fill.avg_px)) == \
        (s_fill.ts_last, float(s_fill.filled_qty), float(s_fill.avg_px))
    assert float(runs[0].equity.iloc[-1]) == pytest.approx(float(runs[1].equity.iloc[-1]), abs=0.01)
    assert sum(f["amount"] for f in runs[0].funding) == 0 and not runs[1].funding


def test_a_study_row_runs_perp_or_spot_by_its_market_setting_alone(tmp_path):
    """HoE: the per-study market, independent of the risk profile and of the params' own market."""
    from sleeve_fund.research.run import StudyRequest

    seen = []
    from sleeve_fund.research import study as study_mod

    def fake(spec, prices, instrument, **kw):
        seen.append(kw["default_params"]["market"])
        raise RuntimeError("stop here")

    base = dict(strategy="buy_and_hold", pair="ETH/USD", minutes=1440, train_days=30, test_days=30, holdout_days=30,
                params={"market": "perp"})
    from sleeve_fund.history import HistoryStore

    hist = HistoryStore(tmp_path / "hist")
    idx = pd.date_range("2026-01-01", periods=3 * 1440, freq="1min", tz="UTC")
    hist.append("KRAKEN", "ETH/USD", pd.DataFrame({"open": 2000.0, "high": 2000.0, "low": 2000.0, "close": 2000.0,
                                                   "volume": 5.0}, index=idx), cursor="x")
    mp = pytest.MonkeyPatch()
    mp.setattr(study_mod, "run_study", fake)
    try:
        from sleeve_fund.research.run import run_store_study

        for market in (None, "spot-lowfee"):
            with pytest.raises(RuntimeError, match="stop here"):
                run_store_study(StudyRequest(**base, market=market), ledger_path=tmp_path / "l.jsonl",
                                out_dir=tmp_path / "ts", history=hist)
    finally:
        mp.undo()
    assert seen == ["perp", "spot-lowfee"]
    from sleeve_fund.research.run import study_market

    assert study_market(StudyRequest(**base)) == {}
    with pytest.raises(ValueError, match="a study's market"):
        study_market(StudyRequest(**base, market="spot"))


def test_a_study_list_row_runs_perp_or_spot_by_its_market_column_alone(tmp_path, capfd):
    """HoE: the same list row runs perp, or spot-lowfee by its optional fifth | market | column, and nothing else."""
    from sleeve_fund.research import rescore

    text = (rescore.ROOT / "configs" / "studies" / "qr-rescore-list.md").read_text()
    row = "| 1 | qr01_trend_ema_long | QR-B1 trend EMA 21/55 | 4h |"
    assert row in text
    seen = []

    def run(req, store=None):
        seen.append((req.label, req.params, req.market))
        return tmp_path / "sheet.md"

    for market in ("", " perp |", " spot-lowfee |"):
        path = tmp_path / f"list{len(seen)}.md"
        path.write_text(text.replace(row, row + market))
        assert rescore.main([str(path), "--rows", "1"], run=run, journal=lambda: None) == 0
    assert [m for _, _, m in seen] == [None, "perp", "spot-lowfee"]
    assert seen[0][:2] == seen[1][:2] == seen[2][:2]  # same definition, same params: only the market differs

    path = tmp_path / "bad.md"
    path.write_text(text.replace(row, row + " spot |"))
    called = []
    assert rescore.main([str(path), "--rows", "1"], run=lambda *a, **k: called.append(1), journal=lambda: None) == 2
    assert called == [] and "a study's market is perp or spot-lowfee" in capfd.readouterr().out


def test_the_perp_line_on_eth_runs_on_eths_own_tiers_and_assumes_none(spot_study_eth):
    """CR M1: the perp comparison of a spot-lowfee study on ETH ran with no tiers and crashed. TEC-TIERS3-A
    (PAPER-TIERS-3): ETH's own table is kept now, so the perp line runs on it and the sheet assumes none (R-28 gone)."""
    r, sheet = spot_study_eth
    assert r.spot_vs_perp["perp"]["funding_paid"] > 0 and r.spot_vs_perp["spot"]["funding_paid"] == 0
    assert "margin_note" not in r.spot_vs_perp
    assert "Margin tiers" not in sheet


def test_the_entry_tier_line_reads_its_fees_from_the_schedule(spot_study):
    """CR F1: the line's words come from SPOT_ENTRY_TIER_FEES, not a literal."""
    from sleeve_fund.research.study import SPOT_ENTRY_TIER_FEES

    _, sheet = spot_study
    assert (f"Spot at our current entry-tier fee ({float(SPOT_ENTRY_TIER_FEES.maker):.2%} maker, "
            f"{float(SPOT_ENTRY_TIER_FEES.taker):.2%} taker)") in sheet
