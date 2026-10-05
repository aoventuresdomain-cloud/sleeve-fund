"""Venue profiles: one source of fees for every mode, and a new venue needs no engine change."""

import re
from decimal import Decimal
from pathlib import Path

import pytest

from sleeve_fund import venues
from sleeve_fund.data import synthetic_ohlcv
from sleeve_fund.instruments import FeeSchedule
from sleeve_fund.paper.config import SleeveConfig, load_sleeve

ROOT = Path(__file__).resolve().parent.parent


def test_every_mode_uses_the_venue_profile_fees():
    from sleeve_fund.dashboard import preview

    k = venues.venue("KRAKEN")
    inst = k.instrument("BTC", "USD")  # research
    paper = SleeveConfig(name="x", strategy="trend_filter", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                         starting_balance=1000)
    preview._history.clear()
    bt = preview.run("buy_and_hold", "BTC/USD", {}, fetch=lambda pair: synthetic_ohlcv(days=120, seed=1))
    rates = {
        "research": (inst.maker_fee, inst.taker_fee),
        "paper": (paper.fees.maker, paper.fees.taker),
        "backtest page": (Decimal(str(bt["fee_schedule"]["maker"])), Decimal(str(bt["fee_schedule"]["taker"]))),
    }
    for mode, (maker, taker) in rates.items():
        assert (maker, taker) == (k.fees.maker, k.fees.taker), mode
    assert "0.40% maker, 0.80% taker" in bt["fee_schedule"]["text"]


def test_a_sleeve_file_cannot_override_the_venue_fees(tmp_path):
    src = (ROOT / "configs" / "examples" / "btc_trend_daily.toml").read_text()
    path = tmp_path / "s.toml"
    path.write_text(src + "\n[fees]\nmaker = 0.001\ntaker = 0.002\n")
    with pytest.raises(ValueError, match="venue profile"):
        load_sleeve(path)


def test_unknown_venue_is_rejected():
    with pytest.raises(ValueError, match="unknown venue"):
        SleeveConfig(name="x", strategy="trend_filter", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                     starting_balance=1000, venue="NOWHERE")


@pytest.fixture
def test_venue():
    profile = venues.register(venues.VenueProfile(
        name="TESTX", label="Test venue", fees=FeeSchedule(maker=Decimal("0.001"), taker=Decimal("0.002")),
        fee_basis="test", daily_history=lambda pair: synthetic_ohlcv(days=200, seed=4, vol=0.03),
    ))
    yield profile
    venues.VENUES.pop("TESTX")


def test_a_second_venue_backtests_without_engine_changes(test_venue):
    from sleeve_fund.dashboard import preview

    preview._history.clear()
    res = preview.run("trend_filter", "ABC/USD", {"fast": 5, "slow": 20}, venue="TESTX", detail=True)
    assert res["fee_schedule"]["taker"] == 0.002 and res["trades"]["trades"] > 0
    from sleeve_fund.research.runner import run_backtest

    bt = run_backtest("trend_filter", test_venue.daily_history("ABC/USD"), test_venue.instrument("ABC", "USD"),
                      {"fast": 5, "slow": 20})
    notional = (bt.fills["filled_qty"].astype(float) * bt.fills["avg_px"].astype(float)).sum()
    assert bt.fees_paid == pytest.approx(notional * 0.002, rel=1e-4)  # the test venue's taker rate, not Kraken's
    cfg = SleeveConfig(name="x", strategy="trend_filter", instrument="ABC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                       starting_balance=1000, venue="testx")
    assert cfg.instrument_id == "ABC/USD.TESTX" and cfg.fees == test_venue.fees


def test_engine_code_names_no_venue():
    """Strategies, runtime, research and the backtest path read the sleeve's venue profile; only
    venues.py (and the venue's own data loaders) may name a venue."""
    paths = [*(ROOT / "sleeve_fund" / "strategies").glob("*.py"), *(ROOT / "sleeve_fund" / "research").glob("*.py"),
             *(ROOT / "sleeve_fund" / "paper").glob("*.py"), ROOT / "sleeve_fund" / "dashboard" / "preview.py",
             ROOT / "sleeve_fund" / "dashboard" / "charts.py", ROOT / "sleeve_fund" / "instruments.py",
             ROOT / "sleeve_fund" / "markets.py", ROOT / "sleeve_fund" / "funding.py"]
    offenders = [p.name for p in paths if re.search(r"kraken|binance", p.read_text(), re.I)]
    # safety.py lists venue credential prefixes so it can refuse any of them; that is the point of it.
    assert offenders == ["safety.py"]


def test_accounts_and_g2_name_no_venue_but_from_data():
    """Account creation and the G2 checklist take a venue from the registered profiles, never from their own
    code (the static page copy waits for the UI redesign)."""
    import inspect

    from sleeve_fund.store import Store
    from sleeve_fund.venues import VENUES

    names = "|".join([*VENUES, "BYBIT", "DERIBIT"])
    texts = {"gates.py": (ROOT / "sleeve_fund" / "dashboard" / "gates.py").read_text(),
             "store accounts": inspect.getsource(Store._ensure_paper_account) + inspect.getsource(Store.create_account)}
    # accounts.py may name the old paper note it replaces, nothing else.
    texts["accounts.py"] = "\n".join(line for line in (ROOT / "sleeve_fund" / "accounts.py").read_text().splitlines()
                                     if not line.startswith("PAPER_NOTES_BEFORE ="))
    assert [k for k, t in texts.items() if re.search(names, t, re.IGNORECASE)] == []
