"""R-28 (PM, 21 Oct 2026 11:01 UK: "use the same margins, margins don't change per asset"; HoE scope): a research study
on a perp pair whose venue tiers aren't kept runs on BTC's kept table, and its sheet says so. Paper, the start check and
every other caller still refuse that pair (VS-1). Expected values are written out from the BTC source
(research/venue-data/paper-venue-risk-limit-BTCUSDT-2026-10-15.json), not read from the code under test."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from sleeve_fund import markets
from sleeve_fund.margin import Tier
from sleeve_fund.venues import venue

pytestmark = [pytest.mark.no_maintenance_tables]
WORDS = "Margin tiers: BTC's table assumed for ETH/USDT (PM, 21 Oct)"


def test_paper_and_the_start_check_still_refuse_an_eth_perp_whatever_its_settings_carry():
    from nautilus_trader.model import BarType

    from sleeve_fund.strategies.base import LongFlatConfig

    with pytest.raises(markets.NoMaintenanceTable, match="ETH/USDT"):
        markets.terms({"market": "perp"}, "BINANCE", "ETH/USDT")
    for params in ({"market": "perp"}, {"market": "perp", "assumed_tiers_pair": "BTC/USDT"}):
        with pytest.raises(markets.NoMaintenanceTable, match="ETH/USDT"):
            markets.check_kept_tiers([SimpleNamespace(params=params, venue="BINANCE", instrument="ETH/USDT")])
    inst = venue("BINANCE").instrument("ETH", "USDT")
    with pytest.raises(markets.NoMaintenanceTable):  # a paper config sets no assumed pair
        LongFlatConfig(instrument_id=inst.id, bar_type=BarType.from_str(f"{inst.id}-1-MINUTE-LAST-EXTERNAL"),
                       assumed_taker_fee=0.00055, market="perp")


def test_a_research_run_on_eth_takes_btcs_tiers_and_last_ceiling_and_says_so():
    assert markets.research_tiers_pair({"market": "perp"}, "BINANCE", "ETH/USDT") == "BTC/USDT"
    t = markets.terms({"market": "perp"}, "BINANCE", "ETH/USDT", assume_from="BTC/USDT")
    assert t.maintenance_margin[0] == Tier(300_000, 0.0033, 0)
    assert t.maintenance_margin[1] == Tier(2_000_000, 0.005, 510)
    assert len(t.maintenance_margin) == 35 and t.maintenance_margin[-1][1:] == (0.6, 242_382_510)
    assert t.max_notional == 1_200_000_000
    assert t.label.endswith(WORDS)


def test_btc_and_spot_need_no_assumption_and_btcs_terms_are_unchanged():
    assert markets.research_tiers_pair({"market": "perp"}, "BINANCE", "BTC/USDT") is None
    assert markets.research_tiers_pair({}, "BINANCE", "ETH/USDT") is None  # spot
    own = markets.terms({"market": "perp"}, "BINANCE", "BTC/USDT")
    assert markets.terms({"market": "perp"}, "BINANCE", "BTC/USDT", assume_from="BTC/USDT") == own
    assert "assumed" not in own.label


def test_a_study_on_an_eth_perp_runs_on_btcs_tiers_and_its_sheet_carries_the_line(tmp_path, monkeypatch):
    from sleeve_fund import funding
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.research import study, tearsheet
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.strategies.buy_and_hold import SPEC as HOLD

    monkeypatch.setattr(funding, "DEFAULT_ROOT", tmp_path / "funding")
    ledger = IdeaLedger(tmp_path / "l.jsonl")
    r = study.run_study(HOLD, synthetic_ohlcv(days=200, seed=4), venue("BINANCE").instrument("ETH", "USDT"),
                        dataset="syn", ledger=ledger, synthetic=True, holdout_days=0, train_days=60, test_days=60,
                        default_params={"market": "perp"})
    assert r.margin_note == WORDS
    assert f"> {WORDS}." in tearsheet.render(r, ledger)
