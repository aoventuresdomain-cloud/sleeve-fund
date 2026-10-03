import re
from pathlib import Path

import pytest
from nautilus_trader.common import Environment
from nautilus_trader.model import Bar, BarType, InstrumentId, Price, Quantity

from sleeve_fund.paper.config import SleeveConfig, load_sleeve
from sleeve_fund.paper.node import build_node
from sleeve_fund.paper.safety import PaperSafetyError, assert_keyless
from sleeve_fund.strategies import TrendFilter, TrendFilterConfig

ROOT = Path(__file__).resolve().parent.parent
PAPER_SRC = ROOT / "sleeve_fund" / "paper"
SLEEVES = sorted((ROOT / "configs" / "sleeves").glob("*.toml"))


def test_paper_code_cannot_build_a_real_execution_client():
    # The only execution client allowed in paper is the local sandbox.
    for path in PAPER_SRC.glob("*.py"):
        src = path.read_text()
        assert "ExecutionClientFactory" not in src.replace("SandboxExecutionClientFactory", ""), path
        assert not re.search(r"\.add_exec_client\(", src), path
        assert "api_key=" not in src and "api_secret=" not in src, path


@pytest.mark.parametrize("var", ["KRAKEN_SPOT_API_KEY", "KRAKEN_SPOT_API_SECRET", "binance_api_key"])
def test_refuses_to_start_with_credentials(var):
    with pytest.raises(PaperSafetyError):
        assert_keyless({var: "x"})
    assert_keyless({var: ""})  # empty is fine
    assert_keyless({"PATH": "/usr/bin"})


@pytest.mark.parametrize("path", SLEEVES, ids=lambda p: p.stem)
def test_shipped_sleeves_build_in_sandbox(path, monkeypatch):
    for k in list(__import__("os").environ):
        if k.upper().startswith("KRAKEN_"):
            monkeypatch.delenv(k)
    node = build_node(load_sleeve(path), log_level="ERROR", asset_fetch=dict)
    try:
        assert node.environment == Environment.SANDBOX
    finally:
        node.dispose()


def _sleeve(**over):
    base = dict(
        name="t", strategy="trend_filter", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
        starting_balance=1000.0,
    )
    return SleeveConfig(**{**base, **over})


@pytest.mark.parametrize(
    "over",
    [{"strategy": "nope"}, {"bar_spec": "1-TICK-LAST-INTERNAL"}, {"instrument": "BTCUSD"}, {"instrument": "SUI/"},
     {"instrument": "SUI/USD; rm"},
     {"starting_balance": 0}, {"max_notional": -1}, {"warmup_bars": -1}],
)
def test_bad_sleeve_config_rejected(over):
    with pytest.raises(ValueError):
        _sleeve(**over)


@pytest.mark.parametrize("pair,expected", [("SUI/USD", "SUI/USD"), (" xrp/gbp ", "XRP/GBP"), ("ETH/USDT", "ETH/USDT")])
def test_any_kraken_pair_accepted(pair, expected):
    cfg = _sleeve(instrument=pair)
    assert cfg.instrument == expected and cfg.instrument_id == f"{expected}.KRAKEN"


def test_typo_in_strategy_params_rejected():
    with pytest.raises(TypeError):
        TrendFilterConfig(
            instrument_id=InstrumentId.from_str("BTC/USD.KRAKEN"),
            bar_type=BarType.from_str("BTC/USD.KRAKEN-1-MINUTE-LAST-INTERNAL"),
            fsat=5,
        )


def test_warmup_and_live_bars_never_double_count():
    bt = BarType.from_str("BTC/USD.KRAKEN-1-MINUTE-LAST-INTERNAL")
    cfg = TrendFilterConfig(instrument_id=InstrumentId.from_str("BTC/USD.KRAKEN"), bar_type=bt, fast=2, slow=3)
    s = TrendFilter(cfg)

    def bar(i, px):
        p = Price(px, 2)
        return Bar(bt, p, p, p, p, Quantity(1, 8), i * 60_000_000_000, i * 60_000_000_000)

    history = [bar(i, 100 + i) for i in range(1, 4)]
    for b in history:
        assert s._accept(b)
    # The same bar arriving again (overlap between warm-up and live) is ignored.
    assert not s._accept(history[-1])
    assert s.slow.value == pytest.approx(102.0)  # (101+102+103)/3; 102.67 if double-counted
    assert s._accept(bar(4, 110))
    assert s.slow.value == pytest.approx(105.0)


def test_kraken_asset_codes_use_the_venue_names():
    from sleeve_fund.paper.node import kraken_asset_codes

    pairs = {"result": {
        "XXBTZUSD": {"wsname": "XBT/USD", "base": "XXBT", "quote": "ZUSD"},
        "SUIUSD": {"wsname": "SUI/USD", "base": "SUI", "quote": "ZUSD"},
        "ETHUSDT": {"wsname": "ETH/USDT", "base": "XETH", "quote": "USDT"},
    }}
    assert kraken_asset_codes("BTC/USD", fetch=lambda: pairs) == ("XXBT", "ZUSD")
    assert kraken_asset_codes("SUI/USD", fetch=lambda: pairs) == ("SUI", "ZUSD")
    assert kraken_asset_codes("ETH/USDT", fetch=lambda: pairs) == ("XETH", "USDT")
    assert kraken_asset_codes("ABC/GBP", fetch=lambda: pairs) == ("ABC", "GBP")  # unlisted: plain codes

    def down():
        raise OSError("no network")

    assert kraken_asset_codes("SUI/USD", fetch=down) == ("SUI", "USD")
