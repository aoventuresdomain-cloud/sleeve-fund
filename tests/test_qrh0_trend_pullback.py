"""Family H round 0, H0 (research/quant-researcher/next-round/prereg-family-h-trend-pullback-31oct.md; Advisor cleared
31 Oct 05:56 UTC as family B try 15): the definition is the pre-registration's TOML, hash 2161c99b4cf29d08; its daily
SMA(200) reads only the last completed daily close at or before each 4h decision (the Advisor's check 4); row 3 of the
round's list counts as family B try 15; and a perp basket with SOL gets its Without SOL line end to end (CR F1 on
#341). Figures in fixtures are synthetic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.research import regime_split as rs
from sleeve_fund.research import rescore, rounds
from sleeve_fund.research.pinned import parse
from sleeve_fund.research.study import continuous_warmup, idea_of
from sleeve_fund.research.variants import PREREGISTERED, definition_path
from sleeve_fund.strategies.definitions import (
    block_value,
    check_definition,
    definition_hash,
    load_definition_file,
    to_params,
)
from sleeve_fund.strategies.indicators.classic import Sma
from sleeve_fund.strategies.timeframes import SlowerCandles

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "configs" / "studies" / "next-round-pinned-dates-list.md"
PERP = {"market": "perp", "allow_short": True}


def _h0() -> dict:
    return load_definition_file(definition_path(PREREGISTERED["QRH0_TREND_PULLBACK"]["definition"]))


def test_h0_is_its_pre_registrations_toml():
    d = _h0()
    assert definition_hash(d) == definition_hash(d, "BINANCE") == "2161c99b4cf29d08"
    checked = check_definition(d, bar_minutes=240)
    assert d["blocks"] == {"trend": {"kind": "sma", "period": 200, "timeframe": "1d"}, "rsi": {"kind": "rsi", "period": 2},
                           "sma5": {"kind": "sma", "period": 5}, "atr": {"kind": "atr", "period": 14}}
    assert d["long"]["entry"] == {"all": [{"left": "close", "op": ">", "right": "trend"},
                                          {"left": "rsi", "op": "<", "right": 10}]}
    assert d["short"]["entry"] == {"all": [{"left": "close", "op": "<", "right": "trend"},
                                           {"left": "rsi", "op": ">", "right": 90}]}
    assert d["long"]["exit"] == {"any": [{"left": "rsi", "op": ">", "right": 70},
                                         {"left": "close", "op": ">", "right": "sma5"}]}
    assert d["short"]["exit"] == {"any": [{"left": "rsi", "op": "<", "right": 30},
                                          {"left": "close", "op": "<", "right": "sma5"}]}
    for side, op in (("long", "sub"), ("short", "add")):
        assert d["exits"][side] == {"stop": {"level": {op: ["close", {"mul": [2.5, "atr"]}]}},
                                    "time_stop": {"bars": 10}}
        assert "take_profit" not in d["exits"][side]
    assert checked.warmup[1440] == 200
    params = {**to_params(d, "BINANCE"), **PERP}
    assert continuous_warmup("rules", params, 240) * 240 / 1440 == 201  # inside 2024, the train year
    # its own idea: neither qr02 leg's (already among family B's 13 stated tries) nor B0's, so never counted twice
    others = [load_definition_file(ROOT / "configs" / "definitions" / f"{n}.toml") for n in ("qr02_dip_long",
                                                                                         "qr02_dip_short")]
    others.append(load_definition_file(definition_path("qrb0_range_reversion_adx")))
    assert idea_of(params, None) not in {idea_of({**to_params(o, "BINANCE"), **PERP}, None) for o in others}


def _four_hour(n_days=320, seed=4):
    """4h candles stamped at their close, 00:00 included, as the engine's decision candles are."""
    rng = np.random.default_rng(seed)
    n = 6 * n_days
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = np.concatenate([[c[0]], c[:-1]])
    idx = pd.date_range("2024-01-01 04:00", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.004, "low": np.minimum(o, c) * 0.996, "close": c,
                         "volume": 1e3}, index=idx)


def _trend_block():
    """H0's own trend block (period and timeframe from the committed definition), fed as the strategy feeds it."""
    b = _h0()["blocks"]["trend"]
    block = Sma(b["period"])
    return block, SlowerCandles(1440, 240, blocks=[block])


def test_check_4_the_daily_sma200_reads_the_last_completed_daily_close_only():
    """Advisor's check 4: at each 4h decision the gate's SMA(200) is the mean of the last 200 completed daily closes,
    the latest being the day that closed at or before the decision, never the day still forming."""
    bars = _four_hour()
    ref = rs.daily(bars)["close"].rolling(200).mean()  # daily bars stamped at each day's close
    days = ref.index.as_unit("ns").asi8
    block, slow = _trend_block()
    seen = 0
    for ts, r in zip(bars.index.as_unit("ns").asi8, bars.itertuples()):
        slow.update(r.open, r.high, r.low, r.close, r.volume, int(ts))
        last = int(np.searchsorted(days, ts, side="right")) - 1  # the latest day closed at or before this decision
        if last >= 199:
            assert block_value(block, None) == pytest.approx(ref.iloc[last], rel=1e-12)  # what the rules read
            seen += 1
        else:
            assert block_value(block, None) is None  # no gate value, so no entry, before 200 completed days
    assert seen > 600
    # A decision at 12:00 reads the day that closed at 00:00, however wildly that day's own candles have moved since.
    noon = bars.index[bars.index.hour == 12][-5]
    values = []
    for wild in (False, True):
        part = bars.loc[:noon].copy()
        if wild:
            part.iloc[-3:, :4] *= 1.5  # the 04:00, 08:00 and 12:00 candles of the day still forming
        b, s = _trend_block()
        for ts, r in zip(part.index.as_unit("ns").asi8, part.itertuples()):
            s.update(r.open, r.high, r.low, r.close, r.volume, int(ts))
        assert s.last.end == noon.normalize().value
        values.append(block_value(b, None))
    assert values[0] == values[1] == pytest.approx(ref.loc[noon.normalize()], rel=1e-12)


def test_row_3_is_h0_on_the_sealed_dates_and_counts_as_family_b_try_15():
    text = LIST.read_text()
    p = parse(text)
    assert (3, "qrh0_trend_pullback", 240) in rescore.rows_of(text)
    req = rescore.request("qrh0_trend_pullback", 240, "binance", "BTC/USDT,ETH/USDT,SOL/USDT", p)
    assert tuple(req.pairs) == ("BTC/USDT", "ETH/USDT", "SOL/USDT") and req.params["market"] == "perp"
    assert req.params["allow_short"] is True
    assert (req.use_holdout, req.train_days, req.test_days, req.holdout_days) == (False, 366, 485, 183)
    req.validate()
    r = rounds.round_of(text, [n for n, _, _ in rescore.rows_of(text)])
    assert r.family_of(3) == "B" and r.families["B"] == (2, 3)
    # B0 is in the register under family B's ideas (its row 2), so H0 runs with one family B try used
    assert "family B try 15 (13 stated before the round, 1 in the trials register" in rounds.try_line(r, 3, 1)


def test_a_perp_basket_with_sol_shows_itself_without_it():
    """CR F1 on #341: end to end on perpetual symbols ("SOLUSDT-PERP"), the basket study re-runs itself without SOL
    and the sheet prints the line. The fixture's figures are synthetic."""
    sys.path.insert(0, str(Path(__file__).parent))
    from nautilus_trader.model import Venue
    from test_b38_basket_study import (
        DAYS,
        FEES,
        LOOSE,
        RESEARCH_END,
        WINDOWS,
        _daily,
        _hourly,
        _register,
    )
    from test_study_asset_list import DEFINITION

    from sleeve_fund.instruments import perpetual
    from sleeve_fund.research import basket as bk

    def asset(base, seed):
        h = _hourly(DAYS, seed, vol=0.007)
        return bk.BasketAsset(perpetual(base, "USDT", FEES, Venue("BINANCE"), f"{base}USDT-PERP"), _daily(h), h)

    import tempfile

    basket = bk.Basket("sal-trend", "rules", {**to_params(DEFINITION, "BINANCE"), **PERP},
                       (asset("BTC", 2), asset("SOL", 9)), "2026-10-21")
    assert basket.names == ["BTCUSDT-PERP", "SOLUSDT-PERP"]
    study = bk.basket_study(basket, WINDOWS, RESEARCH_END, register=_register(Path(tempfile.mkdtemp())),
                            dataset="syn", capital=30_000.0, profile=LOOSE, warm_days=40, ladder=False, filled=True)
    assert study.without_sol is not None and study.without_sol["assets"] == ["BTCUSDT-PERP"]
    line = next(x for x in bk.render(study).splitlines() if x.startswith("**Without SOL** (BTCUSDT-PERP;"))
    assert "diagnostic only, never a condition or a try" in line and "net " in line and "Sharpe " in line
