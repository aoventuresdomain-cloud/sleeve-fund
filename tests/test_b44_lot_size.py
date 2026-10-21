"""B-44 (HoE 20 Oct; Advisor B36b-TICK-SOURCE: "the lot size (qtyStep, minimum quantity) comes from the same listing"):
the paper venue's perpetual is built with the quantity step and minimum order quantity its own instrument listing gives,
the listing B-43 reads the price tick from. Sizing rounds down to that step and refuses below that minimum. Venue
answers are the recorded listing (tests/data/fee_live/instruments-info-BTCUSDT.json), edited where a cell needs it."""

import json
from decimal import Decimal as D
from pathlib import Path

import pytest

from sleeve_fund.portfolio.limits import Book, Intent, decide
from sleeve_fund.risk import PORTFOLIO
from sleeve_fund.strategies.base import LongFlatStrategy
from sleeve_fund.venues import paper_venue_lot, venue

RECORDED = json.loads((Path(__file__).parent / "data" / "fee_live" / "instruments-info-BTCUSDT.json").read_text())
PAPER = venue("BINANCE")
CONTRACT = {"price_precision": 1, "size_precision": 3, "min_quantity": 0.001, "min_notional": 5.0}  # the data venue's


def listing(step=None, least=None, drop=False):
    answer = json.loads(json.dumps(RECORDED))
    f = answer["result"]["list"][0]["lotSizeFilter"]
    if drop:
        f.pop("qtyStep")
    if step is not None:
        f["qtyStep"] = step
    if least is not None:
        f["minOrderQty"] = least
    return lambda url: answer


class Sizer:
    """The strategy's own lot readers, on a given instrument."""

    _lot = LongFlatStrategy._lot
    _min_qty = LongFlatStrategy._min_qty

    def __init__(self, instrument):
        self.instrument = instrument


@pytest.fixture
def built(monkeypatch):
    monkeypatch.setattr(PAPER, "contract", lambda pair: dict(CONTRACT))

    def build(live_lot):
        monkeypatch.setattr(PAPER, "live_lot", live_lot)
        return PAPER.instrument("BTC", "USDT")
    return build


def test_the_lot_size_comes_from_the_recorded_listing():
    assert paper_venue_lot("BTC/USDT", get_json=listing()) == (D("0.001"), D("0.001"))
    assert paper_venue_lot("BTC/USDT", get_json=listing("0.010", "0.020")) == (D("0.01"), D("0.02"))


@pytest.mark.parametrize("bad", [dict(drop=True), dict(step="0"), dict(step="0.001", least="0.0015")])
def test_a_listing_with_no_usable_lot_size_is_refused(bad):
    with pytest.raises(ValueError):
        paper_venue_lot("BTC/USDT", get_json=listing(**bad))


@pytest.mark.parametrize("step,least", [("0.005", "0.005"), ("0.002", "0.002"), ("0.25", "0.5")])
def test_qd_a_step_that_isnt_a_power_of_ten_is_refused_so_the_kept_lot_stands(built, step, least):
    # QD MUST: the engine rounds with quantize, which keeps only the step's exponent: 0.007 stays 0.007 on a 0.005 step
    with pytest.raises(ValueError, match="power of ten"):
        paper_venue_lot("BTC/USDT", get_json=listing(step, least))
    inst = built(lambda pair: paper_venue_lot(pair, get_json=listing(step, least)))
    assert inst.size_increment.as_decimal() == D("0.001") and inst.min_quantity.as_decimal() == D("0.001")


def test_the_instrument_carries_the_listings_step_and_minimum_not_the_data_venues(built):
    inst = built(lambda pair: paper_venue_lot(pair, get_json=listing("0.01", "0.02")))
    assert inst.size_increment.as_decimal() == D("0.01") and inst.min_quantity.as_decimal() == D("0.02")
    s = Sizer(inst)
    assert s._lot() == D("0.01") and s._min_qty() == D("0.02")


def test_sizing_rounds_down_to_the_step_and_refuses_below_the_minimum(built):
    s = Sizer(built(lambda pair: paper_venue_lot(pair, get_json=listing("0.01", "0.02"))))

    def intent(qty):
        return Intent("BTC", 1, D(qty), D("60000"), D("60000"), D("600"), s._lot(), s._min_qty())

    ok = decide(Book(D("100000")), intent("0.0379"), PORTFOLIO)
    assert ok.approved_qty == D("0.03")
    small = decide(Book(D("100000")), intent("0.019"), PORTFOLIO)
    assert small.outcome == "rejected" and small.limit_hit == "below_min" and small.approved_qty == 0


def test_an_unreadable_listing_takes_the_kept_lot_and_says_so(built, capfd):
    def down(pair):
        raise ConnectionError("no answer")

    inst = built(down)
    assert inst.size_increment.as_decimal() == D("0.001") and inst.min_quantity.as_decimal() == D("0.001")
    assert "the kept step 0.001 and minimum 0.001 are used" in capfd.readouterr().err


def test_a_live_lot_that_differs_from_the_kept_one_wins_and_says_so(built, capfd):
    inst = built(lambda pair: (D("0.01"), D("0.01")))
    assert inst.size_increment.as_decimal() == D("0.01") and inst.min_quantity.as_decimal() == D("0.01")
    assert "differs from the kept" in capfd.readouterr().err


@pytest.mark.parametrize("qty", ["0.004", "0.0199"])
def test_a_size_between_0_and_the_minimum_after_rounding_is_refused_in_words_never_sent_as_0(built, qty):
    s = Sizer(built(lambda pair: paper_venue_lot(pair, get_json=listing("0.01", "0.02"))))
    d = decide(Book(D("100000")), Intent("BTC", 1, D(qty), D("60000"), D("60000"), D("600"), s._lot(), s._min_qty()),
               PORTFOLIO)
    assert d.outcome == "rejected" and d.approved_qty == 0 and d.limit_hit == "below_min"
    assert d.reason.startswith("Entry rejected") and "0.02" in d.reason


@pytest.mark.parametrize("live_lot", ["unreadable", None])
def test_cr2_with_no_live_lot_the_kept_lot_wins_over_the_other_venues_contract(monkeypatch, live_lot):
    # CR2 MUST: the contract differs from the kept lot, so only the kept lot can give 0.001 / 0.001
    def down(pair):
        raise ConnectionError("no answer")

    monkeypatch.setattr(PAPER, "contract", lambda pair: {**CONTRACT, "size_precision": 2, "min_quantity": 0.01})
    monkeypatch.setattr(PAPER, "live_lot", down if live_lot == "unreadable" else None)
    inst = PAPER.instrument("BTC", "USDT")
    assert inst.size_increment.as_decimal() == D("0.001") and inst.min_quantity.as_decimal() == D("0.001")


# --- Advisor B44-STEP (22:05 UK): a listed step the engine can't round to is an error at start; entries refused, exits run

def test_b44_step_an_unusable_listed_step_is_said_with_both_steps_and_cleared_by_a_good_read(built):
    PAPER.lot_refusals.clear()
    built(lambda pair: paper_venue_lot(pair, get_json=listing("0.005", "0.005")))
    why = PAPER.lot_refusals["BTC/USDT"]
    assert "BTCUSDT-PERP lists a quantity step of 0.005" in why and "the kept step 0.001" in why
    assert "new entries are refused" in why and "exits and stops still run" in why
    built(lambda pair: paper_venue_lot(pair, get_json=listing()))
    assert "BTC/USDT" not in PAPER.lot_refusals


def _strategy(monkeypatch):
    from types import SimpleNamespace

    from test_fee_live_wiring import T, _store

    store = _store()
    monkeypatch.setitem(PAPER.lot_refusals, "BTC/USDT", "BTCUSDT-PERP lists a quantity step of 0.005, ...")
    return store, SimpleNamespace(_backtest=False, runtime=SimpleNamespace(store=store, name="p1", venue="BINANCE",
                                                                           now=lambda: T))


def test_b44_step_entries_on_that_symbol_are_refused_and_the_start_raises_an_error(monkeypatch):
    store, s = _strategy(monkeypatch)
    assert LongFlatStrategy._venue_terms_refusal(s).startswith("Entry refused: BTCUSDT-PERP lists a quantity step")
    LongFlatStrategy._say_lot(s)
    (e,) = store.events_of(("lot_step_refused",))
    assert e["level"] == "error" and "0.005" in e["message"]


def test_b44_step_exits_still_run_while_entries_are_refused(prices, instrument, full_margin, monkeypatch):
    """Entries ask _venue_terms_refusal; exits never do. Allowed once (the entry), refused after: the exit still fills."""
    from sleeve_fund.research.runner import run_backtest
    from test_backtest import _path

    asked = []

    def refusal(self):
        asked.append(1)
        return None if len(asked) == 1 else "Entry refused: BTCUSDT-PERP lists a quantity step of 0.005"

    monkeypatch.setattr(LongFlatStrategy, "_venue_terms_refusal", refusal)
    feed = _path(prices, [100.0, 100.5, 100.8, 101.5, 101.5, 101.0, 100.0, 99.0, 98.5])
    res = run_backtest("ping_pong", feed, instrument, {"market": "perp"}, half_spread=0, risk_profile="aggressive")
    sides = list(res.fills["side"]) if not res.fills.empty else []
    assert len(asked) >= 2 and sides[:2] == ["BUY", "SELL"], sides


def test_qd_a_power_of_ten_step_above_1_is_kept_and_sizing_rounds_to_it(built):
    # QD MUST 2: a 10 step must not fall back to 1 (size_precision 0); quantize honours 1E+1
    from decimal import ROUND_DOWN

    inst = built(lambda pair: paper_venue_lot(pair, get_json=listing("10", "10")))
    assert inst.size_increment.as_decimal() == D("10") and inst.min_quantity.as_decimal() == D("10")
    s = Sizer(inst)
    assert s._lot() == D("10") and D("23").quantize(s._lot(), ROUND_DOWN) == D("20")
