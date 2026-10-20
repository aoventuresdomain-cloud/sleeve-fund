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
    assert paper_venue_lot("BTC/USDT", get_json=listing("0.005", "0.010")) == (D("0.005"), D("0.01"))


@pytest.mark.parametrize("bad", [dict(drop=True), dict(step="0"), dict(step="0.001", least="0.0015")])
def test_a_listing_with_no_usable_lot_size_is_refused(bad):
    with pytest.raises(ValueError):
        paper_venue_lot("BTC/USDT", get_json=listing(**bad))


def test_the_instrument_carries_the_listings_step_and_minimum_not_the_data_venues(built):
    inst = built(lambda pair: paper_venue_lot(pair, get_json=listing("0.005", "0.010")))
    assert inst.size_increment.as_decimal() == D("0.005") and inst.min_quantity.as_decimal() == D("0.010")
    s = Sizer(inst)
    assert s._lot() == D("0.005") and s._min_qty() == D("0.010")


def test_sizing_rounds_down_to_the_step_and_refuses_below_the_minimum(built):
    s = Sizer(built(lambda pair: paper_venue_lot(pair, get_json=listing("0.005", "0.010"))))

    def intent(qty):
        return Intent("BTC", 1, D(qty), D("60000"), D("60000"), D("600"), s._lot(), s._min_qty())

    ok = decide(Book(D("100000")), intent("0.0237"), PORTFOLIO)
    assert ok.approved_qty == D("0.020")
    small = decide(Book(D("100000")), intent("0.008"), PORTFOLIO)
    assert small.outcome == "rejected" and small.limit_hit == "below_min" and small.approved_qty == 0


def test_an_unreadable_listing_takes_the_kept_lot_and_says_so(built, capfd):
    def down(pair):
        raise ConnectionError("no answer")

    inst = built(down)
    assert inst.size_increment.as_decimal() == D("0.001") and inst.min_quantity.as_decimal() == D("0.001")
    assert "the kept step 0.001 and minimum 0.001 are used" in capfd.readouterr().err


def test_a_live_lot_that_differs_from_the_kept_one_wins_and_says_so(built, capfd):
    inst = built(lambda pair: (D("0.002"), D("0.002")))
    assert inst.size_increment.as_decimal() == D("0.002") and inst.min_quantity.as_decimal() == D("0.002")
    assert "differs from the kept" in capfd.readouterr().err


@pytest.mark.parametrize("qty", ["0.004", "0.0099"])
def test_a_size_between_0_and_the_minimum_after_rounding_is_refused_in_words_never_sent_as_0(built, qty):
    s = Sizer(built(lambda pair: paper_venue_lot(pair, get_json=listing("0.005", "0.010"))))
    d = decide(Book(D("100000")), Intent("BTC", 1, D(qty), D("60000"), D("60000"), D("600"), s._lot(), s._min_qty()),
               PORTFOLIO)
    assert d.outcome == "rejected" and d.approved_qty == 0 and d.limit_hit == "below_min"
    assert d.reason.startswith("Entry rejected") and "0.01" in d.reason


@pytest.mark.parametrize("live_lot", ["unreadable", None])
def test_cr2_with_no_live_lot_the_kept_lot_wins_over_the_other_venues_contract(monkeypatch, live_lot):
    # CR2 MUST: the contract differs from the kept lot, so only the kept lot can give 0.001 / 0.001
    def down(pair):
        raise ConnectionError("no answer")

    monkeypatch.setattr(PAPER, "contract", lambda pair: {**CONTRACT, "size_precision": 2, "min_quantity": 0.01})
    monkeypatch.setattr(PAPER, "live_lot", down if live_lot == "unreadable" else None)
    inst = PAPER.instrument("BTC", "USDT")
    assert inst.size_increment.as_decimal() == D("0.001") and inst.min_quantity.as_decimal() == D("0.001")
