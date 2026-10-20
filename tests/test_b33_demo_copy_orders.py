"""B-33: the demo copy sends one demo order per paper ORDER, and every exit goes reduce-only.

On 15 Oct a paper order filled in pieces became several demo orders, and an exit larger than the demo position
opened the other side (the demo showed "Open Short"). Fake demo endpoint only: nothing here reaches a real account."""
import json
import os
import uuid

import pytest
from sqlalchemy import create_engine, text

from sleeve_fund import mirror
from sleeve_fund.store import CLOSING_INTENTS, Store


class FakeDemo(mirror.BybitDemo):
    """The demo account's answers, without the network. A reduce-only order never moves the position past flat."""

    def __init__(self, held=0.0):
        super().__init__(mirror.Settings("k", "s"), http=self.http)
        self.orders, self.held, self.fail = [], held, False

    def http(self, method, url, headers, body):
        if "/v5/position/list" in url:
            side = "Buy" if self.held > 0 else "Sell" if self.held < 0 else ""
            return {"retCode": 0, "result": {"list": [{"symbol": "BTCUSDT", "side": side, "size": str(abs(self.held))}]}}
        if "/v5/order/realtime" in url:
            return {"retCode": 0, "result": {"list": [{"avgPrice": "60000"}]}}
        if self.fail:
            return {"retCode": 110007, "retMsg": "ab not enough for new order"}
        o = json.loads(body)
        self.orders.append(o)
        move = (1 if o["side"] == "Buy" else -1) * float(o["qty"])
        if o.get("reduceOnly") and (self.held == 0 or (self.held > 0) == (move > 0)):
            return {"retCode": 110017, "retMsg": "reduce-only order would increase the position"}
        if o.get("reduceOnly"):
            move = max(-abs(self.held), min(abs(self.held), move)) if self.held else 0.0
        self.held = round(self.held + move, 8)
        return {"retCode": 0, "result": {"orderId": str(len(self.orders))}}


PG = os.environ.get("TEST_DATABASE_URL")
_SCHEMAS: list[str] = []
_ENGINES: list = []


@pytest.fixture(scope="module", autouse=True)
def _drop_schemas():
    yield
    for eng in _ENGINES:
        eng.dispose()
    if PG and _SCHEMAS:
        admin = create_engine(PG)
        for name in _SCHEMAS:
            with admin.begin() as c:
                c.execute(text(f"DROP SCHEMA IF EXISTS {name} CASCADE"))
        admin.dispose()


@pytest.fixture(autouse=True)
def _fresh_warnings():
    """The mirror warns once per gap or failure per process; each cell starts as a fresh process."""
    getattr(mirror, "_GAP_ALERTED", {}).clear()
    mirror._CATCH_UP_FAILED.clear()
    yield


def _journal():
    """SQLite in memory, or a fresh Postgres schema when TEST_DATABASE_URL is set (CI): both engines."""
    if not PG:
        return Store.in_memory()
    name = f"b33_{uuid.uuid4().hex[:12]}"
    admin = create_engine(PG)
    with admin.begin() as c:
        c.execute(text(f"CREATE SCHEMA {name}"))
    admin.dispose()
    _SCHEMAS.append(name)
    eng = create_engine(PG, connect_args={"options": f"-csearch_path={name}"})
    _ENGINES.append(eng)
    return Store(engine=eng)


def _store(*names):
    store = _journal()
    for name in names or ("bn-ls",):
        store.create_sleeve(name=name, strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                            starting_balance=10_000, params={"market": "perp", "allow_short": True, "demo_mirror": True},
                            venue="binance")
    return store


def _order(store, name, oid, side, qty, intent="entry", reason=""):
    store.record_order(name, order_id=oid, side=side, qty=qty, intent=intent, reason=reason)


def _piece(store, name, oid, side, qty, n, status):
    store.record_fill(name, side=side, qty=qty, price=60_000.0, fee=1.0, order_id=oid, trade_id=f"{oid}-t{n}")
    store.update_order(oid, status=status, fill_qty=qty, fill_px=60_000.0)


def _start(store, demo):
    targets = {"BYBIT": demo}
    mirror.mirror_once(store, targets)  # first sight: the watermark
    return targets


def test_a_paper_order_filled_in_three_pieces_is_one_demo_order_for_the_sum():
    store, demo = _store(), FakeDemo()
    targets = _start(store, demo)
    _order(store, "bn-ls", "o1", "BUY", 0.3)
    _piece(store, "bn-ls", "o1", "BUY", 0.1, 1, "partially_filled")
    _piece(store, "bn-ls", "o1", "BUY", 0.1, 2, "partially_filled")
    _piece(store, "bn-ls", "o1", "BUY", 0.1, 3, "filled")
    mirror.mirror_once(store, targets)
    assert [(o["side"], o["qty"]) for o in demo.orders] == [("Buy", "0.3")]
    assert demo.held == pytest.approx(0.3)
    assert mirror.mirror_once(store, targets) == 0  # nothing sent twice


def test_an_order_still_filling_is_not_copied_until_it_is_done():
    store, demo = _store(), FakeDemo()
    targets = _start(store, demo)
    _order(store, "bn-ls", "o1", "BUY", 0.2)
    _piece(store, "bn-ls", "o1", "BUY", 0.1, 1, "partially_filled")
    assert mirror.mirror_once(store, targets) == 0 and demo.orders == []
    _piece(store, "bn-ls", "o1", "BUY", 0.1, 2, "filled")
    assert mirror.mirror_once(store, targets) == 1
    assert [o["qty"] for o in demo.orders] == ["0.2"]


def test_catch_up_between_two_pieces_of_one_order_sends_nothing():
    store, demo = _store(), FakeDemo()
    targets = _start(store, demo)
    _order(store, "bn-ls", "o1", "BUY", 0.2)
    _piece(store, "bn-ls", "o1", "BUY", 0.1, 1, "partially_filled")
    mirror.mirror_once(store, targets)
    seen = mirror.catch_up(store, targets, {})
    seen = mirror.catch_up(store, targets, seen)  # the same gap twice: still being copied, not a missed copy
    assert demo.orders == []
    _piece(store, "bn-ls", "o1", "BUY", 0.1, 2, "filled")
    mirror.mirror_once(store, targets)
    mirror.catch_up(store, targets, mirror.catch_up(store, targets, {}))
    assert [o["qty"] for o in demo.orders] == ["0.2"] and demo.held == pytest.approx(0.2)


@pytest.mark.parametrize("intent", sorted({*CLOSING_INTENTS, "liquidation"}))
def test_every_closing_order_is_sent_reduce_only(intent):
    store, demo = _store(), FakeDemo()
    targets = _start(store, demo)
    _order(store, "bn-ls", "o1", "BUY", 0.1)
    _piece(store, "bn-ls", "o1", "BUY", 0.1, 1, "filled")
    mirror.mirror_once(store, targets)
    _order(store, "bn-ls", "o2", "SELL", 0.1, intent=intent)
    _piece(store, "bn-ls", "o2", "SELL", 0.1, 1, "filled")
    mirror.mirror_once(store, targets)
    entry, close = demo.orders
    assert not entry.get("reduceOnly") and close.get("reduceOnly") is True
    assert demo.held == 0.0


def test_a_reset_close_is_reduce_only():
    store, demo = _store(), FakeDemo()
    targets = _start(store, demo)
    _order(store, "bn-ls", "o1", "BUY", 0.1)
    _piece(store, "bn-ls", "o1", "BUY", 0.1, 1, "filled")
    mirror.mirror_once(store, targets)
    _order(store, "bn-ls", "o2", "SELL", 0.1, intent="pm_flatten", reason="Reset strategy")
    _piece(store, "bn-ls", "o2", "SELL", 0.1, 1, "filled")
    mirror.mirror_once(store, targets)
    assert demo.orders[-1].get("reduceOnly") is True and "-reset-" in demo.orders[-1]["orderLinkId"]


def test_an_exit_larger_than_the_demo_position_never_opens_the_other_side():
    """The 15 Oct case: the demo held less than the paper book, and the full exit opened a short."""
    store, demo = _store(), FakeDemo()
    targets = _start(store, demo)
    _order(store, "bn-ls", "o1", "BUY", 0.1)
    _piece(store, "bn-ls", "o1", "BUY", 0.1, 1, "filled")
    mirror.mirror_once(store, targets)
    demo.held = 0.04  # part of the copy never landed
    _order(store, "bn-ls", "o2", "SELL", 0.1, intent="exit")
    _piece(store, "bn-ls", "o2", "SELL", 0.1, 1, "filled")
    mirror.mirror_once(store, targets)
    assert demo.held >= 0.0  # flat at worst, never short
    # capped by the mirror itself, not left to the venue: only what the account holds is sent
    assert demo.orders[-1]["qty"] == "0.04" and demo.orders[-1]["reduceOnly"] is True
    assert store.mirror_positions() == {"BTCUSDT": pytest.approx(0.06)}  # 0.1 put on, 0.04 taken off


def test_a_close_when_the_demo_account_holds_nothing_on_that_side_sends_nothing():
    store, demo = _store(), FakeDemo()
    targets = _start(store, demo)
    _order(store, "bn-ls", "o1", "BUY", 0.1)
    _piece(store, "bn-ls", "o1", "BUY", 0.1, 1, "filled")
    mirror.mirror_once(store, targets)
    demo.held = 0.0  # closed by hand on the demo account
    _order(store, "bn-ls", "o2", "SELL", 0.1, intent="stop_loss")
    _piece(store, "bn-ls", "o2", "SELL", 0.1, 1, "filled")
    assert mirror.mirror_once(store, targets) == 0
    assert len(demo.orders) == 1 and demo.held == 0.0
    assert store.mirror_rows("bn-ls")[0]["status"] == "skipped"


def test_two_long_names_never_share_a_demo_link_id():
    a, b = "candle-step-btcusdt-1m-" + "z" * 17 + "a", "candle-step-btcusdt-1m-" + "z" * 17 + "b"
    assert len(a) == len(b) == 41
    store, demo = _store(a, b), FakeDemo()
    targets = _start(store, demo)
    for n, name in enumerate((a, b)):
        _order(store, name, f"o{n}", "BUY", 0.1)
        _piece(store, name, f"o{n}", "BUY", 0.1, 1, "filled")
    mirror.mirror_once(store, targets)
    links = [o["orderLinkId"] for o in demo.orders]
    assert len(links) == 2 and len(set(links)) == 2 and all(len(x) <= 36 for x in links)


def test_two_long_names_never_share_a_reset_link_id():
    a, b = "x" * 20 + "-one", "x" * 20 + "-two"
    store, demo = _store(a, b), FakeDemo()
    targets = _start(store, demo)
    for n, name in enumerate((a, b)):
        _order(store, name, f"e{n}", "BUY", 0.1)
        _piece(store, name, f"e{n}", "BUY", 0.1, 1, "filled")
        _order(store, name, f"r{n}", "SELL", 0.1, intent="pm_flatten", reason="Reset strategy")
        _piece(store, name, f"r{n}", "SELL", 0.1, 1, "filled")
    mirror.mirror_once(store, targets)
    resets = [o["orderLinkId"] for o in demo.orders if "-reset-" in o["orderLinkId"]]
    assert len(resets) == 2 and len(set(resets)) == 2


# --- Advisor ruling B33-DEMO-COPY (20 Oct): copies sized by their effect on the shared paper net ------------------

def test_an_exit_that_moves_the_shared_net_away_from_zero_goes_as_a_plain_order():
    """A long 1, B short 2: net -1, the demo -1. A's exit of 1 takes the paper net to -2, so the demo sells 1 as a
    plain order (reduce-only would be refused on a short and fixed late by catch_up)."""
    store, demo = _store("a", "b"), FakeDemo()
    targets = _start(store, demo)
    _order(store, "a", "a1", "BUY", 1.0)
    _piece(store, "a", "a1", "BUY", 1.0, 1, "filled")
    _order(store, "b", "b1", "SELL", 2.0)
    _piece(store, "b", "b1", "SELL", 2.0, 1, "filled")
    mirror.mirror_once(store, targets)
    assert demo.held == pytest.approx(-1.0)
    _order(store, "a", "a2", "SELL", 1.0, intent="exit")
    _piece(store, "a", "a2", "SELL", 1.0, 1, "filled")
    mirror.mirror_once(store, targets)
    last = demo.orders[-1]
    assert (last["side"], last["qty"], last.get("reduceOnly")) == ("Sell", "1", None)
    assert demo.held == pytest.approx(-2.0)


def test_the_15_oct_exit_is_reduce_only_and_cannot_open_a_short():
    store, demo = _store(), FakeDemo()
    targets = _start(store, demo)
    _order(store, "bn-ls", "o1", "BUY", 1.0)
    _piece(store, "bn-ls", "o1", "BUY", 1.0, 1, "filled")
    mirror.mirror_once(store, targets)
    _order(store, "bn-ls", "o2", "SELL", 1.0, intent="exit")
    _piece(store, "bn-ls", "o2", "SELL", 1.0, 1, "filled")
    mirror.mirror_once(store, targets)
    last = demo.orders[-1]
    assert (last["side"], last["qty"], last["reduceOnly"]) == ("Sell", "1", True) and demo.held == 0.0


def test_a_reversal_is_a_reduce_only_part_to_zero_then_a_plain_part_to_the_paper_net():
    store, demo = _store(), FakeDemo()
    targets = _start(store, demo)
    _order(store, "bn-ls", "o1", "BUY", 0.5)
    _piece(store, "bn-ls", "o1", "BUY", 0.5, 1, "filled")
    mirror.mirror_once(store, targets)
    _order(store, "bn-ls", "o2", "SELL", 0.8, intent="entry")  # close and reverse
    _piece(store, "bn-ls", "o2", "SELL", 0.8, 1, "filled")
    mirror.mirror_once(store, targets)
    assert [(o["qty"], o.get("reduceOnly")) for o in demo.orders[1:]] == [("0.5", True), ("0.3", None)]
    assert demo.held == pytest.approx(-0.3)


def test_an_entry_never_takes_the_demo_past_the_paper_net():
    store, demo = _store(), FakeDemo(held=0.2)  # bought by hand on the demo account
    targets = _start(store, demo)
    _order(store, "bn-ls", "o1", "BUY", 0.3)
    _piece(store, "bn-ls", "o1", "BUY", 0.3, 1, "filled")
    mirror.mirror_once(store, targets)
    assert [o["qty"] for o in demo.orders] == ["0.1"] and demo.held == pytest.approx(0.3)


def test_a_restart_between_two_partial_fills_still_gives_exactly_one_order():
    """Held fills live in the journal (pending rows), not the process: a fresh mirror (a restart) rebuilds them."""
    store, demo = _store(), FakeDemo()
    targets = _start(store, demo)
    _order(store, "bn-ls", "o1", "BUY", 0.2)
    _piece(store, "bn-ls", "o1", "BUY", 0.1, 1, "partially_filled")
    mirror.mirror_once(store, targets)
    getattr(mirror, "_GAP_ALERTED", {}).clear()  # the restart: the process's memory is gone; only the journal is left
    mirror._CATCH_UP_FAILED.clear()
    _piece(store, "bn-ls", "o1", "BUY", 0.1, 2, "filled")
    mirror.mirror_once(store, {"BYBIT": demo})
    mirror.mirror_once(store, {"BYBIT": demo})
    assert [o["qty"] for o in demo.orders] == ["0.2"] and demo.held == pytest.approx(0.2)


def test_a_resting_order_still_open_after_two_cycles_sends_what_has_filled_then_groups_the_rest():
    from datetime import timedelta

    from sleeve_fund.store import utcnow

    store, demo = _store(), FakeDemo()
    targets = _start(store, demo)
    t0 = utcnow()
    _order(store, "bn-ls", "o1", "BUY", 0.3)
    _piece(store, "bn-ls", "o1", "BUY", 0.1, 1, "partially_filled")
    mirror.mirror_once(store, targets, now=t0)
    assert mirror.mirror_once(store, targets, now=t0 + timedelta(seconds=mirror.POLL_SECONDS)) == 0
    assert mirror.mirror_once(store, targets, now=t0 + timedelta(seconds=mirror.HOLD_SECONDS)) == 1
    assert [o["qty"] for o in demo.orders] == ["0.1"]
    _piece(store, "bn-ls", "o1", "BUY", 0.1, 2, "partially_filled")
    _piece(store, "bn-ls", "o1", "BUY", 0.1, 3, "filled")
    mirror.mirror_once(store, targets, now=t0 + timedelta(seconds=3 * mirror.HOLD_SECONDS))
    assert [o["qty"] for o in demo.orders] == ["0.1", "0.2"] and demo.held == pytest.approx(0.3)


def test_catch_up_never_sends_past_the_paper_net():
    store, demo = _store(), FakeDemo()
    targets = _start(store, demo)
    _order(store, "bn-ls", "o1", "BUY", 0.1)
    _piece(store, "bn-ls", "o1", "BUY", 0.1, 1, "filled")
    demo.fail = True  # the copy is refused
    mirror.mirror_once(store, targets)
    demo.fail = False
    demo.held = 0.25  # and the account was bought up by hand meanwhile
    seen = mirror.catch_up(store, targets, mirror.catch_up(store, targets, {}))
    assert demo.held == pytest.approx(0.25) and demo.orders == []  # already past the paper net: nothing bought
    assert seen == {"bn-ls": pytest.approx(0.1)}


def test_a_refused_copy_catch_up_cannot_close_raises_an_alert():
    store, demo = _store(), FakeDemo()
    targets = _start(store, demo)
    _order(store, "bn-ls", "o1", "BUY", 0.1)
    _piece(store, "bn-ls", "o1", "BUY", 0.1, 1, "filled")
    demo.fail = True
    mirror.mirror_once(store, targets)
    seen = mirror.catch_up(store, targets, {})
    mirror.catch_up(store, targets, seen)  # two checks, still refused: not closed
    alerts = [e for e in store.events("bn-ls") if e["kind"] == "mirror_gap"]
    assert len(alerts) == 1 and alerts[0]["level"] == "error"
    mirror.catch_up(store, targets, seen)
    assert len([e for e in store.events("bn-ls") if e["kind"] == "mirror_gap"]) == 1  # once per gap, not a flood


# CR B-33 MUSTs -------------------------------------------------------------------------------------------------------
class WentInButErrored(FakeDemo):
    """A copy whose answer was lost: the order went in at the venue, but the mirror saw an error."""

    lose = False

    def http(self, method, url, headers, body):
        answer = super().http(method, url, headers, body)
        if self.lose and "/v5/order/create" in url:
            return {"retCode": 10016, "retMsg": "server timeout"}
        return answer


def _gap_alerts(store, *names):
    return [e for n in names for e in store.events(n) if e["kind"] == "mirror_gap"]


def test_a_copy_that_errored_but_went_in_raises_no_gap_alert_now_or_after_a_restart():
    """CR MUST 1 (probe b33-false-gap-alert-probe.py): B's sell errored but filled, so the demo is level with the
    paper net at 0 while B's own gap stays open. That is no gap on the account: nothing to send, nothing to alert."""
    store, demo = _store("a-long", "b-short"), WentInButErrored()
    targets = _start(store, demo)
    _order(store, "a-long", "oa", "BUY", 0.1)
    _piece(store, "a-long", "oa", "BUY", 0.1, 1, "filled")
    mirror.mirror_once(store, targets)
    demo.lose = True
    _order(store, "b-short", "ob", "SELL", 0.1)
    _piece(store, "b-short", "ob", "SELL", 0.1, 1, "filled")
    mirror.mirror_once(store, targets)
    demo.lose = False
    sent = len(demo.orders)
    seen = mirror.catch_up(store, targets, mirror.catch_up(store, targets, {}))
    mirror._GAP_ALERTED.clear()  # a restart: only the journal is left
    mirror.catch_up(store, targets, mirror.catch_up(store, targets, seen))
    assert demo.held == pytest.approx(0.0) and len(demo.orders) == sent  # nothing sent twice
    assert _gap_alerts(store, "a-long", "b-short") == []


def test_an_open_gap_reaches_the_pms_alert_channel_once_not_once_per_poll_or_restart():
    """Advisor B33-DEMO-COPY 3(b): an alert, not a log line. The alerts thread sends every error event to the PM's
    channel (ALERT_WEBHOOK_URL, ntfy); the gap's event is written once, so it is sent once."""
    from sleeve_fund import alerts

    store, demo = _store(), FakeDemo()
    targets = _start(store, demo)
    sent = []
    sender = alerts.Forwarder(store, environ={"ALERT_WEBHOOK_URL": "https://ntfy.sh/pm-topic"},
                           send=lambda url, text: sent.append(text), pinger=lambda url: None)
    _order(store, "bn-ls", "o1", "BUY", 0.1)
    _piece(store, "bn-ls", "o1", "BUY", 0.1, 1, "filled")
    demo.fail = True
    mirror.mirror_once(store, targets)
    seen = {}
    for _ in range(4):  # four polls with the gap open
        seen = mirror.catch_up(store, targets, seen)
        sender.step()
    mirror._GAP_ALERTED.clear()  # a restart
    for _ in range(2):
        seen = mirror.catch_up(store, targets, seen)
        sender.step()
    assert len(_gap_alerts(store, "bn-ls")) == 1
    assert sum("still 0.1 BTCUSDT short of the paper position" in t for t in sent) == 1


def test_a_gap_that_closes_and_opens_again_is_alerted_again():
    store, demo = _store(), FakeDemo()
    targets = _start(store, demo)
    _order(store, "bn-ls", "o1", "BUY", 0.1)
    _piece(store, "bn-ls", "o1", "BUY", 0.1, 1, "filled")
    demo.fail = True
    mirror.mirror_once(store, targets)
    seen = mirror.catch_up(store, targets, mirror.catch_up(store, targets, {}))
    demo.fail = False
    seen = mirror.catch_up(store, targets, seen)  # caught up
    assert demo.held == pytest.approx(0.1)
    mirror.catch_up(store, targets, seen)  # level: the alert is closed
    demo.fail = True  # the next order's copy is refused too: a gap of the same size, so the same words
    _order(store, "bn-ls", "o2", "BUY", 0.1)
    _piece(store, "bn-ls", "o2", "BUY", 0.1, 1, "filled")
    mirror.mirror_once(store, targets)
    mirror.catch_up(store, targets, mirror.catch_up(store, targets, {}))
    kinds = [e["kind"] for e in reversed(store.events("bn-ls")) if e["kind"].startswith("mirror_gap")]
    assert kinds == ["mirror_gap", "mirror_gap_closed", "mirror_gap"]


class FakeTestnet:
    """The Deribit testnet's sizing and its reduce-only rule, without the network."""

    label, host, unit = "the Deribit testnet", mirror.TESTNET_HOST, "USD"
    size, owns = staticmethod(mirror.Testnet.size), staticmethod(mirror.Testnet.owns)

    def __init__(self, held=0.0):
        self.orders, self.held = [], held

    def market(self, side, instrument, amount, label, reduce_only=False):
        self.orders.append((side, amount, reduce_only))
        return f"d{len(self.orders)}", 60_000.0

    def position(self, instrument):
        return self.held


@pytest.mark.parametrize("intent", ["exit", "stop_loss"])
def test_on_the_second_venue_a_close_goes_reduce_only_and_never_past_the_position(intent):
    """CR MUST 2: Deribit keeps both (Advisor B33): a closing copy is reduce-only, capped at what the testnet holds.
    The closing = False mutant fails here."""
    store = _journal()
    store.create_sleeve(name="dr-ls", strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params={"market": "perp", "allow_short": True, "demo_mirror": True})
    venue = FakeTestnet(held=3000.0)
    mirror.mirror_once(store, {"DERIBIT": venue})
    _order(store, "dr-ls", "o1", "SELL", 0.1, intent=intent)
    _piece(store, "dr-ls", "o1", "SELL", 0.1, 1, "filled")
    mirror.mirror_once(store, {"DERIBIT": venue})
    assert venue.orders == [("SELL", 3000.0, True)]  # 0.1 at 60,000 is 6,000 USD; the testnet held 3,000
