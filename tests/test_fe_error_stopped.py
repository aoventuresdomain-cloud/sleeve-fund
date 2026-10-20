"""HoQA NOTE 2 on Head 4's live check (HoE, 20 Oct): candle-step-btcusdt-1m still reads "error" with the stale reason
from B-29's start crash. A Strategy left in error that the PM has stopped (or put away, which stops it first) reads as
stopped on every page: its error was its last run's, and nothing retries it. The stored status is untouched (changing
the live row needs the PM's word); one that is meant to run and errored still reads as error, waiting for the PM."""

import re

from test_fe_a20_fund_book_value import AUTH, _hold, _make, _text, c  # noqa: F401
from test_halt_ref import store  # noqa: F401

NAME = "candle-step-btcusdt-1m"
STALE = "node exited 1"


def _errored(store, desired="stopped"):  # noqa: F811
    _hold(store)
    _make(store, NAME, 0)
    _make(store, "timed-test", 0)
    store.set_status(NAME, "error", STALE)
    if desired != "stopped":
        store.set_desired_state(NAME, desired)


def test_stopped_after_an_error_home_reads_stopped_and_nobody_waits(c, store):  # noqa: F811
    _errored(store)
    home = c.get("/", auth=AUTH).text
    assert "halted and waiting" not in _text(home)
    rows = re.findall(rf'<tr[^>]*>(?:(?!</tr>).)*href="/sleeves/{NAME}"(?:(?!</tr>).)*</tr>', home, re.DOTALL)
    row = next(r for r in rows if 'class="chip' in r)  # the Strategies table's row, with its status
    assert f'title="{NAME}: stopped"><span class="sdot stopped"' in home  # the side list too
    assert 'class="chip stopped">Stopped' in row and "chip error" not in row, row


def test_stopped_after_an_error_its_page_reads_stopped_without_the_stale_reason(c, store):  # noqa: F811
    _errored(store)
    page = c.get(f"/sleeves/{NAME}", auth=AUTH).text
    head = re.search(r"<h1>.*?</div>", page, re.DOTALL).group(0)
    assert 'class="chip stopped">Stopped' in head and "chip error" not in head, head
    assert "Error:" not in _text(page) and STALE not in _text(page)


def test_stopped_after_an_error_risk_and_health_names_nothing_bad(c, store):  # noqa: F811
    _errored(store)
    risk = _text(c.get("/risk", auth=AUTH).text)
    assert "is error" not in risk and STALE not in risk, risk[:800]


def test_meant_to_run_and_errored_still_reads_error_and_waits(c, store):  # noqa: F811
    _errored(store, desired="running")
    home = c.get("/", auth=AUTH).text
    assert "1 strategy halted and waiting for you" in _text(home)
    page = _text(c.get(f"/sleeves/{NAME}", auth=AUTH).text)
    assert f"Error: {STALE}" in page


def test_stopped_after_an_error_but_still_holding_stays_error(c, store):  # noqa: F811
    """HoE: only a flat one reads as stopped; one still holding a position is in error and waits for the PM."""
    _errored(store)
    store.record_order(NAME, order_id="held-1", side="BUY", qty=0.1, intent="entry", reason="dipped")
    store.update_order("held-1", fill_qty=0.1, fill_px=60_000.0, fee=3.6)
    store.record_fill(NAME, side="BUY", qty=0.1, price=60_000.0, fee=3.6, order_id="held-1", trade_id="held-T")
    store.record_equity(NAME, equity=-3.6, cash=-6_003.6, qty=0.1, price=60_000.0, benchmark=0)
    assert store.holds_any(NAME)
    assert "1 strategy halted and waiting for you" in _text(c.get("/", auth=AUTH).text)
    assert f"Error: {STALE}" in _text(c.get(f"/sleeves/{NAME}", auth=AUTH).text)


def test_stopped_after_an_error_with_a_working_order_stays_error(c, store):  # noqa: F811
    _errored(store)
    store.record_order(NAME, order_id="work-1", side="BUY", qty=0.1, intent="entry", reason="dipped")
    assert not store.holds_any(NAME)
    assert "1 strategy halted and waiting for you" in _text(c.get("/", auth=AUTH).text)


def test_the_shown_status_is_display_only():
    """HoE: no gate, brake, risk or order path reads it; only the dashboard does."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "sleeve_fund"
    readers = {str(p.relative_to(root)) for p in root.rglob("*.py") if "shown_status" in p.read_text()}
    assert readers and all(r.startswith("dashboard/") for r in readers), readers
