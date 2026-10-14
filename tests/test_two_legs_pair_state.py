"""P2-8 Advisor M3, the journal (DA 14 Oct 05:02 UK): a two-leg Strategy's coordinator state is one pair_state row per
parent, moved by compare-and-set with one event per move; its summed-guard figures are set at the first mark, the
high-water mark and halt reference only ever rise, the day's start is written once a day or re-based on a Resume."""

import os
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal as D

import pytest

from sleeve_fund.store import Store, make_engine, metadata

T = datetime(2026, 10, 14, 9, 0, tzinfo=UTC)
SPOT = {"instrument": "BTC/USDT", "params": {}, "venue": "KRAKEN", "side": 1}
PERP = {"instrument": "BTC/USDT", "params": {"market": "perp"}, "venue": "KRAKEN", "side": -1}


@pytest.fixture
def store(tmp_path):
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        eng = make_engine(url)
        metadata.drop_all(eng)
        st = Store(engine=eng)
    else:
        st = Store(f"sqlite:///{tmp_path}/pair.db")
    st.create_two_legs(name="carry", strategy="buy_and_hold", bar_spec="1-DAY-LAST-EXTERNAL", starting_balance=10_000,
                       first=SPOT, second=PERP)
    yield st
    if url:
        metadata.drop_all(st.engine)


def test_each_move_is_a_compare_and_set_with_one_event(store):
    assert store.set_pair_state("carry", expected=None, state="flat", ts=T)
    assert not store.set_pair_state("carry", expected=None, state="flat", ts=T)  # it has its row already
    assert store.set_pair_state("carry", expected="flat", state="legging", trade_id="t1", deadline=T, ts=T)
    assert not store.set_pair_state("carry", expected="flat", state="legging", trade_id="t2", deadline=T, ts=T)  # lost
    row = store.pair_state("carry")
    assert (row["state"], row["trade_id"]) == ("legging", "t1")
    moves = [e["message"] for e in store.events("carry", limit=50) if e["kind"] == "pair_state"]
    assert sorted(moves) == sorted(["new -> flat", "flat -> legging (t1)"])


def test_a_leg_row_never_has_a_pair_state(store):
    with pytest.raises(ValueError, match="first leg"):
        store.set_pair_state("carry~2", expected=None, state="flat", ts=T)


def test_the_figures_are_set_together_and_the_high_only_rises(store):
    store.set_pair_state("carry", expected=None, state="flat", ts=T)
    row = store.mark_pair("carry", 10_000, date(2026, 10, 14), T)
    assert (row["reference_equity"], row["hwm"], row["day_start_equity"]) == (D(10_000),) * 3
    store.mark_pair("carry", 1, date(2026, 10, 14), T)  # set once: a second first mark changes nothing
    store.raise_pair_peak("carry", 10_500, T)
    store.raise_pair_peak("carry", 10_200, T)  # a stale writer's lower figure
    row = store.pair_state("carry")
    assert row["hwm"] == row["reference_equity"] == D(10_500)
    store.rebase_pair("carry", 9_000, T)  # a Resume after a halt: the reference only
    row = store.pair_state("carry")
    assert (row["reference_equity"], row["hwm"]) == (D(9_000), D(10_500))
    store.raise_pair_peak("carry", 9_500, T)  # above the reference, below the high-water mark
    row = store.pair_state("carry")
    assert (row["reference_equity"], row["hwm"]) == (D(9_500), D(10_500))
    store.rebase_pair("carry", 8_000, T, hwm=True)  # a book reset: both
    assert store.pair_state("carry")["hwm"] == D(8_000)


def test_the_days_start_is_written_once_a_day_or_on_a_resume(store):
    store.set_pair_state("carry", expected=None, state="flat", ts=T)
    store.mark_pair("carry", 10_000, date(2026, 10, 14), T)
    store.set_pair_day("carry", date(2026, 10, 14), 9_000, T)  # the same day: kept
    assert store.pair_state("carry")["day_start_equity"] == D(10_000)
    store.set_pair_day("carry", date(2026, 10, 15), 9_000, T + timedelta(days=1))
    row = store.pair_state("carry")
    assert (row["day_start"], row["day_start_equity"]) == (date(2026, 10, 15), D(9_000))
    store.set_pair_day("carry", date(2026, 10, 15), 9_500, T + timedelta(days=1), rebase=True)
    assert store.pair_state("carry")["day_start_equity"] == D(9_500)


def _leg_runtime(store, name, clock):
    from sleeve_fund.paper.runtime import SleeveRuntime

    rt = SleeveRuntime(store, name, now=lambda: clock[0])
    rt.paired = True  # as the Pair sets it
    rt.on_start(0.0005)
    return rt


def test_a_restart_mid_day_keeps_the_summed_peak_and_days_start(store):
    """CR P2-8 4 (Advisor M3): a leg's drawdown reference and daily-loss baseline are both legs summed. Restarted mid-day
    they come back from pair_state, not from its own row (which holds one leg): the summed book that fell 6% from
    today's start is still paused after the restart, and its peak is still the summed one."""
    clock = [T]
    for name in ("carry", "carry~2"):
        store.set_desired_state(name, "running")
    rt = _leg_runtime(store, "carry", clock)
    rt.tick(equity=5_000.0, cash=5_000.0, qty=0.0, price=60_000.0, other_equity=5_000.0)
    clock[0] = T + timedelta(minutes=1)
    rt.tick(equity=5_300.0, cash=5_300.0, qty=0.0, price=60_000.0, other_equity=5_000.0)  # a summed high: 10,300
    row = store.pair_state("carry")
    assert (row["hwm"], row["day_start_equity"]) == (D(10_300), D(10_000))
    clock[0] = T + timedelta(hours=2)
    again = _leg_runtime(store, "carry", clock)  # restarted mid-day
    again.tick(equity=4_800.0, cash=4_800.0, qty=0.0, price=60_000.0, other_equity=4_600.0)  # 9,400: -6% on the day
    assert again.peak == 10_300.0 and again._day_open == 10_000.0
    assert store.sleeve("carry").status == "paused"  # balanced: 5% daily loss, judged on the summed start


def test_with_a_fresh_journal_the_first_summed_mark_sets_every_figure(store):
    clock = [T]
    rt = _leg_runtime(store, "carry~2", clock)
    rt.tick(equity=5_000.0, cash=5_000.0, qty=0.0, price=60_000.0, other_equity=4_000.0)
    row = store.pair_state("carry")
    assert (row["state"], row["reference_equity"], row["day_start"]) == ("flat", D(9_000), T.date())


def test_a_stale_writer_from_an_earlier_trade_cant_move_a_later_one(store):
    """DA should-fix 1: trade A ends (legging -> closing -> flat) and trade B starts (flat -> legging); a slow writer
    still holding A's 'legging' finds the state it expects, but not A's trade, so it moves nothing."""
    assert store.set_pair_state("carry", expected=None, state="flat", ts=T)
    for frm, to, trade, prior in (("flat", "legging", "A", None), ("legging", "closing", "A", "A"),
                                  ("closing", "flat", None, "A"), ("flat", "legging", "B", None)):
        assert store.set_pair_state("carry", expected=frm, state=to, trade_id=trade,
                                    deadline=None if to == "flat" else T, ts=T, expected_trade_id=prior)
    before = store.events("carry", limit=50)
    assert not store.set_pair_state("carry", expected="legging", state="closing", trade_id="A", deadline=T, ts=T,
                                    expected_trade_id="A")
    row = store.pair_state("carry")
    assert (row["state"], row["trade_id"]) == ("legging", "B") and store.events("carry", limit=50) == before


@pytest.mark.parametrize("frm, to", [(None, "legging"), ("flat", "hedged"), ("flat", "closing"), ("flat", "flat"),
                                     ("hedged", "legging"), ("closing", "legging"), ("closing", "hedged"),
                                     ("legging", "legging"), ("hedged", "hedged"), ("closing", "closing")])
def test_a_move_a_pair_never_makes_is_refused_before_anything_is_written(store, frm, to):
    """DA should-fix 2: only new -> flat, flat -> legging, legging -> hedged | closing | flat, hedged -> closing | flat,
    closing -> flat."""
    before = store.events("carry", limit=50)
    with pytest.raises(ValueError, match="isn't a move"):
        store.set_pair_state("carry", expected=frm, state=to, trade_id="t", deadline=T, ts=T, expected_trade_id="t")
    assert store.pair_state("carry") is None and store.events("carry", limit=50) == before


def test_a_move_from_a_held_trade_names_it(store):
    with pytest.raises(ValueError, match="names the trade"):
        store.set_pair_state("carry", expected="legging", state="closing", trade_id="t", deadline=T, ts=T)
