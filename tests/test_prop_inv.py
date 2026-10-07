"""PROP-INV skeleton (QA, design quant-review/v2-p2/prop-inv-design.md): random paper sessions built from real code
paths, with the money invariants checked after every step.

A run is 1-3 steps on one strategy (ping_pong, perp or spot, fees on). Each step is a recorded session of random price
legs (an optional 0-minute gap leg, which on a perp can liquidate) replayed through the paper runtime into the same
store, eight hours after the step before and starting ten minutes before a funding settlement (07:50, 15:50, 23:50
UTC), so a perp holding a position pays funding in it. The runtime stops and a new one starts on the same journal, so
each step after the first is a restart.

Invariants on 1709cd9 (design items 1-5; 6-8 switch on with CASH-ONE-PATH, NIGHTLY-RECON and DA-9):
  I1 equity = journal cash + position at the mark, at the last equity mark, and the mark's cash equals the journal's
     cash at that time: under 1 cent, or until CASH-ONE-PATH under half a cent a fill (HALF_CENT_A_FILL, HoQA ruling
     (a)): the engine books each fill's cash to the cent, the journal unrounded. The largest drift a run saw is
     reported as a warning; test_prop_inv_i1_strict_on_the_soak_examples keeps the strict form (strict xfail)
  I2 position = signed sum of fills (exact, in Decimal) = the journal's position = the mark's position
  I3 every fill carries one fee (a number, >= 0) and no fill is booked twice (trade_id unique per strategy)
  I4 each funding settlement is charged at most once per strategy and kind, and its amount is the schedule's: the
     harness stub (_funding_stub, rows shaped like store.funding: ts, amount, kind) works out every settlement the
     position was held at from the recorded prices and the journal's fills. The recorded perp is simulated, so every
     settlement is charged the baseline rate. The DA's source replaces the stub when CASH-ONE-PATH lands.
  I5 kill and restart at minute k (HoQA ruling, 7 Oct): run A goes straight through a session; run B is the same
     feed killed just after minute k's candle closes and restarted on its journal (the paper node's restart
     balances). Before k the fills are identical. After k every exit and stop fill A makes while closing the position
     held at k is in B with the same qty and a price within SLIP; every entry B makes is one of A's, at the same
     time, with a price within SLIP and a qty to a cent of notional (B may skip an entry while it warms up again;
     after a restart the venue holds the position at the price it was put back at, not the journal's entry, so an
     entry can size a few lots apart); no fill is booked twice. When B skipped no entry, its final cash and position
     (as notional) equal A's to the cent. Compared on money, quantity and status columns only (DA 20:35: no ids,
     created_at/updated_at, order_timings, heartbeats, feed_seen or event timestamps). Two findings, open with HoQA:
     R-I5-1 a restart after a stop loses the exit lock (strict xfail, ..._while_an_exit_lock_holds; the green
     property leaves the entries after k unchecked when the lock holds at k); R-I5-2 a kill inside a candle loses
     that candle's decision, so a close-based exit lands a candle later (not drawn).

Known failures (strict xfail, no shrinking in CI): the liquidation-gap runs (strict I1: cent drift, IR-1), the strict
I1 on the nightly soak's two shrunk examples, and R-I5-1 (more models in test_r_i5_1_exit_lock.py).

Hypothesis (pinned dev dependency, HoE 7 Oct): profile "ci" is derandomized with a bounded max_examples, so a PR run is
reproducible; "nightly" draws fresh seeds. Select with HYPOTHESIS_PROFILE. A failure prints the shrunk step list.
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from hypothesis import HealthCheck, Phase, event, given, settings
from hypothesis import strategies as st
from hypothesis.errors import InvalidArgument

from paper_scenarios import NAME, PARAMS, replay_into
from sleeve_fund import markets
from sleeve_fund.store import FINISHED_ORDER_STATUSES, Store, replay_book
from test_long_short import _meta

settings.register_profile("ci", derandomize=True, report_multiple_bugs=False, max_examples=15, deadline=None,
                          print_blob=True,
                          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
settings.register_profile("nightly", max_examples=500, report_multiple_bugs=False, deadline=None, print_blob=True,
                          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))

START = 10_000.0
CENT = 0.01  # "to the cent": strictly less than one cent apart
# I1's bound until CASH-ONE-PATH (HoQA 7 Oct, ruling (a)): the engine books each fill's cash to the cent (a margin
# account's realised P&L rounded half away from zero), the journal unrounded, so they drift by up to half a cent a
# fill. Both nightly failures were traced to exactly that (prop-inv/revision-2.md). Strict (I1_STRICT, CENT) again,
# and this bound removed, once money.fill_cash lands: test_prop_inv_i1_strict_on_the_soak_examples then XPASSes.
HALF_CENT_A_FILL = 0.005
I1_STRICT = False
DRIFT = {"cents": 0.0, "fills": 0, "where": ""}  # the largest I1 drift seen in this run, reported at the end
SLIP = 0.001  # an exit after a restart fills within 0.1% of where it filled straight through
SPOT = {k: v for k, v in PARAMS.items() if k not in ("market", "allow_short")}
HOUR0 = 7 + 50 / 60  # step i starts at HOUR0 + 8 i hours after test_replay.START: ten minutes before a settlement
_TICK = timedelta(microseconds=1)

def _event(text):
    """hypothesis.event, which only a @given test may call; the plain cells run the same code without it."""
    try:
        event(text)
    except InvalidArgument:
        pass


# --- the steps -----------------------------------------------------------------------------------------------

leg = st.tuples(st.integers(2, 10), st.floats(-0.03, 0.03, allow_nan=False).map(lambda m: round(m, 4)))
gap = st.sampled_from([0.6, -0.4])  # through a short's / a long's stop and liquidation price


def _run_of(gaps: bool):
    step = st.fixed_dictionaries({
        "legs": st.lists(leg, min_size=1, max_size=3),
        "gap": gap if gaps else st.none(),
    })
    return st.fixed_dictionaries({"perp": st.booleans(), "steps": st.lists(step, min_size=1, max_size=3)})


def _legs(s):
    legs = [(5, 0.0), *s["legs"]]  # five flat minutes first: the strategy warms up on a fresh runtime
    return legs + [(0, s["gap"]), (3, 0.0)] if s["gap"] is not None else legs


def _restart_meta(store, params, i):
    """The recorded session's header as the paper node opens a restart (sleeve_fund/paper/node.py): from the journal,
    a perp's margin account with the cash it had when the position opened, a spot account with its cash and coins."""
    meta = _meta(START, params)
    if i == 0:
        return meta
    book = store.journal_book(NAME, START)
    if params.get("market") == "perp":
        meta["balances"] = [f"{book['cash'] + book['qty'] * (book['entry_px'] or 0.0):.2f} USD"]
    else:
        meta["balances"] = [f"{book['cash']:.2f} USD"] + ([f"{book['qty']:.8f} BTC"] if book["qty"] > 0 else [])
    return meta


def _prices(legs, px):
    """The trade price each second, as test_long_short._record makes them (a 0-minute leg is one trade)."""
    out = []
    for minutes, move in legs:
        step = (1 + move) ** (1 / (minutes * 60)) if minutes else 1 + move
        for _ in range(minutes * 60 or 1):
            px *= step
            out.append(px)
    return out


def _start_ns(hours):
    import test_replay

    return test_replay.START + int(hours * 3600) * 1_000_000_000


def _record_prices(path, meta, prices, hours, first=0, at=None, sizes=None):
    """test_long_short._record on a given price path, from its tick `first` on: tick s is at `hours` after
    test_replay.START plus at[s] seconds (s by default), with s's own trade id and aggressor, so a session cut at
    tick k and resumed there is the same feed as the session straight through. sizes[s]: the trade's size (0.05)."""
    from nautilus_trader.model import AggressorSide, Price, Quantity, QuoteTick, TradeId, TradeTick

    from sleeve_fund.paper.recorder import Recorder
    from sleeve_fund.venues import venue

    inst = venue("KRAKEN").instrument("BTC", "USD", price_precision=1)
    rec = Recorder(path)
    rec.meta = meta
    rec.start(inst)
    base = _start_ns(hours)
    for s in range(first, len(prices)):
        px, t = prices[s], base + (s if at is None else at[s]) * 1_000_000_000
        rec.quote(QuoteTick(inst.id, Price(px - 0.5, 1), Price(px + 0.5, 1), Quantity(1.0, 8), Quantity(1.0, 8),
                            t, t + 1000))
        rec.trade(TradeTick(inst.id, Price(px, 1), Quantity(0.05 if sizes is None else sizes[s], 8),
                            AggressorSide.BUY if s % 2 else AggressorSide.SELL, TradeId(str(s)), t + 2000, t + 3000))
    rec.close()


# --- the funding stub (I4) -----------------------------------------------------------------------------------

def _at(ns):
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)


def _aware(ts):
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _funding_stub(sessions, fills, params):
    """The funding a perp owes, as rows shaped like store.funding (ts, amount, kind), oldest first: at every 00:00,
    08:00 and 16:00 from the first recorded session's start to the last one's end, the position held at it (the
    journal's fills before it; a fill at the settlement instant is after it) pays |qty| x price x the baseline rate,
    whichever side is held (a simulated perp has no venue rates: Advisor 6 Oct, D9). The price is the last trade
    before it, or for a settlement between two sessions (the process was down) the next session's first trade, which
    is when the restarted process books it. `sessions` is [(start ns, prices)], oldest first. A stand-in for the DA's
    funding source; swapped for it when CASH-ONE-PATH lands."""
    terms = markets.terms(params)
    if terms is None or not sessions:
        return []
    rate = markets.baseline_rate(terms)
    fills = sorted(fills, key=lambda f: (_aware(f["ts"]), f["id"]))
    first, end = _at(sessions[0][0]), _at(sessions[-1][0]) + timedelta(seconds=len(sessions[-1][1]) - 1)
    rows, ts = [], first.replace(minute=0, second=0, microsecond=0, hour=first.hour - first.hour % 8)
    while ts <= end:
        if ts > first:
            held = sum((Decimal(repr(f["qty"])) * (1 if f["side"] == "BUY" else -1)
                        for f in fills if _aware(f["ts"]) < ts), Decimal(0))
            px = None
            for start, prices in sessions:
                s = int((ts - _at(start)).total_seconds())
                if 0 < s <= len(prices) - 1 or (s <= 0 and px is None):
                    px = prices[s - 1] if s > 0 else prices[0]
                    break
            if abs(held) >= Decimal("0.000000005") and px is not None:
                rows.append((ts, -abs(float(held)) * round(px, 1) * rate, "baseline"))
        ts += timedelta(hours=8)
    return rows


# --- the invariants ------------------------------------------------------------------------------------------

def _check(store, where, expected_funding=None):
    fills = list(reversed(store.fills(NAME, limit=100_000)))
    last = store.last_equity(NAME)
    if last is None:
        return
    # I1: at the last mark, from the fills, funding and insurance booked up to it
    upto = [f for f in fills if f["ts"] <= last["ts"]]
    book = replay_book(upto, START, store.funding_total(NAME, before=last["ts"] + _TICK),
                       store.insurance_total(NAME, before=last["ts"] + _TICK))
    marked = book["cash"] + book["qty"] * last["price"]
    drift = max(abs(book["cash"] - last["cash"]), abs(marked - last["equity"]))
    if not I1_STRICT and drift * 100 > DRIFT["cents"]:
        DRIFT.update(cents=round(drift * 100, 4), fills=len(upto), where=where)
    _event(f"I1 drift: {'under 0.5' if drift < 0.005 else 'under 1' if drift < CENT else '1 or more'} cent")
    bound = CENT if I1_STRICT else max(CENT, HALF_CENT_A_FILL * len(upto))
    assert abs(book["cash"] - last["cash"]) < bound, (where, "I1 cash", book["cash"], last["cash"], len(upto))
    assert abs(marked - last["equity"]) < bound, (where, "I1 equity", marked, last["equity"], len(upto))
    # I2
    signed = sum((Decimal(repr(f["qty"])) * (1 if f["side"] == "BUY" else -1) for f in upto), Decimal(0))
    assert abs(Decimal(repr(book["qty"])) - signed) < Decimal("1e-10"), (where, "I2 journal", signed, book["qty"])
    assert abs(Decimal(repr(last["qty"])) - signed) < Decimal("1e-10"), (where, "I2 mark", signed, last["qty"])
    # I3
    assert all(isinstance(f["fee"], (int, float, Decimal)) and f["fee"] >= 0 for f in fills), (where, "I3 fee")
    ids = [f["trade_id"] for f in fills]
    assert len(ids) == len(set(ids)), (where, "I3 a fill booked twice", sorted(i for i in ids if ids.count(i) > 1))
    # I4: at most once per settlement and kind, and (with the stub) the schedule's amount at the schedule's times
    rows = sorted(((_aware(r["ts"]), r["amount"], r["kind"]) for r in store.funding(NAME, limit=100_000)),
                  key=lambda r: (r[0], r[2]))
    keys = [(ts, kind) for ts, _, kind in rows if kind in ("settled", "baseline")]
    assert len(keys) == len(set(keys)), (where, "I4 a settlement charged twice", rows)
    if expected_funding is not None:
        net: dict = {}
        for ts, amount, _ in rows:  # a true-up or reversal corrects its own settlement: compared net per settlement
            net[ts] = net.get(ts, 0.0) + amount
        want = {ts: amount for ts, amount, _ in expected_funding if ts <= _aware(last["ts"])}
        got = {ts: amount for ts, amount in net.items() if ts <= _aware(last["ts"]) and abs(amount) >= 1e-9}
        assert got.keys() == want.keys(), (where, "I4 settlements charged", sorted(got), sorted(want))
        _event(f"I4 funding settlements charged: {min(len(want), 2)}{'+' if len(want) >= 2 else ''}")
        bad = {ts: (got[ts], want[ts]) for ts in want if abs(got[ts] - want[ts]) > 1e-6}
        assert bad == {}, (where, "I4 funding amount", bad)


def _session(r):
    params = dict(PARAMS) if r["perp"] else dict(SPOT)
    store = Store.in_memory()
    px, sessions = 60_000.0, []
    with tempfile.TemporaryDirectory() as tmp:
        for i, s in enumerate(r["steps"]):
            legs, hours = _legs(s), HOUR0 + 8 * i
            prices = _prices(legs, px)
            path = pathlib.Path(tmp) / f"s{i}.jsonl.gz"
            _record_prices(path, _restart_meta(store, params, i), prices, hours)
            replay_into(store, path)
            sessions.append((_start_ns(hours), prices))
            _check(store, f"step {i}", _funding_stub(sessions, store.fills(NAME, limit=100_000), params))
            px = prices[-1]


@given(_run_of(gaps=False))
def test_prop_inv_money_invariants_hold_after_every_step(r):
    """Green on main 1709cd9: I1-I4 after every step, restarts and funding settlements included."""
    _session(r)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="FOLLOW-UP (HoQA 7 Oct), two causes. (1) I1 cent "
                   "drift: the engine books each fill's cash in whole cents, the journal unrounded, so marks drift up "
                   "to half a cent per fill (spot gap: 1.35 cents after 6 fills); fixed in CASH-ONE-PATH by "
                   "money.fill_cash, used by replay_book. (2) after a liquidation the marks sit off (IR-1, P1-GLC-3)")
@settings(phases=(Phase.explicit, Phase.generate))  # a known failure: no shrinking in CI
@given(_run_of(gaps=True))
def test_prop_inv_with_liquidation_gaps(r):
    """Held to the strict I1 (under 1 cent), as before the bound: it XPASSes only once both causes are fixed."""
    global I1_STRICT
    I1_STRICT = True
    try:
        _session(r)
    finally:
        I1_STRICT = False


@pytest.fixture(autouse=True, scope="module")
def _largest_drift():
    """Reports the largest I1 drift the module's runs saw, as a warning (shown in every CI log)."""
    yield
    if DRIFT["cents"]:
        import warnings

        warnings.warn(f"PROP-INV I1 largest drift {DRIFT['cents']} cents over {DRIFT['fills']} fills "
                      f"({DRIFT['where']}); bound {HALF_CENT_A_FILL * 100} cent a fill until CASH-ONE-PATH",
                      stacklevel=1)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="FOLLOW-UP until CASH-ONE-PATH money.fill_cash: I1 under "
                   "1 cent on the nightly soak's shrunk examples (1.15 and 1.20 cents, the engine's per-fill cent "
                   "rounding). XPASS once it lands: then I1 goes back to under 1 cent and HALF_CENT_A_FILL goes")
@pytest.mark.parametrize("soak", ["every_step", "kill_A"])
def test_prop_inv_i1_strict_on_the_soak_examples(soak, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "I1_STRICT", True)
    if soak == "every_step":
        _session({"perp": True, "steps": [{"legs": [(2, 0.0119)], "gap": None}, {"legs": [(4, 0.0234)], "gap": None}]})
    else:
        _kill_and_restart(True, [(5, 0.0), (2, 0.012), (2, -0.012), (2, -0.025)], 6, through_a_lock=False)


# --- I5: kill and restart at minute k ------------------------------------------------------------------------

EXITS = ("exit", "stop_loss", "liquidation")


def _fills_with_intent(store):
    intent = {o["order_id"]: o["intent"] for o in store.orders(NAME, limit=100_000)}
    return [{**f, "ts": _aware(f["ts"]), "intent": intent.get(f["order_id"])}
            for f in sorted(store.fills(NAME, limit=100_000), key=lambda f: (_aware(f["ts"]), f["id"]))]


def _money(f):
    """A fill's money, quantity and status columns (no ids, no wall-clock stamps)."""
    return (f["ts"], f["side"], f["intent"], f["qty"], f["price"], f["fee"])


def _orders_before(store, cut):
    """Orders decided before the kill, on their money, quantity and status columns."""
    return sorted((_aware(o["ts"]), o["side"], o["order_type"], o["intent"], o["qty"], o["status"], o["filled_qty"],
                   o["avg_px"], o["fee"]) for o in store.orders(NAME, limit=100_000) if _aware(o["ts"]) < cut)


def _same_fill(a, b, cent=False):
    """Same side and intent, a price within SLIP, and the same qty: exactly, or (cent) to a cent of notional."""
    qty = abs(a["qty"] - b["qty"]) * a["price"] < CENT if cent else abs(a["qty"] - b["qty"]) < 1e-12
    return a["side"] == b["side"] and a["intent"] == b["intent"] and qty and \
        abs(b["price"] - a["price"]) <= SLIP * a["price"]


# moves big enough to take ping_pong past its 1% rise, its 0.5% dip and its 2% stop, so the position held at k is closed
# after k in most runs (a derandomized draw of `leg` stays near 0 and rarely closes anything)
kill_leg = st.tuples(st.integers(2, 6), st.sampled_from([0.012, -0.025, 0.025, -0.008, 0.0, 0.006, -0.012]))


def _locked_at(store, cut):
    """Whether the journal at the kill shows a stop, target or liquidation after its last entry: the exit lock the
    run holds then (base.py _exit_lock: no re-entry until the signal moves off the side it closed)."""
    from sleeve_fund.strategies.base import LOCKING_INTENTS

    for o in sorted(store.orders(NAME, limit=100_000), key=lambda o: _aware(o["ts"]), reverse=True):
        if _aware(o["ts"]) >= cut or o["filled_qty"] <= 0:
            continue
        if o["intent"] == "entry":
            return False
        if o["intent"] in LOCKING_INTENTS:
            return True
    return False


@given(st.fixed_dictionaries({"perp": st.booleans(), "legs": st.lists(kill_leg, min_size=2, max_size=4),
                              "k": st.floats(0.05, 0.95)}))
def test_prop_inv_kill_and_restart_at_minute_k(r):
    """Green on main 1709cd9, except a kill while an exit lock holds: there the entries after k aren't compared
    (R-I5-1, test_prop_inv_kill_and_restart_while_an_exit_lock_holds)."""
    legs = [(5, 0.0), *r["legs"]]
    minutes = sum(m for m, _ in legs)
    _kill_and_restart(r["perp"], legs, min(max(6, round(r["k"] * minutes)), minutes - 1), through_a_lock=False)


@settings(phases=(Phase.explicit, Phase.generate))  # a known failure: no shrinking in CI
@given(st.fixed_dictionaries({"perp": st.booleans(), "legs": st.lists(kill_leg, min_size=0, max_size=2),
                              "k": st.floats(0.0, 1.0)}))
def test_prop_inv_kill_and_restart_while_an_exit_lock_holds(r):
    """Two 2.5% falls take the long through its 2% stop at 07:56; B is killed after it and a 1.2% fall follows."""
    legs = [(5, 0.0), (2, -0.025), (2, -0.025), *r["legs"], (3, -0.012)]
    minutes = sum(m for m, _ in legs)
    _kill_and_restart(r["perp"], legs, 7 + round(r["k"] * (minutes - 4 - 7)), through_a_lock=True)


def _kill_and_restart(perp, legs, k, through_a_lock):
    params = dict(PARAMS) if perp else dict(SPOT)
    prices = _prices(legs, 60_000.0)
    # killed just after minute k's candle closed (its first trade seen): a kill inside a candle loses that candle's
    # decision, so a close-based exit comes a candle later (R-I5-2, open with HoQA)
    cut = _at(_start_ns(HOUR0)) + timedelta(minutes=k, seconds=1)
    a, b = Store.in_memory(), Store.in_memory()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        _record_prices(tmp / "a.jsonl.gz", _meta(START, params), prices, HOUR0)
        replay_into(a, tmp / "a.jsonl.gz")
        _record_prices(tmp / "b0.jsonl.gz", _meta(START, params), prices[:k * 60 + 1], HOUR0)
        replay_into(b, tmp / "b0.jsonl.gz")
        _check(b, "B killed at k")
        before_b = _orders_before(b, cut)
        locked = _locked_at(b, cut)
        _record_prices(tmp / "b1.jsonl.gz", _restart_meta(b, params, 1), prices, HOUR0, first=k * 60 + 1)
        replay_into(b, tmp / "b1.jsonl.gz")
    _event(f"I5 exit lock at k: {'yes' if locked else 'no'}")
    stub = _funding_stub([(_start_ns(HOUR0), prices)], a.fills(NAME, limit=100_000), params)
    _check(a, "A", stub)
    _check(b, "B restarted", _funding_stub([(_start_ns(HOUR0), prices)], b.fills(NAME, limit=100_000), params))

    fa, fb = _fills_with_intent(a), _fills_with_intent(b)
    pre_a, pre_b = [_money(f) for f in fa if f["ts"] < cut], [_money(f) for f in fb if f["ts"] < cut]
    assert pre_a == pre_b, ("I5 fills before k", k, pre_a, pre_b)
    # orders decided before k: the same orders; one finished when B was killed finished the same way in A
    oa, ob = _orders_before(a, cut), _orders_before(b, cut)
    assert [o[:5] for o in oa] == [o[:5] for o in before_b] == [o[:5] for o in ob], ("I5 orders before k", k, oa, ob)
    done = [(x, y) for x, y in zip(oa, before_b) if y[5] in FINISHED_ORDER_STATUSES]
    assert all(x == y for x, y in done), ("I5 an order finished before k differs", k, [d for d in done if d[0] != d[1]])
    post_a, post_b = [f for f in fa if f["ts"] >= cut], [f for f in fb if f["ts"] >= cut]

    if locked and not through_a_lock:
        return  # R-I5-1: B is flat at k, so nothing below is checked but the entries, which a lost lock changes
    # every entry B makes after k is one of A's
    unmatched, left = [], [f for f in post_a if f["intent"] == "entry"]
    for f in (f for f in post_b if f["intent"] == "entry"):
        hit = next((g for g in left if g["ts"] == f["ts"] and _same_fill(g, f, cent=True)), None)
        if hit is None:
            unmatched.append(_money(f))
        else:
            left.remove(hit)
    assert unmatched == [], ("I5 an entry after the restart that A never made", k, unmatched)
    skipped = bool(left)
    _event("I5 B skipped an entry" if skipped else "I5 B skipped no entry")

    # every exit or stop A makes while closing the position held at k is in B
    held = sum(f["qty"] * (1 if f["side"] == "BUY" else -1) for f in fa if f["ts"] < cut)
    closing = []
    for f in post_a:
        if abs(held) < 1e-9:
            break
        if f["intent"] in EXITS:
            closing.append(f)
        held += f["qty"] * (1 if f["side"] == "BUY" else -1)
    _event(f"I5 A closes the position held at k after k: {'yes' if closing else 'no'}")
    pool = [f for f in post_b if f["intent"] in EXITS]
    missing = []
    for f in closing:
        hit = next((g for g in pool if _same_fill(f, g)), None)
        if hit is None:
            missing.append(_money(f))
        else:
            pool.remove(hit)
    assert missing == [], ("I5 an exit A made that B lost after the restart", k, missing, [_money(g) for g in post_b])

    if not skipped:
        ja, jb = a.journal_book(NAME, START), b.journal_book(NAME, START)
        assert abs(ja["cash"] - jb["cash"]) < CENT and abs(ja["qty"] - jb["qty"]) * prices[-1] < CENT, \
            ("I5 B skipped no entry but ends elsewhere", k, ja, jb, [_money(f) for f in post_a],
             [_money(f) for f in post_b])
