"""P2-2 portfolio limits (Platform Engineer 1): strict xfails written by QA BEFORE the build. PE1 makes each one pass
and removes its mark. Pin v1 to the Advisor's ruling, NOT to the spec's trim text.

Rulings (every trading-behaviour expectation cites a tag; advisor-rulings.md line "P2-2 portfolio limits (6 Oct ~21:55)"):
- [PM] PM 6 Oct 14:17: gross 1.5x, net per instrument 0.5x, margin 50%, open risk 5%, drawdown halt 15%, daily pause 3%.
- [R-TRIM] v1 TRIMS to the max qty that holds all four limits (step-rounded down; below the venue minimum = refused),
  first come under a lock with a journaled sequence number. Bar-wide pro rata comes with the gate service.
- [R-BIND] A limit binds only if the entry RAISES that figure AND post-trade > limit (= the limit passes, inclusive).
  Mark drift forces nothing. A net-reducing entry passes the net check but is still checked on the other three.
- [R-BOOK] Book = whole-fund marked equity (all strategies + unallocated cash), never "running only"; the HWM resets only
  on a book reset; cash flows adjust the HWM and the day-start. A single-strategy backtest's book = strategy equity /
  allocation share (others flat, labelled), never its own equity alone.
- [R-MARGIN] Margin = posted isolated margin (notional / leverage); spot = full notional. [R-NET] Net by underlying
  across venues and contract types.
- [R-RISK] Open risk: existing positions from the MARK (+ exit fee + stop slippage), floored at 0 per position; proposed
  = qty x (expected fill - stop) + both fees + slippage; a definition with a stop counts its planned stop; stopless (no
  stop rule) = notional x max(10%, 3 x ATR) (15:30 rule). The 40% share is OUT of v1.
- [R-HALT] The 15% halt flattens ALL (exit path, never gated), PM-only clear; Resume re-bases the halt reference to the
  book at resume (true HWM kept in the record). The 3% pause blocks entries, keeps positions and stops, auto-clears at
  00:00 UTC; each clears only by its own condition.
- [R-FAIL] Fail closed: lock/DB error = no entry + event; a book mark older than 60 s = "portfolio_state_stale"; exits
  never go to the gate; reservations are released on fill/reject/cancel and TTL-swept with an alert.
- [R-OUT] Whole-book net 1.0x and the correlated-cluster net 1.0x are OUT of v1 and must land before a second
  underlying trades on paper: pinned below as xfails with their own tag.
- [SPEC] v2/phase2-spec.md P2-2 Done-when: each limit binding in turn; pro-rata trimming with id tie-breaks; the drop
  below the venue minimum; the same result in backtest and paper; a gate timeout gives no entry and an event, exits
  pass; the halt and pause act on all strategies; every decision journaled.

ASSUMED INTERFACES (adapt the names, never the assertions):
- sleeve_fund.risk.PortfolioProfile(gross=1.5, net_instrument=0.5, margin=0.5, open_risk=0.05, drawdown=0.15,
  daily_loss=0.03): defaults are the PM numbers; fields are floats.
- sleeve_fund.portfolio.book: book_equity(strategy_equities: list[float], unallocated: float) -> float;
  backtest_book(strategy_equity, allocation_share) -> (book: float, label: str) with "others flat" in the label.
- sleeve_fund.portfolio.gate:
  Position(strategy_id, instrument, venue, side +1/-1, qty Decimal, mark, leverage (None = spot), stop_price | None,
  atr_frac), PortfolioState(equity, positions, hwm, day_start_equity, mark_ts: datetime, halt_reference = None),
  Intent(strategy_id, instrument, venue, side, qty Decimal, price, leverage, stop_frac | None, atr_frac, lot, min_qty,
  fee_rate=0.0, slippage=0.0, reduce_only=False, risk_budget=0.0),
  Gate(profile, state_provider: callable -> PortfolioState, clock: callable -> datetime, mode="paper"|"backtest"):
    .decide(intent) -> Decision(outcome "approved"|"trimmed"|"rejected", approved_qty Decimal, requested_qty,
       limit_hit | None, reason, numbers {"limit", "post"}, seq int, id, event | None); also appended to .journal.
    .release(decision, why) frees its reservation; .sweep() -> list of alert dicts {"kind": ...} (RESERVATION TTL =
       gate.RESERVATION_TTL_SECONDS); .decide_bar(list[Intent]) -> list[Decision] (gate service, later).
  evaluate(state, profile, now) -> Verdict(halt, pause, flatten, reasons); resume(state) -> PortfolioState with
  halt_reference re-based to the equity at resume and hwm kept; apply_cash_flow(state, amount) -> PortfolioState;
  next_utc_midnight(now) -> datetime.
Hand calculations: book 10,000, every mark 100, lot and venue minimum 0.01, no fees or slippage; each scenario is built
so that ONLY the limit named binds (tight 0.1% stops keep open risk negligible, 5x leverage keeps margin small).
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

import pytest

NOW = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)
BOOK = 10_000.0
TAG = "P2-2 portfolio limits (PE1)"
TAG_LATER = "P2-2 gate service (later, Advisor 21:55)"
TAG_OUT = "P2-2b: before a second underlying trades on paper (Advisor 21:55)"


import dataclasses
import importlib
import inspect


class _Strict:
    """Every interface lookup goes through here: a missing module, name or signature is AssertionError("not built: ..."),
    never an ImportError, AttributeError or TypeError that a strict xfail(raises=AssertionError) would let through
    (QA house rule: once PE1 builds with different names, a crash must not read as 'still xfail')."""

    def __init__(self, obj, name):
        object.__setattr__(self, "_o", obj)
        object.__setattr__(self, "_n", name)

    def __getattr__(self, a):
        o, n = object.__getattribute__(self, "_o"), object.__getattribute__(self, "_n")
        try:
            v = getattr(o, a)
        except AttributeError:
            raise AssertionError(f"not built: {n}.{a}") from None
        return _wrap(v, f"{n}.{a}")

    def __eq__(self, other):
        return _raw(self) == _raw(other)

    def __hash__(self):
        return hash(id(object.__getattribute__(self, "_o")))


def _raw(x):
    if isinstance(x, _Strict):
        return object.__getattribute__(x, "_o")
    if isinstance(x, list):
        return [_raw(i) for i in x]
    return x


def _wrap(v, name):
    if isinstance(v, (str, bytes, int, float, bool, type(None), D, datetime, timedelta, dict)):
        return v
    if isinstance(v, (list, tuple)):
        return type(v)(_wrap(i, name) for i in v) if isinstance(v, list) else v
    if callable(v):
        def call(*args, **kw):
            args, kw = [_raw(a) for a in args], {k: _raw(x) for k, x in kw.items()}
            try:
                inspect.signature(v).bind(*args, **kw)
            except TypeError as e:
                raise AssertionError(f"not built: {name} signature: {e}") from None
            except ValueError:
                pass
            return _wrap(v(*args, **kw), f"{name}()")
        return call
    return _Strict(v, name)


def _mod(path):
    try:
        return _Strict(importlib.import_module(path), path)
    except ImportError:
        raise AssertionError(f"not built: {path}") from None


def _replace(obj, **kw):
    return _wrap(dataclasses.replace(_raw(obj), **kw), "state")


def PP():
    return _mod("sleeve_fund.risk").PortfolioProfile()


def xf(done: str, tag: str = TAG):
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason=f"{tag} Done-when: {done}")


def _gate_mod():
    return _mod("tests.portfolio_qa_adapter")


def pos(strategy, inst, side, notional, lev=5, stop=0.001, mark=100.0, stop_price="auto", atr=0.0, venue="BINANCE"):
    """An existing position of `notional` at the mark, with a stop `stop` from the mark (tight: open risk negligible)."""
    g = _gate_mod()
    sp = mark * (1 - side * stop) if stop_price == "auto" else stop_price
    return g.Position(strategy_id=strategy, instrument=inst, venue=venue, side=side,
                      qty=D(str(round(notional / mark, 8))), mark=mark, leverage=lev, stop_price=sp, atr_frac=atr)


def intent(strategy, inst, side, notional, lev=5, stop_frac=0.001, atr=0.0, venue="BINANCE", **kw):
    g = _gate_mod()
    base = dict(strategy_id=strategy, instrument=inst, venue=venue, side=side,
                qty=D(str(round(notional / 100.0, 8))), price=100.0, leverage=lev, stop_frac=stop_frac, atr_frac=atr,
                lot=D("0.01"), min_qty=D("0.01"), fee_rate=0.0, slippage=0.0, reduce_only=False)
    base.update(kw)
    return g.Intent(**base)


def state(positions, equity=BOOK, hwm=None, day_start=None, age=0.0, **kw):
    g = _gate_mod()
    return g.PortfolioState(equity=equity, positions=list(positions), hwm=hwm if hwm is not None else equity,
                            day_start_equity=day_start if day_start is not None else equity,
                            mark_ts=NOW - timedelta(seconds=age), **kw)


def gate_for(st, mode="paper", provider=None):
    g = _gate_mod()
    src = provider or (lambda: st)
    return g.Gate(PP(), lambda: _raw(src()), lambda: NOW, mode=mode)


def decide(positions, it, **skw):
    gt = gate_for(state(positions, **skw))
    return gt, gt.decide(it)


def _qty(d):
    return d.approved_qty


# ---- the profile and the book ------------------------------------------------------------------------------------

def test_the_portfolio_profile_defaults_are_the_pm_accepted_limits():
    p = PP()
    assert (p.gross, p.net_instrument, p.margin, p.open_risk, p.drawdown, p.daily_loss) == (1.5, 0.5, 0.5, 0.05, 0.15, 0.03)


def test_the_book_is_the_whole_fund_marked_equity():
    book_equity = _mod("tests.portfolio_qa_adapter").book_equity
    assert book_equity([3_000.0, 3_000.0], 4_000.0) == pytest.approx(10_000.0)  # not 6,000
    assert book_equity([], 10_000.0) == pytest.approx(10_000.0)  # nothing running: the cash is still the book


def test_a_single_strategy_backtests_book_is_its_equity_over_its_allocation_share():
    backtest_book = _mod("tests.portfolio_qa_adapter").backtest_book
    book, label = backtest_book(2_000.0, 0.2)
    assert book == pytest.approx(10_000.0) and "others flat" in label


# ---- each limit binding in turn [R-TRIM] [SPEC] ------------------------------------------------------------------

def test_gross_binds_and_trims_to_the_headroom():
    held = [pos("s1", "AAA/USDT", 1, 4_000), pos("s2", "BBB/USDT", 1, 4_000), pos("s3", "DDD/USDT", -1, 4_000)]
    gt, d = decide(held, intent("s4", "CCC/USDT", 1, 5_000))
    assert (d.outcome, d.approved_qty, d.limit_hit) == ("trimmed", D("30"), "gross")
    assert d.requested_qty == D("50") and d.numbers["limit"] == pytest.approx(15_000.0) and d.numbers["post"] == pytest.approx(17_000.0)
    assert gt.journal == [d] and d.seq == 1 and d.reason  # every decision is journaled with its numbers


def test_net_per_instrument_binds_and_trims():
    _, d = decide([pos("s1", "AAA/USDT", 1, 3_000)], intent("s2", "AAA/USDT", 1, 4_000))
    assert (d.outcome, d.approved_qty, d.limit_hit) == ("trimmed", D("20"), "net_instrument")


def test_landing_exactly_on_the_limit_passes_untouched():
    _, d = decide([pos("s1", "AAA/USDT", 1, 3_000)], intent("s2", "AAA/USDT", 1, 2_000))
    assert (d.outcome, d.approved_qty, d.limit_hit) == ("approved", D("20"), None)


def test_margin_binds_and_trims():
    held = [pos("s1", "AAA/USDT", 1, 4_000, lev=2), pos("s2", "BBB/USDT", 1, 4_000, lev=2)]
    _, d = decide(held, intent("s3", "CCC/USDT", 1, 3_000, lev=2))
    assert (d.outcome, d.approved_qty, d.limit_hit) == ("trimmed", D("20"), "margin")


@pytest.mark.parametrize("lev,want", [
    pytest.param(None, ("trimmed", D("20"), "margin"), id="spot-counts-full-notional"),
    pytest.param(2, ("approved", D("30"), None), id="perp-counts-notional-over-leverage"),
])
def test_spot_counts_full_notional_as_margin(lev, want):
    _, d = decide([pos("s1", "AAA/USDT", 1, 3_000, lev=1)], intent("s2", "CCC/USDT", 1, 3_000, lev=lev))
    assert (d.outcome, d.approved_qty, d.limit_hit) == want


def _risk_book(extra=()):
    """300 of open risk: two longs of 3,000 with stops 5% from the mark (150 each)."""
    return [pos("s1", "AAA/USDT", 1, 3_000, stop=0.05), pos("s2", "BBB/USDT", 1, 3_000, stop=0.05), *extra]


def test_open_risk_binds_and_trims():
    _, d = decide(_risk_book(), intent("s3", "CCC/USDT", 1, 4_000, stop_frac=0.10))
    assert (d.outcome, d.approved_qty, d.limit_hit) == ("trimmed", D("20"), "open_risk")


@xf("a stop already through the mark is floored at 0 risk, never a negative credit that buys headroom [R-RISK]")
def test_an_existing_stop_through_the_mark_is_floored_at_zero_not_negative():
    gapped = pos("s9", "DDD/USDT", 1, 3_000, stop_price=105.0)  # a long's stop ABOVE its mark: raw risk -150
    _, d = decide(_risk_book([gapped]), intent("s3", "CCC/USDT", 1, 4_000, stop_frac=0.10))
    assert (d.approved_qty, d.limit_hit) == (D("20"), "open_risk")  # 300 held, NOT 150: still 20, not 35


def test_existing_open_risk_is_from_the_mark():
    runner = pos("s1", "AAA/USDT", 1, 3_300, mark=110.0, stop_price=105.0)
    held = [runner, pos("s2", "BBB/USDT", 1, 3_000, stop=0.05)]
    _, d = decide(held, intent("s3", "CCC/USDT", 1, 4_000, stop_frac=0.10))
    assert (d.approved_qty, d.limit_hit) == (D("20"), "open_risk")  # 150 + 150 = 300 held; 200 of headroom -> 2,000


@pytest.mark.parametrize("kw,want", [
    pytest.param(dict(stop_frac=None, atr=0.02), D("20"), id="stopless-atr2-uses-the-10pct-floor"),
    pytest.param(dict(stop_frac=None, atr=0.05), D("13.33"), id="stopless-atr5-uses-3-atr-15pct"),
    pytest.param(dict(stop_frac=0.01), D("40"), id="a-planned-stop-counts-not-stopless"),
])
def test_stopless_and_planned_stop_open_risk(kw, want):
    _, d = decide(_risk_book(), intent("s3", "CCC/USDT", 1, 4_000, **kw))
    assert d.approved_qty == want


def test_trim_rounds_down_to_the_step_and_below_the_minimum_is_refused():
    held = [pos("s1", "AAA/USDT", 1, 4_000), pos("s2", "BBB/USDT", 1, 4_000), pos("s3", "DDD/USDT", -1, 4_000)]
    # headroom 3,000 = 30 units, lot 1: a request of 50 trims to 30; with lot 7 -> 28 (floor to the step)
    _, d = decide(held, intent("s4", "CCC/USDT", 1, 5_000, lot=D("7"), min_qty=D("7")))
    assert (d.outcome, d.approved_qty) == ("trimmed", D("28"))
    held2 = [pos("s1", "AAA/USDT", 1, 4_000), pos("s2", "BBB/USDT", 1, 4_000), pos("s3", "DDD/USDT", -1, 6_999.5)]
    _, d = decide(held2, intent("s4", "CCC/USDT", 1, 5_000))  # 0.5 of headroom = 0.005 units < the 0.01 minimum
    assert (d.outcome, d.approved_qty, d.limit_hit) == ("rejected", D("0"), "gross")


# ---- binds only when it raises the figure [R-BIND] ---------------------------------------------------------------

def test_a_net_reducing_entry_passes_the_net_check():
    _, d = decide([pos("s1", "AAA/USDT", 1, 6_000)], intent("s2", "AAA/USDT", -1, 2_000))  # net 6,000 -> 4,000
    assert (d.outcome, d.approved_qty, d.limit_hit) == ("approved", D("20"), None)


def test_a_net_reducing_entry_is_still_checked_on_the_other_limits():
    held = [pos("s1", "AAA/USDT", 1, 6_000), pos("s2", "BBB/USDT", 1, 4_500), pos("s3", "CCC/USDT", 1, 4_000)]
    _, d = decide(held, intent("s4", "AAA/USDT", -1, 2_000))
    assert (d.outcome, d.approved_qty, d.limit_hit) == ("trimmed", D("5"), "gross")


@pytest.mark.parametrize("notional,want", [
    pytest.param(8_000, ("approved", D("80"), None), id="flip-to-exactly-minus-0.5x-passes"),
    pytest.param(9_000, ("trimmed", D("80"), "net_instrument"), id="flip-past-minus-0.5x-trims-to-80"),
])
def test_a_flip_is_judged_on_the_absolute_post_trade_net(notional, want):
    _, d = decide([pos("s1", "AAA/USDT", 1, 3_000)], intent("s2", "AAA/USDT", -1, notional))
    assert (d.outcome, d.approved_qty, d.limit_hit) == want


def test_net_is_by_underlying_across_venues_and_contract_types():
    perp = pos("s1", "BTC/USDT", 1, 3_000, venue="BINANCE")
    _, d = decide([perp], intent("s2", "BTC/USD", 1, 4_000, lev=None, venue="KRAKEN"))
    assert (d.approved_qty, d.limit_hit) == (D("20"), "net_instrument")


@pytest.mark.parametrize("case", ["gross", "net", "margin"])
def test_backtest_and_paper_decide_the_same(case):
    held, it = {
        "gross": ([pos("s1", "AAA/USDT", 1, 4_000), pos("s2", "BBB/USDT", 1, 4_000), pos("s3", "DDD/USDT", -1, 4_000)],
                  intent("s4", "CCC/USDT", 1, 5_000)),
        "net": ([pos("s1", "AAA/USDT", 1, 3_000)], intent("s2", "AAA/USDT", 1, 4_000)),
        "margin": ([pos("s1", "AAA/USDT", 1, 4_000, lev=2), pos("s2", "BBB/USDT", 1, 4_000, lev=2)],
                   intent("s3", "CCC/USDT", 1, 3_000, lev=2)),
    }[case]
    got = []
    for mode in ("paper", "backtest"):
        d = gate_for(state(held), mode=mode).decide(it)
        got.append((d.outcome, d.approved_qty, d.limit_hit, d.numbers))
    assert got[0] == got[1] and got[0][0] == "trimmed"


# ---- first come under the lock, reservations [R-TRIM] [R-FAIL] ---------------------------------------------------

def _crowded():
    return [pos("s1", "AAA/USDT", 1, 4_000), pos("s2", "BBB/USDT", 1, 4_000), pos("s3", "DDD/USDT", -1, 4_000)]  # headroom 3,000


def test_first_come_reservations_and_release():
    gt = gate_for(state(_crowded()))
    d1 = gt.decide(intent("s4", "CCC/USDT", 1, 2_000))
    d2 = gt.decide(intent("s5", "EEE/USDT", 1, 2_000))
    assert (d1.outcome, d1.approved_qty, d1.seq) == ("approved", D("20"), 1)
    assert (d2.outcome, d2.approved_qty, d2.limit_hit, d2.seq) == ("trimmed", D("10"), "gross", 2)  # 1,000 left
    gt.release(d1, "cancelled")  # a cancel (or reject) frees what it held
    d3 = gt.decide(intent("s6", "FFF/USDT", 1, 2_000))
    assert (d3.outcome, d3.approved_qty) == ("approved", D("20"))  # 3,000 - d2's 1,000 = 2,000


def test_a_stale_reservation_is_swept_with_an_alert():
    g = _gate_mod()
    clock = {"now": NOW}
    gt = g.Gate(PP(), lambda: _raw(_replace(state(_crowded()), mark_ts=clock["now"])), lambda: clock["now"])  # fresh mark
    assert gt.decide(intent("s4", "CCC/USDT", 1, 3_000)).approved_qty == D("30")  # takes ALL the headroom, never fills
    assert gt.decide(intent("s5", "EEE/USDT", 1, 1_000)).outcome == "rejected"  # nothing left
    clock["now"] = NOW + timedelta(seconds=g.RESERVATION_TTL_SECONDS + 1)
    alerts = gt.sweep()
    assert [a["kind"] for a in alerts] == ["reservation_expired"]
    assert gt.decide(intent("s5", "EEE/USDT", 1, 1_000)).outcome == "approved"


# ---- fail closed; exits never gated [R-FAIL] [SPEC] --------------------------------------------------------------

def test_a_stale_book_mark_refuses_the_entry():
    for age, want in ((61.0, "rejected"), (60.0, "approved")):
        gt = gate_for(state([], age=age))
        d = gt.decide(intent("s1", "AAA/USDT", 1, 1_000))
        assert d.outcome == want, (age, d)
        if want == "rejected":
            assert d.limit_hit == "portfolio_state_stale"


def test_a_stale_mark_alerts_once_and_never_blocks_a_stop():
    held = [pos("s1", "AAA/USDT", 1, 6_000)]
    gt = gate_for(state(held, age=300.0))
    first = gt.decide(intent("s2", "BBB/USDT", 1, 1_000))
    second = gt.decide(intent("s3", "CCC/USDT", 1, 1_000))
    assert (first.outcome, first.limit_hit) == ("rejected", "portfolio_state_stale") and first.event is not None  # alert
    assert second.outcome == "rejected" and second.event is None  # not an alert per blocked entry
    stop = gt.decide(intent("s1", "AAA/USDT", -1, 6_000, reduce_only=True))  # the position's stop trips mid-stale
    assert (stop.outcome, stop.approved_qty) == ("approved", D("60")) and stop.limit_hit is None
    close = gt.decide(intent("s1", "AAA/USDT", -1, 3_000, reduce_only=True))  # and a partial close
    assert close.outcome == "approved"
    fresh = gate_for(state(held, age=5.0)).decide(intent("s2", "BBB/USDT", 1, 1_000))
    assert fresh.outcome == "approved"  # the block lifts as soon as the mark is fresh again


@pytest.mark.parametrize("why", ["halt", "pause", "stale"])
def test_a_trimmed_entry_still_goes_through_the_entry_block(why):
    held = _crowded()  # 3,000 of gross headroom: a 5,000 entry would be TRIMMED to 30 on a healthy book
    healthy = gate_for(state(held)).decide(intent("s4", "CCC/USDT", 1, 5_000))
    assert (healthy.outcome, healthy.approved_qty) == ("trimmed", D("30"))  # control: it trims when nothing blocks
    blocked = {"halt": state(held, equity=8_400.0, hwm=10_000.0), "pause": state(held, equity=9_650.0, day_start=10_000.0),
               "stale": state(held, age=300.0)}[why]
    d = gate_for(blocked).decide(intent("s4", "CCC/USDT", 1, 5_000))
    assert (d.outcome, d.approved_qty) == ("rejected", D("0"))  # not "trimmed" to some smaller entry
    assert why in (d.limit_hit or "").lower() or why in d.reason.lower()


def test_a_state_error_gives_no_entry_and_an_event():
    def boom():
        raise RuntimeError("db down")
    d = gate_for(None, provider=boom).decide(intent("s1", "AAA/USDT", 1, 1_000))
    assert d.outcome == "rejected" and d.approved_qty == D("0") and d.event is not None


@pytest.mark.parametrize("why", ["stale", "error", "halt", "pause"])
def test_exits_always_pass(why):
    st = {"stale": state([pos("s1", "AAA/USDT", 1, 6_000)], age=300.0),
          "halt": state([pos("s1", "AAA/USDT", 1, 6_000)], equity=8_000.0, hwm=10_000.0),
          "pause": state([pos("s1", "AAA/USDT", 1, 6_000)], equity=9_600.0, day_start=10_000.0),
          "error": None}[why]
    def provider():
        if st is None:
            raise RuntimeError("db down")
        return st
    d = gate_for(st, provider=provider).decide(intent("s1", "AAA/USDT", -1, 6_000, reduce_only=True))
    assert (d.outcome, d.approved_qty) == ("approved", D("60"))


# ---- drawdown halt and daily pause [R-HALT] ----------------------------------------------------------------------

def test_the_drawdown_halt_flattens_all():
    g = _gate_mod()
    held = [pos("s1", "AAA/USDT", 1, 3_000), pos("s2", "BBB/USDT", -1, 3_000)]
    v = g.evaluate(state(held, equity=8_400.0, hwm=10_000.0), PP(), NOW)
    assert v.halt and v.flatten and not g.evaluate(state(held, equity=8_600.0, hwm=10_000.0), PP(), NOW).halt
    d = gate_for(state(held, equity=8_400.0, hwm=10_000.0)).decide(intent("s3", "CCC/USDT", 1, 1_000))
    assert d.outcome == "rejected" and "halt" in (d.limit_hit or d.reason).lower()


def test_resume_rebases_the_halt_reference_and_keeps_the_true_hwm():
    g = _gate_mod()
    st = g.resume(state([], equity=8_400.0, hwm=10_000.0))
    assert st.halt_reference == pytest.approx(8_400.0) and st.hwm == pytest.approx(10_000.0)
    assert not g.evaluate(_replace(st, equity=7_200.0), PP(), NOW).halt
    assert g.evaluate(_replace(st, equity=7_100.0), PP(), NOW).halt


def test_cash_flows_adjust_the_hwm_and_the_day_start():
    g = _gate_mod()
    out = g.apply_cash_flow(state([], equity=10_000.0, hwm=10_000.0), -3_000.0)
    assert (out.equity, out.hwm, out.day_start_equity) == (7_000.0, 7_000.0, 7_000.0)
    assert not g.evaluate(out, PP(), NOW).halt  # not a 30% "drawdown"
    dep = g.apply_cash_flow(state([], equity=10_000.0, hwm=10_000.0), 2_000.0)
    assert (dep.equity, dep.hwm) == (12_000.0, 12_000.0)
    assert g.evaluate(_replace(dep, equity=10_100.0), PP(), NOW).halt  # 15.8% below the adjusted HWM


def test_the_daily_pause_blocks_entries_keeps_positions_and_clears_at_midnight():
    g = _gate_mod()
    held = [pos("s1", "AAA/USDT", 1, 3_000)]
    paused = state(held, equity=9_650.0, day_start=10_000.0)  # -3.5%
    v = g.evaluate(paused, PP(), NOW)
    assert v.pause and not v.halt and not v.flatten  # positions and stops stay
    d = gate_for(paused).decide(intent("s2", "BBB/USDT", 1, 1_000))
    assert d.outcome == "rejected" and "pause" in (d.limit_hit or d.reason).lower()
    assert g.evaluate(_replace(paused, day_start_equity=9_650.0), PP(), g.next_utc_midnight(NOW)).pause is False
    assert g.next_utc_midnight(NOW) == datetime(2026, 10, 7, 0, 0, tzinfo=timezone.utc)


def test_each_halt_and_pause_clears_only_by_its_own_condition():
    g = _gate_mod()
    paused = state([], equity=9_650.0, day_start=10_000.0)
    assert g.evaluate(g.resume(paused), PP(), NOW).pause  # Resume can't clear the daily pause
    halted = state([], equity=8_400.0, hwm=10_000.0, day_start=8_400.0)
    rolled = g.evaluate(halted, PP(), g.next_utc_midnight(NOW))
    assert rolled.halt  # the 00:00 roll is not the PM's Resume


# ---- later: the gate service, and the limits that are out of v1 ----------------------------------------------------

@xf("bar-wide pro rata by risk budget across ALL of the bar's entries, ties by strategy id, rounded down [R-TRIM, gate service]", TAG_LATER)
def test_bar_wide_pro_rata_by_risk_budget_with_id_tiebreaks():
    gt = gate_for(state(_crowded()))  # headroom 3,000
    its = [intent("s1x", "CCC/USDT", 1, 4_000, risk_budget=100.0), intent("s2x", "EEE/USDT", 1, 4_000, risk_budget=100.0),
           intent("s3x", "FFF/USDT", 1, 4_000, risk_budget=200.0)]
    out = gt.decide_bar(its)
    assert [d.approved_qty for d in out] == [D("7.50"), D("7.50"), D("15.00")]  # 750 / 750 / 1,500 of the 3,000
    assert sum(d.approved_qty for d in out) * 100 <= 3_000


@xf("whole-book net directional <= 1.0x: 8,000 net long held + 4,000 asked -> trimmed to 20 [R-OUT]", TAG_OUT)
def test_whole_book_net_directional_binds():
    held = [pos("s1", "AAA/USDT", 1, 4_000), pos("s2", "BBB/USDT", 1, 4_000)]
    _, d = decide(held, intent("s3", "CCC/USDT", 1, 4_000))
    assert (d.approved_qty, d.limit_hit) == (D("20"), "net_book")


@xf("the correlated cluster (majors as one) net <= 1.0x: BTC and ETH longs count together, a short non-major does not offset them [R-OUT]", TAG_OUT)
def test_the_majors_cluster_net_binds_and_a_non_major_is_outside_it():
    held = [pos("s1", "BTC/USDT", 1, 3_500), pos("s2", "ETH/USDT", 1, 3_500), pos("s3", "XYZ/USDT", -1, 3_500)]
    _, d = decide(held, intent("s4", "SOL/USDT", 1, 4_000))
    # cluster 7,000 held + 4,000 = 11,000 > 10,000 -> 3,000 (30); were the short XYZ counted it would read 7,500 and pass
    assert (d.approved_qty, d.limit_hit) == (D("30"), "net_cluster")
