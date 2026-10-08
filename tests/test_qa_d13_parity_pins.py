"""QA Tester 2, 7 Oct: D13 parity pins for PR #178 (HoQA's list a-g of 01:07). They are plain tests that must pass on
#178 and keep passing; each was checked by mutating #178 (see README "7 Oct round"). Helpers come by import from QA's
#146 harness (`../v2-p1/hub-146-scripts/`, not edited): the hub-fed paper node (`paper`) and the research backtest.

- b/g  research and paper book the same entry and exit price (mid +- exactly the half spread used) and the same fee
- fee  the venue fee is charged on the BOOKED price (the ask for a buy, the bid for a sell), not on the mid
- c(a) a stop books at level x (1 -+ max(half spread, 0.05 %)), to 0.1 bp, spread below and above the floor
       (the point-in-time half stays with spread-pit-xfails)
- c(b) one-sided parity, Advisor D13-STOP-PARITY (7 Oct 00:20): the backtest is never better than hub paper's fill and the
       gap is at most the floor plus 7.0 bp; the low bound is NOT moved to -7.5 bp
- d    the paper harness sends quotes from its first second (ticks()), so an entry in a restart never fills without one
- e    report price vs journal price: |diff| <= 1.5 x CENT / qty + 1e-9. Derivation: the engine's charge is rounded with a
       carry (|carry| <= 0.5 cent, the rounding 0.5 cent: 1.0 cent), and the report moves charge - round(exact fee, 2)
       into the price (another 0.5 cent): 1.5 cents / qty. Checked over every fill of a 100+ fill run.
- f    an exact touch of a stop (a minute's low == the level)
"""
import os

# Repo copy of QA's D13 parity pins (quant-review/d13-xfails/test_d13_parity_pins.py, md5 c8fc9ae6). Only the imports
# are adapted: the #146 harness is the repo's tests/test_hub_146_qa.py (QA's post-178 harness, carried marks-only)
# and test_hub_path_parity is tests/test_hub_path_parity.py, both beside this file, never the project folder, so CI
# collects it. Every mark and assertion is the master's.
os.environ.setdefault("BACKTEST_ISOLATE", "0")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import test_hub_146_qa as qa  # noqa: E402
from test_hub_146_qa import (BASE, SPREAD, _probe, adverse, favourable, flat_prices, minute, minutes_of,  # noqa: E402,F401
                             paper, shape)

HALF = SPREAD / 2 / BASE  # 1 bp: below the 0.05 % floor
FLOOR = 0.0005
CENT = 0.0100001
SETUPS = [("spot-long", False, "aggressive", 1), ("perp-1x-long", True, "conservative", 1),
          ("perp-3x-long", True, "aggressive", 1), ("perp-3x-short", True, "aggressive", -1)]
IDS = [s[0] for s in SETUPS]


def _bt(prices, *, side=1, perp=False, profile="aggressive", half=HALF, capital=10_000, leave=30, stop=0.01, tp=None,
        enter=5):
    from sleeve_fund.instruments import BOOK_SHARE
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.venues import venue

    inst = venue("KRAKEN").instrument("BTC", "USD", price_precision=1)
    params = {"enter": enter, "leave": leave, "side": side, "after": 0}
    if stop:
        params["stop_loss"] = stop
    if tp:
        params["take_profit"] = tp
    if perp:
        params.update(market="perp", allow_short=True)
    m = minutes_of(prices)
    bars = m.set_axis(m.index + pd.Timedelta(minutes=1))
    bars["volume"] = 60 / BOOK_SHARE
    return run_backtest("probe", bars, inst, params=params, starting_capital=capital, risk_profile=profile,
                        bar_minutes=1, half_spread=half), params


def _orders(res):
    j = res.journal
    orders = list(j.orders_.values()) if isinstance(j.orders_, dict) else list(j.orders_)
    return sorted(orders, key=lambda o: (o["ts"], o["id"]))


def _fills(res, order_id):
    return [f for f in res.journal.fills_ if f["order_id"] == order_id]


def _seq(res):
    """(intent, side, minute, avg price, qty, fee) per order, in time order."""
    out = []
    for o in _orders(res):
        fs = _fills(res, o["order_id"])
        if not fs:
            continue
        q = sum(f["qty"] for f in fs)
        px = sum(f["qty"] * f["price"] for f in fs) / q
        t = pd.Timestamp(fs[0]["ts"]).tz_convert("UTC")
        out.append((o["intent"], o["side"], t if t == t.floor("1min") else t.ceil("1min"), px, q,
                    sum(f["fee"] for f in fs)))
    return sorted(out, key=lambda r: r[2])


def _paper_seq(run):
    by = {}
    for f in run.fills:
        q, n, fee, ts = by.get(f["order_id"], (0.0, 0.0, 0.0, f["ts"]))
        by[f["order_id"]] = (q + f["qty"], n + f["qty"] * f["price"], fee + f["fee"], ts)
    intents = {o["order_id"]: (o["intent"], o["side"]) for o in run.orders}
    out = []
    for oid, (q, n, fee, ts) in by.items():
        t = pd.Timestamp(ts).tz_convert("UTC")
        out.append((*intents[oid], t if t == t.floor("1min") else t.ceil("1min"), n / q, q, fee))
    return sorted(out, key=lambda r: r[2])


def _taker(perp, side):
    from sleeve_fund import markets
    from sleeve_fund.venues import venue

    params = {"market": "perp", "allow_short": True} if perp else {}
    return float(markets.fees_for(params, venue("KRAKEN").fees).taker)


def _stop_path(side):
    """Entry at minute 5, then 2 % against the position from 00:10:30 for ten seconds: through a 1 % stop (a jump)."""
    p = flat_prices(40).copy()
    p[630:640] = BASE * adverse(side, 0.02)
    return p


def _stop_ramp(side):
    """A smooth ramp 3 % against the position over five minutes from minute 10 (about 1 bp a second), as c5's target ramp:
    the tape passes through the stop level, so paper fills where the level traded."""
    p = flat_prices(40).copy()
    s = np.arange(600, 900)
    p[600:900] = BASE * (1 - side * 0.03 * (s - 600) / 300)
    p[900:] = p[899]
    return p


# --------------------------------------------------------------------------------------------------- b, g, fee base
@pytest.mark.parametrize("label, perp, profile, side", SETUPS, ids=IDS)
def test_research_and_paper_book_the_same_entry_and_exit_price_and_fee(label, perp, profile, side):
    """b + g: the same strategy on the same minutes, research (backtest) and the hub-fed paper node: the entry fills at
    mid +- exactly the half spread used (a buy on the ask, a sell on the bid), the market exit the same, in both, and
    each order's venue fee is the taker rate on THAT price."""
    p = flat_prices(40)
    res, _ = _bt(p, side=side, perp=perp, profile=profile)
    run = paper(p, side=side, perp=perp, profile=profile, leave=30, stop=0.01)
    bt, pp = _seq(res), _paper_seq(run)
    assert [r[0] for r in bt][:2] == [r[0] for r in pp][:2] == ["entry", "exit"], (bt, pp)
    taker = _taker(perp, side)
    for b, q in zip(bt[:2], pp[:2]):
        mid_at = {"entry": 5, "exit": 30}[b[0]]
        mid = float(p[mid_at * 60 - 1])  # the last trade before the decision bar's close
        want = mid + (qa.SPREAD / 2 if b[1] == "BUY" else -qa.SPREAD / 2)
        for who, r in (("research", b), ("paper", q)):
            assert r[3] == pytest.approx(want, abs=0.1 + 1.5 * CENT / r[4]), (who, r, want)
            assert r[5] == pytest.approx(r[4] * r[3] * taker, abs=1.5 * CENT), (who, "fee on the booked price", r)
        assert b[3] == pytest.approx(q[3], abs=0.1 + 1.5 * CENT / min(b[4], q[4])), ("research vs paper", b, q)
        assert b[5] == pytest.approx(q[5], abs=1.5 * CENT + 0.1 * b[4] * taker), ("fee, research vs paper", b, q)


@pytest.mark.parametrize("side, perp", [(1, False), (-1, True)], ids=["long", "short"])
def test_the_fee_is_charged_on_the_booked_price_not_the_mid(side, perp):
    """fee base. At a 0.1 % half spread on a 5 BTC-class book the fee on the ask and on the mid differ by a few
    dollars: the buy is charged on the ask, the sell on the bid (Advisor 20:55; HoE's 'the fee on the ask')."""
    p = flat_prices(40)
    res, _ = _bt(p, side=side, perp=perp, half=0.001, capital=50_000_000)
    taker = _taker(perp, side)
    seq = _seq(res)
    assert len(seq) >= 2 and seq[0][4] > 50, "set-up: a position big enough for the spread to show in the fee"
    for r in seq[:2]:
        assert r[5] == pytest.approx(r[4] * r[3] * taker, abs=1.5 * CENT), r
        mid = float(p[{"entry": 5, "exit": 30}[r[0]] * 60 - 1])
        on_mid = r[4] * mid * taker
        assert abs(r[5] - on_mid) > 1.0, ("the fee equals the fee on the mid: the spread is not in its base", r)


# ------------------------------------------------------------------------------------------------------------- c (a)
@pytest.mark.parametrize("half", [HALF, 0.002], ids=["half-below-floor", "half-above-floor"])
@pytest.mark.parametrize("label, perp, profile, side", SETUPS, ids=IDS)
def test_a_stop_books_at_its_level_less_max_half_spread_or_floor_to_a_tenth_of_a_basis_point(label, perp, profile, side,
                                                                                            half):
    res, _ = _bt(_stop_path(side), side=side, perp=perp, profile=profile, half=half)
    seq = _seq(res)
    stop = next(r for r in seq if r[0] == "stop_loss")
    entry = seq[0]
    level = entry[3] * (1 - side * 0.01)  # levels derive from the BOOKED entry (D13)
    want = level * (1 - side * max(half, FLOOR))
    assert stop[3] == pytest.approx(want, rel=1e-5, abs=0.1 + 1.5 * CENT / stop[4]), (stop, level, want)


# ------------------------------------------------------------------------------------------------------------- c (b)
@pytest.mark.parametrize("label, perp, profile, side", SETUPS, ids=IDS)
def test_stop_one_sided_parity_the_backtest_is_never_better_than_paper_and_within_floor_plus_7bp(label, perp, profile,
                                                                                                 side):
    """D13-STOP-PARITY (b). On a smooth ramp through the stop (about 1 bp a second) paper fills where the tape trades
    (the level); the backtest books level less the slippage floor. (A jump through the stop inside one minute is a
    different case: see the report's INFO.) Backtest never better: side x (paper - backtest) >= 0; the gap at most
    the floor plus 7 bp. The low bound is not moved to -7.5 bp."""
    p = _stop_ramp(side)
    res, _ = _bt(p, side=side, perp=perp, profile=profile)
    run = paper(p, side=side, perp=perp, profile=profile, leave=30, stop=0.01)
    bt = next(r for r in _seq(res) if r[0] == "stop_loss")
    pp = next(r for r in _paper_seq(run) if r[0] == "stop_loss")
    gap = side * (pp[3] - bt[3]) / bt[3]
    assert 0 <= gap <= FLOOR + 0.0007, (gap * 1e4, "bp", pp, bt)


@pytest.mark.parametrize("label, perp, profile, side", SETUPS, ids=IDS)
def test_c5_target_one_sided_parity_follows_the_ruling(label, perp, profile, side):
    """c5 as the Advisor rules it: the backtest target (level less the floor) is never better than paper's real fill
    and the gap is at most the floor plus 7 bp (12 bp), where c5 used 10 bp. 59109.57 / 0.9995 = 59139.14 is the
    Advisor's worked case."""
    p = flat_prices(40).copy()
    s = np.arange(600, 900)
    p[600:900] = BASE * (1 + side * 0.03 * (s - 600) / 300)
    p[900:] = p[899]
    run = paper(p, side=side, perp=perp, profile=profile, tp=0.02, leave=30)
    res, _ = _bt(p, side=side, perp=perp, profile=profile, tp=0.02, stop=0.01)
    pp = next(r for r in _paper_seq(run) if r[0] == "take_profit")
    bt = next(r for r in _seq(res) if r[0] == "take_profit")
    level = round(_seq(res)[0][3] * (1 + side * 0.02), 1)
    assert bt[3] == pytest.approx(level * (1 - side * max(HALF, FLOOR)), rel=1e-5, abs=0.1), (bt, level)
    gap = side * (pp[3] - bt[3]) / bt[3]
    assert 0 <= gap <= FLOOR + 0.0007, (gap * 1e4, "bp", pp, bt)


# ----------------------------------------------------------------------------------------------------------------- d
def test_the_paper_harness_sends_a_quote_before_every_entry():
    """d: a quote precedes the first trade second, so a restart replay never fills an entry without one (it cannot
    'loosen' by missing quotes). Pinned on the harness's own tick builder."""
    from nautilus_trader.model import QuoteTick

    from sleeve_fund.venues import venue

    inst = venue("KRAKEN").instrument("BTC", "USD", price_precision=1)
    ticks = qa.ticks(inst, flat_prices(10))
    first_quote = min(t.ts_event for t in ticks if isinstance(t, QuoteTick))
    assert first_quote <= qa.START + 5 * qa.S, "a quote within the first seconds, long before an entry at minute 5"
    assert all(any(isinstance(t, QuoteTick) and t.ts_event <= qa.START + k * qa.S for t in ticks) for k in (1, 5, 9))


# ----------------------------------------------------------------------------------------------------------------- e
def _walk(n=30000, seed=11):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.00008, 0.0012, n)
    for k in range(300, n, 170):  # sharp dips every 170 minutes, so the legacy RSI model trades
        r[k:k + 6] -= 0.0025
    c = 60_000 * np.exp(np.cumsum(r))
    o = np.concatenate([[c[0]], c[:-1]])
    h = np.maximum(o, c) * (1 + rng.uniform(0, 3e-4, n))
    lo = np.minimum(o, c) * (1 - rng.uniform(0, 3e-4, n))
    idx = pd.date_range("2025-10-03 00:01", periods=n, freq="1min", tz="UTC")
    from sleeve_fund.instruments import BOOK_SHARE

    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": 60 / BOOK_SHARE}, index=idx)


def test_report_price_matches_the_journal_within_one_and_a_half_rounding_cents_over_many_fills():
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.venues import venue

    inst = venue("KRAKEN").instrument("BTC", "USD", price_precision=1)
    worst, n = 0.0, 0
    for seed in range(11, 21):  # a drawdown halt ends one run early: several seeds make 100+ filled orders
        res = run_backtest("rsi_bands", _walk(12000, seed), inst, params={}, starting_capital=10_000,
                           risk_profile="aggressive", bar_minutes=1, half_spread=0.0003)
        by = {}
        for f in res.journal.fills_:
            by.setdefault(f["order_id"], []).append(f)
        for oid, fs in by.items():
            if oid not in res.fills.index:
                continue
            q = sum(f["qty"] for f in fs)
            jp = sum(f["qty"] * f["price"] for f in fs) / q
            rp = float(res.fills.loc[oid, "avg_px"])
            worst = max(worst, abs(rp - jp) * q / CENT)
            assert abs(rp - jp) <= 1.5 * CENT / q + 1e-9, (seed, oid, rp, jp, q, abs(rp - jp) * q / CENT)
            n += 1
        if n >= 150:
            break
    assert n >= 100, f"set-up: only {n} filled orders"
    print(f"\nreport vs journal: {n} orders, worst |diff| x qty = {worst:.3f} x CENT (bound 1.5)")


# ----------------------------------------------------------------------------------------------------------------- f
@pytest.mark.parametrize("label, perp, profile, side", [SETUPS[0], SETUPS[3]], ids=["spot-long", "perp-3x-short"])
def test_an_exact_touch_of_a_stop_triggers_books_the_same_and_matches_paper(label, perp, profile, side):
    """f: a minute whose low (a long's) or high (a short's) equals the stop level exactly: it triggers (touch is
    enough, as for the target, Advisor 19:40), books at level less max(half, 0.05 %), and paper's market-on-touch trigger
    does the same one-sidedly."""
    first, _ = _bt(flat_prices(40), side=side, perp=perp, profile=profile)
    level = round(_seq(first)[0][3] * (1 - side * 0.01), 1)
    p = flat_prices(40).copy()
    p[625:635] = level
    res, _ = _bt(p, side=side, perp=perp, profile=profile)
    assert round(_seq(res)[0][3] * (1 - side * 0.01), 1) == level  # the same entry
    row = minutes_of(p).iloc[10]
    assert float(row["low" if side > 0 else "high"]) == level  # an exact touch, nothing through it
    stop = [r for r in _seq(res) if r[0] == "stop_loss"]
    assert stop and stop[0][2] == minute(11), _seq(res)
    assert stop[0][3] == pytest.approx(level * (1 - side * max(HALF, FLOOR)), rel=1e-5, abs=0.1), (stop, level)
    run = paper(p, side=side, perp=perp, profile=profile, leave=30, stop=0.01)
    pst = [r for r in _paper_seq(run) if r[0] == "stop_loss"]
    assert pst, "paper triggers on the exact touch too"
    assert 0 <= side * (pst[0][3] - stop[0][3]) / stop[0][3] <= FLOOR + 0.0007, (pst, stop)


# ------------------------------------------------------------------------------------- probes (7 Oct), kept as pins
@pytest.mark.parametrize("half", [0.0003, 0.002], ids=["3bp", "20bp"])
@pytest.mark.parametrize("label, perp, profile, side", [SETUPS[0], SETUPS[2], SETUPS[3]], ids=["spot-long", "perp-long", "perp-short"])
def test_a_round_trip_on_flat_prices_costs_exactly_the_spread_once_plus_the_two_fees(label, perp, profile, side, half):
    """No double charge (Advisor 20:55: 'the spread leaves the separate cost line'): with the price unchanged, the loss is
    the booked entry and exit spreads (half x notional each) plus the two taker fees, and not a cent more, to the report's
    equity and to the journal's cash."""
    from sleeve_fund.store import replay_book

    p = flat_prices(40)
    res, _ = _bt(p, side=side, perp=perp, profile=profile, half=half, stop=None)
    seq = _seq(res)
    entry, exit_ = seq[0], seq[1]
    assert entry[0] == "entry" and exit_[0] == "exit"
    mid_in, mid_out = float(p[5 * 60 - 1]), float(p[30 * 60 - 1])
    qty = entry[4]
    gross = side * qty * (mid_out - mid_in)  # the mid-to-mid move: the drift of the flat tape
    spread_cost = qty * mid_in * half + qty * mid_out * half
    taker = _taker(perp, side)
    fees = entry[4] * entry[3] * taker + exit_[4] * exit_[3] * taker  # the exact taker fee on each BOOKED price, not the journal's
    want = 10_000 + gross - spread_cost - fees
    assert float(res.equity.iloc[-1]) == pytest.approx(want, abs=0.05 + 1.5 * CENT), (float(res.equity.iloc[-1]), want)
    book = replay_book(res.journal.fills_, 10_000, res.journal.funding_total("backtest"),
                       res.journal.insurance_total("backtest"))
    assert book["cash"] == pytest.approx(float(res.equity.iloc[-1]), abs=2 * CENT)


def test_c5_stops_and_targets_through_the_hub_book_the_ruled_one_sided_gap_to_the_backtest(monkeypatch):
    """D13-F1 (hub-path parity, the repo's test_c5_stops_and_targets): same orders, minutes and sizes. Entries and plain
    exits agree to 0.3 bp. A stop: the backtest is NEVER better than paper and is worse by at most the floor + 7 bp
    (Advisor D13-STOP-PARITY): backtest/hub - 1 in [-(5 + 7), +0.3] bp. Measured on ed7f484: -4.4 to -5.0 bp (the 5 bp
    floor against paper's near-level fill). Replaces the repo's TOL["stop_loss"] = (-2.5, 7.0), which predates D13 and
    allowed the backtest to be 7 bp BETTER than paper while rejecting the ruled-worse side. Needs tests/test_hub_path_parity.py
    beside this file's harness (a scratch worktree copy)."""
    import numpy as np

    import test_hub_path_parity as hpp
    from sleeve_fund.strategies import REGISTRY

    monkeypatch.setitem(REGISTRY, "probe", (hpp.Probe, hpp.ProbeConfig))  # that file's probe, not the hub-146 one
    s = np.arange(240 * 60)
    prices = 60_000 * (1 + 0.012 * np.sin(s / 500) + 0.0015 * np.sin(s / 13))
    params = {"period": 9, "stop_loss": 0.004, "take_profit": 0.025}
    o, f, _, _ = hpp.hub_paper(prices, params)
    hub = hpp.per_order(o, f)
    bt = hpp.per_order(*hpp.backtest(prices, params))
    assert {"entry", "stop_loss"} <= {r[1] for r in hub}
    assert [r[:2] for r in hub] == [r[:2] for r in bt]
    floor_bp = 5.0
    stops = 0
    for h, b in zip(hub, bt):
        gap = (b[4] / h[4] - 1) * 1e4
        assert b[3] == pytest.approx(h[3], rel=2e-3), (h, b)
        assert abs((b[2] - h[2]).total_seconds()) <= 60, (h, b)
        if h[1] == "stop_loss":
            stops += 1
            assert -(floor_bp + 7.0) <= gap <= 0.3, (gap, h, b)
        elif h[1] in ("entry", "exit"):
            assert abs(gap) <= 0.3, (gap, h, b)
        else:  # take_profit: the repo's band
            assert -6.0 <= gap <= 1.0, (gap, h, b)
    assert stops >= 3
