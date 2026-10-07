"""GAP-LIQ guards-on cases (Advisor ruling "GAP-LIQ", advisor-rulings.md; Head of QA + HoE 6 Oct ~20:08). A paper
ping_pong on a perp with a REAL stop (stop_loss 10%, no further than half the distance to liquidation at 2x and 3x),
every production guard on (no markers: check_perp_stop, the open-risk limit and the HC start refusal all live), and a
60% gap through BOTH the stop and the liquidation price. The Advisor rules such a stop books as a LIQUIDATION (D3
loss, halt, incident, X/Y, reset after liquidation required). The engine change is not built, so each case is a strict
xfail "GAP-LIQ"; stop-safety may merge with them as xfails, GAP-LIQ may not merge until they pass plain.

Each test checks its set-up first (stop placed and within half the liquidation distance; the gap fill beyond both the
stop and the liquidation price), then makes ONE final assertion of everything GAP-LIQ must give, so an xfail can only
come from the missing engine change. Seen on eb737bd: the gap books as a stop_loss past the bankruptcy price (the
insurance fund covers the shortfall) and the strategy is only paused for the day's loss, not halted.

  cd <scripts dir> && BACKTEST_ISOLATE=0 PYTHONPATH=<checkout>:<checkout>/tests \
    /tmp/claude-0/v/bin/python -m pytest -q -p no:cacheprovider -c <checkout>/pyproject.toml \
    --rootdir=<checkout> test_gap_liq_qa.py -rxX          (add --runxfail to see each one fail on its final assert)
"""
import dataclasses
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from test_degraded_155_qa import PERP, Store, _record_session, _reg, _session_meta, risk  # noqa: E402,F401
from test_degraded_155_qa import _equity_at_entry  # noqa: E402  (Y's base, Advisor 20:37)

GAP_LIQ = pytest.mark.xfail(strict=True, raises=AssertionError, reason="GAP-LIQ")
STOP = 0.10
NAME = "ping-pong-test"
LIQ_TEXT = "Position margin lost (liquidated): "


def _gap_through_stop(tmp_path, *, side, profile, after=()):
    """_liquidate_then (test_degraded_155_qa) with a real stop and nothing lifted: margin cap 10% of equity, a 60% gap
    against the position, then each of `after` ("resume" through the store; "stopstart": the PM's Stop then Start
    through the dashboard's command path and the supervisor's step, with what each did recorded, not asserted), each
    followed by a restarted process on a session two hours later that trades both ways."""
    import test_replay
    from sleeve_fund.research.replay import replay
    for nm, p in list(risk.PROFILES.items()):  # _reg puts the profiles back
        risk.PROFILES[nm] = dataclasses.replace(p, max_position_pct=0.1)
    params = {"rise": 0.01, "dip": 0.005, "stop_loss": STOP, **PERP}

    def meta(bal):
        m = _session_meta(bal, params)
        m["sleeve"].update(risk_profile=profile, starting_balance=10_000.0)
        return m
    store = Store(f"sqlite:///{tmp_path}/t.db")
    up = side == "short"
    gap = tmp_path / "gap.jsonl.gz"
    _record_session(gap, meta(10_000.0), [(5, 0.0), (20, 0.015 if up else -0.012), (0, 0.6 if up else -0.6), (5, 0.0)])
    orders = replay(gap, store=store)
    s = store.sleeve(NAME)
    out = {"orders": orders, "status": s.status, "reason": s.status_reason or "", "params": dict(s.params or {})}
    fills = store.fills(NAME, limit=200)
    entry = [o for o in orders if o["intent"] == "entry"][-1]  # the position the gap hit, grouped by order_id
    closing = [o for o in orders[orders.index(entry) + 1:]]
    opened = [f for f in fills if f["order_id"] == entry["order_id"]]
    closed = [f for f in fills if f["order_id"] in {o["order_id"] for o in closing}]
    lev = risk.profile(profile).max_leverage
    out.update(entry=entry, closing=closing, opened=opened, closed=closed)
    # X (Advisor 18:17 point 4): the whole position's margin plus its entry fee and the liquidation fee
    out["x"] = (sum(f["qty"] * f["price"] for f in opened) / lev + sum(f["fee"] for f in opened)
                + sum(f["fee"] for f in closed))
    close_ts = min(pd.Timestamp(f["ts"]) for f in closed) if closed else None
    held = [m for m in store.equity_series(NAME)
            if close_ts is not None and pd.Timestamp(m["ts"]) <= close_ts and m["qty"] != 0 and m["equity"] > 0]
    out["equity_before"] = held[-1]["equity"] if held else None  # MTM equity just before (18:17's Y base, superseded)
    # Y's base (Advisor 20:37, P1-D20): the strategy's equity when the position was opened, at its first entry fill
    out["equity_at_entry"] = _equity_at_entry(store, NAME, close_ts) if close_ts is not None else None
    left = store.journal_book(NAME, 10_000.0)["cash"]
    store.create_sleeve = lambda **kw: store.sleeve(kw["name"])
    start0, steps = test_replay.START, []
    px = 60_000.0 * (1.015 * 1.6 if up else 0.988 * 0.4)
    try:
        for i, step in enumerate(after):
            rec = {"step": step}
            if step == "resume":
                store.command(NAME, "resume", "try again")
            else:
                from fastapi.testclient import TestClient
                from sleeve_fund import supervisor as sup
                from sleeve_fund.dashboard import app as app_mod
                os.environ.setdefault("DASHBOARD_PASSWORD", "qa-pw")
                os.environ.setdefault("TEARSHEET_DIR", str(tmp_path))

                class FakePopen:
                    pid, returncode = 4242, None
                    poll = lambda self: None  # noqa: E731
                    send_signal = wait = kill = lambda self, *a, **k: 0  # noqa: E731
                c = TestClient(app_mod.create_app(store))
                auth, same = ("pm", os.environ["DASHBOARD_PASSWORD"]), {"origin": "http://testserver"}
                real_popen, sup.subprocess.Popen = sup.subprocess.Popen, lambda *a, **k: FakePopen()
                try:
                    sv = sup.Supervisor(store)
                    sv.procs[NAME] = sup.Proc()
                    sv.procs[NAME].popen, sv.procs[NAME].started_at = FakePopen(), sup.utcnow()
                    assert c.post(f"/sleeves/{NAME}/command", data={"command": "stop", "reason": "QA stop"}, auth=auth,
                                  headers=same, follow_redirects=False).status_code == 303  # harness
                    sv.step()
                    s = store.sleeve(NAME)
                    rec.update(after_stop={"desired": s.desired_state, "running": sv.procs[NAME].popen is not None,
                                           "status": s.status})
                    r = c.post(f"/sleeves/{NAME}/command", data={"command": "start", "reason": "QA start"}, auth=auth,
                               headers=same, follow_redirects=False)
                    rec["start_refused"] = r.status_code == 303 and "command_error" in r.headers.get("location", "")
                    rec["desired_after_start"] = store.sleeve(NAME).desired_state
                    sv.procs[NAME].popen = None
                    sv.step()
                finally:
                    sup.subprocess.Popen = real_popen
            test_replay.START = start0 + (i + 1) * 2 * 3600 * 10**9
            path = tmp_path / f"after{i}.jsonl.gz"
            _record_session(path, meta(left), [(3, -0.02), (3, 0.03), (4, -0.02)], px=px)
            got = replay(path, store=store)
            s = store.sleeve(NAME)
            rec.update(new_orders=len(got) - len(orders), status=s.status, reason=s.status_reason or "")
            steps.append(rec)
            orders = got
    finally:
        test_replay.START = start0
    out["steps"] = steps
    out["events"] = store.events(NAME, limit=5000)
    return out


def _set_up_holds(out, side):
    """Set-up only, never GAP-LIQ's outcome: a real stop placed within half the liquidation distance (the shipped
    rule), and the gap's closing fill beyond both the stop level and the liquidation price."""
    entry = out["entry"]
    assert out["params"].get("stop_loss") == STOP and out["opened"], (out["params"], entry)  # a real stop, stored
    sgn = 1 if side == "long" else -1
    entry_px = sum(f["qty"] * f["price"] for f in out["opened"]) / sum(f["qty"] for f in out["opened"])
    liq = entry["signal"]["liquidation_px"]
    assert STOP <= 0.5 * abs(1 - liq / entry_px), (STOP, liq, entry_px)  # stop <= half the distance to liquidation
    stop_px = entry_px * (1 - sgn * STOP)
    assert out["closed"], out["closing"]
    worst = min(f["price"] for f in out["closed"]) if sgn > 0 else max(f["price"] for f in out["closed"])
    assert sgn * (worst - stop_px) < 0 and sgn * (worst - liq) < 0, (worst, stop_px, liq)  # gap through both


def _x_y(reason):
    """X and Y from the ruled text (Advisor 17:57, 20:37): "...: X, Y% of strategy equity at entry"; Y is None
    without "at entry"."""
    m = re.search(r"Position margin lost \(liquidated\): ([\d,]+\.\d\d), (\d+(?:\.\d+)?)% of strategy equity"
                  r"( at entry)?", reason)
    return (float(m.group(1).replace(",", "")), m.group(2) if m.group(3) else None) if m else (None, None)


# PE2 (stop-safety, master a476a042): passes (mark removed)
@pytest.mark.parametrize("side,profile", [("short", "balanced"), ("long", "aggressive")])
def test_gap_liq_a_stop_gapped_through_liquidation_gives_x_and_y(tmp_path, side, profile):
    """X = margin + entry fee + liquidation fee over the whole position (grouped by order_id), to the cent; Y = X over
    the strategy's equity when the position was opened (Advisor 20:37), "Y% of strategy equity at entry", to the
    figures shown; booked as a liquidation and halted."""
    out = _gap_through_stop(tmp_path, side=side, profile=profile)
    _set_up_holds(out, side)
    x, y = _x_y(out["reason"])
    y_ok = (y is not None and out["equity_at_entry"] is not None and float(y) > 0 and abs(
        float(y) - 100 * x / out["equity_at_entry"]) <= 0.5 * 10 ** -len(y.partition(".")[2]) + 1e-9)
    got = {"booked_as": [o["intent"] for o in out["closing"]], "status": out["status"],
           "x": None if x is None else round(x, 2), "y_matches": y_ok}
    want = {"booked_as": ["liquidation"] * len(out["closing"]), "status": "halted", "x": round(out["x"], 2),
            "y_matches": True}
    assert got == want, (got, want, out["reason"])


# PE2 (stop-safety, master a476a042): passes (mark removed)
@pytest.mark.parametrize("side,profile", [("short", "balanced"), ("long", "aggressive")])
def test_gap_liq_the_strategy_stays_halted_until_a_reset_after_liquidation(tmp_path, side, profile):
    """Incident opened; a PM resume, the PM's Stop and Start, and each restart after them leave it halted with the
    liquidation's text and no new order. Only a reset after liquidation clears it."""
    out = _gap_through_stop(tmp_path, side=side, profile=profile, after=("resume", "stopstart"))
    _set_up_holds(out, side)
    kinds = {e["kind"] for e in out["events"]}
    got = {"booked_as": [o["intent"] for o in out["closing"]], "incident": bool(kinds & {"liquidation", "incident"}),
           "steps": [(st["step"], st["new_orders"], st["status"], st["reason"].startswith(LIQ_TEXT))
                     for st in out["steps"]]}
    want = {"booked_as": ["liquidation"] * len(out["closing"]), "incident": True,
            "steps": [("resume", 0, "halted", True), ("stopstart", 0, "halted", True)]}
    assert got == want, (got, want)


# PE2 (stop-safety, master a476a042): passes (mark removed)
@pytest.mark.parametrize("side,profile", [("short", "balanced"), ("long", "aggressive")])
def test_gap_liq_hc_after_the_pms_stop_it_is_not_trading_still_halted_and_start_is_refused(tmp_path, side, profile):
    """HC (Advisor 18:17) on a stop gapped through liquidation: after Stop the strategy is not trading (desired state
    stopped, no process) and the liquidation halt stays; Start is refused and changes nothing; the restart after
    trades nothing and is still halted with the liquidation's text."""
    out = _gap_through_stop(tmp_path, side=side, profile=profile, after=("stopstart",))
    _set_up_holds(out, side)
    (st,) = out["steps"]
    got = {"booked_as": [o["intent"] for o in out["closing"]], "after_stop": st["after_stop"],
           "start_refused": st["start_refused"], "desired_after_start": st["desired_after_start"],
           "after_restart": (st["new_orders"], st["status"], st["reason"].startswith(LIQ_TEXT))}
    want = {"booked_as": ["liquidation"] * len(out["closing"]),
            "after_stop": {"desired": "stopped", "running": False, "status": "halted"},
            "start_refused": True, "desired_after_start": "stopped", "after_restart": (0, "halted", True)}
    assert got == want, (got, want)


# --- P1-L22 restart regression pins (HoE 22:36; Advisor 22:36 "L22 liquidation basis") -------------------------------
# P1-L22 (QA, #146 round on e07ec26): after a paper perp restart the liquidation price was worked out from the restore
# fill (`_liq` -> `_net_position()`, the simulated venue holding the position at the restart's price), not from the
# journal's entry. A gap past the real liquidation price then booked a stop_loss past it and the liquidation hook never
# ran. Advisor 22:36: the liquidation price is computed ONCE from the position's own entry (average entry if there were
# adds) and its posted margin, with the engine's formula, persisted and restored, never recomputed from the restore
# price. HoE 22:36: fixed on #146 with the journal-entry basis GAP-LIQ shares; these pins keep it fixed on the
# stop-safety / GAP-LIQ side.
#
# Harness: #146's QA hub-fed paper node (quant-review/v2-p1/hub-146-scripts/test_hub_146_qa.py: `restart`, the `probe`
# strategy holding `side` from minute 5, a journaled position of 0.05 entered at minute 5 at ~60,000, Kraken BTC/USD
# simulated perp). Imported read-only; nothing in that folder is changed.
#   - "live" pins: the restarted process is up from 00:12 (nothing missed); the price is already 10% against a 3x
#     short (30% against a 2x long) when it comes back, and the gap through the stop and the liquidation price happens
#     LIVE at 00:15. No outage replay is involved, so they hold on any head with GAP-LIQ and the journal-entry basis
#     (845df4d: pass; main 9ae1f8a: xfail, it books a stop_loss past liquidation from a restore-based price).
#   - "replay" pins: HoE's exact case, restart(p, 6, 12): the minute to 00:08 opens 40% against the 3x short (55% against
#     the 2x long, past its ~49.7% liquidation distance) while the process is down; on return the price is 10% (30%)
#     against. They need #146's NA-3 outage replay as well, so they are strict xfail "L22" on every head today.
import importlib  # noqa: E402

from sleeve_fund.strategies.base import LongFlatStrategy as _LFS  # noqa: E402

L22_HARNESS = "/mnt/project-files/sleeve-fund/quant-review/v2-p1/hub-146-scripts"
_HAS_GAP_LIQ = hasattr(_LFS, "_gap_liquidation")  # the GAP-LIQ engine change (PE2 5e00eb2) is on this head
L22_LIVE = pytest.mark.xfail(not _HAS_GAP_LIQ, strict=True, raises=AssertionError,
                             reason="L22 (+GAP-LIQ): no journal-entry liquidation basis after a restart on this head")
L22_REPLAY = pytest.mark.xfail(strict=True, raises=AssertionError,
                               reason="L22: needs #146's NA-3 outage replay with the journal-entry liquidation basis")
# (label, side, profile, price against the position on return, the gap, a stop restorable after the restart that the
# return does not trip: beyond the return, inside the gap)
L22_CASES = [pytest.param(-1, "aggressive", 0.10, 0.40, 0.15, id="3x-short"),
             pytest.param(1, "balanced", 0.30, 0.55, 0.35, id="2x-long")]


def _l22_harness(monkeypatch):
    try:
        h = importlib.import_module("test_hub_146_qa")
    except ImportError:
        sys.path.insert(0, L22_HARNESS)
        try:
            h = importlib.import_module("test_hub_146_qa")
        finally:
            sys.path.remove(L22_HARNESS)
    from sleeve_fund.strategies import REGISTRY

    monkeypatch.setitem(REGISTRY, "probe", h._probe_classes())
    return h


def _l22_run(monkeypatch, side, profile, back, gap, stop, replayed):
    """The restart, with every liquidation price the engine works out (`_liq`) and every liquidation-hook call
    recorded. Returns (run, expected liquidation price from the journal entry before the restart, the engine's liq
    prices after the restart, hook calls)."""
    from sleeve_fund import markets
    from sleeve_fund.store import replay_book

    h = _l22_harness(monkeypatch)
    seen, hook = [], []
    real_liq = _LFS._liq

    def liq_spy(self, cash, qty):
        v = real_liq(self, cash, qty)
        if qty:
            seen.append(v)
        return v
    monkeypatch.setattr(_LFS, "_liq", liq_spy)
    if hasattr(_LFS, "_on_liquidation"):  # #146's L18 hook
        real_hook = _LFS._on_liquidation

        def hook_spy(self, *a, **k):
            hook.append(a)
            return real_hook(self, *a, **k)
        monkeypatch.setattr(_LFS, "_on_liquidation", hook_spy)
    p = h.flat_prices(30)
    if replayed:  # the minute to 00:08 opens past liquidation while the process is down, back at 00:12 `back` against
        p = h.shape(h.shape(p, 7.0, 8.0, h.adverse(side, gap)), 8.0, 30, h.adverse(side, back))
        run = h.restart(p, 6, 12, side=side, perp=True, profile=profile, stop=0.01)
    else:  # back at 00:12 already `back` against; the gap comes live at 00:15
        p = h.shape(h.shape(p, 12.0, 15.0, h.adverse(side, back)), 15.0, 30, h.adverse(side, gap))
        run = h.restart(p, 12, 12, side=side, perp=True, profile=profile, stop=stop)
    held = [f for f in run.fills if f["order_id"] == "O-held"]
    book = replay_book(held, 10_000)  # the journal as it stood before the restart: the entry only
    from sleeve_fund import risk

    want = markets.isolated_liquidation(book["cash"], book["qty"], book["entry_px"],
                                        risk.profile(profile).max_leverage,
                                        markets.terms({"market": "perp"}, "KRAKEN").maintenance_margin)
    return run, want, seen, hook


def _l22_assert(run, want, seen, hook, side, gap):
    # The orders that traded (QA, 7 Oct ~00:40): the stop-safety head journals the stop it watches as an order row
    # ("STOP (watched)", cancelled unfilled when it is replaced or the position goes), which closes nothing; "booked"
    # means filled, as before.
    closing = [o for o in run.orders if o["order_id"] != "O-held" and o["filled_qty"] > 0]
    liq_events = run.kinds("liquidation")
    first = closing[0] if closing else {}
    got = {
        "first_exit": first.get("intent"),
        "liq_px_recorded": round(float((first.get("signal") or {}).get("liquidation_px", 0.0)), 1),
        "stop_loss_orders": [round(float(o.get("avg_px") or 0.0), 1) for o in closing if o["intent"] == "stop_loss"],
        "liq_after_restart": round(seen[0], 1) if seen else None,
        "liquidation_events": len(liq_events),
        "hook_calls": len(hook) if hasattr(_LFS, "_on_liquidation") else len(liq_events),
    }
    exp = {"first_exit": "liquidation", "liq_px_recorded": round(want, 1), "stop_loss_orders": [],
           "liq_after_restart": round(want, 1), "liquidation_events": 1, "hook_calls": 1}
    # set-up: the gap is past the journal-entry liquidation price (else nothing here is about L22)
    gap_px = 60_000.0 * (1 - side * gap)
    assert side * (gap_px - want) < 0, (gap_px, want)
    assert got == exp, (got, exp, [(o["intent"], o["reason"][:120]) for o in closing])


@L22_LIVE
@pytest.mark.parametrize("side,profile,back,gap,stop", L22_CASES)
def test_l22_after_a_restart_the_liquidation_price_is_the_journal_entrys_and_a_live_gap_books_a_liquidation(
        monkeypatch, side, profile, back, gap, stop):
    """Restarted 10% (3x short) / 30% (2x long) against the entry; the liquidation price the engine works out after the
    restart equals the one from the journal's entry before it (Advisor 22:36: never recomputed from the restore price),
    and the live gap through the stop and that price books ONE liquidation recorded at it, not a stop_loss past it."""
    run, want, seen, hook = _l22_run(monkeypatch, side, profile, back, gap, stop, replayed=False)
    _l22_assert(run, want, seen, hook, side, gap)


# PE2 (stop-safety, master a476a042): passes (mark removed)
@pytest.mark.parametrize("side,profile,back,gap,stop", L22_CASES)
def test_l22_restart_replay_of_a_minute_past_liquidation_books_the_liquidation_at_the_entry_based_price(
        monkeypatch, side, profile, back, gap, stop):
    """HoE 22:36, P1-L22 as found: restart(p, 6, 12); the minute to 00:08 opened past the entry-based liquidation price
    (3x short: ~79,604, 40% against; 2x long: ~30,152, 55% against) and the price is still 10% / 30% against on return.
    The replay books ONE liquidation recorded at the journal-entry liquidation price (the hook called once), never the
    e07ec26 stop_loss at ~84,045 past it; the engine's liquidation price after the restart is the journal entry's."""
    run, want, seen, hook = _l22_run(monkeypatch, side, profile, back, gap, stop, replayed=True)
    _l22_assert(run, want, seen, hook, side, gap)
