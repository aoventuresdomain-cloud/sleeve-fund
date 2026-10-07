"""QA pins for P1-SG7 (a fill that raced the block's cancel), per the Independent Quant Advisor's ruling of 6 Oct 23:05
(advisor-rulings.md): a raced fill is treated exactly as the block treats positions already held.

- A drawdown halt or a liquidation flattens: the raced fill is SOLD at once through the exit path, journaled
  "raced fill flattened by halt".
- A strategy-level daily pause follows the profile's own rule. risk.PROFILES (845df4d and main): every profile's
  daily-loss pause flattens ("daily_loss: ... flatten and pause"; SleeveRuntime.tick flattens on pause_day whatever the
  profile), so the raced fill is sold on conservative, balanced and aggressive alike. If a profile's pause ever stops
  flattening, its cell here must move to the kept set.
- The PM's Stop, stale data, missing funding and Retire don't flatten: the raced fill is KEPT with its stop.

Harness: the exposure-gate master's own (stop-safety QA imports it read-only; agent a76842b owns that file and its
raced-remainder cells, which still carry the 20:52/20:56 "kept" expectation for every status). The race, as there: a
resting stop entry (BUY 0.01) placed 20 minutes before the block fills one second after it; the supervisor steps; the
strategy's process (if the supervisor runs one) starts again two hours later on the same journal.

Run (SQLite or Postgres), from a checkout whose tests/ (or a copy with the masters overlaid) is on the path:
  TEST_DATABASE_URL=... BACKTEST_ISOLATE=0 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=<checkout>:<checkout>/tests:<quant-review>/exposure-gate-xfails:<quant-review>/stop-safety-xfails \
  python -m pytest -q -p no:cacheprovider test_gate_845_u35_sg7.py
"""
from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

import test_exposure_gate_xfails as egx
from sleeve_fund import risk
from test_exposure_gate_xfails import _offline, client, store  # noqa: F401 (fixtures)

SG7 = ("P1-SG7 (Advisor 23:05): a raced fill at a flattening halt is sold at once through the exit path, journaled "
       "'raced fill flattened by halt'")
WORDING = "raced fill flattened by halt"
FLATTEN_INTENTS = ("risk_halt", "risk_pause", "liquidation", "exit", "pm_flatten")  # the exit path's own intents


def xf(reason=SG7, condition=True):
    return pytest.mark.xfail(condition, strict=True, raises=AssertionError, reason=reason)


class _ProfiledRecorder(egx.Recorder):
    """The recorded session's strategy runs on `PROFILE` (egx.run writes "balanced")."""
    PROFILE = "balanced"

    def start(self, instrument):
        self.meta.setdefault("sleeve", {})["risk_profile"] = self.PROFILE
        super().start(instrument)


def _on_profile(monkeypatch, profile):
    rec = type("Rec", (_ProfiledRecorder,), {"PROFILE": profile})
    monkeypatch.setattr(egx, "Recorder", rec)


def _session_one(tmp_path, store, monkeypatch, cell):  # noqa: F811
    """The block starts. `cell` is a reason name, or "daily_pause@<profile>": the day opened `daily_loss` + 1% higher
    than the equity at the block (journalled at 00:00 UTC, as the exposure-gate set-up does with 6%), so that profile's
    own daily-loss pause fires, and no drawdown halt (the drawdown stays under every profile's limit)."""
    if "@" not in cell:
        _on_profile(monkeypatch, "balanced")
        return egx._session_one(tmp_path, store, monkeypatch, cell), cell
    reason_name, profile = cell.split("@")
    _on_profile(monkeypatch, profile)
    reason = egx.REASONS[reason_name]
    plan = egx._plan_for(reason, tag="s1")
    at = egx.M(plan.t0, reason.block_at)
    loss = risk.PROFILES[profile].daily_loss + 0.01

    def daily(rt, kw, t0=plan.t0):
        day_open = round(kw["equity"] / (1 - loss), 2)
        store.record_equity(egx.NAME, equity=day_open, cash=day_open, qty=0, price=kw["price"], benchmark=day_open,
                            ts=t0.normalize().to_pydatetime())
        rt._day_open = day_open

    plan.hooks.append((at, daily))
    egx.run(tmp_path, store, plan, monkeypatch)
    egx.check_entered(reason, store, at)
    started = [e["message"] for e in egx._events(store, ("start",))]
    assert started and f"({profile} risk profile)" in started[0], f"setup: not on {profile}: {started}"
    return at.to_pydatetime(), reason_name


FALL_AT = 10  # minutes into the restarted session: the price falls 3%, through the raced fill's 2% stop


def _restart_balance(store):  # noqa: F811
    """The account a restart opens with, as paper.node builds it from the journal (the exposure-gate master's
    _restart_balance, a76842b): a perp's margin account opens with the cash it had when the position was opened, and the
    restore order puts the position back. (QA's first version passed the journal's spot-style cash, which takes the
    position's notional off twice: the "engine cash vs journal" mismatch seen after a raced fill was that, a harness
    artefact.)"""
    book = store.journal_book(egx.NAME, 10_000)
    if store.sleeve(egx.NAME).params.get("market") == "perp":
        return book["cash"] + book["qty"] * (book["entry_px"] or 0.0)
    return book["cash"]


def _session_two_falling(tmp_path, store, monkeypatch, reason_name, hours=2, minutes=20):  # noqa: F811
    """egx._session_two (the process starting again two hours later, its signal long throughout), with the price
    falling 3% at FALL_AT: through the raced fill's 2% stop (60,015 x 0.98 = 58,814.7), so its stop must close it."""
    reason = egx.REASONS[reason_name]
    t0 = reason.t0 + pd.Timedelta(hours=hours)
    k = FALL_AT * 60
    plan = egx.Plan(t0=t0, minutes=minutes, tag="s2", bar=reason.bar,
                    price=lambda s: (60_000 + s * 0.01) * (0.97 if s >= k else 1.0),
                    windows=[(t0 - pd.Timedelta(minutes=10), egx.M(t0, minutes), 1)],
                    balance=_restart_balance(store))
    egx.run(tmp_path, store, plan, monkeypatch)
    return t0.to_pydatetime()


def _race_then_restart(tmp_path, store, monkeypatch, cell, fall=False):  # noqa: F811
    at, reason_name = _session_one(tmp_path, store, monkeypatch, cell)
    raced = egx._raced_remainder(store, reason_name, at)
    sup = egx._supervisor(store, monkeypatch)
    sup.step()
    restart = None
    if sup.procs.get(egx.NAME) is not None and sup.procs[egx.NAME].alive:  # its process runs: what it does
        if fall:
            restart = _session_two_falling(tmp_path, store, monkeypatch, reason_name)
        else:
            restart = egx._session_two(tmp_path, store, monkeypatch, reason_name)
    sup.step()
    return at, raced, restart


def _after(store, raced):  # noqa: F811
    return [(f["ts"], f["side"], f["qty"], f["order_id"]) for f in egx._fills(store) if f["ts"] >= raced]


FLATTENING = ["drawdown_halt", "liquidation"] + [f"daily_pause@{p}" for p in risk.PROFILES]
IDS = {"drawdown_halt": "u35-halted", "liquidation": "u35-liquidated",
       **{f"daily_pause@{p}": f"u35-paused-{p}" for p in risk.PROFILES}}


@pytest.mark.parametrize("cell", FLATTENING, ids=[IDS[c] for c in FLATTENING])
def test_sg7_a_raced_fill_at_a_flattening_halt_is_sold_at_once_through_the_exit_path(tmp_path, store, client,  # noqa: F811
                                                                                     monkeypatch, cell):
    """Advisor 23:05: the halt's own rule covers the raced fill. Its process, started again on the same journal, sells
    the raced 0.01 on its first tick (within the restart's first minute) with an order of the exit path (the halt's or
    pause's own flatten intent), opens nothing after it, and is flat."""
    at, raced, restart = _race_then_restart(tmp_path, store, monkeypatch, cell)
    assert restart is not None, "setup: the supervisor ran no process for the halted strategy"
    sells = [f for f in egx._fills(store) if f["ts"] >= raced and f["side"] == "SELL"]
    orders = {o["order_id"]: o for o in egx._orders(store)}
    assert sells, f"the raced fill was not sold: {_after(store, raced)}"
    first = sells[0]
    assert orders[first["order_id"]]["intent"] in FLATTEN_INTENTS, (
        f"sold, but not through the exit path: intent {orders[first['order_id']]['intent']!r}")
    assert first["ts"] - restart <= timedelta(minutes=1), f"not at once: sold at {first['ts']}, restart {restart}"
    assert not [f for f in egx._fills(store) if f["ts"] >= first["ts"] and f["side"] == "BUY"], _after(store, raced)
    assert abs(egx._net(store)) < 1e-9, f"still holding {egx._net(store)}"


@pytest.mark.parametrize("cell", FLATTENING, ids=[IDS[c] for c in FLATTENING])
# PE2 (stop-safety, master 26f172db): passes (mark removed)
def test_sg7_the_raced_fill_flattened_by_a_halt_is_journaled_in_the_ruled_words(tmp_path, store, client,  # noqa: F811
                                                                                monkeypatch, cell):
    """Advisor 23:05: journaled "raced fill flattened by halt" (any level or kind; the words are the ruling's). 845df4d
    sells it through the flatten retry, journaled "still holding 0.01 after a flatten that didn't complete; closing it
    again (attempt 1 of 3; ...)"."""
    at, raced, restart = _race_then_restart(tmp_path, store, monkeypatch, cell)
    said = [e["message"] for e in egx._events(store) if e["ts"] >= raced - timedelta(seconds=1)]
    assert any(WORDING in m.lower() for m in said), f"no '{WORDING}' in the journal: {[m[:90] for m in said]}"


KEPT = ["stopped", "retired", "funding_missing", "stale_data"]
# A head without the gate branch (main 9ae1f8a): a stopped or retired holder has no process (no U35), so nothing
# watches the raced fill's stop there. Strict xfails on that condition only.
PRE_GATE = not hasattr(egx.SleeveRuntime, "entry_blocked")
KEPT_PARAMS = [pytest.param(c, id=f"u35-{c}", marks=xf("U35 not on this head", PRE_GATE) if c in ("stopped", "retired")
                            else ()) for c in KEPT]


@pytest.mark.parametrize("cell", KEPT_PARAMS)
def test_sg7_a_raced_fill_at_a_block_that_does_not_flatten_is_kept_with_its_stop(tmp_path, store, client,  # noqa: F811
                                                                                 monkeypatch, cell):
    """Advisor 23:05 (and 20:52/20:56): the PM's Stop, stale data, missing funding and Retire keep the raced fill with
    its stop. Its process starts again two hours later (none: nothing watches it, which fails here): the raced fill is
    still held (nothing sells it) until the price falls 3% through its 2% stop, and then its stop closes it (a
    stop_loss order fills; nothing but the stop sells). Stale data ends by itself at the next complete bar, so that
    restarted process may add on its long signal: the stop then closes the lot. (Behaviour, not mechanism: the paper
    engine watches a stop each tick; an open stop_loss order in the journal isn't asked for.)"""
    at, raced, restart = _race_then_restart(tmp_path, store, monkeypatch, cell, fall=True)
    assert restart is not None, (f"no process runs for the strategy holding the raced fill (status "
                                 f"{store.sleeve(egx.NAME).status}, archived {egx.NAME in store.archived()}): "
                                 "nothing watches its stop")
    fall = restart + timedelta(minutes=FALL_AT)
    orders = {o["order_id"]: o for o in egx._orders(store)}
    sold = [f for f in egx._fills(store) if f["ts"] >= raced and f["side"] == "SELL"]
    early = [f for f in sold if f["ts"] < fall or orders[f["order_id"]]["intent"] != "stop_loss"]
    assert not early, f"the raced fill was sold, not by its stop: {_after(store, raced)}"
    held = round(sum(egx._signed(f) for f in egx._fills(store) if f["ts"] < fall), 9)
    assert held >= egx.RACED_QTY - 1e-9, f"the raced fill was not kept until the fall: net {held}"
    assert sold and abs(egx._net(store)) < 1e-9, (
        f"its stop didn't close it after the 3% fall: net {egx._net(store)}, {_after(store, raced)}")
