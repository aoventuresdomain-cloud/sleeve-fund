"""F283-O3: a reset after liquidation (RAL) for a two-leg Strategy (Advisor 14 Oct 13:15 UK, approved with additions).

The PM sends it to the Strategy (its first leg's row) naming a liquidated leg's incident; it is journaled on that leg's
row (one reset per incident) with the other leg's half (PAIR_REBASE) under it. It is refused until the pair is flat
with nothing working and the incident's note also says why the pair's close at half its distance to liquidation didn't
prevent it. Taken, it re-bases the pair's SUMMED high-water mark, halt reference and day's start on both legs and in
pair_state; never the book's figures; a summed drawdown halt stays until a Resume; each liquidated leg needs its own.

Liquidations here are the runtime's own (`ruined`, as the engine reports a gap past the bankruptcy price), with fills
journaled so each leg's cash is what the journal says. Postgres when TEST_DATABASE_URL is set, else SQLite."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal as D

import pytest

from sleeve_fund.dashboard import trading
from sleeve_fund.paper.runtime import WIPED_OUT, SleeveRuntime, liquidation_head, pair_ral_figures
from sleeve_fund.store import LIQUIDATION_RESET, PAIR_REBASE, PAIR_REBASED, RAL, Store, make_engine, metadata

T = datetime(2026, 10, 14, 9, 0, tzinfo=UTC)
SPOT = {"instrument": "BTC/USDT", "params": {}, "venue": "KRAKEN", "side": 1}
PERP = {"instrument": "BTC/USDT", "params": {"market": "perp"}, "venue": "KRAKEN", "side": -1}
LEGS = ("carry", "carry~2")
PX = 60_000.0
WHY = "The price gapped through the resting stop and the liquidation price in one trade"
CLAMP = "The gap crossed half the distance and the liquidation price in the same trade, before any close could fill"


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


class Pair:
    """Both legs' runtimes in one process, as the two-leg node runs them, on one clock."""

    def __init__(self, store: Store, at: datetime = T) -> None:
        self.store, self.clock = store, [at]
        self.rt = {n: self._runtime(n) for n in LEGS}

    def _runtime(self, name: str) -> SleeveRuntime:
        rt = SleeveRuntime(self.store, name, now=lambda: self.clock[0])
        rt.paired = True  # as the Pair sets it
        rt.on_start(0.0005)
        return rt

    def cash(self, name: str) -> float:
        return float(self.store.journal_book(name, self.store.sleeve(name).starting_balance)["cash"])

    def tick(self, at: datetime, order=LEGS, ruined: str | None = None) -> None:
        """Each leg marked at its journal cash (both flat), judged with the other's: the summed book."""
        self.clock[0] = at
        for name in order:
            other = next(n for n in LEGS if n != name)
            eq = self.cash(name)
            self.rt[name].tick(equity=eq, cash=eq, qty=0.0, price=PX, other_equity=self.cash(other),
                               ruined=ruined if name == "carry~2" else None)

    def restart(self) -> None:
        self.rt = {n: self._runtime(n) for n in LEGS}


def _trip(store, name, pnl, ts, tag):
    """A closed round trip moving `name`'s cash by `pnl`, fee free."""
    store.record_fill(name, side="BUY", qty=1.0, price=1_000.0, fee=0.0, order_id=f"{tag}b", trade_id=f"{tag}b", ts=ts)
    store.record_fill(name, side="SELL", qty=1.0, price=1_000.0 + pnl, fee=0.0, order_id=f"{tag}s",
                      trade_id=f"{tag}s", ts=ts)


def _liquidate(store, *, loss=400.0, unwind=100.0, leg="carry~2", at=T + timedelta(minutes=1)) -> tuple[Pair, int]:
    """The pair marked flat at T, then `leg` liquidated (losing `loss`) and the other leg's forced unwind (`unwind`),
    both closed by `at`. Returns the pair's runtimes and the liquidation's incident, noted with both causes."""
    p = Pair(store)
    p.tick(T)
    other = next(n for n in LEGS if n != leg)
    _trip(store, leg, -loss, at, f"liq-{leg}")
    _trip(store, other, unwind, at, f"unw-{other}")
    p.clock[0] = at
    p.rt[leg].tick(equity=p.cash(leg), cash=p.cash(leg), qty=0.0, price=PX, other_equity=p.cash(other),
                   ruined=WIPED_OUT)
    p.rt[other].tick(equity=p.cash(other), cash=p.cash(other), qty=0.0, price=PX, other_equity=p.cash(leg))
    return p, _incident(store, leg, at)


def _incident(store, leg, at, clamp=CLAMP) -> int:
    store.event(leg, "error", "incident", f"Liquidated: {leg} lost its position margin", ts=at)
    iid = store.last_event(leg, ("incident",))["id"]
    store.write_incident_note(iid, author="Head of Engineering", why_stop_did_not_protect=WHY,
                              why_clamp_did_not_prevent=clamp)
    return iid


def _ral(store, iid, actor="PM", name="carry"):
    store.command(name, RAL, "reset after liquidation: both causes written", actor=actor, incident=iid)


def _summed(p: Pair) -> float:
    return sum(p.cash(n) for n in LEGS)


def _written(store):
    """Everything a refused RAL must leave as it was."""
    return ([store.pending_commands(n) for n in LEGS], [store.events(n, limit=500) for n in LEGS],
            store.pair_state("carry"), store.decisions(limit=500))


# --- (1) taken once the pair is flat; both legs and the pair re-based on the summed equity ---------------------------

@pytest.mark.parametrize("order", [LEGS, LEGS[::-1]], ids=["first-leg-ticks-first", "liquidated-leg-ticks-first"])
def test_a_liquidated_legs_reset_rebases_the_summed_figures_on_both_legs_and_the_pair(store, order):
    p, iid = _liquidate(store)
    old = store.pair_state("carry")["hwm"]
    assert old == D(10_000)
    _ral(store, iid)
    # Journaled on the liquidated leg's row; the other leg's half under it, naming that command.
    mine = [c for c in store.pending_commands("carry~2") if c["command"] == RAL]
    half = [c for c in store.pending_commands("carry") if c["command"] == RAL]
    assert len(mine) == len(half) == 1 and mine[0]["incident"] == iid and half[0]["incident"] is None
    assert half[0]["reason"].startswith(PAIR_REBASE) and half[0]["reason"].endswith(f"(command #{mine[0]['id']})")
    p.tick(T + timedelta(minutes=2), order=order)
    p.tick(T + timedelta(minutes=3), order=order)  # the other leg's half waits for the liquidated leg's reset
    summed = _summed(p)
    assert summed == pytest.approx(9_700.0)
    assert not store.pending_commands("carry") and not store.pending_commands("carry~2")
    for n in LEGS:
        assert p.rt[n].peak == pytest.approx(summed) and p.rt[n]._day_open == pytest.approx(summed), n
    row = store.pair_state("carry")
    assert row["hwm"] == row["reference_equity"] == row["day_start_equity"] == pytest.approx(D(str(summed)))
    assert liquidation_head(store, "carry~2") is None and not trading.liquidated_since_reset(store, "carry")
    assert [p.rt[n].status for n in LEGS] == ["running", "running"]
    assert p.rt["carry~2"].can_open() and p.rt["carry"].can_open()
    # One reset event on the liquidated leg, one re-base on the other; the old figures stay in the journal.
    resets = [e for e in store.events("carry~2", limit=500) if e["kind"] == LIQUIDATION_RESET]
    rebased = [e for e in store.events("carry", limit=500) if e["kind"] == PAIR_REBASED]
    assert len(resets) == len(rebased) == 1
    assert "10,000.00 becomes 9,700.00" in resets[0]["message"] and "both legs' equity summed" in resets[0]["message"]
    assert "10,000.00 becomes 9,700.00" in rebased[0]["message"]
    assert not [e for e in store.events("carry", limit=500) if e["kind"] == LIQUIDATION_RESET]


def test_the_ral_is_refused_until_the_pair_is_flat_with_nothing_held_or_working(store):
    """Advisor (c): the refusal names what is still open; nothing is written."""
    p, iid = _liquidate(store)
    moves = [("flat", "legging", "t1", None), ("legging", "closing", "t1", "t1"), ("closing", "flat", None, "t1")]
    for frm, to, trade, prior in moves:
        assert store.set_pair_state("carry", expected=frm, state=to, trade_id=trade, ts=T,
                                    deadline=None if to == "flat" else T, expected_trade_id=prior)
        if to != "flat":
            before = _written(store)
            with pytest.raises(ValueError, match=f"the pair is still {to}"):
                _ral(store, iid)
            assert _written(store) == before
    # The other leg still holds (its unwind not filled), then has its unwind working.
    store.record_fill("carry", side="BUY", qty=0.5, price=PX, fee=0.0, order_id="h1", trade_id="h1",
                      ts=T + timedelta(minutes=1))
    before = _written(store)
    with pytest.raises(ValueError, match="carry still holds 0.5"):
        _ral(store, iid)
    assert _written(store) == before
    store.record_fill("carry", side="SELL", qty=0.5, price=PX, fee=0.0, order_id="h2", trade_id="h2",
                      ts=T + timedelta(minutes=1))
    store.record_order("carry", order_id="w1", side="SELL", qty=0.1, intent="leg_unwind", reason="unwind",
                       ts=T + timedelta(minutes=1))
    before = _written(store)
    with pytest.raises(ValueError, match="carry has an order working"):
        _ral(store, iid)
    assert _written(store) == before
    store.update_order("w1", status="canceled")
    _ral(store, iid)  # flat, nothing working: taken
    assert [c["command"] for c in store.pending_commands("carry~2")] == [RAL]


# --- (2) a summed drawdown halt stays ----------------------------------------------------------------------------

def test_a_summed_drawdown_halt_with_the_liquidation_stays_until_a_resume(store):
    """A 25% summed loss (balanced: 20%): the other leg halts on the summed drawdown on the liquidation's tick. The
    RAL lifts only the liquidation's halt; the drawdown halt stays, its journal keeping the drawdown from the OLD
    high-water mark, and the pair opens nothing until a Resume."""
    p, iid = _liquidate(store, loss=3_000.0, unwind=500.0)
    assert store.sleeve("carry").status == "halted"
    dd = [e for e in store.events("carry", limit=500) if e["kind"] == "risk_halt"]
    assert len(dd) == 1 and "drawdown 25.0%" in dd[0]["message"]
    _ral(store, iid)
    p.tick(T + timedelta(minutes=2))
    p.tick(T + timedelta(minutes=3))
    assert not store.pending_commands("carry") and not store.pending_commands("carry~2")
    assert store.sleeve("carry").status == "halted" and not p.rt["carry"].can_open()
    assert [e["message"] for e in store.events("carry", limit=500) if e["kind"] == "risk_halt"] == [dd[0]["message"]]
    # A Resume on the Strategy reaches both legs and is what clears it.
    store.command("carry", "resume", "the drawdown is understood")
    p.tick(T + timedelta(minutes=4))
    assert [store.sleeve(n).status for n in LEGS] == ["running", "running"]


# --- (3) both legs liquidated: one reset each ---------------------------------------------------------------------

def test_both_legs_liquidated_need_one_reset_each(store):
    p, first = _liquidate(store)
    at = T + timedelta(minutes=1)
    p.clock[0] = at
    p.rt["carry"].tick(equity=p.cash("carry"), cash=p.cash("carry"), qty=0.0, price=PX,
                       other_equity=p.cash("carry~2"), ruined=WIPED_OUT)
    second = _incident(store, "carry", at)
    _ral(store, first)
    p.tick(T + timedelta(minutes=2))
    p.tick(T + timedelta(minutes=3))
    # The first leg's half re-based it, but answered nothing: it is still liquidated, and so is the Strategy.
    assert liquidation_head(store, "carry~2") is None and liquidation_head(store, "carry") is not None
    assert trading.liquidated_since_reset(store, "carry") and not p.rt["carry"].can_open()
    _ral(store, second)
    p.tick(T + timedelta(minutes=4))
    p.tick(T + timedelta(minutes=5))
    assert liquidation_head(store, "carry") is None and not trading.liquidated_since_reset(store, "carry")
    summed = _summed(p)
    assert store.pair_state("carry")["hwm"] == pytest.approx(D(str(summed)))  # the second re-base: idempotent
    assert all(p.rt[n].peak == pytest.approx(summed) for n in LEGS)


# --- (4) and (10) refusals -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("case", ["wrong-incident", "other-legs-incident", "second-leg-row", "not-the-pm", "none"])
def test_a_ral_on_the_wrong_incident_row_or_actor_is_refused_and_writes_nothing(store, case):
    p, iid = _liquidate(store)
    store.event("carry", "error", "incident", "not a liquidation", ts=T + timedelta(minutes=1))
    others = store.last_event("carry", ("incident",))["id"]
    store.write_incident_note(others, author="x", why_stop_did_not_protect=WHY, why_clamp_did_not_prevent=CLAMP)
    before = _written(store)
    kw, words = {
        "wrong-incident": ({"iid": 999_999}, "is not the incident"),
        "other-legs-incident": ({"iid": others}, "isn't halted after a liquidation"),
        "second-leg-row": ({"iid": iid, "name": "carry~2"}, "second leg of carry"),
        "not-the-pm": ({"iid": iid, "actor": "HoE"}, "only the PM"),
        "none": ({"iid": None}, "name the liquidation's incident"),
    }[case]
    with pytest.raises(ValueError, match=words):
        _ral(store, **kw)
    assert _written(store) == before


def test_the_ral_is_refused_until_the_incident_says_why_the_clamp_did_not_prevent_it(store):
    """Advisor (d) 2 and cell (10)."""
    p, _ = _liquidate(store)
    bare = _incident(store, "carry~2", T + timedelta(minutes=1), clamp="")
    before = _written(store)
    with pytest.raises(ValueError, match="why closing the pair at half its distance to liquidation did not prevent"):
        _ral(store, bare)
    assert _written(store) == before
    store.write_incident_note(bare, author="Head of Engineering", why_stop_did_not_protect=WHY,
                              why_clamp_did_not_prevent=CLAMP)
    _ral(store, bare)
    assert [c["incident"] for c in store.pending_commands("carry~2")] == [bare]


def test_the_other_legs_forced_unwind_is_no_liquidation(store):
    """Advisor (d) 4: no incident, no RAL for it."""
    p, _ = _liquidate(store)
    assert liquidation_head(store, "carry") is None
    assert not [e for e in store.events("carry", limit=500) if e["kind"] == "incident"]


# --- (5) a crash between the journal and the applied mark -----------------------------------------------------------

@pytest.mark.parametrize("leg", ["carry~2", "carry"], ids=["liquidated-leg", "other-legs-half"])
def test_a_crash_after_journaling_finishes_the_reset_once_on_the_restart(store, monkeypatch, leg):
    p, iid = _liquidate(store)
    _ral(store, iid)
    if leg == "carry":
        p.tick(T + timedelta(minutes=2), order=("carry~2",))  # the liquidated leg's reset first
    real = store.mark_applied
    crashed = []

    def crash(command_id):
        if not crashed:
            crashed.append(command_id)
            raise RuntimeError("the process died here")
        real(command_id)

    monkeypatch.setattr(store, "mark_applied", crash)
    with pytest.raises(RuntimeError, match="died"):
        p.tick(T + timedelta(minutes=3), order=(leg,))
    monkeypatch.setattr(store, "mark_applied", real)
    p.restart()
    p.tick(T + timedelta(minutes=4))
    p.tick(T + timedelta(minutes=5))
    assert not store.pending_commands("carry") and not store.pending_commands("carry~2")
    assert len([e for e in store.events("carry~2", limit=500) if e["kind"] == LIQUIDATION_RESET]) == 1
    assert len([e for e in store.events("carry", limit=500) if e["kind"] == PAIR_REBASED]) == 1
    summed = _summed(p)
    assert all(p.rt[n].peak == pytest.approx(summed) for n in LEGS)
    assert store.pair_state("carry")["hwm"] == pytest.approx(D(str(summed)))


# --- (6) a stopped pair -------------------------------------------------------------------------------------------

def test_a_stopped_pair_is_reset_by_the_store_and_a_start_restores_the_summed_figures(store):
    p, iid = _liquidate(store)
    store.set_desired_state("carry", "stopped")
    _ral(store, iid)
    assert not store.pending_commands("carry") and not store.pending_commands("carry~2")  # applied here
    summed = _summed(p)
    assert store.pair_state("carry")["hwm"] == pytest.approx(D(str(summed)))
    assert liquidation_head(store, "carry~2") is None
    assert len([e for e in store.events("carry", limit=500) if e["kind"] == PAIR_REBASED]) == 1
    store.set_desired_state("carry", "running")
    p.restart()
    p.tick(T + timedelta(hours=1))
    assert all(p.rt[n].peak == pytest.approx(summed) for n in LEGS)
    assert p.rt["carry~2"].status != "halted" and p.rt["carry~2"].liquidated is None


def test_a_ral_waiting_when_the_pair_is_stopped_is_applied_with_its_half(store):
    p, iid = _liquidate(store)
    _ral(store, iid)
    store.set_desired_state("carry", "stopped")
    assert store.apply_waiting_ral("carry")
    assert not store.pending_commands("carry") and not store.pending_commands("carry~2")
    assert store.pair_state("carry")["hwm"] == pytest.approx(D(str(_summed(p))))


# --- (8) book figures, (9) X/Y on the summed equity, (11) a fresh run -------------------------------------------------

def test_the_books_figures_are_unchanged_by_a_pairs_reset(store):
    p, iid = _liquidate(store)
    led = store._gate_ledger()
    before = led.state()
    _ral(store, iid)
    p.tick(T + timedelta(minutes=2))
    p.tick(T + timedelta(minutes=3))
    assert led.state() == before


def test_x_and_y_are_on_the_summed_equity_and_the_net_loss_is_shown(store):
    p, iid = _liquidate(store, loss=400.0, unwind=100.0)
    liq = store.last_event("carry~2", ("risk_halt",))
    f = pair_ral_figures(store, "carry~2", liq)
    assert f["before"] == pytest.approx(10_000.0) and f["x"] == pytest.approx(400.0)
    assert f["y"] == pytest.approx(4.0)  # 400 of both legs' 10,000, not of the leg's own equity
    assert f["unwind"] == pytest.approx(100.0) and f["net"] == pytest.approx(300.0) and f["after"] == 9_700.0
    _ral(store, iid)
    p.tick(T + timedelta(minutes=2))
    words = store.last_event("carry~2", (LIQUIDATION_RESET,))["message"]
    assert "this leg lost 400.00, 4.0% of the pair" in words and "the other leg's unwind +100.00" in words
    assert "the pair -300.00 in all" in words


def test_a_ral_starts_a_fresh_run_on_both_legs(store):
    """RAL-ANCHOR on both legs (Advisor (d) 3): the fills before it are no cycle to pick up from; the coordinator's
    journal is flat with no trade."""
    p, iid = _liquidate(store)
    assert all(p.rt[n].last_fill_this_run() is not None for n in LEGS)
    _ral(store, iid)
    p.tick(T + timedelta(minutes=2))
    p.tick(T + timedelta(minutes=3))
    assert all(p.rt[n].last_fill_this_run() is None for n in LEGS)
    row = store.pair_state("carry")
    assert (row["state"], row["trade_id"]) == ("flat", None)


# --- the dashboard: the Strategy's page offers its liquidated leg's reset with the pair's figures --------------------

AUTH, SAME = ("pm", "test-pw"), {"origin": "http://testserver"}


@pytest.fixture
def client(store, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from sleeve_fund.dashboard import app as app_mod

    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    monkeypatch.setenv("TEARSHEET_DIR", str(tmp_path))
    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    monkeypatch.setattr(app_mod, "LEDGER", tmp_path / "idea_ledger.jsonl")
    return TestClient(app_mod.create_app(store))


def test_the_strategys_page_takes_both_causes_and_sends_the_legs_reset(store, client):
    p, _ = _liquidate(store)
    store.event("carry~2", "error", "incident", "Liquidated: carry~2 lost its position margin",
                ts=T + timedelta(minutes=1))
    iid = store.last_event("carry~2", ("incident",))["id"]
    page = client.get("/sleeves/carry", auth=AUTH).text
    assert "Reset after liquidation: carry, its leg carry~2" in page
    assert "lost 400.00" in page and "4.0% of both legs" in page
    assert 'name="why_clamp_did_not_prevent"' in page
    r = client.post("/sleeves/carry/incident-note", data={"incident": str(iid), "author": "Head of Engineering",
                                                          "why_stop_did_not_protect": WHY},
                    auth=AUTH, headers=SAME, follow_redirects=False)
    assert r.status_code == 303 and "saved" in r.headers["location"].lower()
    r = client.post("/sleeves/carry/command", data={"command": RAL, "incident": str(iid), "reason": "both written"},
                    auth=AUTH, headers=SAME, follow_redirects=False)
    assert "why+closing+the+pair" in r.headers["location"] and not store.pending_commands("carry~2")
    client.post("/sleeves/carry/incident-note", data={"incident": str(iid), "author": "Head of Engineering",
                                                      "why_stop_did_not_protect": WHY,
                                                      "why_clamp_did_not_prevent": CLAMP},
                auth=AUTH, headers=SAME, follow_redirects=False)
    r = client.post("/sleeves/carry/command", data={"command": RAL, "incident": str(iid), "reason": "both written"},
                    auth=AUTH, headers=SAME, follow_redirects=False)
    assert r.status_code == 303 and "command_error" not in r.headers["location"], r.headers["location"]
    assert [c["incident"] for c in store.pending_commands("carry~2")] == [iid]
