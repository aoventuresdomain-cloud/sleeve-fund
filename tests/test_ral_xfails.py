"""RAL, "reset after liquidation" (Platform Engineer 2, after the stop-safety PR): strict xfails written by QA BEFORE
the build, from the Advisor's rulings. The engineer makes each one pass (and removes its mark) before handing over.
v2: tightened to the Advisor's 18:17 ruling. Halt clearing (Stop/Start, restarts, Start refusal) is in
test_halt_clearing_xfails.py, which imports the helpers below.

Sources (every expectation below cites one):
- [R17:57] Advisor 6 Oct 17:57 to HoE: "#155 post-liquidation: strategy stays halted through resume/restart until PM
  explicit "reset after liquidation" (new high-water mark from remainder) or retire; reset requires an incident note
  on why the half-liq stop failed. (a) text: "Position margin lost (liquidated): X, Y% of strategy equity"."
- [HoE17:55] RAL is a labelled action, not a resume; a new high-water mark from the remaining equity; allowed only once
  an incident note explains why the half-liquidation stop failed.
- [R18:17] Advisor 6 Oct 18:17 to QA ("RAL + halts", agrees QA's 8 defaults): the engine opens an incident on every
  liquidation (paper + outage replay); RAL refused until the incident has a written cause incl. the required field
  "why the half-liquidation stop did not protect the position", the note names its author. RAL resets the strategy's
  HWM + day baseline to the remaining equity; NEVER the book-level limits (15% DD, 3% daily, open-risk total). Y% on
  MTM equity just before the liquidation; X = margin + entry fee + liq fee. Retire = Stop + Archive until P2-6. Track
  record never reset (Sharpe, trials, G1/G2 evidence, fills-vs-model, trade log continuous). RAL only from a PM UI
  action (never replay/automation/API retry); repeat = no-op; journal who/when/incident id/equity before + after.
- [B63] (HoE) an incident is an error-severity event with kind "incident" plus an entry in the alerts inbox.

ASSUMED INTERFACES (adapt the names, never the assertions):
- RAL = "reset_after_liquidation", a PM command: Store.command(name, RAL, reason, actor="PM", incident=<events.id>).
  A refused RAL raises ValueError in words at that call, so nothing is queued. Only actor "PM" may send it.
- The engine's incident: an events row for the strategy, level "error", kind "incident", at or after the liquidation
  (so it is in the alerts inbox). Until the engine opens one, the RAL tests write a stand-in the same way (the
  engine's own incident is pinned by test_the_engine_opens_an_incident_on_every_liquidation).
- The note: Store.write_incident_note(incident_id, *, author: str, why_stop_did_not_protect: str) -> None, raising
  ValueError (naming the field) when the field or the author is blank. The dashboard offers it on the liquidated
  strategy's page (inputs named "author" and "why_stop_did_not_protect", labelled "Why the half-liquidation stop did
  not protect the position").
- The dashboard sends RAL through POST /sleeves/{name}/command with command=RAL, incident=<id>, reason; refusals come
  back as command_error. A retry of the same RAL is refused.
- Applied (by the runtime on its next tick, or by the store at once), RAL sets the status running, the drawdown
  reference (SleeveRuntime.peak, rebuilt on a restart) and the day's loss baseline to the remaining equity, and writes
  one info event of kind "liquidation_reset" (the kind #164 e14acec reads to end a liquidation) naming the PM, the
  incident ("#<id>"), the note's author, the old and the new high-water mark,
  and the equity just before the liquidation and after it (each "{:,.2f}"). The decision row Store.command writes
  (action RAL, actor, ts, reason) names the incident ("#<id>").
- The liquidation halt's status_reason contains the ruled text (a).
Liquidations are real: a recorded paper session (a short at about 60,629 with a 2% stop-loss, then a 60% gap up through
the stop and the liquidation price) replayed through the paper runtime, as tests/test_long_short.py does. The stop is
there so the setups also run under the stop-safety gate (stopless capped at 1x). Where a head books that gap close as a
stop-loss (3be572a, ba4f533), the setups anchor on the halt it caused on the same tick; the liquidation journalling is
pinned on its own (test_a_stop_filled_through_the_liquidation_price_is_journalled_as_a_liquidation).

Copy this file into tests/ (it imports test_long_short and test_replay). With TEST_DATABASE_URL set the store is
Postgres, else SQLite. Tests named test_guard_... are GUARDS: no mark, they pass on 3be572a and must keep passing.
"""

import inspect
import os
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from test_long_short import PERP, _meta, _record

REASON = "QA RAL: not built yet (Advisor 17:57)"
xf = pytest.mark.xfail(strict=True, reason=REASON)

RAL = "reset_after_liquidation"
# The journal kind RAL writes, the contract #164 and #155 read: sleeve_fund.store's constant (#164 60cb17d on), else
# trading's (#164 e14acec), else the literal where a head lacks it (3be572a, ba4f533, c998d08, c7a73f1).
try:
    from sleeve_fund.store import LIQUIDATION_RESET
except ImportError:
    try:
        from sleeve_fund.dashboard.trading import LIQUIDATION_RESET
    except ImportError:
        LIQUIDATION_RESET = "liquidation_reset"
NAME = "ping-pong-test"  # the recorded session's strategy (test_long_short._meta)
# ping_pong on a perp, balanced: 2x, 33% margin cap, with a 2% stop-loss. The stop lets it start under the
# stop-safety gate (a stopless strategy is capped at 1x) and keeps its open risk under the interim 5%-of-book limit
# (2% of a 6,600 notional, or of 20,000 with full margin); the 60% gap goes through the stop AND the liquidation price.
PARAMS = {"rise": 0.01, "dip": 0.005, "stop_loss": 0.02, **PERP}
GAP_PX = 97_440.0  # 60,900 x 1.6: where the gap leaves the price
NEXT_DAY = datetime(2025, 10, 4, 1, 0, tzinfo=timezone.utc)  # the recording is 3 Oct 00:00-00:30 UTC
WHY_FIELD = "why the half-liquidation stop did not protect the position"
WHY = ("The price gapped 60% in one trade, through the resting stop and the liquidation price, so the stop could only "
       "fill past bankruptcy")
AUTHOR = "Head of Engineering"
# [R17:57] (a), with X an amount (separators, an optional currency) and Y a percentage.
TEXT = re.compile(r"Position margin lost \(liquidated\): [^\d]*(\d[\d,]*(?:\.\d+)?)[^,]*, (\d+(?:\.\d+)?)% of "
                  r"strategy equity")
AUTH = ("pm", "test-pw")  # the dashboard user
SAME = {"origin": "http://testserver"}


@pytest.fixture
def store(tmp_path):
    # As test_sleeve_runtime: Postgres when TEST_DATABASE_URL is set, else SQLite.
    from sleeve_fund.store import Store

    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        from sleeve_fund.store import make_engine, metadata

        engine = make_engine(url)
        metadata.drop_all(engine)
        return Store(engine=engine)
    return Store(f"sqlite:///{tmp_path}/t.db")


@pytest.fixture
def client(store, tmp_path, monkeypatch):
    # As test_dashboard's client, on this store.
    from fastapi.testclient import TestClient

    from sleeve_fund.dashboard import app as app_mod

    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    monkeypatch.setenv("TEARSHEET_DIR", str(tmp_path))
    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    monkeypatch.setattr(app_mod, "LEDGER", tmp_path / "idea_ledger.jsonl")
    return TestClient(app_mod.create_app(store))


# --- helpers: a real liquidation, restarts, the runtime, incidents ------------------------------------------------

def _record_at(path, meta, legs, px, hours):
    """test_long_short._record, starting `hours` after test_replay.START, so a restart is later than what it follows."""
    import test_replay

    start = test_replay.START
    test_replay.START = start + int(hours * 3600) * 1_000_000_000
    try:
        _record(path, meta, legs, px=px)
    finally:
        test_replay.START = start


def _replay_into(store, path):
    from sleeve_fund.research.replay import replay

    if any(s.name == NAME for s in store.sleeves()):
        store.create_sleeve = lambda **kw: store.sleeve(kw["name"])  # it is there already
    try:
        return replay(path, store=store)
    finally:
        store.__dict__.pop("create_sleeve", None)


def _gap_closed(orders, side):
    """Setup check (passes on every head): the gap closed the position past its liquidation price, by the liquidation
    or by the stop it went through. Returns (the closing order, the entry it closed)."""
    close = orders[-1]
    assert close["intent"] in ("liquidation", "stop_loss") and close["side"] == side, [(o["side"], o["intent"])
                                                                                       for o in orders]
    entry = [o for o in orders if o["intent"] == "entry" and o["side"] != side][-1]
    move = close["avg_px"] / entry["avg_px"] - 1
    assert (move if side == "BUY" else -move) > 0.5, (close["avg_px"], entry["avg_px"])  # past 2x liquidation (~49%)
    return close, entry


def _gap_event(store, after_id=0):
    """The liquidation's event: the engine's "liquidation" event; on a head that books a stop filled through the
    liquidation price as a stop-loss (3be572a, ba4f533), the halt the gap caused, on the same tick."""
    for kinds in (("liquidation",), ("risk_halt",)):
        ev = store.last_event(NAME, kinds)
        if ev is not None and ev["id"] > after_id:
            return ev
    raise AssertionError("the gap left no liquidation or halt event")


def _liquidate(tmp_path, store):
    """A paper short at about 60,629 with a 2% stop, then a 60% gap up through the stop and the liquidation price:
    closed past bankruptcy and halted. On 3be572a (balanced, 33% margin cap) it keeps about 6,733 of 10,000. The
    strategy may exist already."""
    path = tmp_path / "liq.jsonl.gz"
    _record_at(path, _meta(10_000, PARAMS), [(5, 0.0), (20, 0.015), (0, 0.6), (5, 0.0)], 60_000.0, 0)
    orders = _replay_into(store, path)
    close, entry = _gap_closed(orders, "BUY")
    s = store.sleeve(NAME)
    assert s.status == "halted", (s.status, s.status_reason)
    book = store.journal_book(NAME, 10_000)
    assert book["qty"] == 0
    liq = _gap_event(store)
    before = store.equity_at_or_before(NAME, liq["ts"] - timedelta(seconds=1))["equity"]  # MTM just before
    return SimpleNamespace(rem=book["cash"], peak=store.peak_equity(NAME), liq=liq, before=before, close=close,
                           entry=entry, orders=len(orders), reason=s.status_reason)


def _restart(tmp_path, store, legs, hours, px=GAP_PX, tag="restart"):
    """The paper process starting again on the same journal: a recorded session replayed into the store. Returns the
    orders it sent."""
    before = len(store.orders(NAME, limit=100_000))
    path = tmp_path / f"{tag}.jsonl.gz"
    _record_at(path, _meta(store.journal_book(NAME, 10_000)["cash"], PARAMS), legs, px, hours)
    return _replay_into(store, path)[before:]


def _runtime(store, at, equity, price=GAP_PX, name=NAME):
    """A fresh paper runtime on the journal (a restart), started and ticked once at `at` with a flat book."""
    from sleeve_fund.paper.runtime import SleeveRuntime

    rt = SleeveRuntime(store, name, now=lambda: at)
    rt.on_start(0.0005)
    rt.tick(equity=equity, cash=equity, qty=0.0, price=price)
    return rt


def _tick(rt, at, equity, price=GAP_PX):
    rt.now = lambda: at
    rt.tick(equity=equity, cash=equity, qty=0.0, price=price)
    return rt


def _peak(store):
    from sleeve_fund.paper.runtime import SleeveRuntime

    return SleeveRuntime(store, NAME).peak


def _incidents_since(store, ts, sleeve=NAME):
    return [e for e in store.events(sleeve, limit=10_000) if e["kind"] == "incident" and e["ts"] >= ts]


def _incident(store, liq=None, at=None, sleeve=NAME):
    """The liquidation's incident [R18:17]: the engine's, at or after `liq`; else a stand-in written here [B63], so the
    RAL tests reach RAL before the engine part is built. `at`: write one at that time (another strategy's, or one
    from before this liquidation). Returns its id; it is in the alerts inbox."""
    ev = None
    if at is None:
        found = _incidents_since(store, liq["ts"], sleeve)
        ev = found[-1] if found else None  # newest first: the earliest after the liquidation
        at = liq["ts"] + timedelta(minutes=2)
    if ev is None:
        store.event(sleeve, "error", "incident", f"Liquidated: {sleeve} lost its position margin", ts=at)
        ev = store.last_event(sleeve, ("incident",))
    assert any(a["id"] == ev["id"] for a in store.alerts(limit=500))  # in the inbox [B63]
    return ev["id"]


def _note(store, iid, why=WHY, author=AUTHOR):
    # The interface is checked first, so a missing build fails on an assertion, not an AttributeError.
    assert hasattr(store, "write_incident_note"), "not built: Store.write_incident_note (the incident's note)"
    store.write_incident_note(iid, author=author, why_stop_did_not_protect=why)


def _noted(store, liq=None, at=None, sleeve=NAME):
    iid = _incident(store, liq, at, sleeve)
    _note(store, iid)
    return iid


def _ral(store, incident=None, actor="PM", reason="reset after liquidation: incident note written"):
    if incident is not None:  # checked first, so a missing build fails on an assertion, not a TypeError
        assert "incident" in inspect.signature(store.command).parameters, "not built: Store.command(..., incident=)"
    kw = {} if incident is None else {"incident": incident}
    store.command(NAME, RAL, reason, actor=actor, **kw)


def _refused(store, words, **kw):
    """The RAL is refused at the call, in words matching `words` (the command's own name doesn't count)."""
    with pytest.raises(ValueError) as e:
        _ral(store, **kw)
    msg = str(e.value).lower().replace(RAL, "")
    assert re.search(words, msg), str(e.value)


def _state(store):
    s = store.sleeve(NAME)
    return {"status": s.status, "reason": s.status_reason, "desired": s.desired_state, "until": s.paused_until,
            "pending": [c["command"] for c in store.pending_commands(NAME)], "peak": _peak(store)}


def _ral_events(store, name=NAME):
    return [e for e in store.events(name, limit=10_000) if e["kind"] == LIQUIDATION_RESET]


def _ral_decisions(store):
    return store.decisions(NAME, limit=10_000, action=RAL)


def _nothing_changed(store, before, rem):
    """Refused: the strategy, its queue, its drawdown reference and its journal are as they were, and a restart and a
    tick the next day leave it halted."""
    assert _state(store) == before
    assert not _ral_events(store) and not _ral_decisions(store)
    rt = _runtime(store, NEXT_DAY, rem)
    assert rt.status == "halted" and store.sleeve(NAME).status == "halted"


def _post(client, data, headers=SAME, name=NAME):
    return client.post(f"/sleeves/{name}/command", data=data, auth=AUTH, headers=headers, follow_redirects=False)


# --- halted through resume and restart [R17:57] -------------------------------------------------------------------

def test_guard_after_a_liquidation_a_restart_leaves_it_halted(store, tmp_path):
    """GUARD (passes on 3be572a) for [R17:57] "stays halted through ... restart": the process starting again keeps the
    halt and its reason, and sends nothing. (Stop/Start, deploy and supervisor restarts: test_halt_clearing_xfails.)"""
    f = _liquidate(tmp_path, store)
    new = _restart(tmp_path, store, [(5, 0.0)], hours=25)
    s = store.sleeve(NAME)
    assert s.status == "halted" and s.status_reason == f.reason, (s.status, s.status_reason)
    assert new == []


@xf
def test_after_a_liquidation_a_plain_resume_leaves_it_halted_through_a_restart(store, tmp_path):
    """[R17:57] "stays halted through resume/restart"; [HoE17:55] RAL is not a resume. A resume is refused in words,
    or is taken and leaves it halted: either way no trade, and the drawdown reference is not reset."""
    f = _liquidate(tmp_path, store)
    try:
        store.command(NAME, "resume", "try again")
    except ValueError:
        pass
    new = _restart(tmp_path, store, [(5, 0.0)], hours=25)
    assert new == [], [(o["side"], o["intent"]) for o in new]
    s = store.sleeve(NAME)
    assert s.status == "halted" and s.status_reason == f.reason, (s.status, s.status_reason)
    assert store.last_event(NAME, ("drawdown_reset",)) is None
    assert _peak(store) == pytest.approx(f.peak)


@xf
@pytest.mark.parametrize("margin", ["capped", "full"])
def test_the_liquidation_halt_says_how_much_position_margin_was_lost(store, tmp_path, request, margin):
    """[R17:57] (a), [R18:17]: the halt reads "Position margin lost (liquidated): X, Y% of strategy equity", X = the
    position's isolated margin + its entry fee + its liquidation fee (to the cent), Y = X over the mark-to-market
    equity just before the liquidation (to its printed rounding). "capped": balanced's 33% margin cap (X about
    3,328.7, Y about 33%); "full": the whole equity as margin."""
    from sleeve_fund import markets, risk

    if margin == "full":
        request.getfixturevalue("full_margin")
    f = _liquidate(tmp_path, store)
    liq, entry = f.close, f.entry
    fills = store.fills(NAME, limit=1000)
    efills = [x for x in fills if x["order_id"] == entry["order_id"]]
    lfills = [x for x in fills if x["order_id"] == liq["order_id"]]
    qty = sum(x["qty"] for x in efills)
    px = sum(x["qty"] * x["price"] for x in efills) / qty
    want = (markets.isolated_margin(qty, px, risk.profile("balanced").max_leverage)
            + sum(x["fee"] for x in efills) + sum(x["fee"] for x in lfills))

    s = store.sleeve(NAME)
    m = TEXT.search(s.status_reason)
    assert m, s.status_reason
    x, y = float(m.group(1).replace(",", "")), m.group(2)
    assert x == pytest.approx(want, abs=0.02), (x, want, s.status_reason)
    decimals = len(y.split(".")[1]) if "." in y else 0
    assert abs(float(y) - 100 * x / f.before) <= 0.5 * 10 ** -decimals + 1e-9, (y, x, f.before)


@xf
def test_a_stop_filled_through_the_liquidation_price_is_journalled_as_a_liquidation(store, tmp_path):
    """[R17:57] / [R18:17] (the incident's field is "why the half-liquidation stop did not protect the position"): a
    gap through the resting stop AND the liquidation price is a liquidation (the venue takes the position at its
    liquidation price; the stop never protected it), journalled as one: the closing order's intent is "liquidation",
    with an error event of kind "liquidation". On 3be572a and ba4f533 it is journalled as a stop-loss, so nothing
    downstream (halt text, incident, RAL) knows it was a liquidation."""
    f = _liquidate(tmp_path, store)
    assert f.close["intent"] == "liquidation", (f.close["intent"], f.close["reason"])
    ev = store.last_event(NAME, ("liquidation",))
    assert ev is not None and ev["level"] == "error", [(e["kind"], e["message"][:60])
                                                       for e in store.events(NAME, limit=20, min_level="warning")]


@xf
def test_a_drawdown_halt_on_the_same_tick_as_a_liquidation_keeps_the_liquidation_halt(store, tmp_path):
    """[R17:57], [R18:17] (PE2's fix): the gap also breaches the 20% drawdown halt on the same tick. The halt must
    still be the liquidation's (the ruled text, cleared only by RAL), not an ordinary drawdown halt that a plain
    resume would clear: after a resume and a restart it is halted with the same text and sends nothing."""
    f = _liquidate(tmp_path, store)
    assert 1 - f.rem / f.peak >= 0.2  # setup: the same tick breaches the drawdown halt (passes on every head)
    s = store.sleeve(NAME)
    assert TEXT.search(s.status_reason), s.status_reason
    try:
        store.command(NAME, "resume", "it was only a drawdown halt")
    except ValueError:
        pass
    new = _restart(tmp_path, store, [(5, 0.0)], hours=25)
    assert new == [], [(o["side"], o["intent"]) for o in new]
    s2 = store.sleeve(NAME)
    assert s2.status == "halted" and s2.status_reason == s.status_reason, (s2.status, s2.status_reason)


# --- the incident and its note [R17:57, HoE17:55, R18:17] ---------------------------------------------------------

@xf
@pytest.mark.parametrize("where", ["paper", "restart_after_an_outage"])
def test_the_engine_opens_an_incident_on_every_liquidation(store, tmp_path, where):
    """[R18:17] "engine opens incident on every liquidation (paper + outage replay)" [B63]: one error event of kind
    "incident" for the strategy, at or after the liquidation, unacknowledged in the alerts inbox, naming the
    liquidation. "restart_after_an_outage": the process was down while the price went through the short's
    liquidation price; it restarts at 97,440 holding the short (the #146 outage replay's case, NA-3: on these heads a
    restart is the only entry point). The venue liquidated it, so the journal has a liquidation too."""
    if where == "paper":
        liq_ts = _liquidate(tmp_path, store).liq["ts"]
    else:
        import test_replay
        from sleeve_fund.store import replay_book

        opened = datetime.fromtimestamp(test_replay.START / 1e9 - 3600, tz=timezone.utc)
        store.create_sleeve(name=NAME, strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                            starting_balance=10_000, params=PARAMS)
        store.record_equity(NAME, equity=10_000, cash=10_000, qty=0, price=60_600, benchmark=10_000, ts=opened)
        store.record_fill(NAME, side="SELL", qty=0.11, price=60_600.0, fee=3.3, order_id="carried",
                          trade_id="carried", ts=opened)
        book = replay_book([{"side": "SELL", "qty": 0.11, "price": 60_600.0, "fee": 3.3, "order_id": "carried",
                             "trade_id": "carried", "ts": opened}], 10_000.0)
        path = tmp_path / "outage.jsonl.gz"
        _record_at(path, _meta(book["cash"] + book["qty"] * book["entry_px"], PARAMS), [(5, 0.0)], GAP_PX, 0)
        _replay_into(store, path)
        assert store.sleeve(NAME).status == "halted"  # setup: passes on 3be572a
        liq = store.last_event(NAME, ("liquidation",))
        assert liq is not None, [(e["kind"], e["message"][:60]) for e in store.events(NAME, limit=50, min_level="error")]
        liq_ts = liq["ts"]
    found = _incidents_since(store, liq_ts)
    assert len(found) == 1, [(e["kind"], e["message"][:60]) for e in store.events(NAME, limit=50, min_level="error")]
    assert found[0]["level"] == "error" and "liquidat" in found[0]["message"].lower(), found[0]
    assert any(a["id"] == found[0]["id"] for a in store.alerts(limit=500))


@xf
@pytest.mark.parametrize("missing", ["why", "author"])
def test_the_note_needs_why_the_half_liquidation_stop_did_not_protect_the_position_and_its_author(store, tmp_path,
                                                                                                 missing):
    """[R18:17] "required field 'why the half-liquidation stop did not protect the position'", "note names author":
    a note with that field blank, or with no author, is refused naming what is missing, and RAL stays refused."""
    f = _liquidate(tmp_path, store)
    iid = _incident(store, f.liq)
    with pytest.raises(ValueError) as e:
        _note(store, iid, why="  " if missing == "why" else WHY, author="" if missing == "author" else AUTHOR)
    assert (WHY_FIELD if missing == "why" else "author") in str(e.value).lower(), str(e.value)
    before = _state(store)
    _refused(store, r"note|why the half-liquidation stop", incident=iid)
    _nothing_changed(store, before, f.rem)


@xf
@pytest.mark.parametrize("which", ["no_incident_named", "incident_without_its_note"])
def test_ral_without_a_written_note_is_refused_and_changes_nothing(store, tmp_path, client, which):
    """[R17:57] "reset requires an incident note"; [R18:17] "RAL refused until incident has written cause": with no
    incident named, or the incident named but no note written on it, it is refused saying so, from the store and from
    the dashboard, and nothing changes."""
    f = _liquidate(tmp_path, store)
    kw = {} if which == "no_incident_named" else {"incident": _incident(store, f.liq)}
    words = r"incident" if which == "no_incident_named" else r"note|why the half-liquidation stop"
    before = _state(store)
    _refused(store, words, **kw)
    r = _post(client, {"command": RAL, "reason": "reset it", **{k: str(v) for k, v in kw.items()}})
    assert r.status_code == 303 and "command_error" in r.headers["location"], r.headers.get("location")
    assert re.search(words, r.headers["location"].lower().replace(RAL, "").replace("+", " ")), r.headers["location"]
    _nothing_changed(store, before, f.rem)


@xf
@pytest.mark.parametrize("which", ["another_strategys", "before_this_liquidation", "not_an_incident", "no_such_event"])
def test_ral_with_an_incident_that_is_not_about_this_liquidation_is_refused(store, tmp_path, which):
    """[R17:57] the note is on THIS liquidation: another strategy's noted incident, a noted incident from before this
    liquidation, an event that is not an incident (the liquidation's own event) or no event at all is refused,
    saying so, and nothing changes."""
    if which == "before_this_liquidation":
        store.create_sleeve(name=NAME, strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                            starting_balance=10_000, params=PARAMS)
        ref = _noted(store, at=datetime(2025, 10, 2, 23, 0, tzinfo=timezone.utc))  # an hour before the session
        f = _liquidate(tmp_path, store)
    else:
        f = _liquidate(tmp_path, store)
        if which == "another_strategys":
            store.create_sleeve(name="other", strategy="ping_pong", instrument="BTC/USD",
                                bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=10_000, params=PARAMS)
            ref = _noted(store, at=f.liq["ts"] + timedelta(minutes=2), sleeve="other")
        elif which == "not_an_incident":
            ref = f.liq["id"]
        else:
            ref = store.last_event_id() + 1000
    before = _state(store)
    _refused(store, r"incident", incident=ref)
    _nothing_changed(store, before, f.rem)


# --- RAL with the note [R17:57, HoE17:55, R18:17] -----------------------------------------------------------------

@xf
def test_ral_with_the_note_sets_it_running_and_it_trades_again(store, tmp_path):
    """[R17:57] the PM's explicit RAL lifts the halt: the next process runs it and it trades on its next signal
    (ping_pong buys on the first bar). The recorded session then ends, so its process stops ("process stopped"):
    the status is read as not halted here, and as running in the tests that tick a runtime."""
    f = _liquidate(tmp_path, store)
    _ral(store, incident=_noted(store, f.liq))
    new = _restart(tmp_path, store, [(5, 0.0)], hours=25)
    assert [(o["side"], o["intent"]) for o in new][:1] == [("BUY", "entry")], new
    s = store.sleeve(NAME)
    assert s.status != "halted" and s.desired_state == "running", (s.status, s.status_reason)
    assert len(_ral_events(store)) == 1


@xf
def test_ral_starts_a_new_high_water_mark_at_the_remaining_equity(store, tmp_path):
    """[R17:57] "new high-water mark from remainder"; [R18:17] "RAL resets strategy HWM ... to remaining equity":
    straight after, the drawdown reads 0%, the drawdown halt doesn't act, a restart keeps it, and a 3% fall (35% below
    the old mark) doesn't halt."""
    from sleeve_fund import risk

    f = _liquidate(tmp_path, store)
    _ral(store, incident=_noted(store, f.liq))
    rt = _runtime(store, NEXT_DAY, f.rem)
    assert rt.status == "running", store.sleeve(NAME).status_reason
    assert rt.peak == pytest.approx(f.rem, abs=0.01)
    assert 1 - f.rem / rt.peak == pytest.approx(0.0, abs=1e-6)  # drawdown 0%
    assert risk.check(rt.profile, f.rem, rt.peak, f.rem) is None
    assert _peak(store) == pytest.approx(f.rem, abs=0.01)  # what a restart measures from
    _tick(rt, NEXT_DAY + timedelta(minutes=5), f.rem * 0.97)
    assert rt.status == "running" and store.sleeve(NAME).status == "running", store.sleeve(NAME).status_reason


@xf
def test_ral_resets_the_day_baseline_to_the_remaining_equity(store, tmp_path):
    """[R18:17] "RAL resets strategy ... day baseline to remaining equity": on the liquidation's own UTC day (which
    opened at about 10,000, 33% above the remainder) it runs, not paused; a 4% fall stays running, a restart keeps
    the baseline, and 6% below the remainder pauses."""
    f = _liquidate(tmp_path, store)
    _ral(store, incident=_noted(store, f.liq))
    at = f.liq["ts"] + timedelta(minutes=15)  # 3 Oct, 00:40 UTC
    rt = _runtime(store, at, f.rem)
    assert rt.status == "running", store.sleeve(NAME).status_reason
    _tick(rt, at + timedelta(minutes=5), f.rem * 0.96)
    assert rt.status == "running", store.sleeve(NAME).status_reason
    rt2 = _runtime(store, at + timedelta(minutes=10), f.rem * 0.97)  # a restart the same day
    assert rt2.status == "running", store.sleeve(NAME).status_reason
    _tick(rt2, at + timedelta(minutes=15), f.rem * 0.94)
    assert rt2.status == "paused", (rt2.status, store.sleeve(NAME).status_reason)


@xf
def test_ral_keeps_the_old_high_water_mark_in_history(store, tmp_path):
    """[R17:57] / [HoE17:55]: a NEW high-water mark, so the old one is kept: no mark is deleted or moved, the highest
    mark ever and the worst drawdown still show the liquidation, and the RAL's event names the old and the new
    high-water mark. It is labelled, not a resume: no pm_resume."""
    f = _liquidate(tmp_path, store)
    marks = len(store.equity_series(NAME, limit=100_000))
    worst = store.max_drawdown(NAME, 10_000)
    _ral(store, incident=_noted(store, f.liq))
    _runtime(store, NEXT_DAY, f.rem)
    assert len(store.equity_series(NAME, limit=100_000)) == marks + 1
    assert store.peak_equity(NAME) == pytest.approx(f.peak)
    assert store.max_drawdown(NAME, 10_000) == pytest.approx(worst) and worst > 0.3
    ev = _ral_events(store)
    assert len(ev) == 1, ev
    assert f"{f.peak:,.2f}" in ev[0]["message"] and f"{f.rem:,.2f}" in ev[0]["message"], ev[0]["message"]
    assert store.last_event(NAME, ("pm_resume",)) is None


@xf
def test_ral_is_journalled_with_who_when_the_incident_and_the_equity_before_and_after(store, tmp_path, client):
    """[R18:17] "journal who/when/incident id/equity before+after"; [HoE17:55] a labelled action. Sent from the
    dashboard by the PM: one decision row, action RAL, actor PM, its time, naming the incident; the applied event
    names the incident, the note's author, the equity just before the liquidation and the remainder after it."""
    f = _liquidate(tmp_path, store)
    iid = _noted(store, f.liq)
    t0 = datetime.now(timezone.utc)
    r = _post(client, {"command": RAL, "incident": str(iid), "reason": "incident note written"})
    t1 = datetime.now(timezone.utc)
    assert r.status_code == 303 and "command_error" not in r.headers["location"], r.headers["location"]
    d = _ral_decisions(store)
    assert len(d) == 1, d
    assert d[0]["actor"] == "PM" and t0 - timedelta(seconds=1) <= d[0]["ts"] <= t1 + timedelta(seconds=1), d[0]
    assert f"#{iid}" in d[0]["reason"], d[0]["reason"]
    _runtime(store, NEXT_DAY, f.rem)
    ev = _ral_events(store)
    assert len(ev) == 1, ev
    msg = ev[0]["message"]
    assert f"#{iid}" in msg and AUTHOR in msg, msg
    assert f"{f.before:,.2f}" in msg and f"{f.rem:,.2f}" in msg, (msg, f.before, f.rem)
    assert store.sleeve(NAME).status == "running"


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=REASON)
def test_ral_writes_exactly_one_liquidation_reset_event_with_its_author_and_time_after_the_liquidation(store,
                                                                                                         tmp_path):
    """[R18:17] "journal who/when"; #164 (e14acec) judges "liquidated" from the journal: a liquidation stays live until
    an event of kind "liquidation_reset" is written after it. RAL writes exactly one, after the liquidation (by id and
    by time), naming the PM who reset it and the note's author, stamped when the PM sent it or when the runtime
    applied it; the tick after it and a restart write no second one. (Nothing else writes that kind:
    test_guard_nothing_but_ral_writes_a_liquidation_reset, in test_halt_clearing_xfails.py.)"""
    f = _liquidate(tmp_path, store)
    iid = _noted(store, f.liq)
    t0 = datetime.now(timezone.utc)
    _ral(store, incident=iid)
    t1 = datetime.now(timezone.utc)
    rt = _runtime(store, NEXT_DAY, f.rem)  # applied on its next tick (or already, by the store)
    _tick(rt, NEXT_DAY + timedelta(minutes=5), f.rem)
    _runtime(store, NEXT_DAY + timedelta(minutes=10), f.rem)  # a restart
    ev = _ral_events(store)
    assert len(ev) == 1, ev
    e = ev[0]
    assert e["id"] > f.liq["id"] and e["ts"] > f.liq["ts"], (e, f.liq)
    sent = t0 - timedelta(seconds=1) <= e["ts"] <= t1 + timedelta(seconds=1)
    assert sent or e["ts"] == NEXT_DAY, (e["ts"], t0, NEXT_DAY)
    assert re.search(r"\bPM\b", e["message"]) and AUTHOR in e["message"], e["message"]


@xf
def test_ral_never_resets_the_book_level_figures(store, tmp_path):
    """[R18:17] "NEVER book-level limits (15% DD, 3% daily, open-risk total)", and the loss stays in the book's
    figures: the book curve the book-level limits read still has the loss (drawdown from 10,000 to the remainder,
    and its worst), the book's start is unchanged, and RAL is no book or strategy reset (no reset row, no run put
    away, nothing archived)."""
    from sleeve_fund.dashboard import book as bookm

    f = _liquidate(tmp_path, store)
    start = store.book_start()
    _ral(store, incident=_noted(store, f.liq))
    _runtime(store, NEXT_DAY, f.rem)
    curve = bookm.book_curve([{"sleeve": store.sleeve(NAME)}], {NAME: bookm.daily(store, NAME)})
    assert len(_ral_events(store)) == 1
    assert curve["equity"].iloc[-1] == pytest.approx(f.rem, abs=0.01)
    assert curve["drawdown"].iloc[-1] == pytest.approx(1 - f.rem / 10_000, abs=1e-4)
    assert curve["drawdown"].max() >= 1 - f.rem / 10_000 - 1e-4
    assert store.book_start() == start and store.reset_runs() == {} and not store.pending_resets()
    assert not store.archived() and not store.events_of(("book_cleared",))


@xf
def test_ral_never_resets_the_track_record(store, tmp_path):
    """[R18:17] "Track record never reset (Sharpe, trials, G1/G2 evidence, fills-vs-model, trade log continuous)":
    after RAL the strategy keeps its name, creation time (the G2 six-week clock), fills, orders and trades, its daily
    curve (what Sharpe is computed on) from the first day, its P&L and worst drawdown, the trials table, and the G2
    row that shows the halt."""
    from sleeve_fund.dashboard import book as bookm
    from sleeve_fund.dashboard.gates import path_to_live
    from sleeve_fund.dashboard.metrics import sleeve_summary

    f = _liquidate(tmp_path, store)
    s0 = store.sleeve(NAME)
    x0 = sleeve_summary(store, s0)
    counts = (len(store.fills(NAME, limit=10_000)), len(store.orders(NAME, limit=10_000)), len(store.trials()))
    first_day = bookm.daily(store, NAME).index[0]
    _ral(store, incident=_noted(store, f.liq))
    _runtime(store, NEXT_DAY, f.rem)
    assert len(_ral_events(store)) == 1
    s1 = store.sleeve(NAME)
    x1 = sleeve_summary(store, s1)
    assert s1.created_at == s0.created_at
    assert (len(store.fills(NAME, limit=10_000)), len(store.orders(NAME, limit=10_000)), len(store.trials())) == counts
    assert x1["trades"] == x0["trades"] and x1["fills"] == x0["fills"]
    assert x1["pnl"] == pytest.approx(f.rem - 10_000, abs=0.01)
    assert x1["max_drawdown"] == pytest.approx(x0["max_drawdown"])
    daily = bookm.daily(store, NAME)
    assert daily.index[0] == first_day and len(daily) == 2
    rows = path_to_live(store, x1, None, [], NEXT_DAY + timedelta(hours=1))
    halt_row = next(r for r in rows if r["label"].startswith("No risk halt"))
    assert halt_row["ok"] is False and "halted on 03 Oct 2025" in halt_row["detail"], halt_row


# --- only on a liquidation; retire; twice; a second liquidation; only the PM [R17:57, R18:17] ----------------------

def _ordinary(store, kind, name=NAME, t=datetime(2025, 10, 3, 10, 0, tzinfo=timezone.utc)):
    """A strategy halted or paused by its own guard, never liquidated: a drawdown halt (21% from 10,000) or a
    daily-loss pause (6% in a day), at `t`. Returns the runtime."""
    store.create_sleeve(name=name, strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params=PARAMS)
    rt = _runtime(store, t - timedelta(minutes=5), 10_000.0, price=60_000.0, name=name)
    _tick(rt, t, 7_900.0 if kind == "drawdown_halt" else 9_400.0, price=60_000.0)
    s = store.sleeve(name)
    assert s.status == ("halted" if kind == "drawdown_halt" else "paused"), (s.status, s.status_reason)
    return rt


@xf
@pytest.mark.parametrize("kind", ["drawdown_halt", "daily_pause", "drawdown_halt_after_an_earlier_ral"])
def test_ral_is_refused_on_an_ordinary_halt(store, tmp_path, kind):
    """[R17:57], [R18:17] "a halt is cleared only by its own action": RAL is the liquidation's. On a drawdown halt or
    a daily-loss pause it is refused, saying it isn't after a liquidation, even with a noted incident, and nothing
    changes. The third case: liquidated, reset, then halted by an ordinary 21% drawdown from the new mark."""
    if kind == "drawdown_halt_after_an_earlier_ral":
        f = _liquidate(tmp_path, store)
        _ral(store, incident=_noted(store, f.liq))
        rt = _runtime(store, NEXT_DAY, f.rem)
        assert rt.status == "running"
        _tick(rt, NEXT_DAY + timedelta(minutes=5), f.rem * 0.79)
        assert store.sleeve(NAME).status == "halted", store.sleeve(NAME).status_reason
        at = NEXT_DAY + timedelta(minutes=6)
    else:
        _ordinary(store, kind)
        at = datetime(2025, 10, 3, 10, 1, tzinfo=timezone.utc)
    iid = _noted(store, at=at)
    before = _state(store)
    _refused(store, r"liquidat", incident=iid)
    assert _state(store) == before
    assert len(_ral_events(store)) == (1 if kind == "drawdown_halt_after_an_earlier_ral" else 0)


def test_guard_retire_still_works_instead_of_ral(store, tmp_path, client):
    """GUARD (passes on 3be572a) for [R17:57] "or retire", [R18:17] "Retire = Stop+Archive until P2-6": a liquidated
    strategy can be retired with no note, and nothing starts it again."""
    from sleeve_fund import supervisor

    _liquidate(tmp_path, store)
    r = _post(client, {"command": "stop", "reason": "retire after the liquidation"})
    assert r.status_code == 303 and "command_error" not in r.headers["location"], r.headers["location"]
    r = client.post(f"/sleeves/{NAME}/archive", data={"action": "archive", "reason": "retired after the liquidation"},
                    auth=AUTH, headers=SAME, follow_redirects=False)
    assert r.status_code in (200, 303), r.text
    assert NAME in store.archived()
    s = store.sleeve(NAME)
    assert s.desired_state == "stopped" and s.status != "running"
    started = []
    sup = supervisor.Supervisor(store)
    sup._start = lambda name, proc: started.append(name)
    sup.step()
    assert started == []


@xf
def test_a_second_ral_on_the_same_liquidation_is_a_no_op(store, tmp_path):
    """[R18:17] "repeat = no-op": a second RAL (sent twice before it applied, again once running, or through a
    restart) is refused or ignored; there is one RAL event and the mark it set is not moved again."""
    f = _liquidate(tmp_path, store)
    iid = _noted(store, f.liq)
    _ral(store, incident=iid)
    try:
        _ral(store, incident=iid)  # a double click
    except ValueError:
        pass
    rt = _runtime(store, NEXT_DAY, f.rem)
    assert rt.status == "running"
    _tick(rt, NEXT_DAY + timedelta(minutes=5), f.rem + 300)  # a new high
    _tick(rt, NEXT_DAY + timedelta(minutes=10), f.rem + 100)
    try:
        _ral(store, incident=iid)
    except ValueError:
        pass
    _tick(rt, NEXT_DAY + timedelta(minutes=15), f.rem + 100)
    rt2 = _runtime(store, NEXT_DAY + timedelta(minutes=20), f.rem + 100)  # and through a restart
    assert len(_ral_events(store)) == 1, _ral_events(store)
    assert rt.peak == pytest.approx(f.rem + 300, abs=0.01) and rt2.peak == pytest.approx(f.rem + 300, abs=0.01)
    assert store.sleeve(NAME).status == "running"


@xf
def test_a_second_liquidation_after_a_reset_needs_a_new_note(store, tmp_path):
    """[R17:57] the note is per liquidation: reset after the first, liquidated again (a long, a 60% gap down), the
    first note no longer serves; a note on the second liquidation's incident does, and the new mark is the second
    remainder."""
    f = _liquidate(tmp_path, store)
    first = _noted(store, f.liq)
    _ral(store, incident=first)
    new = _restart(tmp_path, store, [(5, 0.0), (0, -0.6), (5, 0.0)], hours=25, tag="second")
    _gap_closed(new, "SELL")  # a long this time
    assert store.sleeve(NAME).status == "halted"
    liq2 = _gap_event(store, after_id=f.liq["id"])
    rem2 = store.journal_book(NAME, 10_000)["cash"]
    before = _state(store)
    _refused(store, r"incident", incident=first)
    assert _state(store) == before
    _ral(store, incident=_noted(store, liq2))
    rt = _runtime(store, NEXT_DAY + timedelta(days=1), rem2)
    assert rt.status == "running" and rt.peak == pytest.approx(rem2, abs=0.01)
    assert len(_ral_events(store)) == 2


@xf
@pytest.mark.parametrize("actor", ["system", "supervisor", "replay"])
def test_only_the_pm_can_reset_after_liquidation(store, tmp_path, actor):
    """[R17:57] "PM explicit", [R18:17] "never replay/automation": from any actor but the PM it is refused, naming the
    PM, and nothing changes."""
    f = _liquidate(tmp_path, store)
    iid = _noted(store, f.liq)
    before = _state(store)
    with pytest.raises(ValueError, match=r"\bPM\b"):
        _ral(store, incident=iid, actor=actor)
    _nothing_changed(store, before, f.rem)


@xf
def test_ral_comes_only_from_a_pm_ui_action_so_retries_are_refused(store, tmp_path, client):
    """[R18:17] "RAL only from a PM UI action (never ... API retry)": the PM's form post is taken; the same post sent
    again (before or after it applied) is refused in words, and a cross-site post is refused outright. One decision,
    one event."""
    f = _liquidate(tmp_path, store)
    iid = _noted(store, f.liq)
    form = {"command": RAL, "incident": str(iid), "reason": "incident note written"}
    r = _post(client, form)
    assert r.status_code == 303 and "command_error" not in r.headers["location"], r.headers["location"]
    retry = _post(client, form)
    assert retry.status_code == 303 and "command_error" in retry.headers["location"], retry.headers.get("location")
    cross = _post(client, form, headers={"origin": "http://elsewhere.example"})
    assert cross.status_code in (400, 403), cross.status_code
    _runtime(store, NEXT_DAY, f.rem)
    late = _post(client, form)
    assert late.status_code == 303 and "command_error" in late.headers["location"], late.headers.get("location")
    assert len(_ral_decisions(store)) == 1 and len(_ral_events(store)) == 1


@xf
def test_the_dashboard_offers_ral_and_the_note_only_on_a_liquidated_strategy(store, tmp_path, client):
    """[R17:57] / [R18:17]: the liquidated strategy's page offers Reset after liquidation with a field for the
    incident, and the note with its required field and author, and doesn't promise that a resume trades again; a
    strategy on an ordinary drawdown halt is offered neither."""
    _liquidate(tmp_path, store)
    page = client.get(f"/sleeves/{NAME}", auth=AUTH).text
    assert f'value="{RAL}"' in page and 'name="incident"' in page
    assert WHY_FIELD in page.lower() and 'name="why_stop_did_not_protect"' in page and 'name="author"' in page
    assert "trades again on its next signal" not in page
    _ordinary(store, "drawdown_halt", name="dd-halted")
    other = client.get("/sleeves/dd-halted", auth=AUTH).text
    assert RAL not in other and WHY_FIELD not in other.lower()
