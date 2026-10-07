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
- [R20:37] Advisor 6 Oct 20:37 (P1-D20; supersedes 18:17's "Y on MTM equity just before"): Y = X / the strategy's
  equity when the liquidated position was OPENED (at its first entry fill); with partial reductions in between keep
  that same entry-time equity. Show the true figure even above 100% (never cap). Halt text: "Position margin lost
  (liquidated): X, Y% of strategy equity at entry". The incident also records the equity remaining after it.
- [RAL7/8] Advisor 6 Oct 21:05 to QA ("RAL 7/8"): Y's denominator stays the equity at the FIRST entry fill even with
  adds; X = the whole liquidated quantity; Y > 100% shown uncapped with "includes adds", flagged as a sizing finding
  only if a cap check at entry or at an add was breached. The incident closes only when the RAL note is written AND
  the PM acknowledges it; neither a reset nor a book reset closes it.
- [CAP21:16] Advisor 6 Oct ~21:16 to QA ("Cap check at an add"): check the WHOLE position after the add, as the
  isolated margin actually POSTED after it (existing posted margin + the add's margin), not the whole position
  re-valued at the add's price. Tests use exactly the engine's own entry/add cap check; if it differs from posted
  margin, that is its own finding.
- [ADDCAP22:23] Advisor 6 Oct ~22:23 to QA ("Add-cap finding"): latent MAJOR confirmed; allowed add notional <= (cap x
  equity just before the add - the position's posted margin) x leverage. RAL 7: keep "includes adds" as it is; on
  balanced, Y > 100% must coincide with a recorded cap breach (else a bug in Y or the cap check).
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
- With adds [RAL7/8]: the halt text says "includes adds" after Y. At the liquidation the engine checks the entry and
  each add against the risk profile's cap at its own time (the whole position's margin at that fill's price within
  max_position_pct of the equity then), from the journal, and raises a sizing finding (an event of kind
  "sizing_finding" for the strategy, naming the cap) only when one was breached.
- The liquidation halt's status_reason contains the ruled text, as [R20:37] words it ("... of strategy equity at
  entry"). The engine's incident message carries the equity remaining after the liquidation ("{:,.2f}").
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
# Two drawdown cells miss max_drawdown by ~8e-7: journal cash and the engine's mark differ by 0.8 cents of the
# insurance credit's rounding (GAP-LIQ-CAP). Never loosened: the reason says it.
xf_cap = pytest.mark.xfail(strict=True, reason=REASON + " (also needs GAP-LIQ-CAP: journal cash vs engine mark)")
REASON78 = "QA RAL 7/8: not built yet (Advisor 21:05)"
xf78 = pytest.mark.xfail(strict=True, raises=AssertionError, reason=REASON78)
SIZING = "sizing_finding"  # [RAL7/8] the event a breached cap check at entry or at an add raises

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
# [R17:57] (a) as [R20:37] words it, with X an amount (separators, an optional currency) and Y a percentage.
TEXT = re.compile(r"Position margin lost \(liquidated\): [^\d]*(\d[\d,]*(?:\.\d+)?)[^,]*, (\d+(?:\.\d+)?)% of "
                  r"strategy equity at entry")
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


def _gap_px(close):
    """The price the gap reached: the liquidation signal's market_px where the head books the fill at the bankruptcy
    price (GAP-LIQ-CAP, Advisor 00:19 (1)-(2)); otherwise the booked avg_px. A set-up input only, never asserted on."""
    sig = close.get("signal") or {}
    return float(sig.get("market_px") or close["avg_px"])


def _gap_closed(orders, side):
    """Setup check (passes on every head): the gap closed the position past its liquidation price, by the liquidation
    or by the stop it went through. Returns (the closing order, the entry it closed)."""
    close = orders[-1]
    assert close["intent"] in ("liquidation", "stop_loss") and close["side"] == side, [(o["side"], o["intent"])
                                                                                       for o in orders]
    entry = [o for o in orders if o["intent"] == "entry" and o["side"] != side][-1]
    move = _gap_px(close) / entry["avg_px"] - 1
    assert (move if side == "BUY" else -move) > 0.5, (_gap_px(close), entry["avg_px"])  # past 2x liquidation (~49%)
    return close, entry


def _gap_event(store, after_id=0):
    """The liquidation's event: the engine's "liquidation" event; on a head that books a stop filled through the
    liquidation price as a stop-loss (3be572a, ba4f533), the halt the gap caused, on the same tick."""
    for kinds in (("liquidation",), ("risk_halt",)):
        ev = store.last_event(NAME, kinds)
        if ev is not None and ev["id"] > after_id:
            return ev
    raise AssertionError("the gap left no liquidation or halt event")


def _equity_at_entry(store, close_id):
    """[R20:37] the strategy's equity at the first entry fill of the position the order `close_id` closed: the fill
    that opened it from flat (or crossed through flat); later adds and partial reductions don't move it. Returns
    (equity just after that fill, from the journal: cash + position at the fill's price; that fill's fee), so a
    figure taken just before the fee is within reach too."""
    from sleeve_fund.store import replay_book

    fills = sorted(store.fills(NAME, limit=100_000), key=lambda x: (x["ts"], x["id"]))
    fills = fills[:next(i for i, x in enumerate(fills) if x["order_id"] == close_id)]
    held, first = 0.0, None
    for i, x in enumerate(fills):
        new = held + (x["qty"] if x["side"] == "BUY" else -x["qty"])
        new = 0.0 if abs(new) < 1e-9 else new
        if new and (not held or (new > 0) != (held > 0)):
            first = i
        held = new
    assert first is not None and held, "setup: no open position before the close"
    book = replay_book(fills[:first + 1], 10_000)
    return book["cash"] + book["qty"] * fills[first]["price"], fills[first]["fee"]


def _y_ok(y, x, at_entry, fee):
    """Y as printed is X over the equity at entry to its printed rounding, uncapped ([R20:37] "never cap"); the
    equity taken just after the opening fill or just before its fee."""
    r = 0.5 * 10 ** -(len(y.split(".")[1]) if "." in y else 0) + 1e-9
    return 100 * x / (at_entry + fee) - r <= float(y) <= 100 * x / at_entry + r


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
    at_entry, entry_fee = _equity_at_entry(store, close["order_id"])
    return SimpleNamespace(rem=book["cash"], peak=store.peak_equity(NAME), liq=liq, before=before, close=close,
                           entry=entry, orders=len(orders), reason=s.status_reason, at_entry=at_entry,
                           entry_fee=entry_fee)


def _restart(tmp_path, store, legs, hours, px=GAP_PX, tag="restart"):
    """The paper process starting again on the same journal: a recorded session replayed into the store. Returns the
    orders it sent, oldest first: the rows replay() returns whose order id the journal didn't hold before. Selected by
    id, never by position: replay() leaves out paper's watched-stop rows (P1-U35, order_type 'STOP (watched)', no
    venue order) while the journal holds them, so a count taken before would cut into the new orders."""
    before = {o["order_id"] for o in store.orders(NAME, limit=100_000)}
    path = tmp_path / f"{tag}.jsonl.gz"
    _record_at(path, _meta(store.journal_book(NAME, 10_000)["cash"], PARAMS), legs, px, hours)
    return [o for o in _replay_into(store, path) if o["order_id"] not in before]


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


# PE2 (stop-safety, master 370303a4): passes (mark removed)
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


# PE2 (stop-safety, master 370303a4): passes (mark removed)
@pytest.mark.parametrize("margin", ["capped", "full"])
def test_the_liquidation_halt_says_how_much_position_margin_was_lost(store, tmp_path, request, margin):
    """[R17:57] (a), [R18:17], [R20:37]: the halt reads "Position margin lost (liquidated): X, Y% of strategy equity
    at entry", X = the position's isolated margin + its entry fee + its liquidation fee (to the cent), Y = X over the
    strategy's equity at the position's first entry fill (to its printed rounding, never capped). "capped": balanced's
    33% margin cap (X about 3,328.7, Y about 33%); "full": the whole equity as margin (Y about 100%, above it if the
    fees take it there)."""
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
    assert _y_ok(y, x, f.at_entry, f.entry_fee), (y, x, f.at_entry, f.before)


# PE2 (stop-safety, master 370303a4): passes (mark removed)
def test_y_keeps_the_equity_at_the_first_entry_fill_through_a_partial_reduce(store, tmp_path):
    """[R20:37] "with partial reductions in between keep that same entry-time equity": a 2x short of 0.11 opened at
    60,600 on 10,000 of equity; the price falls 30% and half the short is bought back (a partial reduce, equity about
    12,000); then the price gaps 120% up through the stop and the liquidation price. Y is X over the equity at the
    opening fill (about 10,000), not at the reduce or just before the gap (both about 12,000). The journal is
    written as the #146 outage case writes it: fills, then the process restarts on them."""
    import test_replay
    from sleeve_fund.store import replay_book

    params = {**PARAMS, "dip": 0.4}  # the short leg's own take-profit is 40% down, so the 30% fall keeps it open
    start = datetime.fromtimestamp(test_replay.START / 1e9, tz=timezone.utc)
    opened, reduced, entry_px, low = start - timedelta(hours=2), start - timedelta(hours=1), 60_600.0, 42_420.0
    store.create_sleeve(name=NAME, strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params=params)
    fills = [{"side": "SELL", "qty": 0.11, "price": entry_px, "fee": 3.33, "order_id": "opened", "trade_id": "opened",
              "ts": opened},
             {"side": "BUY", "qty": 0.055, "price": low, "fee": 1.17, "order_id": "reduced", "trade_id": "reduced",
              "ts": reduced}]
    store.record_equity(NAME, equity=10_000, cash=10_000, qty=0, price=entry_px, benchmark=10_000, ts=opened)
    for i, x in enumerate(fills):
        b = replay_book(fills[:i], 10_000)
        if i:  # marked at the reduce's price just before it
            store.record_equity(NAME, equity=b["cash"] + b["qty"] * x["price"], cash=b["cash"], qty=b["qty"],
                                price=x["price"], benchmark=10_000, ts=x["ts"] - timedelta(seconds=1))
        store.record_fill(NAME, **x)
    book = replay_book(fills, 10_000)
    at_reduce = book["cash"] + book["qty"] * low
    assert at_reduce > 11_900, at_reduce  # setup: the reduce came at about 12,000 of equity
    path = tmp_path / "reduced.jsonl.gz"
    _record_at(path, _meta(book["cash"] + book["qty"] * book["entry_px"], params), [(5, 0.0), (0, 1.2), (5, 0.0)],
               low, 0)
    orders = _replay_into(store, path)
    close = orders[-1]  # setup (passes on every head): the gap closed the rest past its liquidation price
    assert close["intent"] in ("liquidation", "stop_loss") and close["side"] == "BUY", [(o["side"], o["intent"])
                                                                                          for o in orders]
    assert _gap_px(close) / entry_px - 1 > 0.5 and close["qty"] == pytest.approx(0.055), close
    assert store.journal_book(NAME, 10_000)["qty"] == 0
    at_entry, fee = _equity_at_entry(store, close["order_id"])
    assert at_entry == pytest.approx(10_000 - 3.33) and fee == pytest.approx(3.33)  # setup: the opening fill's
    before = store.equity_at_or_before(NAME, close["ts"] - timedelta(seconds=1))["equity"]
    assert before > 11_000, before  # setup: just before the gap the equity is still about 12,000
    s = store.sleeve(NAME)  # a liquidation halts it with the ruled text (on 3be572a: a daily-loss pause, see README)
    assert s.status == "halted", (s.status, s.status_reason)
    m = TEXT.search(s.status_reason)
    assert m, s.status_reason
    x, y = float(m.group(1).replace(",", "")), m.group(2)
    assert _y_ok(y, x, at_entry, fee), (y, x, at_entry, at_reduce, before)


# PE2 (stop-safety, master 370303a4): passes (mark removed)
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


# PE2 (stop-safety, master 370303a4): passes (mark removed)
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

# PE2 (stop-safety, master 370303a4): passes (mark removed)
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


# The engine's own cap check, the one formula the adds cases judge by ([CAP21:16]: "tests must use exactly the formula
# the engine's own entry/add cap check uses"). On 3be572a it is the perp opening order's sizing limit,
# sleeve_fund/strategies/base.py:1279 (ba4f533 :1326, c998d08 :1290, e14acec :1174): the order's notional <=
# SleeveRuntime.position_budget(equity), sleeve_fund/paper/runtime.py:172-173 (ba4f533 :189), = equity x
# risk.position_cap(profile, params), sleeve_fund/risk.py:75-82 (margin cap x leverage cap on a perp), with equity
# marked to market. A perp opens only from flat (base.py:1219-1242, `if side == current: return`), so there is no add
# path, and the formula applied to an add sizes the add alone. Once the add cap is built (test_the_engines_add_cap_
# check_is_on_the_margin_posted_after_the_add, "add-cap formula") position_budget takes the margin already posted and
# _engine_cap_held passes it, so the same call judges by the posted margin after the add.
def _engine_cap_held(store, equity, posted, qty, px):
    """Whether the engine's own guard lets an order of qty at px through, at this equity, with this isolated margin
    already posted: SleeveRuntime.position_budget, called as the engine calls it."""
    from sleeve_fund.paper.runtime import SleeveRuntime

    rt = SleeveRuntime(store, NAME)
    if "posted" in inspect.signature(rt.position_budget).parameters:
        budget = rt.position_budget(equity, posted=posted)
    else:
        budget = rt.position_budget(equity)
    return qty * px <= budget + 1e-6


def _with_adds(store, tmp_path, last, profile="aggressive", share=0.45, adds=2):
    """[RAL7/8] a short at the profile's leverage (by default aggressive, 3x with a 50% margin cap: on balanced, 33% x
    2x, margin posted within the cap can never pass the first-fill equity), opened at 60,600 on 10,000 of equity.
    `adds` times the price halves, half the short is bought back (realising the profit, so the wallet can post more)
    and the short is added to; then a gap up through the stop and the liquidation price closes it. Each opening fill
    takes the posted isolated margin to 45% of
    the equity just before it (`share`; [CAP21:16]: existing posted margin + the fill's own), the last to `last`; every
    margin fits the wallet (cash + position at entry, less what is posted: markets.isolated_margin's balance). The
    journal is written as the #146 outage case writes it; the process restarts on it. Returns the first fill's equity
    and fee, the posted margin of the whole liquidated quantity, the fees paid, whether the engine's own guard held at
    each opening fill (_engine_cap_held), and the halt."""
    import test_replay
    from sleeve_fund import markets, risk
    from sleeve_fund.store import replay_book

    lev = risk.profile(profile).max_leverage
    params = {**PARAMS, "dip": 0.4}  # the short leg's own take-profit is 40% down: it holds through the halvings
    start = datetime.fromtimestamp(test_replay.START / 1e9, tz=timezone.utc)
    store.create_sleeve(name=NAME, strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params=params, risk_profile=profile)
    fills, px, held = [], 60_600.0, []
    plan = [("open", share)] + [("reduce", 0.5), ("open", share)] * (adds - 1) + [("reduce", 0.5), ("open", last)]
    for i, (what, x) in enumerate(plan):
        b = replay_book(fills, 10_000)
        equity, short = b["cash"] + b["qty"] * px, -b["qty"]
        posted = markets.isolated_margin(short, b["entry_px"] or px, lev)
        if what == "open":
            qty = round((x * equity - posted) * lev / px, 6)
            wallet = b["cash"] + b["qty"] * (b["entry_px"] or px) - posted
            assert qty * px / lev <= wallet, (i, qty * px / lev, wallet)  # setup: the wallet can post it
            held.append(_engine_cap_held(store, equity, posted, qty, px))
        else:
            qty = round(short * x, 6)
        at = start - timedelta(hours=len(plan) - i)
        store.record_equity(NAME, equity=equity, cash=b["cash"], qty=b["qty"], price=px, benchmark=10_000,
                            ts=at - timedelta(seconds=1))
        fill = {"side": "SELL" if what == "open" else "BUY", "qty": qty, "price": px, "fee": round(qty * px * 0.0005, 2),
                "order_id": f"{what}-{i}", "trade_id": f"{what}-{i}", "ts": at}
        store.record_fill(NAME, **fill)
        fills.append(fill)
        if what == "open" and i < len(plan) - 1:
            px /= 2
    book = replay_book(fills, 10_000)
    liq_px = markets.isolated_liquidation(book["cash"], book["qty"], book["entry_px"], lev, 0.005)
    assert px > book["entry_px"] * 0.61 and liq_px / px > 1.2, (px, book["entry_px"], liq_px)  # setup: it holds
    meta = _meta(book["cash"] + book["qty"] * book["entry_px"], params)
    meta["sleeve"]["risk_profile"] = profile
    path = tmp_path / "adds.jsonl.gz"
    _record_at(path, meta, [(5, 0.0), (0, liq_px / px * 1.1 - 1), (5, 0.0)], px, 0)
    orders = _replay_into(store, path)
    close = orders[-1]  # setup (passes on every head): the gap closed the whole position past its liquidation price
    assert close["intent"] in ("liquidation", "stop_loss") and close["side"] == "BUY", [(o["side"], o["intent"])
                                                                                          for o in orders]
    assert close["avg_px"] > liq_px and close["qty"] == pytest.approx(-book["qty"]), (close, liq_px)
    assert store.journal_book(NAME, 10_000)["qty"] == 0
    first, fee = _equity_at_entry(store, close["order_id"])
    assert first == pytest.approx(10_000 - fills[0]["fee"]) and fee == fills[0]["fee"]  # setup: the first fill's
    closing = sum(x["fee"] for x in store.fills(NAME, limit=1000) if x["order_id"] == close["order_id"])
    return SimpleNamespace(first=first, fee=fee, margin=markets.isolated_margin(-book["qty"], book["entry_px"], lev),
                           fees=sum(f["fee"] for f in fills) + closing, held=held, s=store.sleeve(NAME))


@xf78
def test_y_with_adds_stays_on_the_first_entry_fill_uncapped_and_says_includes_adds(store, tmp_path):
    """[RAL7/8] "Y denominator stays equity at FIRST entry fill even with adds; X = whole liquidated qty; Y>100% shown
    uncapped with 'includes adds'": opened, reduced and added to twice, each opening fill through the engine's own cap
    check at its time (_engine_cap_held, [CAP21:16]), then liquidated. X is the whole liquidated quantity's isolated
    margin plus fees (at most every fee paid); Y = X over the equity at the first entry fill, about 120%, not capped;
    the text says "includes adds"; no sizing finding is raised."""
    f = _with_adds(store, tmp_path, 0.45)
    assert all(f.held), f.held  # setup: the engine's own guard held at the entry and at each add
    assert f.margin > f.first * 1.1, (f.margin, f.first)  # setup: so Y is above 100%
    assert f.s.status == "halted", (f.s.status, f.s.status_reason)
    m = TEXT.search(f.s.status_reason)
    assert m, f.s.status_reason
    x, y = float(m.group(1).replace(",", "")), m.group(2)
    assert f.margin - 0.01 <= x <= f.margin + f.fees + 0.01, (x, f.margin, f.fees)
    assert float(y) > 100 and _y_ok(y, x, f.first, f.fee), (y, x, f.first)
    assert "includes adds" in f.s.status_reason[m.end():], f.s.status_reason
    found = [e for e in store.events(NAME, limit=10_000) if e["kind"] == SIZING]
    assert not found, [e["message"] for e in found]


@xf78
def test_an_add_that_breached_its_cap_check_raises_a_sizing_finding(store, tmp_path):
    """[RAL7/8] "flagged as sizing finding only if a cap check at entry or an add was breached": the last add takes the
    posted margin to 70% of the equity then, and is on its own more than the engine's position budget, so it breaches
    the engine's own cap check whichever way it is measured (_engine_cap_held); after the liquidation there is a
    sizing finding for the strategy, naming the cap."""
    f = _with_adds(store, tmp_path, 0.70)
    assert f.held == [True, True, False], f.held  # setup: only the last add breached the engine's own guard
    found = [e for e in store.events(NAME, limit=10_000) if e["kind"] == SIZING]
    assert found, f"not built: a sizing finding (an event of kind {SIZING!r}) for an add that breached its cap check"
    assert any("cap" in e["message"].lower() for e in found), [e["message"] for e in found]


@pytest.mark.parametrize("case", [pytest.param("y_above_100_with_a_breached_add", marks=pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="QA RAL 7 cross-check: not built yet (Advisor 22:23)")),
    "y_at_or_below_100_within_every_cap"])  # PE2 (stop-safety, master 370303a4): this cell passes (mark removed)
def test_on_balanced_y_above_100_coincides_with_a_recorded_cap_breach(store, tmp_path, case):
    """[ADDCAP22:23] "on balanced, Y > 100% must coincide with a recorded cap breach (else bug in Y or cap check)",
    both ways. On balanced (33% x 2x) a margin posted within the cap at every fill stays below the first-fill equity,
    so: (1) a liquidation whose last add took the posted margin to 75% of equity (two adds, the engine's own check
    breached at the second, _engine_cap_held) shows Y above 100% (about 121%) AND has a sizing finding recorded; a Y
    above 100% with none recorded fails here; (2) a liquidation after three adds, every fill within the engine's
    check, shows Y at or below 100% and records no sizing finding: nothing is flagged from Y alone. A strict xfail:
    today's engine prints no Y and records no sizing finding, so neither direction holds yet. The "includes adds" note is pinned as it is
    by test_y_with_adds_..."""
    above = case == "y_above_100_with_a_breached_add"
    f = _with_adds(store, tmp_path, 0.75 if above else 0.30, profile="balanced", share=0.30, adds=2 if above else 3)
    assert f.held == [True] * (len(f.held) - 1) + [not above], f.held  # setup: only a breaching add failed the check
    assert len(f.held) == (3 if above else 4)  # setup: the entry and its adds
    assert f.s.status == "halted", (f.s.status, f.s.status_reason)
    m = TEXT.search(f.s.status_reason)
    assert m, f.s.status_reason
    y = float(m.group(2))
    found = [e for e in store.events(NAME, limit=10_000) if e["kind"] == SIZING]
    assert (y > 100) == above, (y, f.s.status_reason)
    assert bool(found) == (y > 100), (y, [e["message"] for e in found])


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="add-cap formula")
def test_the_engines_add_cap_check_is_on_the_margin_posted_after_the_add(store):
    """[CAP21:16] "check the WHOLE position after the add ... the isolated margin actually POSTED after the add
    (existing posted margin + the add's margin) ... if the engine's check differs from posted margin, that is its own
    finding". The engine's only cap check sizes the order alone (3be572a sleeve_fund/strategies/base.py:1279:
    notional <= SleeveRuntime.position_budget(equity), sleeve_fund/paper/runtime.py:172-173, = equity x
    risk.position_cap, sleeve_fund/risk.py:75-82); a perp opens only from flat (base.py:1219-1242), so nothing checks
    an add. Pinned: with margin already posted, the room the guard leaves is the margin cap less what is posted, times
    the leverage; none once the posted margin is at the cap."""
    from sleeve_fund import risk
    from sleeve_fund.paper.runtime import SleeveRuntime

    store.create_sleeve(name=NAME, strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params=PARAMS)
    rt = SleeveRuntime(store, NAME)
    p = risk.profile("balanced")
    assert "posted" in inspect.signature(rt.position_budget).parameters, (
        "not built: SleeveRuntime.position_budget(equity, posted=...): the add cap on the margin already posted "
        f"(today position_budget({10_000}) = {rt.position_budget(10_000):,.0f} of notional whatever is posted)")
    assert rt.position_budget(10_000, posted=2_000) == pytest.approx((p.max_position_pct * 10_000 - 2_000)
                                                                     * p.max_leverage)
    assert rt.position_budget(10_000, posted=p.max_position_pct * 10_000) <= 1e-6


# PE2 (stop-safety, master 370303a4): passes (mark removed)
def test_the_engines_incident_records_the_equity_remaining_after_the_liquidation(store, tmp_path):
    """[R20:37] "Incident also records the equity remaining after the liquidation": the engine's incident for this
    liquidation (one, at or after it) names the strategy's equity once the position is gone ("{:,.2f}")."""
    f = _liquidate(tmp_path, store)
    found = _incidents_since(store, f.liq["ts"])
    assert len(found) == 1, [(e["kind"], e["message"][:60]) for e in store.events(NAME, limit=50, min_level="error")]
    assert f"{f.rem:,.2f}" in found[0]["message"], (found[0]["message"], f.rem)


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


@xf_cap
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


@xf_cap
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
