"""Resets vs liquidation (Advisor 6 Oct 20:41, "U27", for #164): strict xfails written by QA BEFORE the build. The
engineer makes each one pass (and removes its mark) before handing over.

Source [U27] (advisor-rulings.md, "U27 resets vs liquidation", Advisor 6 Oct 20:41): "an ordinary per-strategy Reset
is REFUSED while a strategy is liquidated (points to reset-after-liquidation). A whole-book clean slate may clear it
only after the PM confirms the named list of liquidated strategies; it writes one liquidation_reset per strategy
(author PM, note "book reset"). Guard: the book reset clears the halt but does NOT close the liquidation incident; it
stays open until someone writes why the half-liquidation stop failed. Standing rule: no book resets until DA-6."
(The standing rule is about using it, not building it: these pin how it behaves once it is used.)

ASSUMED INTERFACES (adapt the names, never the assertions), with test_ral_xfails.py's and test_halt_clearing_xfails.py's
(whose helpers and fixtures this file imports: copy all three into tests/):
- Per-strategy Reset: Store.request_reset(name, reason, actor) and POST /sleeves/{name}/reset, as on 3be572a. Refused
  while liquidated: ValueError at the call, and command_error from the dashboard, naming "reset after liquidation".
- Whole-book clean slate: POST /book/reset (the setup page's "Reset all"). While any strategy is liquidated it resets
  nothing and sends the PM to the named list (its redirect, or the page it leads to, names each liquidated strategy).
  POST /book/reset/confirm, with the reason and one "liquidated" field per strategy on that list, carries it out.
- The liquidation_reset a book reset writes is the kind test_ral_xfails.LIQUIDATION_RESET, naming the PM and "book
  reset". A reset puts the run so far away (Store.split_run), so it is looked for under the strategy's name and its
  runs' names ("<name>--<time>").
- An incident is open while it is in the alerts inbox unacknowledged (Store.alerts), wherever its row has moved.
- [RAL7/8] (Advisor 6 Oct 21:05: "Incident closes only when RAL note written AND PM acknowledges; neither reset nor
  book reset closes it"): Store.incident_is_open(incident_id) -> bool. The PM acknowledges it as any alert,
  Store.ack(incident_id, "PM").
"""

from urllib.parse import unquote

import pytest

from test_halt_clearing_xfails import _error, _supervised
from test_ral_xfails import (AUTH, LIQUIDATION_RESET, NAME, NEXT_DAY, PARAMS, REASON78, SAME, _incident,  # noqa: F401
                             _liquidate, _note, _ral, _runtime, client, store)

REASON = "QA U27: resets vs liquidation, not built yet (Advisor 20:41)"
xf = pytest.mark.xfail(strict=True, raises=AssertionError, reason=REASON)

OTHER = "other-test"  # an ordinary strategy on the same book, never liquidated
WHY_RESET = "testing: a clean slate for the next round"
CONFIRM = "/book/reset/confirm"


def _form(client, path, data):
    return client.post(path, data=data, auth=AUTH, headers=SAME, follow_redirects=False)


def _resets(store, name):
    """The liquidation_reset events of a strategy and of the runs a reset put away from it."""
    return [e for e in store.events(None, limit=100_000)
            if e["kind"] == LIQUIDATION_RESET and (e["sleeve"] == name or (e["sleeve"] or "").startswith(f"{name[:46]}--"))]


def _book(store, tmp_path):
    """A liquidated strategy (halted) and an ordinary one beside it on the book."""
    f = _liquidate(tmp_path, store)
    store.create_sleeve(name=OTHER, strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params=PARAMS)
    return f


def _confirmed_book_reset(store, client, monkeypatch):
    """The PM's whole-book clean slate: Reset all (which holds back and lists the liquidated strategy), then the
    confirmation of that named list, carried out by the supervisor. Returns the redirect of the first post."""
    sup = _supervised(store, monkeypatch)
    assert any(getattr(r, "path", None) == CONFIRM for r in client.app.routes), (
        f"not built: POST {CONFIRM} (the PM confirms the named list of liquidated strategies)")
    first = _form(client, "/book/reset", {"reason": WHY_RESET})
    r = _form(client, CONFIRM, {"reason": WHY_RESET, "liquidated": [NAME]})
    assert r.status_code == 303 and "error" not in r.headers["location"], r.headers["location"]
    for _ in range(3):
        sup.step()
    return first


# --- (1) an ordinary per-strategy Reset is refused while liquidated [U27] -----------------------------------------

@pytest.mark.parametrize("via", ["dashboard", "store"])
def test_an_ordinary_reset_is_refused_while_liquidated_naming_reset_after_liquidation(store, tmp_path, client, via):
    """[U27] "an ordinary per-strategy Reset is REFUSED while a strategy is liquidated (points to reset-after-
    liquidation)": from the strategy page or the store, refused in words naming reset after liquidation; nothing is
    queued, no liquidation_reset is written, and it stays halted. On 3be572a the reset is queued."""
    _liquidate(tmp_path, store)
    if via == "dashboard":
        err = _error(_form(client, f"/sleeves/{NAME}/reset", {"reason": WHY_RESET}))
    else:
        try:
            store.request_reset(NAME, WHY_RESET, actor="PM")
            err = None
        except ValueError as e:
            err = str(e)
    assert err and "reset after liquidation" in err.lower(), err
    assert store.pending_reset(NAME) is None, store.pending_reset(NAME)
    assert not _resets(store, NAME) and store.sleeve(NAME).status == "halted"


# --- (2) a whole-book clean slate clears it only after the PM confirms the named list [U27] -----------------------

def test_a_book_reset_clears_a_liquidated_strategy_only_after_the_pm_confirms_the_named_list(store, tmp_path, client,
                                                                                             monkeypatch):
    """[U27] "A whole-book clean slate may clear it only after the PM confirms the named list of liquidated
    strategies; it writes one liquidation_reset per strategy (author PM, note "book reset")": Reset all on its own
    resets nothing while a strategy is liquidated and names it for the PM; once the PM confirms that list, the book
    is reset, the liquidated strategy's halt is cleared and exactly one liquidation_reset is written for it (naming
    the PM and "book reset"), none for the ordinary strategy. Fails "not built" while there is no confirmation route."""
    _book(store, tmp_path)
    sup = _supervised(store, monkeypatch)
    assert any(getattr(r, "path", None) == CONFIRM for r in client.app.routes), (
        f"not built: POST {CONFIRM} (the PM confirms the named list of liquidated strategies)")
    r = _form(client, "/book/reset", {"reason": WHY_RESET})
    assert r.status_code == 303, r.status_code
    shown = unquote(r.headers["location"]) + client.get(r.headers["location"], auth=AUTH).text
    assert NAME in shown, r.headers["location"]  # the named list
    assert store.pending_reset(NAME) is None and store.pending_reset(OTHER) is None  # nothing reset yet
    assert not _resets(store, NAME) and store.sleeve(NAME).status == "halted"
    r = _form(client, CONFIRM, {"reason": WHY_RESET, "liquidated": [NAME]})
    assert r.status_code == 303 and "error" not in r.headers["location"], r.headers["location"]
    for _ in range(3):
        sup.step()
    got = _resets(store, NAME)
    assert len(got) == 1, [(e["sleeve"], e["message"]) for e in got]
    assert "PM" in got[0]["message"] and "book reset" in got[0]["message"].lower(), got[0]["message"]
    assert not _resets(store, OTHER)
    assert store.sleeve(NAME).status != "halted", (store.sleeve(NAME).status, store.sleeve(NAME).status_reason)


# --- (3) the book reset leaves the liquidation incident open until the note [U27] ---------------------------------

def test_after_a_book_reset_the_liquidation_incident_stays_open_until_the_why_note_is_written(store, tmp_path, client,
                                                                                              monkeypatch):
    """[U27] Guard: "the book reset clears the halt but does NOT close the liquidation incident; it stays open until
    someone writes why the half-liquidation stop failed": after the confirmed book reset the incident (the engine's,
    or a stand-in where the engine opens none yet) is still open in the alerts inbox, unacknowledged."""
    f = _book(store, tmp_path)
    iid = _incident(store, f.liq)
    _confirmed_book_reset(store, client, monkeypatch)
    assert _resets(store, NAME), "setup: the book reset ran"
    open_ = [a for a in store.alerts(limit=100_000) if a["id"] == iid]
    assert open_ and open_[0]["acked_at"] is None, [(a["id"], a["kind"]) for a in store.alerts(limit=20)]


# --- [RAL7/8] the incident closes only on the note AND the PM's acknowledgement --------------------------------------

@pytest.mark.parametrize("done", ["reset_after_liquidation_alone", "note_alone", "pm_ack_alone", "book_reset_and_pm_ack",
                                  "note_then_pm_ack", "pm_ack_then_note"])
def test_the_incident_closes_only_on_the_note_and_the_pm_acknowledgement(store, tmp_path, client, monkeypatch, done):
    """[RAL7/8] "Incident closes only when RAL note written AND PM acknowledges; neither reset nor book reset closes
    it": the reset after liquidation (its note written, no acknowledgement), the note alone, the PM's acknowledgement
    alone, or a book reset with the PM's acknowledgement (no note) leave it open; the note (author and why the stop
    failed to protect it) and the PM's acknowledgement, in either order, close it."""
    assert hasattr(store, "incident_is_open"), "not built: Store.incident_is_open (whether the incident is closed)"
    f = _book(store, tmp_path) if done == "book_reset_and_pm_ack" else _liquidate(tmp_path, store)
    iid = _incident(store, f.liq)
    assert store.incident_is_open(iid)
    if done == "reset_after_liquidation_alone":
        _note(store, iid)
        _ral(store, incident=iid)
        _runtime(store, NEXT_DAY, f.rem)
        assert _resets(store, NAME), "setup: the reset after liquidation applied"
    elif done == "book_reset_and_pm_ack":
        _confirmed_book_reset(store, client, monkeypatch)
        store.ack(iid, "PM")
    else:
        for step in done.removesuffix("_alone").split("_then_"):
            if step == "note":
                _note(store, iid)
            else:
                store.ack(iid, "PM")
    closes = done in ("note_then_pm_ack", "pm_ack_then_note")
    assert store.incident_is_open(iid) is (not closes), (done, store.incident_is_open(iid))


# --- a reset pending from before a liquidation does not outlive the reset after liquidation (#167 round) -----------

def test_a_reset_pending_from_before_the_liquidation_does_not_run_after_the_reset_after_liquidation(store, tmp_path,
                                                                                                   monkeypatch):
    """Found in the #167 round (quant-review/v2-p1/ui-v2-152.md:601, on 5afb6a7): a reset pending from before a
    liquidation, the liquidation and its liquidation_reset all before one supervisor pass let the older reset run with
    no fresh confirmation. The reset after liquidation cancels or closes any reset pending from before it: after the
    supervisor's passes the old one is no longer pending and has not run (nothing put away, the remainder still the
    strategy's own). A reset asked for after the liquidation_reset runs normally."""
    store.create_sleeve(name=NAME, strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params=PARAMS)
    store.request_reset(NAME, "testing: start it again", actor="PM")  # pending, before anything happened
    f = _liquidate(tmp_path, store)
    iid = _incident(store, f.liq)
    _note(store, iid)
    _ral(store, incident=iid)
    _runtime(store, NEXT_DAY, f.rem)
    assert _resets(store, NAME), "setup: the reset after liquidation applied"
    sup = _supervised(store, monkeypatch)
    for _ in range(2):
        sup.step()
    assert store.pending_reset(NAME) is None, store.pending_reset(NAME)
    assert not [r for r in store.reset_runs() if r], store.reset_runs()  # the old reset put nothing away
    assert store.journal_book(NAME, 10_000)["cash"] == pytest.approx(f.rem, abs=0.01)
    store.request_reset(NAME, "testing: a fresh reset after the reset after liquidation", actor="PM")
    for _ in range(3):
        sup.step()
    runs = [r for r in store.reset_runs() if r]
    assert len(runs) == 1 and store.pending_reset(NAME) is None, (runs, store.pending_reset(NAME))
