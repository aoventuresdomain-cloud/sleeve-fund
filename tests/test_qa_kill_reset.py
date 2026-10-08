"""QA adversarial probes for PR #188 (m13-U5, supersedes #132): the PM's Kill switch / Pause / Stop / Flatten across a
strategy reset, a clean slate, a refused (liquidated) reset, back-to-back resets and a deploy.

Ported from quant-review/v2-p1/kill-reset-188-scripts/test_kr188_probes.py (round kill-reset-188.md, findings
P1-KR-1/2/3). The strict xfail marks are removed: #188 (b7368b5) fixed them. Only the cells listed in the QA cell
table are ported; the others in the source file are not regression pins.

Real parts: the dashboard routes (TestClient), Store, Supervisor.reset_pending and supervisor.clear, and the paper
SleeveRuntime (on_start + tick apply the PM's commands). Faked: the process lifecycle and the venue. A flatten the
runtime asks for becomes one working market order, which fills at the next tick unless the cell holds it (`fill=False`);
a flatten asked while that order works sends nothing more, as base._sell_all does with a working market order.

Each cell drives the reset to completion and starts the fresh run's process, then checks:
  * the PM's Pause / Flatten / Kill switch is in force on the fresh run (paused, nothing opens) and Stop leaves it stopped;
  * no double flatten: at most one closing order per position, and the position never flips;
  * the system's own flattens (the reset's, the clean slate's) never leave a PM pause on the fresh run;
  * the journal names the PM for the PM's action.
Grades go in quant-review/v2-p1/kill-reset-188.md."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sleeve_fund.paper.runtime import SleeveRuntime, entry_blocked
from sleeve_fund.store import Store
from sleeve_fund.supervisor import Supervisor

AUTH = ("pm", "test-pw")
SAME = {"origin": "http://testserver"}
PX = 100_000.0
START = 5_000.0
QTY = 0.02
NAME = "s1"


class World:
    def __init__(self, tmp_path, monkeypatch, names=(NAME,)):
        monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
        monkeypatch.setenv("TEARSHEET_DIR", str(tmp_path))
        from sleeve_fund.dashboard import app as app_mod

        monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
        monkeypatch.setattr(app_mod, "LEDGER", tmp_path / "idea_ledger.jsonl")
        self.tmp = tmp_path
        # The store's clock steps 2 s at each supervisor pass, so two resets never put a run away in the same second
        # (split_run names the run by the second; real resets are at least one supervisor poll apart).
        import sleeve_fund.store as store_mod
        real = store_mod.utcnow
        self.skew = [0]
        from datetime import timedelta as _td
        monkeypatch.setattr(store_mod, "utcnow", lambda: real() + _td(seconds=self.skew[0]))
        self.store = Store(f"sqlite:///{tmp_path}/t.db")
        self.c = TestClient(app_mod.create_app(self.store))
        self.sup = Supervisor(self.store, python="true")
        self.rt: dict[str, SleeveRuntime] = {}
        self.working: dict[str, float] = {}
        self.orders: dict[str, list[float]] = {}
        self.n = 0
        self.runs_done = 0
        self.said: list[str] = []
        for name in names:
            self.store.create_sleeve(name=name, strategy="trend_filter", instrument="BTC/USD",
                                     bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=START,
                                     params={"fast": 10, "slow": 30})
            self.start_process(name)

    # --- the strategy's process (faked lifecycle, real runtime) ---------------------------------------------------
    def start_process(self, name):
        rt = self.rt[name] = SleeveRuntime(self.store, name)
        rt.on_start(0.0008)
        return rt

    def stop_process(self, name):
        self.rt.pop(name, None)
        self.working.pop(name, None)

    def qty(self, name):
        return self.store.journal_book(name, START)["qty"]

    def fill(self, name, side, qty):
        self.n += 1
        self.store.record_fill(name, side=side, qty=qty, price=PX, fee=0.0, order_id=f"o{self.n}", trade_id=f"t{self.n}")

    def buy(self, name=NAME, qty=QTY):
        self.fill(name, "BUY", qty)
        self.mark(name)

    def mark(self, name):
        b = self.store.journal_book(name, START)
        self.store.record_equity(name, equity=b["cash"] + b["qty"] * PX, cash=b["cash"], qty=b["qty"], price=PX,
                                 benchmark=START)

    def wanted(self, name):
        s = self.store.sleeve(name)
        return s.desired_state == "running" or abs(self.qty(name)) > 1e-12  # a stopped holder runs for its exits (U35)

    def tick(self, name=NAME, fill=True):
        if not self.wanted(name):
            self.stop_process(name)
            return None
        rt = self.rt.get(name) or self.start_process(name)
        b = self.store.journal_book(name, START)
        qty = b["qty"]
        busy = name in self.working
        out = rt.tick(equity=b["cash"] + qty * PX, cash=b["cash"], qty=qty, price=PX, busy=busy)
        if out == "flatten" and abs(qty) > 1e-12 and not busy:
            self.working[name] = qty
            self.orders.setdefault(name, []).append(qty)
        if fill and name in self.working:
            q = self.working.pop(name)
            self.fill(name, "SELL" if q > 0 else "BUY", abs(q))
        self.mark(name)
        return out

    def supervise(self):
        before = len(self.store.reset_runs())
        self.skew[0] += 2
        self.sup.reset_pending()
        if len(self.store.reset_runs()) > before:  # the reset stopped the process and started the run afresh
            for name in list(self.rt):
                if self.store.pending_reset(name) is None:
                    self.stop_process(name)
        self.runs_done = len(self.store.reset_runs())

    # --- the PM, through the dashboard ----------------------------------------------------------------------------
    def pm(self, action, name=NAME, reason="QA probe"):
        if action == "kill":
            r = self.c.post("/book/flatten", data={"reason": reason}, auth=AUTH, headers=SAME, follow_redirects=False)
        elif action == "book_reset":
            r = self.c.post("/book/reset", data={"reason": reason}, auth=AUTH, headers=SAME, follow_redirects=False)
        elif action == "reset":
            r = self.c.post(f"/sleeves/{name}/reset", data={"reason": reason}, auth=AUTH, headers=SAME,
                            follow_redirects=False)
        else:
            r = self.c.post(f"/sleeves/{name}/command", data={"command": action, "reason": reason}, auth=AUTH,
                            headers=SAME, follow_redirects=False)
        self.said.append(f"{action}: {r.status_code} {r.headers.get('location', '')}")
        return r

    # --- what the fresh run does ----------------------------------------------------------------------------------
    def drive(self, name=NAME, steps=8):
        for _ in range(steps):
            self.supervise()
            if self.store.pending_reset(name) is None:
                break  # done: the fresh run's process has not started yet (fresh_opens starts it)
            self.tick(name)

    def fresh_opens(self, name=NAME) -> bool:
        """Whether the fresh run would open a position: its process is wanted and its runtime would let an entry
        through (can_open) with the gate open."""
        if not self.wanted(name):
            return False
        self.stop_process(name)
        rt = self.start_process(name)  # the fresh run's first process start
        self.tick(name)
        self.tick(name)
        return rt.can_open() and not entry_blocked(self.store, name, starting=True)[0]


def _sells(w, name=NAME, run=None):
    fills = w.store.fills(name) + (w.store.fills(run) if run else [])
    return [f for f in fills if f["side"] == "SELL"]


def _pm_rows(w, name, action):
    runs = list(w.store.reset_runs())
    rows = [d for n in [name, *runs] for d in w.store.decisions(n, limit=500)]
    return [d for d in rows if d["actor"] == "PM" and d["action"] == action]


# ----------------------------------------------------------------------------------------------------------------
# 1. The matrix: Kill switch / Pause / Stop / Flatten x when, on a running strategy that holds a long.
#    a1: pressed just before the reset is asked for, and the process has applied it
#    a2: pressed just before the reset is asked for, before the process's next tick
#    b : the reset is pending and its flatten is queued, not yet taken by the process
#    c : the reset's own flatten is working (sent, not filled)
#    d : the reset is done, before the fresh run's first tick
# ----------------------------------------------------------------------------------------------------------------
ACTIONS = ("kill", "pause", "stop", "flatten")
WHENS = ("a1", "a2", "b", "c", "d")

# The finding each cell pinned (source: strict xfail marks). Cells not listed carried no mark in the source.
FINDINGS = {("kill", "a2"): "P1-KR-2", ("pause", "a2"): "P1-KR-2", ("flatten", "a2"): "P1-KR-2",
            ("kill", "b"): "P1-KR-1", ("flatten", "b"): "P1-KR-1", ("stop", "b"): "P1-KR-3", ("stop", "c"): "P1-KR-3"}
# Not ported: kill/pause/flatten at a1 and d, and stop at a1, a2 and d. They pass on #188's base a83de76 and on its
# parent 018bb23 (and the harness does not import before a83de76), so they never go red: no regression pin.
NOT_RED = {("kill", "a1"), ("kill", "d"), ("pause", "a1"), ("pause", "d"), ("stop", "a1"), ("stop", "a2"),
           ("stop", "d"), ("flatten", "a1"), ("flatten", "d")}
MATRIX = [pytest.param(a, w, id=f"{a}-{w}") for a in ACTIONS for w in WHENS if (a, w) not in NOT_RED]


def _scenario(w, action, when, holding=True):
    if holding:
        w.buy()
    w.tick(fill=False)  # running, steady
    if when == "a1":
        w.pm(action)
        w.tick(fill=False)
        w.pm("reset")
    elif when == "a2":
        w.pm(action)
        w.pm("reset")
    elif when == "b":
        w.pm("reset")
        w.supervise()  # the reset's flatten is queued (holding) or the reset is done (flat)
        w.pm(action)
    elif when == "c":
        w.pm("reset")
        w.supervise()
        w.tick(fill=False)  # the process takes the reset's flatten; its order works
        w.pm(action)
    elif when == "d":
        w.pm("reset")
        w.drive()
        assert w.store.pending_reset(NAME) is None
        w.pm(action)
    w.drive()
    assert w.store.pending_reset(NAME) is None, "the reset never finished"


@pytest.mark.parametrize("action,when", MATRIX)
def test_matrix_pm_intent_survives_the_reset(tmp_path, monkeypatch, action, when):
    """Source: quant-review/v2-p1/kill-reset-188-scripts/test_kr188_probes.py::test_matrix_pm_intent_survives_the_reset.
    Findings: P1-KR-2 (kill-a2, pause-a2, flatten-a2), P1-KR-1 (kill-b, flatten-b), P1-KR-3 (stop-b, stop-c); the
    other params (kill-c, pause-b, pause-c, flatten-c) carried no finding mark (m13-U5, kill-reset-188.md: "fixed by
    #188")."""
    w = World(tmp_path, monkeypatch)
    _scenario(w, action, when)
    (run,) = w.store.reset_runs()
    fresh = w.store.sleeve(NAME)
    opens = w.fresh_opens()
    fresh = w.store.sleeve(NAME)
    detail = f"fresh run: status={fresh.status!r} ({fresh.status_reason!r}) desired={fresh.desired_state!r}; PM said {w.said}"
    # never a double flatten, never a flip
    assert len(w.orders.get(NAME, [])) <= 1, f"{len(w.orders[NAME])} closing orders for one position; {detail}"
    assert w.qty(NAME) == 0 and w.store.journal_book(run, START)["qty"] == 0, detail
    assert len(_sells(w, NAME, run)) == 1, detail
    if action == "stop":
        assert fresh.desired_state == "stopped" and not opens, f"the PM's Stop was lost: {detail}"
    else:
        assert fresh.status == "paused" and not opens, f"the PM's {action} was lost: {detail}"
    # the journal names the PM for what the PM did
    want = {"kill": "flatten everything", "pause": "pause", "stop": "stop", "flatten": "flatten"}[action]
    rows = [d for n in (NAME, run) for d in w.store.decisions(n, limit=500)] + w.store.decisions(limit=500)
    assert any(d["action"] == want and d["actor"] == "PM" for d in rows), f"no PM {want} row; {detail}"


@pytest.mark.parametrize("action", ACTIONS)
def test_matrix_flat_strategy_pm_acts_between_the_reset_request_and_the_supervisor(tmp_path, monkeypatch, action):
    """A flat strategy: the reset finishes on the supervisor's next pass. The PM acts in between.

    Source: quant-review/v2-p1/kill-reset-188-scripts/test_kr188_probes.py::
    test_matrix_flat_strategy_pm_acts_between_the_reset_request_and_the_supervisor. Finding: P1-KR-3 ([stop]); the
    other params carried no finding mark (kill-reset-188.md: "fixed by #188")."""
    w = World(tmp_path, monkeypatch)
    w.tick()
    w.pm("reset")
    w.pm(action)
    w.drive()
    fresh = w.store.sleeve(NAME)
    opens = w.fresh_opens()
    fresh = w.store.sleeve(NAME)
    detail = f"fresh: {fresh.status!r} ({fresh.status_reason!r}) desired={fresh.desired_state!r}; {w.said}"
    if action == "stop":
        assert fresh.desired_state == "stopped" and not opens, detail
    elif action == "flatten":
        assert "nothing to flatten" in w.said[-1] or (fresh.status == "paused" and not opens), detail
    else:
        assert fresh.status == "paused" and not opens, detail


# ----------------------------------------------------------------------------------------------------------------
# 2. The kill switch reaches a strategy whose reset flatten is queued (route-level twin of the PR's own test)
# ----------------------------------------------------------------------------------------------------------------
def test_kill_switch_route_acts_on_a_strategy_whose_reset_flatten_is_queued(tmp_path, monkeypatch):
    """Source: quant-review/v2-p1/kill-reset-188-scripts/test_kr188_probes.py::
    test_kill_switch_route_acts_on_a_strategy_whose_reset_flatten_is_queued. Finding: P1-KR-1."""
    w = World(tmp_path, monkeypatch)
    w.buy()
    w.tick(fill=False)
    w.pm("reset")
    w.supervise()
    assert [c["command"] for c in w.store.pending_commands(NAME)] == ["flatten"]  # the reset's, not yet taken
    r = w.pm("kill")
    page = w.c.get("/risk", auth=AUTH).text
    holds = w.store.pending_reset(NAME)
    assert [d for d in _pm_rows(w, NAME, "flatten") if d["reason"].startswith("Book kill switch")], (f"the kill switch skipped it: {r.headers.get('location')}; "
                                          f"PM-facing: {'Waiting to sell: s1' in page}; reset pending {bool(holds)}")


def test_per_strategy_flatten_route_while_the_reset_flatten_is_queued(tmp_path, monkeypatch):
    """Source: quant-review/v2-p1/kill-reset-188-scripts/test_kr188_probes.py::
    test_per_strategy_flatten_route_while_the_reset_flatten_is_queued. Finding: P1-KR-1."""
    w = World(tmp_path, monkeypatch)
    w.buy()
    w.tick(fill=False)
    w.pm("reset")
    w.supervise()
    r = w.pm("flatten")
    assert "command_error" not in r.headers["location"], r.headers["location"]


# ----------------------------------------------------------------------------------------------------------------
# 6. Two resets back to back with a kill switch between them
# ----------------------------------------------------------------------------------------------------------------
def test_two_resets_kill_switch_between_them_before_the_fresh_first_tick(tmp_path, monkeypatch):
    """Source: quant-review/v2-p1/kill-reset-188-scripts/test_kr188_probes.py::
    test_two_resets_kill_switch_between_them_before_the_fresh_first_tick. Finding: P1-KR-2."""
    w = World(tmp_path, monkeypatch)
    w.buy()
    w.tick(fill=False)
    w.pm("reset")
    w.drive()
    assert len(w.store.reset_runs()) == 1
    w.pm("kill")  # fresh run, flat, running: the kill switch queues its flatten (pause)
    w.pm("reset", reason="second")  # before the fresh process ticks
    w.drive()
    assert len(w.store.reset_runs()) == 2
    s = w.store.sleeve(NAME)
    opens = w.fresh_opens()
    assert not opens, f"kill switch lost across the second reset: {s.status!r} {s.status_reason!r}; {w.said}"
