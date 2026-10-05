from datetime import timedelta

import pytest

from sleeve_fund.paper.config import from_store
from sleeve_fund.store import Store, utcnow
from sleeve_fund.supervisor import Proc, clear, decide, seed


class FakePopen:
    pid = 1

    def __init__(self, alive=True, code=None):
        self._alive, self.returncode = alive, code

    def poll(self):
        return None if self._alive else self.returncode


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path}/t.db")


@pytest.fixture
def sleeve(store):
    return store.create_sleeve(name="s", strategy="trend_filter", instrument="BTC/USD",
                               bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=1000)


def test_start_stop_and_backoff(sleeve):
    now = utcnow()
    assert decide(sleeve, Proc(), now) == "start"
    assert decide(sleeve, Proc(next_start=now + timedelta(seconds=30)), now) == "wait"
    assert decide(sleeve, Proc(popen=FakePopen(alive=False, code=1)), now) == "crashed"
    sleeve.desired_state = "stopped"
    assert decide(sleeve, Proc(popen=FakePopen(), started_at=now), now) == "stop"
    assert decide(sleeve, Proc(), now) == "none"


def test_stale_heartbeat_restarts_only_after_grace(sleeve):
    now = utcnow()
    young = Proc(popen=FakePopen(), started_at=now - timedelta(minutes=1))
    old = Proc(popen=FakePopen(), started_at=now - timedelta(minutes=10))
    assert decide(sleeve, young, now) == "none"
    assert decide(sleeve, old, now) == "restart_stale"
    sleeve.heartbeat_at = now - timedelta(seconds=20)
    assert decide(sleeve, old, now) == "none"


def test_seed_is_idempotent(store):
    paths = ["configs/examples/btc_trend_smoke.toml", "configs/examples/btc_trend_daily.toml"]
    assert seed(store, paths) == ["btc-trend-smoke", "btc-trend-daily"]
    assert seed(store, paths) == []
    s = store.sleeve("btc-trend-smoke")
    assert s.params == {"fast": 5, "slow": 20, "max_notional": 1000}
    assert s.risk_profile == "aggressive"


def test_the_seeded_strategies_are_the_two_test_strategies(store):
    import glob

    assert sorted(seed(store, sorted(glob.glob("configs/sleeves/*.toml")))) == [
        "ping-pong-ls-binance", "ping-pong-ls-test", "ping-pong-test", "rsi-bands-15m-ls-test", "rsi-bands-15m-test",
        "rsi-bands-ls-binance", "rsi-bands-ls-test", "rsi-bands-test"]
    # RSI(14) warms up on ten periods of history, whatever the file says (PM, 5 Oct 2026).
    assert store.sleeve("rsi-bands-test").warmup_bars == 140
    m15 = store.sleeve("rsi-bands-15m-ls-test")
    assert (m15.bar_spec, m15.warmup_bars, m15.desired_state) == ("15-MINUTE-LAST-INTERNAL", 140, "stopped")
    assert store.sleeve("rsi-bands-test").bar_spec == "1-MINUTE-LAST-INTERNAL"
    # The long/short pair trade a perpetual and are added stopped, for the PM to start.
    ls = store.sleeve("ping-pong-ls-test")
    assert (ls.params["market"], ls.params["allow_short"], ls.desired_state, ls.status) == ("perp", True, "stopped", "stopped")
    assert store.sleeve("ping-pong-test").desired_state == "running"
    # The Binance copies trade Binance's own perpetual, stopped until the PM starts them; the rest stay on Kraken.
    bn = store.sleeve("rsi-bands-ls-binance")
    assert (bn.venue, bn.instrument, bn.params["market"], bn.desired_state) == ("BINANCE", "BTC/USDT", "perp", "stopped")
    assert from_store(bn).instrument_id == "BTCUSDT-PERP.BINANCE" and ls.venue is None


def test_clear_puts_every_strategy_away_once_and_keeps_its_journal(store, sleeve, tmp_path):
    """The PM's clean slate (4 Oct 2026): every strategy on the book is stopped and archived, its journal
    untouched; the entry applies once, so restarts and strategies added afterwards are left alone."""
    store.set_desired_state("s", "running")
    store.record_fill("s", side="BUY", qty=0.01, price=100.0, fee=0.008, order_id="o1", trade_id="t1")
    store.record_fill("s", side="SELL", qty=0.01, price=101.0, fee=0.008, order_id="o2", trade_id="t2")
    store.create_sleeve(name="old", strategy="buy_and_hold", instrument="ETH/USD",
                        bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=500)
    store.set_desired_state("old", "stopped")
    store.archive("old")
    path = tmp_path / "clear.toml"
    path.write_text('[[clear]]\nid = "2026-10-04"\nreason = "PM asked for a clean slate"\n')
    assert clear(store, str(path)) == ["s"]
    assert store.sleeve("s").desired_state == "stopped" and set(store.archived()) == {"s", "old"}
    assert len(store.fills("s")) == 2
    assert [d["action"] for d in store.decisions("s")][:2] in (["archive", "stop"], ["stop", "archive"])
    seed(store, ["configs/sleeves/ping_pong_test.toml"])
    assert clear(store, str(path)) == []  # applied already: the new strategy stays
    assert "ping-pong-test" not in store.archived()
    assert any(e["kind"] == "book_cleared" for e in store.events())



def test_clear_flattens_a_strategy_still_holding_a_position_before_archiving_it(store, sleeve, tmp_path):
    """Archiving takes a strategy off the book: one still holding a position would drop it from every
    total with nothing watching it (review round 11). It is flattened instead (a PM flatten, which pauses it
    too), started if it was stopped so the flatten can trade, and archived on a later start once flat."""
    store.set_desired_state("s", "stopped")
    store.record_fill("s", side="BUY", qty=0.01, price=100.0, fee=0.008, order_id="o1", trade_id="t1")
    path = tmp_path / "clear.toml"
    path.write_text('[[clear]]\nid = "2026-10-05"\nreason = "clean slate"\n')
    assert clear(store, str(path)) == []
    assert store.sleeve("s").desired_state == "running" and "s" not in store.archived()
    assert [c["command"] for c in store.pending_commands("s")] == ["flatten"]
    assert any(e["kind"] == "clear_held" and "0.01" in e["message"] for e in store.events("s"))
    assert store.decisions(action="clear") == []
    assert clear(store, str(path)) == [] and len(store.pending_commands("s")) == 1  # not asked twice
    store.record_fill("s", side="SELL", qty=0.01, price=101.0, fee=0.008, order_id="o2", trade_id="t2")
    assert clear(store, str(path)) == ["s"]  # flat now: the next start finishes the entry
    assert store.sleeve("s").desired_state == "stopped"
    assert clear(store, str(path)) == []


def test_a_clean_slate_waiting_on_a_flatten_finishes_without_a_restart_and_touches_nothing_added_since(
        store, sleeve, tmp_path, monkeypatch):
    """PM, 5 Oct 2026: a fresh book once the clean slate deploys. One still holding is flattened first; the
    supervisor retries the slate about every minute, so the book clears once it is flat rather than on the
    next deploy, and a strategy the PM added in between is left alone."""
    from sleeve_fund import supervisor
    from sleeve_fund.supervisor import Supervisor

    monkeypatch.setattr(supervisor, "POLL_SECONDS", 0)
    store.set_desired_state("s", "stopped")
    store.record_fill("s", side="BUY", qty=0.01, price=100.0, fee=0.008, order_id="o1", trade_id="t1")
    path = tmp_path / "clear.toml"
    path.write_text('[[clear]]\nid = "2026-10-05"\nreason = "fresh book"\n')
    assert clear(store, str(path)) == []
    store.create_sleeve(name="added-since", strategy="buy_and_hold", instrument="ETH/USD",
                        bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=500)
    store.record_fill("s", side="SELL", qty=0.01, price=101.0, fee=0.008, order_id="o2", trade_id="t2")
    sup = Supervisor(store, clear_path=str(path))
    sup.step = lambda: setattr(sup, "_stopping", True)  # one pass of the loop
    sup.run()
    assert "s" in store.archived() and "added-since" not in store.archived()
    assert store.sleeve("added-since").desired_state == "running"
    assert set(store.previous_book()) == {"s"}


def test_the_deploy_log_says_what_the_book_holds(store, sleeve):
    from sleeve_fund.supervisor import book_line

    store.create_sleeve(name="fresh", strategy="buy_and_hold", instrument="ETH/USD",
                        bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=10000, desired_state="stopped")
    store.record_equity("s", equity=990, cash=990, qty=0, price=1, benchmark=1000)
    assert book_line(store) == "s 1,000 (running, has history); fresh 10,000 (stopped, no history)"


def test_a_strategy_archived_while_still_holding_is_flattened_and_leaves_the_book(store, sleeve, tmp_path):
    """PM, 5 Oct 2026: after the fresh book the Portfolio still read 30,000 with a position open. The 4 Oct
    slate had archived a strategy still long (before holders were flattened first), and later slates
    skipped it as already put away, so it stayed in the book's figures. A slate now flattens it too."""
    from sleeve_fund.store import sleeve_archive_t

    store.set_desired_state("s", "stopped")
    store.record_fill("s", side="BUY", qty=0.01, price=100.0, fee=0.008, order_id="o1", trade_id="t1")
    with store.engine.begin() as conn:  # archived with its position, as the 4 Oct slate left it
        conn.execute(sleeve_archive_t.insert().values(sleeve="s", archived_at=utcnow()))
    store.decide("system", "clear", "2026-10-04: first slate (1 put away)")
    store.create_sleeve(name="kept", strategy="buy_and_hold", instrument="ETH/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=500, desired_state="stopped")
    path = tmp_path / "clear.toml"
    path.write_text('[[clear]]\nid = "2026-10-04"\nreason = "first slate"\n'
                    '[[clear]]\nid = "flat"\nreason = "fresh book"\nkeep = ["kept"]\n')
    assert "s" not in store.previous_book()  # still holding: still in the book
    assert clear(store, str(path)) == []
    assert store.sleeve("s").desired_state == "running"
    assert [c["command"] for c in store.pending_commands("s")] == ["flatten"]
    from sleeve_fund.supervisor import book_figures, book_line

    assert "s 1,000 (archived, still holding 0.01, running)" in book_line(store)
    assert book_figures(store).startswith("from 1,500.00")
    store.record_fill("s", side="SELL", qty=0.01, price=101.0, fee=0.008, order_id="o2", trade_id="t2")
    assert clear(store, str(path)) == ["s"]
    assert store.sleeve("s").desired_state == "stopped" and set(store.previous_book()) == {"s"}
    assert book_figures(store) == "from 500.00, equity 500.00, fees 0.00, positions none, first mark none"
    assert store.sleeve("kept").desired_state == "stopped" and "kept" not in store.archived()


def test_clear_leaves_the_strategies_it_keeps(store, sleeve, tmp_path):
    store.set_desired_state("s", "running")
    store.create_sleeve(name="kept", strategy="buy_and_hold", instrument="ETH/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=500)
    path = tmp_path / "clear.toml"
    path.write_text('[[clear]]\nid = "k"\nreason = "tidy"\nkeep = ["kept"]\n')
    assert clear(store, str(path)) == ["s"]
    assert "kept" not in store.archived() and store.sleeve("kept").desired_state == "running"


def test_the_shipped_clear_file_reads(store):
    assert clear(store, "configs/clear.toml") == []
    assert sorted(d["reason"].split(":")[0] for d in store.decisions(action="clear")) == [
        "2026-10-04", "2026-10-05", "2026-10-05-flat"]


def test_the_shipped_clear_keeps_only_the_binance_strategies(store, tmp_path):
    """PM, 5 Oct 2026: archive every strategy not in use; the two Binance long/short test strategies stay,
    for the PM to start once the Bybit demo mirror is deployed."""
    import glob

    path = tmp_path / "clear.toml"
    path.write_text('[[clear]]\nid = "2026-10-04"\nreason = "first slate"\n')
    clear(store, str(path))  # the book as it stood: the 4 Oct slate applied, then everything seeded
    seed(store, sorted(glob.glob("configs/sleeves/*.toml")))
    put_away = clear(store, "configs/clear.toml")
    assert sorted(put_away) == ["ping-pong-ls-test", "ping-pong-test", "rsi-bands-15m-ls-test", "rsi-bands-15m-test",
                                "rsi-bands-ls-test", "rsi-bands-test"]
    assert sorted(s.name for s in store.sleeves() if s.name not in store.archived()) == [
        "ping-pong-ls-binance", "rsi-bands-ls-binance"]


def test_changed_settings_restart_a_running_strategy_once(store, sleeve, monkeypatch):
    """Saving new settings restarts the running process so it trades under them; the strategy itself
    never acts on the reload, and a stopped strategy just picks them up when it starts."""
    import subprocess

    from sleeve_fund.supervisor import Supervisor

    started = []
    monkeypatch.setattr(subprocess, "Popen", lambda args, env=None: started.append(args) or FakePopen())
    sup = Supervisor(store)
    sup.step()
    assert len(started) == 1
    assert store.change_settings("s", risk_profile="conservative", params={}, warmup_bars=0)
    assert store.change_settings("s", risk_profile="conservative", params={"stop_loss": 0.05}, warmup_bars=0)
    assert len(store.pending_commands("s")) == 1  # two saves, one restart
    monkeypatch.setattr(sup, "_stop", lambda name, proc, why: setattr(proc, "popen", None))
    sup.step()
    assert len(started) == 2 and store.pending_reload("s") is None
    assert any(e["kind"] == "settings_applied" for e in store.events("s"))
    sup.step()
    assert len(started) == 2
    store.set_desired_state("s", "stopped")
    assert store.change_settings("s", risk_profile="balanced", params={}, warmup_bars=0) is False
    assert store.pending_reload("s") is None and store.sleeve("s").risk_profile == "balanced"


def test_the_strategy_leaves_a_reload_to_the_supervisor(store, sleeve):
    from sleeve_fund.paper.runtime import SleeveRuntime

    store.change_settings("s", risk_profile="balanced", params={}, warmup_bars=0)
    rt = SleeveRuntime(store, "s")
    rt.on_start(0.008)
    rt.tick(equity=1000, cash=1000, qty=0, price=1)
    assert store.pending_reload("s") is not None


def test_a_strategy_added_before_its_file_asked_for_the_demo_mirror_gets_it_without_a_restart(store):
    path = "configs/sleeves/ping_pong_ls_binance.toml"
    seed(store, [path])
    s = store.sleeve("ping-pong-ls-binance")
    params = {k: v for k, v in s.params.items() if k != "demo_mirror"}
    store._update_sleeve(s.name, params=params, desired_state="running")  # as seeded before 5 Oct 11:20
    assert seed(store, [path]) == []
    s = store.sleeve("ping-pong-ls-binance")
    assert s.params["demo_mirror"] is True and {k: v for k, v in s.params.items() if k != "demo_mirror"} == params
    assert store.pending_commands(s.name) == []  # no reload: the running strategy is not restarted
    assert store.decisions(s.name)[0]["action"] == "mirror"
    # A PM who turned it off keeps it off.
    store._update_sleeve(s.name, params={**params, "demo_mirror": False})
    seed(store, [path])
    assert store.sleeve("ping-pong-ls-binance").params["demo_mirror"] is False


@pytest.mark.sanity
def test_a_reset_flattens_puts_the_run_away_and_starts_again_at_the_starting_capital(store):
    """PM, 5 Oct 2026. Nothing is deleted: the run so far moves to its own archived name under Previous book."""
    from sleeve_fund.supervisor import Supervisor

    store.create_sleeve(name="bn-ls", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params={"market": "perp", "allow_short": True, "demo_mirror": True},
                        venue="binance")
    store.record_fill("bn-ls", side="BUY", qty=0.076, price=86_000.0, fee=3.27, order_id="o1", trade_id="t1")
    store.record_equity("bn-ls", equity=9_990.0, cash=3_460.0, qty=0.076, price=86_000.0, benchmark=10_000.0)
    store.event("bn-ls", "info", "fill", "BUY 0.076")
    store.request_reset("bn-ls", "Test finished")
    sup = Supervisor(store, python="true")
    sup.reset_pending()  # still long: flattened first, nothing moved yet
    (cmd,) = store.pending_commands("bn-ls")
    assert cmd["command"] == "flatten" and cmd["reason"].startswith("Reset strategy: Test finished")
    sup.reset_pending()
    assert len(store.pending_commands("bn-ls")) == 1  # not sent twice
    store.mark_applied(cmd["id"])
    store.record_fill("bn-ls", side="SELL", qty=0.076, price=86_100.0, fee=3.27, order_id="o2", trade_id="t2")
    sup.reset_pending()
    assert store.pending_reset("bn-ls") is None
    run = next(iter(store.reset_runs()))
    assert run.startswith("bn-ls--") and run in store.archived() and run in store.previous_book()
    fresh = store.sleeve("bn-ls")
    book = store.journal_book("bn-ls", 10_000)
    assert (book["qty"], book["cash"]) == (0.0, 10_000.0)
    assert store.fills("bn-ls") == [] and store.equity_series("bn-ls") == [] and store.events("bn-ls")[0]["kind"] == "reset"
    assert fresh.desired_state == "running" and fresh.params == store.sleeve(run).params
    assert store.sleeve(run).venue == "BINANCE" and store.sleeve(run).desired_state == "stopped"
    assert store.journal_book(run, 10_000)["qty"] == 0.0 and len(store.fills(run)) == 2  # kept, not deleted
    assert store.pending_resyncs()[-1]["sleeve"] == "bn-ls"  # the demo copy is set flat on the paper terms
    assert store.decisions("bn-ls")[0]["reason"].startswith("Started afresh at 10,000")


def test_a_flat_stopped_strategy_resets_at_once_and_stays_stopped(store):
    from sleeve_fund.supervisor import Supervisor

    seed(store, ["configs/sleeves/ping_pong_test.toml"])
    store.set_desired_state("ping-pong-test", "stopped")
    store.request_reset("ping-pong-test", "Settings changed")
    Supervisor(store, python="true").reset_pending()
    assert store.pending_reset() is None and store.sleeve("ping-pong-test").desired_state == "stopped"
    with pytest.raises(ValueError, match="archived"):
        store.request_reset(next(iter(store.reset_runs())), "x")


@pytest.mark.sanity
def test_a_reset_keeps_a_pause_or_halt_and_restarts_a_running_strategy(store):
    """Round 13, U13-4: Reset all un-paused strategies the kill switch had paused (a pause leaves desired_state
    running, and the fresh run started with no status). A pause or halt in force when the reset is asked for is
    still in force after it, so nothing trades until the PM resumes; a running strategy still restarts."""
    from sleeve_fund.supervisor import Supervisor

    for name in ("bn-paused", "bn-halted", "bn-running"):
        store.create_sleeve(name=name, strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                            starting_balance=10_000, params={"market": "perp", "allow_short": True}, venue="binance")
    store.set_status("bn-paused", "paused", "kill switch: paused by PM")
    store.set_status("bn-halted", "halted", "drawdown 21% hit the 20% limit")
    store.set_status("bn-running", "running")
    for name in ("bn-paused", "bn-halted", "bn-running"):
        store.request_reset(name, "Reset all")
    Supervisor(store, python="true").reset_pending()
    assert store.pending_reset() is None
    paused, halted, running = (store.sleeve(n) for n in ("bn-paused", "bn-halted", "bn-running"))
    assert (paused.status, paused.paused_until) == ("paused", None) and "kill switch" in paused.status_reason
    assert halted.status == "halted" and "20% limit" in halted.status_reason
    assert running.status == "stopped" and running.desired_state == "running"  # starts afresh and runs
    # The fresh paper process reads that status and keeps it until a resume (review round 10, B10-3).
