"""B-39 MUST (HoE, 20 Oct; QT3's production-shaped run, quant-review/v3/prod-run/README.md): a perp Strategy saved on
an instrument whose venue tiers aren't kept (VS-1) crashed its node at start with a NoMaintenanceTable traceback. A
Strategy that fails a start check is refused in words, never crashes: the supervisor doesn't start it, and a
--db-sleeve process started by hand stops before it trades. It is marked stopped with the reason shown, and the other
Strategies start as before. On SQLite, and on Postgres where TEST_DATABASE_URL is set (its own schema)."""
from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import create_engine, text

from sleeve_fund.store import Store

# The venue as it is: conftest otherwise gives every pair a stand-in table, which hid this crash (team-learnings A1)
pytestmark = [pytest.mark.no_maintenance_tables]
PERP = {"market": "perp", "allow_short": True, "stop_loss": 0.02}
PG = os.environ.get("TEST_DATABASE_URL")


@pytest.fixture(params=["sqlite", "postgres"])
def url(request, tmp_path, monkeypatch):
    monkeypatch.setenv("SLEEVE_LOCK_DIR", str(tmp_path / "locks"))
    if request.param == "sqlite":
        u = f"sqlite:///{tmp_path}/b39.db"
    else:
        if not PG:
            pytest.skip("Postgres cells run where TEST_DATABASE_URL is set (CI)")
        schema = f"b39_{uuid.uuid4().hex[:10]}"
        admin = create_engine(PG)
        with admin.begin() as c:
            c.execute(text(f"CREATE SCHEMA {schema}"))
        sep = "&" if "?" in PG else "?"
        u = f"{PG}{sep}options=-csearch_path%3D{schema}"
    monkeypatch.setenv("DATABASE_URL", u)
    yield u
    if request.param == "postgres":
        with admin.begin() as c:
            c.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
        admin.dispose()


def _fund(store):
    from datetime import UTC, datetime, timedelta

    at = datetime.now(UTC) - timedelta(minutes=5)
    store.settle_flow(store.fund_flow("deposit", 100_000, actor="pm", reason="test", rebase=True, now=at), at)


def test_a_perp_with_no_kept_tiers_is_refused_in_words_and_the_others_still_start(url, monkeypatch):
    import sleeve_fund.paper.node as node
    from sleeve_fund.paper.node import main

    def built(*a, **k):
        raise AssertionError("a node was built for a Strategy that fails its start check")

    for f in ("build_node", "build_two_leg_node", "build_legs_node"):
        monkeypatch.setattr(node, f, built)
    from sleeve_fund.supervisor import Supervisor

    store = Store(url)
    _fund(store)
    for name, pair in (("eth-perp", "ETH/USDT"), ("btc-perp", "BTC/USDT")):
        store.create_sleeve(name=name, strategy="ping_pong", instrument=pair, bar_spec="1-MINUTE-LAST-INTERNAL",
                            starting_balance=0, params=PERP, venue="BINANCE")  # the paper venue that lists perps
        store.set_desired_state(name, "running")

    assert main(["--db-sleeve", "eth-perp"]) == 1  # refused, not a traceback
    sup = Supervisor(store, python="true")
    assert sup._refused("eth-perp")
    refused = [e["message"] for e in store.events("eth-perp") if e["kind"] == "start_refused"]
    assert len(refused) == 2 and all("ETH/USDT has no maintenance table kept" in m for m in refused), refused
    row = store.sleeve("eth-perp")
    assert (row.status, row.desired_state) == ("stopped", "stopped")
    assert "not started: ETH/USDT has no maintenance table kept" in row.status_reason
    assert not sup._refused("btc-perp")  # the Strategy beside it starts as before
    assert store.sleeve("btc-perp").desired_state == "running"


def test_the_supervisor_never_spawns_an_untiered_perp_and_starts_the_tiered_one(tmp_path, monkeypatch):
    """FE's cell (fe-plans/b39/test_b39_node_tiers_cell.py, adopted): through the supervisor's own step, the ETH/USDT
    perp is stopped with its reason and never spawned, while the BTC/USDT perp beside it is."""
    from types import SimpleNamespace

    from sleeve_fund import supervisor

    store = Store(f"sqlite:///{tmp_path}/t.db")
    for name, pair in (("eth-perp", "ETH/USDT"), ("btc-perp", "BTC/USDT")):
        store.create_sleeve(name=name, strategy="dip_buy", instrument=pair, bar_spec="1-HOUR-LAST-INTERNAL",
                            starting_balance=1_000, venue="binance", params={"market": "perp", "stop_atr": 2.0})
    started = []
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **k: started.append(a[0][-1]) or
                        SimpleNamespace(pid=1, poll=lambda: None))
    supervisor.Supervisor(store).step()
    s = store.sleeve("eth-perp")
    assert "eth-perp" not in started and "btc-perp" in started, started
    assert s.desired_state == "stopped" and "ETH/USDT has no maintenance table kept" in s.status_reason, s.status_reason
    assert [e for e in store.events("eth-perp", limit=10) if e["kind"] == "start_refused"]
