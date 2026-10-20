"""B-40: the framework's cancel takes the order's client id (NautilusTrader 2.0.0rc5 Strategy.cancel_order), never the
order. On 274ecd88 two sites passed the order: the restart reconcile's cancel of a venue-held orphan and the reconcile
halt's cancel of its resting entries. Each raised TypeError, so no cancel went out: a restart left the orphan resting at
the venue, and a cash/qty-mismatch halt left its entries working. The same Strategy code runs with the demo copy on
(mirror.py sends no cancel of its own), so the guard cell over every call in sleeve_fund covers both paths."""

from __future__ import annotations

import ast
import pathlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from nautilus_trader.model import ClientOrderId
from test_gate_ledger import _drop_schemas, _store, _strategies  # noqa: F401

from sleeve_fund.paper import orphan_guard
from sleeve_fund.strategies.base import EXIT_LEGS, LongFlatStrategy

CLOSE = datetime(2026, 10, 8, 9, 0, tzinfo=UTC)
SRC = pathlib.Path(__file__).resolve().parents[1] / "sleeve_fund"


class _Cancels:
    """Strategy.cancel_order's contract: a ClientOrderId, else TypeError, as the engine raises."""

    def __init__(self):
        self.sent = []

    def __call__(self, coid):
        if not isinstance(coid, ClientOrderId):
            raise TypeError(f"Argument 'client_order_id' has incorrect type (expected ClientOrderId, got "
                            f"{type(coid).__name__})")
        self.sent.append(str(coid))


def _order(oid):
    return SimpleNamespace(client_order_id=ClientOrderId(oid))


@pytest.fixture
def store():
    s = _store()
    _strategies(s)
    return s


@pytest.mark.parametrize("demo_copy", [False, True])
def test_a_restart_with_a_venue_held_orphan_sends_exactly_one_cancel(store, demo_copy):
    orphan_guard.qa_resting_entry(store, "a", "o-left", CLOSE)
    left = _order("o-left")
    cancels = _Cancels()
    me = SimpleNamespace(
        _guarding=lambda: True, cancel_order=cancels, strategy_id="S-a",
        _cfg=SimpleNamespace(demo_mirror=demo_copy),
        runtime=SimpleNamespace(store=store, name="a", in_process_venue=None, sole_instance=None),
        cache=SimpleNamespace(orders_open=lambda strategy_id=None: [left],
                              order=lambda coid: left if str(coid) == "o-left" else None),
        clock=SimpleNamespace(timestamp_ns=lambda: int((CLOSE + timedelta(minutes=20)).timestamp() * 1e9)))
    LongFlatStrategy._reconcile_resting(me)
    assert cancels.sent == ["o-left"]  # one cancel, by its id: on 274ecd88 the engine raised and none went out


@pytest.mark.parametrize("demo_copy", [False, True])
def test_a_reconcile_halt_cancels_every_resting_entry_it_means_to(demo_copy):
    entries = [_order("o-e1"), _order("o-e2")]
    stop, target = _order("o-sl"), _order("o-tp")
    cancels = _Cancels()
    me = SimpleNamespace(
        cancel_order=cancels, strategy_id="S-a", _cfg=SimpleNamespace(demo_mirror=demo_copy),
        decisions={"o-e1": {"intent": "entry"}, "o-e2": {"intent": "rebalance"},
                   "o-sl": {"intent": EXIT_LEGS[0]}, "o-tp": {"intent": EXIT_LEGS[1]}},
        cache=SimpleNamespace(orders_open=lambda strategy_id=None: [*entries, stop, target]))
    LongFlatStrategy._cancel_all_but_exits(me)
    assert cancels.sent == ["o-e1", "o-e2"]  # the resting stop and target stay, as the halt means


def _cancel_args():
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "cancel_order":
                yield f"{path.relative_to(SRC.parent)}:{node.lineno}", node.args[0] if node.args else None


def _is_client_id(arg) -> bool:
    if isinstance(arg, ast.Attribute) and arg.attr == "client_order_id":
        return True
    return isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == "ClientOrderId"


def test_every_cancel_in_the_fund_passes_a_client_order_id():
    calls = list(_cancel_args())
    assert len(calls) >= 10  # the scan found the Strategy's cancels
    bad = [where for where, arg in calls if not _is_client_id(arg)]
    assert bad == [], f"cancel_order is passed something that isn't a ClientOrderId at {bad}"
