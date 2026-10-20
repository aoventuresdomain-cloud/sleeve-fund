"""TEC-B05-KEYS (HoE 21:16 UK, Advisor 21:18 UK, 20 Oct): the compensating cells for letting the order client's own
modules hold keys inside paper/. OFF, a paper node never builds a key-holding client; the connect check's read path
can't write; the list of paper modules that may mention keys is closed, by exact path."""

import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

from sleeve_fund.paper.exec_client import Limiter
from sleeve_fund.paper.venue_reads import VenueReads

ROOT = Path(__file__).resolve().parent.parent
SLEEVE = ROOT / "configs" / "sleeves" / "ping_pong_test.toml"

# A node start in a fresh interpreter, so the modules it imports are its own, never a neighbouring test's. A client
# can't be built without its module, so a module never imported is a client never built.
_START = """
import json, sys
from sleeve_fund.paper.config import load_sleeve
from sleeve_fund.paper.node import build_node
try:
    build_node(load_sleeve(sys.argv[1]), log_level="ERROR", asset_fetch=dict).dispose()
    outcome = "built"
except Exception as e:  # noqa: BLE001 - the cell reads what refused it
    outcome = type(e).__name__
print(json.dumps({"outcome": outcome, "modules": sorted(m for m in sys.modules if m.startswith("sleeve_fund."))}))
"""
KEY_HOLDERS = ("sleeve_fund.paper.venue_reads", "sleeve_fund.paper.venue_stops", "nautilus_trader.adapters.bybit")


def _start(env_over):
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("EXEC_", "KRAKEN_"))}
    env.update(env_over)
    out = subprocess.run([sys.executable, "-c", _START, str(SLEEVE)], capture_output=True, text=True, env=env,
                         cwd=ROOT, timeout=300, check=False)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_off_a_paper_node_start_never_builds_or_imports_a_key_holding_client():
    got = _start({})
    assert got["outcome"] == "built"
    assert not [m for m in KEY_HOLDERS if m in got["modules"]]


def test_on_a_paper_node_start_is_refused_before_any_client_is_built():
    got = _start({"EXEC_MODE": "demo"})
    assert got["outcome"] == "PaperSafetyError"  # assert_off: ON is refused at the node while there is no data hub
    assert not [m for m in KEY_HOLDERS if m in got["modules"]]


# By exact path. Adding a module here needs the HoE's and the Advisor's sign-off, recorded on the board (TEC-B05-KEYS).
MAY_HOLD_KEYS = {"sleeve_fund/paper/exec_client.py", "sleeve_fund/paper/venue_reads.py",
                 "sleeve_fund/paper/venue_stops.py"}


def test_only_the_listed_paper_modules_mention_a_key():
    mention = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "sleeve_fund" / "paper").rglob("*.py")
                     if "api_key" in p.read_text() or "api_secret" in p.read_text())
    assert mention == sorted(MAY_HOLD_KEYS)


def test_test_paper_exempts_only_the_read_path_and_only_its_read_only_client_call():
    """CR K1: the exemption from test_paper's key check is venue_reads.py alone, and its one key-passing line is the
    read-only VenueStops it builds."""
    import test_paper

    assert test_paper.ORDER_CLIENT == {"sleeve_fund/paper/venue_reads.py"}
    src = (ROOT / "sleeve_fund" / "paper" / "venue_reads.py").read_text()
    passing = [line.strip() for line in src.splitlines() if "api_key=" in line or "api_secret=" in line]
    assert len(passing) == 1 and passing[0].startswith("self._client = VenueStops(")
    assert re.search(r"VenueStops\([^)]*read_only=True\)", src)


@pytest.fixture
def requests_seen(monkeypatch):
    seen = []

    def fake(req, timeout=None):
        seen.append(req.get_method())
        raise OSError("no venue in cells")

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return seen


@pytest.mark.parametrize("write", ["place", "cancel", "_post"])
def test_the_connect_check_read_path_cannot_write(requests_seen, write):
    reads = VenueReads("https://venue.invalid", api_key="k", api_secret="s", limiter=Limiter({}))
    call = {"place": lambda c: c.place("sf-1", side="SELL", qty=1, level=1),
            "cancel": lambda c: c.cancel("sf-1"),
            "_post": lambda c: c._post("order/create", {})}[write]
    with pytest.raises(PermissionError):
        call(reads._client)
    assert "POST" not in requests_seen and requests_seen == []


def test_the_read_path_still_reads(requests_seen):
    reads = VenueReads("https://venue.invalid", api_key="k", api_secret="s", limiter=Limiter({}))
    assert reads.server_ms() is None  # unanswered here, so None: the check fails on it, never a pass
    assert requests_seen == ["GET"]
