"""Live smoke check for the market data hub (v2 P1-1): run a venue's hub for a few minutes against its public
feed, subscribe two clients to the same instrument, and fail unless both receive trades, quotes and the same
closed 1-minute bars (values and close stamps). Public data only; no credentials.

    python scripts/hub_smoke.py --venue kraken --pair BTC/USD --minutes 4"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time

from sleeve_fund.hub.__main__ import build
from sleeve_fund.venues import venue


def _client(port: int, iid: str, out: dict) -> None:
    for _ in range(100):  # the relay subscribes when the node starts
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=120)
            break
        except OSError:
            time.sleep(0.2)
    s.sendall(json.dumps({"v": 1, "sub": [iid]}).encode() + b"\n")
    for line in s.makefile("rb"):
        m = json.loads(line)
        out.setdefault(m["t"], []).append(m)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", default="kraken")
    ap.add_argument("--pair", default="BTC/USD")
    ap.add_argument("--minutes", type=float, default=4)
    args = ap.parse_args()
    profile = venue(args.venue)
    iid = f"{profile.symbol_of(args.pair)}.{profile.name}"
    node, fanout, relay = build(profile, 0)
    a, b = {}, {}
    for out in (a, b):
        threading.Thread(target=_client, args=(fanout.port, iid, out), daemon=True).start()
    threading.Timer(args.minutes * 60, node.handle().stop).start()
    node.run()
    fanout.stop()
    live = lambda o: [(m["ts"], m["o"], m["h"], m["l"], m["c"], m["v"]) for m in o.get("bar", []) if not m["refilled"]]  # noqa: E731
    print(f"{iid}: client A {', '.join(f'{len(v)} {k}' for k, v in sorted(a.items()))}")
    print(f"{iid}: client B {', '.join(f'{len(v)} {k}' for k, v in sorted(b.items()))}")
    for m in a.get("bar", []):
        print("bar", m)
    for m in a.get("gap", []):
        print("gap", m)
    trades = a.get("trade", [])
    if trades:
        lag = sorted((t["recv"] - t["ts"]) / 1e6 for t in trades)
        print(f"trade receive lag: median {lag[len(lag) // 2]:.0f} ms, p95 {lag[int(len(lag) * 0.95)]:.0f} ms")
    failures = []
    if not a.get("hb"):
        failures.append("no heartbeat")
    if not trades:
        failures.append("no trades")
    if not a.get("quote"):
        failures.append("no quotes")
    if len(live(a)) < 1:
        failures.append("no live closed bar")
    if live(a)[:len(live(b))] != live(b)[:len(live(a))]:
        failures.append("the two clients got different bars")
    if any(m["ts"] % 60_000_000_000 for m in a.get("bar", [])):
        failures.append("a bar not stamped on a minute close")
    print("FAIL: " + "; ".join(failures) if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
