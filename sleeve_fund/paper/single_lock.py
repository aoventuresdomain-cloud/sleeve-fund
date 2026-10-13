"""SINGLE-LOCK: one paper process per strategy (Independent Quant Advisor, 13 Oct 2026 17:55 and 18:10 UK). A second
process for a running strategy would double its orders and positions, so it takes the strategy's lock before it connects
to market data or the venue, and stops there when another live process holds it. The OS drops the lock when its holder
dies, so a crashed process never blocks its own restart."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path

EXIT_DUPLICATE = 75  # a paper process refused because another holds its strategy's lock (EX_TEMPFAIL)
DUPLICATE_REFUSED, INSTANCE_LOCK = "duplicate_refused", "instance_lock"


def instance_lock(name: str, root=None):
    """The strategy's single-instance lock: an exclusive, non-blocking flock on <root>/<name>.lock (root: SLEEVE_LOCK_DIR,
    else data/locks), held by the returned open file for as long as the process keeps it. None when another live
    process holds it."""
    root = Path(root or os.environ.get("SLEEVE_LOCK_DIR", Path(__file__).resolve().parents[2] / "data" / "locks"))
    root.mkdir(parents=True, exist_ok=True)
    fh = open(root / f"{name}.lock", "a+")  # noqa: SIM115 - held open for the process's life
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        return None
    fh.seek(0)
    fh.truncate()
    fh.write(f"{os.getpid()}\n")  # who holds it, for a person checking
    fh.flush()
    return fh


def duplicate_refused(store, name: str) -> int:
    """A second process for `name` stops before it connects to anything: journalled and alerted once while it lasts
    (until a process takes the lock again). Returns its exit code."""
    last = store.last_event(name, (DUPLICATE_REFUSED, INSTANCE_LOCK))
    if last is None or last["kind"] != DUPLICATE_REFUSED:
        store.event(name, "error", DUPLICATE_REFUSED, f"Duplicate refused: another process already runs this strategy "
                    f"(it holds its lock), so this one (pid {os.getpid()}) stopped before placing anything")
    return EXIT_DUPLICATE
