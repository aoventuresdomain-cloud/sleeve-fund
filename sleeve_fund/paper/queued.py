"""Journal writes off the decision path (v2 P1-2): a paper strategy's journal, with the writes a decision or a fill
makes (the order and why, its timing, its status, the fill, events, marks) handed to one writer thread instead of
made inline, so the order goes to the venue without waiting on the database.

Nothing reads a stale journal: any other call (a read) first waits for the writes queued before it, in order, so
the runtime and the strategy see their own writes exactly as before. A write that failed is raised by the next
read, where the strategy's handler reports it, rather than lost. The writer's connections come from the store's
pool, which keeps them open between writes."""

from __future__ import annotations

import queue
import threading

from sleeve_fund.store import Store

QUEUED = frozenset({"record_order", "record_timing", "update_order", "record_fill", "event", "record_equity",
                    "heartbeat", "set_signal_state"})


class QueuedStore:
    def __init__(self, store: Store) -> None:
        self._store = store
        self._q: queue.Queue = queue.Queue()
        self._failed: BaseException | None = None
        threading.Thread(target=self._write, name="journal-writer", daemon=True).start()

    def _write(self) -> None:
        while True:
            name, args, kwargs = self._q.get()
            try:
                getattr(self._store, name)(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - kept for the next read to raise
                print(f"journal: {name} failed: {exc!r}")
                self._failed = self._failed or exc
            finally:
                self._q.task_done()

    def flush(self) -> None:
        """Wait for every queued write; raise the first one that failed."""
        self._q.join()
        failed, self._failed = self._failed, None
        if failed is not None:
            raise RuntimeError(f"a journal write failed: {failed!r}") from failed

    def __getattr__(self, name: str):
        if name in QUEUED:
            return lambda *args, **kwargs: self._q.put((name, args, kwargs))
        self.flush()
        return getattr(self._store, name)
