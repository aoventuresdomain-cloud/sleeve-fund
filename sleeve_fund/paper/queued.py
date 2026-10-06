"""Journal writes off the decision path (v2 P1-2): a paper strategy's journal, with the writes a decision or a fill
makes (the order and why, its timing, its status, the fill, events, marks) handed to one writer thread instead of
made inline, so the order goes to the venue without waiting on the database.

Nothing reads a stale journal: any other call (a read) first waits for the writes queued before it, in order, so
the runtime and the strategy see their own writes exactly as before. A write that failed is raised by the next
read, where the strategy's handler reports it, rather than lost. The writer's connections come from the store's
pool, which keeps them open between writes.

An order that opens or adds (entry, rebalance) is journaled inline, before the venue sees it, and a failed write
stops it there: nothing is opened without its reason on record. Any other order (an exit, a stop) is queued and
goes to the venue whatever happens to its row: a failed write raises an incident in the alerts inbox and is
retried (HoE ruling on QA P1-L5, 6 Oct 2026)."""

from __future__ import annotations

import queue
import threading
import time

from sleeve_fund.store import Store

QUEUED = frozenset({"record_order", "record_timing", "update_order", "record_fill", "event", "record_equity",
                    "heartbeat", "set_signal_state"})
OPENING = ("entry", "rebalance")  # as LongFlatStrategy's OPENING_INTENTS
RETRY_SECONDS = (1.0, 5.0, 30.0)  # an exit's journal row: retried after each wait before the failure is kept


class QueuedStore:
    def __init__(self, store: Store, retry_seconds: tuple[float, ...] = RETRY_SECONDS) -> None:
        self._store = store
        self._q: queue.Queue = queue.Queue()
        self._failed: BaseException | None = None
        self.retry_seconds = retry_seconds
        threading.Thread(target=self._write, name="journal-writer", daemon=True).start()

    def record_order(self, *args, **kwargs) -> None:
        if kwargs.get("intent") in OPENING:
            self.flush()  # in order after what is queued, and nothing opens while an earlier write has failed
            self._store.record_order(*args, **kwargs)  # raises: the strategy's _submit stops before the venue
            return
        self._q.put(("record_order", args, kwargs))

    def _write(self) -> None:
        while True:
            name, args, kwargs = self._q.get()
            try:
                self._apply(name, args, kwargs)
            except BaseException as exc:  # noqa: BLE001 - kept for the next read to raise
                print(f"journal: {name} failed: {exc!r}")
                self._failed = self._failed or exc
            finally:
                self._q.task_done()

    def _apply(self, name: str, args, kwargs) -> None:
        """One write. An exit's order row that fails is an incident, retried: the order is at the venue already."""
        for attempt, wait in enumerate((*self.retry_seconds, None)):
            try:
                getattr(self._store, name)(*args, **kwargs)
                return
            except BaseException as exc:  # noqa: BLE001 - an exit's row is retried; anything else is kept
                if name != "record_order" or wait is None:
                    raise
                if attempt == 0:
                    self._incident(args, kwargs, exc)
                time.sleep(wait)

    def _incident(self, args, kwargs, exc: BaseException) -> None:
        sleeve = args[0] if args else kwargs.get("sleeve", "?")
        try:
            self._store.event(sleeve, "error", "incident",
                              f"The journal row of {kwargs.get('intent', 'an')} order {kwargs.get('order_id', '?')} "
                              f"couldn't be written ({type(exc).__name__}: {exc}); the order went to the venue "
                              "anyway, as an exit always does. Retrying the write; check the journal against the "
                              "venue if this stays open")
        except Exception as again:  # noqa: BLE001 - the database is gone for events too: the log has it
            print(f"journal: incident for {kwargs.get('order_id')} not recorded: {again!r}")

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
