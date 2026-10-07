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
retried (HoE ruling on QA P1-L5, 6 Oct 2026). The retry waits off the writer, so other writes (fills, heartbeats, an
entry's flush) keep flowing; only that order's own later rows (its status, fill and timing, which need its row) are
held back until it is written, then written in order."""

from __future__ import annotations

import queue
import threading
import time

from sleeve_fund.store import Store

QUEUED = frozenset({"record_order", "record_timing", "update_order", "record_fill", "event", "record_equity",
                    "heartbeat", "set_signal_state", "merge_order_signal"})
OPENING = ("entry", "rebalance")  # as LongFlatStrategy's OPENING_INTENTS
RETRY_SECONDS = (1.0, 5.0, 30.0)  # an exit's journal row: retried after each wait before the failure is kept


def _order_of(name: str, args, kwargs) -> str | None:
    """The order a queued write belongs to, for the writes that need its row."""
    if name in ("record_fill", "record_order"):
        return kwargs.get("order_id")
    if name in ("update_order", "merge_order_signal"):
        return args[0] if args else kwargs.get("order_id")
    if name == "record_timing":
        return args[1] if len(args) > 1 else kwargs.get("order_id")
    return None


class QueuedStore:
    def __init__(self, store: Store, retry_seconds: tuple[float, ...] = RETRY_SECONDS) -> None:
        self._store = store
        self._q: queue.Queue = queue.Queue()
        self._failed: BaseException | None = None
        self.retry_seconds = retry_seconds
        # An exit whose row failed, by order id: the row, the retries spent, and that order's later writes held.
        # Touched only by the writer thread; a retry comes back through the queue.
        self._held: dict[str, dict] = {}
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
                if name == "_retry":
                    self._retry(*args)
                elif (oid := _order_of(name, args, kwargs)) in self._held:
                    self._held[oid]["after"].append((name, args, kwargs))  # needs the order's row: after it
                else:
                    self._apply(name, args, kwargs)
            except BaseException as exc:  # noqa: BLE001 - kept for the next read to raise
                print(f"journal: {name} failed: {exc!r}")
                self._failed = self._failed or exc
            finally:
                self._q.task_done()

    def _apply(self, name: str, args, kwargs) -> None:
        """One write. An exit's order row that fails is an incident, retried after a wait: the order is at the
        venue already."""
        try:
            getattr(self._store, name)(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - an exit's row is retried; anything else is kept
            if name != "record_order" or not self.retry_seconds:
                raise
            self._held[kwargs["order_id"]] = {"row": (args, kwargs), "tries": 0, "after": []}
            self._incident(args, kwargs, f"couldn't be written ({type(exc).__name__}: {exc}); the order went to "
                           "the venue anyway, as an exit always does. Retrying the write; check the journal against "
                           "the venue if this stays open")
            self._later(kwargs["order_id"])

    def _later(self, oid: str) -> None:
        """Retry the held row after its next wait, through the queue, so the writer never sleeps on it."""
        wait = self.retry_seconds[self._held[oid]["tries"]]
        timer = threading.Timer(wait, self._q.put, args=(("_retry", (oid,), {}),))
        timer.daemon = True
        timer.start()

    def _retry(self, oid: str, last: bool = False) -> None:
        """One more try at a held row; `last` (settle's deadline, as the process stops) gives up if it fails."""
        held = self._held.get(oid)
        if held is None:
            return  # given up on at settle's deadline before this wait ran out
        args, kwargs = held["row"]
        try:
            self._store.record_order(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - retried until the waits run out
            held["tries"] += 1
            if held["tries"] < len(self.retry_seconds) and not last:
                self._later(oid)
                return
            del self._held[oid]
            self._incident(args, kwargs, f"still couldn't be written after {held['tries'] + 1} tries "
                           f"({type(exc).__name__}: {exc}). Its fill may be missing from the journal too: "
                           f"{len(held['after'])} later row(s) of this order (status, fill, timing) need its row and "
                           "were not written. Reconcile the journal against the venue's fills for this order")
            raise
        del self._held[oid]
        for name, a, k in held["after"]:  # its status, fill and timing, in the order they came
            try:
                self._apply(name, a, k)
            except BaseException as exc:  # noqa: BLE001 - kept for the next read to raise
                print(f"journal: {name} failed: {exc!r}")
                self._failed = self._failed or exc

    def _incident(self, args, kwargs, what: str) -> None:
        sleeve = args[0] if args else kwargs.get("sleeve", "?")
        try:
            self._store.event(sleeve, "error", "incident",
                              f"The journal row of {kwargs.get('intent', 'an')} order {kwargs.get('order_id', '?')} "
                              + what)
        except Exception as again:  # noqa: BLE001 - the database is gone for events too: the log has it
            print(f"journal: incident for {kwargs.get('order_id')} not recorded: {again!r}")

    def settle(self, timeout: float = 60.0) -> None:
        """Wait until no exit's row is waiting on a retry (each written, or given up on), then flush. At the
        deadline each row still held has one last try, and the give-up incident if that fails too."""
        deadline = time.monotonic() + timeout
        while True:
            self.flush()
            if not self._held:
                return
            if time.monotonic() > deadline:
                for oid in list(self._held.copy()):  # the writer owns _held: the last try goes through the queue
                    self._q.put(("_retry", (oid, True), {}))
                self.flush()
                return
            time.sleep(0.01)

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
