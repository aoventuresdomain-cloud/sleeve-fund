"""Backtests run in the background, one at a time, so a long run neither blocks the page that
started it nor shares the server's memory with a second long run. The page shows how far it is.

Each run is its own process: the engine can abort the whole process (a Rust panic, or the kernel
ending it for memory), and that must end the run, not the dashboard. Its memory goes back to the
server when it finishes, and the pages stay responsive while it works."""

from __future__ import annotations

import multiprocessing
import os
import secrets
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

KEEP = 100  # finished jobs remembered, so a progress page reloaded later still finds its result


@dataclass
class Job:
    id: str
    key: str
    title: str
    status: str = "queued"  # queued | running | done | error
    progress: float = 0.0  # 0 to 1, from the run's simulated clock
    run_id: str | None = None  # the saved backtest, when done
    error: str = ""
    created: float = field(default_factory=time.time)
    done_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def view(self) -> dict:
        return {"id": self.id, "status": self.status, "progress": round(self.progress, 4), "run_id": self.run_id,
                "error": self.error, "title": self.title}


class JobError(Exception):
    """A run that failed, already in words for the page."""


class Jobs:
    def __init__(self, workers: int = 1, isolate: bool | None = None) -> None:
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="backtest")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        # Tests run in-process, where their stand-in price feeds apply; the server always isolates.
        self.isolate = os.environ.get("BACKTEST_ISOLATE", "1") != "0" if isolate is None else isolate

    def submit(self, key: str, title: str, fn, *args) -> Job:
        """Queue fn(progress, job_id, *args) -> saved run id, where progress(f) reports 0 to 1. With
        isolation fn and args go to a fresh process, so both must pickle (fn a module-level function).
        The same settings already queued or running share one job."""
        with self._lock:
            for j in self._jobs.values():
                if j.key == key and j.status in ("queued", "running"):
                    return j
            job = Job(id=secrets.token_hex(6), key=key, title=title)
            self._jobs[job.id] = job
            for old in sorted(self._jobs.values(), key=lambda j: j.created)[:-KEEP]:
                if old.status in ("done", "error"):
                    self._jobs.pop(old.id, None)
        self._pool.submit(self._run, job, fn, args)
        return job

    def _run(self, job: Job, fn, args: tuple) -> None:
        job.status = "running"
        try:
            if self.isolate:
                job.run_id = _in_child(job, fn, args)
            else:
                job.run_id = fn(lambda f: setattr(job, "progress", f), job.id, *args)
            job.progress, job.status = 1.0, "done"
        except JobError as exc:
            job.error, job.status = str(exc), "error"
        except Exception as exc:  # noqa: BLE001 - the page shows what failed
            job.error, job.status = _message(exc), "error"
        finally:
            job.done_event.set()

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def ahead_of(self, job: Job) -> int:
        """Jobs that will run before this one."""
        with self._lock:
            return sum(1 for j in self._jobs.values()
                       if j.status in ("queued", "running") and j.created < job.created)


def _in_child(job: Job, fn, args: tuple):
    # Spawned, not forked: the dashboard has threads and open connections a fork would copy mid-use.
    ctx = multiprocessing.get_context("spawn")
    progress = ctx.Value("d", 0.0, lock=False)
    recv, send = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(fn, job.id, args, progress, send), daemon=True,
                       name=f"backtest-{job.id}")
    proc.start()
    send.close()
    while proc.is_alive():
        job.progress = progress.value
        proc.join(0.5)
    job.progress = progress.value
    try:
        kind, value = recv.recv() if recv.poll() else (None, None)
    except EOFError:  # ended without a word: poll() is also true at end of file
        kind = value = None
    if kind == "ok":
        return value
    raise JobError(value if kind == "error" else _crash(proc.exitcode))


def _child(fn, job_id: str, args: tuple, progress, send) -> None:
    try:  # if memory runs out, the kernel ends this run rather than the dashboard
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass

    def report(f: float) -> None:
        progress.value = f

    try:
        send.send(("ok", fn(report, job_id, *args)))
    except Exception as exc:  # noqa: BLE001 - sent to the page
        send.send(("error", _message(exc)))


def _crash(code: int | None) -> str:
    """Why a run's process ended without an answer."""
    if code == -signal.SIGKILL:
        return "the backtest ran out of memory and was stopped. Try a shorter period or a longer interval."
    return (f"the backtest engine stopped unexpectedly (exit code {code}). The dashboard is unaffected; "
            "the settings may have hit an engine limit.")


def _message(exc: Exception) -> str:
    from sleeve_fund.dashboard.app import backtest_error

    return backtest_error(exc)
