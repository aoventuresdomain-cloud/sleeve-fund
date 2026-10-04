"""Backtests run in the background, one at a time, so a long run neither blocks the page that
started it nor shares the server's memory with a second long run. The page shows how far it is."""

from __future__ import annotations

import secrets
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


class Jobs:
    def __init__(self, workers: int = 1) -> None:
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="backtest")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, key: str, title: str, work) -> Job:
        """Queue work(job) -> saved run id. The same settings already queued or running share one job."""
        with self._lock:
            for j in self._jobs.values():
                if j.key == key and j.status in ("queued", "running"):
                    return j
            job = Job(id=secrets.token_hex(6), key=key, title=title)
            self._jobs[job.id] = job
            for old in sorted(self._jobs.values(), key=lambda j: j.created)[:-KEEP]:
                if old.status in ("done", "error"):
                    self._jobs.pop(old.id, None)
        self._pool.submit(self._run, job, work)
        return job

    def _run(self, job: Job, work) -> None:
        job.status = "running"
        try:
            job.run_id = work(job)
            job.progress, job.status = 1.0, "done"
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


def _message(exc: Exception) -> str:
    from sleeve_fund.dashboard.app import backtest_error

    return backtest_error(exc)
