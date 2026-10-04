"""Idea counter: an append-only log of every variant ever evaluated.

Nothing is ever removed. The tear sheet reads the counts from here, so a
strategy found after 200 tries is judged against a 200-try bar.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


class IdeaLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, *, idea: str, family: str, params: dict, dataset: str, stage: str, sharpe: float) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "idea": idea,
            "family": family,
            "params": params,
            "dataset": dataset,
            "stage": stage,
            "sharpe": round(float(sharpe), 4),
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def counts(self) -> dict:
        rows = [e for e in self.entries() if e["family"] != "benchmark"]
        variants = {(e["idea"], json.dumps(e["params"], sort_keys=True), e["dataset"]) for e in rows}
        by_family: dict[str, set] = {}
        for e in rows:
            by_family.setdefault(e["family"], set()).add(e["idea"])
        return {
            "ideas": len({e["idea"] for e in rows}),
            "variants": len(variants),
            "evaluations": len(rows),
            "ideas_by_family": {k: len(v) for k, v in sorted(by_family.items())},
        }

    def sharpes(self, family: str | None = None) -> list[float]:
        return [
            e["sharpe"]
            for e in self.entries()
            if e["family"] != "benchmark" and (family is None or e["family"] == family)
        ]

    def holdout_used(self, idea: str, dataset: str) -> bool:
        """Whether this idea's holdout on this data was opened before, at any bar length: the same days
        seen once on daily bars are not fresh on hourly ones."""
        base = _without_bars(dataset)
        return any(
            e["stage"] == "holdout" and e["idea"] == idea and _without_bars(e["dataset"]) == base for e in self.entries()
        )


def _without_bars(dataset: str) -> str:
    """A dataset name without its bar-length suffix ("venue-btcusd-store-60m" -> "venue-btcusd-store")."""
    return re.sub(r"-\d+m$", "", dataset)
