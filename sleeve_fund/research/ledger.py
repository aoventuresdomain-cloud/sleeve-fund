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
        return self.holdout_opened(idea, dataset) is not None

    def holdout_opened(self, idea: str, dataset: str) -> dict | None:
        """The ledger entry that first opened this idea's holdout on this data, at any bar length, or None."""
        return self.holdouts().get((idea, _without_bars(dataset)))

    def holdouts(self) -> dict[tuple[str, str], dict]:
        """The first holdout opening of each idea on each dataset, keyed by (idea, dataset without bars)."""
        out: dict[tuple[str, str], dict] = {}
        for e in self.entries():
            if e["stage"] == "holdout":
                out.setdefault((e["idea"], _without_bars(e["dataset"])), e)
        return out


def _without_bars(dataset: str) -> str:
    """A dataset name without its bar-length suffix ("venue-btcusd-store-60m" -> "venue-btcusd-store")."""
    return re.sub(r"-\d+m$", "", dataset)


def opened_words(entry: dict) -> str:
    """When and on what bars a holdout was opened, in words: "on 04 Oct 2026, on daily bars"."""
    m = re.search(r"-(\d+)m$", entry["dataset"])
    minutes = int(m.group(1)) if m else 1440
    bars = ("daily" if minutes == 1440 else f"{minutes // 60}-hour" if minutes % 60 == 0 else f"{minutes}-minute")
    day = datetime.fromisoformat(entry["ts"]).strftime("%d %b %Y")
    return f"on {day}, on {bars} bars"
