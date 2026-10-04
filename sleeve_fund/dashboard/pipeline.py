"""The research pipeline: every strategy, how far it has got, and the evidence behind it."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

from sleeve_fund.strategies import REGISTRY

STAGES = ["Idea", "Tested", "Passed G1", "Paper", "Passed G2", "Live"]
_G1 = re.compile(r"^\|\s*G1 test[^|]*\|\s*(PASS|FAIL|WARN)\s*\|\s*([^|]*)\|", re.M)
_DATASET = re.compile(r"^Dataset `([^`]+)`", re.M)
_NAME = re.compile(r"^# Tear sheet: (\S+)", re.M)
_TESTED = re.compile(r"^Tested on `([^`]+)` at (\d+)-minute bars", re.M)


def sheet_facts(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    g1, ds, name, tested = _G1.search(text), _DATASET.search(text), _NAME.search(text), _TESTED.search(text)
    return {"name": path.stem, "strategy": name.group(1) if name else None,
            "g1": g1.group(1) if g1 else None, "evidence": g1.group(2).strip() if g1 else "",
            "dataset": ds.group(1) if ds else "unknown", "mtime": path.stat().st_mtime,
            # Sheets from before the instrument and bars were written down can't vouch for either.
            "instrument": tested.group(1).upper() if tested else None,
            "minutes": int(tested.group(2)) if tested else None}


def _sheets(tearsheets: Path) -> list[dict]:
    return [sheet_facts(p) for p in sorted(tearsheets.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)]


def _real(sheets: list[dict], strategy: str) -> list[dict]:
    """A strategy's sheets on real data (synthetic runs prove plumbing, not edge), newest first."""
    return [s for s in sheets if s["strategy"] == strategy and s["dataset"] != "synthetic"]


def g1_for(tearsheets: Path, strategy: str, instrument: str, minutes: int) -> str | None:
    """G1 for exactly this strategy on this instrument at this bar length: the newest real tear sheet
    of that combination decides. A pass on one instrument or interval says nothing about another."""
    hit = next((s for s in _real(_sheets(tearsheets), strategy)
                if s["instrument"] == instrument.upper() and s["minutes"] == minutes and s["g1"]), None)
    return hit["g1"] if hit else None


def _every(minutes: int) -> str:
    return {1440: "daily", 60: "hourly"}.get(minutes, f"{minutes}-minute")


def strategies(tearsheets: Path, sleeves: list) -> list[dict]:
    from sleeve_fund.data import spec_minutes

    sheets = _sheets(tearsheets)
    out = []
    for name in sorted(REGISTRY):
        spec = importlib.import_module(f"sleeve_fund.strategies.{name}").SPEC
        mine = [s for s in sheets if s["strategy"] == name]
        real = _real(sheets, name)
        # The newest verdict per instrument and bar length; a pass lists where it holds.
        latest: dict[tuple, str] = {}
        for s in real:
            if s["g1"] and s["instrument"]:
                latest.setdefault((s["instrument"], s["minutes"]), s["g1"])
        passed_on = sorted(f"{i}@{m}" for (i, m), v in latest.items() if v == "PASS")
        # No pass that says where it holds: show the latest other verdict (an old, unplaced pass is none).
        g1 = "PASS" if passed_on else next((s["g1"] for s in real if s["g1"] and s["g1"] != "PASS"), None)
        running = [s for s in sleeves if s.strategy == name]
        backed = [s for s in running if f"{s.instrument.upper()}@{spec_minutes(s.bar_spec)}" in passed_on]
        if running:
            stage = 3  # in paper; without a pass on its own instrument and bars it is an observation
        elif passed_on:
            stage = 2
        else:
            stage = 1 if mine else 0
        where = [f"{i} {_every(m)}" for i, m in sorted(k for k, v in latest.items() if v == "PASS")]
        out.append({"name": name, "spec": spec, "sheets": mine, "g1": g1, "passed_on": passed_on, "passed_where": where,
                    "stage": stage,
                    "sleeves": running, "observation": len(backed) < len(running)})
    return out
