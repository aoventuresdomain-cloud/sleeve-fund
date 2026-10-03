"""The research pipeline: every strategy, how far it has got, and the evidence behind it."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

from sleeve_fund.strategies import REGISTRY

STAGES = ["Idea", "Tested", "Passed G1", "Paper", "Passed G2", "Live"]
_G1 = re.compile(r"^\|\s*G1 test[^|]*\|\s*(PASS|FAIL|WARN)\s*\|\s*([^|]*)\|", re.M)
_DATASET = re.compile(r"^Dataset `([^`]+)`", re.M)


def sheet_facts(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    g1, ds = _G1.search(text), _DATASET.search(text)
    return {"name": path.stem, "g1": g1.group(1) if g1 else None, "evidence": g1.group(2).strip() if g1 else "",
            "dataset": ds.group(1) if ds else "unknown", "mtime": path.stat().st_mtime}


def strategies(tearsheets: Path, sleeves: list) -> list[dict]:
    sheets = [sheet_facts(p) for p in sorted(tearsheets.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)]
    out = []
    for name in sorted(REGISTRY):
        spec = importlib.import_module(f"sleeve_fund.strategies.{name}").SPEC
        mine = [s for s in sheets if s["name"] == name or s["name"].startswith(name + "_")]
        real = [s for s in mine if s["dataset"] != "synthetic"]  # synthetic runs prove plumbing, not edge
        g1 = next((s["g1"] for s in real if s["g1"]), None)
        running = [s for s in sleeves if s.strategy == name]
        if g1 == "PASS":
            stage = 3 if running else 2
        elif running:
            stage = 3  # in paper without a real G1 pass: an observation sleeve
        else:
            stage = 1 if mine else 0
        out.append({"name": name, "spec": spec, "sheets": mine, "g1": g1, "stage": stage,
                    "sleeves": running, "observation": bool(running) and g1 != "PASS"})
    return out
