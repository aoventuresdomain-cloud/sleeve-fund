"""The research pipeline: every strategy, how far it has got, and the evidence behind it."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

from sleeve_fund import markets
from sleeve_fund.research.guardrails import G1_RULES
from sleeve_fund.strategies import REGISTRY

STAGES = ["Idea", "Tested", "Passed G1", "Paper", "Passed G2", "Live"]
_CHECK = re.compile(r"^\|\s*([^|]*?)\s*\|\s*(PASS|FAIL|WARN|INFO|NOT JUDGED|N/A)\s*\|\s*([^|]*)\|", re.M)
_DATASET = re.compile(r"^Dataset `([^`]+)`", re.M)
_NAME = re.compile(r"^# Tear sheet: (\S+)", re.M)
_TESTED = re.compile(r"^Tested on `([^`]+)` at (\d+)-minute bars", re.M)
_SETTINGS = re.compile(r"^Settings: (.+)$", re.M)
_RULES = re.compile(r"^G1 rules: (\S+)$", re.M)


def _g1(text: str) -> tuple[str | None, str, list[str]]:
    """(verdict, Sharpe evidence, failed checks) from a sheet's check rows. G1 passes only when every
    check that can fail passed, whatever the Sharpe row says, so sheets written before the verdict
    line existed are read the same strict way."""
    rows = _CHECK.findall(text)
    sharpe = next((r for r in rows if r[0].startswith("G1 test")), None)
    if sharpe is None:
        return None, "", []
    unjudged = [(label, ev.strip()) for label, verdict, ev in rows if verdict == "NOT JUDGED"]
    if unjudged:  # neither a pass nor a fail: the study's runs can't be judged (tearsheet.g1_verdict)
        return "NOT JUDGED", "; ".join(ev or label for label, ev in unjudged), [label for label, _ in unjudged]
    failed = [(label, ev.strip()) for label, verdict, ev in rows if verdict not in ("PASS", "INFO", "N/A")]
    rules = _RULES.search(text)
    if not failed and (rules is None or rules.group(1) != G1_RULES):
        # Passed under older, looser rules (a 10-trade bar, no nearby-settings check, or one centred on the
        # defaults): not a pass now, and never one the path to live can count, until re-run (QA F3).
        then = f"the G1 rules of {rules.group(1)}" if rules else "G1 rules"
        return ("NOT JUDGED", f"Old bar: passed under {then} older than the current {G1_RULES}; re-run the study "
                "to judge it", ["G1 rules"])
    evidence = sharpe[2].strip()
    if failed:
        # Lead with what failed: a strong Sharpe beside "Fail" otherwise reads like a pass.
        evidence = "Failed: " + "; ".join(f"{label} ({ev})" if ev else label for label, ev in failed) + "."
        if not any(label.startswith("G1 test") for label, _ in failed):  # else its evidence is already there
            evidence += " " + sharpe[2].strip()
    return ("FAIL" if failed else "PASS"), evidence, [label for label, _ in failed]


def sheet_facts(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    ds, name, tested, settings = _DATASET.search(text), _NAME.search(text), _TESTED.search(text), _SETTINGS.search(text)
    g1, evidence, failed = _g1(text)
    return {"name": path.stem, "strategy": name.group(1) if name else None,
            "g1": g1, "evidence": evidence, "failed": failed,
            "dataset": ds.group(1) if ds else "unknown", "mtime": path.stat().st_mtime,
            "settings": settings.group(1) if settings else "",
            # Sheets from before the instrument and bars were written down can't vouch for either.
            "instrument": tested.group(1).upper() if tested else None,
            "minutes": int(tested.group(2)) if tested else None}


def _sheets(tearsheets: Path) -> list[dict]:
    return [sheet_facts(p) for p in sorted(tearsheets.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)]


def _real(sheets: list[dict], strategy: str) -> list[dict]:
    """A strategy's sheets on real data (synthetic runs prove plumbing, not edge; a sheet that doesn't
    say what it was tested on proves nothing), newest first."""
    return [s for s in sheets if s["strategy"] == strategy and s["dataset"] not in ("synthetic", "unknown")
            and "synthetic" not in s["dataset"]]


def studied_as(params: dict | None) -> bool:
    """Whether a G1 study tests this configuration: studies run spot, long only, so a strategy on a
    perpetual or one that shorts has no G1 evidence until perp studies come (L4/L5). A spot long-only
    pass must never tick G1 for a long/short strategy (round 12, M12-U3)."""
    return not markets.is_perp(params) and not (params or {}).get("allow_short")


def g1_for(tearsheets: Path, strategy: str, instrument: str, minutes: int, params: dict | None = None) -> str | None:
    """G1 for exactly this strategy on this instrument at this bar length: the newest real tear sheet
    of that combination decides. A pass on one instrument or interval says nothing about another, and
    none says anything about a perpetual or long/short configuration (studied_as)."""
    if not studied_as(params):
        return None
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
        if not spec.listed:
            continue
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
        backed = [s for s in running if f"{s.instrument.upper()}@{spec_minutes(s.bar_spec)}" in passed_on
                  and studied_as(s.params)]
        if running:
            stage = 3  # in paper; without a pass on its own instrument and bars it is an observation
        elif passed_on:
            stage = 2
        else:
            stage = 1 if mine else 0
        where = [f"{i} {_every(m)}" for i, m in sorted(k for k, v in latest.items() if v == "PASS")]
        observing = [s for s in running if s not in backed]
        out.append({"name": name, "spec": spec, "sheets": mine, "g1": g1, "passed_on": passed_on, "passed_where": where,
                    "stage": stage, "sleeves": running, "observation": bool(observing),
                    "observing": [{"name": s.name, "where": f"{s.instrument} {_every(spec_minutes(s.bar_spec))}"}
                                  for s in observing]})
    return out
