"""The Results page for one study (v2 P1-6): the guardrail figures on one screen, each read from the tear sheet
the study wrote, never recomputed."""

from __future__ import annotations

import re
from datetime import timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse

from sleeve_fund.dashboard import development as dev
from sleeve_fund.dashboard.pipeline import _CHECK
from sleeve_fund.research.guardrails import MIN_OOS_TRADES

NEARBY = "Holds at nearby settings"
OOS_TRADES = "Enough out-of-sample trades to judge"
_VARIANTS = re.compile(r"^- (\d+) distinct variants? of this idea tried so far", re.M)
_PROJECT = re.compile(r"^- (\d+) ideas? and (\d+) distinct variants? tested so far", re.M)
_DSR = re.compile(r"^- Deflated Sharpe: (?:([\d.]+%) probability|(can't be computed here: .*?)(?= N uncertain|$))", re.M)
_UNCERTAIN = re.compile(r"N uncertain: [^\n]*")
UK = ZoneInfo("Europe/London")


def facts(path: Path) -> dict:
    """The figures the page shows, as the tear sheet states them."""
    text = path.read_text(encoding="utf-8")
    s = dev.read_sheet(path)
    checks = {name: (verdict, evidence.strip()) for name, verdict, evidence in _CHECK.findall(text)}
    variants, project, dsr = _VARIANTS.search(text), _PROJECT.search(text), _DSR.search(text)
    uncertain = _UNCERTAIN.search(text)
    trades = s.get("oos_trades")
    near = checks.get(NEARBY)
    bar_met = trades is not None and trades >= MIN_OOS_TRADES
    return dict(
        s=s, banner=dev.banner(s), breakeven=dev.breakeven_text(s),
        when=s["when"].astimezone(UK) if s["when"].tzinfo else s["when"].replace(tzinfo=timezone.utc).astimezone(UK),
        variants=int(variants.group(1)) if variants else None,
        project_ideas=int(project.group(1)) if project else None,
        project_variants=int(project.group(2)) if project else None,
        dsr=dsr.group(1) if dsr and dsr.group(1) else None,
        dsr_why=dsr.group(2) if dsr and dsr.group(2) else None,
        uncertain=uncertain.group(0).rstrip(".") if uncertain else None,
        nearby=near, trades=trades, min_trades=MIN_OOS_TRADES, trades_met=bar_met,
        trades_check=checks.get(OOS_TRADES),
        not_judged=(s["evidence"] if s.get("g1") == "NOT JUDGED" else None),
    )


def register(app: FastAPI, *, page, sheet_path, require_pm) -> None:
    @app.get("/results/{sheet}", response_class=HTMLResponse)
    def results(request: Request, sheet: str, _: str = Depends(require_pm)):
        path = sheet_path(sheet)
        return page(request, "results.html", title=sheet, candle_words=dev.candle_words, **facts(path))
