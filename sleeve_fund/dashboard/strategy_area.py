"""B-16: the Strategy area (was Development). Every built Strategy, read from its rule file in configs/definitions/ at
page load, beside the hand-coded Models: a list with status chips, and a page for each with why it should work, its
rules in plain English (strategy_words, written from the checked definition), indicators, exits and its settings
budget, its test history, and where it runs on the fund. Results, Data coverage and Archived tabs.

Spec: /mnt/project-files/sleeve-fund/fe-plans/b16-strategy-tab-spec.md; look: ui-v2/strategy-area-mockup.html (v5).

Nothing here creates, starts, stops or archives a Strategy on the fund, and nothing runs a test: Run is shown
disabled while research is on hold, and its route refuses. Archiving a built Strategy only writes its archive row
and the decision log (the Data Architect's definition_archive table, Store.archive_definition); the rule file, its
hash, its tries and its test history never change."""

from __future__ import annotations

import importlib
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from sleeve_fund.dashboard import strategy_words as words
from sleeve_fund.research.candidates import BUDGET_PER_IDEA, configurations, idea_hash
from sleeve_fund.store import (
    ARCHIVE_REASONS,  # the pick-list (mock-up v5, section 5), shared with the table's CHECK
)
from sleeve_fund.strategies.definitions import (
    check_definition,
    definition_hash,
    load_definition_file,
    timeframe_minutes,
)

log = logging.getLogger(__name__)
UK = ZoneInfo("Europe/London")
DEFINITIONS = Path(__file__).resolve().parents[2] / "configs" / "definitions"
# Research is on hold until the PM's go (Researcher hold, 14 Oct): Run shows disabled, and its route refuses.
RESEARCH_ON_HOLD = True
ON_HOLD = "Research is on hold until your go"
# Rule files written as examples of the builder before the QR set (spec: the list's third group).
EXAMPLES = ("rsi-bands", "rsi-pullback")
# The hand-coded Models Development listed (spec: the list's second group). The timed test and the rule builder's
# own model are not Strategies to research.
MODELS = ("trend_filter", "rsi_pullback", "rsi_bands", "rsi_cross", "dip_buy", "donchian", "buy_and_hold", "ping_pong")
# A rule file's family, as the mock-up groups them; a file not here shows its idea's word from the file name.
FAMILIES = {"qr01-": "Trend", "qr01b-": "Trend", "qr01c-": "Breakout", "qr02-": "Dip", "qr03-": "Squeeze",
            "qr10-": "Exhaustion", "qr11-": "Exhaustion", "qr12-": "Breakout", "qr13-": "Level continuation",
            "rsi-": "Mean reversion"}
# Status chips: key -> (words, chip tone).
STATUS = {"untested": ("Built, not tested", ""), "tested": ("Tested", "paused"), "passed": ("Passed G1", "running"),
          "killed": ("Killed", "halted"), "on_paper": ("On paper", "running"), "refused": ("Refused", "halted")}
TABS = ("overview", "test", "history", "runs")
_DECISION = re.compile(r"Decision candle:\s*(\d+[mhd])", re.IGNORECASE)


@dataclass
class Entry:
    """One row of the list and the page behind it."""

    key: str  # the definition's name (its file id, e.g. qr02-dip-long), or "model:<name>" for a hand-coded Model
    group: str  # "built", "model" or "example"
    title: str
    file_id: str
    family: str
    sides: tuple = ()
    decision: str | None = None  # the decision candle the file is written for (4h), when it says
    reason: str = ""
    path: Path | None = None
    text: str = ""  # the file as written, for "Show the file"
    defn: dict | None = None
    checked: object = None
    refused: str | None = None
    hash: str | None = None
    idea: str | None = None
    version: int | None = None
    file_date: datetime | None = None
    secondaries: list = field(default_factory=list)
    rules: list = field(default_factory=list)
    indicators: list = field(default_factory=list)
    words_error: str | None = None
    status: str = "untested"
    tries: int = 0
    holdouts: list = field(default_factory=list)
    trials: list = field(default_factory=list)
    sheets: list = field(default_factory=list)
    runs_on: list = field(default_factory=list)
    archived: dict | None = None

    @property
    def name(self) -> str:
        """The shared name of a long and short pair ("Trend Cross (21/55, 4h)"), the part of the title before " · ";
        for a file with no title yet, its id without the side (qr02-dip)."""
        if " · " in self.title:
            return self.title.split(" · ")[0]
        stem, _, side = self.title.rpartition("-")
        return stem if side in ("long", "short") and stem else self.title

    @property
    def side_label(self) -> str:
        """What tells the sides of a pair apart: the title after " · " ("Long", or "buy the dip · Long")."""
        return self.title.split(" · ", 1)[1] if " · " in self.title else " & ".join(s.capitalize() for s in self.sides)

    @property
    def chip(self) -> tuple[str, str]:
        return STATUS[self.status]

    @property
    def model(self) -> str | None:
        return self.key.split(":", 1)[1] if self.group == "model" else None


def _family(file_id: str) -> str:
    return next((v for k, v in sorted(FAMILIES.items(), key=lambda kv: -len(kv[0])) if file_id.startswith(k)), "Rules")


def _comments(text: str) -> list[str]:
    """The file's leading comment lines, '# ' stripped."""
    out = []
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        out.append(line.lstrip("#").strip())
    return out


def secondaries(text: str) -> list[str]:
    """The pre-registered secondary settings the file's header names ("Pre-registered secondaries (...): a; b; c"),
    each one a try against the idea's budget."""
    head = " ".join(_comments(text))
    m = re.search(r"Pre-registered secondaries[^:]*:\s*(.*?)(?:\.\s|$)", head)
    if not m:
        return []
    return [s.strip().rstrip(".") for s in m.group(1).split(";") if s.strip()]


def load_file(path: Path) -> Entry:
    """One rule file as an entry; a file that fails its check is kept, refused, with the reason."""
    text = path.read_text(encoding="utf-8")
    try:
        defn = load_definition_file(path)
    except Exception as exc:  # noqa: BLE001 - a file that doesn't parse is shown refused, not a page error
        file_id = path.stem.replace("_", "-")
        return Entry(key=file_id, group="built", title=file_id, file_id=file_id, family=_family(file_id), path=path,
                     text=text, refused=f"the file can't be read: {exc}", status="refused")
    file_id = str(defn.get("name") or path.stem.replace("_", "-"))
    head = " ".join(_comments(text))
    m = _DECISION.search(head)
    decision = m.group(1).lower() if m else None
    e = Entry(key=file_id, group="example" if file_id in EXAMPLES else "built",
              title=str(defn.get("title") or file_id), file_id=file_id, family=_family(file_id),
              sides=tuple(s for s in ("long", "short") if s in defn), decision=decision,
              reason=str(defn.get("reason") or "").strip(), path=path, text=text, defn=defn,
              version=defn.get("version"), secondaries=secondaries(text),
              file_date=datetime.fromtimestamp(path.stat().st_mtime, UTC))
    try:
        e.checked = check_definition(defn)
    except ValueError as exc:
        e.refused, e.status = str(exc), "refused"
        return e
    e.hash, e.idea = definition_hash(defn), idea_hash(defn)
    try:
        minutes = timeframe_minutes(decision) if decision else None
        e.rules = words.rules(e.checked, minutes)
        e.indicators = words.indicators(e.checked, decision)
    except (ValueError, KeyError) as exc:  # the page shows why, in place of the sentences, and the file below
        e.words_error = str(exc)
    return e


def load_model(name: str) -> Entry:
    from sleeve_fund.dashboard import development as dev
    from sleeve_fund.research.trials import legacy_idea_hash

    spec = importlib.import_module(f"sleeve_fund.strategies.{name}").SPEC
    return Entry(key=f"model:{name}", group="model", title=dev.label(name), file_id=name.replace("_", "-"),
                 family=str(spec.family).replace("_", " ").replace("-", " ").capitalize(), reason=spec.idea,
                 idea=legacy_idea_hash(name))


def catalogue(directory: Path | None = None) -> list[Entry]:
    """Every built Strategy (rule files, read now; nothing is written) and every hand-coded Model."""
    directory = directory or DEFINITIONS
    out = [load_file(p) for p in sorted(directory.glob("*.toml"))]
    for name in MODELS:
        try:
            out.append(load_model(name))
        except Exception as exc:  # noqa: BLE001 - a model that can't be read is left out and logged
            log.warning(f"couldn't read the {name} model for the Strategy list: {exc!r}")
    return out


def _sheet_matches(e: Entry, s: dict) -> bool:
    return s.get("strategy") in ({e.model} if e.model else {e.file_id, e.key})


def annotate(entries: list[Entry], store, sheets: list[dict]) -> list[Entry]:
    """Each entry's status, tries, holdout state, test history and where it runs, read from the store. Read only."""
    from sleeve_fund.research.trials import TrialsRegister

    register = TrialsRegister(store)
    try:
        archived = {r["file_id"]: r for r in store.archived_definitions()}
    except AttributeError:  # the archive table isn't in this build yet: nothing is archived
        archived = {}
    put_away = set(store.archived())
    live = [s for s in store.sleeves() if s.name not in put_away and s.lifecycle != "retired"]
    for e in entries:
        e.archived = archived.get(e.key)
        if e.refused:
            continue
        if e.idea:
            e.tries = len(configurations(register, e.idea))
            e.trials = [t for t in register._counted(e.idea)]
            e.holdouts = store.holdout_locks(e.idea)
        e.sheets = [s for s in sheets if _sheet_matches(e, s)]
        if e.model:
            e.runs_on = [s for s in live if s.strategy == e.model]
        else:
            e.runs_on = [s for s in live if isinstance(s.params.get("definition"), dict)
                         and _same_definition(s.params["definition"], e.hash)]
        verdicts = [s["g1"] for s in e.sheets if s.get("g1")]
        e.status = ("on_paper" if e.runs_on else "passed" if verdicts and verdicts[0] == "PASS"
                    else "killed" if verdicts and verdicts[0] == "FAIL" else "tested" if e.trials or e.sheets
                    else "untested")
    return entries


def _same_definition(defn: dict, h: str | None) -> bool:
    try:
        return h is not None and definition_hash(defn) == h
    except ValueError:
        return False


def groups(entries: list[Entry]) -> list[dict]:
    """The list's sections in order, each holding its pairs: built Strategies (a long and short side under their
    shared name), then the hand-coded Models, then the earlier examples."""
    out = []
    for group, label in (("built", "Built Strategies"), ("model", "Models (hand-coded)"), ("example", "Earlier examples")):
        rows = [e for e in entries if e.group == group]
        if not rows:
            continue
        pairs: dict[str, list[Entry]] = {}
        for e in rows:
            pairs.setdefault(e.name if group == "built" else e.key, []).append(e)
        out.append({"key": group, "label": label, "pairs": [{"name": k, "rows": v} for k, v in pairs.items()]})
    return out


def other_side(entries: list[Entry], e: Entry) -> Entry | None:
    """The other side of a long and short pair: the same file id but for its last word."""
    if e.group == "model":
        return None
    stem, _, side = e.file_id.rpartition("-")
    want = {"long": "short", "short": "long"}.get(side)
    return next((x for x in entries if want and x.file_id == f"{stem}-{want}"), None)


def stop_note(e: Entry, side: dict) -> str:
    """Where the stop sits against the leverage caps, as far as the file alone can say."""
    if side["stopless"]:
        return "No stop: held to 1×."
    return ("The order rules keep the stop within half the distance to liquidation, at 1×, 2× or 3×. This stop is "
            "set from the market at entry, so where it sits against the caps is known only then.")


def uk(t) -> str:
    if t is None:
        return ""
    t = t if t.tzinfo else t.replace(tzinfo=UTC)
    return t.astimezone(UK).strftime("%d %b %Y %H:%M UK")


def register(app: FastAPI, *, page, st, tearsheets, stored_history, venue_choices, research_venue) -> None:
    """The Strategy area's routes. page/st: app.py's; the rest are app.py's module helpers, passed so this module
    doesn't import app.py."""
    from sleeve_fund.dashboard import development as dev
    from sleeve_fund.dashboard.app import require_pm, same_origin

    def sheets() -> list[dict]:
        root = tearsheets()
        return [dev.read_sheet(p) for p in sorted(root.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)]

    def everything():
        found = sheets()
        return annotate(catalogue(), st(), found), found

    def coverage() -> list[dict]:
        rows = []
        for v in venue_choices():
            profile = research_venue(v["key"])
            for h in stored_history(st(), profile):
                rows.append(dict(h, venue_key=v["key"], market=dev.venue_label(profile.name),
                                 chip=dev.history_chip(h)))
        return rows

    def area(request: Request, view: str, **ctx) -> HTMLResponse:
        entries = ctx.pop("entries", None)
        if entries is None:
            entries, _ = everything()
        shown = [e for e in entries if not e.archived]
        return page(request, "strategy_area.html", view=view, entries=entries, shown=shown,
                    counts={"strategies": len(shown), "archived": len(entries) - len(shown),
                            "results": ctx.pop("n_results", None)},
                    on_hold=RESEARCH_ON_HOLD, on_hold_words=ON_HOLD, statuses=STATUS, budget=BUDGET_PER_IDEA,
                    reasons=ARCHIVE_REASONS, uk=uk, **ctx)

    @app.get("/strategy", response_class=HTMLResponse)
    def strategies(request: Request, s: str = "", tab: str = "overview", _: str = Depends(require_pm)):
        entries, found = everything()
        shown = [e for e in entries if not e.archived]
        chosen = next((e for e in entries if e.key == s), None) if s else (shown[0] if shown else None)
        if s and chosen is None:
            raise HTTPException(404, "no such Strategy")
        tab = tab if tab in TABS else "overview"
        if chosen is not None and chosen.archived and tab in ("test", "runs"):
            tab = "overview"  # an archived Strategy opens read-only: Overview and Test history
        basket = coverage() if tab == "test" else []
        return area(request, "strategies", entries=entries, chosen=chosen, tab=tab, groups=groups(shown),
                    other=other_side(entries, chosen) if chosen else None, stop_note=stop_note, basket=basket,
                    n_results=len(found), notice=request.query_params.get("done", ""))

    @app.get("/strategy/results", response_class=HTMLResponse)
    def results(request: Request, _: str = Depends(require_pm)):
        entries, found = everything()
        by_sheet = {s["name"]: e for e in entries for s in e.sheets}
        rows = [dict(s, entry=by_sheet.get(s["name"]), breakeven_text=dev.breakeven_text(s)) for s in found]
        return area(request, "results", entries=entries, rows=rows, n_results=len(found))

    @app.get("/strategy/coverage", response_class=HTMLResponse)
    def data_coverage(request: Request, _: str = Depends(require_pm)):
        entries, found = everything()
        rows = coverage()
        for r in rows:
            r["used_by"] = [e for e in entries if any(s.instrument.upper() == r["pair"] for s in e.runs_on)
                            or any(r["pair"].replace("/", "").lower() in str(t.get("dataset", "")).replace("/", "")
                                   for t in e.trials)]
        return area(request, "coverage", entries=entries, rows=rows, n_results=len(found))

    @app.get("/strategy/archived", response_class=HTMLResponse)
    def archived(request: Request, _: str = Depends(require_pm)):
        entries, found = everything()
        return area(request, "archived", entries=entries, rows=[e for e in entries if e.archived],
                    n_results=len(found), notice=request.query_params.get("done", ""))

    def _entry(key: str) -> tuple[Entry, list[Entry]]:
        entries, _ = everything()
        e = next((x for x in entries if x.key == key), None)
        if e is None:
            raise HTTPException(404, "no such Strategy")
        return e, entries

    def _store_or_refuse():
        store = app.state.store
        if not hasattr(store, "archive_definition"):
            raise HTTPException(409, "Archiving isn't available in this build yet")
        return store

    @app.post("/strategy/archive")
    def archive(key: str = Form(...), reason: str = Form(""), note: str = Form(""), other: str = Form(""),
                actor: str = Depends(require_pm), _o: None = Depends(same_origin)):
        e, entries = _entry(key)
        if reason not in ARCHIVE_REASONS:
            raise HTTPException(400, "pick why it is archived")
        if e.archived:
            raise HTTPException(409, "it is already archived")
        if e.runs_on:
            raise HTTPException(409, "it has a copy on the fund: stop and archive it on the fund first")
        store = _store_or_refuse()
        todo = [e]
        twin = other_side(entries, e)
        if other == "on" and twin is not None and not twin.archived and not twin.runs_on:
            todo.append(twin)
        for x in todo:
            said = note.strip() if x is e else f"archived with its other side, {e.file_id}" + (
                f": {note.strip()}" if note.strip() else "")
            store.archive_definition(x.key, definition_hash=x.hash, status=x.status, reason=reason,
                                     note=said or None, actor=actor)
        return RedirectResponse(f"/strategy/archived?done={'+'.join(x.key for x in todo)}", status_code=303)

    @app.post("/strategy/restore")
    def restore(key: str = Form(...), reason: str = Form(""), actor: str = Depends(require_pm),
                _o: None = Depends(same_origin)):
        e, _ = _entry(key)
        if not reason.strip():
            raise HTTPException(400, "a reason is required")
        if not e.archived:
            raise HTTPException(409, "it isn't archived")
        _store_or_refuse().restore_definition(e.key, reason=reason.strip(), actor=actor)
        return RedirectResponse(f"/strategy?s={e.key}&done=restored", status_code=303)

    @app.post("/strategy/test")
    def run_test(_: str = Depends(require_pm), _o: None = Depends(same_origin)):
        """Running a test from the Strategy page: refused while research is on hold (spec Done-when). Whether it runs
        after the PM's go is for the HoE to wire then."""
        raise HTTPException(409, ON_HOLD)
