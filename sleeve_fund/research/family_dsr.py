"""FAMILY-DSR (Advisor, 31 Oct 2026, on H0): a basket try's deflated Sharpe judged across its whole idea family, not
only its own idea's rows.

A round file pins each family's members and which of their register rows the Sharpe spread uses:

    - **Family members, B:** 66fb76f17c2a5ec3, 0e13cf32e1775235, ...
    - **Valid-fill rows, B:** 5f5556da26db40ba/0123456789abcdef (stand-ins 12%) (sheet STEM), ...
      or .../<id> (hand-checked by Researcher: fill times off the 15-minute grid), when its sheet isn't kept
    - **Excluded rows, B:** 5f5556da26db40ba/fedcba9876543210 (coarse re-run), ...

Members are idea hashes (structural ideas, or a legacy ledger idea's). A valid-fill row is "definition hash / trials
id" for a run whose sheet shows 1-minute execution fired (Advisor's check 1), chosen by hand from the sheets and pinned
here, so the rule is written down and never inferred. Each pin carries its evidence: its sheet's stem, when the sheet
is in the tear-sheet folder (the step runs the try's off-grid test on it), else who hand-checked it and on what; a pin
with neither refuses, and the report says which each used (Advisor 31 Oct). The stand-in share is noted beside it: funding stand-ins don't
exclude a row from a family whose signal isn't funding (Advisor, 31 Oct). An excluded row carries one of three reasons:
"coarse re-run", "corrected by row <id>" (the correcting row is the one included) or "Sharpe unverified".

- **N:** the larger of the stated count (the round's "Prior tries" line plus the round rows' tries in the register)
  and the register's (every distinct variant under the members), flagged when they differ.
- **Spread:** one Sharpe per variant, from its included row. The register can't tell a correction from a re-run (a
  correction copies every field but its id, time, Sharpe and error), so every row of a pinned row's variant that is
  newer than it must itself be pinned, included or excluded with its reason, or the step refuses (Advisor and CR,
  31 Oct): a correction can't slip past as a re-run. A row pinned as included while another is marked as its
  correction refuses too. The try being judged joins without a pin when its own sheet shows a fill off the 15-minute
  grid (1-minute execution fired), and refuses otherwise. With fewer than 3 included rows, the spread is floored
  at 0.5 a year. The report names the bound that set the spread, and every member variant left out, with why.

    python -m sleeve_fund.research.family_dsr --list configs/studies/LIST.md --family B --sheet STEM

reads the sheet's pooled daily returns (STEM-pooled.csv, the test days) and prints the report: ids and Sharpes, so the
workflow encrypts it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sleeve_fund.research import rounds, trials
from sleeve_fund.research.metrics import PERIODS_PER_YEAR, deflated_sharpe_probability

SPREAD_FLOOR = 0.5  # a year, below MIN_ROWS valid-fill rows (B's clearance, 30 Oct)
MIN_ROWS = 3
HEX16 = r"[0-9a-f]{16}"
MEMBERS_LINE = re.compile(r"^-\s*\*\*Family members, ([A-Za-z0-9]{1,16}):\*\*\s*(.+?)\s*$", re.MULTILINE)
VALID_LINE = re.compile(r"^-\s*\*\*Valid-fill rows, ([A-Za-z0-9]{1,16}):\*\*\s*(.+?)\s*$", re.MULTILINE)
EXCLUDED_LINE = re.compile(r"^-\s*\*\*Excluded rows, ([A-Za-z0-9]{1,16}):\*\*\s*(.+?)\s*$", re.MULTILINE)
ENTRY = re.compile(rf"({HEX16})/({HEX16})(?:\s*\(stand-ins (\d{{1,3}})%\))?"
                   r"(?:\s*\((?:sheet ([A-Za-z0-9_][A-Za-z0-9_.-]*)|hand-checked by ([^(),:]+): ([^(),]+))\))?")
XENTRY = re.compile(rf"({HEX16})/({HEX16})\s*\((coarse re-run|corrected by row ({HEX16})|Sharpe unverified)\)")
UNVERIFIED = "Sharpe unverified"
MIN_DAYS = 30  # deflated_sharpe_probability's own minimum; below it the step refuses rather than print nan
STEM = re.compile(r"(?P<label>[A-Za-z0-9_.-]+?)_(?P<dataset>[a-z0-9]+-basket-[a-z0-9-]+?)_\d{8}-\d{6}(?:-\d+)?")


@dataclass(frozen=True)
class Family:
    name: str
    members: tuple  # idea hashes
    valid: tuple  # (definition hash, trials id, stand-in share % or None, how its 1-minute fills were checked)
    excluded: tuple = ()  # (definition hash, trials id, reason, the correcting row's id or None)


def families_of(text: str) -> dict:
    """Each family's pinned members and valid-fill rows, refused in words when a line is malformed."""
    out: dict = {}
    for f, spec in MEMBERS_LINE.findall(text):
        if f in out:
            raise ValueError(f"family {f}'s members are listed twice")
        members = [m.strip() for m in spec.split(",")]
        if not members or any(not re.fullmatch(HEX16, m) for m in members) or len(set(members)) != len(members):
            raise ValueError(f"family {f}'s members are distinct 16-character idea hashes, comma-separated")
        out[f] = Family(f, tuple(members), ())
    seen = set()
    for f, spec in VALID_LINE.findall(text):
        if f not in out:
            raise ValueError(f"valid-fill rows are pinned for family {f}, whose members aren't")
        if f in seen:
            raise ValueError(f"family {f}'s valid-fill rows are listed twice")
        seen.add(f)
        parts = [p.strip() for p in spec.split(",")]
        entries = [ENTRY.fullmatch(p) for p in parts]
        if any(e is None for e in entries):
            raise ValueError(f"family {f}'s valid-fill rows read <definition hash>/<trials id> (stand-ins N%)")
        valid = tuple((e[1], e[2], None if e[3] is None else int(e[3]),
                       ("sheet", e[4]) if e[4] else ("hand", e[5].strip(), e[6].strip()) if e[5] else None)
                      for e in entries)
        if len({v[1] for v in valid}) != len(valid):
            raise ValueError(f"family {f} pins a trials id twice")
        out[f] = Family(f, out[f].members, valid)
    seen = set()
    for f, spec in EXCLUDED_LINE.findall(text):
        if f not in out:
            raise ValueError(f"excluded rows are pinned for family {f}, whose members aren't")
        if f in seen:
            raise ValueError(f"family {f}'s excluded rows are listed twice")
        seen.add(f)
        entries = [XENTRY.fullmatch(p.strip()) for p in spec.split(",")]
        if any(e is None for e in entries):
            raise ValueError(f"family {f}'s excluded rows read <definition hash>/<trials id> (coarse re-run), "
                             "(corrected by row <trials id>) or (Sharpe unverified)")
        excluded = tuple((e[1], e[2], "corrected by row" if e[4] else e[3], e[4]) for e in entries)
        ids = [v[1] for v in out[f].valid] + [x[1] for x in excluded]
        if len(set(ids)) != len(ids):
            for x in excluded:  # the named case first: an original kept in while its correction is marked
                if x[3] and x[1] in {v[1] for v in out[f].valid}:
                    raise ValueError(f"refused: row {x[1]} is pinned as included, but row {x[3]} is marked as its "
                                     f"correction: include {x[3]} and exclude {x[1]}")
            raise ValueError(f"family {f} pins a trials id twice")
        out[f] = Family(f, out[f].members, out[f].valid, excluded)
    if len(re.findall(r"^-\s*\*\*(Family members|Valid-fill rows|Excluded rows)", text,
                      re.MULTILINE | re.IGNORECASE)) != (len(MEMBERS_LINE.findall(text)) + len(VALID_LINE.findall(text))
                                                         + len(EXCLUDED_LINE.findall(text))):
        raise ValueError("a family line reads - **Family members, B:** <hashes>, - **Valid-fill rows, B:** <rows> "
                         "or - **Excluded rows, B:** <rows>")
    return out


@dataclass(frozen=True)
class Result:
    family: str
    n_stated: int
    n_register: int
    spread_rows: tuple  # (trials id, definition hash, Sharpe, stand-in %, how its 1-minute fills were checked)
    floored: bool
    dsr: float
    binding: tuple = ("", math.nan)  # (which bound set the spread, its value a year): Advisor 31 Oct, shown
    left_out: tuple = ()  # (what: a variant's latest row or a member idea, why): never ran, not valid-fill, ...
    list_sha: str = ""  # the list the report read: the container's deployed copy can lag main (CR F3)

    @property
    def n(self) -> int:
        return max(self.n_stated, self.n_register)


def _params_idea(t: dict) -> str | None:
    """A basket row's idea: its definition's (its settings' params), as rounds.used counts it."""
    from sleeve_fund.research.study import idea_of

    if not str(t["definition_name"]).startswith("basket:"):
        return None
    try:  # the register keeps settings as JSON text
        params = json.loads(t["settings"] or "{}").get("params")
    except (TypeError, ValueError, AttributeError):
        params = None
    return idea_of(params if isinstance(params, dict) else None, None)


def _member(t: dict, members) -> str | None:
    """The member idea a counted row is a try of (rounds.used's rule), or None."""
    if t["idea_hash"] in members:
        return t["idea_hash"]
    idea = _params_idea(t)
    return idea if idea in members else None


def spread_rows(register: trials.TrialsRegister, fam: Family, try_id: str | None = None,
                sheet_ok=None) -> tuple:
    """(the included rows as the register holds them, one per variant: (id, definition, Sharpe, stand-ins %, check);
    the member variants and ideas left out, with why). try_id: the judged try's own row, included without a pin.
    sheet_ok(stem): whether a pinned row's own sheet, in the tear-sheet folder, shows 1-minute fills (main's
    one_minute_fills); a pin names that sheet, or says who hand-checked it and on what (Advisor 31 Oct)."""
    counted = register._counted()  # oldest first
    pos = {t["id"]: k for k, t in enumerate(counted)}
    by_id = {t["id"]: t for t in counted}
    included = list(fam.valid)
    excluded = {x[1]: x for x in fam.excluded}
    if try_id is not None:
        if try_id in excluded:
            raise ValueError(f"refused: the try's own row {try_id} is pinned as excluded")
        if try_id not in {v[1] for v in included}:
            if try_id not in by_id:
                raise ValueError(f"refused: no counted row {try_id} for the try")
            included.append((by_id[try_id]["definition_hash"], try_id, None, ("own sheet",)))
    pinned = [(d, i) for d, i, _, _ in included] + [(x[0], x[1]) for x in fam.excluded]
    for definition, row_id in pinned:
        t = by_id.get(row_id)
        if t is None:
            raise ValueError(f"refused: no counted row {row_id} (a correction not yet applied?)")
        if t["definition_hash"] != definition:
            raise ValueError(f"refused: row {row_id} is definition {t['definition_hash']}, not {definition}")
        if _member(t, fam.members) is None:
            raise ValueError(f"refused: row {row_id} is a try of none of family {fam.name}'s members")
    for definition, row_id, _, how in included:
        _check_fills(by_id[row_id], how, sheet_ok)
    in_ids = {i for _, i, _, _ in included}
    for _, row_id, _, by in fam.excluded:
        if by is None:
            continue
        if by not in in_ids:
            raise ValueError(f"refused: row {row_id} is corrected by row {by}, which isn't included")
        if trials._variant(by_id[by]) != trials._variant(by_id[row_id]) or pos[by] < pos[row_id]:
            raise ValueError(f"refused: row {by} is not a newer row of row {row_id}'s variant, so can't correct it")
    pinned_ids = in_ids | set(excluded)
    for _, row_id in pinned:  # a newer row of a pinned row's variant may be its correction: it must be pinned too
        v = trials._variant(by_id[row_id])
        for t in counted[pos[row_id] + 1:]:
            if trials._variant(t) == v and t["id"] not in pinned_ids:
                raise ValueError(f"refused: row {t['id']} is newer than pinned row {row_id}, of its variant, and not "
                                 "pinned: include it or exclude it (coarse re-run, corrected by row <id>, Sharpe "
                                 "unverified)")
    out, variants = [], set()
    for definition, row_id, standins, how in included:
        t = by_id[row_id]
        if trials._variant(t) in variants:
            raise ValueError(f"refused: row {row_id}'s variant is pinned twice")
        variants.add(trials._variant(t))
        if t["sharpe"] is None or not math.isfinite(t["sharpe"]):
            raise ValueError(f"refused: row {row_id} has no finite Sharpe")
        out.append((row_id, definition, float(t["sharpe"]), standins, how))
    return tuple(out), _left_out(counted, fam, variants, excluded)


def _check_fills(t: dict, how, sheet_ok) -> None:
    """A pin's 1-minute evidence: its own sheet, run through the same off-grid test as the try's, when the sheet is
    kept; else a named hand check. Neither refuses."""
    if how is None:
        raise ValueError(f"refused: pin {t['id']} names neither its sheet nor who hand-checked its 1-minute fills")
    if how[0] != "sheet":
        return
    m = STEM.fullmatch(how[1])
    if (m is None or t["definition_name"] != f"basket:{m['label']}" or t["dataset"] != m["dataset"]):
        raise ValueError(f"refused: sheet {how[1]} is not row {t['id']}'s basket sheet (a single study's row is "
                         "hand-checked)")
    if sheet_ok is None or not sheet_ok(how[1]):
        raise ValueError(f"refused: row {t['id']}'s sheet {how[1]} shows no fill off the {COARSEST_FINE}-minute grid")


def _left_out(counted: list, fam: Family, included: set, excluded: dict) -> tuple:
    """Every member variant with no included row, and every member idea with no row at all (Advisor 31 Oct, (a))."""
    rows: dict = {}
    ran = set()
    for t in counted:
        idea = _member(t, fam.members)
        if idea is not None:
            ran.add(idea)
            rows.setdefault(trials._variant(t), []).append(t)
    out = [(f"idea {m}", "never ran") for m in fam.members if m not in ran]
    for v, ts in rows.items():
        if v in included:
            continue
        unverified = all(t["sharpe"] is None or not math.isfinite(t["sharpe"]) for t in ts) or any(
            excluded.get(t["id"], (None, None, None))[2] == UNVERIFIED for t in ts)
        out.append((f"row {ts[-1]['id']} ({v[0]})", UNVERIFIED if unverified else "not valid-fill"))
    return tuple(out)


def compute(register: trials.TrialsRegister, text: str, family: str, returns: pd.Series, row_ideas: set,
            try_id: str | None = None, sheet_ok=None) -> Result:
    """H0's (or any try's) deflated Sharpe across its family. row_ideas: the ideas the round's rows of this family
    count under (rescore derives them as it runs), for the stated count. try_id: the try's own register row, which
    joins the spread unpinned (main passes it only once its sheet shows 1-minute execution)."""
    fam = families_of(text).get(family)
    if fam is None:
        raise ValueError(f"refused: the list pins no members for family {family}")
    if not fam.valid and try_id is None:
        raise ValueError(f"refused: the list pins no valid-fill rows for family {family}")
    if len(returns.dropna()) < MIN_DAYS:
        raise ValueError(f"refused: {len(returns.dropna())} return days, under the {MIN_DAYS} a deflated Sharpe needs")
    rnd = rounds.round_of(text, [n for n in _rows(text)])
    prior = rnd.prior.get(family, 0) if rnd is not None else 0
    n_stated = prior + rounds.used(register, row_ideas)
    n_register = rounds.used(register, set(fam.members))
    rows, left_out = spread_rows(register, fam, try_id, sheet_ok)
    sharpes = [s for _, _, s, _, _ in rows]
    floored = len(sharpes) < MIN_ROWS
    dsr = deflated_sharpe_probability(returns, max(n_stated, n_register, 1), sharpes,
                                      spread_floor=SPREAD_FLOOR if floored else None)
    if not math.isfinite(dsr):
        raise ValueError("refused: the deflated Sharpe is undefined on these returns (flat?)")
    return Result(family, n_stated, n_register, rows, floored, dsr, binding(returns, sharpes, floored), left_out,
                  hashlib.sha256(text.encode("utf-8")).hexdigest())


def binding(returns: pd.Series, sharpes: list, floored: bool) -> tuple:
    """Which of the spread's bounds is the largest, and its value a year, as deflated_sharpe_probability takes it:
    the valid-fill rows' measured spread (2 rows or more), the no-skill error 1/sqrt(T-1), and the 0.5 floor (below
    3 rows). The Advisor (31 Oct): the reader sees why the hurdle is where it is."""
    year = math.sqrt(PERIODS_PER_YEAR)
    bounds = [("the no-skill error 1/sqrt(T-1)", year / math.sqrt(max(len(returns.dropna()) - 1, 1)))]
    if len(sharpes) > 1:
        bounds.append(("the valid-fill rows' spread", float(np.std(sharpes, ddof=1))))
    if floored:
        bounds.append((f"the {SPREAD_FLOOR} floor", SPREAD_FLOOR))
    return max(bounds, key=lambda b: b[1])


def _rows(text: str) -> list:
    from sleeve_fund.research.rescore import rows_of

    return [n for n, _, _ in rows_of(text)]


def pooled_returns(path: Path) -> pd.Series:
    """The sheet's pooled daily returns over its test days (basket_run's pooled.csv)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "pooled_return" not in rows[0]:
        raise ValueError(f"refused: {path.name} has no pooled_return column")
    return pd.Series([float(r["pooled_return"]) for r in rows], index=pd.to_datetime([r["date"] for r in rows]))


def try_row(register: trials.TrialsRegister, stem: str) -> str:
    """The register row of the basket try a sheet shows: its stem is <label>_<dataset>_<time> (basket_run), and the
    try is the counted row basket_run recorded for that label on that dataset (the latest, if more than one)."""
    m = STEM.fullmatch(stem)
    if m is None:
        raise ValueError(f"refused: {stem} is not a basket sheet's stem (<label>_<venue>-basket-..._<time>)")
    rows = [t for t in register._counted() if t["definition_name"] == f"basket:{m['label']}"
            and t["dataset"] == m["dataset"]]
    if not rows:
        raise ValueError(f"refused: no counted row for the try basket:{m['label']} on {m['dataset']}")
    return rows[-1]["id"]


COARSEST_FINE = 15  # minutes: a fill off this grid can only come from 1-minute bars (Advisor 31 Oct)


def one_minute_fills(trades: Path, stem: str) -> bool:
    """Whether the try's own sheet shows 1-minute execution fired: an intraday basket (its dataset names its candle
    length) with at least one fill stamped off the 15-minute grid. A fill merely inside a decision candle isn't
    enough: a 4-hour rule filled on 15-minute bars has those too, and that is the coarse case (Advisor 31 Oct)."""
    m = STEM.fullmatch(stem)
    if m is None or re.search(r"-\d+m$", m["dataset"]) is None:  # a daily basket fills on hourly bars
        return False
    with open(trades, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for when in (row.get("entry_time"), row.get("exit_time")):
                if when:
                    ts = pd.Timestamp(when)
                    if ts.second or ts.microsecond or ts.minute % COARSEST_FINE:
                        return True
    return False


def _how(how) -> str:
    if how[0] == "own sheet":
        return "1-minute fills: the try's own sheet"
    if how[0] == "sheet":
        return f"1-minute fills: sheet {how[1]}"
    return f"1-minute fills: hand-checked by {how[1]} ({how[2]})"


def report(r: Result) -> list:
    lines = [f"list SHA-256 {r.list_sha}", f"family {r.family}: N {r.n} (stated {r.n_stated}, register {r.n_register})"
             + (" FLAG: the counts differ" if r.n_stated != r.n_register else "")]
    for row_id, definition, sharpe, standins, how in r.spread_rows:
        lines.append(f"  spread row {row_id} ({definition}): Sharpe {sharpe:.4f}"
                     + (f", funding stand-ins {standins}%" if standins is not None else "") + f", {_how(how)}")
    for what, why in r.left_out:
        lines.append(f"  left out: {what}: {why}")
    lines.append(f"spread: {len(r.spread_rows)} valid-fill rows" + (f", floored at {SPREAD_FLOOR} a year (fewer than "
                                                                     f"{MIN_ROWS})" if r.floored else ""))
    lines.append(f"spread bound: {r.binding[0]}, {r.binding[1]:.2f} a year")
    lines.append(f"deflated Sharpe probability {r.dsr:.4f}")
    return lines


def main(argv=None, store=None, sheets: Path | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sleeve_fund.research.family_dsr")
    ap.add_argument("--list", required=True, help="the round's study list, under configs/studies/")
    ap.add_argument("--family", required=True, help="the family, e.g. B")
    ap.add_argument("--sheet", required=True, help="the try's tear sheet stem, in TEARSHEET_DIR")
    a = ap.parse_args(argv)
    if (not re.fullmatch(r"[A-Za-z0-9]{1,16}", a.family) or not re.fullmatch(r"[A-Za-z0-9._-]+", a.sheet)
            or ".." in a.sheet or not re.fullmatch(r"configs/studies/[A-Za-z0-9._-]+\.md", a.list) or ".." in a.list):
        print("refused: --list is a configs/studies/ file, --family letters or digits, --sheet a plain file stem")
        return 2
    if store is None:  # the research CLI's journal (allowlisted research, as rescore's): never a default file
        from sleeve_fund.__main__ import _journal

        store = _journal()
        if store is None:
            print("refused: no DATABASE_URL, so no register to read")
            return 2
    from sleeve_fund.research import rescore
    from sleeve_fund.research.run import TEARSHEETS
    from sleeve_fund.research.study import idea_of
    from sleeve_fund.research.trials import legacy_idea_hash
    from sleeve_fund.venues import RESEARCH_VENUE

    try:
        text = Path(a.list).read_text(encoding="utf-8")
        pinned = rescore.parse_pinned(text)
        rnd = rounds.round_of(text, _rows(text))
        market = rescore.markets_of(text)  # a row's own market, as rescore runs it (CR F2)
        ideas = set()
        for n, name, minutes in rescore.rows_of(text):
            if rnd is not None and rnd.family_of(n) == a.family:
                req = rescore.request(name, minutes, RESEARCH_VENUE.lower(), "BTC/USDT", pinned,  # rescore's default
                                      market=market.get(n))
                ideas.add(idea_of(req.params, None) or legacy_idea_hash(req.strategy))
        register = trials.TrialsRegister(store)
        own = try_row(register, a.sheet)
        folder = sheets or TEARSHEETS
        returns = pooled_returns(folder / f"{a.sheet}-pooled.csv")
        if not one_minute_fills(folder / f"{a.sheet}-basket-trades.csv", a.sheet):
            raise ValueError(f"refused: the try's sheet shows no fill off the {COARSEST_FINE}-minute grid, so 1-minute "
                             f"execution isn't shown and its row {own} can't join the spread")
        result = compute(register, text, a.family, returns, ideas, try_id=own, sheet_ok=lambda stem: (
            folder / f"{stem}-basket-trades.csv").is_file() and one_minute_fills(folder / f"{stem}-basket-trades.csv",
                                                                                    stem))
    except (ValueError, OSError, KeyError) as exc:
        print(exc if str(exc).startswith("refused:") else f"refused: {exc}")
        return 1
    print("\n".join(report(result)))
    print(f"done: family {a.family}, sheet {a.sheet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
