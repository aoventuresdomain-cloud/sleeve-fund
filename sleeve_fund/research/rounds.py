"""M1 items 4-6 (PM 30 Oct, next strategy round): a study list as the round's registered tries, each family's try budget,
and the round's count so far, read off the trials register rather than counted by hand.

A study list (configs/studies/) becomes a round with a block of common settings like

    - **Round:** next-1
    - **Budget per family:** 20 tries
    - **Family A:** rows 1-8
    - **Family B:** rows 9-16, 18

Its rows are the round's registered tries, fixed before anything runs, and the list's SHA-256 names exactly that list:
any edit to it (a row added, a setting changed, a date moved) is a new hash on every sheet after it. Every row is in
exactly one family. A try is one evaluation in the register (one study of one definition on one asset, or one basket),
counted under its definition's idea (study.idea_of), so a family's tries are every counted evaluation of its rows'
ideas, ever, earlier rounds' included: never fewer than were really tried. A row is refused before it runs once its
family has used its budget."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

ROUND_LINE = re.compile(r"^-\s*\*\*Round:\*\*\s*([A-Za-z0-9][A-Za-z0-9 ._-]{0,63}?)\s*$", re.M)
BUDGET_LINE = re.compile(r"^-\s*\*\*Budget per family:\*\*\s*(\d{1,4})\s*tries\s*$", re.M | re.I)
FAMILY_LINE = re.compile(r"^-\s*\*\*Family ([A-Za-z0-9]{1,16}):\*\*\s*rows\s+(.+?)\s*$", re.M)
ANY_LINE = re.compile(r"^-\s*\*\*(Round|Budget per family|Family [^*]*):\*\*", re.M | re.I)


@dataclass(frozen=True)
class Round:
    name: str
    sha256: str  # of the whole list's text
    budget: int  # tries per family
    families: dict  # family -> its rows, sorted
    rows: tuple  # every row of the list, in its order

    def family_of(self, row: int) -> str:
        return next(f for f, rows in self.families.items() if row in rows)

    def place(self, row: int) -> tuple[int, int]:
        """The row's place among the round's registered tries, and how many there are."""
        return self.rows.index(row) + 1, len(self.rows)


def _rows_in(spec: str, where: str) -> tuple[int, ...]:
    out = set()
    for part in spec.split(","):
        m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", part)
        if m is None:
            raise ValueError(f"{where}: rows are numbers and ranges like 1-8, 11, not {part.strip()!r}")
        a, b = int(m[1]), int(m[2] or m[1])
        if b < a:
            raise ValueError(f"{where}: the range {a}-{b} runs backwards")
        out.update(range(a, b + 1))
    return tuple(sorted(out))


def round_of(text: str, rows: list[int]) -> Round | None:
    """The list's round, or None for a list that names none. Refused in words: a round without its budget or
    families, a row in no family or two, a family naming a row the list doesn't have."""
    if not ANY_LINE.search(text):
        return None
    name, budget, fams = ROUND_LINE.findall(text), BUDGET_LINE.findall(text), FAMILY_LINE.findall(text)
    if len(name) != 1:
        raise ValueError("a round names itself once: - **Round:** <name>")
    if len(budget) != 1 or int(budget[0]) < 1:
        raise ValueError("a round sets its budget once: - **Budget per family:** <N> tries")
    if not fams:
        raise ValueError("a round puts its rows in families: - **Family A:** rows 1-8")
    families: dict = {}
    for f, spec in fams:
        if f in families:
            raise ValueError(f"family {f} is listed twice")
        families[f] = _rows_in(spec, f"family {f}")
    listed = [r for rs in families.values() for r in rs]
    unknown = sorted(set(listed) - set(rows))
    if unknown:
        raise ValueError(f"the families name rows the list doesn't have: {', '.join(map(str, unknown))}")
    twice = sorted({r for r in listed if listed.count(r) > 1})
    if twice:
        raise ValueError(f"a row is in two families: {', '.join(map(str, twice))}")
    homeless = [r for r in rows if r not in listed]
    if homeless:
        raise ValueError(f"every row of a round is in a family; not {', '.join(map(str, homeless))}")
    return Round(name[0], hashlib.sha256(text.encode("utf-8")).hexdigest(), int(budget[0]), families, tuple(rows))


def used(register, ideas) -> int:
    """The counted evaluations of these ideas in the trials register: a family's tries so far."""
    return sum(register.counts(i)["evaluations"] for i in set(ideas) if i is not None)


def try_line(r: Round, row: int, family_used: int) -> str:
    """The sheet's line for one of the round's registered tries."""
    k, n = r.place(row)
    f = r.family_of(row)
    return (f"Round {r.name} (study list SHA-256 {r.sha256[:12]}): registered try {k} of {n}, family {f}; the family "
            f"had used {family_used} of its {r.budget} tries before this one (trials register).")


def summary(r: Round, used_by_family: dict) -> str:
    """The round in one line: each family's tries used of its budget, from the trials register."""
    parts = [f"family {f} {used_by_family.get(f, 0)} of {r.budget} tries" for f in r.families]
    return f"Round {r.name} (study list SHA-256 {r.sha256[:12]}), {len(r.rows)} registered tries: " + ", ".join(parts)
