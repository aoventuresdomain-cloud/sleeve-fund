"""R6: the one door an optimiser's output goes through (v2 P2, PE1 plan section 2, Independent Quant Advisor 7 Oct
14:05, 18:20 and 18:25 UK). No optimiser exists yet; this is built first so it can't be written around.

An optimiser may only edit a definition's strategy: its blocks, its long and short rules and its exits. Everything
that judges or protects a result is out of its reach: costs, the definition's version, name and reason, and, by
module, the engine, execution, sizing and risk caps, and the evaluation (guardrails and G1 thresholds, the cost
model, the trials register, the holdout lock, the random-entry benchmark). validate_candidate() enforces the first
part on every candidate; tests/test_candidates.py scans the optimiser package for the second.

What comes out is a Candidate: data only, with no path to paper. A Strategy starts on paper only when the PM adds
its definition.

A candidate is refused when:
- it changes anything outside blocks, long, short and exits, or the checker refuses it (before any backtest runs);
- it is an ablation variant, or its parent is one (Advisor 18:25 Q8, MUST: this is what makes leaving ablations out
  of N safe), read from the ablation rows' kind and their settings (parent_definition_hash, removed_path);
- its idea differs from its parent's and it isn't flagged as a new idea, so restructuring can't escape an idea's N;
- its idea has used its budget of BUDGET_PER_IDEA configurations, counted by the trials register, unless the PM's
  override is given (Advisor 18:20 Q4: sensitivity grid included, ablations not)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime

from sleeve_fund.strategies.definitions import (
    _canonical,
    check_definition,
    definition_hash,
)

EDITABLE = ("blocks", "long", "short", "exits")
BUDGET_PER_IDEA = 20  # configurations per idea (Advisor 18:20 Q4); the PM may change it
# Where an optimiser lives, once one exists; tests/test_candidates.py scans it.
OPTIMISER_PACKAGE = "sleeve_fund/research/optimiser"
# What an optimiser may never import or write to (Advisor 7 Oct 14:05): engine, execution, sizing and risk caps,
# and evaluation. Prefixes of module names.
PROTECTED_MODULES = (
    "sleeve_fund.strategies.base", "sleeve_fund.strategies.rules", "sleeve_fund.paper", "sleeve_fund.hub",
    "sleeve_fund.venues", "sleeve_fund.router", "sleeve_fund.supervisor", "sleeve_fund.liquidation",
    "sleeve_fund.margin", "sleeve_fund.portfolio", "sleeve_fund.risk", "sleeve_fund.open_risk",
    "sleeve_fund.gate_ledger", "sleeve_fund.fees", "sleeve_fund.spreads", "sleeve_fund.markets",
    "sleeve_fund.funding", "sleeve_fund.store", "sleeve_fund.research.guardrails", "sleeve_fund.research.study",
    "sleeve_fund.research.tearsheet", "sleeve_fund.research.trials", "sleeve_fund.research.holdout",
    "sleeve_fund.research.random_entry", "sleeve_fund.research.runner", "sleeve_fund.research.metrics",
    "sleeve_fund.research.ablation", "sleeve_fund.research.ledger",
)


def idea_hash(defn: dict) -> str:
    """The definition with its settings left out: its blocks' kinds and wiring, its rules' shape, comparisons and
    references, and which exits it uses, every number blanked. Two definitions that differ only in numbers are one
    idea; adding, removing or rewiring a block, a condition or an exit is another."""
    body = {k: v for k, v in defn.items() if k in EDITABLE}

    def blank(x):
        if isinstance(x, dict):
            return {k: blank(v) for k, v in x.items()}
        if isinstance(x, list):
            return [blank(v) for v in x]
        return "#" if isinstance(x, float) else x

    text = json.dumps(blank(_canonical(body)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class PmOverride:
    """The PM's go-ahead for one more configuration past an idea's budget, with who and why, for the record."""

    by: str
    reason: str
    at: datetime

    def __post_init__(self) -> None:
        if self.by != "pm" or not self.reason.strip():
            raise ValueError("a budget override is the PM's, with the reason written down")


@dataclass(frozen=True)
class Candidate:
    """A definition an optimiser proposes. Data only: nothing starts it. status is always "candidate"."""

    definition: dict
    definition_hash: str
    idea_hash: str
    parent_hash: str
    new_idea: bool = False
    override: PmOverride | None = None
    status: str = "candidate"


def _ablation_hashes(register) -> set[str]:
    """The definition hashes of every ablation variant, read from their rows, whatever idea they were filed under.
    A row whose settings lack the parent's hash or the removed path has lost its lineage, so nothing can be checked
    against it: refused."""
    out = set()
    for t in register.store.trials():
        if t.get("kind") != "ablation":
            continue
        settings = json.loads(t["settings"] or "{}")
        if not settings.get("parent_definition_hash") or "removed_path" not in settings:
            raise ValueError(f"ablation row {t['id']} has no parent hash or removed path, so a candidate's lineage "
                             "can't be checked")
        out.add(t["definition_hash"])
    return out


def configurations(register, idea: str) -> set[str]:
    """The configurations an idea has tried: distinct definition hashes among the rows the register counts (kind
    'single'), so the sensitivity grid counts and ablations don't."""
    return {t["definition_hash"] for t in register._counted(idea)}


def validate_candidate(parent: dict, candidate: dict, *, register, new_idea: bool = False,
                       override: PmOverride | None = None, venue: str | None = None) -> Candidate:
    """The candidate as a Candidate, or ValueError naming why it is refused. register: the trials register, which
    holds the idea's ablation rows and counts its budget."""
    if not isinstance(candidate, dict):
        raise TypeError("a candidate is a definition")
    changed = sorted(k for k in set(parent) | set(candidate)
                     if k not in EDITABLE and parent.get(k) != candidate.get(k))
    if changed:
        raise ValueError(f"a candidate may change only {', '.join(EDITABLE)}; it changed {', '.join(changed)}")
    check_definition(candidate)  # refused before anything runs
    parent_hash, cand_hash = definition_hash(parent, venue), definition_hash(candidate, venue)
    parent_idea, cand_idea = idea_hash(parent), idea_hash(candidate)
    variants = _ablation_hashes(register)
    if cand_hash in variants:
        raise ValueError("the candidate is an ablation variant: those are diagnostic only and never promoted")
    if parent_hash in variants:
        raise ValueError("the candidate is built from an ablation variant: those are diagnostic only")
    if cand_idea != parent_idea and not new_idea:
        raise ValueError("the candidate restructures its parent (another idea); it can only be proposed flagged as "
                         "a new idea, which starts its own count")
    idea = cand_idea
    tried = configurations(register, idea)
    if cand_hash not in tried and len(tried) >= BUDGET_PER_IDEA and override is None:
        raise ValueError(f"this idea has tried {len(tried)} configurations, its budget of {BUDGET_PER_IDEA}; another "
                         "needs the PM's override")
    return Candidate(definition=candidate, definition_hash=cand_hash, idea_hash=idea, parent_hash=parent_hash,
                     new_idea=new_idea, override=override)
