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
  of N safe), read from the ablation rows' kind and their settings (parent_definition_hash, removed_path). An
  ablation row that has lost that lineage refuses every candidate in its idea and raises an alert; one that has lost
  its idea too refuses every candidate (Advisor 13 Oct 15:40 UK);
- its idea differs from its parent's and it isn't flagged as a new idea with the Researcher's written hypothesis, so
  restructuring can't escape an idea's budget. A new idea still counts in the project-wide N the deflated Sharpe
  uses, as every trial does;
- its idea has used its budget of BUDGET_PER_IDEA configurations, counted by the trials register, unless the PM's
  override is given (Advisor 18:20 Q4: sensitivity grid included, ablations not). The override goes in the PM's
  decision log right after the configuration's trial row (log_override, DA 13 Oct).

A candidate's evaluations count under its own idea (CR F236-1): record_candidate() writes its row, a study takes
run_study(idea_hash=candidate.idea_hash), and an ablation is filed under its parent's idea (ablation.ablate). The
scan in tests/test_candidates.py refuses an optimiser that records any other way, or rebinds anything here."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime

from sleeve_fund.strategies.definitions import check_definition, definition_hash

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


# Indicators that are one idea whatever their smoothing (Advisor 13 Oct 15:40 UK): an SMA, EMA or WMA is a moving
# average, and ATR is ATR whether Wilder's or a simple average. Every other kind is its own indicator.
FAMILIES = {"sma": "ma", "ema": "ma", "wma": "ma", "atr": "atr", "atr_sma": "atr"}
# Comparison variants that are one idea: > and >= ask the same question, as do < and <=.
SAME_OP = {">=": ">", "<=": "<"}
# Price series that are one input: a condition on the close or on the high asks the same question. Volume is not one.
PRICES = ("open", "high", "low", "close")


def idea_hash(defn: dict) -> str:
    """The definition with every tuning knob left out (Advisor 13 Oct 15:40 UK): what it measures and how the pieces
    are wired, not how each is tuned. These are one idea, and use up its budget:
    - numbers;
    - a comparison's variant (> or >=, < or <=);
    - the input series (open, high, low or close, and a block's settings such as its source);
    - the moving-average type or smoothing within one indicator (SMA, EMA, WMA; ATR or its simple average);
    - the lookback or interval (a block's settings and timeframe), and a block's name.
    These make a new idea: adding or removing a block, condition or exit; changing the and/or wiring; changing which
    indicator a condition measures (RSI to volume, say); changing the sides traded. When in doubt it is the same
    idea: the budget limits selection, and flipping a knob must not reset it."""
    blocks = defn.get("blocks") or {}
    kinds = {str(bid): FAMILIES.get(b.get("kind"), b.get("kind")) for bid, b in blocks.items() if isinstance(b, dict)}

    def ref(x: str) -> str:
        if x in PRICES:
            return "price"
        bid, dot, output = x.partition(".")
        return f"{kinds[bid]}{dot}{output}" if bid in kinds else x

    def shape(x, key=None):
        if isinstance(x, dict):
            out = {k: shape(v, k) for k, v in x.items()}
            for k in ("all", "any"):  # members in one order, duplicates kept: two conditions are not one
                if isinstance(out.get(k), list):
                    out[k] = sorted(out[k], key=lambda m: json.dumps(m, sort_keys=True))
            return out
        if isinstance(x, list):
            return [shape(v) for v in x]
        if isinstance(x, bool) or x is None:
            return x
        if isinstance(x, (int, float)) or key == "tz":
            return "#"
        if key == "op":
            return SAME_OP.get(x, x)
        return ref(x) if isinstance(x, str) else x

    body = {k: shape(defn[k]) for k in ("long", "short", "exits") if k in defn}
    # A block is its indicator and what it reads; its settings, timeframe and name are knobs.
    body["blocks"] = sorted(
        (json.dumps({"kind": kinds[bid], "input": ref(str(b.get("input") or "close"))},
                    sort_keys=True) for bid, b in blocks.items() if isinstance(b, dict)))
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
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
class NewIdea:
    """A candidate flagged as a new idea, with the Researcher's written hypothesis for it (Advisor 13 Oct 15:40 UK).
    It starts its own budget; it still counts in the project-wide N."""

    by: str
    hypothesis: str

    def __post_init__(self) -> None:
        if self.by != "researcher" or not self.hypothesis.strip():
            raise ValueError("a new idea is flagged by the Researcher, with the hypothesis written down")


@dataclass(frozen=True)
class Candidate:
    """A definition an optimiser proposes. Data only: nothing starts it. status is always "candidate"."""

    definition: dict
    definition_hash: str
    idea_hash: str
    parent_hash: str
    new_idea: NewIdea | None = None
    override: PmOverride | None = None
    configuration: int = 0  # this configuration's number in its idea's count; past the budget only with override
    status: str = "candidate"


def _ablation_hashes(register, ideas: set[str]) -> set[str]:
    """The definition hashes of every ablation variant, read from their rows, whatever idea they were filed under.
    A row whose settings lack the parent's hash or the removed path has lost its lineage, so nothing in its idea can be
    checked against it: a candidate of that idea (ideas: the parent's and the candidate's) is refused, with an alert.
    A row that has lost its idea as well could be anyone's, so every candidate is refused (Advisor 13 Oct 15:40 UK)."""
    out = set()
    for t in register.store.trials():
        if t.get("kind") != "ablation":
            continue
        settings = json.loads(t["settings"] or "{}")
        if settings.get("parent_definition_hash") and "removed_path" in settings:
            out.add(t["definition_hash"])
            continue
        if t.get("idea_hash") and t["idea_hash"] not in ideas:
            continue
        message = (f"ablation row {t['id']} has no parent hash or removed path, so a candidate's lineage can't be "
                   f"checked; candidates of idea {str(t.get('idea_hash') or 'unknown')[:12]} are refused until "
                   "it is fixed")
        if t.get("idea_hash"):
            event = getattr(register.store, "event", None)
            if event is not None:
                event(None, "warning", "ablation_lineage", message)
        raise ValueError(message if t.get("idea_hash") else
                         f"ablation row {t['id']} has no lineage and no idea, so no candidate's lineage can be checked")
    return out


def configurations(register, idea: str) -> set[str]:
    """The configurations an idea has tried: distinct definition hashes among the rows the register counts (kind
    'single'), so the sensitivity grid counts and ablations don't."""
    return {t["definition_hash"] for t in register._counted(idea)}


def validate_candidate(parent: dict, candidate: dict, *, register, new_idea: NewIdea | None = None,
                       override: PmOverride | None = None, venue: str | None = None) -> Candidate:
    """The candidate as a Candidate, or ValueError naming why it is refused. register: the trials register, which
    holds the idea's ablation rows and counts its budget."""
    if not isinstance(candidate, dict):
        raise TypeError("a candidate is a definition")
    if new_idea is not None and not isinstance(new_idea, NewIdea):
        raise TypeError("a new idea is flagged with NewIdea, carrying the Researcher's hypothesis")
    changed = sorted(k for k in set(parent) | set(candidate)
                     if k not in EDITABLE and parent.get(k) != candidate.get(k))
    if changed:
        raise ValueError(f"a candidate may change only {', '.join(EDITABLE)}; it changed {', '.join(changed)}")
    check_definition(candidate)  # refused before anything runs
    parent_hash, cand_hash = definition_hash(parent, venue), definition_hash(candidate, venue)
    parent_idea, cand_idea = idea_hash(parent), idea_hash(candidate)
    variants = _ablation_hashes(register, {parent_idea, cand_idea})
    if cand_hash in variants:
        raise ValueError("the candidate is an ablation variant: those are diagnostic only and never promoted")
    if parent_hash in variants:
        raise ValueError("the candidate is built from an ablation variant: those are diagnostic only")
    if cand_idea != parent_idea and new_idea is None:
        raise ValueError("the candidate restructures its parent (another idea); it can only be proposed flagged as "
                         "a new idea, with the Researcher's hypothesis, which starts its own count")
    if cand_idea == parent_idea and new_idea is not None:
        raise ValueError("the candidate only retunes its parent, so it is the same idea and uses its budget; it "
                         "can't be flagged as a new one")
    idea = cand_idea
    tried = configurations(register, idea)
    number = len(tried) if cand_hash in tried else len(tried) + 1
    if number > BUDGET_PER_IDEA and override is None:
        raise ValueError(f"this idea has tried {len(tried)} configurations, its budget of {BUDGET_PER_IDEA}; another "
                         "needs the PM's override")
    return Candidate(definition=candidate, definition_hash=cand_hash, idea_hash=idea, parent_hash=parent_hash,
                     new_idea=new_idea, override=override if number > BUDGET_PER_IDEA else None,
                     configuration=number)


def record_candidate(register, candidate: Candidate, **trial) -> str:
    """The one way an evaluation of a candidate is counted outside a study (CR F236-1): a trials row under the
    candidate's own idea and definition, which configurations() and the lineage check read, then the PM's override,
    if any, in the decision log. trial: the rest of the row, as TrialsRegister.record takes it. A study counts a
    candidate's runs under its idea with run_study(idea_hash=candidate.idea_hash). Returns the row's id."""
    keyed = sorted(k for k in ("definition_hash", "idea_hash") if k in trial)
    if keyed:
        raise ValueError(f"a candidate's row is keyed by the candidate itself; {', '.join(keyed)} can't be given")
    trial_id = register.record(definition_hash=candidate.definition_hash, idea_hash=candidate.idea_hash, **trial)
    log_override(register.store, candidate, trial_id)
    return trial_id


def log_override(store, candidate: Candidate, trial_id: str) -> None:
    """Write the PM's budget override to the PM's decision log, right after the configuration's trial row is
    recorded (DA 13 Oct: a decision, not an incident or a trials column). Nothing to write within the budget."""
    if candidate.override is None:
        return
    store.decide("PM", "budget_override",
                 f"{candidate.override.reason.strip()} (idea {candidate.idea_hash[:12]}, trial {trial_id}, "
                 f"configuration {candidate.configuration} of {BUDGET_PER_IDEA})", sleeve=None)
