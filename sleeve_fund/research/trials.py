"""Trials register: every variant ever tested, so a result is judged against how many tries found it (v2 P1-6).

A definition is hashed whole (its `definition_hash`, one variant) and with its tunable settings left out (its
`idea_hash`, which groups the variants of one idea). The indicator library's own version goes with each trial,
so a changed indicator makes a new variant even when the definition is unchanged. The register lives in the
database (Store.add_trials); the older research idea counter, a JSONL file, is folded in at start-up by
`import_ledger`, which can run any number of times.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd

from sleeve_fund.research.metrics import deflated_sharpe_probability
from sleeve_fund.store import Store

INDICATORS = Path(__file__).resolve().parent.parent / "strategies" / "indicators"
# Trials imported from the idea counter predate the indicator library.
LEGACY_CODE = "pre-v2"

# The start of a paper strategy's seen data: the beginning of time, as it was chosen with all of history in view.
OPEN_START = datetime(1970, 1, 1, tzinfo=timezone.utc)


def content_hash(obj) -> str:
    """A stable SHA-256 of any JSON-able value: the same content gives the same hash whatever its key order."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def code_version() -> str:
    """The indicator library's version: a SHA-1 of its source files, so it changes exactly when an indicator's
    code does, and needs no git checkout on the server."""
    h = hashlib.sha1()
    for path in sorted(INDICATORS.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


class TrialsRegister:
    def __init__(self, store: Store) -> None:
        self.store = store

    def record(self, *, definition_hash: str, idea_hash: str, name: str, family: str, settings: dict, dataset: str,
               stage: str, source: str, sharpe: float | None, trades: int | None = None,
               oos_trades: int | None = None, backtest_id: str | None = None, row_id: str | None = None,
               data_start=None, data_end=None) -> str:
        """Add one evaluation, over the bars from data_start to data_end. Returns its id. row_id, when given, is
        kept, so a row recorded twice (a study that also writes the idea counter) counts once."""
        row_id = row_id or secrets.token_hex(8)
        self.store.add_trials([{
            "id": row_id, "definition_hash": definition_hash, "idea_hash": idea_hash, "code_version": code_version(),
            "definition_name": name, "family": family, "settings": json.dumps(settings, sort_keys=True),
            "dataset": dataset, "stage": stage, "source": source, "sharpe": _finite(sharpe), "trades": trades,
            "oos_trades": oos_trades, "backtest_id": backtest_id, "data_start": data_start, "data_end": data_end,
            # To the microsecond, so a variant's latest evaluation is known even within a second (sharpes()).
            "created_at": datetime.now(timezone.utc),
        }])
        return row_id

    def _counted(self, idea_hash: str | None = None) -> list[dict]:
        """Every evaluation that counts: not benchmarks, and not runs deliberately marked as engineering
        (fixtures), which nothing marks by default (Advisor, 6 Oct 2026)."""
        return [t for t in self.store.trials(idea_hash) if t["family"] != "benchmark" and t["source"] != "engineering"]

    def counts(self, idea_hash: str | None = None) -> dict:
        """ideas: distinct ideas; variants: distinct (definition, indicator code, dataset); evaluations: rows.
        With idea_hash, one idea family's: everything tried under that core idea, across settings, instruments,
        timeframes and registrations. That is the N a result is judged by; the project-wide total, without
        idea_hash, is shown for awareness and never gates (Advisor, 6 Oct 2026)."""
        rows = self._counted(idea_hash)
        return {
            "ideas": len({t["idea_hash"] for t in rows}),
            "variants": len({_variant(t) for t in rows}),
            "evaluations": len(rows),
        }

    def failed(self, idea_hash: str | None = None) -> int:
        """Runs whose count failed (QA P1-T8): each still counts as one variant tried in counts(), and their number
        says how far that count rests on rows with no result."""
        return sum(1 for t in self._counted(idea_hash) if t["status"] == "failed")

    def record_failed(self, *, strategy: str, params: dict, source: str, error: str, setup: dict | None = None,
                      dataset: str | None = None, backtest_id: str | None = None) -> str:
        """Record, against its idea, a run whose count failed (QA P1-T8): its own source, its settings and the
        error. It has no Sharpe, trades or dates, so it counts as a variant tried and a holdout treats the idea as
        having read undated data: the safe side on both. With the run's setup and dataset it is keyed as its
        variant, so a retry that counts joins it rather than adding one (Data Architect); without them, by its
        strategy and settings alone, so the same failing run retried is still one variant."""
        try:
            from sleeve_fund.research.run import spec_of

            family = spec_of(strategy).family
        except ValueError:
            family = "unknown"
        definition = (legacy_definition_hash(strategy, params, setup) if setup is not None else
                      content_hash({"unkeyed": strategy, "params": params}))
        row_id = secrets.token_hex(8)
        self.store.add_trials([{
            "id": row_id, "definition_hash": definition, "idea_hash": legacy_idea_hash(strategy),
            "code_version": code_version(), "definition_name": strategy, "family": family,
            "settings": json.dumps({"params": params}, sort_keys=True), "dataset": dataset or "unknown",
            "stage": "in_sample", "source": source, "sharpe": None, "trades": None, "oos_trades": None,
            "backtest_id": backtest_id, "status": "failed", "error": error,
            "created_at": datetime.now(timezone.utc),
        }])
        return row_id

    def ideas_by_family(self) -> dict[str, int]:
        """How many ideas each family (trend, breakout...) holds, project-wide."""
        by_family: dict[str, set] = {}
        for t in self._counted():
            by_family.setdefault(t["family"], set()).add(t["idea_hash"])
        return {k: len(v) for k, v in sorted(by_family.items())}

    def sharpes(self, idea_hash: str | None = None) -> list[float]:
        """One Sharpe per variant: its latest evaluation's (Advisor, 6 Oct 2026), so a variant re-run many times
        doesn't weigh more in the spread than one run once."""
        latest: dict[tuple, dict] = {}
        for t in self._counted(idea_hash):
            if t["sharpe"] is not None:
                latest[_variant(t)] = t  # trials() is oldest first
        return [t["sharpe"] for t in latest.values()]

    def deflated_sharpe(self, returns: pd.Series, idea_hash: str) -> float:
        """Probability the result's true Sharpe beats the best of every variant of its idea tried so far by luck:
        the deflated Sharpe ratio (Bailey and Lopez de Prado). N is the idea family's variant count; the spread is
        that family's trial Sharpes, one per variant, floored at the no-skill error 1/sqrt(T). The tear sheet and
        the Research page read this one number (QA P1-T2)."""
        return deflated_sharpe_probability(returns, max(self.counts(idea_hash)["variants"], 1), self.sharpes(idea_hash))

    def import_ledger(self, path: str | Path) -> int:
        """Fold the JSONL idea counter into the register. Each line gets an id from a hash of the line itself,
        so lines already imported are skipped and running this again changes nothing. The file is left as it is.
        Returns how many rows were added."""
        path = Path(path)
        if not path.exists():
            return 0
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            rows.append({
                "id": line_id(line),
                "definition_hash": legacy_definition_hash(e["idea"], e["params"]),
                "idea_hash": legacy_idea_hash(e["idea"]),
                "code_version": LEGACY_CODE,
                "definition_name": e["idea"], "family": e["family"],
                "settings": json.dumps(e["params"], sort_keys=True),
                "dataset": e["dataset"], "stage": str(e["stage"])[:16], "source": "ledger_import",
                "sharpe": _finite(e.get("sharpe")), "trades": None, "oos_trades": None, "backtest_id": None,
                "created_at": datetime.fromisoformat(e["ts"]),
            })
        return self.store.add_trials(rows)


def record_model_run(store: Store, *, strategy: str, params: dict, dataset: str, source: str, setup: dict,
                     sharpe: float | None = None, data_start=None, data_end=None, backtest_id: str | None = None,
                     trades: int | None = None) -> str:
    """Count one run of a hand-coded model outside a study: a backtest, or a paper strategy created, cloned or
    re-set (QA P1-T1). Each is a variant tried, so the deflated Sharpe's N counts it. The whole run reads every
    bar it was given, so it is in-sample."""
    if source == "strategy" and data_end is None:
        # Chosen having seen everything up to the moment it was made or edited (Advisor, 6 Oct 2026, QA P1-T4 and
        # P1-T7): it read from the open start to then, so no holdout before that is unseen. Watching it trade
        # afterwards reads nothing new into the choice; each edit is a new row, dated again.
        data_start, data_end = OPEN_START, datetime.now(timezone.utc)
    try:
        from sleeve_fund.research.run import spec_of

        family = spec_of(strategy).family
    except ValueError:
        family = "unknown"
    return TrialsRegister(store).record(
        definition_hash=legacy_definition_hash(strategy, params, setup), idea_hash=legacy_idea_hash(strategy),
        name=strategy,
        family=family, settings=params, dataset=dataset, stage="in_sample", source=source, sharpe=sharpe,
        trades=trades, backtest_id=backtest_id, data_start=data_start, data_end=data_end)


def _variant(t: dict) -> tuple:
    return t["definition_hash"], t["code_version"], t["dataset"]


def line_id(line: str) -> str:
    """A trial id from an idea-counter line, so the same evaluation is one row however it arrives."""
    return hashlib.sha256(line.strip().encode("utf-8")).hexdigest()[:16]


def legacy_idea_hash(idea: str) -> str:
    """The idea hash of a hand-coded model, known by its name until it becomes a definition (P1-5)."""
    return content_hash({"idea": idea})


def legacy_definition_hash(idea: str, params: dict, setup: dict | None = None) -> str:
    """A hand-coded model's variant. setup holds what else someone could choose to make a result look better:
    the risk profile (and so the sizing), the walk-forward windows and the fee assumed when choosing (Advisor,
    6 Oct 2026). Fee-ladder rungs are not part of it. Counter lines imported from before it kept none."""
    return content_hash({"idea": idea, "params": params} if setup is None else
                        {"idea": idea, "params": params, "setup": setup})


def run_setup(*, risk_profile: str | None, fee: float, windows: tuple[int, int, int] | None = None,
              period: str | None = None) -> dict:
    """The setup part of a variant's key: risk profile, walk-forward windows (train, test, holdout days; None
    for a single run over all the bars), the fee per side assumed when choosing, rounded to a basis point
    hundredth so a float's last digit doesn't make a new variant, and a backtest's period (backtest_period)."""
    out = {"risk_profile": risk_profile, "windows": list(windows) if windows else None, "fee": round(fee, 6)}
    if period is not None:  # left out when not given, so studies' keys are as before
        out["period"] = period
    return out


def backtest_period(days: int | None = None, start=None, end=None) -> str:
    """A backtest's period in the variant key (Independent Quant Advisor, 6 Oct 2026, QA P1-T6): a preset by its
    name, so re-running "last 365 days" next week is the same variant, and a custom range by its exact dates.
    The dates actually read are stored on the row either way."""
    if start is not None or end is not None:
        return f"{pd.Timestamp(start):%Y-%m-%d} to {pd.Timestamp(end):%Y-%m-%d}"
    return f"last {int(days)} days" if days else "all history"


def _finite(x) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if x == x and abs(x) != float("inf") else None

