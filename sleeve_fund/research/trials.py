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
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd

from sleeve_fund.research.metrics import deflated_sharpe_probability
from sleeve_fund.store import Store

INDICATORS = Path(__file__).resolve().parent.parent / "strategies" / "indicators"
# Trials imported from the idea counter predate the indicator library.
LEGACY_CODE = "pre-v2"


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
               oos_trades: int | None = None, backtest_id: str | None = None) -> str:
        """Add one evaluation. Returns its id."""
        row_id = secrets.token_hex(8)
        self.store.add_trials([{
            "id": row_id, "definition_hash": definition_hash, "idea_hash": idea_hash, "code_version": code_version(),
            "definition_name": name, "family": family, "settings": json.dumps(settings, sort_keys=True),
            "dataset": dataset, "stage": stage, "source": source, "sharpe": _finite(sharpe), "trades": trades,
            "oos_trades": oos_trades, "backtest_id": backtest_id,
        }])
        return row_id

    def _counted(self) -> list[dict]:
        return [t for t in self.store.trials() if t["family"] != "benchmark"]

    def counts(self) -> dict:
        """ideas: distinct ideas; variants: distinct (definition, indicator code, dataset); evaluations: rows."""
        rows = self._counted()
        return {
            "ideas": len({t["idea_hash"] for t in rows}),
            "variants": len({(t["definition_hash"], t["code_version"], t["dataset"]) for t in rows}),
            "evaluations": len(rows),
        }

    def sharpes(self) -> list[float]:
        return [t["sharpe"] for t in self._counted() if t["sharpe"] is not None]

    def deflated_sharpe(self, returns: pd.Series) -> float:
        """Probability the result's true Sharpe beats the best of every variant tried so far by luck: the deflated
        Sharpe ratio (Bailey and Lopez de Prado), with this register's variant count. The spread of trial Sharpes
        is the no-skill sampling error, as the tear sheet uses, so fee-destroyed variants don't distort it."""
        return deflated_sharpe_probability(returns, max(self.counts()["variants"], 1))

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
                "id": hashlib.sha256(line.encode("utf-8")).hexdigest()[:16],
                "definition_hash": content_hash({"idea": e["idea"], "params": e["params"]}),
                "idea_hash": content_hash({"idea": e["idea"]}),
                "code_version": LEGACY_CODE,
                "definition_name": e["idea"], "family": e["family"],
                "settings": json.dumps(e["params"], sort_keys=True),
                "dataset": e["dataset"], "stage": str(e["stage"])[:16], "source": "ledger_import",
                "sharpe": _finite(e.get("sharpe")), "trades": None, "oos_trades": None, "backtest_id": None,
                "created_at": datetime.fromisoformat(e["ts"]),
            })
        return self.store.add_trials(rows)


def _finite(x) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if x == x and abs(x) != float("inf") else None

