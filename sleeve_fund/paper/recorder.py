"""Record what a paper sleeve saw from the venue, so a backtest can replay it exactly.

A recording is one gzipped JSON-lines file: a header with the instrument as the venue defined it,
the opening balances and the sleeve's settings, then every quote and trade in the order they
arrived. Prices and sizes are kept as the venue's strings, so nothing is rounded on the way.
sleeve_fund.research.replay feeds it back through the backtest engine with the same runtime.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path


class Recorder:
    FLUSH_EVERY = 500

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = gzip.open(self.path, "wt", encoding="utf-8")
        self._n = 0
        self.closed = False
        # Set by whoever builds the node; written once the venue's instrument is known (on_start).
        self.meta: dict = {}

    def _write(self, row: dict) -> None:
        if self.closed:
            return
        self._fh.write(json.dumps(row, separators=(",", ":")) + "\n")
        self._n += 1
        if self._n % self.FLUSH_EVERY == 0:
            self._fh.flush()

    def start(self, instrument) -> None:
        self.header(instrument=instrument, balances=self.meta.get("balances", []), sleeve=self.meta.get("sleeve", {}))

    def header(self, *, instrument, balances: list[str], sleeve: dict) -> None:
        i = instrument
        self._write({
            "k": "header", "version": 1, "sleeve": sleeve, "balances": balances,
            "instrument": {
                "id": str(i.id), "raw_symbol": str(i.raw_symbol), "base": str(i.base_currency.code),
                "quote": str(i.quote_currency.code), "price_precision": i.price_precision,
                "size_precision": i.size_precision, "price_increment": str(i.price_increment),
                "size_increment": str(i.size_increment),
                "min_quantity": str(i.min_quantity) if i.min_quantity is not None else None,
                "min_notional": str(i.min_notional) if i.min_notional is not None else None,
                "lot_size": str(i.lot_size) if i.lot_size is not None else None,
            },
        })

    def quote(self, q) -> None:
        self._write({"k": "q", "e": q.ts_event, "i": q.ts_init, "b": str(q.bid_price), "a": str(q.ask_price),
                     "bs": str(q.bid_size), "as": str(q.ask_size)})

    def trade(self, t) -> None:
        self._write({"k": "t", "e": t.ts_event, "i": t.ts_init, "p": str(t.price), "s": str(t.size),
                     "side": int(t.aggressor_side), "id": str(t.trade_id)})

    def close(self) -> None:
        if not self.closed:
            self._fh.close()
            self.closed = True


def read(path: Path | str) -> tuple[dict, list[dict]]:
    """The header and the market data rows, in arrival order."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if not rows or rows[0].get("k") != "header":
        raise ValueError(f"{path} is not a sleeve recording (no header)")
    return rows[0], rows[1:]
