"""NIGHTLY-RECON, positions by (Strategy, asset) (Drop B; DA 0028; HoQA b4/b4b): every journal row's position rebuilt
from its fills alone against the position its orders record, exactly, and every row the DA's reconcile reader finds
written under another asset.

The key. Each journal row (a sleeve) trades one instrument, so the row is the unit and the key is read from it:
(COALESCE(parent, name), the row's instrument). A multi-asset Strategy's parent trades its first asset only, so its key
is (parent, first asset) and its legs are never summed into it (DA MUST 2). A two-leg carry's rows trade one symbol on
spot and on perp, so a "~2" row's asset carries its contract, "BTC/USDT (perp)": the carry's legs never net to 0 (DA
MUST 1). Every row counts but saved backtests, reset runs included: a run put away that isn't flat is what this is for.

The two sides, both exact Decimal sums of the EXACT columns' values (summed here, row by row: SQLite's SUM is a
float's), each rounded to 8 dp and then compared exactly, with no tolerance (DA MUST 3; F-RECON-DUST):
- rebuilt: the row's fills, BUY less SELL (rebuild_positions).
- stored: what its orders say filled (stored_positions, Store.positions_by_asset): an order's filled_qty, which the
  runtime's book_fill moves in the fill's own transaction, or, for an order marked filled with nothing booked onto it
  (a fill recorded apart from its order), its whole qty. A fill whose order has no row (one sent before the journal
  had orders) has only its own figure, so it counts the same on both sides and is listed as unanchored, never a gap.
A fill changed, lost or booked under another order shows as the two sides apart on its row, named by its Strategy, asset
and row. The equity marks' positions are the per-poll book check's (supervisor.positions_breaks), which the night also
runs.

Both sides are read in one connection and one snapshot (REPEATABLE READ on Postgres; one read transaction on SQLite), so
a fill landing mid-read can't show as a gap. Reads only: nothing here writes to the journal.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal

from sqlalchemy import select

from sleeve_fund.money import to_decimal

ZERO = Decimal(0)
STEP = Decimal("1e-8")  # F-RECON-DUST: both sides are rounded to 8 dp, then compared exactly
MISMATCHES = 1000  # asset_mismatches rows listed a night; past it the night says more exist


@dataclass(frozen=True)
class Finding:
    """One break: the Strategy and asset it is keyed by, the journal row to fix, what is wrong and the two figures."""

    strategy: str
    asset: str
    row: str
    kind: str  # "position" (the two sides apart) or "instrument" (a journal row written under another asset)
    rebuilt: Decimal | None
    stored: Decimal | None
    what: str

    def __str__(self) -> str:
        return self.what


@contextmanager
def snapshot(store):
    """One connection holding one snapshot for every read in it."""
    eng = store.engine
    if eng.dialect.name == "postgresql":
        with eng.connect().execution_options(isolation_level="REPEATABLE READ") as c, c.begin():
            yield c
    else:
        with eng.connect() as c, c.begin():
            yield c


def _rows(conn) -> dict[str, tuple[str, str]]:
    """Every journal row but saved backtests', with its key: {row: (Strategy, asset)}."""
    from sleeve_fund import markets
    from sleeve_fund.store import _not_backtest, sleeves_t

    t = sleeves_t
    q = select(t.c.name, t.c.parent, t.c.instrument, t.c.params).where(_not_backtest(t.c.name))
    out = {}
    for r in conn.execute(q):
        asset = r.instrument
        if r.name.endswith("~2"):  # a carry's second leg: the parent's symbol on the other contract (DA MUST 1)
            asset = f"{r.instrument} ({'perp' if markets.is_perp(r.params) else 'spot'})"
        out[r.name] = (r.parent or r.name, asset)
    return out


def _add(out: dict[str, Decimal], row: str, side: str, qty) -> None:
    q = to_decimal(qty, "qty")
    out[row] = out.get(row, ZERO) + (q if side == "BUY" else -q)


def _rebuilt_rows(conn) -> dict[str, Decimal]:
    """Each row's fills, BUY less SELL. Summed here row by row as the column's exact Decimals, not by the database:
    SQLite keeps a REAL and its SUM is a float's (DA-9)."""
    from sleeve_fund.store import fills_t

    f, out = fills_t.c, {}
    for r in conn.execute(select(f.sleeve, f.side, f.qty)):
        _add(out, r.sleeve, r.side, r.qty)
    return out


def _stored_rows(conn) -> tuple[dict[str, Decimal], dict[str, int]]:
    """The orders' side per row, and per row how many fills have no order row (unanchored)."""
    from sleeve_fund.store import fills_t, orders_t

    o, f = orders_t.c, fills_t.c
    stored: dict[str, Decimal] = {}
    for r in conn.execute(select(o.sleeve, o.side, o.qty, o.status, o.filled_qty)):
        filled = to_decimal(r.filled_qty, "filled_qty")
        _add(stored, r.sleeve, r.side, r.qty if r.status == "filled" and filled == 0 else filled)
    unanchored: dict[str, int] = {}
    lone = (select(f.sleeve, f.side, f.qty).select_from(fills_t.outerjoin(orders_t, o.order_id == f.order_id))
            .where(o.id.is_(None)))
    for r in conn.execute(lone):
        _add(stored, r.sleeve, r.side, r.qty)
        unanchored[r.sleeve] = unanchored.get(r.sleeve, 0) + 1
    return stored, unanchored


def _twins(rows: dict[str, tuple[str, str]]) -> dict[tuple[str, str], list[str]]:
    """Keys more than one row gives (0028's asset index rules it out; never a row hidden behind another)."""
    by_key: dict[tuple[str, str], list[str]] = {}
    for row, key in rows.items():
        by_key.setdefault(key, []).append(row)
    return {k: sorted(v) for k, v in by_key.items() if len(v) > 1}


def _keyed(rows: dict[str, tuple[str, str]], by_row: dict[str, Decimal]) -> dict[tuple[str, str], Decimal]:
    if twins := _twins(rows):
        raise ValueError("journal rows share a (Strategy, asset) key: " + "; ".join(
            f"{k[0]}, {k[1]}: {', '.join(v)}" for k, v in sorted(twins.items())))
    return {key: by_row.get(row, ZERO) for row, key in rows.items()}


def rebuild_positions(store, conn=None) -> dict[tuple[str, str], Decimal]:
    """{(Strategy, asset): net quantity} rebuilt from fills alone, every row but saved backtests'."""
    if conn is None:
        with snapshot(store) as c:
            return rebuild_positions(store, c)
    return _keyed(_rows(conn), _rebuilt_rows(conn))


def stored_positions(store, conn=None) -> dict[tuple[str, str], Decimal]:
    """{(Strategy, asset): net quantity} as the orders record it (Store.positions_by_asset)."""
    if conn is None:
        with snapshot(store) as c:
            return stored_positions(store, c)
    return _keyed(_rows(conn), _stored_rows(conn)[0])


def _8dp(x: Decimal) -> Decimal:
    """F-RECON-DUST (HoE, HoQA 15 Oct): a quantity at 8 dp, half to even. Rows from before the journal kept quantities
    exact (DA-9) can carry an 18-place residue in an order's filled_qty; at 8 dp it is equal, and one 8-dp step still
    differs, so no tolerance band hides a real break (DA MUST 3)."""
    return x.quantize(STEP, ROUND_HALF_EVEN)


def nightly(store) -> list[Finding]:
    """Every break, [] when clean: each row whose rebuilt position isn't exactly its stored one, then each journal row
    written under another asset (Store.asset_mismatches, the DA's reconcile reader)."""
    with snapshot(store) as c:
        rows = _rows(c)
        rebuilt, (stored, _) = _rebuilt_rows(c), _stored_rows(c)
    out = []
    for row, (strategy, asset) in sorted(rows.items(), key=lambda kv: (kv[1], kv[0])):
        a, b = rebuilt.get(row, ZERO), stored.get(row, ZERO)
        if _8dp(a) != _8dp(b):
            out.append(Finding(strategy, asset, row, "position", a, b,
                               f"{strategy}, {asset} (row {row}): its fills come to {a.normalize():f} but its orders "
                               f"record {b.normalize():f}, {(a - b).normalize():f} apart"))
    for (strategy, asset), twins in sorted(_twins(rows).items()):
        out.append(Finding(strategy, asset, ", ".join(twins), "key", None, None,
                           f"{strategy}, {asset}: rows {' and '.join(twins)} share one key, so neither may stand for it"))
    reader = getattr(store, "asset_mismatches", None)
    found = reader(limit=MISMATCHES + 1) if reader is not None else []
    if len(found) > MISMATCHES:
        found = found[:MISMATCHES]
        out.append(Finding("", "", "", "instrument", None, None, f"more than {MISMATCHES} journal rows are written "
                           "under another asset: the first are listed, the rest are not"))
    for m in found:
        strategy, asset = rows.get(m["sleeve"], (m["sleeve"], m["expected"] or "no row"))
        out.append(Finding(strategy, asset, m["sleeve"], "instrument", None, None,
                           f"{strategy}, {asset} (row {m['sleeve']}): {m['table']} row {m['id']} is written under "
                           f"{m['instrument'] or 'no asset'}" + ("" if m["expected"] else ", and its row is gone")))
    return out


def unanchored(store) -> dict[str, int]:
    """{row: fills with no order row}: counted the same on both sides, so never a gap, listed so the night says so."""
    with snapshot(store) as c:
        return _stored_rows(c)[1]
