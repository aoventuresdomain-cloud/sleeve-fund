"""A two-leg Strategy on the dashboard (P2-8 step 7, spec pe2-plans/p2-8-step7-dashboard.md). Display only.

Each leg is a sleeves row of its own (the second "<name>~2", whose parent is the first), so every money path and the
book's figures keep both rows. The lists the PM reads show the Strategy once, under its first leg's name, with its
figures the two legs summed and the legs nested underneath. Its drawdown is the summed guard's (Advisor E): from the
pair's high-water mark in pair_state once the pair is marked."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sleeve_fund import markets
from sleeve_fund.dashboard.book import spark_path
from sleeve_fund.dashboard.metrics import costs
from sleeve_fund.store import LEG, Store
from sleeve_fund.wording import leg_name

# The pair's alerts the Strategy's row and page flag in red until acknowledged (M4): no coordinator, a broken pair
# (a leg unwound), the liquidation-distance close (ruling B) and a pair state that couldn't be saved.
FLAG_KINDS = ("two_leg_no_pair", "two_leg_broken", "two_leg_liq_distance", "pair_state_unsaved")
FLAG_WORDS = {"two_leg_no_pair": "no coordinator", "two_leg_broken": "a leg was unwound",
              "two_leg_liq_distance": "closed near liquidation", "pair_state_unsaved": "its state wasn't saved"}
# pair_state's states as the PM reads them: "legging" is the first leg in, the second on its way.
STATE_WORDS = {"flat": "Flat", "legging": "Opening", "hedged": "Hedged", "closing": "Closing"}
# Open alerts read for the flag. An open alert is one the PM hasn't acknowledged, so far fewer than this.
_ALERTS_READ = 500


def uk_time(t: datetime | None) -> str:
    """A pair's deadline as the PM reads it, in UK time to the second (a legged state lasts about a minute)."""
    if t is None:
        return ""
    t = t if t.tzinfo else t.replace(tzinfo=UTC)
    try:
        return t.astimezone(ZoneInfo("Europe/London")).strftime("%H:%M:%S UK")
    except ZoneInfoNotFoundError:  # no time zone data on the host: UTC, said so
        return t.astimezone(UTC).strftime("%H:%M:%S UTC")


def label(name: str | None) -> str:
    """A strategy row's name as the PM reads it: a second leg's "<name>~2" as "<name>, leg 2" (wording.leg_name)."""
    return leg_name(name)


def legs_of(sleeves) -> dict[str, str]:
    """{first leg's name: second leg's name} for every two-leg Strategy among `sleeves` (parent set on the second)."""
    return {s.parent: s.name for s in sleeves if getattr(s, "parent", None)}


def flags(store: Store) -> dict[str, list[dict]]:
    """Unacknowledged M4 alerts by the Strategy they belong to (the first leg's name), newest first."""
    out: dict[str, list[dict]] = {}
    for a in store.alerts(limit=_ALERTS_READ):
        if a["kind"] in FLAG_KINDS and a.get("sleeve"):
            name = a["sleeve"].removesuffix(LEG)
            out.setdefault(name, []).append({**a, "words": FLAG_WORDS[a["kind"]]})
    return out


def leg_row(x: dict, n: int) -> dict:
    """One leg as the Strategy's nested row shows it."""
    s = x["sleeve"]
    side = (s.params.get("leg") or {}).get("side")
    return {"n": n, "name": s.name, "instrument": s.instrument, "perp": markets.is_perp(s.params),
            "held": bool(x["qty"]), "short": x["qty"] < 0 if x["qty"] else side == -1,
            "qty": x["qty"], "equity": x["equity"], "pnl": x["pnl"], "fees": x["fees"], "funding": x["funding"]}


def pair_view(store: Store, parent: str) -> dict:
    """The pair's state from pair_state: {state, words, trade_id, deadline}. Flat before the coordinator's first write."""
    row = store.pair_state(parent) or {}
    state = row.get("state") or "flat"
    return {"state": state, "words": STATE_WORDS.get(state, state.capitalize()),
            "trade_id": row.get("trade_id"), "deadline": row.get("deadline"), "hwm": row.get("hwm"),
            "day_start_equity": row.get("day_start_equity"),
            "open": state in ("legging", "closing")}


def _summed_curve(frames: list) -> list[float]:
    frames = [f["equity"] for f in frames if f is not None and len(f)]
    if not frames:
        return []
    joined = frames[0].to_frame("a").join([f.rename(f"l{i}") for i, f in enumerate(frames[1:])], how="outer")
    return joined.ffill().dropna().sum(axis=1).tolist()


def fold_one(store: Store, first: dict, second: dict, frames: dict | None = None) -> dict:
    """The Strategy's row: the first leg's summary with its figures the two legs summed, and `legs` nested."""
    parent = first["sleeve"].name
    pair = pair_view(store, parent)
    start = first["sleeve"].starting_balance + second["sleeve"].starting_balance
    equity = first["equity"] + second["equity"]
    curve = _summed_curve([(frames or {}).get(x["sleeve"].name) for x in (first, second)])
    # The summed guard's peak (pair_state.hwm) once the pair is marked; until then the summed curve's, from the start.
    peak = max(float(pair["hwm"]) if pair["hwm"] is not None else max(curve, default=0.0), start, equity)
    dd = 1 - equity / peak if peak else 0.0
    max_dd = first["profile"].max_drawdown
    day_pnl = first["day_pnl"] + second["day_pnl"]
    out = {
        **first,
        "equity": equity, "pnl": equity - start, "ret": equity / start - 1 if start else 0.0,
        "benchmark": first["benchmark"] + second["benchmark"],
        "bench_ret": (first["benchmark"] + second["benchmark"]) / start - 1 if start else 0.0,
        "day_pnl": day_pnl, "day_ret": day_pnl / (equity - day_pnl) if equity - day_pnl else 0.0,
        "week_pnl": first["week_pnl"] + second["week_pnl"], "mtd_pnl": first["mtd_pnl"] + second["mtd_pnl"],
        "peak": peak, "drawdown": dd, "dd_used": min(dd / max_dd, 1.0) if max_dd else 0.0,
        "room": max(equity - peak * (1 - max_dd), 0.0),
        # Gross: a hedged pair's net is near nothing, and its size is both legs'.
        "exposure": (abs(first["position_value"]) + abs(second["position_value"])) / equity if equity else 0.0,
        "fees": first["fees"] + second["fees"], "funding": first["funding"] + second["funding"],
        "fills": first["fills"] + second["fills"],
        "healthy": first["healthy"] and second["healthy"],
        "legs": [leg_row(first, 1), leg_row(second, 2)], "pair": pair,
    }
    out.update(costs(out["pnl"], out["fees"], out["funding"]))
    if curve:
        tail = curve[-60:]
        out.update(spark=spark_path(tail), spark_up=len(tail) < 2 or tail[-1] >= tail[0])
    return out


def fold(store: Store, summaries: list[dict], frames: dict | None = None) -> list[dict]:
    """The summaries as the PM's lists show them: a two-leg Strategy once, at its first leg's place, folded
    (fold_one); its second leg's own row left out. A second leg whose first isn't in the list stays as it is, so
    nothing held is ever hidden."""
    by_name = {x["sleeve"].name: x for x in summaries}
    pairs = {p: n for p, n in legs_of([x["sleeve"] for x in summaries]).items() if p in by_name}
    folded = set(pairs.values())
    return [fold_one(store, x, by_name[pairs[x["sleeve"].name]], frames) if x["sleeve"].name in pairs else x
            for x in summaries if x["sleeve"].name not in folded]


def without_legs(sleeves) -> list:
    """The strategy rows without a second leg whose first is among them (the rail's and the menus' lists)."""
    names = {s.name for s in sleeves}
    return [s for s in sleeves if not (getattr(s, "parent", None) and s.parent in names)]
