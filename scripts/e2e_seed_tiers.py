"""stack-e2e only (TEC-E2E-TIERS-SEED, HoE 20 Oct): CI's stack can't reach the venue, so the margin tiers it would pull
daily never arrive and the hand-checked paste (a fetch dated the day it was read) ages past the 72-hour refusal. Before
the supervisor starts, this keeps the same pasted tables as a good tiers fetch dated now, labelled "CI seed", for every
venue that has them. The 72-hour guard itself is untouched, and is proved first on a scratch journal: the same seed
dated over 72 hours ago still refuses entries, and dated now does not. Never run against a server's journal."""

from __future__ import annotations

import math
import sys
from datetime import UTC, datetime, timedelta

from sleeve_fund import venue_terms as vt
from sleeve_fund.store import Store
from sleeve_fund.venues import VENUES

CI_SEED = "CI seed"


def seed(store, profile, at: datetime) -> int:
    """The profile's pasted tables as good tiers fetches dated `at`, as seed_pasted_tiers keeps them; how many."""
    n = 0
    for pair, table in profile.maintenance_tiers.items():
        last = profile.maintenance_limits.get(pair)
        rows = [{"ceiling": vt._num(last if math.isinf(t.ceiling) and last else t.ceiling), "mmr": vt._num(t.rate),
                 "imr": "", "max_leverage": "", "deduction": vt._num(t.amount)} for t in table]
        store.record_venue_fetch(profile.name, "tiers", pair, source=CI_SEED, ok=True, at=at,
                                 payload={"symbol": vt.symbol_of(pair), "tiers": rows})
        n += 1
    return n


def guard_holds(now: datetime) -> None:
    """The seed doesn't bypass the guard: dated over 72 hours ago it refuses, dated now it doesn't."""
    for profile in (p for p in VENUES.values() if p.maintenance_tiers):
        for age, refused in ((timedelta(hours=73), True), (timedelta(0), False)):  # the PM rule: over 72 h refuses
            scratch = Store("sqlite://")
            seed(scratch, profile, now - age)
            for pair in profile.maintenance_tiers:
                t = vt.term(scratch, profile, "tiers", pair, now)
                if (t.state == vt.REFUSED) != refused:
                    sys.exit(f"FAIL: a {pair} tiers seed {age} old reads {t.state}, expected "
                             f"{'refused' if refused else 'not refused'}: {t.words}")
    print("tiers guard: a seed over 72 hours old refuses entries; one dated now does not")


def main() -> int:
    now = datetime.now(UTC)
    guard_holds(now)
    store = Store()
    n = sum(seed(store, p, now) for p in VENUES.values() if p.maintenance_tiers)
    print(f"seeded {n} tiers table(s) as '{CI_SEED}', dated now")
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
