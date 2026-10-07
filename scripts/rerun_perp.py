"""Re-run every perpetual backtest the journal holds, after P1-D13's fill fix (Advisor, 6 Oct 19:00): a bars-only
liquidation check that ignored the bar's worst price was a MAJOR, so every perp result made before the fix is suspect.

    python scripts/rerun_perp.py            # inside the dashboard container, which has the journal and the store

Read-only: each saved backtest on a perpetual is re-run from its own settings over today's data and nothing is
saved. Prints, per run, the saved and the re-run total return and the fills label, then the trials register's
studies on perpetual venues, which re-run from the Research page. Exit 0 whatever the numbers; 1 when the journal
can't be read.
"""

from __future__ import annotations

import sys
from urllib.parse import parse_qsl


def main() -> int:
    from sleeve_fund import markets
    from sleeve_fund.dashboard import preview
    from sleeve_fund.dashboard.app import _backtest_args
    from sleeve_fund.fees import resolve as resolve_fees
    from sleeve_fund.spreads import resolve as resolve_spread
    from sleeve_fund.store import Store
    from sleeve_fund.venues import VENUES

    try:
        store = Store()
        saved = store.backtests(limit=100_000)
    except Exception as exc:  # noqa: BLE001 - the report says what failed
        print(f"couldn't read the journal: {exc!r}")
        return 1
    print(f"# Perpetual backtests re-run after P1-D13 ({len(saved)} saved backtests in all)\n")
    print("| Saved | Title | Saved return | Re-run return | Fills |")
    print("| --- | --- | --- | --- | --- |")
    perps = 0
    for row in saved:
        try:
            full = store.backtest(row["id"])
            args = _backtest_args(dict(parse_qsl(full["query"])))
        except Exception as exc:  # noqa: BLE001 - one bad row never stops the report
            print(f"| {row['created_at']:%d %b %H:%M} | {row['title']} | n/a | not re-run: {exc} | |")
            continue
        if not markets.is_perp(args["params"]):
            continue
        perps += 1
        before = (full["result"].get("strategy") or {}).get("total_return")
        try:
            now = preview.run(args["strategy"], args["pair"], args["params"], starting=args["starting"],
                              days=args["days"], minutes=args["minutes"], risk_profile=args["risk_profile"],
                              venue=args["venue"], fee_quote=resolve_fees(args["venue"], store),
                              spread_quote=resolve_spread(args["venue"], args["pair"], store))
            after, labels = now["strategy"]["total_return"], "; ".join(now["execution"].get("labels") or []) or "exact"
        except Exception as exc:  # noqa: BLE001
            after, labels = None, f"failed: {exc!r}"
        fmt = (lambda x: "n/a" if x is None else f"{x:+.2%}")  # noqa: E731
        print(f"| {row['created_at']:%d %b %H:%M} | {row['title']} | {fmt(before)} | {fmt(after)} | {labels} |")
    print(f"\n{perps} on perpetuals re-run.\n")
    perp_venues = tuple(v.name.lower() for v in VENUES.values() if v.perpetual)
    try:
        from sqlalchemy import select

        from sleeve_fund.store import trials_t

        with store.engine.connect() as c:
            rows = c.execute(select(trials_t.c.definition_name, trials_t.c.dataset, trials_t.c.settings)
                             .distinct()).all()
    except Exception as exc:  # noqa: BLE001
        print(f"couldn't read the trials register: {exc!r}")
        return 0
    studies = sorted({(r.definition_name, r.dataset, r.settings) for r in rows
                      if str(r.dataset or "").lower().startswith(perp_venues)})
    print(f"# Trials register entries on perpetual venues ({len(studies)}): re-run each from the Research page\n")
    for name, dataset, settings in studies:
        print(f"- {name} on {dataset}: {settings}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
