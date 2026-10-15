"""CAP5-RESET (Advisor, 14 Oct ~21:35 UK; bridge Drop A): the PM's clean-slate reset of the paper book, then a deposit
into the fund's cash that starts the book afresh.

The only place a fresh start (carry False) comes from: the deposit is a re-base deposit on the fund's flows (DA's
fund_flows, 0025), and the supervisor's poll applies it only onto an empty book (PE2's empty_book_pockets): settled to a
book of exactly the deposit, nothing carried, or refused naming what is left. An ordinary withdrawal never carries that
flag, so withdrawing everything and depositing again still carries the drawdown and any halt (H4). Run from
reset-book.yml, a dry run first, by an operator on the PM's word.

It runs in steps and can be run again at any point (a restart between them loses nothing):
1. the clean slate: every Strategy stopped and archived; one still holding is flattened first (supervisor.clear);
2. once nothing is waiting to leave the book and the book is empty, the re-base deposit is written to the fund's
   flows; the poll applies it in journal order, behind any outflow still pending.
A book that can't be emptied is refused, naming what is left, and nothing is deposited. Once the fund holds the book
(bridge_start), a reset run's cash and an archived Strategy are outside it (PE2). Before that, on the first reset,
they still count: each is taken out as a withdrawal at the next mark (an archive from before F-C3B-A by
withdraw-archived, a reset run's cash by a put-away; W13 keeps runs out of withdraw-archived), so the book reaches
exactly 0 by the journal. One holding a loss no withdrawal can take is refused up front, by name."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from sleeve_fund import supervisor
from sleeve_fund.gate_ledger import DbLedger
from sleeve_fund.store import Store, is_backtest, utcnow

UK = ZoneInfo("Europe/London")
RESET_ACTION = "book_reset_pm"  # the decisions row each step writes, keyed by the slate's id
CLOSE_OUT = "old-book-close-out"  # the key of a close-out's cash rows (Store.close_out_run): once per run, any slate
WAIT, REFUSED, DONE = 2, 1, 0  # the command's exit codes: run again later; refused, nothing more written; deposited


def strategies(store: Store) -> list[str]:
    """The Strategies the clean slate puts away: every one still on the book, not archived or taken out."""
    out, archived = store.withdrawn(), store.archived()
    return [s.name for s in store.sleeves() if s.name not in out and s.name not in archived
            and not is_backtest(s.name) and not supervisor.outside_fund_book(store, s)]


def _holding(store: Store, names) -> list[str]:
    return [f"{n} holds {float(qty):g}" for n in names for qty, flat in [store.lifecycle_flat(n)] if not flat]


def _old_pockets(store: Store) -> dict[str, Decimal]:
    """Before the fund holds the book: what of the old one no clean slate puts away, by name: each archive from before
    F-C3B-A (put away with nothing withdrawn) and each reset run's cash. Empty once bridge_start is set (PE2)."""
    if supervisor.bridge_start(store) is not None:
        return {}
    runs, out, waiting = store.reset_runs(), store.withdrawn(), set(store.withdrawal_pending())
    old = set(store.archived())  # those archived since F-C3B-A are already waiting or out
    left = {}
    for s in store.sleeves():
        if s.name in out or s.name in waiting or is_backtest(s.name) or supervisor.outside_fund_book(store, s):
            continue
        if s.name in runs or s.name in old:
            equity = supervisor.strategy_equity(store, s, runs)
            if equity != 0:
                left[s.name] = equity
    return left


def _close_outs(store: Store) -> list[tuple[str, str, Decimal]]:
    """RESET-NEG (Advisor, 15 Oct 03:10 UK): each reset run of the old book below zero, as (run, the Strategy it was
    reset from, its loss), to be closed out from the old book's pooled cash before the empty-book check. The pooled
    cash is that Strategy's: a run's capital is its Strategy's (split_run), so the loss goes back where the capital came
    from, and the book's total doesn't move. Only while that Strategy is still in the book and carries every loss of
    its runs at or above 0 (a withdrawal is never negative); else the loss stays a pocket no withdrawal takes out, and
    the reset is refused by name (_stuck). A run at 0 has nothing to close: a rerun closes out nothing."""
    pockets, came_from, out = _old_pockets(store), store.reset_from(), store.withdrawn()
    by_name = {s.name: s for s in store.sleeves()}
    losses: dict[str, list[tuple[str, Decimal]]] = {}
    for run, eq in pockets.items():
        into = came_from.get(run)
        if eq < 0 and into in by_name and into not in out and not is_backtest(into) and \
                not supervisor.outside_fund_book(store, by_name[into]):
            losses.setdefault(into, []).append((run, -eq))
    runs = store.reset_runs()
    return [(run, into, loss) for into, each in losses.items()
            if supervisor.strategy_equity(store, by_name[into], runs) >= sum(loss for _, loss in each)
            for run, loss in each]


def _stuck(store: Store, closing: list | None = None) -> list[str]:
    """What the slate can't flatten or take out, so the book could never be empty (CAP5-RESET: refused, named): an
    archived Strategy still holding a position, and an old pocket holding a loss (a withdrawal is never negative) that
    no close-out covers. A Strategy still on the book is not one: the slate flattens it first."""
    out = store.withdrawn()
    closed = {run for run, _, _ in (_close_outs(store) if closing is None else closing)}
    held = _holding(store, [n for n in store.archived() if n not in out and not is_backtest(n)])
    return held + [f"{n} {eq:,.2f}, a loss no withdrawal takes out" for n, eq in _old_pockets(store).items()
                   if eq < 0 and n not in closed]


def _minus(amount: Decimal) -> str:
    return f"\u2212{amount:,.2f}"


def _close_out_words(into: str, loss: Decimal, now: datetime) -> str:
    """The journal's words for one close-out (HoE 03:10 UK): by the Strategy's name only, never the run's, whose id
    names its venue."""
    return (f"Old book close-out: {into} ended at {_minus(loss)} from paper losses before the "
            f"{now.astimezone(UK):%-d %b} restart; covered from the old book's cash. No effect on the old book total.")


def _count_words(n: int) -> str:
    return ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")[n] if n < 10 else str(n)


def _closed_words(book: Decimal, closing: list, deposit: Decimal) -> str:
    """The PM's line for the reset once old losses were closed out (HoE 03:10 UK)."""
    n, total = len(closing), sum((loss for _, _, loss in closing), Decimal(0))
    fresh = f"{deposit:,.2f}".removesuffix(".00")
    which = "retired Strategy that had" if n == 1 else "retired Strategies that had"
    return (f"Old book closed at {book:,.2f}, including {total:,.2f} of losses on {_count_words(n)} {which} run below "
            f"zero; the new fund starts at {fresh}.")


def empty_book_pockets(store: Store) -> list[str]:
    """What keeps the book from being empty, by PE2's one definition (the poll's, which settles or refuses the deposit).
    Empty means nothing left."""
    check = getattr(supervisor, "empty_book_pockets", None)
    if check is None:
        raise NotImplementedError("the fund's re-base (PE2's empty_book_pockets) isn't in this build")
    return check(store)


def pause_words(store: Store, now: datetime) -> str | None:
    """The day's pause in force, in the PM's words: a Strategy put away leaves the book only once it ends at 00:00 UTC
    (apply_flow). None when there is none."""
    st = DbLedger(store, lambda: ()).state()
    if st.paused_until is None or now >= st.paused_until:
        return None
    return (f"The fund is paused for the day ({st.paused}), so what was put away leaves the book only once the pause "
            f"ends at {st.paused_until.astimezone(UK):%H:%M} UK ({st.paused_until:%H:%M} UTC). Nothing is lost by "
            "waiting: run the reset again after then.")


def _done(store: Store, slate: str) -> list[str]:
    return [d["reason"] for d in store.decisions(action=RESET_ACTION, limit=10_000) if d["reason"].startswith(f"{slate}:deposit")]


def _withdrawn_fund(store: Store, slate: str) -> bool:
    return any(d["reason"].startswith(f"{slate}:withdraw-fund") for d in
               store.decisions(action=RESET_ACTION, limit=10_000))


def _fund_leaving(store: Store, pocket: str) -> bool:
    """The fund's own cash, while a withdrawal of it waits on the flows: the poll takes it out before the deposit."""
    return pocket.startswith("the fund's cash and P&L") and any(f["kind"] == "withdraw" for f in store.pending_flows())


def plan(store: Store, slate: str, deposit: Decimal) -> list[str]:
    """The dry run's lines for the public log: what is in the book, what each step would do, and what would hold it.
    Writes nothing."""
    names = strategies(store)
    lines = [f"reset {slate}: {'already deposited' if _done(store, slate) else 'not run'}",
             f"the book now: {supervisor.fund_equity(store):,.2f}",
             f"would put away: {', '.join(names) or 'nothing'}"]
    if waiting := store.withdrawal_pending():  # a put-away still in the book: the deposit waits for it (HoQA R1)
        runs, by_name = store.reset_runs(), {s.name: s for s in store.sleeves()}
        lines.append("waiting to leave the book first, at the supervisor's next mark: " + ", ".join(
            f"{n} {supervisor.strategy_equity(store, by_name[n], runs):,.2f}" for n in waiting))
    if holders := _holding(store, names):
        lines.append(f"would flatten first: {'; '.join(holders)}")
    if closing := _close_outs(store):
        lines.append("would close out first, from the old book's cash (no effect on its total): "
                     + ", ".join(f"{into} {_minus(loss)}" for _, into, loss in closing)
                     + f"; {sum(loss for _, _, loss in closing):,.2f} in all")
    if old := {n: eq for n, eq in _old_pockets(store).items() if eq > 0}:
        lines.append("would take out of the old book at the next mark: "
                     + ", ".join(f"{n} {eq:,.2f}" for n, eq in old.items()))
    if stuck := _stuck(store, closing):
        lines.append(f"would refuse, nothing written: the book can't be emptied while {'; '.join(stuck)}")
    lines.append(f"then deposit {deposit:,.2f} into the fund's cash, once the book is empty, as its fresh start")
    return lines


def run(store: Store, actor: str, commit: str, slate: str, deposit: Decimal,
        now: datetime | None = None) -> tuple[int, list[str]]:
    """One pass of the reset: as far as it can get now. Returns (WAIT, REFUSED or DONE, the lines it logs)."""
    actor, commit, slate = actor.strip(), commit.strip(), slate.strip()
    if not actor or not commit or not slate:
        raise ValueError("an actor, the deployed commit and the reset's id are required")
    if deposit <= 0:
        raise ValueError("the deposit must be more than 0")
    now = now or utcnow()
    if _done(store, slate):
        return DONE, [f"reset {slate}: deposited already, nothing to do"]
    closing = _close_outs(store)
    if stuck := _stuck(store, closing):  # before anything is written: a reset that can't empty the book doesn't start
        return REFUSED, [f"refused, nothing written: the book can't be emptied while {'; '.join(stuck)}"]
    said = []
    if closing:  # RESET-NEG: first, so each Strategy's put-away below takes out what is left of its cash
        book = supervisor.fund_equity(store)
        for run, into, loss in closing:
            store.close_out_run(run, into, loss, CLOSE_OUT, now, actor, RESET_ACTION,
                                f"{slate}:close-out {_close_out_words(into, loss, now)}")
        store.decide(actor, RESET_ACTION, f"{slate}:closed {_closed_words(book, closing, deposit)}", ts=now)
        said.append("closed out from the old book's cash: "
                    + ", ".join(f"{into} {_minus(loss)}" for _, into, loss in closing))
    reason = f"{slate}: PM reset of the paper book, every Strategy put away and {deposit:,.2f} deposited afresh"
    if put_away := supervisor.clear_entries(store, [{"id": slate, "reason": reason}]):
        said.append(f"put away: {', '.join(put_away)}")
    if left := strategies(store):
        return WAIT, said + [f"waiting on a flatten before the slate can finish: {'; '.join(_holding(store, left))}"]
    if old := _old_pockets(store):
        runs = store.reset_runs()
        queued = supervisor.withdraw_archived(store, actor, commit) if any(n not in runs for n in old) else []
        for name in old:
            if name in runs:
                with store._put_away(name, now, f"{slate}: a reset run's cash leaves the old book with the PM's reset"):
                    pass
                queued.append(name)
        store.decide(actor, RESET_ACTION, f"{slate}:withdraw {', '.join(queued)} at commit {commit}")
        said.append(f"queued to leave the old book at the supervisor's next mark: {', '.join(queued)}")
    if waiting := store.withdrawal_pending():
        lines = said + [f"waiting for the supervisor's next mark to take out: {', '.join(waiting)}"]
        if (paused := pause_words(store, now)) is not None:
            lines.append(paused)
        return WAIT, lines
    if supervisor.bridge_start(store) is not None and not _withdrawn_fund(store, slate):
        # The fund holds the book (a reset after the first): its cash and P&L leave as one withdrawal on the fund's
        # flows, journalled ahead of the deposit, so the poll applies it first (in id order) and the re-base lands on 0.
        book = supervisor.fund_book(store).total
        if book < 0:
            return REFUSED, said + [f"refused, nothing more written: the fund's book is {book:,.2f}, a loss no "
                                    "withdrawal takes out"]
        if book > 0:
            flow = store.fund_flow("withdraw", book, actor="pm", reason=f"{slate}: the fund's cash leaves with the "
                                   "PM's reset, ahead of its fresh deposit", now=now)
            store.decide(actor, RESET_ACTION, f"{slate}:withdraw-fund {book:,.2f} as fund flow {flow} at commit {commit}")
            said.append(f"withdrawal of the fund's {book:,.2f} written as fund flow {flow}, ahead of the deposit")
    if left := [p for p in empty_book_pockets(store) if not _fund_leaving(store, p)]:
        return REFUSED, said + [f"refused, no deposit written: the book isn't empty; still in it: {'; '.join(left)}"]
    flow = store.fund_flow("deposit", deposit, actor="pm", reason=reason, rebase=True, now=now)
    store.decide(actor, RESET_ACTION, f"{slate}:deposit {deposit:,.2f} as fund flow {flow} at commit {commit}")
    closed = [d["reason"].removeprefix(f"{slate}:closed ") for d in store.decisions(action=RESET_ACTION, limit=10_000)
              if d["reason"].startswith(f"{slate}:closed ")]
    return DONE, said + [f"deposit {deposit:,.2f} written as fund flow {flow}: the supervisor's next mark applies it",
                         *closed[:1]]


def main(args) -> int:
    store = supervisor.Store(portfolio_gate=True)  # the supervisor's, so a test's stand-in reaches it
    deposit = Decimal(args.deposit)
    if args.dry_run:
        lines = plan(store, args.id, deposit)
        print("dry run, nothing written:", *lines, sep="\n")
        return REFUSED if any(line.startswith("would refuse") for line in lines) else DONE
    code, lines = run(store, args.actor, args.commit, args.id, deposit)
    print(*lines, sep="\n")
    return code


__all__ = ["DONE", "REFUSED", "WAIT", "empty_book_pockets", "main", "pause_words", "plan", "run", "strategies"]
