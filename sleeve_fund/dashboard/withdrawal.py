"""The fund's limits in use against the book after a Strategy is put away (F-C3B-A-UI, a follow-up to QD's F-C3B-A).
Display only.

Archiving, retiring or a clean slate takes the Strategy's equity out of the fund's book as a flow (Store._put_away),
and the portfolio limits are shares of that book, so the same positions use more of each limit afterwards (W10). The
confirm view shows the book after the withdrawal and the limits in use against it; the page after shows them against
the book as it now is. Every figure is the entry gate's own: its last mark less what _put_away takes out, and its
book's gross, net, margin and open risk (portfolio.limits.Book, the one the gate checks an entry against)."""

from __future__ import annotations

from decimal import Decimal

from sleeve_fund.dashboard.riskops import NEAR_LIMIT
from sleeve_fund.portfolio.gate import EMPTY
from sleeve_fund.portfolio.limits import Book
from sleeve_fund.risk import PORTFOLIO
from sleeve_fund.store import WITHDRAWAL_DEFERRED, Store

# The supervisor's word when a flat Strategy's equity couldn't leave the book at its mark (Supervisor._withdraw_flat).
WITHDRAWAL_WAITS = "withdrawal_waits"


def waiting(store: Store, names) -> dict[str, str]:
    """{name: why} for each of `names` not settled flat (Store.withdrawal_blocker): archived now, it stays in the book
    in full and its equity leaves once it is flat (Advisor 14 Oct 10:45 UK), so nothing leaves at archive."""
    out = store.withdrawn()
    return {n: why for n in dict.fromkeys(names) if n not in out and (why := store.withdrawal_blocker(n))}


def leaving(store: Store) -> dict[str, str | None]:
    """{name: the supervisor's word on why it waits, or None} for each Strategy put away and flat whose equity leaves
    the book at the supervisor's next mark (F286-CR1: only the poll withdraws). The word is its latest withdrawal_waits
    since it was put away: a book a day's pause keeps marked, or one left empty while something else can trade."""
    names = [n for n in store.withdrawal_pending() if store.withdrawal_blocker(n) is None]
    if not names:
        return {}
    said: dict[str, dict] = {}
    for e in store.events_of((WITHDRAWAL_DEFERRED, WITHDRAWAL_WAITS), limit=10_000):  # newest first
        said.setdefault(e["sleeve"], e)
    return {n: said[n]["message"] if n in said and said[n]["kind"] == WITHDRAWAL_WAITS else None for n in names}


def _put_away_after_bridge(store: Store, name: str) -> bool:
    """Whether `name` was put away once the fund held the cash (supervisor.bridge_start): only then does a withdrawal
    of 0 mean nothing of its own was in the book, rather than its own capital lost before the bridge (F-WN2-2)."""
    from sleeve_fund.supervisor import bridge_start

    start, at = bridge_start(store), store.archived().get(name)
    return start is not None and at is not None and at >= start


def state_words(store: Store, name: str) -> str:
    """What became of a put-away Strategy's equity, as the PM reads it after "Archived, " or "Retired, "; "" for one
    archived before F-C3B-A (still counted until supervisor withdraw-archived) or a reset's earlier run."""
    if (out := store.withdrawn().get(name)) is not None:
        if out == 0 and _put_away_after_bridge(store, name):  # F-WN-2 (PE2's F-WN-1); F-WN2-2: not a pre-bridge wipe-out
            return "it left with nothing to move: its P&L stays in the fund's cash, which isn't a loss"
        return "its equity left the fund's book, which isn't a loss"
    if name in store.withdrawal_pending():
        if (why := store.withdrawal_blocker(name)) is not None:
            return f"waiting to go flat: {why}. It stays in the fund's book in full until then"
        waits = leaving(store).get(name)
        return ("its equity leaves the fund's book at the supervisor's next mark, which isn't a loss"
                + (f". {waits.rstrip('.')}" if waits else ""))  # the banner adds its own full stop
    if name in (none := store.nothing_withdrawn()):
        return f"nothing to withdraw (equity {none[name]:,.2f}). It stays in the fund's book at that figure"
    return ""


def taken_out(store: Store, names) -> Decimal:
    """What putting `names` away takes out of the fund's book now: each one's in-book equity (Store.in_book_equity),
    as _put_away takes it; none for one already out, one not flat yet (waiting), a saved backtest or a test account."""
    out, held = store.withdrawn(), waiting(store, names)
    return sum((store.in_book_equity(n) for n in dict.fromkeys(names) if n not in out and n not in held), Decimal(0))


def _limit(key: str, words: str, used: Decimal, limit: float, book: Decimal, what: str = "") -> dict:
    share = used / book
    cap = Decimal(repr(limit))  # compared exactly, as the gate compares (limits.decide)
    tone = "loss" if share >= cap else "warn" if share >= cap * Decimal(repr(NEAR_LIMIT)) else ""
    return {"key": key, "words": words, "what": what, "used": float(share), "limit": limit, "tone": tone}


def limits_in_use(store: Store, put_away=()) -> dict:
    """The fund's book after `put_away` leaves it (none: the book as it is) and the limits in use against it, from the
    journal itself (the Store, not the dashboard's float view: the gate's figures are exact):
    {source, book, taken, after, empty, waiting: {name: why}, leaving: {name: word}, limits: [{key, words, what, used,
    limit, tone}], over: [words]}. One of `put_away` not flat yet is in `waiting` and takes nothing out. One put away
    and flat (leaving) counts as gone, as its equity leaves at the supervisor's next mark (F286-CR1): about, as it
    moves with its P&L until then.

    source: "gate" when read from the entry gate; "unmarked" before the supervisor's first mark (the gate then takes
    no entry); "unreadable" when its book couldn't be read (printed). tone: "loss" at or over a limit, where the gate takes no
    entry that adds to it (the withdrawal itself is allowed and closes nothing: Advisor, W10), "warn" from NEAR_LIMIT
    of it."""
    from sleeve_fund.paper.portfolio_gate import PaperGate

    view = {"source": "gate", "book": None, "taken": None, "after": None, "empty": False, "waiting": {}, "leaving": {},
            "limits": [], "over": []}
    try:
        # Put away and flat, its equity leaves at the supervisor's next mark, seconds on (F286-CR1): counted as gone.
        view["leaving"] = leaving(store)
        put_away = [*put_away, *view["leaving"]]
        view["waiting"] = waiting(store, put_away)
        taken = taken_out(store, put_away)
        ledger = PaperGate(store).ledger
        book = ledger.state().equity
        if book is None:
            return {**view, "source": "unmarked", "taken": float(taken)}
        _, held = ledger.positions()
        held = (*held, *(r.holding for r in ledger.reservations()))
    except Exception as exc:  # noqa: BLE001 - a book the gate can't read leaves the line out, said so
        print(f"limits after withdrawal: can't read the entry gate's book: {exc!r}")
        return {**view, "source": "unreadable"}
    after = book - taken
    view.update(book=float(book), taken=float(taken), after=float(after))
    if after < EMPTY:  # what apply_flow empties: no book left to measure a share of
        return {**view, "empty": True}
    gate = Book(after, held)
    nets = sorted(((abs(gate.net(u)), u) for u in {h.underlying for h in held}), reverse=True)
    net, top = nets[0] if nets else (Decimal(0), "")
    limits = [
        _limit("gross", "Total size", gate.gross(), PORTFOLIO.gross, after),
        _limit("net", "Largest net", net, PORTFOLIO.net_instrument, after, top),
        _limit("margin", "Margin", gate.margin(), PORTFOLIO.margin, after),
        _limit("open_risk", "Open risk", gate.open_risk(), PORTFOLIO.open_risk, after),
    ]
    return {**view, "limits": limits, "over": [x["words"] for x in limits if x["tone"] == "loss"]}


def words(view: dict) -> str:
    """The line as plain text, for the command line (supervisor withdraw-archived)."""
    if view["source"] == "unmarked":
        return "The fund's book has no mark yet, so there are no limits in use to show."
    if view["source"] == "unreadable":
        return "The limits in use can't be read now."
    book = (f"will be about {view['after']:,.2f} once {', '.join(view['leaving'])} leave"
            f"{'s' if len(view['leaving']) == 1 else ''} it at the supervisor's next mark" if view["leaving"]
            else f"is now {view['after']:,.2f}")
    if view["empty"]:
        return f"The fund's book {book}: no capital, so nothing can trade until some is added."
    parts = [f"{x['words']}{' (' + x['what'] + ')' if x['what'] else ''} {x['used']:.1%} of {x['limit']:.0%}"
             for x in view["limits"]]
    line = f"The fund's book {book}. Limits in use: " + ", ".join(parts) + "."
    if view["over"]:
        line += (f" At or over the limit: {', '.join(view['over'])}. New entries that add to it are blocked until it is back"
                 " under; nothing open is closed.")
    return line
