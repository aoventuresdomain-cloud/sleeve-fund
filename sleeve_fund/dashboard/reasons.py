"""Why the PM took an action: one short list per action, plus Other with a written note.

Shared by every page that asks for a reason (the strategy page's dialogs, and the book-level Flatten on
Risk & health): templates/_reasons.html draws the picker from ACTION_REASONS, and the handlers turn what was
picked into the one `reason` string the decision log already stores, "<picked>" or "<picked>: <note>"
(PM, 5 Oct 2026, combined build F4). The lists are for filtering and counting decisions later, so each
action has its own.
"""

from __future__ import annotations

OTHER = "Other"
NOTE_MIN = 10  # characters an Other note needs before Confirm is enabled

# Each action's reasons as (reason, what it means), in the order the dialog lists them.
ACTION_REASONS: dict[str, list[tuple[str, str]]] = {
    "pause": [
        ("Market event ahead", "A data release, a venue announcement or unusual volatility"),
        ("Data or venue problem", "Feed lag, rejected orders or an outage"),
        ("Reviewing a recent trade", "Hold new entries until you've checked it"),
        ("Settings change coming", "Pause, change the settings, then resume"),
        ("Too much risk across the book", "Cut new exposure while other strategies hold positions"),
    ],
    "resume": [
        ("Issue resolved", "The data, venue or order problem is fixed"),
        ("Event passed", "The market event is over"),
        ("Review done", "The trade you checked was correct"),
        ("Settings changed", "The new settings are saved"),
    ],
    "stop": [
        ("Test finished", "It ran as long as planned"),
        ("Kill rule hit", "The rule set before the test says stop"),
        ("Replaced by a new version", "A clone with new settings takes over"),
        ("Not behaving like its backtest", "Trades or returns are outside the expected range"),
        ("Long-lasting venue or data problem", "It can't trade properly for now"),
    ],
    # Flatten, Close position and the book-level Flatten share one list.
    "flatten": [
        ("Stop or risk limit reached", "A stop, the drawdown or the daily-loss line is close"),
        ("Market event", "Get flat before or during an event"),
        ("Reducing exposure", "Take risk off on purpose"),
        ("Position looks wrong", "A suspected bug, a bad fill or the wrong size"),
        ("Data or venue problem", "Feed lag, rejected orders or an outage"),
    ],
    "start": [
        ("New test", "First run with these settings"),
        ("Restart after a fix", "The problem that stopped it is fixed"),
        ("Restart after review", "The review found nothing wrong"),
    ],
    "archive": [
        ("Test finished, results recorded", "The tear sheet and trades are kept in Records"),
        ("Failed G1 or its kill rule", "Not worth running again"),
        ("Superseded by a new version", "A newer clone replaces it"),
    ],
    "restore": [
        ("Running it again", "Back in the list, ready to start"),
        ("Archived by mistake", "It shouldn't have been put away"),
    ],
    "save": [
        ("Research recommendation", "The researcher proposed these settings"),
        ("Fix after reviewing trades", "A trade review showed the setting was wrong"),
        ("Change the risk level", "More or less risk on purpose"),
        ("Match the backtest", "Make paper use the settings that were tested"),
    ],
    "move": [
        ("Separate the books", "Keep this test apart from the others"),
        ("Move onto the demo mirror", "Check fills against the demo account"),
    ],
    # Reset strategy: flatten, put the run away under Previous book, restart at starting capital (PM, 5 Oct 2026).
    "reset": [
        ("Test finished; starting a clean run", "Keep the run so far under Previous book and begin again"),
        ("Settings changed; a fresh run to compare", "Compare the new settings from a clean start"),
        ("Demo copy out of line with paper", "Set paper and the demo account level again"),
        ("After a fix to the engine or the mirror", "Start over on the fixed code"),
    ],
}
ACTION_REASONS["close"] = ACTION_REASONS["flatten"]
ACTION_REASONS["book_flatten"] = ACTION_REASONS["flatten"]


def compose(action: str, picked: str, note: str = "") -> str:
    """The reason as the decision log stores it: "<picked>" or "<picked>: <note>". Raises ValueError, in
    words for the page, when nothing was picked, the pick isn't one of the action's, or Other has no note
    of NOTE_MIN characters."""
    picked, note = (picked or "").strip(), " ".join((note or "").split())
    choices = [r for r, _ in ACTION_REASONS.get(action, [])]
    if not picked:
        raise ValueError("pick a reason")
    if picked == OTHER:
        if len(note) < NOTE_MIN:
            raise ValueError(f"Other needs a note of at least {NOTE_MIN} characters")
        return f"{OTHER}: {note}"
    if picked not in choices:
        raise ValueError(f"{picked!r} isn't one of the reasons for this action")
    return f"{picked}: {note}" if note else picked


def from_form(action: str, form) -> str:
    """The reason from a submitted form: the picker's reason_pick and reason_note when the form has them,
    else its plain `reason` field as before (older forms and scripts keep working). Empty if neither."""
    if form.get("reason_pick") is not None:
        return compose(action, str(form.get("reason_pick", "")), str(form.get("reason_note", "")))
    return str(form.get("reason", "") or "").strip()
