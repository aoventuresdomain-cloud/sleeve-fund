"""Research's Development tab and result page: each model's study plan (research's recommended settings,
the question it answers and its kill rule), its status in plain words, and a tear sheet read back as a
verdict, four figures and the cost ladder.

Layout and copy only: nothing here changes how a study runs or how G1 is judged. Plans are proposed by
the researcher and decided by the PM (standing rule); a model without one falls back to the study form's
standing defaults."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import markdown

# Research's recommended settings per model, with why, the question it answers and the pre-registered kill
# rule. rsi_cross: strategy sprint, run 1 (research/strategy-sprint/run-1-rsi-15m.md, PM 5 Oct 2026).
PLANS: dict[str, dict] = {
    "rsi_cross": {
        "label": "RSI cross-back",
        "question": "Does buying after a sharp 15-minute dip make money after fees?",
        "kill_rule": "if break-even is under 0.05% per side, or it loses on 2 or more of the 4 instruments.",
        "why": ("15 minutes trades often enough to judge in a year; the sealed year is opened once, after "
                "settings are frozen. Start with BTC/USDT: the deepest book, so a fail there kills the idea."),
        "defaults": {"venue": "binance", "instrument": "BTC/USDT", "minutes": 15, "train_days": 730,
                     "test_days": 365, "holdout_days": 365, "use_holdout": False, "risk_profile": "balanced",
                     "stop_atr": 2, "atr_bars": 14},
    },
}
# The study form's standing defaults, for a model research hasn't given settings yet.
FALLBACK = {"minutes": 1440, "train_days": 365, "test_days": 180, "holdout_days": 365, "use_holdout": False,
            "risk_profile": "balanced"}
GENERIC_WHY = ("The study form's standing settings: research hasn't recommended settings for this model yet, "
               "so check them before you run it.")
GENERIC_KILL = "if G1 fails: the out-of-sample Sharpe after fees doesn't clearly beat buy-and-hold."
ACRONYMS = {"rsi", "ema", "sma", "atr", "macd"}
EXIT_KEYS = ("stop_loss_pct", "stop_atr", "atr_bars", "stop_swing_bars", "take_profit_pct", "take_profit_r",
             "risk_per_trade_pct")
FORM_KEYS = ("minutes", "train_days", "test_days", "holdout_days", "risk_profile") + EXIT_KEYS


def label(name: str) -> str:
    """A model's display name: the plan's, else its code name in words ("rsi_bands" -> "RSI bands")."""
    if name in PLANS and PLANS[name].get("label"):
        return PLANS[name]["label"]
    words = name.split("_")
    out = [w.upper() if w in ACRONYMS else w for w in words]
    if out and out[0] == words[0]:
        out[0] = out[0].capitalize()
    return " ".join(out)


def _short(text: str, limit: int = 96) -> str:
    """The first sentence of a spec's idea, cut at a word within `limit` characters."""
    first = re.split(r"(?<=[.?!])\s", text.strip(), maxsplit=1)[0]
    if len(first) <= limit:
        return first
    return first[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def plan(name: str, spec) -> dict:
    """The study plan for a model: form values (research's, else the standing defaults), and the words."""
    p = PLANS.get(name)
    values = dict(FALLBACK)
    if p:
        values.update(p["defaults"])
    return {"name": name, "label": label(name), "recommended": bool(p),
            "question": p["question"] if p else _short(spec.idea),
            # The header's "Answers:" line: the plan's question, else G1's own question about the idea.
            "answers": p["question"] if p else f"Does it beat buy-and-hold after fees, out of sample? The idea: {_short(spec.idea)}",
            "kill_rule": p["kill_rule"] if p else GENERIC_KILL,
            "why": p["why"] if p else GENERIC_WHY, "values": values}


def status(row: dict, recommended: bool) -> tuple[str, str, str]:
    """(key, label, chip tone) for a model's card, from the pipeline's G1 (the newest real-data verdict):
    Passed G1, Killed (a FAIL), Not judged; untested, Ready when research has given it settings to run,
    else Not tested."""
    g1 = row.get("g1")
    if g1 == "PASS":
        return "passed", "Passed G1", "running"
    if g1 == "FAIL":
        return "killed", "Killed", "halted"
    if g1 == "NOT JUDGED":
        return "unjudged", "Not judged", "paused"
    if recommended:
        return "ready", "Ready", "running"
    return "untested", "Not tested", ""


def form_values(chosen: dict, pre: dict) -> dict:
    """What the study form shows for the chosen model: its plan, or what came back on the address (a running
    study's settings, or a form that was refused) when that is for this model. A form sent back in full
    replaces the plan; just a model, venue or instrument (a link, or a history request) only picks those."""
    values = dict(chosen["values"])
    if pre.get("strategy") not in (None, "", chosen["name"]):
        return values
    if any(k in pre for k in ("minutes", "train_days", "test_days", "holdout_days")):
        for k in FORM_KEYS:
            values.pop(k, None)
        values.update({k: v for k, v in pre.items() if k in FORM_KEYS and v not in (None, "")})
        values["use_holdout"] = pre.get("use_holdout") == "on"
    for k in ("venue", "instrument"):
        if pre.get(k):
            values[k] = str(pre[k])
    return values


def candle_words(minutes) -> str:
    m = int(minutes)
    return "Daily" if m == 1440 else f"{m // 60}-hour" if m % 60 == 0 else f"{m}-minute"


def _span(days: int, next_: bool = False) -> str:
    if days % 365 == 0:
        n = days // 365
        return ("the next year" if next_ else "1 year") if n == 1 else f"{'the next ' if next_ else ''}{n} years"
    return f"{'the next ' if next_ else ''}{days} days"


def how_sentence(v: dict) -> str:
    """Step 2 in one sentence: "15-minute candles. Learn on 2 years, test on the next year. Latest year sealed."
    console.js's twin is in research.html (howSentence); keep them in step."""
    hold = int(v.get("holdout_days") or 0)
    held = "year" if hold == 365 else f"{hold} days"
    sealed = ("Nothing held back." if not hold else f"Latest {held} opened for this read-out."
              if v.get("use_holdout") in (True, "on") else f"Latest {held} sealed.")
    return (f"{candle_words(v.get('minutes') or 1440)} candles. Learn on {_span(int(v.get('train_days') or 365))}, "
            f"test on {_span(int(v.get('test_days') or 180), True)}. {sealed}")


def _num(x) -> str:
    f = float(x)
    return str(int(f)) if f.is_integer() else f"{f:g}"


def exits_sentence(v: dict) -> str:
    """Step 3 in one sentence: the stop, the target and the risk profile (twin: exitsSentence in research.html)."""
    if v.get("stop_atr"):
        stop = (f"Stop {_num(v['stop_atr'])} average true ranges below entry "
                f"({_num(v.get('atr_bars') or 14)} bars).")
    elif v.get("stop_loss_pct"):
        stop = f"Stop {_num(v['stop_loss_pct'])}% below entry."
    elif v.get("stop_swing_bars"):
        stop = f"Stop at the lowest low of the last {_num(v['stop_swing_bars'])} bars."
    else:
        stop = "No stop."
    if v.get("take_profit_pct"):
        target = f"Target {_num(v['take_profit_pct'])}% above entry."
    elif v.get("take_profit_r"):
        target = f"Target {_num(v['take_profit_r'])}R after costs."
    else:
        target = "No target."
    parts = [stop, target]
    if v.get("risk_per_trade_pct"):
        parts.append(f"Risk {_num(v['risk_per_trade_pct'])}% of capital a trade.")
    prof = v.get("risk_profile") or "balanced"
    parts.append("No risk profile." if prof == "none" else f"{prof.capitalize()} profile.")
    return " ".join(parts)


def history_chip(h: dict) -> dict:
    """The chip beside a study's instrument: how much is stored and whether the collector is current."""
    if h["first"] is None:
        when = h.get("requested")
        return {"text": f"Asked for {when:%d %b}, nothing stored yet" if when else "Asked for, nothing stored yet",
                "tone": "paused"}
    if h["state"] != "current":
        return {"text": f"Catching up, from {h['first']:%d %b %Y}", "tone": "paused",
                "days": (h["last"] - h["first"]).days if h.get("last") else 0}
    days = (h["last"] - h["first"]).days
    span = f"{days / 365.25:.1f} years" if days >= 365 else f"{days} day{'s' if days != 1 else ''}"
    return {"text": f"{span} stored · current", "tone": "running", "days": days}


MONTH = 30.44  # days, for the study-window sentence


def months_words(days: float) -> str:
    """2 years 4 months; 7 months; 20 days. Twin: monthsWords in research.html."""
    m = round(days / MONTH)
    if m < 1:
        return f"{int(days)} day{'s' if int(days) != 1 else ''}"
    y, m = divmod(m, 12)
    parts = ([f"{y} year{'s' if y != 1 else ''}"] if y else []) + ([f"{m} month{'s' if m != 1 else ''}"] if m else [])
    return " ".join(parts)


def fit_windows(v: dict, days: int | None) -> tuple[dict, str]:
    """Study windows sized to the stored history (UI v2, item 8): the plan's learn, test and sealed days when
    they fit inside `days`, else the history split 2:1:1 (2:1 with nothing sealed). Returns the new windows and
    the sentence the page shows. Twin: fitWindows in research.html."""
    if not days:
        return {}, ""
    train, test, hold = (int(v.get(k) or d) for k, d in (("train_days", 365), ("test_days", 180), ("holdout_days", 0)))
    have = months_words(days)
    if train + test + hold <= days:
        return {}, f"You have {have} of history; these windows fit inside it."
    unit = days / (4 if hold else 3)
    out = {"train_days": max(int(2 * unit), 30), "test_days": max(int(unit), 30), "holdout_days": int(unit) if hold else 0}
    words = (f"learn on {months_words(out['train_days'])}, test on the next {months_words(out['test_days'])}"
             + (f", newest {months_words(out['holdout_days'])} sealed" if hold else ""))
    return out, f"You have {have} of history, so: {words}."


def variants(spec) -> int:
    from sleeve_fund.research.study import grid

    return len(grid(spec.param_grid))


def venue_label(code: str | None) -> str | None:
    if not code:
        return None
    from sleeve_fund.venues import venue

    try:
        return venue(code).label
    except ValueError:
        return code


# --- a tear sheet read back ------------------------------------------------------------------------
_VENUE = re.compile(r"^Tested on `[^`]+` at \d+-minute bars on `([^`]+)`", re.M)
_FEES = re.compile(r"^Dataset .*?fees: (.*)$", re.M)
_TAKER = re.compile(r"([\d.]+)% taker")
_BREAKEVEN = re.compile(r"^\*\*Break-even fee:\*\* (.+?)\.?$", re.M)
_ABOUT = re.compile(r"about ([\d.]+)% per side")
_OOS = re.compile(r"^\| Walk-forward out-of-sample \((\d+) folds?, (\d+) days\) \| ([^|]+)\| ([^|]+)\| ([^|]+)\| ([^|]+)\|",
                  re.M)
_TRADES = re.compile(r"^\| Enough out-of-sample trades to judge \| [^|]+ \| (?:not judged: )?(\d+) closed", re.M)
_RUNG = re.compile(r"^\| ([\d.]+)% \| ([^|]+) \| ([^|]+) \| (\d+) \| ([^|]+) \|", re.M)
_SLIP = re.compile(r"plus ([\d.]+)% slippage")
_MEANS = re.compile(r"^## What it means\s*$(.*?)(?=^## |\Z)", re.M | re.S)
_STAMP = re.compile(r"_(\d{8})-(\d{6})(?:-\d+)?$")


def _pct(s: str) -> float | None:
    s = s.strip().replace("−", "-")
    try:
        return float(s.rstrip("%")) / 100
    except ValueError:
        return None


def _float(s: str) -> float | None:
    try:
        return float(s.strip())
    except ValueError:
        return None


def verdict_word(g1: str | None) -> tuple[str, str]:
    """The result's word and its chip tone: Pass, Kill (a G1 FAIL) or Not judged."""
    return {"PASS": ("Pass", "running"), "FAIL": ("Kill", "halted")}.get(g1 or "", ("Not judged", "paused"))


def read_sheet(path: Path) -> dict:
    """What a tear sheet says, for the Results tab and the result page. Older sheets miss some of it, which
    shows as None."""
    from datetime import datetime, timezone

    from sleeve_fund.dashboard.pipeline import sheet_facts

    text = path.read_text(encoding="utf-8")
    f = sheet_facts(path)
    venue = _VENUE.search(text)
    fees = _FEES.search(text)
    taker = _TAKER.search(fees.group(1)) if fees else None
    be = _BREAKEVEN.search(text)
    oos = _OOS.search(text)
    trades = _TRADES.search(text)
    ladder_at = text.find("## Cost ladder")
    ladder_text = text[ladder_at:text.find("\n## ", ladder_at + 3)] if ladder_at >= 0 else ""
    rungs = [{"fee": float(m.group(1)) / 100, "ret": _pct(m.group(2)), "ret_text": m.group(2).strip(),
              "sharpe": m.group(3).strip(), "trips": int(m.group(4))} for m in _RUNG.finditer(ladder_text)]
    slip = _SLIP.search(ladder_text)
    means = _MEANS.search(text)
    stamp = _STAMP.search(path.stem)
    when = (datetime.strptime("".join(stamp.groups()), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc) if stamp
            else datetime.fromtimestamp(f["mtime"], timezone.utc))
    word, tone = verdict_word(f["g1"])
    out = dict(f, venue=venue.group(1) if venue else None, venue_label=venue_label(venue.group(1) if venue else None),
               fee=float(taker.group(1)) / 100 if taker else None, when=when, word=word, tone=tone,
               breakeven_words=be.group(1) if be else None, breakeven=None, breakeven_kind=None,
               rungs=rungs, slippage=float(slip.group(1)) / 100 if slip else None,
               means=markdown.markdown(means.group(1).strip(), extensions=["tables"]) if means and means.group(1).strip()
               else None, label=label(f["strategy"]) if f["strategy"] else path.stem.replace("_", " "))
    if be:
        words = be.group(1)
        about = _ABOUT.search(words)
        if about:
            out.update(breakeven=float(about.group(1)) / 100, breakeven_kind="at")
        elif words.startswith("loses money"):
            out.update(breakeven=0.0, breakeven_kind="none")
        elif words.startswith("still makes money"):
            out.update(breakeven=rungs[-1]["fee"] if rungs else None, breakeven_kind="above")
    if oos:
        out.update(oos_days=int(oos.group(2)), oos_cagr=oos.group(3).strip(), bench_cagr=oos.group(4).strip(),
                   oos_sharpe=oos.group(5).strip(), bench_sharpe=oos.group(6).strip())
    out["oos_trades"] = int(trades.group(1)) if trades else None
    return out


def _p(x: float) -> str:
    return f"{x * 100:.2f}%"


def breakeven_text(s: dict) -> str:
    """The break-even fee as a figure: "0.03%", "none" (loses even with no fees), "over 0.80%"."""
    if s["breakeven_kind"] == "at":
        return _p(s["breakeven"])
    if s["breakeven_kind"] == "none":
        return "None"
    if s["breakeven_kind"] == "above":
        return f"Over {_p(s['breakeven'])}"
    return "–"


def banner(s: dict) -> str:
    """The verdict's one plain sentence, break-even first, built from the sheet's own words."""
    fee, who = s.get("fee"), s.get("venue_label") or "The venue"
    kind = s["breakeven_kind"]
    charges = f"{who} charges {_p(fee)}" if fee is not None else None
    if kind == "at":
        lead = f"Break-even fee is {_p(s['breakeven'])} per side."
        if charges:
            lead += (f" {charges}, so fees eat the edge." if s["breakeven"] < fee
                     else f" {charges}, so the edge survives its fee.")
    elif kind == "none":
        lead = "It loses money even with no fees, so there is no edge for fees to eat."
    elif kind == "above":
        lead = f"It still makes money at {_p(s['breakeven'])} per side, the top of the cost ladder."
        if charges:
            lead += f" {charges}."
    else:
        lead = "This tear sheet has no cost ladder, so it names no break-even fee."
    if s["g1"] == "FAIL" and s.get("failed"):
        lead += " G1 failed on: " + "; ".join(x[0].lower() + x[1:] if x[1:2].islower() else x
                                              for x in s["failed"]) + "."
    elif s["g1"] == "NOT JUDGED":
        why = re.sub(r"^not judged: ", "", s.get("evidence") or "", flags=re.I).rstrip(".")
        lead += f" Not judged: {why}." if why else " The study's runs can't be judged."
    elif s["g1"] is None:
        lead += " The sheet carries no G1 checks."
    return lead


def per_day(trades: int | None, days: int | None) -> str:
    if not trades or not days:
        return "no trades" if trades == 0 else ""
    rate = trades / days
    if rate >= 1:
        return f"about {rate:.1f} a day"
    if rate * 7 >= 1:
        return f"about {rate * 7:.1f} a week"
    return f"about {rate * 30.4:.1f} a month"


def ladder_chart(rungs: list[dict], fee: float | None, width: int = 460, height: int = 190) -> dict | None:
    """Bars for the cost ladder's return at each rung, laid out for an inline SVG. A rung far below the others
    (the 0.80% stress rung often is) is drawn cut short and faded, its real value still on its label."""
    rets = [r["ret"] for r in rungs if r["ret"] is not None]
    if not rungs or len(rets) != len(rungs):
        return None
    left, right, top, bottom = 44, 10, 22, 26
    plot_h = height - top - bottom
    mags = sorted(abs(x) for x in rets)
    cap = max(mags[-2] * 3 if len(mags) > 1 else mags[-1], 0.01)
    lo, hi = min(0.0, *[max(x, -cap) for x in rets]), max(0.0, *[min(x, cap) for x in rets])
    span = (hi - lo) or 0.01
    y = lambda v: top + (hi - v) / span * plot_h  # noqa: E731
    zero = y(0)
    slot = (width - left - right) / len(rungs)
    bars = []
    for i, r in enumerate(rungs):
        v = max(min(r["ret"], cap), -cap)
        cut = v != r["ret"]
        y0, y1 = sorted((y(v), zero))
        x = left + i * slot + slot * 0.18
        here = fee is not None and abs(r["fee"] - fee) < 1e-9
        bars.append({"x": round(x, 1), "w": round(slot * 0.64, 1), "y": round(y0, 1), "h": round(max(y1 - y0, 1.5), 1),
                     "cx": round(x + slot * 0.32, 1), "gain": r["ret"] > 0, "cut": cut, "here": here,
                     "value": r["ret_text"], "fee": f"{r['fee'] * 100:.2f}%",
                     "ly": round(y0 - 5 if r["ret"] > 0 else min(y1 + 13, height - bottom - 2), 1)})
    # The venue's fee between two rungs: a dashed marker where it falls, as the ladder's own rungs are a scale.
    marker = None
    if fee is not None and not any(b["here"] for b in bars):
        fees = [r["fee"] for r in rungs]
        for i in range(len(fees) - 1):
            if fees[i] < fee < fees[i + 1]:
                t = (fee - fees[i]) / (fees[i + 1] - fees[i])
                marker = round(bars[i]["cx"] + t * (bars[i + 1]["cx"] - bars[i]["cx"]), 1)
    # Round ticks: 0% and a few steps either side, so the scale reads at a glance.
    step = next(st for st in (0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0) if span / st <= 4)
    uniq = [{"y": round(y(k * step), 1), "label": f"{k * step * 100:+.0f}%" if k else "0%"}
            for k in range(int(lo / step), int(hi / step) + 1)]
    return {"w": width, "h": height, "left": left, "right": width - right, "zero": round(zero, 1), "bars": bars,
            "ticks": uniq, "marker": marker, "base": height - 8, "marker_top": top - 6}


def plans(rows: list[dict], sheets: list[dict]) -> list[dict]:
    """One card per model in the library: its plan, status and a meta line (the last verdict and where)."""
    out = []
    for r in rows:
        spec = r["spec"] if "spec" in r else importlib.import_module(f"sleeve_fund.strategies.{r['name']}").SPEC
        p = plan(r["name"], spec)
        key, word, tone = status(r, p["recommended"])
        real = [s for s in sheets if s["strategy"] == r["name"] and s["dataset"] not in ("synthetic", "unknown")
                and "synthetic" not in s["dataset"] and s["g1"]]
        last = real[0] if real else None
        if last:
            where = ", ".join(x for x in (last.get("instrument"), last.get("venue_label")) if x)
            meta = f"{last['word']}{' on ' + where if where else ''}"
            if last.get("breakeven_kind") == "at":
                meta += f" · break-even {_p(last['breakeven'])}"
        else:
            meta = "synthetic data only" if r.get("sheets") else "not tested yet"
        out.append(dict(p, row=r, family=spec.family, status=key, status_word=word, tone=tone, meta=meta,
                        studies=len(real), variants=variants(spec), last=last))
    order = {"ready": 0, "passed": 1, "unjudged": 2, "untested": 3, "killed": 4}
    return sorted(out, key=lambda c: (order[c["status"]], c["name"]))
