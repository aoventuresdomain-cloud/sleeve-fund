"""PM dashboard v1: see the book, start and stop paper sleeves, pause/flatten/resume,
read tear sheets and the decision log.

Read-mostly. Every control that changes a sleeve asks for a reason and lands in
the decision log. Protected by one PM password (HTTP Basic, behind HTTPS).

    DASHBOARD_PASSWORD=... uvicorn --factory sleeve_fund.dashboard.app:create_app
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import secrets
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode, urlparse

import markdown
from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sleeve_fund import liquidation, markets
from sleeve_fund.dashboard import book as bookm
from sleeve_fund.dashboard import development as dev
from sleeve_fund.dashboard import gates, reasons, reports, riskops, trading
from sleeve_fund.dashboard.jobs import Jobs
from sleeve_fund.dashboard.metrics import STALE, sleeve_summary
from sleeve_fund.data import spec_minutes
from sleeve_fund.fees import resolve as resolve_fees
from sleeve_fund.history import REQUEST_YEARS
from sleeve_fund.instruments import price_decimals
from sleeve_fund.spreads import resolve as resolve_spread
from sleeve_fund.paper.config import (
    ALLOWED_BAR_SPECS,
    VENUE_WARMUP_BARS,
    SleeveConfig,
    auto_warmup,
    check_hub_bar_spec,
    to_store_kwargs,
)
from sleeve_fund.research import run as study_run
from sleeve_fund.research.ledger import IdeaLedger, opened_words
from sleeve_fund.research.holdout import HoldoutLocks
from sleeve_fund.research.trials import TrialsRegister
from sleeve_fund.risk import PROFILES
from sleeve_fund.store import BACKTEST_PREFIX, Store, is_backtest, utcnow
from sleeve_fund.paper.runtime import RESUMABLE, entry_blocked
from sleeve_fund.strategies import REGISTRY, check_perp_sizing, check_perp_stop
from sleeve_fund.strategies.base import exit_warmup, maker_orders_enabled
from sleeve_fund.wording import no_venues

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TEARSHEETS = study_run.TEARSHEETS
LEDGER = study_run.LEDGER
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,40}$")
VERSION = os.environ.get("APP_VERSION", "dev")[:12]
ORDER_HISTORY_ROWS = 15  # the strategy page's Order history; the Blotter has the rest

security = HTTPBasic(realm="Multi-Strategy Fund")


def _password() -> str:
    pw = os.environ.get("DASHBOARD_PASSWORD", "")
    if not pw and os.environ.get("DASHBOARD_INSECURE_DEV") != "1":
        raise RuntimeError("DASHBOARD_PASSWORD must be set (or DASHBOARD_INSECURE_DEV=1 for local dev)")
    return pw


def require_pm(creds: HTTPBasicCredentials = Depends(security)) -> str:
    pw = _password()
    if pw and not (
        secrets.compare_digest(creds.username.encode(), b"pm")
        and secrets.compare_digest(creds.password.encode(), pw.encode())
    ):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "wrong user or password",
                            headers={"WWW-Authenticate": 'Basic realm="Multi-Strategy Fund"'})
    return "PM"


def same_origin(request: Request) -> None:
    """Basic auth is sent automatically by browsers, so block cross-site form posts."""
    origin = request.headers.get("origin") or request.headers.get("referer")
    if not origin or urlparse(origin).netloc != request.headers.get("host"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "cross-site request blocked")


def create_app(store: Store | None = None) -> FastAPI:
    _password()  # fail at start-up, not on first request
    app = FastAPI(title="Multi-Strategy Fund", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store = store or Store()
    app.state.jobs = Jobs()
    try:
        study_run.seed(LEDGER, TEARSHEETS)
    except OSError as exc:  # the pages still work; research shows what it has
        logging.getLogger(__name__).warning(f"couldn't bring the repository's research into {TEARSHEETS}: {exc}")
    try:  # the idea counter folded into the trials register and the holdout locks; safe to repeat on every start
        TrialsRegister(app.state.store).import_ledger(LEDGER)
        HoldoutLocks(app.state.store).import_ledger(LEDGER)
    except (OSError, ValueError, KeyError) as exc:
        logging.getLogger(__name__).warning(f"couldn't fold the idea counter into the trials register: {exc}")
    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.globals["maker_enabled"] = maker_orders_enabled
    templates.env.globals["market_choices"] = market_choices
    templates.env.globals["venue_choices"] = venue_choices
    templates.env.globals["venue_label"] = dev.venue_label  # "perpetual" or "spot": never the venue's name (QA U8)
    templates.env.globals["exit_ways"] = trading.exit_ways
    templates.env.filters["no_venues"] = no_venues  # stored reasons and messages name no venue (QA U18)
    templates.env.filters["pct"] = lambda x: f"{x:+.2%}"
    templates.env.filters["pct0"] = lambda x: f"{x:.0%}"
    templates.env.filters["money"] = lambda x: f"{x:,.2f}"
    # NaN-safe versions for stats that don't exist until a trade has closed.
    templates.env.filters["pctn"] = lambda x: "n/a" if x != x else f"{x:+.1%}"
    templates.env.filters["numn"] = lambda x: "n/a" if x != x else ("∞" if x == float("inf") else f"{x:.2f}")
    templates.env.filters["pct1"] = lambda x: f"{x:.1%}"
    templates.env.filters["pctn0"] = lambda x: "n/a" if x != x else f"{x:.0%}"
    templates.env.filters["smoney"] = lambda x: ("+" if x > 0.005 else ("−" if x < -0.005 else "")) + f"{abs(x):,.2f}"
    templates.env.filters["pct1s"] = lambda x: f"{x:+.1%}"
    templates.env.filters["smoney0"] = lambda x: ("+" if x >= 0.5 else ("−" if x <= -0.5 else "")) + f"{abs(x):,.0f}"
    templates.env.filters["ago"] = _ago
    templates.env.filters["held"] = _held
    templates.env.filters["bytes"] = _bytes
    templates.env.filters["px"] = lambda x: "n/a" if x is None or x != x else f"{x:,.{price_decimals(x)}f}"
    templates.env.filters["qty"] = _qty
    templates.env.globals["bar_label"] = _bar_label
    templates.env.globals["bar_short"] = _bar_short
    templates.env.globals["bar_choices"] = _bar_choices
    from sleeve_fund.dashboard.glossary import GLOSSARY

    templates.env.globals["glossary"] = GLOSSARY
    templates.env.globals["action_reasons"] = reasons.ACTION_REASONS  # templates/_reasons.html
    templates.env.globals["reason_note_min"] = reasons.NOTE_MIN
    templates.env.filters["action_words"] = action_words
    templates.env.filters["kind_words"] = kind_words
    templates.env.filters["rmult"] = lambda r: "–" if r is None else f"{r:+.2f}R"
    # The year only when it isn't this one, as a backtest's or an old journal's dates need it.
    templates.env.filters["ts"] = lambda t: (t.strftime("%d %b %H:%M UTC" if t.year == utcnow().year
                                                        else "%d %b %Y %H:%M UTC") if t else "never")
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    def st() -> Store:
        return app.state.store

    def recent_reasons() -> list[dict]:
        """The PM's own reasons, newest first, for every reason picker's "You used recently" (UI v2, item 9)."""
        try:
            return reasons.recent(st().decisions(limit=200))
        except Exception as exc:  # noqa: BLE001 - an unreadable log leaves the group out, not the dialog
            logging.getLogger(__name__).warning(f"couldn't read the recent reasons: {exc!r}")
            return []

    templates.env.globals["recent_reasons"] = recent_reasons

    def current_sleeves():
        """The current book's strategies: every one except those an earlier clean slate put away."""
        earlier = st().previous_book()
        return [s for s in st().sleeves() if s.name not in earlier]

    def shell(sleeves=None) -> dict:
        """What the frame shows on every page: mode, health and open alerts."""
        sleeves = current_sleeves() if sleeves is None else sleeves
        now = utcnow()
        wanted = [x for x in sleeves if x.desired_state == "running"]
        put_away = st().archived()
        return {
            "mode": "paper",  # live mode arrives with the G2 work; until then nothing can trade real money
            "live_locked": True,
            "now": now,
            "version": VERSION,
            "alerts": st().open_alert_count(),
            "total": len(sleeves),
            "running": sum(1 for x in sleeves if x.status == "running"),
            "attention": sum(1 for x in sleeves if x.status in ("halted", "error")),
            "unhealthy": sum(1 for x in wanted if not (x.heartbeat_at and now - x.heartbeat_at < STALE)),
            # The rail lists every strategy not put away, so each is one click from anywhere.
            "strategies": [{"name": x.name, "status": x.status} for x in sleeves if x.name not in put_away],
        }

    def page(request: Request, name: str, **ctx) -> HTMLResponse:
        ctx.setdefault("shell", shell())
        return templates.TemplateResponse(request, name, ctx)

    def _saved_backtest_summary(name: str | None) -> dict | None:
        """A saved backtest's strategy summary, as the screens use for paper, or None."""
        if not is_backtest(name):
            return None
        try:
            s = st().sleeve(name)
        except KeyError:
            return None
        x = bookm.sleeve_extras(st(), sleeve_summary(st(), s), bookm.daily(st(), name))
        x["run_id"] = name[len(BACKTEST_PREFIX):]
        return x

    def book_data():
        sleeves = current_sleeves()
        frames = {s.name: bookm.daily(st(), s.name) for s in sleeves}
        summaries = [bookm.sleeve_extras(st(), sleeve_summary(st(), s), frames[s.name]) for s in sleeves]
        return sleeves, frames, summaries

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict:
        return {"ok": True}

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request, _: str = Depends(require_pm)):
        sleeves, frames, summaries = book_data()
        put_away = st().archived()
        earlier = st().previous_book()
        names = [s.name for s in sleeves]
        # The bottom tabs: the book's positions, working orders, latest fills and funding, from the same
        # journal the strategy pages read. Every strategy in the book counts, archived ones still holding too.
        working = [trading.order_view(o) for o in st().orders(None, trading.STATUS_TABS["open"][1], limit=200)
                   if o["sleeve"] in names]
        book = bookm.book_view(st(), summaries, frames)
        positions = trading.book_positions(st(), summaries)
        funding = trading.book_funding(st(), summaries)
        # Costs as money paid: fees plus funding paid (funding totals are + received, - paid).
        book["funding_paid"] = -funding["total"]
        book["fees_funding"] = book.get("costs", book["fees"] - funding["total"])
        # Margin used and Open risk, as the position figures compute them (UI v2, item 7), once they exist.
        if "open_risk" in positions:
            book.update(margin_used=positions["margin"], open_risk=positions["open_risk"],
                        unbounded=len(positions["unbounded"]), trailing=len(positions["trailing"]))
        return page(request, "home.html", summaries=[x for x in summaries if x["sleeve"].name not in put_away],
                    archived=[x for x in summaries if x["sleeve"].name in put_away],
                    earlier=[st().sleeve(n) for n in earlier], book_start=st().book_start(),
                    earlier_resets=bool(set(earlier) & set(st().reset_runs())),
                    book=book, alerts=st().alerts(limit=30), shell=shell(sleeves),
                    positions=positions, working=working, holdings=bookm.holdings(positions["rows"], book["equity"]),
                    book_fills=trading.book_fills(st(), sleeves), funding=funding,
                    fill_count=sum(x["fills"] for x in summaries))

    def _recent_json(sleeves, days: int, daily) -> JSONResponse:
        """The last day or week at fine resolution, in the shape the charts read."""
        if days not in bookm.RECENT_STEP:
            raise HTTPException(400, "days must be 1 or 7")
        cutoff = utcnow() - timedelta(days=days)
        prior = daily["equity"][daily.index < cutoff] if len(daily) else daily
        # The peak before the window counts the starting capital too, as the whole history does (M12-F1): from
        # the daily closes alone a wiped-out strategy's was 0, and 0/0 made the range a 500 (mF2-1, mF2-2).
        start = sum(s.starting_balance for s in sleeves)
        curve = bookm.recent_curve(st(), sleeves, days, max(float(prior.max()) if len(prior) else 0.0, start))
        return JSONResponse({
            "res": "intraday",
            "t": [t.isoformat() for t in curve.index],
            "equity": [round(v, 2) for v in curve["equity"]],
            "benchmark": [round(v, 2) for v in curve["benchmark"]],
            "drawdown": [round(v, 5) for v in curve["drawdown"]],
        })

    @app.get("/api/book/equity")
    def book_equity_json(days: int | None = None, _: str = Depends(require_pm)):
        sleeves, frames, summaries = book_data()
        curve = bookm.book_curve(summaries, frames)  # every strategy, as the book figures count it
        if days:
            return _recent_json(sleeves, days, curve)
        return JSONResponse({
            "start": sum(x["sleeve"].starting_balance for x in summaries),  # the chart's baseline for Change
            "t": [t.isoformat() for t in curve.index],
            "equity": [round(v, 2) for v in curve["equity"]],
            "benchmark": [round(v, 2) for v in curve["benchmark"]],
            "drawdown": [round(v, 5) for v in curve["drawdown"]] if len(curve) else [],
        })

    @app.get("/risk", response_class=HTMLResponse)
    def risk_page(request: Request, _: str = Depends(require_pm)):
        sleeves, frames, summaries = book_data()
        book = bookm.book_view(st(), summaries, frames)
        kill = _kill_targets(summaries)
        risk = riskops.risk_view(st(), summaries, book)
        health = riskops.health_view(st(), summaries)
        running = sum(1 for x in summaries if x["sleeve"].status == "running")
        # The drawdown chart's halt line only when the whole book shares one risk profile's limit.
        profiles = {x["profile"].name: x["profile"].max_drawdown for x in summaries}
        halt = next(iter(profiles.items())) if len(profiles) == 1 else None
        chart = riskops.drawdown_chart(bookm.book_curve(summaries, frames), halt=halt[1] if halt else None)
        return page(request, "risk.html", book=book, risk=risk, health=health, chart=chart,
                    halt_name=halt[0] if halt else None, stress=riskops.stress_bars(risk["scenarios"]),
                    status=riskops.status_word(riskops.status_items(risk["rows"], health), running),
                    ops=riskops.ops_view(st(), summaries), shell=shell(sleeves), kill=kill,
                    kill_error=request.query_params.get("kill_error"), killed=request.query_params.get("killed"))

    def _kill_targets(summaries) -> dict:
        """Who the kill switch acts on: every strategy still trading (running and not halted) and every
        one holding a position, stopped or halted included. A halted, flat strategy is left as it is."""
        # One already told to flatten is waiting on its process, not a target again: firing twice would
        # queue a second sale (review round 8, R8-11).
        flattening = [x["sleeve"].name for x in summaries
                      if any(c["command"] == "flatten" for c in st().pending_commands(x["sleeve"].name))]
        live = [x for x in summaries if x["sleeve"].name not in flattening]
        trading = [x for x in live if x["sleeve"].desired_state == "running" and x["sleeve"].status != "halted"]
        held = [x for x in live if x["qty"] and x not in trading]
        # What the switch trades: every open position, longs and shorts, by its size (round 12, M12-U4: shorts
        # were left out, so the dialog showed a seventh of the notional it would trade).
        holding = [x for x in trading + held if x["qty"]]
        return {"trading": trading, "held": held, "all": trading + held, "flattening": flattening,
                "stopped": [x["sleeve"].name for x in held if x["sleeve"].desired_state != "running"],
                "holding": holding, "gross": sum(abs(x["position_value"]) for x in holding),
                "shorts": sum(1 for x in holding if x["qty"] < 0)}

    @app.post("/book/flatten")
    def book_flatten(reason: str = Form(""), reason_pick: str | None = Form(None), reason_note: str = Form(""),
                     actor: str = Depends(require_pm), _o: None = Depends(same_origin)):
        """The book's kill switch: every strategy still trading or holding a position sells to cash at
        market and pauses, each with the PM's reason in its decision log. A stopped strategy holding a
        position is started so its process can sell; it pauses once flat, as the others do."""
        try:
            why = _reason("book_flatten", reason, reason_pick, reason_note)
        except ValueError:
            why = ""
        if not why:
            return RedirectResponse("/risk?kill_error=reason", status_code=303)
        _, _, summaries = book_data()
        targets = _kill_targets(summaries)
        if not targets["all"]:
            # Fired again, or nothing to sell: no decision to log, as nothing was done (review round 9, N1).
            return RedirectResponse("/risk?killed=0", status_code=303)
        for x in targets["all"]:
            name = x["sleeve"].name
            st().command(name, "flatten", f"Book kill switch: {why}", actor=actor)
            if x["sleeve"].desired_state != "running":
                st().set_desired_state(name, "running")
                st().decide(actor, "start", f"Book kill switch: started to sell its position ({why})", name)
        n = len(targets["all"])
        st().decide(actor, "flatten everything", f"{why} ({n} strateg{'y' if n == 1 else 'ies'})")
        return RedirectResponse(f"/risk?killed={n}", status_code=303)

    @app.get("/ops")
    def ops_page(_: str = Depends(require_pm)):
        """Operations is the System tab of Risk & health now."""
        return RedirectResponse("/risk#system", status_code=303)

    @app.get("/alerts", response_class=HTMLResponse)
    def alerts_page(request: Request, _: str = Depends(require_pm), show: str = "open"):
        rows = st().alerts(limit=300, include_acked=(show == "all"))
        return page(request, "alerts.html", alerts=rows, show=show)

    @app.post("/alerts/{event_id}/ack")
    def ack_alert(event_id: int, note: str = Form(""), next: str = Form("/alerts"),
                  actor: str = Depends(require_pm), _o: None = Depends(same_origin)):
        try:
            st().ack(event_id, actor, note)
        except KeyError:
            raise HTTPException(404, "no such alert") from None
        target = next if next.startswith("/") and not next.startswith("//") else "/alerts"
        return RedirectResponse(target, status_code=303)

    @app.get("/sleeves/new", response_class=HTMLResponse)
    def new_sleeve_form(request: Request, _: str = Depends(require_pm), error: str = "", strategy: str = ""):
        from sleeve_fund.dashboard import pipeline

        g1 = {r["name"]: "|".join(r["passed_on"]) for r in pipeline.strategies(TEARSHEETS, st().sleeves())}
        chosen = strategy if strategy in REGISTRY else "trend_filter"
        try:
            hints = _hints(request.query_params.get("venue"))
        except ValueError:
            hints = _hints()
        return page(request, "new_sleeve.html", strategies=_strategy_choices(), instruments=hints,
                    bar_specs=sorted(ALLOWED_BAR_SPECS, key=spec_minutes), profiles=PROFILES, error=error, g1=g1, chosen=chosen,
                    pre=dict(request.query_params), accounts=st().accounts(), costs=exit_costs())

    @app.post("/sleeves/new")
    async def new_sleeve(request: Request, actor: str = Depends(require_pm), _o: None = Depends(same_origin)):
        form = dict(await request.form())
        try:
            name = str(form.get("name", "")).strip()
            reason = reasons.from_form("start", form)
            strategy = str(form.get("strategy", ""))
            if not NAME_RE.match(name):
                raise ValueError("name: lower-case letters, digits and dashes, 2 to 41 characters")
            if not reason:
                raise ValueError("a reason is required")
            _check_listed(st(), str(form.get("venue") or "") or None, str(form.get("instrument", "")))
            params = _form_params(form, strategy)
            tested = str(form.get("tested_bar_spec") or BACKTEST_BAR_SPEC)
            if form.get("from") == "backtest" and form.get("bar_spec") != tested:
                raise ValueError(f"interval: the backtest decided on {_bar_short(tested)}, so this strategy must "
                                 "too; backtest another interval before changing it")
            bar_spec = str(form.get("bar_spec", ""))
            # Blank means automatic: enough history for the model's longest look-back, as a backtest has.
            auto = strategy in REGISTRY and bar_spec in ALLOWED_BAR_SPECS and not str(form.get("warmup_bars", "")).strip()
            warmup = _warmup_for(strategy, form, bar_spec) if auto else int(form.get("warmup_bars", 0) or 0)
            cfg = SleeveConfig(name=name, strategy=strategy, instrument=str(form.get("instrument", "")),
                               bar_spec=bar_spec, starting_balance=float(form.get("starting_balance", 0) or 0),
                               params=params, warmup_bars=warmup, risk_profile=str(form.get("risk_profile", "")),
                               venue=_venue_name(form.get("venue")))
            _check_strategy_params(cfg, resolve_spread(cfg.venue, cfg.instrument, st()).half_spread)
            check_hub_bar_spec(cfg.venue, cfg.bar_spec)
            needed = max(REGISTRY[strategy][0].warmup_needed({**_defaults(strategy), **params}, spec_minutes(bar_spec)),
                         exit_warmup(params))
            if any(s.name == name for s in st().sleeves()):
                raise ValueError(f"a strategy called {name} already exists")
            account = str(form.get("account", "") or "paper")
            kinds = {a["name"]: a["kind"] for a in st().accounts()}
            if account not in kinds:
                raise ValueError(f"account: no account called {account}")
            if _retired(account):
                raise ValueError(f"account: {account} is retired")
            if kinds[account] == "live":  # the shell's live lock: nothing trades real money before G2
                raise ValueError("account: live accounts are locked until G2 is approved; choose a paper account")
            trial, uncounted = _trial(_strategy_trial_metrics, st(), cfg, strategy, params, cfg.risk_profile,
                                      fallback={"strategy": strategy, "params": params, "source": "strategy"})
            _saved("strategy", st().create_sleeve, name=name, strategy=strategy, instrument=cfg.instrument,
                   bar_spec=cfg.bar_spec, starting_balance=cfg.starting_balance, params=params,
                   risk_profile=cfg.risk_profile, warmup_bars=cfg.warmup_bars, venue=to_store_kwargs(cfg)["venue"],
                   trial=trial)
            st().assign_account(name, account)
            st().decide(actor, "create", reason, name)
            _uncounted_event(st(), name, f"strategy {name}", uncounted)
            if needed > cfg.warmup_bars:
                # A warning (an alert) when the most that can load falls short; a note when it was chosen.
                st().event(name, "warning" if auto else "info", "warmup_short",
                           f"The model's longest look-back needs {needed:,} bars of history but {cfg.warmup_bars:,} "
                           f"load at start, so until {needed:,} live bars have passed its signals can differ from "
                           "a backtest's.")
        except (ValueError, TypeError) as exc:
            # Send the form back filled in, so a typo doesn't cost the PM everything they entered.
            kept = {k: str(v) for k, v in form.items() if isinstance(v, str) and v}
            return RedirectResponse(f"/sleeves/new?{urlencode({'error': str(exc), **kept})}", status_code=303)
        return RedirectResponse(f"/sleeves/{name}", status_code=303)

    def _tested(run_id: str | None) -> str:
        """The period a saved backtest replayed, for its strategy screen."""
        if not run_id:
            return ""
        try:
            r = st().backtest(run_id)["result"]
        except KeyError:
            return ""
        return f"{r['from']} to {r['to']}" if r.get("from") and r.get("to") else ""

    @app.get("/sleeves/{name}", response_class=HTMLResponse)
    def sleeve_detail(request: Request, name: str, _: str = Depends(require_pm)):
        try:
            s = st().sleeve(name)
        except KeyError:
            raise HTTPException(404, "no such strategy") from None
        bt_id = name[len(BACKTEST_PREFIX):] if is_backtest(name) else None
        x = bookm.sleeve_extras(st(), sleeve_summary(st(), s), bookm.daily(st(), name))
        fills = st().fills(name, limit=100_000)
        events = st().events(name, limit=400)
        orders = trading.orders_by_id(st(), name)
        plans = st().exit_plans(name)
        perp = markets.is_perp(s.params)
        funding = st().funding(name, limit=100_000) if perp else []
        trips = trading.trips(fills, st().events(name, limit=5000), orders, plans, perp, funding if perp else None,
                              st().insurance(name) if perp else None)
        feed = _feed(events, request.query_params.get("feed", "all"))
        recent = [trading.order_view(o) for o in st().orders(name, limit=ORDER_HISTORY_ROWS)]
        order_total = sum(st().order_counts(name).values())
        position = trading.open_position(x, fills, orders, plans)
        perp_x = trading.perp_view(x, position, funding) if perp else None
        # Its position's margin was lost (liquidated): no Start, Resume or Reset until the PM resets it after
        # liquidation with an incident note (Advisor 6 Oct 17:57 and 20:41), whatever its status meanwhile.
        liquidated = not bt_id and _liquidated(name)
        q = request.query_params
        # The settings form: what was typed when a change was refused, else the settings as they are.
        typed = {k[2:]: v for k, v in q.items() if k.startswith("f_")}
        settings_pre = typed or {"risk_profile": s.risk_profile, **_risk_form(s.params)}
        path = None if bt_id else gates.path_to_live(st(), x, _g1_of(s.strategy, s.instrument, spec_minutes(s.bar_spec),
                                                                     s.params), st().accounts(), utcnow())
        # The Overview's Position table is the Portfolio's, one row; Open orders lists working orders only.
        positions = trading.book_positions(st(), [x])
        working = [trading.order_view(o) for o in st().orders(name, trading.STATUS_TABS["open"][1], limit=200)]
        fees_funding = x.get("costs", x["fees"] - (perp_x["funding_total"] if perp_x else 0.0))  # funding is + received
        return page(request, "sleeve.html", x=x, fills=fills[:200], trips=trips, feed=feed, orders=recent,
                    order_total=order_total, liquidated=liquidated,
                    positions=positions, working=working, fees_funding=fees_funding,
                    timing=None if bt_id else trading.timing_view(st().timings(name)),
                    price_feed=None if bt_id else _price_feed(s, st().last_feed(name)),
                    account=st().account_of(name), accounts=st().accounts(), settings_pre=settings_pre,
                    settings_error=q.get("settings_error", ""), saved=q.get("saved", ""), profiles=PROFILES,
                    command_error=q.get("command_error", ""),
                    # The journal's position, as the flatten and account checks read it: a stopped strategy's
                    # last mark may be older than its last fill.
                    held=0.0 if bt_id else st().journal_book(name, s.starting_balance)["qty"],
                    costs=exit_costs(), reload=st().pending_reload(name),
                    position=position, perp=perp_x,
                    feed_kind=request.query_params.get("feed", "all"), decisions=st().decisions(name, limit=50),
                    pending=st().pending_commands(name), risk=_risk_view(x, position),
                    resetting=None if bt_id else st().pending_reset(name),
                    idea=_idea(s.strategy, s.params), archived=name in st().archived(),
                    signals=None if bt_id else _signals_view(s, st().signal_state(name)),
                    timeline=_timeline(events, st().decisions(name, limit=50)),
                    pos_history=_position_history(events, funding, position),
                    then_stop=q.get("then_stop", ""), done=q.get("done", ""),
                    flatten_waits=any(c["command"] == "flatten" for c in st().pending_commands(name)),
                    clone_qs=_clone_qs(s), backtest_id=bt_id, tested=_tested(bt_id),
                    demo=None if bt_id else _demo_copy(st(), s),
                    strategy_errors=st().strategy_errors(name, since_start=not bt_id),
                    path=path, journey=None if bt_id else _journey(s, x, path, st().mirror_rows(name, limit=200)))

    def _strategy_indicators(name: str) -> list[dict]:
        """The strategy's own indicator values for the chart (P1-3s, agreed shape v2/chart-indicators-shape.md):
        as the platform recorded them, passed on untouched and never recomputed here. The store gives them once
        the Quant Developer's recording lands; until then, or if it fails, the chart has none and still draws."""
        source = getattr(st(), "chart_indicators", None)
        if source is None:
            return []
        try:
            return list(source(name))
        except Exception:  # an overlay must never take the chart down
            logging.getLogger(__name__).exception("chart indicators for %s", name)
            return []

    def _strategy_decisions(name: str) -> list[dict]:
        """Fills and missed entries the strategy recorded (P1-3m): [{kind: fill|missed, side, t, signal_t, price,
        reason, code}], passed on untouched. Empty until the platform's journal read lands, or if it fails."""
        source = getattr(st(), "chart_decisions", None)
        if source is None:
            return []
        try:
            return list(source(name))
        except Exception:  # an overlay must never take the chart down
            logging.getLogger(__name__).exception("chart decisions for %s", name)
            return []

    @app.get("/api/sleeves/{name}/candles")
    def candles_json(name: str, interval: str = "", pair: str = "", _: str = Depends(require_pm)):
        from sleeve_fund.dashboard import charts

        try:
            s = st().sleeve(name)
        except KeyError:
            raise HTTPException(404, "no such strategy") from None
        interval = interval if interval in charts.INTERVALS else charts.default_interval(s.bar_spec)
        minutes = charts.INTERVALS[interval]
        pair = pair.strip().upper()
        if pair and pair != s.instrument and not is_backtest(name):
            # Another security on the same chart, for comparison: the venue's candles, without this strategy's trades.
            if not PAIR_RE.fullmatch(pair):
                raise HTTPException(400, "instrument must look like BASE/QUOTE")
            try:
                df, note = charts.candles(pair, minutes, venue=s.venue), ""
            except (OSError, ValueError, KeyError):
                df, note = charts.from_marks([], minutes), f"The venue has no candles for {pair}."
            data = charts.payload(df, minutes, [], {}, [], "venue")
            data.update(intervals=list(charts.INTERVALS), chosen=interval, pair=pair, home=s.instrument,
                        pairs=_chart_pairs(s.instrument, [b.instrument for b in st().sleeves()]))
            if note:
                data["note"] = note
            return JSONResponse(data)
        try:
            if is_backtest(name):  # the venue's recent candles aren't the replayed period
                raise ValueError("backtest")
            df, source = charts.candles(s.instrument, minutes, venue=s.venue), "venue"
        except (OSError, ValueError):  # venue unreachable or pair unknown: chart the sleeve's own marks
            df, source = charts.from_marks(st().equity_series(name, limit=500_000), minutes), "marks"
        fills = list(reversed(st().fills(name, limit=100_000)))
        orders = trading.orders_by_id(st(), name)
        x = bookm.sleeve_extras(st(), sleeve_summary(st(), s), bookm.daily(st(), name))
        position = trading.open_position(x, list(reversed(fills)), orders, st().exit_plans(name))
        # A live screen shows the latest candles; a backtest's chart covers its whole period.
        data = charts.payload(df, minutes, fills, orders, charts.position_lines(position), source,
                              limit=None if is_backtest(name) else 720)
        data["intervals"], data["chosen"] = list(charts.INTERVALS), interval
        data["indicators"] = _strategy_indicators(name)
        data["decisions"] = _strategy_decisions(name)
        data.update(pair=s.instrument, home=s.instrument, pairs=_chart_pairs(s.instrument, [b.instrument for b in st().sleeves()]))
        if is_backtest(name):
            data["note"] = "Candles built from the run's price marks."
        return JSONResponse(data)

    @app.get("/api/instruments")
    def instruments_json(options: bool = False, _: str = Depends(require_pm)):
        from sleeve_fund.dashboard import charts

        if options:  # the forms' instrument pick-list: every venue's listing, grouped, with a history badge
            return JSONResponse({"options": instrument_options(st(), charts.instruments)})
        try:
            return JSONResponse({"instruments": charts.instruments(), "source": "venue"})
        except (OSError, ValueError, KeyError):  # venue unreachable: the usual ones, and any other can still be typed
            return JSONResponse({"instruments": _hints(), "source": "fallback"})

    @app.get("/api/history/coverage")
    def history_coverage_json(venue: str | None = None, _: str = Depends(require_pm)):
        """Each instrument's stored history on a venue, for the pick-list's badges (UI v2, items 9 and 10): first
        and last candle, gaps, and the badge's state and words. An instrument not listed reads "not stored yet"."""
        try:
            profile = _research_venue(venue)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return JSONResponse({"venue": profile.name.lower(), "instruments": [
            {"pair": h["pair"], "first": h["first"].isoformat() if h["first"] is not None else None,
             "last": h["last"].isoformat() if h["last"] is not None else None,
             "gaps": [[a.isoformat(), b.isoformat()] for a, b in h["gaps"]], **h["badge"]}
            for h in _stored_history(st(), profile)]})

    @app.get("/api/sleeves/{name}/equity")
    def equity_json(name: str, days: int | None = None, _: str = Depends(require_pm)):
        try:
            s = st().sleeve(name)
        except KeyError:
            raise HTTPException(404, "no such strategy") from None
        if days:
            resp = _recent_json([s], days, bookm.daily(st(), name))
            body = json.loads(resp.body)
            cutoff = utcnow() - timedelta(days=days)
            body["fills"] = [{"t": f["ts"].isoformat(), "side": f["side"], "qty": f["qty"], "price": f["price"]}
                             for f in st().fills(name, limit=2000) if f["ts"] >= cutoff]
            return JSONResponse(body)
        rows = st().equity_series(name, limit=500_000)
        intraday = bool(rows) and (rows[-1]["ts"] - rows[0]["ts"]).total_seconds() < 3 * 86400
        if intraday:
            step = max(1, len(rows) // 1500)  # keep the chart light
            rows = rows[::step] + ([rows[-1]] if rows and (len(rows) - 1) % step else [])
            t = [r["ts"] for r in rows]
            eq, bench = [r["equity"] for r in rows], [r["benchmark"] for r in rows]
        else:
            frame = bookm.daily(st(), name)
            t, eq, bench = list(frame.index), list(frame["equity"]), list(frame["benchmark"])
        peak, dd = s.starting_balance, []  # from the starting balance too, as the book's (M12-F1)
        for v in eq:
            peak = max(peak, v)
            dd.append(round(1 - v / peak, 5) if peak else 0.0)
        fills = [{"t": f["ts"].isoformat(), "side": f["side"], "qty": f["qty"], "price": f["price"]}
                 for f in st().fills(name, limit=2000)]
        return JSONResponse({
            "res": "intraday" if intraday else "daily",
            "start": s.starting_balance,
            "t": [x.isoformat() for x in t],
            "equity": [round(v, 2) for v in eq],
            "benchmark": [round(v, 2) for v in bench],
            "drawdown": dd,
            "worst": round(st().max_drawdown(name, s.starting_balance), 5),  # over every mark: the curve above is thinned or daily
            "fills": fills,
        })

    @app.post("/sleeves/{name}/command")
    def sleeve_command(name: str, command: str = Form(...), reason: str = Form(""),
                       reason_pick: str | None = Form(None), reason_note: str = Form(""), reason_for: str = Form(""),
                       then: str = Form(""), actor: str = Depends(require_pm), _o: None = Depends(same_origin)):
        """A PM command, with its reason: picked from the action's own list (reasons.py) or typed. Flatten with
        then=stop is the Stop dialog's "Flatten and stop": the flatten goes now, and the page asks to stop once
        the strategy is flat, as a stop sent now would drop the waiting flatten."""
        then_stop = ""
        if command == "flatten-stop":  # the Stop dialog's other button, with the Stop dialog's reasons
            command, then, reason_for = "flatten", "stop", "stop"
        try:
            action = reason_for if reason_for in ("stop", "close") and command == "flatten" else command
            reason = _reason(action, reason, reason_pick, reason_note)
            if not reason:
                raise ValueError("every command needs a reason")
            if command in ("start", "resume") and _liquidated(name):
                # The page says so too; a stale page or a direct post must not restart it (QA P1-U25, U31).
                raise ValueError(liquidation.REFUSAL)
            if command == "flatten" and then == "stop":
                then_stop = reason
            if command in ("start", "stop"):
                if not reason.strip():
                    raise ValueError("a reason is required")
                if command == "start" and _retired(st().account_of(name)):
                    raise ValueError(f"its account {st().account_of(name)} is retired; move it to another "
                                     "account or reinstate that one first")
                if command == "start":
                    s = st().sleeve(name)
                    check_perp_sizing(s.strategy, s.params)
                    check_perp_stop(s.strategy, s.params, s.risk_profile)
                    blocked, why = entry_blocked(st(), name, utcnow(), starting=True)  # CHOKE
                    if blocked and not (s.status == "halted" and any(
                            c["command"] in ("resume", "reset_after_liquidation") for c in st().pending_commands(name))):
                        raise ValueError(f"not started. {why}")  # a halt is cleared only by its own action (HC)
                # A Stop's acceptance, stamped to the microsecond just before it is committed: the time a raced fill
                # and a resting entry's cancel are measured from (P1-SG15, Advisor 7 Oct 05:01 and 05:47).
                accepted = datetime.now(timezone.utc) if command == "stop" else None
                st().set_desired_state(name, "running" if command == "start" else "stopped")
                if command == "stop":
                    # A command still waiting when its process stops would act on the next start, maybe
                    # weeks later; it lapses instead, and the decision log says so. Stop is always taken, a
                    # waiting flatten included (QA P1-D23): a strategy still holding runs for its exits only, its
                    # stop or a safety stop watching the position (P1-U35).
                    st().drop_pending(name, "lapsed: the strategy was stopped before it acted")
                st().decide(actor, command, reason, name, ts=accepted)
            elif (command == "resume" and (why := entry_blocked(st(), name, utcnow(), starting=True)[1])
                  and not set(why.codes) <= set(RESUMABLE)):
                raise ValueError(f"a resume can't clear it. {why}")
            elif (command == "resume" and st().sleeve(name).status == "running"
                  and not any(c["command"] in ("pause", "flatten") for c in st().pending_commands(name))):
                # Nothing to resume, and the runtime would reset the day's loss baseline (review round 10, m10-3).
                raise ValueError("it is already running, so there is nothing to resume")
            elif command == "flatten" and any(c["command"] == "flatten" for c in st().pending_commands(name)):
                # A second would sell again whatever the first left (review round 10, m5).
                raise ValueError("a flatten is already waiting for the strategy to act on it")
            elif command == "flatten" and st().sleeve(name).desired_state != "running":
                # A stopped strategy's process isn't there to act on a flatten, which would wait for its next
                # start, maybe weeks later (review round 8, M8-2). Holding a position, it starts to sell it,
                # as the kill switch does; flat, there is nothing to sell.
                s = st().sleeve(name)
                if abs(st().journal_book(name, s.starting_balance)["qty"]) <= 1e-12:
                    raise ValueError("it is stopped and holds no position, so there is nothing to flatten")
                st().command(name, command, reason, actor=actor)
                st().set_desired_state(name, "running")
                st().decide(actor, "start", f"Started to sell its position: {reason.strip()}", name)
            else:
                st().command(name, command, reason, actor=actor)
        except KeyError:
            raise HTTPException(404, "no such strategy") from None
        except ValueError as exc:
            # Back on the page, in words: a stale page can reach these, and raw JSON is no answer (round 9, N8).
            return RedirectResponse(f"/sleeves/{name}?{urlencode({'command_error': str(exc)})}", status_code=303)
        done = {"pause": "Pause sent", "resume": "Resume sent", "stop": "Stopped", "start": "Started",
                "flatten": "Close sent: it sells at market, then pauses" if reason_for == "close" else
                "Flatten sent: it closes at market, then pauses"}.get(command, "Sent")
        # Flatten and stop keeps then_stop in the address (the page asks to stop once flat), so no one-off toast.
        q = {"then_stop": then_stop} if then_stop else {"done": f"{done}. Reason: {reason}."}
        return RedirectResponse(f"/sleeves/{name}?{urlencode(q)}", status_code=303)

    def _liquidated(name: str) -> bool:
        """The dashboard half of the CHOKE gate: whether Start, Resume and Reset are refused because the
        strategy's position margin was lost with no Reset after liquidation since. The routes and the page
        both ask this, so they can't disagree. To read the engine's entry_blocked state once #155 has it."""
        return trading.liquidated_since_reset(st(), name)

    @app.post("/sleeves/{name}/reset")
    def sleeve_reset(name: str, reason: str = Form(""), reason_pick: str | None = Form(None),
                     reason_note: str = Form(""), actor: str = Depends(require_pm),
                     _o: None = Depends(same_origin)):
        """PM, 5 Oct 2026: reset a strategy during testing. The supervisor flattens it (and its demo copy),
        puts the run so far away under Previous book, and restarts it at its starting capital."""
        try:
            if _liquidated(name):  # an ordinary reset would put the liquidation away unanswered (P1-U27)
                raise ValueError(liquidation.REFUSAL)
            st().request_reset(name, _reason("reset", reason, reason_pick, reason_note), actor=actor)
        except KeyError:
            raise HTTPException(404, "no such strategy") from None
        except ValueError as exc:
            return RedirectResponse(f"/sleeves/{name}?{urlencode({'command_error': str(exc)})}", status_code=303)
        return RedirectResponse(f"/sleeves/{name}", status_code=303)

    @app.post("/book/reset")
    def book_reset(reason: str = Form(""), reason_pick: str | None = Form(None), reason_note: str = Form(""),
                   actor: str = Depends(require_pm), _o: None = Depends(same_origin)):
        """Reset every strategy on the current book (not archived, not a backtest, none already resetting)."""
        try:
            reason = _reason("reset", reason, reason_pick, reason_note)
        except ValueError as exc:
            reason = ""
            error = str(exc)
        else:
            error = "a reason is required"
        if not reason:
            return RedirectResponse(f"/setup?{urlencode({'reset_error': error})}", status_code=303)
        gone, liquidated = set(st().archived()), []
        for s in st().sleeves():
            if s.name in gone or st().pending_reset(s.name):
                continue
            if _liquidated(s.name):  # put away unanswered otherwise; it waits for Reset after liquidation
                liquidated.append(s.name)
                continue
            st().request_reset(s.name, reason, actor=actor)
        q = {"reset": "1", **({"not_reset": ", ".join(liquidated)} if liquidated else {})}
        return RedirectResponse(f"/setup?{urlencode(q)}", status_code=303)

    def _retired(account: str) -> bool:
        return any(a["name"] == account and a["retired_at"] for a in st().accounts())

    @app.post("/sleeves/{name}/settings")
    async def sleeve_settings(request: Request, name: str, actor: str = Depends(require_pm),
                              _o: None = Depends(same_origin)):
        """Change a strategy's risk settings in place: risk profile, stop and target, risk per trade and
        largest order. Logged with the PM's reason; a running strategy restarts to trade under them."""
        from sleeve_fund.paper.config import MAX_WARMUP_BARS as MAX_STORED_WARMUP_BARS
        from sleeve_fund.paper.config import from_store

        form = dict(await request.form())
        try:
            s = st().sleeve(name)
        except KeyError:
            raise HTTPException(404, "no such strategy") from None
        try:
            reason = reasons.from_form("save", form)
            if not reason:
                raise ValueError("a reason is required")
            if is_backtest(name):
                raise ValueError("a saved backtest's settings are what it tested; run a new backtest instead")
            if name in st().archived():
                raise ValueError("restore the strategy before changing its settings")
            profile = str(form.get("risk_profile", ""))
            if profile not in PROFILES:
                raise ValueError(f"risk profile: no profile called {profile}")
            # Only the risk settings change; the model's own parameters and the order type stay as they are.
            params = {k: v for k, v in s.params.items() if k not in RISK_KEYS}
            params.update({k: v for k, v in _form_params(form, "").items() if k in RISK_KEYS})
            cfg = from_store(replace(s, params=params, risk_profile=profile))
            half_spread = resolve_spread(cfg.venue, cfg.instrument, st()).half_spread
            _check_strategy_params(cfg, half_spread)
            _check_open_stop(st(), s, params, PROFILES[profile], float(cfg.fees.taker) + half_spread,
                             bool(form.get("confirm_looser")))
            held = st().journal_book(name, s.starting_balance)["qty"]
            changes = _risk_changes(s.risk_profile, s.params, profile, params, (held > 0) - (held < 0))
            if not changes:
                raise ValueError("nothing changed")
            cap = MAX_STORED_WARMUP_BARS if s.bar_spec.endswith("INTERNAL") else MAX_WARMUP_BARS
            warmup = max(s.warmup_bars, min(cap, exit_warmup(params)))
            trial, uncounted = _trial(_strategy_trial_metrics, st(), cfg, s.strategy, params, profile,
                                      fallback={"strategy": s.strategy, "params": params, "source": "strategy"})
            restart = _saved("settings change", st().change_settings, name, risk_profile=profile, params=params,
                             warmup_bars=warmup, trial=trial)
            text = "; ".join(changes)
            st().decide(actor, "change_settings", f"{text}. {reason}", name)
            _uncounted_event(st(), name, f"strategy {name}", uncounted)
            exits_changed = any(c.startswith(("Stop-loss", "Take-profit")) for c in changes)
            # The strategy reads "exits_change" on restart: an open position takes the new stop and target.
            st().event(name, "info", "exits_change" if exits_changed else "settings_change",
                       f"Settings changed by {actor}: {text}. " + ("Restarting to apply them."
                                                                  if restart else "They apply when it next starts."))
        except (ValueError, TypeError) as exc:
            kept = {k: str(v) for k, v in form.items() if isinstance(v, str) and v and k != "reason"}
            q = urlencode({"settings_error": str(exc), **{f"f_{k}": v for k, v in kept.items()}})
            return RedirectResponse(f"/sleeves/{name}?{q}#settings", status_code=303)
        return RedirectResponse(f"/sleeves/{name}?saved=settings#settings", status_code=303)

    @app.post("/sleeves/{name}/account")
    def sleeve_account(name: str, account: str = Form(...), reason: str = Form(""), reason_pick: str | None = Form(None),
                       reason_note: str = Form(""), actor: str = Depends(require_pm), _o: None = Depends(same_origin)):
        try:
            s = st().sleeve(name)
        except KeyError:
            raise HTTPException(404, "no such strategy") from None
        try:
            reason = _reason("move", reason, reason_pick, reason_note)
            if not reason:
                raise ValueError("a reason is required")
            # Paper processes don't use the account yet (fees come from the venue's schedule), so a running,
            # flat strategy moves without a restart.
            old = st().move_sleeve(name, account, st().journal_book(name, s.starting_balance)["qty"])
            st().decide(actor, "move_account", f"from {old} to {account}: {reason.strip()}", name)
            st().event(name, "info", "account_move", f"Moved from account {old} to {account} by {actor}")
        except ValueError as exc:
            return RedirectResponse(f"/sleeves/{name}?{urlencode({'settings_error': str(exc)})}#settings", status_code=303)
        return RedirectResponse(f"/sleeves/{name}?saved=account#settings", status_code=303)

    @app.post("/sleeves/{name}/archive")
    def sleeve_archive(name: str, action: str = Form(...), reason: str = Form(""), reason_pick: str | None = Form(None),
                       reason_note: str = Form(""), actor: str = Depends(require_pm), _o: None = Depends(same_origin)):
        try:
            reason = _reason(action if action in ("archive", "restore") else "archive", reason, reason_pick, reason_note)
            if not reason:
                raise ValueError("a reason is required")
            if is_backtest(name):
                raise ValueError("a saved backtest can't be archived")
            if action == "archive":
                st().archive(name)
            elif action == "restore":
                st().unarchive(name)
            else:
                raise ValueError("unknown action")
            st().decide(actor, action, reason, name)
        except KeyError:
            raise HTTPException(404, "no such strategy") from None
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return RedirectResponse("/" if action == "archive" else f"/sleeves/{name}", status_code=303)

    def research_page(request: Request, job=None, error: str = "", pre: dict | None = None,
                      notice: str = "", tab: str | None = None):
        """Research: Development (a plan per model and the study form), Results and History. Every venue's
        stored history is on the page, so a study's venue changes in place (no reload)."""
        from sleeve_fund.dashboard import development as dev
        from sleeve_fund.dashboard import pipeline

        pre = pre or {}
        sheets = [dev.read_sheet(p) for p in sorted(TEARSHEETS.glob("*.md"), key=lambda p: p.stat().st_mtime,
                                                    reverse=True)]
        put_away = st().archived()  # an archived strategy is not in paper any more (QA U6)
        rows = pipeline.strategies(TEARSHEETS, [s for s in st().sleeves() if s.name not in put_away])
        cards = dev.plans(rows, sheets)
        chosen = next((c for c in cards if c["name"] == pre.get("strategy")), cards[0])
        values = dev.form_values(chosen, pre)
        try:
            profile = _research_venue(values.get("venue"))
        except ValueError:
            profile = _research_venue()
        values["venue"] = profile.name.lower()
        venues, stored_all = [], []
        for v in venue_choices():
            vp = _research_venue(v["key"])
            stored = _stored_history(st(), vp)
            stored_all += [dict(h, venue=vp.label, venue_key=v["key"], perpetual=vp.perpetual, chip=dev.history_chip(h))
                           for h in stored]
            venues.append(dict(v, fee=float(resolve_fees(vp.name, st()).fees.taker),
                               suggest=[h["pair"] for h in stored] or _hints(vp.name),
                               held={h["pair"]: dev.history_chip(h) for h in stored}))
        here = next(v for v in venues if v["key"] == values["venue"])
        if not values.get("instrument"):
            values["instrument"] = "BTC/USD" if "BTC/USD" in here["suggest"] else here["suggest"][0]
        # Study windows sized to the stored history, unless the form came back with its own (UI v2, item 8).
        held = here["held"].get(str(values["instrument"]).upper())
        fitted, fit_text = dev.fit_windows(values, held.get("days") if held else None)
        if not any(k in pre for k in ("train_days", "test_days", "holdout_days")):
            values.update(fitted)
        ledger = IdeaLedger(LEDGER)
        spent = {f"{idea}|{base}": opened_words(e) for (idea, base), e in ledger.holdouts().items()}
        # What the study panel's script needs to switch model and venue in place.
        study_data = {"chosen": chosen["name"], "default_venue": _research_venue().name.lower(),
                      "plans": {c["name"]: {"values": c["values"], "variants": c["variants"]} for c in cards},
                      "venues": {v["key"]: {k: v[k] for k in ("label", "perpetual", "fee", "suggest", "held")}
                                 for v in venues}}
        # The trials register's counts, which include single backtests and paper strategies (QA P1-T1).
        return page(request, "research.html", sheets=sheets, study_data=study_data, counts=TrialsRegister(st()).counts(), rows=rows, cards=cards,
                    chosen=chosen, values=values, venues=venues, on_venue=here, stored_all=stored_all,
                    stages=pipeline.STAGES, job=job, error=error, notice=notice,
                    tab=tab, msg_in=tab or "development", request_years=REQUEST_YEARS, spent_holdouts=spent, fit_text=fit_text,
                    profiles=PROFILES,
                    study_candles=[(m, "1d" if m == 1440 else f"{m // 60}h" if m % 60 == 0 else f"{m}m")
                                   for m in study_run.STUDY_MINUTES],
                    costs=exit_costs(backtest=True),
                    how=dev.how_sentence(values), exits=dev.exits_sentence(values),
                    breakeven_text=dev.breakeven_text, candle_words=dev.candle_words)

    @app.get("/research", response_class=HTMLResponse)
    def research(request: Request, _: str = Depends(require_pm)):
        job = app.state.jobs.get(request.query_params.get("job", ""))
        view = dict(job.view(), ahead=app.state.jobs.ahead_of(job)) if job is not None else None
        # The study's own settings come back on the redirect, so the form shows what is running.
        pre = {k: v for k, v in request.query_params.items() if k != "job"}
        lost = request.query_params.get("job") and job is None
        return research_page(request, job=view, pre=pre, error=LOST_JOB if lost else "")

    @app.post("/research/run")
    async def research_run(request: Request, _: str = Depends(require_pm), _o: None = Depends(same_origin)):
        """Run a G1 study from the page, as `python -m sleeve_fund study --store` does, in the background."""
        form = dict((await request.form()).items())
        try:
            req = _study_request(form)
            req.validate()
        except (ValueError, KeyError) as exc:
            return research_page(request, error=str(exc), pre=form)
        from sleeve_fund.dashboard import development as dev

        # No history yet: say so now rather than from a failed job (R8-M6). There is no Collect button (UI v2,
        # item 10): a study on an instrument not stored yet asks the collector for it itself.
        profile = _research_venue(req.venue)
        held = next((h for h in _stored_history(st(), profile) if h["pair"] == req.pair), None)
        if held is None or held["first"] is None:
            if held:
                why = f"{req.pair}: being filled · the study waits until some is stored."
            else:
                try:
                    why = f"{req.pair}: not stored yet. {_ask_history(profile, req.pair)}"
                except ValueError as exc:
                    why = f"{req.pair}: not stored yet, and the collector wasn't asked: {exc}"
            return research_page(request, error=why, pre=form)
        if why := dev.blocked_by_gaps(req.pair, held["gaps"]):
            return research_page(request, error=why, pre=form)
        jobs = app.state.jobs
        target = st().url if jobs.isolate and st().url else st()
        key = "study|" + "|".join(f"{k}={v}" for k, v in sorted(vars(req).items()))
        title = (f"G1 study of {req.strategy.replace('_', ' ')} on {req.pair} ({dev.venue_label(profile.name) or 'venue'}), "
                 f"{study_run._bars(req.minutes)} bars")
        job = jobs.submit(key, title, run_study_job, target, req, str(LEDGER), str(TEARSHEETS))
        return RedirectResponse(f"/research?{urlencode({'job': job.id, **form})}", status_code=303)

    @app.post("/research/history")
    async def research_history(request: Request, _: str = Depends(require_pm), _o: None = Depends(same_origin)):
        """Ask the history collector to store an instrument, so a study can run on it before any strategy
        trades it (review round 8, R8-M6)."""
        form = dict((await request.form()).items())
        pair = str(form.get("instrument", "")).strip().upper()
        # From a study's "Not stored: Collect", the answer shows in that study; else on the History tab.
        tab = "development" if form.get("from") == "study" else "history"

        def page_(**kw):
            return research_page(request, tab=tab, **kw)

        try:
            profile = _research_venue(form.get("venue"))
        except ValueError as exc:
            return page_(error=str(exc))
        here = {"instrument": pair, "venue": profile.name.lower()}
        if str(form.get("strategy", "")) in REGISTRY:
            here["strategy"] = str(form["strategy"])
        try:
            notice = _ask_history(profile, pair)
        except ValueError as exc:
            # No button to ask again: the same request would be refused the same way.
            return page_(error=str(exc), pre=here)
        return page_(pre=here, notice=notice)

    def _ask_history(profile, pair: str) -> str:
        """Ask the history collector to store an instrument, and say what happened; ValueError when it can't be
        asked for. From the history request and from a study on an instrument not stored yet."""
        if not re.fullmatch(r"[A-Z0-9]{1,12}/[A-Z0-9]{2,6}", pair):
            raise ValueError(f"{pair or 'that'} isn't an instrument: write it as base and quote with a slash, "
                             f"like {_hints(profile.name)[0]}")
        held = next((h for h in _stored_history(st(), profile) if h["pair"] == pair and h["first"] is not None),
                    None)
        if held is not None:
            return (f"{pair} is already stored, from {held['first']:%d %b %Y} to {held['last']:%d %b %Y %H:%M} UTC "
                    f"({held['state']}); the collector keeps it current, and a study can run on it now.")
        if pair in profile.core_pairs:
            # The collector always keeps its core list from each instrument's listing (sleeve_fund.history), so
            # a request would change nothing, and its "from five years back" would misstate where it starts.
            return (f"{pair} is on the collector's core list for the venue: it is stored from its listing and "
                    "kept current, and this list shows how far it has got.")
        if profile.minute_loader is None:
            raise ValueError("this venue has no history loader")
        if profile.check_listed is not None:
            try:
                profile.check_listed(pair)
            except ValueError:
                raise
            except Exception as exc:  # noqa: BLE001 - venue unreachable
                # Asked for unchecked, a pair the venue doesn't list sat "Asked for" for good, with no way
                # to take it back (review round 9, N5): nothing is asked for until the venue confirms it.
                logging.getLogger(__name__).warning(f"couldn't check {profile.label} lists {pair}: {exc!r}")
                raise ValueError(f"couldn't reach the venue to check it lists {pair}, so nothing was asked "
                                 "for; try again when the venue answers") from None
        since = (utcnow() - timedelta(days=365 * REQUEST_YEARS)).replace(hour=0, minute=0, second=0, microsecond=0)
        new = st().request_history(profile.name, pair, since)
        return (f"Asked the collector for {pair}: it backfills from {since:%d %b %Y}, then keeps it current. "
                "A study can run once some is stored; this list shows how far it has got."
                if new else f"{pair} was already asked for; this list shows how far the collector has got.")

    @app.get("/strategies/{name}", response_class=HTMLResponse)
    def strategy_page(request: Request, name: str, _: str = Depends(require_pm)):
        from sleeve_fund.dashboard import pipeline

        row = next((r for r in pipeline.strategies(TEARSHEETS, st().sleeves()) if r["name"] == name), None)
        if row is None:
            raise HTTPException(404, "no such strategy")
        return page(request, "strategy.html", r=row, stages=pipeline.STAGES, summary=_idea(name))

    def _sheet_path(sheet: str) -> Path:
        path = (TEARSHEETS / f"{sheet}.md").resolve()
        if path.parent != TEARSHEETS.resolve() or not path.exists():
            raise HTTPException(404, "no such tear sheet")
        return path

    @app.get("/research/run")
    @app.get("/research/history")
    def research_form_reloaded(request: Request, venue: str = "", _: str = Depends(require_pm)):
        """A refresh or Back after a study or a history request asks for the form's own address by GET, which
        would otherwise be read as a tear sheet's name: back to the research page, on the same venue."""
        try:
            name = _research_venue(venue or None).name.lower()
        except ValueError:
            name = _research_venue().name.lower()
        anchor = "#history" if request.url.path.endswith("/history") else ""
        return RedirectResponse(f"/research?venue={name}{anchor}", status_code=303)

    @app.get("/research/{sheet}/download")
    def tearsheet_download(sheet: str, _: str = Depends(require_pm)):
        """The tear sheet as its Markdown file, to hand to the research thread as it is."""
        path = _sheet_path(sheet)
        return Response(path.read_bytes(), media_type="text/markdown; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{path.name}"'})

    @app.get("/research/{sheet}", response_class=HTMLResponse)
    def tearsheet(request: Request, sheet: str, _: str = Depends(require_pm)):
        path = _sheet_path(sheet)
        html = markdown.markdown(path.read_text(encoding="utf-8"), extensions=["tables"])
        # Results as status chips, so a FAIL can't be missed in a wall of text.
        for word, tone, label in (("PASS", "running", "Pass"), ("FAIL", "halted", "Fail"), ("WARN", "paused", "Warn"),
                                  ("INFO", "stopped", "Info"), ("NOT JUDGED", "paused", "Not judged"),
                                  ("N/A", "stopped", "Not judged")):
            html = html.replace(f"<td>{word}</td>", f'<td><span class="chip {tone}">{label}</span></td>')
        from sleeve_fund.dashboard import development as dev

        # On top of the sheet: its verdict in one sentence, four figures and the cost ladder, all read from it.
        s = dev.read_sheet(path)
        return page(request, "tearsheet.html", title=sheet, body=html, s=s, banner=dev.banner(s),
                    breakeven=dev.breakeven_text(s), per_day=dev.per_day(s.get("oos_trades"), s.get("oos_days")),
                    chart=dev.ladder_chart(s["rungs"], s["fee"]), candle_words=dev.candle_words)

    def _moved(request: Request, to: str, hash_: str = "") -> RedirectResponse:
        """An old page's address: the same query on its new page, so bookmarks and filter links still work."""
        q = request.url.query
        return RedirectResponse(f"{to}{'?' + q if q else ''}{hash_}", status_code=303)

    @app.get("/decisions")
    def decisions(request: Request, _: str = Depends(require_pm)):
        return _moved(request, "/records", "#log")

    @app.get("/decisions.csv")
    def decisions_csv(request: Request, _: str = Depends(require_pm)):
        rows = st().decisions(limit=100_000, **_decision_filters(request)["query"])
        return _csv("decisions", reports.to_csv(rows, ["ts", "actor", "action", "sleeve", "reason"]))

    @app.get("/reports")
    def reports_page(request: Request, _: str = Depends(require_pm)):
        return _moved(request, "/records")

    @app.get("/records", response_class=HTMLResponse)
    def records_page(request: Request, _: str = Depends(require_pm)):
        """Performance and the decision log under one period and strategy filter, which the exports follow."""
        q = request.query_params
        now = utcnow()
        sleeves, frames, summaries = book_data()
        names = [s.name for s in st().sleeves()]
        chosen = q.get("sleeve") if q.get("sleeve") in names else ""
        if chosen and chosen not in frames:  # a strategy an earlier clean slate put away: its own figures
            s = st().sleeve(chosen)
            frames = {chosen: bookm.daily(st(), chosen)}
            summaries = [bookm.sleeve_extras(st(), sleeve_summary(st(), s), frames[chosen])]
        elif chosen:
            summaries = [x for x in summaries if x["sleeve"].name == chosen]
        w = reports.window(q.get("period", ""), q.get("month", ""), now)
        fills = {x["sleeve"].name: st().fills(x["sleeve"].name, limit=100_000) for x in summaries}
        perf = reports.performance(summaries, frames, [f for fs in fills.values() for f in fs], w["start"], w["end"])
        m = reports.monthly(summaries, frames, fills)
        months = [r for r in m["rows"] if reports.in_window(r["key"], w)]
        keys = {r["month"] for r in months}
        m = {**m, "rows": months, "months": [mo for mo in m.get("months", []) if mo in keys]}
        # The log: the same window, unless an old Decisions link brought its own dates or action.
        f = _decision_filters(request)
        dq = {"sleeve": chosen or None, "action": f["query"]["action"],
              "since": f["query"].get("since") or w["start"], "until": f["query"].get("until") or w["end"]}
        decisions = st().decisions(limit=500, **dq)
        decisions.sort(key=lambda d: d["ts"], reverse=True)  # the timeline reads newest first by time
        log = reports.timeline(decisions, now)
        kinds = {k: sum(1 for d in decisions if reports.decision_kind(d["action"]) in
                        (("started", "stopped") if k == "startstop" else (k,))) for k, _l in reports.KIND_FILTERS}
        kinds["all"] = len(decisions)
        keep = {"period": w["period"] if w["period"] != reports.DEFAULT_PERIOD else "", "sleeve": chosen,
                "month": w["month"] or ""}

        def qs(**over) -> str:
            out = urlencode({k: v for k, v in {**keep, **over}.items() if v})
            return "?" + out if out else ""

        dl = {"sleeve": chosen} if chosen else {}
        dec_dl = {**dl, **({"from": dq["since"].strftime("%Y-%m-%d")} if dq["since"] else {}),
                  **({"to": (dq["until"] - timedelta(days=1)).strftime("%Y-%m-%d")} if dq["until"] else {}),
                  **({"action": dq["action"]} if dq["action"] else {})}
        exports = [(label, f"/exports/{kind}.csv" + ("?" + urlencode(dl) if dl else ""), f"{kind}.csv", what)
                   for kind, label, what in EXPORTS]
        exports.insert(4, ("Decision log", "/decisions.csv" + ("?" + urlencode(dec_dl) if dec_dl else ""),
                           "decisions.csv", "Every decision, with who made it and why."))
        return page(request, "records.html", perf=perf, m=m, w=w, chosen=chosen, sleeves=names,
                    periods=reports.PERIODS, log=log, latest=decisions[:5], total=len(decisions),
                    kind_filters=reports.KIND_FILTERS, kind_counts=kinds, kinds=reports.DECISION_KINDS,
                    qs=qs, exports=exports, action=dq["action"], shell=shell(sleeves))

    @app.get("/exports/{kind}.csv")
    def export_csv(kind: str, sleeve: str = "", _: str = Depends(require_pm)):
        names = [s.name for s in st().sleeves()]
        if sleeve and sleeve not in names and not _saved_backtest_summary(sleeve):
            raise HTTPException(404, "no such strategy")
        chosen = [sleeve] if sleeve else names
        if kind == "fills":
            rows = [f for n in chosen for f in reversed(st().fills(n, limit=1_000_000))]
            cols = ["ts", "sleeve", "side", "qty", "price", "fee", "order_id", "trade_id"]
        elif kind == "equity":
            rows = []
            for n in chosen:
                for ts, r in bookm.daily(st(), n).iterrows():
                    rows.append({"date": ts.date(), "sleeve": n, **{k: round(float(r[k]), 8) for k in r.index}})
            cols = ["date", "sleeve", "equity", "benchmark", "cash", "qty", "price"]
        elif kind == "trades":
            rows = []
            for n in chosen:
                orders = trading.orders_by_id(st(), n)
                for t in trading.trips(st().fills(n, limit=1_000_000), st().events(n, limit=5000), orders,
                                       st().exit_plans(n), _shorts(st(), n),
                                       st().funding(n, limit=1_000_000) if _shorts(st(), n) else None,
                                       st().insurance(n) if _shorts(st(), n) else None):
                    rows.append({"sleeve": n, **t, "held_hours": round(t["held"].total_seconds() / 3600, 2)
                                 if t["held"] else None})
            cols = ["sleeve", "opened", "closed", "held_hours", "side", "qty", "entry_px", "exit_px", "cost", "fees",
                    "funding", "insurance", "pnl",
                    "ret", "r", "planned_r", "exits_edited", "exit_kind", "entry_why", "exit_why", "entry_order",
                    "exit_order"]
        elif kind == "audit":
            from sleeve_fund.venues import DEFAULT_VENUE

            rows, signal_cols = [], []
            for n in chosen:
                s = st().sleeve(n)
                got, keys = trading.audit_rows(s, list(reversed(st().fills(n, limit=1_000_000))),
                                               trading.orders_by_id(st(), n), getattr(s, "venue", None) or DEFAULT_VENUE)
                rows += got
                signal_cols += [k for k in keys if k not in signal_cols]
            cols = trading.AUDIT_COLUMNS + signal_cols
        elif kind == "orders":
            rows = [dict(o, signal=json.dumps(o["signal"], sort_keys=True))
                    for n in chosen for o in reversed(st().orders(n, limit=1_000_000))]
            cols = ["ts", "sleeve", "order_id", "side", "order_type", "qty", "status", "filled_qty", "avg_px", "fee",
                    "intent", "reason", "signal", "message", "updated_at"]
        else:
            raise HTTPException(404, "unknown export")
        return _csv(f"{kind}-{sleeve or 'all'}", reports.to_csv(rows, cols))

    def exit_costs(backtest: bool = False) -> dict:
        """What each leg of a trade costs, for the plan line under the exit fields: the taker fee and
        each instrument's half spread (measured, else the venue's assumption), exactly as the backtest
        and paper charge them, so the form and the backtest quote the same round trip. `markets`: a
        perpetual's own taker fee, and in a backtest its assumed half spread (paper pays the live one)."""
        default = resolve_spread(None, "?/?", None).half_spread
        venue_fees = resolve_fees(None, st()).fees
        per_market = {}
        for m in markets.MARKETS:
            t = markets.terms({"market": m})
            if t is None:
                continue
            per_market[m] = {"taker": float(markets.fees_for({"market": m}, venue_fees).taker)}
            if backtest and t.half_spread is not None:
                per_market[m]["half_spread"] = t.half_spread
        return {"taker": float(venue_fees.taker), "default_spread": default, "markets": per_market,
                "spreads": {p: resolve_spread(None, p, st()).half_spread for p in _hints()}}

    def backtest_form(request: Request, q, *, result=None, error="", job=None, saved=None):
        """The backtest page: the settings form, plus a result, an error, or a run in progress."""
        from sleeve_fund.dashboard import pipeline

        strategy = q.get("strategy") if q.get("strategy") in REGISTRY else "trend_filter"
        period = q.get("period") if q.get("period") in BACKTEST_PERIODS else "all"
        bar_spec = q.get("bar_spec") if q.get("bar_spec") in ALLOWED_BAR_SPECS else BACKTEST_BAR_SPEC
        try:
            venue = _venue_name(q.get("venue"))
        except ValueError:
            venue = _venue_name(None)
        # The sleeve must decide on the bars that were tested, with indicators warm from its first bar.
        carry = {k: v for k, v in q.items() if k not in ("run", "period") and v}
        carry.update(bar_spec=bar_spec, tested_bar_spec=bar_spec, warmup_bars=_warmup_for(strategy, q, bar_spec))
        g1 = {r["name"]: "|".join(r["passed_on"]) for r in pipeline.strategies(TEARSHEETS, st().sleeves())}
        pair = q.get("instrument", "").strip().upper()
        g1_here = (pipeline.g1_for(TEARSHEETS, strategy, pair, spec_minutes(bar_spec),
                                   {"market": q.get("market") if q.get("market") in markets.MARKETS else None,
                                    "allow_short": str(q.get("allow_short", "")).lower() in ("1", "true", "on", "yes")})
                   if pair else None)
        chart = None
        if result:
            chart = {"t": result["t"], "equity": result["equity"], "benchmark": result["benchmark"],
                     "drawdown": result["drawdown"], "fills": result["fills"], "res": "daily",  # equity is daily
                     "worst": round(-result["strategy"]["max_drawdown"], 5)}  # over every mark, as the table
        market = q.get("market") if q.get("market") in markets.MARKETS else markets.SPOT
        shorts = market != markets.SPOT and str(q.get("allow_short", "")).lower() in ("1", "true", "on", "yes")
        try:
            t = markets.terms({"market": market}, venue) if market != markets.SPOT else None
            label = "Perpetual" if t and t.funding_venue else (t.label if t else "")  # never the venue's name (QA U8)
        except ValueError:  # a spot market on a perpetual venue: the run itself is refused, with the reason
            label = markets.terms({"market": market}).label
        market_words = ("Spot, long only" if market == markets.SPOT else
                        f"{label}, {'long and short' if shorts else 'long only'}")
        return page(request, "backtest.html", result=result, error=error, job=job, saved=saved, pre=dict(q),
                    market_words=market_words,
                    chosen=strategy, strategies=_strategy_choices(), instruments=_hints(venue), g1=g1, g1_here=g1_here,
                    period=period, periods=BACKTEST_PERIODS, profiles=PROFILES, bar_spec=bar_spec,
                    bar_specs=sorted(ALLOWED_BAR_SPECS, key=spec_minutes),
                    sleeve_qs=urlencode({**carry, "from": "backtest"}), chart=chart, stored=_stored(venue),
                    runs=st().backtests(limit=BACKTEST_KEEP), costs=exit_costs(backtest=True))

    @app.get("/backtest", response_class=HTMLResponse)
    def backtest_page(request: Request, _: str = Depends(require_pm)):
        """Run any strategy and settings over the venue's history on the paper runtime. The run goes to
        the background; a quick one comes straight back as its saved result, a long one shows progress."""
        q = request.query_params
        if not q.get("run"):
            lost = q.get("job") and app.state.jobs.get(q["job"]) is None
            return backtest_form(request, q, error=LOST_JOB if lost else "")
        try:
            args = _backtest_args(q)
        except (ValueError, TypeError, KeyError) as exc:
            return backtest_form(request, q, error=backtest_error(exc))
        # The same settings, fees and spread within a few minutes show the saved run rather than a copy.
        key = _backtest_key(q, resolve_fees(args["venue"], st()).text,
                            resolve_spread(args["venue"], args["pair"], st()).text)
        hit = st().fresh_backtest(key, utcnow() - BACKTEST_FRESH)
        if hit:
            return RedirectResponse(f"/backtest/{hit['id']}", status_code=303)
        query = urlencode([(k, v) for k, v in q.items() if k != "run" and v != ""])
        jobs = app.state.jobs
        # A process of its own opens the journal by its address; in-process it shares this one.
        target = st().url if jobs.isolate and st().url else st()
        job = jobs.submit(key, args["title"], run_backtest_job, target, args, key, query)
        job.done_event.wait(BACKTEST_WAIT)
        if job.status == "done":
            return RedirectResponse(f"/backtest/{job.run_id}", status_code=303)
        if job.status == "error":
            return backtest_form(request, q, error=job.error)
        return backtest_form(request, q, job=dict(job.view(), ahead=app.state.jobs.ahead_of(job)))

    @app.get("/api/backtest/jobs/{job_id}")
    def backtest_job(job_id: str, _: str = Depends(require_pm)):
        job = app.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such run; it may have finished before the server restarted")
        return JSONResponse(dict(job.view(), ahead=app.state.jobs.ahead_of(job)))

    @app.get("/backtest/{run_id}", response_class=HTMLResponse)
    def saved_backtest(request: Request, run_id: str, _: str = Depends(require_pm)):
        """A saved run: its result, and every trade rebuilt from its journal, as the Trades screen does."""
        try:
            row = st().backtest(run_id)
        except KeyError:
            raise HTTPException(404, "no such backtest; older runs are cleared to save space") from None
        from urllib.parse import parse_qsl

        q = dict(parse_qsl(row["query"]))
        result, name = row["result"], row["sleeve"]
        result["trips"] = trading.trips(st().fills(name, limit=1_000_000), st().events(name, limit=10_000),
                                        trading.orders_by_id(st(), name), shorts=_shorts(st(), name),
                                        funding=st().funding(name, limit=1_000_000) if _shorts(st(), name) else None,
                                        insurance=st().insurance(name) if _shorts(st(), name) else None)
        result["orders"] = sum(st().order_counts(name).values())
        rs = [t["r"] for t in result["trips"] if t["r"] is not None]
        result["expectancy_r"] = sum(rs) / len(rs) if rs else None
        return backtest_form(request, q, result=result, saved=row)

    @app.get("/trades", response_class=HTMLResponse)
    def trades_page(request: Request, _: str = Depends(require_pm)):
        sleeves, _frames, summaries = book_data()
        names = [s.name for s in sleeves]
        sleeve = request.query_params.get("sleeve") or None
        backtest = _saved_backtest_summary(sleeve)
        if backtest:  # one saved backtest's trades, in the same screen as paper's
            summaries = [backtest]
        elif sleeve not in names:
            sleeve = None
        h = trading.history(st(), summaries, sleeve)
        return page(request, "trades.html", h=h, sleeve=sleeve, sleeves=names, shell=shell(sleeves),
                    book_equity=sum(x["equity"] for x in summaries), backtest=backtest)

    @app.get("/orders", response_class=HTMLResponse)
    def orders_page(request: Request, _: str = Depends(require_pm)):
        names = [s.name for s in st().sleeves()]
        q = request.query_params
        backtest = _saved_backtest_summary(q.get("sleeve"))
        sleeve = q.get("sleeve") if q.get("sleeve") in names or backtest else None
        tab = q.get("status") if q.get("status") in trading.STATUS_TABS else "all"
        rows = [trading.order_view(o) for o in st().orders(sleeve, trading.STATUS_TABS[tab][1], limit=1000)]
        counts = st().order_counts(sleeve)
        tabs = [(k, label, sum(counts.get(x, 0) for x in sts) if sts else sum(counts.values()))
                for k, (label, sts) in trading.STATUS_TABS.items()]
        return page(request, "orders.html", orders=rows, tab=tab, tabs=tabs, sleeve=sleeve, sleeves=names,
                    backtest=backtest)

    @app.get("/accounts")
    def accounts_page(request: Request, _: str = Depends(require_pm)):
        return _moved(request, "/setup", "#accounts")

    def _accounts_rows() -> list[dict]:
        from sleeve_fund import accounts as acc

        rows = st().accounts()
        sleeves = st().sleeves()
        running = {s.name for s in sleeves if s.desired_state == "running"}
        held = {s.name for s in sleeves if s.desired_state != "running"
                and abs(st().journal_book(s.name, s.starting_balance)["qty"]) > 1e-12}
        for r in rows:
            r["env"] = acc.env_names(r["name"], r["venue"]) if r["kind"] == "live" else None
            r["running"] = [n for n in r["sleeves"] if n in running]
            r["held"] = [n for n in r["sleeves"] if n in held]  # stopped, still holding a position
        return rows

    @app.post("/accounts/new")
    async def new_account(request: Request, actor: str = Depends(require_pm), _o: None = Depends(same_origin)):
        form = dict(await request.form())
        name = str(form.get("name", "")).strip()
        try:
            reason = str(form.get("reason", "")).strip()
            if not reason:
                raise ValueError("a reason is required")
            kind = str(form.get("kind", ""))
            st().create_account(name, kind, str(form.get("note", "")).strip()[:200], venue=str(form.get("venue", "")))
            on = f" on the {dev.venue_label(str(form.get('venue')))} venue" if kind == "live" else ""
            st().decide(actor, "create_account", f"{kind} account {name}{on}: {reason}")
        except ValueError as exc:
            kept = {k: str(v) for k, v in form.items() if isinstance(v, str) and v}
            return RedirectResponse(f"/setup?{urlencode({'error': str(exc), **kept})}#add", status_code=303)
        return RedirectResponse(f"/setup?saved={name}#acct-{name}", status_code=303)

    @app.post("/accounts/{name}/note")
    def account_note(name: str, note: str = Form(""), reason: str = Form(...),
                     actor: str = Depends(require_pm), _o: None = Depends(same_origin)):
        try:
            if not reason.strip():
                raise ValueError("a reason is required")
            st().set_account_note(name, note)
            st().decide(actor, "account_note", f"{name}: note set to \"{note.strip()[:200]}\". {reason.strip()}")
        except ValueError as exc:
            return RedirectResponse(f"/setup?{urlencode({'error': str(exc)})}#accounts", status_code=303)
        return RedirectResponse(f"/setup?saved={name}#acct-{name}", status_code=303)

    @app.post("/accounts/{name}/retire")
    def account_retire(name: str, action: str = Form(...), reason: str = Form(...),
                       actor: str = Depends(require_pm), _o: None = Depends(same_origin)):
        try:
            if not reason.strip():
                raise ValueError("a reason is required")
            if action == "retire":
                st().retire_account(name)
            elif action == "reinstate":
                st().reinstate_account(name)
            else:
                raise ValueError("unknown action")
            st().decide(actor, f"{action}_account", f"{name}: {reason.strip()}")
        except ValueError as exc:
            return RedirectResponse(f"/setup?{urlencode({'error': str(exc)})}#accounts", status_code=303)
        return RedirectResponse(f"/setup?saved={name}#acct-{name}", status_code=303)

    @app.get("/settings")
    def settings_page(request: Request, _: str = Depends(require_pm)):
        return _moved(request, "/setup", "#settings")

    @app.get("/setup", response_class=HTMLResponse)
    def setup_page(request: Request, _: str = Depends(require_pm), error: str = ""):
        """Path to live, one card per setting area, and today's Accounts and Settings content as its detail."""
        from sleeve_fund.dashboard import setup_view
        from sleeve_fund.venues import VENUES

        # Named by market, not venue: venue names appear only where the PM sets up keys (QA U8 ruling).
        fee_quotes = [{"venue_label": "Perpetual venue" if v.perpetual else "Spot venue", "source": q.source, "taker": float(q.fees.taker),
                       "rates": f"{float(q.fees.maker):.2%} maker / {float(q.fees.taker):.2%} taker",
                       "basis": q.basis if q.source == "published" else
                       f"account {q.account}, {q.fetched_at:%d %b %Y %H:%M} UTC",
                       "assumed_spread": f"{2 * v.assumed_half_spread:.2%}"}
                      for v in VENUES.values() for q in [resolve_fees(v.name, st())]]
        sleeves = current_sleeves()
        frame = shell(sleeves)
        mirror = setup_view.mirror_state(st(), sleeves)
        here = setup_view.stage(frame, sleeves, mirror)
        accounts = _accounts_rows()
        return page(request, "setup.html", profiles=PROFILES, venues=VENUES.values(), fee_quotes=fee_quotes,
                    tearsheets=str(TEARSHEETS), counts=st().table_sizes(), accounts=accounts,
                    error=error, pre=dict(request.query_params), shell=frame,
                    steps=setup_view.path(here), stage_n=here, next_words=setup_view.NEXT[here], mirror=mirror,
                    paper_count=sum(1 for x in sleeves if x.name not in st().archived()),
                    resettable=[x.name for x in sleeves if x.name not in st().archived()],
                    alerts_out=setup_view.outside_alerts(st()), backup=setup_view.backup_state(frame["now"]))

    return app


BACKTEST_PERIODS = {"180": ("6 months", 180), "365": ("1 year", 365), "all": ("All available", None)}
PAIR_RE = re.compile(r"[A-Z0-9]{1,12}/[A-Z0-9]{2,6}")
# How long the page waits for a backtest before showing its progress instead; most daily runs finish.
BACKTEST_WAIT = float(os.environ.get("BACKTEST_WAIT_SECONDS", "8"))
BACKTEST_FRESH = timedelta(minutes=15)  # the same settings within this long show the saved run again
BACKTEST_KEEP = 50  # saved runs kept, with their journals; older ones are deleted


def backtest_error(exc: Exception) -> str:
    """A backtest failure in words for the page."""
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return str(exc).strip("'")
    if isinstance(exc, OSError):  # the venue unreachable
        return f"could not reach the venue for price history ({exc}). Try again in a minute."
    return f"the backtest failed: {exc}"


def _backtest_args(q) -> dict:
    """The backtest page's settings, checked, as preview.run's arguments."""
    strategy = q.get("strategy") if q.get("strategy") in REGISTRY else "trend_filter"
    period = q.get("period") if q.get("period") in BACKTEST_PERIODS else "all"
    bar_spec = q.get("bar_spec") if q.get("bar_spec") in ALLOWED_BAR_SPECS else BACKTEST_BAR_SPEC
    venue = _venue_name(q.get("venue"))
    pair = q.get("instrument", "").strip().upper()
    if not PAIR_RE.fullmatch(pair):
        raise ValueError(f"instrument: write it as BASE/QUOTE, for example {_hints(venue)[0]}")
    starting = float(q.get("starting_balance") or 10_000)
    if not 100 <= starting <= 1e9:
        raise ValueError("capital: between 100 and 1,000,000,000")
    params = _form_params(q, strategy)
    _profile_cap(q)  # validates the profile name
    markets.check_venue(params, venue)  # a perpetual venue has no spot
    check_perp_sizing(strategy, params)
    if spec_minutes(bar_spec) < 1440:
        have = [r["pair"] for r in _stored(venue)]
        if pair not in have:
            raise ValueError(f"interval: {pair} has no stored minute history here, so it can only be backtested on "
                             f"daily bars. Instruments with stored minutes: {', '.join(have) or 'none yet'}.")
    _check_gaps(venue, pair)
    title = (f"{strategy.replace('_', ' ').capitalize()} on {pair}"
             f"{'' if venue == _venue_name(None) else ' (' + (dev.venue_label(venue) or 'venue') + ')'}, "
             f"{_bar_short(bar_spec)}, {BACKTEST_PERIODS[period][0].lower()}")
    return {"strategy": strategy, "pair": pair, "venue": venue, "params": params, "starting": starting,
            "days": BACKTEST_PERIODS[period][1], "minutes": spec_minutes(bar_spec),
            "risk_profile": q.get("risk_profile") or "balanced", "title": title, "bar_spec": bar_spec}


def _check_gaps(venue: str, pair: str) -> None:
    """A backtest waits while the instrument's stored history has gaps (UI v2, item 10), with the badge's words."""
    from sleeve_fund.dashboard import development as dev
    from sleeve_fund.history import HistoryStore

    try:
        gaps = HistoryStore().gaps(venue, pair)
    except OSError:  # an unreadable store: the run itself says what failed
        return
    if why := dev.blocked_by_gaps(pair, gaps):
        raise ValueError(why)


def _stored(venue: str | None = None) -> list[dict]:
    from sleeve_fund.dashboard import preview

    try:
        return preview.stored(venue)
    except OSError:  # an unreadable store reads as empty; the run itself says what failed
        return []


def _backtest_key(q, *costs: str) -> str:
    import hashlib

    items = sorted((k, v) for k, v in q.items() if k != "run" and v != "")
    return hashlib.sha256("|".join([urlencode(items), *costs]).encode()).hexdigest()[:32]


def run_backtest_job(progress, run_id: str, store: Store | str, args: dict, key: str, query: str) -> str:
    """Run one backtest on the paper runtime and save it, with its journal, as `run_id`. Runs in a
    process of its own (see jobs.py), where `store` is the journal's database address."""
    from sleeve_fund.dashboard import preview

    if isinstance(store, str):
        store = Store(store)
    keep: dict = {}
    result = preview.run(args["strategy"], args["pair"], args["params"], starting=args["starting"],
                         days=args["days"], detail=True, minutes=args["minutes"], risk_profile=args["risk_profile"],
                         venue=args["venue"], fee_quote=resolve_fees(args["venue"], store),
                         spread_quote=resolve_spread(args["venue"], args["pair"], store),
                         progress=progress, keep=keep)
    result.pop("trips", None)  # rebuilt from the saved journal, as the Trades screen does
    trial, uncounted = _trial(_backtest_trial_metrics, store, args, result, run_id,
                              fallback={"strategy": args["strategy"], "params": args["params"], "source": "backtest",
                                        "backtest_id": run_id})
    # One transaction: a backtest is never saved uncounted, nor counted unsaved (QA P1-T8).
    _saved("backtest", store.save_backtest, keep["journal"], run_id=run_id, key=key, title=args["title"], query=query,
           result=result, bar_spec=args["bar_spec"], trial=trial)
    _uncounted_event(store, None, f"backtest {args['title']!r}", uncounted)
    store.prune_backtests(keep=BACKTEST_KEEP)
    return run_id


def _trial(run, *args, fallback: dict) -> tuple[dict, Exception | None]:
    """A run's row for the trials register, written by the caller in the same transaction as the run's own save
    (QA P1-T8, Head of Engineering), so a failed write loses both and the request says so. run(*args) gives the
    run's key and result. If working that out raises, or the row would be refused, the run is not lost for it (Data Architect): it gets a
    failed row, which still counts as a variant tried and keeps G1 from judging its idea until it is re-counted
    (Advisor), keyed as its variant where the key itself was worked out, else by `fallback` (strategy, params,
    source). Returns the row and the error, if any."""
    from sleeve_fund.research.trials import failed_row, model_run_row
    from sleeve_fund.store import check_trial

    key = None
    try:
        key = run(*args)
        row = model_run_row(**key)
        check_trial(row)  # a row the register would refuse is a failed count too, never a refused save
        return row, None
    except Exception as exc:  # noqa: BLE001 - any failure here must not lose the run
        logging.getLogger(__name__).exception("couldn't work out a run's row for the trials register")
        keyed = key or fallback
        return failed_row(strategy=keyed["strategy"], params=keyed["params"], source=keyed["source"], error=repr(exc),
                          setup=keyed.get("setup"), dataset=keyed.get("dataset"),
                          backtest_id=keyed.get("backtest_id")), exc


def _saved(what: str, save, *args, **kwargs):
    """Run a save that writes its trial row on the same transaction. A database failure loses both, and says
    so in words the page shows (QA P1-T8)."""
    from sqlalchemy.exc import SQLAlchemyError

    try:
        return save(*args, **kwargs)
    except SQLAlchemyError as exc:
        logging.getLogger(__name__).exception(f"couldn't save the {what} with its trial row")
        raise ValueError(f"The {what} was not saved: writing it with its count in the trials register failed "
                         f"({getattr(exc, 'orig', None) or exc}), so nothing was kept. Try again.") from exc


def _uncounted_event(store: Store, sleeve: str | None, what: str, exc: Exception | None) -> None:
    """Show on the dashboard that a run went into the trials register as failed (QA P1-T8)."""
    if exc is not None:
        store.event(sleeve, "error", "trials_count_failed",
                    f"The {what} was saved, but its count in the trials register failed ({exc!r}), so it is recorded "
                    "against its idea as a failed run: it still counts as a variant tried, and G1 won't judge the "
                    "idea until it is re-counted")

def _count_backtest(store: Store, args: dict, result: dict, run_id: str) -> None:
    """Every backtest is a variant tried: the trials register counts it, so a setting picked from many runs is
    judged against all of them (QA P1-T1)."""
    from sleeve_fund.research.trials import record_model_run

    record_model_run(store, **_backtest_trial_metrics(store, args, result, run_id))


def _backtest_trial_metrics(store: Store, args: dict, result: dict, run_id: str) -> dict:
    """A backtest as the register keys it: its variant, its dataset, and every bar from its first day to its last."""
    import pandas as pd

    from sleeve_fund.research.run import dataset_name
    from sleeve_fund.research.trials import backtest_period, run_setup

    start = pd.Timestamp(result["from"], tz="UTC")
    end = pd.Timestamp(result["to"], tz="UTC") + pd.Timedelta(days=1)
    fees = result.get("fee_schedule") or {}
    setup = run_setup(risk_profile=args["risk_profile"],
                      fee=float(fees.get("taker", 0.0)) + float((result.get("spread") or {}).get("half", 0.0)),
                      period=backtest_period(args.get("days")))
    return dict(strategy=args["strategy"], params=args["params"], setup=setup,
                dataset=dataset_name(_venue_name(args["venue"]), args["pair"], args["minutes"]), source="backtest",
                sharpe=result["strategy"]["sharpe"], data_start=start, data_end=end, backtest_id=run_id,
                trades=(result.get("trades") or {}).get("trades"))


def _strategy_trial_metrics(store: Store, cfg, strategy: str, params: dict, risk_profile: str) -> dict:
    """A paper strategy created, cloned or re-set, as the register keys it: a variant chosen to run, counted with
    no Sharpe yet. cfg is its paper config (venue, instrument, bars and fees)."""
    from sleeve_fund.research.run import dataset_name
    from sleeve_fund.research.trials import run_setup

    fee = float(cfg.fees.taker) + resolve_spread(cfg.venue, cfg.instrument, store).half_spread
    return dict(strategy=strategy, params=params, source="strategy",
                setup=run_setup(risk_profile=risk_profile, fee=fee),
                dataset=dataset_name(_venue_name(cfg.venue), cfg.instrument, spec_minutes(cfg.bar_spec)))


LOST_JOB = ("that run is no longer known, most likely because the server restarted while it ran; "
            "run it again (review round 10, m9)")


def _qty(x: float) -> str:
    """A quantity to six significant figures, without exponents: 23,350.1 and 0.0765 rather than 2.335e+04
    (review round 10, m3)."""
    if not x:
        return "0"
    d = min(8, max(0, 5 - math.floor(math.log10(abs(x)))))
    return f"{x:,.{d}f}".rstrip("0").rstrip(".") if d else f"{x:,.0f}"


def _research_venue(name: str | None = None):
    from sleeve_fund.venues import venue

    return venue(name or None)


def _venue_name(value) -> str:
    """A venue picked on a form (any case; blank is the default venue), as its profile names it. Raises
    ValueError on a venue with no profile."""
    return _research_venue(str(value or "").strip() or None).name


def _hints(venue: str | None = None) -> list[str]:
    """The instruments a form suggests on this venue (VenueProfile.hints); any other can still be typed."""
    return list(_research_venue(venue).hints)


def venue_choices() -> list[dict]:
    """Every venue a strategy can be researched, backtested or paper traded on, for the forms' venue field."""
    from sleeve_fund.venues import VENUES

    return [{"key": v.name.lower(), "label": v.label, "perpetual": v.perpetual, "pairs": _hints(v.name)}
            for v in VENUES.values() if v.data_client is not None or v.minute_loader is not None]


def _check_listed(store: Store, venue: str | None, pair: str) -> None:
    """A typed instrument ("Use '…' as typed") is asked of the venue before a strategy can start on it (UI v2,
    item 9). The usual, stored and already-listed ones need no question. Raises ValueError in words for the page."""
    from sleeve_fund.dashboard import charts

    pair = pair.strip().upper()
    profile = _research_venue(venue)
    if not PAIR_RE.fullmatch(pair):
        raise ValueError(f"instrument: write it as BASE/QUOTE, for example {_hints(profile.name)[0]}")
    listed = charts._listed.get(profile.name, (0, []))[1]
    if pair in {*_hints(profile.name), *listed} or profile.check_listed is None:
        return
    if any(h["pair"] == pair for h in _stored_history(store, profile)):
        return
    try:
        profile.check_listed(pair)
    except ValueError:
        raise ValueError(f"instrument: the venue doesn't list {pair}; pick one from the list") from None
    except Exception as exc:  # noqa: BLE001 - venue unreachable
        logging.getLogger(__name__).warning(f"couldn't check {profile.label} lists {pair}: {exc!r}")
        raise ValueError(f"instrument: couldn't reach the venue to check it lists {pair}; try again, or pick one "
                         "from the list") from None


def instrument_options(store: Store, listing=None) -> list[dict]:
    """Every instrument the venues offer, for the forms' instrument pick-list (UI v2, item 9): perpetuals
    first, then spot, each with its history badge. A venue whose listing can't be fetched offers what its
    history store keeps. Labels say perpetual or spot, never the venue's name; `venue` is the form value."""
    from sleeve_fund.dashboard import development as dev

    out = []
    for v in sorted(venue_choices(), key=lambda v: not v["perpetual"]):
        stored = {h["pair"]: dev.history_chip(h) for h in _stored_history(store, _research_venue(v["key"]))}
        try:
            listed = list(listing(venue=v["key"])) if listing else []
        except Exception as exc:  # noqa: BLE001 - venue unreachable: the stored and usual ones, any other typed
            logging.getLogger(__name__).info(f"couldn't list {v['label']}'s instruments: {exc!r}")
            listed = []
        kind = "perpetual" if v["perpetual"] else "spot"
        pairs = list(dict.fromkeys([*stored, *v["pairs"], *sorted(listed)]))
        out += [{"value": p, "venue": v["key"], "group": "Perpetuals" if v["perpetual"] else "Spot",
                 "label": f"{p} {kind}", "badge": stored[p]["text"] if p in stored else "not stored yet",
                 "tone": stored[p]["tone"] if p in stored else "", "stored": p in stored} for p in pairs]
    return out


def _stored_history(store: Store, profile=None) -> list[dict]:
    """Each instrument research can use or has asked for, on the research venue: what is stored, whether
    the collector is current or still catching up, any gaps, and the badge that says so. One unreadable
    series (a coverage file being rewritten, say) is left out and logged rather than taking the Research
    page down with it."""
    from sleeve_fund.dashboard import development as dev
    from sleeve_fund.history import HistoryStore

    profile = profile or _research_venue()
    hist = HistoryStore()
    now = utcnow()
    log = logging.getLogger(__name__)
    try:
        asked = {r["instrument"]: r for r in store.history_requests(profile.name)}
    except Exception as exc:  # noqa: BLE001 - an old database without the table: nothing asked for
        log.warning(f"couldn't read the history requests for {profile.label}: {exc!r}")
        asked = {}
    out = []
    try:
        series = hist.series()
    except OSError as exc:
        log.warning(f"couldn't list the stored history: {exc!r}")
        series = []
    for v, pair in series:
        if v != profile.name:
            continue
        try:
            cov = hist.coverage(v, pair)
            # The last candle is the newest the store records as closed, once the hub writes (append_bars);
            # the REST loader's newest minute may still be forming.
            first, last = cov.first, cov.closed if cov.closed is not None else cov.last
            if last.tzinfo is None:  # stored without a zone: UTC, as the collector writes it
                first, last = first.tz_localize("UTC"), last.tz_localize("UTC")
            behind = now - last.to_pydatetime() > study_run.STALE_HISTORY
            gaps = hist.gaps(v, pair)
            kinds = [e.get("kind") for e in hist.provenance(v, pair)]
            funding_gaps = _funding_health(v, pair, hist.root, log)
        except Exception as exc:  # noqa: BLE001 - see the docstring
            log.warning(f"couldn't read the stored history of {pair} on {profile.label}: {exc!r}")
            continue
        row = {"pair": pair, "first": first, "last": last, "requested": asked.get(pair, {}).get("requested_at"),
               "state": "catching up" if behind else "current", "gaps": gaps,
               "refills": kinds.count("refill"), "conflicts": kinds.count("conflict"), **funding_gaps}
        out.append({**row, "badge": dev.history_badge(row)})
    held = {h["pair"] for h in out}
    out += [{"pair": p, "first": None, "last": None, "requested": r["requested_at"], "state": "asked for", "gaps": [],
             "badge": dev.history_badge({"first": None})}
            for p, r in asked.items() if p not in held]
    return sorted(out, key=lambda h: h["pair"])


def _funding_health(venue: str, pair: str, root, log) -> dict:
    """Missed funding settlements and possible holes at a change of settlement interval (QA P1-O9/O11), for the
    instrument's history chip. Empty where no funding is kept."""
    from sleeve_fund import funding

    try:
        return {"funding_gaps": len(funding.gaps(venue, pair, root)),
                "funding_maybe": len(funding.interval_changes(venue, pair, root))}
    except Exception as exc:  # noqa: BLE001 - the prices' badge stands without it
        log.warning(f"couldn't read the funding kept for {pair}: {exc!r}")
        return {}


def _study_request(form: dict) -> "study_run.StudyRequest":
    def num(name: str, default=None, cast=float):
        v = str(form.get(name, "")).strip()
        return cast(v) if v else default

    def pct(name: str):
        v = num(name)
        return None if v is None else v / 100

    strategy = str(form.get("strategy", ""))
    if strategy not in REGISTRY:
        raise ValueError(f"unknown model {strategy!r}")
    profile = str(form.get("risk_profile", "balanced"))
    if profile != "none" and profile not in PROFILES:
        raise ValueError(f"unknown risk profile {profile!r}")
    return study_run.StudyRequest(
        strategy=strategy, pair=str(form.get("instrument", "")).strip().upper(), venue=_venue_name(form.get("venue")),
        minutes=num("minutes", 1440, int), risk_profile=None if profile == "none" else profile,
        train_days=num("train_days", 3 * 365, int), test_days=num("test_days", 365, int),
        holdout_days=num("holdout_days", 365, int), use_holdout=form.get("use_holdout") == "on",
        stop_loss=pct("stop_loss_pct"), take_profit=pct("take_profit_pct"), risk_per_trade=pct("risk_per_trade_pct"),
        stop_atr=num("stop_atr"), stop_swing_bars=num("stop_swing_bars", cast=int), atr_bars=num("atr_bars", cast=int),
        take_profit_r=num("take_profit_r"))


def run_study_job(progress, job_id: str, store: Store | str, req, ledger: str, tearsheets: str) -> str:
    """Run one G1 study and write its tear sheet; returns the sheet's name. Runs in a process of its
    own (see jobs.py), where `store` is the journal's database address."""
    if isinstance(store, str):
        store = Store(store)
    path = study_run.run_store_study(req, store=store, progress=progress, ledger_path=Path(ledger),
                                     out_dir=Path(tearsheets))
    return path.stem


def _reason(action: str, reason: str, pick: str | None, note: str) -> str:
    """The PM's reason for an action: composed from the dialog's pick and note (reasons.compose, which says in
    words what is missing), or the plain reason field older forms send."""
    return reasons.compose(action, pick, note) if pick is not None else (reason or "").strip()



def _demo_copy(store, s) -> dict | None:
    """What the strategy page shows of its demo copy: only for a perpetual strategy copied to a demo account the
    mirror trades to an exact quantity (mirror.exact_copy)."""
    from sleeve_fund import mirror

    label = mirror.exact_copy(s)
    if label is None or not markets.is_perp(s.params):
        return None
    put_on = sum(float(r["amount"] or 0.0) for r in store.mirror_rows(s.name, limit=100_000)
                 if r["status"] == "filled")
    last = store.last_resync(s.name)
    return {"label": "the demo account", "put_on": put_on, "last": last,
            "leverage": PROFILES[s.risk_profile].max_leverage}



EXIT_KINDS = trading.EXIT_EVENTS


def _config_defaults(strategy: str) -> dict:
    import inspect

    _, config_cls = REGISTRY[strategy]
    return {k: p.default for k, p in inspect.signature(config_cls.__init__).parameters.items()
            if p.default is not inspect.Parameter.empty and isinstance(p.default, (int, float, str))}


def _idea(strategy: str, params: dict | None = None) -> str:
    """The strategy in one sentence, using this sleeve's settings rather than the defaults."""
    import importlib

    try:
        spec = importlib.import_module(f"sleeve_fund.strategies.{strategy}").SPEC
    except (ImportError, AttributeError):
        return ""
    if not spec.summary:
        return spec.idea
    values = _config_defaults(strategy)
    values.update(params or {})
    try:
        return spec.summary.format(**values)
    except (KeyError, IndexError, ValueError):
        return spec.idea



FEEDS = {
    "all": lambda e: e["kind"] != "reconcile",  # routine passes are summarised in the risk panel
    "alerts": lambda e: e["level"] in ("warning", "error"),
    "trades": lambda e: e["kind"] in ("fill", "order_rejected", "order_denied", *EXIT_KINDS),
    "pm": lambda e: e["kind"].startswith("pm_") or e["kind"] in ("start", "restart", "restore", "resume"),
}


def _feed(events: list[dict], kind: str) -> list[dict]:
    keep = FEEDS.get(kind, FEEDS["all"])
    return [e for e in events if keep(e)][:120]


def _chart_pairs(home: str, book: list[str]) -> list[str]:
    """The top of a chart's instrument dropdown: the strategy's own, then the book's. The venue's full list follows."""
    return list(dict.fromkeys([home, *book]))


def _risk_view(x: dict, position: dict | None = None) -> dict:
    p = x["profile"]
    stop_px = position["stop_px"] if position else None
    margin = position.get("margin", 0.0) if position else 0.0
    margin_cap = p.max_position_pct * max(x.get("equity", 0.0), 0.0)
    return {
        "stop_px": stop_px,
        "target_px": position["target_px"] if position else None,
        # The move from here to the stop (round 9, N3): a drop for a long, a rise for a short.
        "to_stop": abs(1 - stop_px / x["price"]) if stop_px and x["price"] else None,
        # Not capped at 100%: drift past the entry cap shows its true share, in amber (P1-U13).
        "cap_used": abs(x["exposure"]) / cap if (cap := x.get("cap", p.max_position_pct)) else 0.0,
        "day_used": min(max(-x["day_ret"], 0.0) / p.daily_loss, 1.0) if p.daily_loss else 0.0,
        # The limit bars (UI v2, item 6): the position's margin against the most the profile lets it put up.
        "margin": margin,
        "margin_cap": margin_cap,
        "margin_used": margin / margin_cap if margin_cap > 0 else 0.0,
    }


def _held(td) -> str:
    if td is None:
        return ""
    secs = max(td.total_seconds(), 0.0)  # a file stamped a moment after the clock read is 0, not -0
    hours = secs / 3600
    return f"{hours / 24:.1f} d" if hours >= 48 else f"{hours:.0f} h" if hours >= 1 else f"{secs / 60:.0f} min"


# The Records page's Export menu: (file, what it is, what it's for), in the order the menu lists them.
EXPORTS = [("trades", "Closed trades", "Round trips with P&L after fees and their reasons. Best for comparing."),
           ("audit", "Full audit", "Every fill with its P&L, reason and indicator values."),
           ("orders", "Orders", "Every order, filled or refused, with why."),
           ("equity", "Daily equity", "Book and strategy value, cash and position each day."),
           ("fills", "Fills", "Every fill as the venue reported it: time, side, quantity, price, fee.")]
DECISION_ACTIONS = ["create", "start", "stop", "pause", "resume", "flatten", "change_settings", "move_account", "archive",
                    "restore", "flatten everything", "create_account", "account_note", "retire_account",
                    "reinstate_account"]
# The decision log's actions and the feeds' event kinds as the PM reads them (review round 8, R8-4, R8-9).
ACTION_WORDS = {"change_settings": "Changed settings", "move_account": "Moved account", "create_account": "Created account",
                "account_note": "Account note", "retire_account": "Retired account",
                "reinstate_account": "Reinstated account", "flatten everything": "Flattened everything"}
KIND_WORDS = {"handler_failed": "Strategy error", "maker_fill_above_tape": "Maker fill ahead of the tape",
              "maker_fill_settled": "Maker fill settled", "crossing_trade_unseen": "Fill on an unseen trade",
              "risk_halt": "Risk halt", "risk_pause": "Risk pause", "reconcile_mismatch": "Reconcile mismatch",
              "liquidation": "Liquidated", "liquidation_cut": "Cut before liquidation",
              "insurance_fund": "Insurance fund",
              "instrument_not_found": "Instrument not found", "tick_failed": "Risk check failed",
              "mark_unavailable": "No price to value the book", "price_feed_back": "Price feed back",
              "heartbeat_stale": "Heartbeat late", "process_crash": "Process crashed", "process_start": "Process started",
              "process_stop": "Process stopped", "supervisor_error": "Supervisor error",
              "supervisor_start": "Supervisor started", "fee_fetch_failed": "Couldn't fetch fees", "fees": "Fees",
              "alert_send_failed": "Couldn't send alert", "alerts_sent": "Alerts sent", "alerts_config": "Alerts set up",
              "backup_problem": "Backup problem", "exits_change": "Exits changed", "settings_applied": "Settings applied",
              "stop_rejected": "Stop refused", "stop_reset": "Stop reset", "exits_applied": "Exits applied", "warmup_short": "Short warm-up",
              "account_move": "Moved account"}


def action_words(action: str) -> str:
    return ACTION_WORDS.get(action) or action.replace("_", " ").capitalize()


def kind_words(kind: str) -> str:
    return KIND_WORDS.get(kind) or kind.replace("_", " ").capitalize()


def _decision_filters(request: Request) -> dict:
    from datetime import datetime, timedelta, timezone

    q = request.query_params
    out = {"sleeve": q.get("sleeve") or None, "action": q.get("action") or None}
    raw = {"from": q.get("from", ""), "to": q.get("to", "")}
    for key, field, extra in (("from", "since", 0), ("to", "until", 1)):
        try:
            if raw[key]:
                out[field] = datetime.strptime(raw[key], "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=extra)
        except ValueError:
            raw[key] = ""  # an unreadable date is ignored rather than failing the page
    return {"query": out, "raw": raw, "qs": urlencode({k: v for k, v in q.items() if v})}


def _csv(name: str, body: str) -> Response:
    stamp = utcnow().strftime("%Y%m%d")
    return Response(body, media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="fund-{name}-{stamp}.csv"'})


def _bytes(n) -> str:
    if n is None:
        return "n/a"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024
    return ""


def _ago(t) -> str:
    if not t:
        return "never"
    secs = int((utcnow() - t).total_seconds())
    if secs < 0:
        return t.strftime("%d %b %H:%M UTC")
    for size, unit in ((86400, "d"), (3600, "h"), (60, "min")):
        if secs >= size:
            return f"{secs // size} {unit} ago"
    return "just now"


FEED_FRESH_SECONDS = riskops.FEED_FRESH_SECONDS  # past this the strategy page's price feed reads as stale


def _price_feed(s, seen) -> dict:
    """The strategy's price feed for the page header: how long since its venue last sent it a trade or quote,
    to the second. Labelled "Prices", never with the venue's name (QA U8)."""
    label = "Prices"
    if s.desired_state != "running":
        return {"label": label, "state": "off", "age": "off while stopped"}
    if seen is None:
        return {"label": label, "state": "off", "age": "waiting for the first trade"}
    secs = max(0, int((utcnow() - seen).total_seconds()))
    age = f"{secs} s ago" if secs < 60 else f"{secs // 60} min ago" if secs < 3600 else f"{secs // 3600} h ago"
    return {"label": label, "state": "ok" if secs <= FEED_FRESH_SECONDS else "stale", "age": age}


def _bar_short(spec: str) -> str:
    step, unit = spec.split("-")[:2]
    return f"{step}{ {'SECOND': 's', 'MINUTE': 'm', 'HOUR': 'h', 'DAY': 'd'}.get(unit, unit.lower())} bars"


def _bar_choices(specs) -> list[tuple[str, str]]:
    """Candle-length chips: "15m", "1h"; the venue's own hourly candles say so beside the ones built from trades."""
    out = []
    for spec in specs:
        step, unit, _, source = spec.split("-")
        text = f"{step}{ {'SECOND': 's', 'MINUTE': 'm', 'HOUR': 'h', 'DAY': 'd'}.get(unit, unit.lower())}"
        twin = sum(1 for o in specs if o.split("-")[:2] == [step, unit]) > 1
        out.append((spec, f"{text} venue" if twin and source == "EXTERNAL" else text))
    return out


def _bar_label(spec: str) -> str:
    step, unit, _, source = spec.split("-")
    unit = unit.lower() + ("s" if step != "1" else "")
    how = "built from live trades" if source == "INTERNAL" else "the venue's candles, supports warm-up"
    return f"{step} {unit} ({how})"


def _strategy_choices() -> list[dict]:
    import importlib

    out = []
    for name in sorted(REGISTRY):
        spec = importlib.import_module(f"sleeve_fund.strategies.{name}").SPEC
        out.append({"name": name, "idea": _idea(name), "params": spec.default_params, "family": spec.family,
                    "tpl": spec.summary, "defaults": _config_defaults(name)})
    return out


def _g1_of(strategy: str, instrument: str, minutes: int, params: dict | None = None) -> str | None:
    from sleeve_fund.dashboard import pipeline

    return pipeline.g1_for(TEARSHEETS, strategy, instrument, minutes, params)


# Exit settings the form takes as stored (the % ones are converted above): an ATR or swing-low stop,
# and a target in R after costs.
EXIT_SETTINGS = {"stop_atr": float, "atr_bars": int, "stop_swing_bars": int, "take_profit_r": float}


# The settings the strategy's Settings tab changes in place (with the risk profile). Everything else in
# params is the model's own or the order type, which stay as created.
RISK_KEYS = ("max_notional", "stop_loss", "take_profit", "risk_per_trade", *EXIT_SETTINGS)


def _risk_form(params: dict) -> dict:
    """The risk settings in params as the form enters them (the % ones as percentages)."""
    q = {}
    if "max_notional" in params:
        q["max_notional"] = f"{params['max_notional']:g}"
    for key in ("stop_loss", "take_profit", "risk_per_trade"):  # stored as fractions, entered as %
        if key in params:
            q[f"{key}_pct"] = f"{params[key] * 100:g}"
    for key in EXIT_SETTINGS:
        if key in params:
            q[key] = f"{params[key]:g}"
    return q


def _risk_words(profile: str, params: dict, side: int = 0) -> dict[str, str]:
    """Each risk setting in words, for the decision log's before and after, in the terms of the position
    held (side: 1 long, -1 short, 0 flat): a short's stop is above its entry (review round 11, M11-7)."""
    p = params
    w = trading.exit_ways(params, side)
    held = " (short)" if side < 0 else ""
    if p.get("stop_atr"):
        stop = f"{p['stop_atr']:g} simple average true ranges ({p.get('atr_bars', 14)} bars) {w['stop']} the entry{held}"
    elif p.get("stop_swing_bars"):
        stop = f"at the {w['swing']} of {p['stop_swing_bars']} bars{held}"
    elif p.get("stop_loss"):
        stop = f"{p['stop_loss'] * 100:g}% {w['stop']} the entry{held}"
    else:
        stop = "none"
    target = (f"{p['take_profit_r']:g}R after costs" if p.get("take_profit_r")
              else f"{p['take_profit'] * 100:g}% {w['tp']} the entry{held}" if p.get("take_profit") else "none")
    return {"Risk profile": profile, "Stop-loss": stop, "Take-profit": target,
            "Risk per trade": f"{p['risk_per_trade'] * 100:g}%" if p.get("risk_per_trade") else "none",
            "Largest order": f"{p['max_notional']:,.2f}" if p.get("max_notional") else "no cap"}


def _risk_changes(old_profile: str, old: dict, new_profile: str, new: dict, side: int = 0) -> list[str]:
    before, after = _risk_words(old_profile, old, side), _risk_words(new_profile, new, side)
    return [f"{k} {before[k]} to {after[k]}" for k in before if before[k] != after[k]]


MARKET_KEYS = ("market", "allow_short", "demo_mirror")  # form fields of their own, not model parameters


def market_choices() -> list[tuple[str, str]]:
    return [(markets.SPOT, "Spot: long only, the venue's fees"),
            (markets.PERP, f"{markets.LOW_FEE_PERP.label}: {markets.LOW_FEE_PERP.fees.maker:.2%} maker, "
                           f"{markets.LOW_FEE_PERP.fees.taker:.2%} taker, funding"),
            (markets.PERP_VENUE_FEES, f"{markets.VENUE_FEE_PERP.label}: funding")]


def _market_form(params: dict) -> dict:
    """A strategy's market settings as the form's own fields."""
    out = {}
    if params.get("market"):
        out["market"] = params["market"]
    for key in ("allow_short", "demo_mirror"):
        if params.get(key):
            out[key] = "1"
    return out


def _clone_qs(s) -> str:
    """The new-sleeve form filled in with this sleeve's settings, for "Clone with changes"."""
    params = {k: v for k, v in s.params.items() if k not in RISK_KEYS and k not in MARKET_KEYS}
    q = {"strategy": s.strategy, "instrument": s.instrument, "bar_spec": s.bar_spec,
         "starting_balance": f"{s.starting_balance:g}", "risk_profile": s.risk_profile, "warmup_bars": s.warmup_bars,
         "name": f"{s.name[:38]}-v2", "from": "clone", "source": s.name, **_risk_form(s.params),
         **_market_form(s.params), **({"venue": s.venue.lower()} if s.venue else {})}
    if "maker_wait_minutes" in params:
        q.update(execution="maker", maker_wait_minutes=params.pop("maker_wait_minutes"))
    q.update({f"p_{s.strategy}__{k}": v for k, v in params.items()})
    return urlencode(q)


# The backtest page's default interval. A sleeve made from a backtest decides on the bars it tested.
BACKTEST_BAR_SPEC = "1-DAY-LAST-EXTERNAL"
MAX_WARMUP_BARS = VENUE_WARMUP_BARS  # what the venue returns in one request; bars built from trades load from the store


def _profile_cap(q) -> float:
    """The chosen risk profile's position cap, as paper applies it (balanced when none is given)."""
    name = q.get("risk_profile") or "balanced"
    if name not in PROFILES:
        raise ValueError(f"risk profile: no profile called {name}")
    return PROFILES[name].max_position_pct


def _defaults(strategy: str) -> dict:
    import importlib

    return dict(importlib.import_module(f"sleeve_fund.strategies.{strategy}").SPEC.default_params)


def _warmup_for(strategy: str, q, bar_spec: str = BACKTEST_BAR_SPEC) -> int:
    """Bars to load at start so the slowest indicator is settled on the sleeve's first bar, as the
    strategy itself says (paper.config.auto_warmup)."""
    try:
        params = _form_params(q, strategy)
    except ValueError:
        params = {}
    return auto_warmup(strategy, params, bar_spec)


def _form_params(form, strategy: str) -> dict:
    """Strategy parameters and exits from the new-sleeve form (also used by the preview)."""
    # Each strategy's parameter inputs are named p_<strategy>__<param>; only the chosen one counts.
    prefix = f"p_{strategy}__"
    params = _coerce_params({k[len(prefix):]: v for k, v in form.items() if k.startswith(prefix) and v != ""})
    # A setting the form can't honour is refused, never dropped: a long/short strategy's market came
    # through "Clone with changes" as a model parameter the form had no field for, and the backtest quietly
    # ran it as long-only spot (review round 11, B11-1).
    known = set(_defaults(strategy)) | _config_keys(strategy) if strategy in REGISTRY else set()
    unknown = sorted(set(params) - known)
    if unknown:
        raise ValueError(f"{', '.join(unknown)}: unknown setting for the {strategy.replace('_', ' ')} model; it "
                         "can't be honoured here, so nothing was run")
    market = str(form.get("market", "") or markets.SPOT)
    if market not in markets.MARKETS:
        raise ValueError(f"market: one of {', '.join(markets.MARKETS)}")
    if market != markets.SPOT:
        params["market"] = market
    for key in ("allow_short", "demo_mirror"):
        if str(form.get(key, "")).strip().lower() in ("1", "true", "on", "yes"):
            if market == markets.SPOT:
                raise ValueError(f"{key.replace('_', ' ')}: only on a perpetual; spot is long only")
            params[key] = True
    if str(form.get("max_notional", "")).strip():
        params["max_notional"] = float(form["max_notional"])
    for key in ("stop_loss", "take_profit", "risk_per_trade"):  # entered as %, stored as fractions
        raw = str(form.get(f"{key}_pct", "")).strip()
        if raw:
            params[key] = round(float(raw) / 100, 6)
    for key, cast in EXIT_SETTINGS.items():  # entered as they are stored
        raw = str(form.get(key, "")).strip()
        if raw:
            try:
                params[key] = cast(raw)
            except ValueError:
                raise ValueError(f"{key.replace('_', ' ')}: a number") from None
    if "atr_bars" in params and "stop_atr" not in params:
        params.pop("atr_bars")  # only means something with an ATR stop
    execution = str(form.get("execution", "") or "market")
    if execution not in ("market", "maker"):
        raise ValueError("order type: market or maker first")
    if execution == "maker" and not maker_orders_enabled():
        raise ValueError("order type: maker-first orders are switched off for now; every order goes at market")
    if execution == "maker":
        try:
            params["maker_wait_minutes"] = int(str(form.get("maker_wait_minutes", "")).strip() or 15)
        except ValueError:
            raise ValueError("go to market after: a whole number of minutes") from None
    return params


def _config_keys(strategy: str) -> set[str]:
    """The keyword settings a model's config takes beyond the common ones."""
    import inspect

    sig = inspect.signature(REGISTRY[strategy][1].__init__)
    return {n for n, p in sig.parameters.items() if p.kind == p.KEYWORD_ONLY}


def _coerce_params(raw: dict) -> dict:
    out = {}
    for k, v in raw.items():
        if not re.fullmatch(r"[a-z_]{1,32}", k):
            raise ValueError(f"bad parameter name {k!r}")
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                raise ValueError(f"parameter {k} must be a number") from None
    return out


def _shorts(store: Store, name: str) -> bool:
    """Whether a strategy's (or saved backtest's) journal can hold shorts: it trades a perpetual."""
    try:
        return markets.is_perp(store.sleeve(name).params)
    except (KeyError, ValueError):
        return False


def _check_open_stop(store: Store, s, params: dict, profile, leg: float, confirmed: bool) -> None:
    """A stop the PM loosens on an open position is checked against its size before it saves (review
    round 8, M8-1: a 3% to 50% edit saved, about 17R at risk). A stop set from the market only ever
    tightens the one working (LongFlatStrategy._replan), so this is about a % stop, or removing the stop.
    - Refused when the loss from here to the new stop is more than is left before the profile's
      drawdown halt: the halt, not the stop, would close the position.
    - Otherwise a looser stop saves only with the PM's confirmation, which the error asks for with the
      risk in money and in R of the entry's risk."""
    book = store.journal_book(s.name, s.starting_balance)
    if not book["qty"] or not book["entry_px"] or params.get("stop_atr") or params.get("stop_swing_bars"):
        return
    lot = trading.open_lot(store.fills(s.name, limit=100_000), shorts=markets.is_perp(s.params))
    entry = trading.orders_by_id(store, s.name).get(lot["order_id"]) if lot else None
    plan = store.exit_plans(s.name).get(lot["order_id"]) if lot else None
    working, _ = trading.exit_fracs(s.params, entry["signal"] if entry else None, plan)
    new = params.get("stop_loss")
    if working is None or (new is not None and new <= working):
        return
    # A short works the same way mirrored: its stop is above the entry and its loss is a rise.
    side = 1 if book["qty"] > 0 else -1
    qty, entry_px = abs(book["qty"]), book["entry_px"]
    last = store.last_equity(s.name)
    price = last["price"] if last and last.get("price") else entry_px
    if new is None:
        if not confirmed:
            raise ValueError(f"removing the stop leaves the open position ({qty * price:,.2f}) with none; tick "
                             "\"Accept a looser stop on the open position\" to save it")
        return
    level = entry_px * (1 - side * new)
    loss = max(side * qty * (price - level), 0.0) + qty * level * leg  # from here to the stop, and the fee to close
    equity = last["equity"] if last else s.starting_balance
    peak = max(store.peak_equity(s.name) or s.starting_balance, equity)
    headroom = equity - peak * (1 - profile.max_drawdown)
    if loss > headroom:
        raise ValueError(f"a {new:.1%} stop on the open position would lose {loss:,.2f} from here, more than the "
                         f"{max(headroom, 0):,.2f} left before the {profile.name} profile's "
                         f"{profile.max_drawdown:.0%} drawdown halt, which would close it first; choose a tighter stop")
    if not confirmed:
        risk = (plan or {}).get("risk_amount") or ((entry or {}).get("signal") or {}).get("risk_amount")
        in_r = f" ({qty * entry_px * (new + leg + (1 - side * new) * leg) / risk:.1f}R of the entry's risk)" if risk else ""
        raise ValueError(f"a {new:.1%} stop is looser than the {working:.1%} the open position works to now: it "
                         f"risks {loss:,.2f} from here{in_r}. Tick \"Accept a looser stop on the open position\" "
                         "to save it")


# The strategy page's Activity chips (combined build F2): every event and decision, each under one kind.
_RISK_KINDS = {"risk_halt", "risk_pause", "reconcile_mismatch", "liquidation", "liquidation_cut", "stop_loss",
               "take_profit", "flatten_retry", "flatten_failed", "drawdown_reset"}
_TRADE_KINDS = {"fill", "order_rejected", "order_denied"}


def _timeline(events: list[dict], decisions: list[dict]) -> list[dict]:
    """Events and PM decisions in one list, newest first, each with a kind for the filter chips: decisions
    (yours, and the strategy acting on them), risk (halts, pauses, stops, warnings and errors), trades (fills
    and refused orders) and system (starts, warm-up, the feed and the rest). Routine reconcile passes stay out,
    as before."""
    rows = []
    for e in events:
        k = e["kind"]
        if k == "reconcile":
            continue
        kind = ("decisions" if k.startswith("pm_") else "trades" if k in _TRADE_KINDS
                else "risk" if k in _RISK_KINDS or e["level"] in ("warning", "error") else "system")
        rows.append({"ts": e["ts"], "kind": kind, "level": e["level"], "title": kind_words(k), "text": e["message"]})
    for d in decisions:
        rows.append({"ts": d["ts"], "kind": "decisions", "level": "info", "title": action_words(d["action"]),
                     "text": f"by {d['actor']}: {d['reason']}"})
    rows.sort(key=lambda r: r["ts"], reverse=True)
    return rows[:150]


_HISTORY_KINDS = {"fill": "Fill", "exits_applied": "Stop and target changed", "stop_reset": "Stop set again",
                  "stop_loss": "Stop-loss", "take_profit": "Take-profit", "liquidation_cut": "Cut back"}


def _position_history(events: list[dict], funding: list[dict], position: dict | None) -> list[dict]:
    """The open position's story, newest first: its fills (the entry and any adds), stop moves and funding."""
    if not position or not position.get("opened"):
        return []
    since = position["opened"]
    rows = [{"ts": e["ts"], "what": _HISTORY_KINDS[e["kind"]], "text": e["message"]}
            for e in events if e["kind"] in _HISTORY_KINDS and e["ts"] >= since]
    rows += [{"ts": f["ts"], "what": "Funding", "text": f"{f['amount']:+,.2f} at {f['rate'] * 100:.4f}%"}
             for f in funding if f["ts"] >= since]
    rows.sort(key=lambda r: r["ts"], reverse=True)
    return rows[:60]


def _journey(s, x: dict, path: list[dict] | None, mirror: list[dict]) -> list[dict]:
    """The strategy's road to live as five steps, from what the checklist and the mirror already show: Research
    (passed G1), Paper (six weeks, ten trades, clean), Demo check (fills copied to a demo account), G2 approval
    (yours) and Live. The first step not done is where it is now. Read-only: nothing here approves anything."""
    if not path:
        return []
    ok = {r["label"]: r for r in path}
    rows = list(ok.values())
    g1, weeks, trades = rows[0], rows[1], rows[2]
    clean = all(r["ok"] for r in rows[3:5])
    filled = [m for m in mirror if m["status"] == "filled"]
    errors = [m for m in mirror[:20] if m["status"] == "error"]
    mirrored = bool(s.params.get("demo_mirror"))
    steps = [
        {"label": "Research", "done": bool(g1["ok"]), "detail": "passed G1" if g1["ok"] else "not passed G1"},
        {"label": "Paper", "done": bool(weeks["ok"] and trades["ok"] and clean),
         "detail": f"{weeks['detail']} · {trades['detail']}"},
        {"label": "Demo check", "done": bool(filled) and not errors,
         "detail": (f"{len(filled)} fill{'s' if len(filled) != 1 else ''} copied" + (", with errors" if errors else ""))
         if filled else "mirror on, no fills yet" if mirrored else "not mirrored to a demo account"},
        {"label": "G2 approval", "done": False, "detail": "your decision"},
        {"label": "Live", "done": x.get("mode") == "live", "detail": "live" if x.get("mode") == "live" else "keys and the switch"},
    ]
    here = next((i for i, st_ in enumerate(steps) if not st_["done"]), None)
    for i, st_ in enumerate(steps):
        st_["state"] = "done" if st_["done"] else "here" if i == here else "todo"
    return steps


SIGNALS_FRESH_SECONDS = 60  # older than this, the Signals tab waits for the model rather than show old lights
_OP_WORDS = {"<=": "≤", ">=": "≥", "<": "<", ">": ">"}


def _signal_row(c: dict, guard: bool = False) -> dict:
    """One condition as the Signals tab draws it: value, threshold, distance and the gauge's positions (%).
    guard: a stop or target, which reads as how far away it is, or hit."""
    unit, v, t = c.get("unit", "pts"), c.get("value"), c.get("threshold") or 0.0
    fmt = (lambda n: f"{n:+.2f}%") if unit == "%" else (lambda n: f"{n:.1f}")
    row = {"name": c["name"], "note": c.get("note", ""), "met": bool(c.get("met")),
           "threshold": f"{_OP_WORDS.get(c.get('op'), c.get('op'))} {fmt(t)}", "value": "–", "distance": "", "gauge": None}
    if v is None:
        row["distance"] = "met" if row["met"] else "not yet"
        return row
    row["value"] = fmt(v)
    gap = abs(v - t)
    gap_words = f"{gap:.2f}%" if unit == "%" else f"{gap:.1f} pts"
    if guard:
        row["distance"] = "hit · exits at once" if row["met"] else f"{gap_words} away"
    else:
        row["distance"] = f"met · {gap_words} inside" if row["met"] else f"{gap_words} to go"
    lo, hi = c.get("gauge_min"), c.get("gauge_max")
    if lo is not None and hi is not None and hi > lo:
        at = lambda n: round(min(max((n - lo) / (hi - lo), 0.0), 1.0) * 100, 2)  # noqa: E731
        tick = at(t)
        below = c.get("op") in ("<=", "<")
        row["gauge"] = {"tick": tick, "dot": at(v), "zone_left": 0.0 if below else tick,
                        "zone_width": tick if below else round(100 - tick, 2)}
    return row


def _signal_card(side: int, rows: list[dict] | None, guards: list[dict], held: int, flat_why: str) -> dict:
    """One side's card: its rules (entry, or exit while the model is on that side), the count, and the
    stop and target while a position on that side is open."""
    card = {"label": "Long" if side > 0 else "Short", "key": "long" if side > 0 else "short", "flat": flat_why,
            "rows": [], "exit": False, "met": 0, "total": 0, "all": False, "guards": []}
    if flat_why:
        return card
    card["rows"] = [_signal_row(r) for r in rows or []]
    card["exit"] = bool(rows) and all(r.get("exit") for r in rows)
    card["met"] = sum(r["met"] for r in card["rows"])
    card["total"] = len(card["rows"])
    card["all"] = card["total"] > 0 and card["met"] == card["total"]
    if held == side:
        card["guards"] = [_signal_row(g, guard=True) for g in guards]
    return card


def _signals_view(s, row: dict | None) -> dict:
    """The strategy page's Signals tab, from the latest conditions its paper process wrote
    (LongFlatStrategy.signal_state): the long and short cards, the lights for the tab label, and when the
    model next acts. Display only."""
    cls = REGISTRY.get(s.strategy, (None,))[0]
    from sleeve_fund.strategies.base import LongFlatStrategy

    view = {"supported": cls is not None and cls.conditions is not LongFlatStrategy.conditions, "state": "waiting",
            "age": "", "cards": [], "lights": [], "every": "", "close_in": "", "close_at": 0, "warming": False}
    if not view["supported"]:
        return view
    if s.desired_state != "running":
        view["state"] = "stopped"
        return view
    if row is None:
        return view
    now = utcnow()
    secs = max(0, int((now - row["ts"]).total_seconds()))
    view["age"] = f"{secs} s ago" if secs < 60 else f"{secs // 60} min ago" if secs < 3600 else _ago(row["ts"])
    if secs > SIGNALS_FRESH_SECONDS:
        view["state"] = "stale"
        return view
    p = row["payload"]
    held = int(p.get("held") or 0)
    if not markets.is_perp(s.params):
        short_flat = "held flat on spot"
    elif not s.params.get("allow_short"):
        short_flat = "held flat: shorts are off in its settings"
    else:
        short_flat = ""
    view["cards"] = [_signal_card(1, p.get("long"), p.get("guards") or [], held, ""),
                     _signal_card(-1, p.get("short"), p.get("guards") or [], held, short_flat)]
    view["warming"] = not view["cards"][0]["rows"]
    view["lights"] = [r["met"] for r in view["cards"][0]["rows"]]
    view["state"] = "live"
    # The model acts at its bar's close: the next one after the last bar it decided on.
    minutes = int(p.get("bar_minutes") or 1)
    step = minutes * 60
    last = (p.get("bar_ts") or 0) / 1e9
    t = now.timestamp()
    nxt = last + step * max(1, math.ceil((t - last) / step)) if last else (t // step + 1) * step
    left = max(0, int(nxt - t))
    view["close_at"] = int(nxt * 1000)
    view["close_in"] = f"{left // 3600}:{left // 60 % 60:02d}:{left % 60:02d}" if left >= 3600 else f"{left // 60:02d}:{left % 60:02d}"
    view["every"] = ("daily" if minutes == 1440 else f"{minutes // 60}-hour" if minutes % 60 == 0 else f"{minutes}-minute")
    return view


def _check_strategy_params(cfg: SleeveConfig, half_spread: float = 0.0) -> None:
    """Build the strategy config once so bad parameters fail here, not in the sleeve process. With the
    spread paper will charge, so a target that can't cover its costs is refused here at the same round
    trip the backtest quotes (review round 7: 1.61% here against 1.71% there)."""
    from nautilus_trader.model import BarType, InstrumentId

    check_perp_sizing(cfg.strategy, cfg.params)
    check_perp_stop(cfg.strategy, cfg.params, cfg.risk_profile)
    _, config_cls = REGISTRY[cfg.strategy]
    params = dict(cfg.params)
    params.pop("max_notional", None)
    config_cls(instrument_id=InstrumentId.from_str(cfg.instrument_id), bar_type=BarType.from_str(cfg.bar_type),
               assumed_taker_fee=float(cfg.fees.taker), assumed_half_spread=half_spread, **params)

