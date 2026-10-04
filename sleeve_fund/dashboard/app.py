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
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlencode, urlparse

import markdown
from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sleeve_fund import markets
from sleeve_fund.dashboard import book as bookm
from sleeve_fund.dashboard import gates, reports, riskops, trading
from sleeve_fund.dashboard.jobs import Jobs
from sleeve_fund.dashboard.metrics import STALE, sleeve_summary
from sleeve_fund.data import spec_minutes
from sleeve_fund.fees import resolve as resolve_fees
from sleeve_fund.history import REQUEST_YEARS
from sleeve_fund.instruments import price_decimals
from sleeve_fund.spreads import resolve as resolve_spread
from sleeve_fund.paper.config import ALLOWED_BAR_SPECS, SleeveConfig
from sleeve_fund.research import run as study_run
from sleeve_fund.research.ledger import IdeaLedger, opened_words
from sleeve_fund.risk import PROFILES
from sleeve_fund.store import BACKTEST_PREFIX, Store, is_backtest, utcnow
from sleeve_fund.strategies import REGISTRY
from sleeve_fund.strategies.base import exit_warmup, maker_orders_enabled

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TEARSHEETS = study_run.TEARSHEETS
LEDGER = study_run.LEDGER
# Suggestions only: the field accepts any instrument Kraken spot lists.
INSTRUMENT_HINTS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "SUI/USD", "ADA/USD", "DOGE/USD", "BTC/GBP", "ETH/GBP"]
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,40}$")
VERSION = os.environ.get("APP_VERSION", "dev")[:12]

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
    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.globals["maker_enabled"] = maker_orders_enabled
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
    from sleeve_fund.dashboard.glossary import GLOSSARY

    templates.env.globals["glossary"] = GLOSSARY
    templates.env.filters["action_words"] = action_words
    templates.env.filters["kind_words"] = kind_words
    templates.env.filters["rmult"] = lambda r: "–" if r is None else f"{r:+.2f}R"
    # The year only when it isn't this one, as a backtest's or an old journal's dates need it.
    templates.env.filters["ts"] = lambda t: (t.strftime("%d %b %H:%M UTC" if t.year == utcnow().year
                                                        else "%d %b %Y %H:%M UTC") if t else "never")
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    def st() -> Store:
        return app.state.store

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
        return page(request, "home.html", summaries=[x for x in summaries if x["sleeve"].name not in put_away],
                    archived=[x for x in summaries if x["sleeve"].name in put_away],
                    earlier=[st().sleeve(n) for n in earlier], book_start=st().book_start(),
                    book=bookm.book_view(st(), summaries, frames), alerts=st().alerts(limit=30), shell=shell(sleeves))

    def _recent_json(sleeves, days: int, daily) -> JSONResponse:
        """The last day or week at fine resolution, in the shape the charts read."""
        if days not in bookm.RECENT_STEP:
            raise HTTPException(400, "days must be 1 or 7")
        cutoff = utcnow() - timedelta(days=days)
        prior = daily["equity"][daily.index < cutoff] if len(daily) else daily
        curve = bookm.recent_curve(st(), sleeves, days, float(prior.max()) if len(prior) else None)
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
        return page(request, "risk.html", book=book, risk=riskops.risk_view(st(), summaries, book),
                    shell=shell(sleeves), kill=kill, reasons=COMMON_REASONS,
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
        return {"trading": trading, "held": held, "all": trading + held, "flattening": flattening,
                "stopped": [x["sleeve"].name for x in held if x["sleeve"].desired_state != "running"]}

    @app.post("/book/flatten")
    def book_flatten(reason: str = Form(""), actor: str = Depends(require_pm), _o: None = Depends(same_origin)):
        """The book's kill switch: every strategy still trading or holding a position sells to cash at
        market and pauses, each with the PM's reason in its decision log. A stopped strategy holding a
        position is started so its process can sell; it pauses once flat, as the others do."""
        why = reason.strip()
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

    @app.get("/ops", response_class=HTMLResponse)
    def ops_page(request: Request, _: str = Depends(require_pm)):
        sleeves, frames, summaries = book_data()
        return page(request, "ops.html", ops=riskops.ops_view(st(), summaries), shell=shell(sleeves))

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
        return page(request, "new_sleeve.html", strategies=_strategy_choices(), instruments=INSTRUMENT_HINTS,
                    bar_specs=sorted(ALLOWED_BAR_SPECS), profiles=PROFILES, error=error, g1=g1, chosen=chosen,
                    pre=dict(request.query_params), accounts=st().accounts(), costs=exit_costs())

    @app.post("/sleeves/new")
    async def new_sleeve(request: Request, actor: str = Depends(require_pm), _o: None = Depends(same_origin)):
        form = dict(await request.form())
        try:
            name = str(form.get("name", "")).strip()
            reason = str(form.get("reason", "")).strip()
            strategy = str(form.get("strategy", ""))
            if not NAME_RE.match(name):
                raise ValueError("name: lower-case letters, digits and dashes, 2 to 41 characters")
            if not reason:
                raise ValueError("a reason is required")
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
                               params=params, warmup_bars=warmup, risk_profile=str(form.get("risk_profile", "")))
            _check_strategy_params(cfg, resolve_spread(cfg.venue, cfg.instrument, st()).half_spread)
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
            st().create_sleeve(name=name, strategy=strategy, instrument=cfg.instrument, bar_spec=cfg.bar_spec,
                               starting_balance=cfg.starting_balance, params=params,
                               risk_profile=cfg.risk_profile, warmup_bars=cfg.warmup_bars)
            st().assign_account(name, account)
            st().decide(actor, "create", reason, name)
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
        trips = trading.trips(fills, st().events(name, limit=5000), orders, plans, markets.is_perp(s.params))
        feed = _feed(events, request.query_params.get("feed", "all"))
        recent = [trading.order_view(o) for o in st().orders(name, limit=15)]
        position = trading.open_position(x, fills, orders, plans)
        q = request.query_params
        # The settings form: what was typed when a change was refused, else the settings as they are.
        typed = {k[2:]: v for k, v in q.items() if k.startswith("f_")}
        settings_pre = typed or {"risk_profile": s.risk_profile, **_risk_form(s.params)}
        return page(request, "sleeve.html", x=x, fills=fills[:200], trips=trips, feed=feed, orders=recent,
                    account=st().account_of(name), accounts=st().accounts(), settings_pre=settings_pre,
                    settings_error=q.get("settings_error", ""), saved=q.get("saved", ""), profiles=PROFILES,
                    command_error=q.get("command_error", ""),
                    # The journal's position, as the flatten and account checks read it: a stopped strategy's
                    # last mark may be older than its last fill.
                    held=0.0 if bt_id else st().journal_book(name, s.starting_balance)["qty"],
                    costs=exit_costs(), reload=st().pending_reload(name),
                    position=position,
                    feed_kind=request.query_params.get("feed", "all"), decisions=st().decisions(name, limit=50),
                    pending=st().pending_commands(name), risk=_risk_view(x, position), reasons=COMMON_REASONS,
                    idea=_idea(s.strategy, s.params), archived=name in st().archived(),
                    clone_qs=_clone_qs(s), backtest_id=bt_id, tested=_tested(bt_id),
                    strategy_errors=st().strategy_errors(name, since_start=not bt_id),
                    path=None if bt_id else gates.path_to_live(st(), x, _g1_of(s.strategy, s.instrument, spec_minutes(s.bar_spec)),
                                                               st().accounts(), utcnow()))

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
                df, note = charts.candles(pair, minutes), ""
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
            df, source = charts.candles(s.instrument, minutes), "venue"
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
        data.update(pair=s.instrument, home=s.instrument, pairs=_chart_pairs(s.instrument, [b.instrument for b in st().sleeves()]))
        if is_backtest(name):
            data["note"] = "Candles built from the run's price marks."
        return JSONResponse(data)

    @app.get("/api/instruments")
    def instruments_json(_: str = Depends(require_pm)):
        from sleeve_fund.dashboard import charts

        try:
            return JSONResponse({"instruments": charts.instruments(), "source": "venue"})
        except (OSError, ValueError, KeyError):  # venue unreachable: the usual ones, and any other can still be typed
            return JSONResponse({"instruments": INSTRUMENT_HINTS, "source": "fallback"})

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
        peak, dd = 0.0, []
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
            "worst": round(st().max_drawdown(name), 5),  # over every mark: the curve above is thinned or daily
            "fills": fills,
        })

    @app.post("/sleeves/{name}/command")
    def sleeve_command(name: str, command: str = Form(...), reason: str = Form(...),
                       actor: str = Depends(require_pm), _o: None = Depends(same_origin)):
        try:
            if command in ("start", "stop"):
                if not reason.strip():
                    raise ValueError("a reason is required")
                if command == "start" and _retired(st().account_of(name)):
                    raise ValueError(f"its account {st().account_of(name)} is retired; move it to another "
                                     "account or reinstate that one first")
                st().set_desired_state(name, "running" if command == "start" else "stopped")
                if command == "stop":
                    # A command still waiting when its process stops would act on the next start, maybe
                    # weeks later; it lapses instead, and the decision log says so.
                    st().drop_pending(name, "lapsed: the strategy was stopped before it acted")
                st().decide(actor, command, reason, name)
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
        return RedirectResponse(f"/sleeves/{name}", status_code=303)

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
            reason = str(form.get("reason", "")).strip()
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
            changes = _risk_changes(s.risk_profile, s.params, profile, params)
            if not changes:
                raise ValueError("nothing changed")
            cap = MAX_STORED_WARMUP_BARS if s.bar_spec.endswith("INTERNAL") else MAX_WARMUP_BARS
            warmup = max(s.warmup_bars, min(cap, exit_warmup(params)))
            restart = st().change_settings(name, risk_profile=profile, params=params, warmup_bars=warmup)
            text = "; ".join(changes)
            st().decide(actor, "change_settings", f"{text}. {reason}", name)
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
    def sleeve_account(name: str, account: str = Form(...), reason: str = Form(...),
                       actor: str = Depends(require_pm), _o: None = Depends(same_origin)):
        try:
            s = st().sleeve(name)
        except KeyError:
            raise HTTPException(404, "no such strategy") from None
        try:
            if not reason.strip():
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
    def sleeve_archive(name: str, action: str = Form(...), reason: str = Form(...),
                       actor: str = Depends(require_pm), _o: None = Depends(same_origin)):
        try:
            if not reason.strip():
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

    def research_page(request: Request, job=None, error: str = "", pre: dict | None = None, collect: str = "",
                      notice: str = ""):
        from sleeve_fund.dashboard import pipeline

        sheets = [pipeline.sheet_facts(p) for p in sorted(TEARSHEETS.glob("*.md"), key=lambda p: p.stat().st_mtime,
                                                         reverse=True)]
        stored = _stored_history(st())
        ledger = IdeaLedger(LEDGER)
        spent = {f"{idea}|{base}": opened_words(e) for (idea, base), e in ledger.holdouts().items()}
        return page(request, "research.html", sheets=sheets, counts=ledger.counts(),
                    rows=pipeline.strategies(TEARSHEETS, st().sleeves()), stages=pipeline.STAGES, job=job,
                    error=error, pre=pre or {}, strategies=_strategy_choices(),
                    instruments=[h["pair"] for h in stored] or INSTRUMENT_HINTS, stored=stored, collect=collect,
                    notice=notice, request_years=REQUEST_YEARS, history_venue=_research_venue().label,
                    research_venue=_research_venue().name.lower(), spent_holdouts=spent,
                    profiles=PROFILES, study_minutes=study_run.STUDY_MINUTES, costs=exit_costs())

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
        # No history yet: say so now, with the way to get it, rather than from a failed job (R8-M6).
        held = next((h for h in _stored_history(st()) if h["pair"] == req.pair), None)
        if held is None or held["first"] is None:
            label = _research_venue().label
            why = (f"{req.pair}'s history was asked for on {held['requested']:%d %b %Y}; the collector hasn't stored "
                   "any yet." if held else f"there is no stored {label} history for {req.pair} yet.")
            return research_page(request, error=why, pre=form, collect="" if held else req.pair)
        jobs = app.state.jobs
        target = st().url if jobs.isolate and st().url else st()
        key = "study|" + "|".join(f"{k}={v}" for k, v in sorted(vars(req).items()))
        title = f"G1 study of {req.strategy.replace('_', ' ')} on {req.pair}, {study_run._bars(req.minutes)} bars"
        job = jobs.submit(key, title, run_study_job, target, req, str(LEDGER), str(TEARSHEETS))
        return RedirectResponse(f"/research?{urlencode({'job': job.id, **form})}", status_code=303)

    @app.post("/research/history")
    async def research_history(request: Request, _: str = Depends(require_pm), _o: None = Depends(same_origin)):
        """Ask the history collector to store an instrument, so a study can run on it before any strategy
        trades it (review round 8, R8-M6)."""
        form = dict((await request.form()).items())
        pair = str(form.get("instrument", "")).strip().upper()
        profile = _research_venue()
        try:
            if not re.fullmatch(r"[A-Z0-9]{1,12}/[A-Z0-9]{2,6}", pair):
                raise ValueError(f"{pair or 'that'} isn't an instrument: write it as base and quote with a slash, "
                                 "like BTC/USD")
            held = next((h for h in _stored_history(st()) if h["pair"] == pair and h["first"] is not None), None)
            if held is not None:
                return research_page(request, pre={"instrument": pair}, notice=(
                    f"{pair} is already stored, from {held['first']:%d %b %Y} to {held['last']:%d %b %Y %H:%M} UTC "
                    f"({held['state']}); the collector keeps it current, and a study can run on it now."))
            if profile.minute_loader is None:
                raise ValueError(f"{profile.label} has no history loader")
            if profile.check_listed is not None:
                try:
                    profile.check_listed(pair)
                except ValueError:
                    raise
                except Exception as exc:  # noqa: BLE001 - venue unreachable
                    # Asked for unchecked, a pair the venue doesn't list sat "Asked for" for good, with no way
                    # to take it back (review round 9, N5): nothing is asked for until the venue confirms it.
                    logging.getLogger(__name__).warning(f"couldn't check {profile.label} lists {pair}: {exc!r}")
                    raise ValueError(f"couldn't reach {profile.label} to check it lists {pair}, so nothing was asked "
                                     "for; try again when the venue answers") from None
        except ValueError as exc:
            # No button to ask again: the same request would be refused the same way.
            return research_page(request, error=str(exc), pre={"instrument": pair})
        since = (utcnow() - timedelta(days=365 * REQUEST_YEARS)).replace(hour=0, minute=0, second=0, microsecond=0)
        new = st().request_history(profile.name, pair, since)
        notice = (f"Asked the collector for {pair}: it backfills from {since:%d %b %Y}, then keeps it current. "
                  "A study can run once some is stored; this list shows how far it has got."
                  if new else f"{pair} was already asked for; this list shows how far the collector has got.")
        return research_page(request, pre={"instrument": pair}, notice=notice)

    @app.get("/strategies/{name}", response_class=HTMLResponse)
    def strategy_page(request: Request, name: str, _: str = Depends(require_pm)):
        from sleeve_fund.dashboard import pipeline

        row = next((r for r in pipeline.strategies(TEARSHEETS, st().sleeves()) if r["name"] == name), None)
        if row is None:
            raise HTTPException(404, "no such strategy")
        return page(request, "strategy.html", r=row, stages=pipeline.STAGES, summary=_idea(name))

    @app.get("/research/{sheet}", response_class=HTMLResponse)
    def tearsheet(request: Request, sheet: str, _: str = Depends(require_pm)):
        path = (TEARSHEETS / f"{sheet}.md").resolve()
        if path.parent != TEARSHEETS.resolve() or not path.exists():
            raise HTTPException(404, "no such tear sheet")
        html = markdown.markdown(path.read_text(encoding="utf-8"), extensions=["tables"])
        # Results as status chips, so a FAIL can't be missed in a wall of text.
        for word, tone, label in (("PASS", "running", "Pass"), ("FAIL", "halted", "Fail"), ("WARN", "paused", "Warn"),
                                  ("INFO", "stopped", "Info"), ("NOT JUDGED", "paused", "Not judged"),
                                  ("N/A", "stopped", "Not judged")):
            html = html.replace(f"<td>{word}</td>", f'<td><span class="chip {tone}">{label}</span></td>')
        return page(request, "tearsheet.html", title=sheet, body=html)

    @app.get("/decisions", response_class=HTMLResponse)
    def decisions(request: Request, _: str = Depends(require_pm)):
        f = _decision_filters(request)
        return page(request, "decisions.html", decisions=st().decisions(limit=500, **f["query"]), f=f,
                    sleeves=[s.name for s in st().sleeves()], actions=DECISION_ACTIONS)

    @app.get("/decisions.csv")
    def decisions_csv(request: Request, _: str = Depends(require_pm)):
        rows = st().decisions(limit=100_000, **_decision_filters(request)["query"])
        return _csv("decisions", reports.to_csv(rows, ["ts", "actor", "action", "sleeve", "reason"]))

    @app.get("/reports", response_class=HTMLResponse)
    def reports_page(request: Request, _: str = Depends(require_pm)):
        sleeves, frames, summaries = book_data()
        fills = {s.name: st().fills(s.name, limit=100_000) for s in sleeves}
        return page(request, "reports.html", m=reports.monthly(summaries, frames, fills),
                    sleeves=[s.name for s in sleeves], shell=shell(sleeves))

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
                                       st().exit_plans(n), _shorts(st(), n)):
                    rows.append({"sleeve": n, **t, "held_hours": round(t["held"].total_seconds() / 3600, 2)
                                 if t["held"] else None})
            cols = ["sleeve", "opened", "closed", "held_hours", "qty", "entry_px", "exit_px", "cost", "fees", "pnl",
                    "ret", "r", "planned_r", "exits_edited", "exit_kind", "entry_why", "exit_why", "entry_order",
                    "exit_order"]
        elif kind == "orders":
            rows = [dict(o, signal=json.dumps(o["signal"], sort_keys=True))
                    for n in chosen for o in reversed(st().orders(n, limit=1_000_000))]
            cols = ["ts", "sleeve", "order_id", "side", "order_type", "qty", "status", "filled_qty", "avg_px", "fee",
                    "intent", "reason", "signal", "message", "updated_at"]
        else:
            raise HTTPException(404, "unknown export")
        return _csv(f"{kind}-{sleeve or 'all'}", reports.to_csv(rows, cols))

    def exit_costs() -> dict:
        """What each leg of a trade costs, for the plan line under the exit fields: the taker fee and
        each instrument's half spread (measured, else the venue's assumption), exactly as the backtest
        and paper charge them, so the form and the backtest quote the same round trip."""
        default = resolve_spread(None, "?/?", None).half_spread
        return {"taker": float(resolve_fees(None, st()).fees.taker), "default_spread": default,
                "spreads": {p: resolve_spread(None, p, st()).half_spread for p in INSTRUMENT_HINTS}}

    def backtest_form(request: Request, q, *, result=None, error="", job=None, saved=None):
        """The backtest page: the settings form, plus a result, an error, or a run in progress."""
        from sleeve_fund.dashboard import pipeline

        strategy = q.get("strategy") if q.get("strategy") in REGISTRY else "trend_filter"
        period = q.get("period") if q.get("period") in BACKTEST_PERIODS else "all"
        bar_spec = q.get("bar_spec") if q.get("bar_spec") in ALLOWED_BAR_SPECS else BACKTEST_BAR_SPEC
        # The sleeve must decide on the bars that were tested, with indicators warm from its first bar.
        carry = {k: v for k, v in q.items() if k not in ("run", "period") and v}
        carry.update(bar_spec=bar_spec, tested_bar_spec=bar_spec, warmup_bars=_warmup_for(strategy, q, bar_spec))
        g1 = {r["name"]: "|".join(r["passed_on"]) for r in pipeline.strategies(TEARSHEETS, st().sleeves())}
        pair = q.get("instrument", "").strip().upper()
        g1_here = pipeline.g1_for(TEARSHEETS, strategy, pair, spec_minutes(bar_spec)) if pair else None
        chart = None
        if result:
            chart = {"t": result["t"], "equity": result["equity"], "benchmark": result["benchmark"],
                     "drawdown": result["drawdown"], "fills": result["fills"], "res": "daily",  # equity is daily
                     "worst": round(-result["strategy"]["max_drawdown"], 5)}  # over every mark, as the table
        return page(request, "backtest.html", result=result, error=error, job=job, saved=saved, pre=dict(q),
                    chosen=strategy, strategies=_strategy_choices(), instruments=INSTRUMENT_HINTS, g1=g1, g1_here=g1_here,
                    period=period, periods=BACKTEST_PERIODS, profiles=PROFILES, bar_spec=bar_spec,
                    bar_specs=sorted(ALLOWED_BAR_SPECS, key=spec_minutes),
                    sleeve_qs=urlencode({**carry, "from": "backtest"}), chart=chart, stored=_stored(),
                    runs=st().backtests(limit=BACKTEST_KEEP), costs=exit_costs())

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
        key = _backtest_key(q, resolve_fees(None, st()).text, resolve_spread(None, args["pair"], st()).text)
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
                                        trading.orders_by_id(st(), name), shorts=_shorts(st(), name))
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

    @app.get("/accounts", response_class=HTMLResponse)
    def accounts_page(request: Request, _: str = Depends(require_pm), error: str = ""):
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
        return page(request, "accounts.html", accounts=rows, error=error, pre=dict(request.query_params))

    @app.post("/accounts/new")
    async def new_account(request: Request, actor: str = Depends(require_pm), _o: None = Depends(same_origin)):
        form = dict(await request.form())
        name = str(form.get("name", "")).strip()
        try:
            reason = str(form.get("reason", "")).strip()
            if not reason:
                raise ValueError("a reason is required")
            st().create_account(name, str(form.get("kind", "")), str(form.get("note", "")).strip()[:200])
            st().decide(actor, "create_account", f"{form.get('kind')} account {name}: {reason}")
        except ValueError as exc:
            kept = {k: str(v) for k, v in form.items() if isinstance(v, str) and v}
            return RedirectResponse(f"/accounts?{urlencode({'error': str(exc), **kept})}#add", status_code=303)
        return RedirectResponse(f"/accounts#acct-{name}", status_code=303)

    @app.post("/accounts/{name}/note")
    def account_note(name: str, note: str = Form(""), reason: str = Form(...),
                     actor: str = Depends(require_pm), _o: None = Depends(same_origin)):
        try:
            if not reason.strip():
                raise ValueError("a reason is required")
            st().set_account_note(name, note)
            st().decide(actor, "account_note", f"{name}: note set to \"{note.strip()[:200]}\". {reason.strip()}")
        except ValueError as exc:
            return RedirectResponse(f"/accounts?{urlencode({'error': str(exc)})}", status_code=303)
        return RedirectResponse(f"/accounts?saved={name}#acct-{name}", status_code=303)

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
            return RedirectResponse(f"/accounts?{urlencode({'error': str(exc)})}", status_code=303)
        return RedirectResponse(f"/accounts?saved={name}#acct-{name}", status_code=303)

    @app.get("/settings", response_class=HTMLResponse)
    def settings_page(request: Request, _: str = Depends(require_pm)):
        from sleeve_fund.venues import VENUES

        fee_quotes = [{"venue_label": q.venue_label, "source": q.source,
                       "rates": f"{float(q.fees.maker):.2%} maker / {float(q.fees.taker):.2%} taker",
                       "basis": q.basis if q.source == "published" else
                       f"account {q.account}, {q.fetched_at:%d %b %Y %H:%M} UTC",
                       "assumed_spread": f"{2 * v.assumed_half_spread:.2%}"}
                      for v in VENUES.values() for q in [resolve_fees(v.name, st())]]

        return page(request, "settings.html", profiles=PROFILES, venues=VENUES.values(), fee_quotes=fee_quotes,
                    tearsheets=str(TEARSHEETS), counts=st().table_sizes(), accounts=st().accounts())

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
    pair = q.get("instrument", "").strip().upper()
    if not PAIR_RE.fullmatch(pair):
        raise ValueError("instrument: write it as BASE/QUOTE, for example SOL/USD")
    starting = float(q.get("starting_balance") or 10_000)
    if not 100 <= starting <= 1e9:
        raise ValueError("capital: between 100 and 1,000,000,000")
    params = _form_params(q, strategy)
    _profile_cap(q)  # validates the profile name
    if spec_minutes(bar_spec) < 1440:
        have = [r["pair"] for r in _stored()]
        if pair not in have:
            raise ValueError(f"interval: {pair} has no stored minute history here, so it can only be backtested on "
                             f"daily bars. Instruments with stored minutes: {', '.join(have) or 'none yet'}.")
    title = (f"{strategy.replace('_', ' ').capitalize()} on {pair}, {_bar_short(bar_spec)}, "
             f"{BACKTEST_PERIODS[period][0].lower()}")
    return {"strategy": strategy, "pair": pair, "params": params, "starting": starting,
            "days": BACKTEST_PERIODS[period][1], "minutes": spec_minutes(bar_spec),
            "risk_profile": q.get("risk_profile") or "balanced", "title": title, "bar_spec": bar_spec}


def _stored() -> list[dict]:
    from sleeve_fund.dashboard import preview

    try:
        return preview.stored()
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
                         fee_quote=resolve_fees(None, store), spread_quote=resolve_spread(None, args["pair"], store),
                         progress=progress, keep=keep)
    result.pop("trips", None)  # rebuilt from the saved journal, as the Trades screen does
    store.save_backtest(keep["journal"], run_id=run_id, key=key, title=args["title"], query=query, result=result,
                        bar_spec=args["bar_spec"])
    store.prune_backtests(keep=BACKTEST_KEEP)
    return run_id

LOST_JOB = ("that run is no longer known, most likely because the server restarted while it ran; "
            "run it again (review round 10, m9)")


def _qty(x: float) -> str:
    """A quantity to six significant figures, without exponents: 23,350.1 and 0.0765 rather than 2.335e+04
    (review round 10, m3)."""
    if not x:
        return "0"
    d = min(8, max(0, 5 - math.floor(math.log10(abs(x)))))
    return f"{x:,.{d}f}".rstrip("0").rstrip(".") if d else f"{x:,.0f}"


def _research_venue():
    from sleeve_fund.venues import venue

    return venue(None)


def _stored_history(store: Store) -> list[dict]:
    """Each instrument research can use or has asked for, on the research venue: what is stored, and
    whether the collector is current or still catching up."""
    from sleeve_fund.history import HistoryStore

    profile = _research_venue()
    hist = HistoryStore()
    now = utcnow()
    asked = {r["instrument"]: r for r in store.history_requests(profile.name)}
    out = []
    for v, pair in hist.series():
        if v != profile.name:
            continue
        cov = hist.coverage(v, pair)
        behind = now - cov.last.to_pydatetime() > study_run.STALE_HISTORY
        out.append({"pair": pair, "first": cov.first, "last": cov.last, "requested": asked.get(pair, {}).get("requested_at"),
                    "state": "catching up" if behind else "current"})
    held = {h["pair"] for h in out}
    out += [{"pair": p, "first": None, "last": None, "requested": r["requested_at"], "state": "asked for"}
            for p, r in asked.items() if p not in held]
    return sorted(out, key=lambda h: h["pair"])


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
        strategy=strategy, pair=str(form.get("instrument", "")).strip().upper(),
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


COMMON_REASONS = [
    "Risk limit close; reducing exposure",
    "Market event; standing aside",
    "Strategy behaving outside its backtest range",
    "Data or venue problem",
    "Checked after an alert; safe to continue",
    "Planned change of settings",
]
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
    return {
        "stop_px": stop_px,
        "target_px": position["target_px"] if position else None,
        # The move from here to the stop (round 9, N3): a drop for a long, a rise for a short.
        "to_stop": abs(1 - stop_px / x["price"]) if stop_px and x["price"] else None,
        "cap_used": min(abs(x["exposure"]) / p.max_position_pct, 1.0) if p.max_position_pct else 0.0,
        "day_used": min(max(-x["day_ret"], 0.0) / p.daily_loss, 1.0) if p.daily_loss else 0.0,
    }


def _held(td) -> str:
    if td is None:
        return ""
    secs = max(td.total_seconds(), 0.0)  # a file stamped a moment after the clock read is 0, not -0
    hours = secs / 3600
    return f"{hours / 24:.1f} d" if hours >= 48 else f"{hours:.0f} h" if hours >= 1 else f"{secs / 60:.0f} min"


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


def _bar_short(spec: str) -> str:
    step, unit = spec.split("-")[:2]
    return f"{step}{ {'SECOND': 's', 'MINUTE': 'm', 'HOUR': 'h', 'DAY': 'd'}.get(unit, unit.lower())} bars"


def _bar_label(spec: str) -> str:
    step, unit, _, source = spec.split("-")
    unit = unit.lower() + ("s" if step != "1" else "")
    how = "built from live trades" if source == "INTERNAL" else "Kraken candles, supports warm-up"
    return f"{step} {unit} ({how})"


def _strategy_choices() -> list[dict]:
    import importlib

    out = []
    for name in sorted(REGISTRY):
        spec = importlib.import_module(f"sleeve_fund.strategies.{name}").SPEC
        out.append({"name": name, "idea": _idea(name), "params": spec.default_params, "family": spec.family,
                    "tpl": spec.summary, "defaults": _config_defaults(name)})
    return out


def _g1_of(strategy: str, instrument: str, minutes: int) -> str | None:
    from sleeve_fund.dashboard import pipeline

    return pipeline.g1_for(TEARSHEETS, strategy, instrument, minutes)


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


def _risk_words(profile: str, params: dict) -> dict[str, str]:
    """Each risk setting in words, for the decision log's before and after."""
    p = params
    if p.get("stop_atr"):
        stop = f"{p['stop_atr']:g} average true ranges ({p.get('atr_bars', 14)} bars) below the entry"
    elif p.get("stop_swing_bars"):
        stop = f"at the lowest low of {p['stop_swing_bars']} bars"
    elif p.get("stop_loss"):
        stop = f"{p['stop_loss'] * 100:g}% below the entry"
    else:
        stop = "none"
    target = (f"{p['take_profit_r']:g}R after costs" if p.get("take_profit_r")
              else f"{p['take_profit'] * 100:g}% above the entry" if p.get("take_profit") else "none")
    return {"Risk profile": profile, "Stop-loss": stop, "Take-profit": target,
            "Risk per trade": f"{p['risk_per_trade'] * 100:g}%" if p.get("risk_per_trade") else "none",
            "Largest order": f"{p['max_notional']:,.2f}" if p.get("max_notional") else "no cap"}


def _risk_changes(old_profile: str, old: dict, new_profile: str, new: dict) -> list[str]:
    before, after = _risk_words(old_profile, old), _risk_words(new_profile, new)
    return [f"{k} {before[k]} to {after[k]}" for k in before if before[k] != after[k]]


def _clone_qs(s) -> str:
    """The new-sleeve form filled in with this sleeve's settings, for "Clone with changes"."""
    params = {k: v for k, v in s.params.items() if k not in RISK_KEYS}
    q = {"strategy": s.strategy, "instrument": s.instrument, "bar_spec": s.bar_spec,
         "starting_balance": f"{s.starting_balance:g}", "risk_profile": s.risk_profile, "warmup_bars": s.warmup_bars,
         "name": f"{s.name[:38]}-v2", "from": "clone", "source": s.name, **_risk_form(s.params)}
    if "maker_wait_minutes" in params:
        q.update(execution="maker", maker_wait_minutes=params.pop("maker_wait_minutes"))
    q.update({f"p_{s.strategy}__{k}": v for k, v in params.items()})
    return urlencode(q)


# The backtest page's default interval. A sleeve made from a backtest decides on the bars it tested.
BACKTEST_BAR_SPEC = "1-DAY-LAST-EXTERNAL"
MAX_WARMUP_BARS = 720  # what the venue returns in one request; bars built from trades load from the store


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
    """Bars to load at start so the slowest indicator is ready on the sleeve's first bar, as the
    strategy itself says. Venue candles are capped at what the venue returns in one request; bars
    built from live trades load from the history store, up to the sleeve limit."""
    from sleeve_fund.paper.config import MAX_WARMUP_BARS as MAX_STORED_WARMUP_BARS

    params = _defaults(strategy)
    try:
        params.update(_form_params(q, strategy))
    except ValueError:
        pass
    cap = MAX_STORED_WARMUP_BARS if bar_spec.endswith("INTERNAL") else MAX_WARMUP_BARS
    return min(cap, max(REGISTRY[strategy][0].warmup_needed(params, spec_minutes(bar_spec)), exit_warmup(params)))


def _form_params(form, strategy: str) -> dict:
    """Strategy parameters and exits from the new-sleeve form (also used by the preview)."""
    # Each strategy's parameter inputs are named p_<strategy>__<param>; only the chosen one counts.
    prefix = f"p_{strategy}__"
    params = _coerce_params({k[len(prefix):]: v for k, v in form.items() if k.startswith(prefix) and v != ""})
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


def _check_strategy_params(cfg: SleeveConfig, half_spread: float = 0.0) -> None:
    """Build the strategy config once so bad parameters fail here, not in the sleeve process. With the
    spread paper will charge, so a target that can't cover its costs is refused here at the same round
    trip the backtest quotes (review round 7: 1.61% here against 1.71% there)."""
    from nautilus_trader.model import BarType, InstrumentId

    _, config_cls = REGISTRY[cfg.strategy]
    params = dict(cfg.params)
    params.pop("max_notional", None)
    config_cls(instrument_id=InstrumentId.from_str(cfg.instrument_id), bar_type=BarType.from_str(cfg.bar_type),
               assumed_taker_fee=float(cfg.fees.taker), assumed_half_spread=half_spread, **params)

