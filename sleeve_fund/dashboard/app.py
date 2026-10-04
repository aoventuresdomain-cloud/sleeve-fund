"""PM dashboard v1: see the book, start and stop paper sleeves, pause/flatten/resume,
read tear sheets and the decision log.

Read-mostly. Every control that changes a sleeve asks for a reason and lands in
the decision log. Protected by one PM password (HTTP Basic, behind HTTPS).

    DASHBOARD_PASSWORD=... uvicorn --factory sleeve_fund.dashboard.app:create_app
"""

from __future__ import annotations

import json
import os
import re
import secrets
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlencode, urlparse

import markdown
from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sleeve_fund.dashboard import book as bookm
from sleeve_fund.dashboard import gates, reports, riskops, trading
from sleeve_fund.dashboard.jobs import Jobs
from sleeve_fund.dashboard.metrics import STALE, sleeve_summary
from sleeve_fund.data import spec_minutes
from sleeve_fund.fees import resolve as resolve_fees
from sleeve_fund.spreads import resolve as resolve_spread
from sleeve_fund.paper.config import ALLOWED_BAR_SPECS, SleeveConfig
from sleeve_fund.research.ledger import IdeaLedger
from sleeve_fund.risk import PROFILES
from sleeve_fund.store import BACKTEST_PREFIX, Store, is_backtest, utcnow
from sleeve_fund.strategies import REGISTRY

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TEARSHEETS = Path(os.environ.get("TEARSHEET_DIR", ROOT / "research" / "tearsheets"))
LEDGER = Path(os.environ.get("IDEA_LEDGER", ROOT / "research" / "idea_ledger.jsonl"))
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
    templates = Jinja2Templates(directory=HERE / "templates")
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
    templates.env.filters["px"] = lambda x: "n/a" if x is None or x != x else (f"{x:,.2f}" if x >= 100 else f"{x:,.4f}")
    templates.env.globals["bar_label"] = _bar_label
    templates.env.globals["bar_short"] = _bar_short
    from sleeve_fund.dashboard.glossary import GLOSSARY

    templates.env.globals["glossary"] = GLOSSARY
    templates.env.filters["rmult"] = lambda r: "–" if r is None else f"{r:+.2f}R"
    # The year only when it isn't this one, as a backtest's or an old journal's dates need it.
    templates.env.filters["ts"] = lambda t: (t.strftime("%d %b %H:%M UTC" if t.year == utcnow().year
                                                        else "%d %b %Y %H:%M UTC") if t else "never")
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    def st() -> Store:
        return app.state.store

    def shell(sleeves=None) -> dict:
        """What the frame shows on every page: mode, health and open alerts."""
        sleeves = st().sleeves() if sleeves is None else sleeves
        now = utcnow()
        wanted = [x for x in sleeves if x.desired_state == "running"]
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
        sleeves = st().sleeves()
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
        return page(request, "home.html", summaries=[x for x in summaries if x["sleeve"].name not in put_away],
                    archived=[x for x in summaries if x["sleeve"].name in put_away],
                    book=bookm.book_view(st(), summaries, frames), alerts=st().alerts(limit=30), shell=shell(sleeves))

    @app.get("/api/book/equity")
    def book_equity_json(_: str = Depends(require_pm)):
        _, frames, summaries = book_data()
        active = [x for x in summaries if x["sleeve"].desired_state == "running"] or summaries
        curve = bookm.book_curve(active, frames)
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
        trading = [x for x in summaries if x["sleeve"].desired_state == "running" and x["sleeve"].status != "halted"]
        held = [x for x in summaries if x["qty"] > 0 and x not in trading]
        return {"trading": trading, "held": held, "all": trading + held,
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
                    pre=dict(request.query_params), accounts=st().accounts())

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
            _check_strategy_params(cfg)
            needed = REGISTRY[strategy][0].warmup_needed({**_defaults(strategy), **params}, spec_minutes(bar_spec))
            if any(s.name == name for s in st().sleeves()):
                raise ValueError(f"a strategy called {name} already exists")
            account = str(form.get("account", "") or "paper")
            kinds = {a["name"]: a["kind"] for a in st().accounts()}
            if account not in kinds:
                raise ValueError(f"account: no account called {account}")
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
        trips = trading.trips(fills, st().events(name, limit=5000), orders)
        feed = _feed(events, request.query_params.get("feed", "all"))
        recent = [trading.order_view(o) for o in st().orders(name, limit=15)]
        return page(request, "sleeve.html", x=x, fills=fills[:200], trips=trips, feed=feed, orders=recent,
                    account=st().account_of(name),
                    position=trading.open_position(x, fills, orders),
                    feed_kind=request.query_params.get("feed", "all"), decisions=st().decisions(name, limit=50),
                    pending=st().pending_commands(name), risk=_risk_view(x), reasons=COMMON_REASONS,
                    idea=_idea(s.strategy, s.params), archived=name in st().archived(),
                    clone_qs=_clone_qs(s), backtest_id=bt_id, tested=_tested(bt_id),
                    path=None if bt_id else gates.path_to_live(st(), x, _g1_of(s.strategy, s.instrument, spec_minutes(s.bar_spec)),
                                                               st().accounts(), utcnow()))

    @app.get("/api/sleeves/{name}/candles")
    def candles_json(name: str, interval: str = "", _: str = Depends(require_pm)):
        from sleeve_fund.dashboard import charts

        try:
            s = st().sleeve(name)
        except KeyError:
            raise HTTPException(404, "no such strategy") from None
        interval = interval if interval in charts.INTERVALS else charts.default_interval(s.bar_spec)
        minutes = charts.INTERVALS[interval]
        try:
            if is_backtest(name):  # the venue's recent candles aren't the replayed period
                raise ValueError("backtest")
            df, source = charts.candles(s.instrument, minutes), "venue"
        except (OSError, ValueError):  # venue unreachable or pair unknown: chart the sleeve's own marks
            df, source = charts.from_marks(st().equity_series(name, limit=500_000), minutes), "marks"
        fills = list(reversed(st().fills(name, limit=100_000)))
        orders = trading.orders_by_id(st(), name)
        x = bookm.sleeve_extras(st(), sleeve_summary(st(), s), bookm.daily(st(), name))
        position = trading.open_position(x, list(reversed(fills)), orders)
        # A live screen shows the latest candles; a backtest's chart covers its whole period.
        data = charts.payload(df, minutes, fills, orders, charts.position_lines(position), source,
                              limit=None if is_backtest(name) else 720)
        data["intervals"], data["chosen"] = list(charts.INTERVALS), interval
        if is_backtest(name):
            data["note"] = "Candles built from the run's price marks."
        return JSONResponse(data)

    @app.get("/api/sleeves/{name}/equity")
    def equity_json(name: str, _: str = Depends(require_pm)):
        try:
            s = st().sleeve(name)
        except KeyError:
            raise HTTPException(404, "no such strategy") from None
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
                st().set_desired_state(name, "running" if command == "start" else "stopped")
                if command == "stop":
                    # A command still waiting when its process stops would act on the next start, maybe
                    # weeks later; it lapses instead, and the decision log says so.
                    st().drop_pending(name, "lapsed: the strategy was stopped before it acted")
                st().decide(actor, command, reason, name)
            else:
                st().command(name, command, reason, actor=actor)
        except KeyError:
            raise HTTPException(404, "no such strategy") from None
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return RedirectResponse(f"/sleeves/{name}", status_code=303)

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

    @app.get("/research", response_class=HTMLResponse)
    def research(request: Request, _: str = Depends(require_pm)):
        from sleeve_fund.dashboard import pipeline

        sheets = [pipeline.sheet_facts(p) for p in sorted(TEARSHEETS.glob("*.md"), key=lambda p: p.stat().st_mtime,
                                                         reverse=True)]
        return page(request, "research.html", sheets=sheets, counts=IdeaLedger(LEDGER).counts(),
                    rows=pipeline.strategies(TEARSHEETS, st().sleeves()), stages=pipeline.STAGES)

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
        for word, tone in (("PASS", "running"), ("FAIL", "halted"), ("WARN", "paused"), ("INFO", "stopped")):
            html = html.replace(f"<td>{word}</td>", f'<td><span class="chip {tone}">{word.capitalize()}</span></td>')
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
                for t in trading.trips(st().fills(n, limit=1_000_000), st().events(n, limit=5000), orders):
                    rows.append({"sleeve": n, **t, "held_hours": round(t["held"].total_seconds() / 3600, 2)
                                 if t["held"] else None})
            cols = ["sleeve", "opened", "closed", "held_hours", "qty", "entry_px", "exit_px", "cost", "fees", "pnl",
                    "ret", "exit_kind", "entry_why", "exit_why", "entry_order", "exit_order"]
        elif kind == "orders":
            rows = [dict(o, signal=json.dumps(o["signal"], sort_keys=True))
                    for n in chosen for o in reversed(st().orders(n, limit=1_000_000))]
            cols = ["ts", "sleeve", "order_id", "side", "order_type", "qty", "status", "filled_qty", "avg_px", "fee",
                    "intent", "reason", "signal", "message", "updated_at"]
        else:
            raise HTTPException(404, "unknown export")
        return _csv(f"{kind}-{sleeve or 'all'}", reports.to_csv(rows, cols))

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
                    runs=st().backtests(limit=BACKTEST_KEEP))

    @app.get("/backtest", response_class=HTMLResponse)
    def backtest_page(request: Request, _: str = Depends(require_pm)):
        """Run any strategy and settings over the venue's history on the paper runtime. The run goes to
        the background; a quick one comes straight back as its saved result, a long one shows progress."""
        q = request.query_params
        if not q.get("run"):
            return backtest_form(request, q)
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
                                        trading.orders_by_id(st(), name))
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
        for r in rows:
            r["env"] = acc.env_names(r["name"], r["venue"]) if r["kind"] == "live" else None
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


def _risk_view(x: dict) -> dict:
    p, params = x["profile"], x["sleeve"].params
    entry = x["entry_px"]
    sl, tp = params.get("stop_loss"), params.get("take_profit")
    return {
        "stop_px": entry * (1 - sl) if entry and sl else None,
        "target_px": entry * (1 + tp) if entry and tp else None,
        "to_stop": (x["price"] / (entry * (1 - sl)) - 1) if entry and sl and x["price"] else None,
        "cap_used": min(x["exposure"] / p.max_position_pct, 1.0) if p.max_position_pct else 0.0,
        "day_used": min(max(-x["day_ret"], 0.0) / p.daily_loss, 1.0) if p.daily_loss else 0.0,
    }


def _held(td) -> str:
    if td is None:
        return ""
    secs = max(td.total_seconds(), 0.0)  # a file stamped a moment after the clock read is 0, not -0
    hours = secs / 3600
    return f"{hours / 24:.1f} d" if hours >= 48 else f"{hours:.0f} h" if hours >= 1 else f"{secs / 60:.0f} min"


DECISION_ACTIONS = ["create", "start", "stop", "pause", "resume", "flatten"]


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


def _clone_qs(s) -> str:
    """The new-sleeve form filled in with this sleeve's settings, for "Clone with changes"."""
    params = dict(s.params)
    q = {"strategy": s.strategy, "instrument": s.instrument, "bar_spec": s.bar_spec,
         "starting_balance": f"{s.starting_balance:g}", "risk_profile": s.risk_profile, "warmup_bars": s.warmup_bars,
         "name": f"{s.name[:38]}-v2", "from": "clone", "source": s.name}
    if "max_notional" in params:
        q["max_notional"] = f"{params.pop('max_notional'):g}"
    for key in ("stop_loss", "take_profit", "risk_per_trade"):  # stored as fractions, entered as %
        if key in params:
            q[f"{key}_pct"] = f"{params.pop(key) * 100:g}"
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
    return min(cap, REGISTRY[strategy][0].warmup_needed(params, spec_minutes(bar_spec)))


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
    execution = str(form.get("execution", "") or "market")
    if execution not in ("market", "maker"):
        raise ValueError("order type: market or maker first")
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


def _check_strategy_params(cfg: SleeveConfig) -> None:
    """Build the strategy config once so bad parameters fail here, not in the sleeve process."""
    from nautilus_trader.model import BarType, InstrumentId

    _, config_cls = REGISTRY[cfg.strategy]
    params = dict(cfg.params)
    params.pop("max_notional", None)
    config_cls(instrument_id=InstrumentId.from_str(cfg.instrument_id), bar_type=BarType.from_str(cfg.bar_type),
               assumed_taker_fee=float(cfg.fees.taker), **params)

