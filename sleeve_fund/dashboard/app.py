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
from sleeve_fund.dashboard.metrics import STALE, sleeve_summary
from sleeve_fund.paper.config import ALLOWED_BAR_SPECS, SleeveConfig
from sleeve_fund.research.ledger import IdeaLedger
from sleeve_fund.risk import PROFILES
from sleeve_fund.store import Store, utcnow
from sleeve_fund.strategies import REGISTRY

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TEARSHEETS = Path(os.environ.get("TEARSHEET_DIR", ROOT / "research" / "tearsheets"))
LEDGER = Path(os.environ.get("IDEA_LEDGER", ROOT / "research" / "idea_ledger.jsonl"))
# Suggestions only: the field accepts any instrument Kraken spot lists.
INSTRUMENT_HINTS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "SUI/USD", "ADA/USD", "DOGE/USD", "BTC/GBP", "ETH/GBP"]
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,40}$")
VERSION = os.environ.get("APP_VERSION", "dev")[:12]

security = HTTPBasic(realm="Sleeve Fund")


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
                            headers={"WWW-Authenticate": 'Basic realm="Sleeve Fund"'})
    return "PM"


def same_origin(request: Request) -> None:
    """Basic auth is sent automatically by browsers, so block cross-site form posts."""
    origin = request.headers.get("origin") or request.headers.get("referer")
    if not origin or urlparse(origin).netloc != request.headers.get("host"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "cross-site request blocked")


def create_app(store: Store | None = None) -> FastAPI:
    _password()  # fail at start-up, not on first request
    app = FastAPI(title="Sleeve Fund", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store = store or Store()
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
    templates.env.filters["ts"] = lambda t: t.strftime("%d %b %H:%M UTC") if t else "never"
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
        return page(request, "risk.html", book=book, risk=riskops.risk_view(st(), summaries, book),
                    shell=shell(sleeves))

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

        g1 = {r["name"]: r["g1"] for r in pipeline.strategies(TEARSHEETS, st().sleeves())}
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
            if form.get("from") == "backtest" and form.get("bar_spec") != BACKTEST_BAR_SPEC:
                raise ValueError(f"interval: the backtest decided on daily bars, so this sleeve must too "
                                 f"({BACKTEST_BAR_SPEC}); backtest another interval before changing it")
            cfg = SleeveConfig(name=name, strategy=strategy, instrument=str(form.get("instrument", "")),
                               bar_spec=str(form.get("bar_spec", "")),
                               starting_balance=float(form.get("starting_balance", 0) or 0), params=params,
                               warmup_bars=int(form.get("warmup_bars", 0) or 0),
                               risk_profile=str(form.get("risk_profile", "")))
            _check_strategy_params(cfg)
            if any(s.name == name for s in st().sleeves()):
                raise ValueError(f"a sleeve called {name} already exists")
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
        except (ValueError, TypeError) as exc:
            # Send the form back filled in, so a typo doesn't cost the PM everything they entered.
            kept = {k: str(v) for k, v in form.items() if isinstance(v, str) and v}
            return RedirectResponse(f"/sleeves/new?{urlencode({'error': str(exc), **kept})}", status_code=303)
        return RedirectResponse(f"/sleeves/{name}", status_code=303)

    @app.get("/api/preview")
    def preview_json(request: Request, _: str = Depends(require_pm)):
        """Look-back of the form's settings on Kraken's recent daily history (in-sample, not G1)."""
        from sleeve_fund.dashboard import preview
        from sleeve_fund.paper.config import PAIR_RE

        q = request.query_params
        strategy, pair = q.get("strategy", ""), q.get("instrument", "").strip().upper()
        try:
            if strategy not in REGISTRY:
                raise ValueError("pick a strategy")
            if not PAIR_RE.match(pair):
                raise ValueError("enter an instrument like SOL/USD")
            params = _form_params(q, strategy)
            balance = float(q.get("starting_balance") or 10_000)
            return JSONResponse(preview.run(strategy, pair, params, starting=balance, cap=_profile_cap(q)))
        except (ValueError, TypeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        except OSError as exc:  # Kraken unreachable
            return JSONResponse({"error": f"could not reach Kraken: {exc}"}, status_code=502)
        except Exception as exc:  # noqa: BLE001 - the form shows the message instead of a blank chart
            return JSONResponse({"error": f"look-back failed: {exc}"}, status_code=500)

    @app.get("/sleeves/{name}", response_class=HTMLResponse)
    def sleeve_detail(request: Request, name: str, _: str = Depends(require_pm)):
        try:
            s = st().sleeve(name)
        except KeyError:
            raise HTTPException(404, "no such sleeve") from None
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
                    clone_qs=_clone_qs(s), path=gates.path_to_live(st(), x, _g1_of(s.strategy), st().accounts(), utcnow()))

    @app.get("/api/sleeves/{name}/candles")
    def candles_json(name: str, interval: str = "", _: str = Depends(require_pm)):
        from sleeve_fund.dashboard import charts

        try:
            s = st().sleeve(name)
        except KeyError:
            raise HTTPException(404, "no such sleeve") from None
        interval = interval if interval in charts.INTERVALS else charts.default_interval(s.bar_spec)
        minutes = charts.INTERVALS[interval]
        try:
            df, source = charts.candles(s.instrument, minutes), "venue"
        except (OSError, ValueError):  # venue unreachable or pair unknown: chart the sleeve's own marks
            df, source = charts.from_marks(st().equity_series(name, limit=500_000), minutes), "marks"
        fills = list(reversed(st().fills(name, limit=100_000)))
        orders = trading.orders_by_id(st(), name)
        x = bookm.sleeve_extras(st(), sleeve_summary(st(), s), bookm.daily(st(), name))
        position = trading.open_position(x, list(reversed(fills)), orders)
        data = charts.payload(df, minutes, fills, orders, charts.position_lines(position), source)
        data["intervals"], data["chosen"] = list(charts.INTERVALS), interval
        return JSONResponse(data)

    @app.get("/api/sleeves/{name}/equity")
    def equity_json(name: str, _: str = Depends(require_pm)):
        try:
            s = st().sleeve(name)
        except KeyError:
            raise HTTPException(404, "no such sleeve") from None
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
                st().decide(actor, command, reason, name)
            else:
                st().command(name, command, reason, actor=actor)
        except KeyError:
            raise HTTPException(404, "no such sleeve") from None
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return RedirectResponse(f"/sleeves/{name}", status_code=303)

    @app.post("/sleeves/{name}/archive")
    def sleeve_archive(name: str, action: str = Form(...), reason: str = Form(...),
                       actor: str = Depends(require_pm), _o: None = Depends(same_origin)):
        try:
            if not reason.strip():
                raise ValueError("a reason is required")
            if action == "archive":
                st().archive(name)
            elif action == "restore":
                st().unarchive(name)
            else:
                raise ValueError("unknown action")
            st().decide(actor, action, reason, name)
        except KeyError:
            raise HTTPException(404, "no such sleeve") from None
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
        if sleeve and sleeve not in names:
            raise HTTPException(404, "no such sleeve")
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

    @app.get("/backtest", response_class=HTMLResponse)
    def backtest_page(request: Request, _: str = Depends(require_pm)):
        """Run any strategy and settings over Kraken's daily history and show every trade with its reason."""
        from sleeve_fund.dashboard import pipeline, preview

        q = request.query_params
        strategy = q.get("strategy") if q.get("strategy") in REGISTRY else "trend_filter"
        period = q.get("period") if q.get("period") in BACKTEST_PERIODS else "all"
        result, error = None, ""
        if q.get("run"):
            try:
                pair = q.get("instrument", "").strip().upper()
                if not PAIR_RE.fullmatch(pair):
                    raise ValueError("instrument: write it as BASE/QUOTE, for example SOL/USD")
                starting = float(q.get("starting_balance") or 10_000)
                if not 100 <= starting <= 1e9:
                    raise ValueError("capital: between 100 and 1,000,000,000")
                params = _form_params(q, strategy)
                result = preview.run(strategy, pair, params, starting=starting, days=BACKTEST_PERIODS[period][1],
                                     detail=True, cap=_profile_cap(q))
            except (ValueError, TypeError, KeyError) as exc:
                error = str(exc).strip("'")
            except OSError as exc:  # Kraken unreachable
                error = f"could not reach Kraken for price history ({exc}). Try again in a minute."
            except Exception as exc:  # noqa: BLE001 - say what failed instead of a blank page
                error = f"the backtest failed: {exc}"
        # The sleeve must decide on the bars that were tested, with indicators warm from its first bar.
        carry = {k: v for k, v in q.items() if k not in ("run", "period") and v}
        carry.update(bar_spec=BACKTEST_BAR_SPEC, warmup_bars=_warmup_for(strategy, q))
        g1 = {r["name"]: r["g1"] for r in pipeline.strategies(TEARSHEETS, st().sleeves())}
        chart = None
        if result:
            chart = {"t": result["t"], "equity": result["equity"], "benchmark": result["benchmark"],
                     "drawdown": result["drawdown"], "fills": result["fills"], "res": "daily"}
        return page(request, "backtest.html", result=result, error=error, pre=dict(q), chosen=strategy,
                    strategies=_strategy_choices(), instruments=INSTRUMENT_HINTS, g1=g1, period=period,
                    periods=BACKTEST_PERIODS, profiles=PROFILES, sleeve_qs=urlencode({**carry, "from": "backtest"}), chart=chart)

    @app.get("/trades", response_class=HTMLResponse)
    def trades_page(request: Request, _: str = Depends(require_pm)):
        sleeves, _frames, summaries = book_data()
        names = [s.name for s in sleeves]
        sleeve = request.query_params.get("sleeve") or None
        if sleeve not in names:
            sleeve = None
        h = trading.history(st(), summaries, sleeve)
        return page(request, "trades.html", h=h, sleeve=sleeve, sleeves=names, shell=shell(sleeves),
                    book_equity=sum(x["equity"] for x in summaries))

    @app.get("/orders", response_class=HTMLResponse)
    def orders_page(request: Request, _: str = Depends(require_pm)):
        names = [s.name for s in st().sleeves()]
        q = request.query_params
        sleeve = q.get("sleeve") if q.get("sleeve") in names else None
        tab = q.get("status") if q.get("status") in trading.STATUS_TABS else "all"
        rows = [trading.order_view(o) for o in st().orders(sleeve, trading.STATUS_TABS[tab][1], limit=1000)]
        counts = st().order_counts(sleeve)
        tabs = [(k, label, sum(counts.get(x, 0) for x in sts) if sts else sum(counts.values()))
                for k, (label, sts) in trading.STATUS_TABS.items()]
        return page(request, "orders.html", orders=rows, tab=tab, tabs=tabs, sleeve=sleeve, sleeves=names)

    @app.get("/accounts", response_class=HTMLResponse)
    def accounts_page(request: Request, _: str = Depends(require_pm), error: str = ""):
        from sleeve_fund import accounts as acc

        rows = st().accounts()
        for r in rows:
            r["env"] = acc.env_names(r["name"]) if r["kind"] == "live" else None
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

        return page(request, "settings.html", profiles=PROFILES, venues=VENUES.values(),
                    tearsheets=str(TEARSHEETS), counts=st().table_sizes(), accounts=st().accounts())

    return app


BACKTEST_PERIODS = {"180": ("6 months", 180), "365": ("1 year", 365), "all": ("All, about 2 years", None)}
PAIR_RE = re.compile(r"[A-Z0-9]{1,12}/[A-Z0-9]{2,6}")

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
    hours = td.total_seconds() / 3600
    return f"{hours / 24:.1f} d" if hours >= 48 else f"{hours:.0f} h" if hours >= 1 else f"{td.total_seconds() / 60:.0f} min"


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
                    headers={"Content-Disposition": f'attachment; filename="sleeve-fund-{name}-{stamp}.csv"'})


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


def _g1_of(strategy: str) -> str | None:
    from sleeve_fund.dashboard import pipeline

    return {r["name"]: r["g1"] for r in pipeline.strategies(TEARSHEETS, [])}.get(strategy)


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
    q.update({f"p_{s.strategy}__{k}": v for k, v in params.items()})
    return urlencode(q)


# The backtest page runs on the venue's daily candles, so a sleeve made from it decides daily too.
BACKTEST_BAR_SPEC = "1-DAY-LAST-EXTERNAL"
MAX_WARMUP_BARS = 720  # what the venue returns in one request


def _profile_cap(q) -> float:
    """The chosen risk profile's position cap, as paper applies it (balanced when none is given)."""
    name = q.get("risk_profile") or "balanced"
    if name not in PROFILES:
        raise ValueError(f"risk profile: no profile called {name}")
    return PROFILES[name].max_position_pct


def _warmup_for(strategy: str, q) -> int:
    """Bars to load at start so the slowest indicator is ready on the sleeve's first daily bar, as
    the strategy itself says, capped at what the venue returns in one request."""
    import importlib

    params = dict(importlib.import_module(f"sleeve_fund.strategies.{strategy}").SPEC.default_params)
    try:
        params.update(_form_params(q, strategy))
    except ValueError:
        pass
    return min(MAX_WARMUP_BARS, REGISTRY[strategy][0].warmup_needed(params, 1440))


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

