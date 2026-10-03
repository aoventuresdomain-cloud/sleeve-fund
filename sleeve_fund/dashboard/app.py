"""PM dashboard v1: see the book, start and stop paper sleeves, pause/flatten/resume,
read tear sheets and the decision log.

Read-mostly. Every control that changes a sleeve asks for a reason and lands in
the decision log. Protected by one PM password (HTTP Basic, behind HTTPS).

    DASHBOARD_PASSWORD=... uvicorn --factory sleeve_fund.dashboard.app:create_app
"""

from __future__ import annotations

import os
import re
import secrets
from pathlib import Path
from urllib.parse import urlencode, urlparse

import markdown
from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sleeve_fund.dashboard import book as bookm
from sleeve_fund.dashboard import riskops
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
# Suggestions only: the field accepts any Kraken spot pair.
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
        return page(request, "home.html", summaries=summaries, book=bookm.book_view(st(), summaries, frames),
                    alerts=st().alerts(limit=30), shell=shell(sleeves))

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
                    bar_specs=sorted(ALLOWED_BAR_SPECS), profiles=PROFILES, error=error, g1=g1, chosen=chosen)

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
            cfg = SleeveConfig(name=name, strategy=strategy, instrument=str(form.get("instrument", "")),
                               bar_spec=str(form.get("bar_spec", "")),
                               starting_balance=float(form.get("starting_balance", 0) or 0), params=params,
                               warmup_bars=int(form.get("warmup_bars", 0) or 0),
                               risk_profile=str(form.get("risk_profile", "")))
            _check_strategy_params(cfg)
            if any(s.name == name for s in st().sleeves()):
                raise ValueError(f"a sleeve called {name} already exists")
            st().create_sleeve(name=name, strategy=strategy, instrument=cfg.instrument, bar_spec=cfg.bar_spec,
                               starting_balance=cfg.starting_balance, params=params,
                               risk_profile=cfg.risk_profile, warmup_bars=cfg.warmup_bars)
            st().decide(actor, "create", reason, name)
        except (ValueError, TypeError) as exc:
            return RedirectResponse(f"/sleeves/new?{urlencode({'error': str(exc)})}", status_code=303)
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
                raise ValueError("enter a pair like SOL/USD")
            params = _form_params(q, strategy)
            params.pop("max_notional", None)  # a per-order cap doesn't change a look-back meaningfully
            balance = float(q.get("starting_balance") or 10_000)
            return JSONResponse(preview.run(strategy, pair, params, starting=balance))
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
        fills = st().fills(name, limit=2000)
        events = st().events(name, limit=400)
        trips = _trips(fills, events)
        feed = _feed(events, request.query_params.get("feed", "all"))
        return page(request, "sleeve.html", x=x, fills=fills[:200], trips=trips, feed=feed,
                    feed_kind=request.query_params.get("feed", "all"), decisions=st().decisions(name, limit=50),
                    pending=st().pending_commands(name), risk=_risk_view(x), reasons=COMMON_REASONS,
                    idea=_idea(s.strategy, s.params))

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
        return page(request, "decisions.html", decisions=st().decisions(limit=500))

    return app


COMMON_REASONS = [
    "Risk limit close; reducing exposure",
    "Market event; standing aside",
    "Strategy behaving outside its backtest range",
    "Data or venue problem",
    "Checked after an alert; safe to continue",
    "Planned change of settings",
]
EXIT_KINDS = {"stop_loss": "Stop-loss", "take_profit": "Take-profit", "risk_halt": "Risk halt",
              "risk_pause": "Daily-loss pause", "pm_flatten": "PM flatten"}


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


def _trips(fills: list[dict], events: list[dict]) -> list[dict]:
    """Closed round trips, newest first, with holding time and why they closed."""
    from sleeve_fund.research.metrics import trades

    exits = [e for e in events if e["kind"] in EXIT_KINDS]
    out = []
    for t in reversed(trades(list(reversed(fills)))):
        why = "Signal"
        if t["opened"] and t["closed"]:
            hit = [e for e in exits if t["opened"] <= e["ts"] <= t["closed"]]
            if hit:
                why = EXIT_KINDS[hit[0]["kind"]]
        t["reason"] = why
        t["held"] = (t["closed"] - t["opened"]) if t["opened"] and t["closed"] else None
        out.append(t)
    return out


FEEDS = {
    "all": lambda e: e["kind"] != "reconcile",  # routine passes are summarised in the risk panel
    "alerts": lambda e: e["level"] in ("warning", "error"),
    "trades": lambda e: e["kind"] in ("fill", *EXIT_KINDS),
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
               **params)

