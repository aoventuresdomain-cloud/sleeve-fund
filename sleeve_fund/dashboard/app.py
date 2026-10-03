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

from sleeve_fund.dashboard.metrics import portfolio_summary, sleeve_summary
from sleeve_fund.paper.config import ALLOWED_BAR_SPECS, SleeveConfig
from sleeve_fund.research.ledger import IdeaLedger
from sleeve_fund.risk import PROFILES
from sleeve_fund.store import Store
from sleeve_fund.strategies import REGISTRY

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TEARSHEETS = Path(os.environ.get("TEARSHEET_DIR", ROOT / "research" / "tearsheets"))
LEDGER = Path(os.environ.get("IDEA_LEDGER", ROOT / "research" / "idea_ledger.jsonl"))
# Suggestions only: the field accepts any Kraken spot pair.
INSTRUMENT_HINTS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "SUI/USD", "ADA/USD", "DOGE/USD", "BTC/GBP", "ETH/GBP"]
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,40}$")

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
    templates.env.globals["bar_label"] = _bar_label
    templates.env.filters["ts"] = lambda t: t.strftime("%d %b %H:%M UTC") if t else "never"
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    def st() -> Store:
        return app.state.store

    def page(request: Request, name: str, **ctx) -> HTMLResponse:
        return templates.TemplateResponse(request, name, ctx)

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict:
        return {"ok": True}

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request, _: str = Depends(require_pm)):
        summaries = [sleeve_summary(st(), s) for s in st().sleeves()]
        return page(request, "home.html", summaries=summaries, book=portfolio_summary(summaries),
                    alerts=st().events(min_level="warning", limit=15))

    @app.get("/sleeves/new", response_class=HTMLResponse)
    def new_sleeve_form(request: Request, _: str = Depends(require_pm), error: str = ""):
        return page(request, "new_sleeve.html", strategies=_strategy_choices(), instruments=INSTRUMENT_HINTS,
                    bar_specs=sorted(ALLOWED_BAR_SPECS), profiles=PROFILES, error=error)

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
            # Each strategy's parameter inputs are named p_<strategy>__<param>; only the chosen one counts.
            prefix = f"p_{strategy}__"
            params = _coerce_params({k[len(prefix):]: v for k, v in form.items() if k.startswith(prefix) and v != ""})
            if str(form.get("max_notional", "")).strip():
                params["max_notional"] = float(form["max_notional"])
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

    @app.get("/sleeves/{name}", response_class=HTMLResponse)
    def sleeve_detail(request: Request, name: str, _: str = Depends(require_pm)):
        try:
            s = st().sleeve(name)
        except KeyError:
            raise HTTPException(404, "no such sleeve") from None
        return page(request, "sleeve.html", x=sleeve_summary(st(), s), fills=st().fills(name, limit=100),
                    events=st().events(name, limit=60), decisions=st().decisions(name, limit=50),
                    pending=st().pending_commands(name))

    @app.get("/api/sleeves/{name}/equity")
    def equity_json(name: str, _: str = Depends(require_pm)):
        rows = st().equity_series(name, limit=20_000)
        step = max(1, len(rows) // 1500)  # keep the chart light
        rows = rows[::step] + ([rows[-1]] if rows and (len(rows) - 1) % step else [])
        return JSONResponse({
            "t": [r["ts"].isoformat() for r in rows],
            "equity": [round(r["equity"], 2) for r in rows],
            "benchmark": [round(r["benchmark"], 2) for r in rows],
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
        sheets = sorted(TEARSHEETS.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        counts = IdeaLedger(LEDGER).counts()
        return page(request, "research.html", sheets=[p.stem for p in sheets], counts=counts)

    @app.get("/research/{sheet}", response_class=HTMLResponse)
    def tearsheet(request: Request, sheet: str, _: str = Depends(require_pm)):
        path = (TEARSHEETS / f"{sheet}.md").resolve()
        if path.parent != TEARSHEETS.resolve() or not path.exists():
            raise HTTPException(404, "no such tear sheet")
        html = markdown.markdown(path.read_text(encoding="utf-8"), extensions=["tables"])
        return page(request, "tearsheet.html", title=sheet, body=html)

    @app.get("/decisions", response_class=HTMLResponse)
    def decisions(request: Request, _: str = Depends(require_pm)):
        return page(request, "decisions.html", decisions=st().decisions(limit=500))

    return app


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
        out.append({"name": name, "idea": spec.idea, "params": spec.default_params, "family": spec.family})
    return out


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

