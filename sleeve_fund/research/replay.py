"""Replay a paper sleeve's recording through the backtest engine with the paper runtime.

This is the proof that a backtest is the paper sleeve replayed (plan milestone 3): the same
strategy class, the same runtime (in its live mode: tick by tick, trade-driven marks), the same
simulated venue and fees, fed the quotes and trades the paper sleeve actually received, must send
exactly the orders the paper sleeve sent.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from sleeve_fund.paper.recorder import read


WARMUP_BARS = 200  # candles a replay with history warms up on, at most


def _instrument(h: dict, fees):
    from nautilus_trader.model import Currency, CurrencyPair, InstrumentId, Money, Price, Quantity, Symbol

    i = h["instrument"]
    iid = InstrumentId.from_str(i["id"])
    quote = Currency.from_str(i["quote"])
    opt = lambda key, f: f(i[key]) if i.get(key) else None  # noqa: E731
    return CurrencyPair(
        instrument_id=iid, raw_symbol=Symbol(i["raw_symbol"]), base_currency=Currency.from_str(i["base"]),
        quote_currency=quote, price_precision=i["price_precision"], size_precision=i["size_precision"],
        price_increment=Price.from_str(i["price_increment"]), size_increment=Quantity.from_str(i["size_increment"]),
        min_quantity=opt("min_quantity", Quantity.from_str), lot_size=opt("lot_size", Quantity.from_str),
        min_notional=opt("min_notional", lambda s: Money.from_str(s)),
        margin_init=Decimal(0), margin_maint=Decimal(0), maker_fee=fees.maker, taker_fee=fees.taker,
        ts_event=0, ts_init=0,
    )


def _money(text: str):
    """A recorded balance. Money.from_str only knows registered currencies, and a venue's own codes
    (e.g. a "Z"-prefixed cash code) are made on the fly by Currency.from_str, as the live adapter does."""
    from nautilus_trader.model import Currency, Money

    amount, code = text.split()
    return Money(Decimal(amount), Currency.from_str(code))


def _data(instrument, rows: list[dict]) -> list:
    from nautilus_trader.model import AggressorSide, Price, Quantity, QuoteTick, TradeId, TradeTick

    sides = {int(s): s for s in (AggressorSide.NO_AGGRESSOR, AggressorSide.BUY, AggressorSide.SELL)}
    iid, out = instrument.id, []
    for r in rows:
        if r["k"] == "q":
            out.append(QuoteTick(iid, Price.from_str(r["b"]), Price.from_str(r["a"]), Quantity.from_str(r["bs"]),
                                 Quantity.from_str(r["as"]), r["e"], r["i"]))
        elif r["k"] == "t":
            out.append(TradeTick(iid, Price.from_str(r["p"]), Quantity.from_str(r["s"]), sides[r["side"]],
                                 TradeId(r["id"]), r["e"], r["i"]))
    return out


def _history(path: Path | str, before_ns: int):
    """A warm-up history loader, as the paper node attaches (paper.node.history_loader), from a recording of the market
    the hub saw: its trades built into candles, those closed by `before_ns` (the replay's first event), the latest
    `limit` of them. It lets a replayed restart warm up, and decide the candles it missed, as the node does (R-I5-2)."""
    from nautilus_trader.model import Bar, Price, Quantity

    from sleeve_fund.data import bar_minutes

    trades = [(r["e"], float(r["p"]), float(r["s"])) for r in read(path)[1] if r["k"] == "t" and r["e"] < before_ns]

    def load(instrument, bar_type, limit: int):
        step = bar_minutes(bar_type) * 60_000_000_000
        candles: dict[int, list[float]] = {}
        for ts, px, size in trades:
            c = candles.setdefault((ts // step + 1) * step, [px, px, px, px, 0.0])
            c[1], c[2], c[3], c[4] = max(c[1], px), min(c[2], px), px, c[4] + size
        p, q = instrument.price_precision, instrument.size_precision
        return [Bar(bar_type, Price(o, p), Price(h, p), Price(lo, p), Price(c, p), Quantity(v, q), close, close)
                for close, (o, h, lo, c, v) in sorted(candles.items()) if close <= before_ns][-limit:]

    load.source = "the replayed market's history"
    return load


def replay(path: Path | str, with_fills: bool = False, store=None,
           history: Path | str | None = None) -> list[dict] | tuple[list[dict], list[dict]]:
    """The orders the recorded sleeve sends when replayed, oldest first, as the journal stores them;
    with_fills, the fills too, oldest first. store: the journal to replay into (in memory by default). history: a
    recording of the market over the same span or longer, as the hub's history store holds it: a restart warms up
    from its candles before the recording's first event, as the paper node does (none: no warm-up)."""
    from nautilus_trader.backtest import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.common import LoggerConfig, LogLevel
    from nautilus_trader.model import AccountType, BarType, OmsType, TraderId

    from sleeve_fund import markets
    from sleeve_fund.instruments import FeeSchedule, ScheduleFeeModel, fill_model
    from sleeve_fund.paper.runtime import SleeveRuntime
    from sleeve_fund.store import Store, utcnow
    from sleeve_fund.strategies import REGISTRY

    h, rows = read(path)
    s = h["sleeve"]
    fees = FeeSchedule(Decimal(s["maker_fee"]), Decimal(s["taker_fee"]))
    instrument = _instrument(h, fees)
    store = store or Store.in_memory()
    store.create_sleeve(name=s["name"], strategy=s["strategy"], instrument=s["instrument"], bar_spec=s["bar_spec"],
                        starting_balance=s["starting_balance"], risk_profile=s["risk_profile"], params=s["params"])
    runtime = SleeveRuntime(store, s["name"], tick_seconds=s["tick_seconds"])
    engine = BacktestEngine(BacktestEngineConfig(trader_id=TraderId.from_str("REPLAY-001"),
                                                 logging=LoggerConfig(stdout_level=LogLevel.ERROR)))
    fee_model = ScheduleFeeModel(fees)
    try:
        perp = markets.is_perp(s["params"])  # on margin, as the paper node ran it
        engine.add_venue(venue=instrument.id.venue, oms_type=OmsType.NETTING,
                         account_type=AccountType.MARGIN if perp else AccountType.CASH,
                         default_leverage=markets.VENUE_LEVERAGE if perp else None,
                         base_currency=None, starting_balances=[_money(b) for b in h["balances"]],
                         fee_model=fee_model, fill_model=fill_model())
        engine.add_instrument(instrument)
        data = _data(instrument, rows)
        engine.add_data(data)
        strategy_cls, config_cls = REGISTRY[s["strategy"]]
        config = config_cls(instrument_id=instrument.id, bar_type=BarType.from_str(f"{instrument.id}-{s['bar_spec']}"),
                            max_notional=s.get("max_notional"), assumed_taker_fee=float(fees.taker),
                            warmup_bars=WARMUP_BARS if history is not None else 0, **s["params"])
        strategy = strategy_cls(config).attach_runtime(runtime)
        if history is not None:
            strategy.attach_history(_history(history, min(d.ts_event for d in data)))
        strategy.simulated_venue = True  # as the paper node does
        strategy.fee_model = fee_model
        engine.add_strategy(strategy)
        engine.run()
        orders = [o for o in reversed(store.orders(s["name"], limit=100_000))
                  if not (o.get("signal") or {}).get("watched")]  # a stop watched in the process is not sent
        return (orders, list(reversed(store.fills(s["name"], limit=100_000)))) if with_fills else orders
    finally:
        runtime.now = utcnow  # break the runtime <-> strategy cycle on this thread
        engine.dispose()


def comparable(orders: list[dict]) -> list[tuple]:
    """What must match between paper and replay: what was sent, why, how, and how much."""
    return [(o["side"], o["intent"], o["order_type"], round(float(o["qty"]), 8), o["ts"].replace(microsecond=0))
            for o in orders]
