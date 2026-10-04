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
    (Kraken's ZUSD, XXBT) are made on the fly by Currency.from_str, as the live adapter does."""
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


def replay(path: Path | str) -> list[dict]:
    """The orders the recorded sleeve sends when replayed, oldest first, as the journal stores them."""
    from nautilus_trader.backtest import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.common import LoggerConfig, LogLevel
    from nautilus_trader.model import AccountType, BarType, OmsType, TraderId

    from sleeve_fund.instruments import FeeSchedule, ScheduleFeeModel, fill_model
    from sleeve_fund.paper.runtime import SleeveRuntime
    from sleeve_fund.store import Store, utcnow
    from sleeve_fund.strategies import REGISTRY

    h, rows = read(path)
    s = h["sleeve"]
    fees = FeeSchedule(Decimal(s["maker_fee"]), Decimal(s["taker_fee"]))
    instrument = _instrument(h, fees)
    store = Store.in_memory()
    store.create_sleeve(name=s["name"], strategy=s["strategy"], instrument=s["instrument"], bar_spec=s["bar_spec"],
                        starting_balance=s["starting_balance"], risk_profile=s["risk_profile"], params=s["params"])
    runtime = SleeveRuntime(store, s["name"], tick_seconds=s["tick_seconds"])
    engine = BacktestEngine(BacktestEngineConfig(trader_id=TraderId.from_str("REPLAY-001"),
                                                 logging=LoggerConfig(stdout_level=LogLevel.ERROR)))
    try:
        engine.add_venue(venue=instrument.id.venue, oms_type=OmsType.NETTING, account_type=AccountType.CASH,
                         base_currency=None, starting_balances=[_money(b) for b in h["balances"]],
                         fee_model=ScheduleFeeModel(fees), fill_model=fill_model())
        engine.add_instrument(instrument)
        engine.add_data(_data(instrument, rows))
        strategy_cls, config_cls = REGISTRY[s["strategy"]]
        config = config_cls(instrument_id=instrument.id, bar_type=BarType.from_str(f"{instrument.id}-{s['bar_spec']}"),
                            max_notional=s.get("max_notional"), assumed_taker_fee=float(fees.taker), warmup_bars=0,
                            **s["params"])
        engine.add_strategy(strategy_cls(config).attach_runtime(runtime))
        engine.run()
        return list(reversed(store.orders(s["name"], limit=100_000)))
    finally:
        runtime.now = utcnow  # break the runtime <-> strategy cycle on this thread
        engine.dispose()


def comparable(orders: list[dict]) -> list[tuple]:
    """What must match between paper and replay: what was sent, why, how, and how much."""
    return [(o["side"], o["intent"], o["order_type"], round(float(o["qty"]), 8), o["ts"].replace(microsecond=0))
            for o in orders]
