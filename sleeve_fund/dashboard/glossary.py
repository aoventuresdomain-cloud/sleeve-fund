"""Plain-English definitions behind the ? next to each figure."""

GLOSSARY = {
    "equity": "What the sleeve or book is worth now: cash plus coins at the last Kraken price.",
    "today": "Change in equity since midnight UTC, after fees.",
    "mtd": "Change in equity since the 1st of this month, after fees.",
    "since_start": "Return since the money was first allocated, after fees. 'Holding' is what simply buying the coin on day one would have returned.",
    "vs_hold": "The sleeve's return minus buy-and-hold over the same period. Positive means the strategy added value over just owning the coin.",
    "in_market": "Share of equity held in coins rather than cash. 0% means all cash.",
    "cash": "Uninvested money. Open P&L is the unrealised gain or loss on coins held now.",
    "drawdown": "How far equity is below its highest point so far. Each risk profile halts a sleeve at a set drawdown.",
    "sharpe": "Return per unit of risk: annual return divided by annual volatility. Above 1 is good; buy-and-hold is the bar to beat. Needs at least 30 days to mean much.",
    "fees": "Kraken trading fees paid. Every order is charged the taker fee, the higher of the two.",
    "volatility": "How much daily returns swing, scaled to a year. Higher means a bumpier ride.",
    "cagr": "Compound annual growth rate: the steady yearly return that would give the same end result.",
    "win_rate": "Share of closed trades that made money after both fees. A low win rate can still be profitable if wins are much bigger than losses.",
    "expectancy": "Average return per closed trade after fees. This is what each new trade is worth on average.",
    "profit_factor": "Money made on winning trades divided by money lost on losing ones. Above 1 is profitable; 1.5 or more is solid.",
    "unrealised": "Gain or loss on coins still held, at the last price. It becomes realised when sold, and the exit fee is not included yet.",
    "realised": "Profit or loss on closed trades, after entry and exit fees.",
    "exposure_cap": "The most of a sleeve's equity its risk profile lets it hold in coins.",
    "g1": "Gate 1: the strategy beat buy-and-hold out of sample on real data, with enough trades, after fees. Decided by the research loop.",
    "g2": "Gate 2: at least six weeks of paper trading whose results sit inside the backtest's range. Only you can approve it, and only then can a sleeve trade real money.",
}
