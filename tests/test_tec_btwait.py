"""TEC-BTWAIT (HoE, Drop B): a dashboard backtest cell waits for its job as long as a loaded box needs, not the
production 8 s, and production keeps its own wait."""
from pathlib import Path

from sleeve_fund.dashboard import app as app_mod


def test_a_dashboard_backtest_cell_waits_far_longer_than_production_does():
    assert app_mod.BACKTEST_WAIT >= 300.0


def test_production_still_waits_8_seconds_unless_configured():
    src = Path(app_mod.__file__).read_text()
    assert 'BACKTEST_WAIT = float(os.environ.get("BACKTEST_WAIT_SECONDS", "8"))' in src
