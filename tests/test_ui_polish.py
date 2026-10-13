"""UI polish (during testing, fe-plans/ui-polish-plan.md): pins for each item as it lands."""

import re
from pathlib import Path

DASH = Path(__file__).resolve().parents[1] / "sleeve_fund" / "dashboard"


def test_up6_no_collect_leftovers_in_the_templates_or_the_css():
    """UP-6: the Collect button went in UI v2 (item 10); a study asks the collector itself. Its form styles and any
    "Collect" control are gone. The history request route stays: a study's own request shares its checks."""
    css = (DASH / "static" / "app.css").read_text()
    assert ".collect-" not in css
    for page in (DASH / "templates").glob("*.html"):
        text = page.read_text()
        assert "collect-" not in text, page.name
        assert not re.search(r">\s*Collect\b", text), page.name
