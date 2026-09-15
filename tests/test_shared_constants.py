"""isc_common is the single source for shared constants: no src module may
carry its own literal copy. Static grep, so it runs without heavy imports."""
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COPIES = {
    "fuel-cost table literal":      re.compile(r"^REGION_FUEL_COST\s*=\s*\{", re.M),
    "planner rule literal":         re.compile(r"^W_CAP,\s*W_ESTAB\s*=\s*0\.6", re.M),
    "supplier weights literal":     re.compile(r"0\.50\s*\*\s*\w+\[\"safety_score\"\]"),
    "inventory weights literal":    re.compile(r"0\.40\s*\*\s*(capacity_n|float\(row\[\"capacity_norm\"\]\))"),
    "King's formula inline":        re.compile(r"sqrt\(\s*\w+\s*\*\s*\w+\s*\*\*\s*2\s*\+\s*\w+\s*\*\*\s*2\s*\*\s*\w+\s*\*\*\s*2"),
    "SCMS date-format list":        re.compile(r"^DATE_FMTS\s*=", re.M),
    "SCMS parse_date copy":         re.compile(r"^def parse_date\(", re.M),
    "SCMS clean copy":              re.compile(r"^def clean\(", re.M),
    "SCMS to_float copy":           re.compile(r"^def to_float\(", re.M),
    "quantity zero-fill":           re.compile(r"except ValueError:\s*\n\s*qty\s*=\s*0\.0", re.M),
}


def _src_files():
    return [p for p in glob.glob(os.path.join(ROOT, "src", "*.py"))
            if os.path.basename(p) != "isc_common.py"]


def test_no_literal_copies_of_shared_constants():
    hits = []
    for path in _src_files():
        text = open(path).read()
        for name, pat in COPIES.items():
            if pat.search(text):
                hits.append(f"{os.path.basename(path)}: {name}")
    assert not hits, "literal copies of isc_common values found:\n  " + "\n  ".join(hits)


def test_former_copies_import_from_isc_common():
    expects = {
        "routing.py": "from isc_common import REGION_FUEL_COST",
        "inventory_agent.py": "from isc_common import REGION_FUEL_COST",
        "supplier_agent.py": "from isc_common import SUPPLIER_WEIGHTS",
        "immune_response_engine.py": "from isc_common import SUPPLIER_WEIGHTS, INVENTORY_WEIGHTS",
        "scms_spine.py": "from isc_common import W_CAP, W_ESTAB",
        "scms_backtest.py": "from isc_common import W_CAP, W_ESTAB",
        "scms_counterfactual.py": "from isc_common import W_CAP, W_ESTAB",
        "scms_safety_stock.py": "from isc_common import kings_safety_stock",
        "scms_vendor_scorecard.py": "from isc_common import kings_safety_stock",
        "ir_stages.py": "from isc_common import kings_safety_stock",
    }
    missing = [f for f, imp in expects.items()
               if imp not in open(os.path.join(ROOT, "src", f)).read()]
    assert not missing, f"no longer importing from isc_common: {missing}"


def test_scms_parsing_helpers_behave():
    """The shipped SCMS file never trips these paths, so exercise them directly."""
    import sys
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from isc_common import to_float, parse_date, clean
    assert to_float("1,234.5") == 1234.5
    assert to_float(" 7 ") == 7.0
    assert to_float("") is None and to_float(None) is None and to_float("n/a") is None
    assert parse_date("11/13/06").year == 2006          # the %m/%d/%y form only the 4-format list handled
    assert parse_date("13-Nov-06").year == 2006
    assert parse_date("11/13/2006").year == 2006
    assert parse_date("") is None and parse_date("garbage") is None
    assert clean("  a   b\tc ") == "a b c" and clean(None) == ""
