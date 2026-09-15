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
