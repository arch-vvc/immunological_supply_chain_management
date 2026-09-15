"""
isc_common — the ONE place for constants that several stages share.
====================================================================
Everything here used to be copy-pasted (fuel-cost table in two files and a
YAML, scoring weights in two files, the planner rule in three, King's
safety-stock formula with two different variance estimators). Drift between
copies was found by review; this module is the single source and the tests
check the copies are gone.

Nothing here runs anything at import time.
"""
from __future__ import annotations

import math
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42

# ── Regional shipping-cost multipliers ───────────────────────────────────────
# ILLUSTRATIVE relative multipliers by US state (Midwest hub = 1.0), NOT a
# sourced dataset: they encode "further from the Midwest costs more" with a
# coarse regional gradient and are used only to break ties between otherwise
# similar reroute / transfer options. Two decimals are formatting, not
# precision. The canonical copy lives in config/<domain>.yaml
# (shipping_by_state); this loader reads it so code and config cannot drift.
DEFAULT_FUEL_COST = 1.3


def load_region_fuel_cost(domain: str = "pharma") -> dict[str, float]:
    """Read shipping_by_state from config/<domain>.yaml. Falls back to an
    empty table (every state = DEFAULT_FUEL_COST) if the file is missing."""
    path = os.path.join(ROOT, "config", f"{domain}.yaml")
    try:
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return {}

    def _find(d, key):
        if isinstance(d, dict):
            if key in d:
                return d[key]
            for v in d.values():
                r = _find(v, key)
                if r is not None:
                    return r
        return None

    tbl = _find(cfg, "shipping_by_state") or _find(cfg, "by_state") or {}
    return {str(k).upper(): float(v) for k, v in tbl.items()}


def load_default_fuel_cost(domain: str = "pharma", fallback: float = DEFAULT_FUEL_COST) -> float:
    """shipping_costs.default from config/<domain>.yaml (the code used to hard-
    code 1.3 while the YAML said 1.0 — the YAML is now the single source)."""
    path = os.path.join(ROOT, "config", f"{domain}.yaml")
    try:
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        return float(((cfg.get("shipping_costs") or {}).get("default", fallback)))
    except Exception:
        return fallback


REGION_FUEL_COST  = load_region_fuel_cost("pharma")
DEFAULT_FUEL_COST = load_default_fuel_cost("pharma")


def region_fuel_cost(state, table: dict | None = None) -> float:
    return (table if table is not None else REGION_FUEL_COST).get(
        str(state).upper(), DEFAULT_FUEL_COST)


# ── Digital-antibody scoring weights ─────────────────────────────────────────
# Supplier agent (Stage 14) and the live engine rank backup suppliers with
# the same rule; inventory agent (Stage 15) and the engine rank transfers
# with the same rule. Each is imported from here by both users.
SUPPLIER_WEIGHTS = {"safety": 0.50, "volume": 0.30, "efficiency": 0.20}
INVENTORY_WEIGHTS = {"capacity": 0.40, "safety": 0.35, "fuel": 0.25}

# ── SCMS planner rule (Stages 19, 20, 28) ────────────────────────────────────
# "Alternate vendor = 0.6 x normalised capacity + 0.4 x normalised
# establishment on the molecule". The backtest and counterfactual validate
# THIS rule; the vendor scorecard (Stage 23) is a different, reliability-
# weighted ranking and is reported as such, not as the same rule.
W_CAP, W_ESTAB = 0.6, 0.4


# ── King's safety-stock formula ──────────────────────────────────────────────
Z_95, Z_99 = 1.65, 2.33


def kings_safety_stock(l_mean: float, l_std: float, d_mean: float, d_std: float,
                       z: float = Z_95) -> float:
    """SS = z * sqrt( Lbar * sigma_d^2 + Dbar^2 * sigma_L^2 ).
    Callers must pass SAMPLE standard deviations (statistics.stdev); with
    MIN_SHIPMENTS as small as 3 the population estimator materially understates
    variance, and the two implementations used to disagree on this."""
    return z * math.sqrt(max(0.0, l_mean * d_std ** 2 + d_mean ** 2 * l_std ** 2))


def sample_std(values, default: float = 0.0) -> float:
    """statistics.stdev with the n<2 guard every caller needs."""
    return statistics.stdev(values) if len(values) >= 2 else default
