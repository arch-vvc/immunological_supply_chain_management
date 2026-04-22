"""
Domain Config Loader
====================
Loads a domain YAML and exposes a flat DomainConfig dataclass.
The engine and consumer import this instead of hardcoded constants.

Usage:
    from domain_config import load_domain_config
    cfg = load_domain_config("pharma")   # loads config/pharma.yaml
    cfg = load_domain_config("automotive")
    cfg = load_domain_config("fmcg")
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict
import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

# ── Defaults (pharma values — safe fallback if key missing in YAML) ─────────

DEFAULTS = {
    "domain": {
        "name": "Supply Chain",
        "industry": "generic",
        "node_roles": {
            "tier1": "manufacturer",
            "tier2": "distributor",
            "tier3": "retailer",
        },
    },
    "anomaly": {
        "z_score_threshold": 2.5,
        "rolling_window": 50,
        "quantity_lognormal_mean": 5.5,
        "quantity_lognormal_sigma": 1.2,
        "quantity_max": 50000,
    },
    "cytokine_storm": {
        "threshold": 3,
        "window_seconds": 60,
    },
    "routing": {
        "ppo_candidates": 10,
        "feature_dims": 5,
        "risk_weight": 0.5,
        "gnn_weight": 0.3,
        "path_weight": 0.2,
        "decay_window_minutes": 30,
        "decay_per_use": 0.15,
        "decay_floor": 0.5,
        "memory_top_k": 3,
    },
    "shipping_costs": {
        "default": 1.0,
        "by_state": {},
    },
}


@dataclass
class DomainConfig:
    # Identity
    domain_name:    str
    industry:       str
    tier1_label:    str   # e.g. "manufacturer" / "oem" / "producer"
    tier2_label:    str   # e.g. "distributor" / "tier1_supplier" / "warehouse"
    tier3_label:    str   # e.g. "retailer" / "assembly_plant" / "supermarket"

    # Anomaly detection
    z_score_threshold:      float
    rolling_window:         int
    qty_lognormal_mean:     float
    qty_lognormal_sigma:    float
    qty_max:                int

    # Cytokine storm
    cytokine_threshold:     int
    cytokine_window_secs:   int

    # Routing
    ppo_candidates:         int
    feature_dims:           int
    risk_weight:            float
    gnn_weight:             float
    path_weight:            float
    decay_window_minutes:   float
    decay_per_use:          float
    decay_floor:            float
    memory_top_k:           int

    # Shipping costs
    shipping_default:       float
    shipping_by_state:      Dict[str, float] = field(default_factory=dict)

    def fuel_cost(self, state: str) -> float:
        """Return the shipping cost multiplier for a US state."""
        return self.shipping_by_state.get(str(state).upper(), self.shipping_default)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_domain_config(domain: str = "pharma") -> DomainConfig:
    """
    Load config/`domain`.yaml and return a DomainConfig.
    Unknown keys in YAML are ignored; missing keys fall back to DEFAULTS.
    Raises FileNotFoundError if the YAML doesn't exist.
    """
    yaml_path = CONFIG_DIR / f"{domain}.yaml"
    if not yaml_path.exists():
        available = [p.stem for p in CONFIG_DIR.glob("*.yaml")]
        raise FileNotFoundError(
            f"Domain config '{domain}' not found at {yaml_path}. "
            f"Available: {available}"
        )

    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    cfg = _deep_merge(DEFAULTS, raw)
    roles = cfg["domain"]["node_roles"]
    r     = cfg["routing"]
    a     = cfg["anomaly"]
    cs    = cfg["cytokine_storm"]
    sc    = cfg["shipping_costs"]

    return DomainConfig(
        domain_name          = cfg["domain"]["name"],
        industry             = cfg["domain"]["industry"],
        tier1_label          = roles.get("tier1", "manufacturer"),
        tier2_label          = roles.get("tier2", "distributor"),
        tier3_label          = roles.get("tier3", "retailer"),

        z_score_threshold    = float(a["z_score_threshold"]),
        rolling_window       = int(a["rolling_window"]),
        qty_lognormal_mean   = float(a["quantity_lognormal_mean"]),
        qty_lognormal_sigma  = float(a["quantity_lognormal_sigma"]),
        qty_max              = int(a["quantity_max"]),

        cytokine_threshold   = int(cs["threshold"]),
        cytokine_window_secs = int(cs["window_seconds"]),

        ppo_candidates       = int(r["ppo_candidates"]),
        feature_dims         = int(r["feature_dims"]),
        risk_weight          = float(r["risk_weight"]),
        gnn_weight           = float(r["gnn_weight"]),
        path_weight          = float(r["path_weight"]),
        decay_window_minutes = float(r["decay_window_minutes"]),
        decay_per_use        = float(r["decay_per_use"]),
        decay_floor          = float(r["decay_floor"]),
        memory_top_k         = int(r["memory_top_k"]),

        shipping_default     = float(sc.get("default", 1.0)),
        shipping_by_state    = {
            k.upper(): float(v)
            for k, v in sc.get("by_state", {}).items()
        },
    )
