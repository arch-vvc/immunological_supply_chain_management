"""
IR SCHEMA — Canonical Intermediate Representation for supply-chain data
=======================================================================
Why this file exists
--------------------
The project grew two parallel tracks welded to two datasets: an ARCOS track
(real 3-tier network, no disruptions) and an SCMS track (real disruptions +
lead times + value, thin topology). Every analytics stage read a dataset-
specific file directly, so nothing was portable and any cross-dataset number
crossed an illegitimate seam.

This module defines ONE shape that every dataset maps into. Models are then
written against this shape, never against raw columns. Crucially, every field
is OPTIONAL: a dataset populates what it actually has, and a *capability
manifest* records which fields are present. Analytics run only when their
required capability is satisfied — a capability is NEVER borrowed from another
dataset. That single rule is what keeps "plug-and-play" honest.

Layers:
    raw CSV  --(adapter)-->  IR (this module)  --(manifest)-->  gated analytics

Four tables (all columns nullable):
    nodes  — entities:      node_id, role, country, size_hint
    edges  — who supplies whom: src, dst, product
    flows  — transactions/shipments over time:
             flow_id, timestamp, src, dst, product, qty, value,
             scheduled_date, actual_date, po_date, status
    events — OPTIONAL explicit disruption log (time, location, etype, magnitude)
             (most datasets derive disruptions from flows, so this is often empty)

Pure stdlib on purpose (csv / json / dataclasses): runs anywhere, no pandas.

Usage:
    from ir_schema import IR, Node, Edge, Flow
    ir = IR("scms")
    ir.add_node(Node("Mylan", role="vendor", country=None))
    ir.add_flow(Flow(..., qty=1000, scheduled_date="2010-06-01", actual_date="2010-07-01"))
    ir.finalize()
    print(ir.capabilities())      # {'HAS_TRANSACTIONS': True, ...}
    ir.print_manifest()
    ir.write()                    # -> output/ir/scms/{nodes,edges,flows,events}.csv + manifest.json
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field, asdict, fields
from typing import Optional

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IR_DIR = os.path.join(ROOT, "output", "ir")

# ── Table row types ──────────────────────────────────────────────────────────
# Every field is Optional[...] with a null-ish default. An adapter fills only
# what the source genuinely supports; absence is meaningful, not an error.


@dataclass
class Node:
    node_id:   str
    role:      Optional[str] = None    # tier / function, e.g. manufacturer, vendor, country
    country:   Optional[str] = None
    size_hint: Optional[float] = None  # total volume/experience proxy, if known


@dataclass
class Edge:
    src:     str
    dst:     str
    product: Optional[str] = None      # molecule / lane / SKU carried on this link


@dataclass
class Flow:
    flow_id:        str
    timestamp:      Optional[str] = None   # ISO date the flow is anchored to
    src:            Optional[str] = None
    dst:            Optional[str] = None
    product:        Optional[str] = None
    qty:            Optional[float] = None
    value:          Optional[float] = None
    scheduled_date: Optional[str] = None   # promised delivery (ISO)
    actual_date:    Optional[str] = None   # realised delivery (ISO)
    po_date:        Optional[str] = None    # order placed (ISO) — for lead time
    status:         Optional[str] = None


@dataclass
class Event:
    time:      Optional[str] = None
    location:  Optional[str] = None
    etype:     Optional[str] = None
    magnitude: Optional[float] = None


# ── Capability definitions ───────────────────────────────────────────────────
# A capability is a boolean derived ONLY from which IR fields a dataset actually
# populates. `describe` is documentation surfaced in the manifest. Detection
# logic lives in IR.capabilities() so the thresholds are in one place.

CAPABILITIES = {
    "HAS_TRANSACTIONS": "flows carry a quantity — enables anomaly detection",
    "HAS_TOPOLOGY":     "3+ distinct node roles wired by edges — enables graph risk / routing",
    "HAS_DISRUPTIONS":  "flows have scheduled AND actual dates — enables delay analysis / event replay",
    "HAS_LEAD_TIMES":   "flows have order (PO) AND actual dates — enables safety-stock sizing",
    "HAS_VALUE":        "flows carry monetary value — enables $ value-at-risk / buffer costing",
}

# Fraction of flows that must populate a field for the capability to count.
# Kept deliberately modest: real datasets are patchy (SCMS has 'Date Not
# Captured' rows), and a capability means "enough signal to run the stage",
# not "every row is perfect".
_COVERAGE = 0.30

# Which capabilities each analytics stage requires. This is the gate: the
# runner activates a stage only if ALL its requirements are satisfied. Names
# mirror the existing pipeline stages so the mapping is obvious.
STAGE_REQUIREMENTS = {
    "anomaly_detection":       ["HAS_TRANSACTIONS"],
    "disruption_detection":    ["HAS_DISRUPTIONS"],
    "graph_risk_routing":      ["HAS_TOPOLOGY"],
    "ppo_recovery_routing":    ["HAS_TOPOLOGY"],
    "event_replay":            ["HAS_DISRUPTIONS"],
    "revealed_pref_backtest":  ["HAS_DISRUPTIONS"],
    "safety_stock":            ["HAS_LEAD_TIMES", "HAS_TRANSACTIONS"],
    "value_at_risk":           ["HAS_VALUE", "HAS_DISRUPTIONS"],
}


def _nonempty(v) -> bool:
    """A field counts as 'present' if it is not None and not an empty string."""
    return v is not None and v != ""


class IR:
    """An in-memory canonical representation plus capability detection and I/O."""

    def __init__(self, dataset: str):
        self.dataset = dataset
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
        self.flows: list[Flow] = []
        self.events: list[Event] = []
        self._node_ids: set[str] = set()

    # ── population ──
    def add_node(self, node: Node) -> None:
        if node.node_id in self._node_ids:
            return
        self._node_ids.add(node.node_id)
        self.nodes.append(node)

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    def add_flow(self, flow: Flow) -> None:
        self.flows.append(flow)

    def add_event(self, event: Event) -> None:
        self.events.append(event)

    def finalize(self) -> None:
        """Deduplicate edges (src,dst,product). Nodes already deduped on add."""
        seen = set()
        deduped = []
        for e in self.edges:
            key = (e.src, e.dst, e.product)
            if key not in seen:
                seen.add(key)
                deduped.append(e)
        self.edges = deduped

    # ── capability detection ──
    def _frac(self, pred) -> float:
        if not self.flows:
            return 0.0
        return sum(1 for f in self.flows if pred(f)) / len(self.flows)

    def capabilities(self) -> dict:
        n_roles = len({n.role for n in self.nodes if _nonempty(n.role)})
        has_edges = len(self.edges) > 0

        return {
            "HAS_TRANSACTIONS": bool(self.flows) and self._frac(lambda f: _nonempty(f.qty)) >= _COVERAGE,
            "HAS_TOPOLOGY":     n_roles >= 3 and has_edges,
            "HAS_DISRUPTIONS":  self._frac(lambda f: _nonempty(f.scheduled_date) and _nonempty(f.actual_date)) >= _COVERAGE,
            "HAS_LEAD_TIMES":   self._frac(lambda f: _nonempty(f.po_date) and _nonempty(f.actual_date)) >= _COVERAGE,
            "HAS_VALUE":        self._frac(lambda f: _nonempty(f.value) and (f.value or 0) > 0) >= _COVERAGE,
        }

    def enabled_stages(self) -> dict:
        """{stage_name: (bool_enabled, [missing_capabilities])} for every stage."""
        caps = self.capabilities()
        out = {}
        for stage, reqs in STAGE_REQUIREMENTS.items():
            missing = [r for r in reqs if not caps.get(r)]
            out[stage] = (len(missing) == 0, missing)
        return out

    def summary(self) -> dict:
        return {
            "dataset": self.dataset,
            "nodes":   len(self.nodes),
            "edges":   len(self.edges),
            "flows":   len(self.flows),
            "events":  len(self.events),
            "roles":   sorted({n.role for n in self.nodes if _nonempty(n.role)}),
        }

    # ── reporting ──
    def print_manifest(self) -> None:
        s = self.summary()
        caps = self.capabilities()
        stages = self.enabled_stages()
        print("=" * 62)
        print(f"  IR MANIFEST — dataset: {self.dataset}")
        print("=" * 62)
        print(f"  nodes {s['nodes']:>7}   edges {s['edges']:>7}   "
              f"flows {s['flows']:>7}   events {s['events']:>5}")
        print(f"  roles: {', '.join(s['roles']) or '(none)'}")
        print("  " + "-" * 58)
        print("  CAPABILITIES")
        for cap, desc in CAPABILITIES.items():
            mark = "ON " if caps[cap] else "off"
            print(f"    [{mark}] {cap:<18} {desc}")
        print("  " + "-" * 58)
        print("  STAGES (gated by capability)")
        for stage, (ok, missing) in stages.items():
            if ok:
                print(f"    [RUN ]  {stage}")
            else:
                print(f"    [skip]  {stage:<24} needs {', '.join(missing)}")
        print("=" * 62)

    # ── persistence ──
    def _write_table(self, path: str, rows: list, row_type) -> None:
        cols = [f.name for f in fields(row_type)]
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: ("" if v is None else v) for k, v in asdict(r).items()})

    def write(self, base_dir: str = IR_DIR) -> str:
        out = os.path.join(base_dir, self.dataset)
        os.makedirs(out, exist_ok=True)
        self._write_table(os.path.join(out, "nodes.csv"),  self.nodes,  Node)
        self._write_table(os.path.join(out, "edges.csv"),  self.edges,  Edge)
        self._write_table(os.path.join(out, "flows.csv"),  self.flows,  Flow)
        self._write_table(os.path.join(out, "events.csv"), self.events, Event)
        manifest = {
            "dataset":       self.dataset,
            "summary":       self.summary(),
            "capabilities":  self.capabilities(),
            "stages":        {k: {"enabled": v[0], "missing": v[1]}
                              for k, v in self.enabled_stages().items()},
        }
        with open(os.path.join(out, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=2)
        return out


if __name__ == "__main__":
    # Tiny self-test with a hand-built IR so this module runs with zero data.
    ir = IR("_selftest")
    for r in ("manufacturer", "distributor", "retailer"):
        ir.add_node(Node(f"{r}_1", role=r, country="XX"))
    ir.add_edge(Edge("manufacturer_1", "distributor_1", product="widget"))
    ir.add_edge(Edge("distributor_1", "retailer_1", product="widget"))
    ir.add_flow(Flow("f1", timestamp="2010-01-01", src="distributor_1", dst="retailer_1",
                     product="widget", qty=100.0))
    ir.finalize()
    ir.print_manifest()
    print("\nOK — ir_schema self-test complete.")
