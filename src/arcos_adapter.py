"""
ARCOS ADAPTER — raw ARCOS transactions -> canonical IR
======================================================
The ONLY module that knows ARCOS column names. It reads the normalized
transaction table (data/processed/clean_chain.csv, produced by preprocess.py)
and emits the IR defined in ir_schema.py. No analytics here — an adapter is a
dumb translator, or the "dataset-independent" claim leaks.

ARCOS honestly supports:
    HAS_TOPOLOGY     — 3 tiers: manufacturer -> distributor -> retailer
    HAS_TRANSACTIONS — per-row shipment quantity
It does NOT support disruptions, lead times, or value (those columns don't
exist), so the manifest will correctly leave those capabilities off.

Run:
    python3 src/arcos_adapter.py        # build IR, print manifest, write output/ir/arcos/
"""

from __future__ import annotations

import csv
import os

from ir_schema import IR, Node, Edge, Flow

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "data", "processed", "clean_chain.csv")


def _to_float(s):
    try:
        return float((s or "").replace(",", "").strip() or 0)
    except ValueError:
        return None


def build(path: str = SRC, max_rows: int | None = None) -> IR:
    if not os.path.exists(path):
        raise SystemExit(f"[arcos_adapter] missing input: {path}\n"
                         f"  run the ARCOS pipeline first (python3 main.py --only 1)")

    ir = IR("arcos")
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            mfr = (row.get("manufacturer") or "").strip()
            dst_ = (row.get("distributor") or "").strip()   # distributor
            ret = (row.get("retailer") or "").strip()
            state = (row.get("retailer_state") or "").strip() or None
            qty = _to_float(row.get("quantity"))
            date = (row.get("date") or "").strip() or None

            # nodes (deduped inside IR): one per distinct entity, tagged by tier
            if mfr:
                ir.add_node(Node(mfr, role="manufacturer"))
            if dst_:
                ir.add_node(Node(dst_, role="distributor"))
            if ret:
                ir.add_node(Node(ret, role="retailer", country=state))

            # edges: the 3-tier chain (manufacturer -> distributor -> retailer)
            if mfr and dst_:
                ir.add_edge(Edge(mfr, dst_))
            if dst_ and ret:
                ir.add_edge(Edge(dst_, ret))

            # flow: the actual shipment recorded in the row (distributor -> retailer)
            ir.add_flow(Flow(
                flow_id=f"arcos_{i}",
                timestamp=date,
                src=dst_ or None,
                dst=ret or None,
                product=None,          # clean_chain drops drug name; not needed for topology
                qty=qty,
                value=None,            # ARCOS has no monetary value
                scheduled_date=None,   # no promised date -> no disruption signal
                actual_date=None,
                po_date=None,
            ))

    ir.finalize()
    return ir


if __name__ == "__main__":
    ir = build()
    ir.print_manifest()
    out = ir.write()
    print(f"\nIR written -> {out}")
