"""
DATACO ADAPTER — raw DataCo Supply Chain dataset -> canonical IR
===============================================================
The ONLY module that knows DataCo column names. Third dataset, deliberately
from a DIFFERENT domain (online retail, not health commodities) — the point of
plug-and-play is that a dataset the IR was NOT designed around still maps in
cleanly and lights up exactly the capabilities its own fields support.

DataCo honestly supports (same profile as SCMS, different domain):
    HAS_TRANSACTIONS — Order Item Quantity
    HAS_DISRUPTIONS  — scheduled vs actual shipping dates (real late deliveries)
    HAS_LEAD_TIMES   — order date -> shipping date span
    HAS_VALUE        — Sales
Single-tier (department -> destination country), so HAS_TOPOLOGY stays OFF.

Encoding is latin-1 (the file has non-UTF-8 bytes). Dates look like
'1/31/2018 22:56' -> %m/%d/%Y %H:%M.

Run:
    python3 src/dataco_adapter.py       # build IR, print manifest, write output/ir/dataco/
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta

from ir_schema import IR, Node, Edge, Flow

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATACO_IN = os.path.join(ROOT, "data", "raw", "DataCoSupplyChainDataset.csv")


def parse_dt(s):
    s = (s or "").strip()
    for fmt in ("%m/%d/%Y %H:%M", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _iso(dt):
    return dt.date().isoformat() if dt else None


def _to_float(s):
    try:
        return float((s or "").replace(",", "").strip() or 0)
    except ValueError:
        return None


def build(path: str = DATACO_IN, max_rows: int | None = None) -> IR:
    if not os.path.exists(path):
        raise SystemExit(f"[dataco_adapter] missing input: {path}")

    ir = IR("dataco")
    with open(path, newline="", encoding="latin-1") as fh:
        reader = csv.DictReader(fh)
        for i, r in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            dept    = (r.get("Department Name") or "").strip()
            country = (r.get("Order Country") or "").strip()
            product = (r.get("Category Name") or "").strip() or None
            qty     = _to_float(r.get("Order Item Quantity"))
            value   = _to_float(r.get("Sales"))

            order_dt = parse_dt(r.get("order date (DateOrders)"))
            ship_dt  = parse_dt(r.get("shipping date (DateOrders)"))
            sched_days = _to_float(r.get("Days for shipment (scheduled)"))
            # scheduled delivery date = order date + promised transit days
            sched_dt = (order_dt + timedelta(days=int(sched_days))
                        if order_dt and sched_days is not None else None)

            status = None
            if sched_dt and ship_dt:
                status = "late" if (ship_dt - sched_dt).days > 0 else "on_time"

            if dept:
                ir.add_node(Node(dept, role="department"))
            if country:
                ir.add_node(Node(country, role="market", country=country))
            if dept and country:
                ir.add_edge(Edge(dept, country, product=product))

            ir.add_flow(Flow(
                flow_id=f"dataco_{i}",
                timestamp=_iso(ship_dt) or _iso(order_dt),
                src=dept or None,
                dst=country or None,
                product=product,
                qty=qty,
                value=value,
                scheduled_date=_iso(sched_dt),
                actual_date=_iso(ship_dt),
                po_date=_iso(order_dt),
                status=status,
            ))

    ir.finalize()
    return ir


if __name__ == "__main__":
    ir = build()
    ir.print_manifest()
    out = ir.write()
    print(f"\nIR written -> {out}")
