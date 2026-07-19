"""
SCMS ADAPTER — raw SCMS Delivery History -> canonical IR
========================================================
The ONLY module that knows SCMS column names. It reads the raw SCMS delivery
history and emits the IR defined in ir_schema.py. Date parsing mirrors
scms_spine.py exactly (including the quirk that 'PO Sent to Vendor Date' uses
%m/%d/%y while the other date columns use %d-%b-%y, and the many 'Date Not
Captured' / 'N/A - From RDC' sentinels which simply parse to None).

SCMS honestly supports:
    HAS_TRANSACTIONS — Line Item Quantity per shipment
    HAS_TOPOLOGY     — 3-tier chain manufacturing_site -> vendor -> country
    HAS_DISRUPTIONS  — Scheduled vs Delivered dates (real late shipments)
    HAS_LEAD_TIMES   — PO Sent -> Delivered span
    HAS_VALUE        — Line Item Value

TOPOLOGY NOTE: an earlier version emitted only a 2-tier network (vendor ->
country) and HAS_TOPOLOGY stayed OFF. The raw data actually carries a third,
upstream tier — 'Manufacturing Site' (100% populated, 88 distinct sites) —
so we now wire the real chain manufacturing_site -> vendor -> country. That
flips HAS_TOPOLOGY ON and lets the graph-risk / PPO-routing stages run on a
dataset that ALSO has real disruptions, which ARCOS lacks. Site node ids are
namespaced ('site::…') because two site names collide with vendor names, and
add_node dedups on id — namespacing keeps the three tiers distinct.

Run:
    python3 src/scms_adapter.py         # build IR, print manifest, write output/ir/scms/
"""

from __future__ import annotations

import csv
import os
from datetime import datetime

from ir_schema import IR, Node, Edge, Flow

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCMS_IN = os.path.join(ROOT, "data", "raw", "SCMS_Delivery_History_Dataset.csv")


def parse_date(s):
    """Mirror scms_spine.parse_date, plus the %m/%d/%y form used by PO dates."""
    s = (s or "").strip()
    if not s or s.lower().startswith(("date not captured", "n/a")):
        return None
    for fmt in ("%d-%b-%y", "%m/%d/%Y", "%m/%d/%y", "%d-%b-%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _iso(dt):
    return dt.date().isoformat() if dt else None


def _to_float(s):
    try:
        return float((s or "0").replace(",", "").strip() or 0)
    except ValueError:
        return None


def build(path: str = SCMS_IN, max_rows: int | None = None) -> IR:
    if not os.path.exists(path):
        raise SystemExit(f"[scms_adapter] missing input: {path}")

    ir = IR("scms")
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for i, r in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            site     = (r.get("Manufacturing Site") or "").strip()
            vendor   = (r.get("Vendor") or "").strip()
            country  = (r.get("Country") or "").strip()
            molecule = (r.get("Molecule/Test Type") or "").strip() or None
            qty      = _to_float(r.get("Line Item Quantity"))
            value    = _to_float(r.get("Line Item Value"))

            sched    = parse_date(r.get("Scheduled Delivery Date"))
            deliv    = parse_date(r.get("Delivered to Client Date"))
            po       = parse_date(r.get("PO Sent to Vendor Date"))

            status = None
            if sched and deliv:
                status = "late" if (deliv - sched).days > 0 else "on_time"

            # nodes: three tiers — manufacturing site (upstream), vendor
            # (distributor), destination country. Site ids are namespaced so
            # the 2 site/vendor name clashes don't collapse into one node.
            site_id = f"site::{site}" if site else None
            if site_id:
                ir.add_node(Node(site_id, role="manufacturing_site"))
            if vendor:
                ir.add_node(Node(vendor, role="vendor"))
            if country:
                ir.add_node(Node(country, role="country", country=country))

            # edges: the real chain  site -> vendor -> country  (this molecule).
            if site_id and vendor:
                ir.add_edge(Edge(site_id, vendor, product=molecule))
            if vendor and country:
                ir.add_edge(Edge(vendor, country, product=molecule))

            # flow: one shipment, with the dates that unlock disruption/lead-time
            ir.add_flow(Flow(
                flow_id=f"scms_{i}",
                timestamp=_iso(deliv) or _iso(sched),
                src=vendor or None,
                dst=country or None,
                product=molecule,
                qty=qty,
                value=value,
                scheduled_date=_iso(sched),
                actual_date=_iso(deliv),
                po_date=_iso(po),
                status=status,
            ))

    ir.finalize()
    return ir


if __name__ == "__main__":
    ir = build()
    ir.print_manifest()
    out = ir.write()
    print(f"\nIR written -> {out}")
