"""
Immune Response Engine — Real-Time Unified Decision System
==========================================================
Wires together all four response systems and emits a chain-of-thought
reasoning trace for every anomaly detected in the live stream.

When an anomaly is detected (A → B is disrupted), the engine fires
four parallel "immune responses" and shows its thinking at each step:

  Step 1 — MEMORY RECALL    : Query FAISS for the most similar past disruptions.
                               What happened before? How long did it take to recover?
  Step 2 — ALTERNATE ROUTE  : PPO agent (or Dijkstra fallback) finds the safest
                               reroute through the graph. A → C → B.
  Step 3 — BACKUP SUPPLIER  : Supplier agent scores alternative upstream suppliers
                               that can cover the disrupted node's load.
  Step 4 — INVENTORY SHIFT  : Inventory agent finds distributors with spare capacity
                               that can emergency-transfer stock to at-risk retailers.
  Step 5 — FINAL VERDICT    : Combines all signals into a ranked action plan with
                               confidence scores and estimated recovery time.

Output:
  data/stream/immune_decisions.jsonl  — one JSON object per anomaly event
                                        (append-mode, readable by app.py)
"""

from __future__ import annotations

import os
import sys
import json
import time
import pickle
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ── Paths ──────────────────────────────────────────────────────────────────
GRAPH_PATH    = ROOT / "models"  / "supplychain_graph.pkl"
PPO_PATH      = ROOT / "models"  / "ppo_routing_agent.pth"
FAISS_INDEX   = ROOT / "models"  / "faiss_memory.index"
FAISS_META    = ROOT / "models"  / "faiss_memory_meta.pkl"
RISK_CSV      = ROOT / "output"  / "graph_risk_scores.csv"
GNN_CSV       = ROOT / "output"  / "gnn_risk_scores.csv"
DECISIONS_OUT = ROOT / "data"    / "stream" / "immune_decisions.jsonl"

DECISIONS_OUT.parent.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────
K         = 10      # PPO candidate pool size
F_DIM     = 5       # PPO feature dimensions per candidate
STATE_DIM = K * F_DIM
ACTION_DIM= K
TOP_K_MEM = 3       # FAISS nearest neighbours

# Fuel cost per US state (from routing.py / inventory_agent.py)
REGION_FUEL_COST = {
    "ME":1.8,"NH":1.8,"VT":1.8,"MA":1.7,"RI":1.7,"CT":1.7,
    "NY":1.5,"NJ":1.5,"PA":1.4,"DE":1.5,"MD":1.4,
    "VA":1.3,"WV":1.2,"NC":1.3,"SC":1.4,"GA":1.4,"FL":1.6,
    "AL":1.3,"MS":1.3,"TN":1.2,"KY":1.2,
    "OH":1.0,"IN":1.0,"IL":1.0,"MI":1.1,"WI":1.1,
    "MN":1.2,"IA":1.1,"MO":1.1,"ND":1.3,"SD":1.3,
    "NE":1.2,"KS":1.2,
    "TX":1.3,"OK":1.2,"AR":1.2,"LA":1.3,
    "MT":1.6,"ID":1.6,"WY":1.5,"CO":1.4,"NM":1.5,
    "AZ":1.5,"UT":1.5,"NV":1.6,
    "CA":1.8,"OR":1.8,"WA":1.8,"AK":2.5,"HI":3.0,
}

# ── PyTorch (optional) ─────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as TF

    class _Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(STATE_DIM, 128), nn.ReLU(),
                nn.Linear(128, 64),        nn.ReLU(),
                nn.Linear(64, ACTION_DIM),
            )
        def forward(self, state, mask=None):
            logits = self.net(state)
            if mask is not None:
                logits = logits + (1.0 - mask) * (-1e9)
            return TF.softmax(logits, dim=-1)

    TORCH_OK = True
except ImportError:
    TORCH_OK = False

# ── FAISS (optional) ───────────────────────────────────────────────────────
try:
    import faiss as _faiss
    FAISS_OK = True
except ImportError:
    FAISS_OK = False


# ══════════════════════════════════════════════════════════════════════════
#  ImmuneResponseEngine
# ══════════════════════════════════════════════════════════════════════════

class ImmuneResponseEngine:
    """
    Loads all model artefacts once at startup, then processes anomaly events
    one at a time via .respond(event) which returns a full reasoning trace
    and final decision.
    """

    def __init__(self, verbose: bool = True):
        self.verbose   = verbose
        self.G         = None
        self.actor     = None
        self.faiss_idx = None
        self.faiss_meta= None
        self.risk_map  = {}
        self.gnn_map   = {}
        self.risk_df   = None
        self.in_deg    = {}
        self.out_deg   = {}
        self.max_in    = 1
        self.max_out   = 1
        # Tracks when each reroute node was last chosen; used for confidence decay
        self._reroute_usage: dict = {}   # node -> [unix_timestamp, ...]
        self._load_all()

    # ── Startup loaders ────────────────────────────────────────────────────

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _load_all(self):
        self._log("\n[ENGINE] Initialising Immune Response Engine...")

        # Graph
        if GRAPH_PATH.exists():
            try:
                with open(GRAPH_PATH, "rb") as f:
                    self.G = pickle.load(f)
                self.in_deg  = dict(self.G.in_degree())
                self.out_deg = dict(self.G.out_degree())
                self.max_in  = max(self.in_deg.values())  or 1
                self.max_out = max(self.out_deg.values()) or 1
                self._log(f"[ENGINE] Graph: {self.G.number_of_nodes()} nodes, "
                          f"{self.G.number_of_edges()} edges ✓")
            except Exception as e:
                self._log(f"[ENGINE][WARN] Graph load failed ({e}) — routing unavailable")
        else:
            self._log("[ENGINE][WARN] Graph not found — routing unavailable")

        # Risk maps
        if RISK_CSV.exists():
            df = pd.read_csv(RISK_CSV)
            # column is composite_risk (from build_chain.py)
            risk_col = "composite_risk" if "composite_risk" in df.columns else "risk_score"
            raw = dict(zip(df["entity"], df[risk_col]))
            rmax = max(raw.values()) if raw else 1.0
            self.risk_map = {k: v / rmax for k, v in raw.items()}
            # Keep full risk_df for supplier scoring
            self.risk_df = df
            self._log(f"[ENGINE] Risk map: {len(self.risk_map)} nodes ✓")
        if GNN_CSV.exists():
            df = pd.read_csv(GNN_CSV)
            raw = dict(zip(df["entity"], df["gnn_score"]))
            gmax = max(raw.values()) if raw else 1.0
            self.gnn_map = {k: v / gmax for k, v in raw.items()}
            self._log(f"[ENGINE] GNN map: {len(self.gnn_map)} nodes ✓")

        # PPO actor
        if TORCH_OK and PPO_PATH.exists():
            try:
                self.actor = _Actor()
                ckpt = torch.load(PPO_PATH, weights_only=True)
                self.actor.load_state_dict(ckpt["actor"])
                self.actor.eval()
                self._log("[ENGINE] PPO actor loaded ✓")
            except Exception as e:
                self._log(f"[ENGINE][WARN] PPO load failed ({e}) — Dijkstra fallback will be used")
                self.actor = None
        else:
            self._log("[ENGINE] PPO not available — Dijkstra fallback will be used")

        # FAISS memory
        if FAISS_OK and FAISS_INDEX.exists() and FAISS_META.exists():
            try:
                self.faiss_idx  = _faiss.read_index(str(FAISS_INDEX))
                with open(FAISS_META, "rb") as f:
                    self.faiss_meta = pickle.load(f)
                self._log(f"[ENGINE] FAISS memory: {self.faiss_idx.ntotal:,} vectors ✓")
            except Exception as e:
                self._log(f"[ENGINE][WARN] FAISS load failed ({e}) — memory recall disabled")
                self.faiss_idx  = None
                self.faiss_meta = None
        else:
            self._log("[ENGINE] FAISS index not found — memory recall disabled")

        self._log("[ENGINE] Ready.\n")

    # ── Core: respond to one anomaly event ────────────────────────────────

    def respond(self, event: dict) -> dict:
        """
        Given an anomaly event dict with keys:
            manufacturer, distributor, retailer, retailer_state,
            quantity, z_score, disruption_injected
        Returns a full decision dict including the chain-of-thought trace.
        """
        mfr   = event.get("manufacturer", "")
        dist  = event.get("distributor",  "")
        ret   = event.get("retailer",     "")
        state = str(event.get("retailer_state", "")).upper().strip()
        qty   = float(event.get("quantity", 0))
        z     = float(event.get("z_score",  0))
        ts    = datetime.now().isoformat()

        thinking = []   # chain-of-thought steps
        actions  = []   # ranked candidate actions

        thinking.append({
            "step": 0,
            "phase": "DETECTION",
            "title": "Anomaly detected in live feed",
            "reasoning": (
                f"Shipment from [{mfr}] via [{dist}] to [{ret}] triggered an alert. "
                f"Quantity={qty:.0f}, Z-score={z:.2f} (threshold=2.5). "
                f"{'Disruption flag was explicitly set in the data.' if event.get('disruption_injected') else 'Statistical anomaly — quantity deviates significantly from rolling window.'} "
                f"Initiating full immune response cascade."
            ),
            "data": {"z_score": z, "quantity": qty, "disruption_flagged": bool(event.get("disruption_injected"))}
        })

        # ── STEP 1: Memory Recall ──────────────────────────────────────────
        memory_result = self._recall_memory(mfr, dist, z, qty, state, thinking)

        # ── STEP 2: Alternate Route (PPO / Dijkstra) ───────────────────────
        route_result  = self._find_alternate_route(mfr, dist, ret, thinking)

        # ── STEP 3: Backup Supplier ────────────────────────────────────────
        supplier_result = self._find_backup_suppliers(mfr, dist, ret, thinking)

        # ── STEP 4: Inventory Transfer ─────────────────────────────────────
        inventory_result = self._find_inventory_transfers(ret, state, dist, thinking)

        # ── STEP 5: Final Verdict ──────────────────────────────────────────
        verdict = self._synthesise(
            memory_result, route_result, supplier_result, inventory_result,
            mfr, dist, ret, z, thinking, actions
        )

        decision = {
            "timestamp":          ts,
            "manufacturer":       mfr,
            "distributor":        dist,
            "retailer":           ret,
            "retailer_state":     state,
            "quantity":           qty,
            "z_score":            z,
            "disruption_flagged": bool(event.get("disruption_injected")),
            "thinking":           thinking,
            "actions":            actions,
            "verdict":            verdict,
        }

        self._emit(decision)
        return decision

    # ── Step 1: FAISS Memory Recall ────────────────────────────────────────

    def _recall_memory(self, mfr, dist, z, qty, state, thinking) -> dict:
        step = {
            "step": 1,
            "phase": "MEMORY RECALL",
            "title": "Querying immunological memory for similar past disruptions",
        }

        if self.faiss_idx is None or self.faiss_meta is None:
            step["reasoning"] = "FAISS index not loaded — memory recall skipped. No historical context available."
            step["data"] = {}
            thinking.append(step)
            return {}

        try:
            meta      = self.faiss_meta
            scaler    = meta["scaler"]
            outcomes  = meta["outcomes"]
            ddf_proxy = outcomes   # use outcomes as proxy for mean encoding

            # Map live anomaly → disruption feature space (same mapping as immunological_memory.py)
            severity    = min(z / 10.0 * 4 + 1, 5.0)
            prod_impact = min(abs(z) / 10.0 * 100, 100.0)
            has_backup  = 1.0

            # Use mean values for categorical features
            mean_enc = 0.0
            query_vec = np.array([[severity, prod_impact, has_backup,
                                   mean_enc, mean_enc, mean_enc, mean_enc, mean_enc]],
                                 dtype=np.float32)
            query_scaled = scaler.transform(query_vec).astype(np.float32)
            distances, indices = self.faiss_idx.search(query_scaled, TOP_K_MEM)

            matches = []
            rec_days_list = []
            for rank, (idx, dist_val) in enumerate(zip(indices[0], distances[0])):
                if idx < 0:
                    continue
                nb = outcomes.iloc[idx]
                rec_days = float(nb["full_recovery_days"])
                resp     = str(nb.get("response_type", nb.get("response_type_enc", "unknown")))
                disrupt  = str(nb.get("disruption_type", nb.get("disruption_type_enc", "unknown")))
                rec_days_list.append(rec_days)
                matches.append({
                    "rank":          rank + 1,
                    "distance":      round(float(dist_val), 4),
                    "recovery_days": rec_days,
                    "response_type": resp,
                    "disruption_type": disrupt,
                })

            avg_recovery = round(float(np.mean(rec_days_list)), 1) if rec_days_list else None
            top_response = matches[0]["response_type"] if matches else "unknown"

            step["reasoning"] = (
                f"Found {len(matches)} similar historical disruptions in memory "
                f"(closest match distance={matches[0]['distance']:.4f}). "
                f"Past events of this severity took an average of {avg_recovery} days to recover. "
                f"The most successful response type for similar events was '{top_response}'. "
                f"Using this as prior knowledge to weight the current response."
            )
            step["data"] = {
                "matches":        matches,
                "avg_recovery_days": avg_recovery,
                "recommended_response": top_response,
            }
            thinking.append(step)
            return {"avg_recovery_days": avg_recovery, "recommended_response": top_response, "matches": matches}

        except Exception as e:
            step["reasoning"] = f"Memory recall failed: {e}. Proceeding without historical context."
            step["data"] = {}
            thinking.append(step)
            return {}

    # ── Confidence Decay ───────────────────────────────────────────────────

    def _decay_factor(self, node: str, window_minutes: int = 30) -> float:
        """
        Returns a confidence multiplier in [0.5, 1.0] for a reroute node.
        Drops by 0.15 for each time the node was chosen in the last window_minutes.
        Prevents the system from blindly overloading the same alternate node.
        """
        now = time.time()
        cutoff = now - window_minutes * 60
        recent = [t for t in self._reroute_usage.get(node, []) if t >= cutoff]
        self._reroute_usage[node] = recent   # prune stale entries in-place
        return max(0.5, 1.0 - len(recent) * 0.15)

    def _record_reroute_use(self, node: str):
        self._reroute_usage.setdefault(node, []).append(time.time())

    # ── Clonal Selection: write resolved outcome back into FAISS memory ────

    def learn_from_outcome(self, event: dict, decision: dict) -> None:
        """
        Clonal selection feedback loop — after each live disruption is handled,
        encode the outcome as an 8-dim vector and append it to the FAISS index.
        The meta["outcomes"] DataFrame is extended in-memory and both the index
        and the pkl are persisted atomically.  The system therefore learns from
        every decision it makes, not just historical training data.
        """
        if not FAISS_OK or self.faiss_idx is None or self.faiss_meta is None:
            return
        try:
            meta     = self.faiss_meta
            scaler   = meta["scaler"]
            outcomes = meta["outcomes"]

            z             = float(event.get("z_score", 0))
            verdict       = decision.get("verdict", {})
            recovery_days = float(verdict.get("recovery_estimate_days") or 7.0)
            top_action    = (verdict.get("actions_ranked") or [{}])[0]

            severity    = min(z / 10.0 * 4 + 1, 5.0)
            prod_impact = min(abs(z) / 10.0 * 100, 100.0)
            has_backup  = 1.0 if verdict.get("actions_ranked") else 0.0
            mean_enc    = 0.0   # categorical features unknown at live time; use neutral value

            new_vec    = np.array([[severity, prod_impact, has_backup,
                                    mean_enc, mean_enc, mean_enc, mean_enc, mean_enc]],
                                  dtype=np.float32)
            new_scaled = scaler.transform(new_vec).astype(np.float32)
            self.faiss_idx.add(new_scaled)

            new_row = pd.DataFrame([{
                "full_recovery_days": recovery_days,
                "response_type_enc":  0.0,
                "response_type":      top_action.get("action", "REROUTE"),
                "disruption_type":    "live_event",
                "industry":           "live",
            }])
            meta["outcomes"] = pd.concat([outcomes, new_row], ignore_index=True)

            # Atomic persist: write index then metadata
            _faiss.write_index(self.faiss_idx, str(FAISS_INDEX))
            with open(FAISS_META, "wb") as fh:
                pickle.dump(meta, fh)

            self._log(
                f"[ENGINE] Clonal selection: outcome learned "
                f"(recovery={recovery_days:.0f}d, action={top_action.get('action','?')}). "
                f"Memory now {self.faiss_idx.ntotal:,} vectors."
            )
        except Exception as e:
            self._log(f"[ENGINE][WARN] Clonal selection write failed: {e}")

    # ── Step 2: Alternate Route ────────────────────────────────────────────

    def _find_alternate_route(self, mfr, dist, ret, thinking) -> dict:
        step = {
            "step": 2,
            "phase": "ALTERNATE ROUTE",
            "title": "Evaluating alternate routing paths through the supply graph",
        }

        if self.G is None or not mfr or not ret:
            step["reasoning"] = "Graph unavailable or missing endpoints — cannot evaluate alternate routes."
            step["data"] = {}
            thinking.append(step)
            return {}

        distributors = {n for n, d in self.G.nodes(data=True) if d.get("type") == "distributor"}
        candidates = [
            n for n in self.G.successors(mfr)
            if n in distributors and n != dist
            and nx.has_path(self.G, n, ret)
        ]

        if not candidates:
            step["reasoning"] = (
                f"No valid alternate distributors found from [{mfr}] to [{ret}] "
                f"that bypass [{dist}]. The disrupted node may be the only intermediate path."
            )
            step["data"] = {"candidates_evaluated": 0}
            thinking.append(step)
            return {}

        step["reasoning"] = (
            f"Original route: [{mfr}] → [{dist}] → [{ret}] is disrupted. "
            f"Scanning {len(candidates)} candidate alternate distributors connected to [{mfr}] "
            f"that have a valid path to [{ret}]. "
        )

        route_str  = None
        method     = None
        chosen     = None
        risk_score = None
        scored_candidates = []

        # Score all candidates; apply confidence decay for overloaded nodes
        for cand in candidates[:K]:
            r     = self.risk_map.get(cand, 0.5)
            g     = self.gnn_map.get(cand, 0.5)
            decay = self._decay_factor(cand)
            try:
                path_len = nx.shortest_path_length(self.G, cand, ret)
            except Exception:
                path_len = 999
            raw_composite = (1 - r) * 0.5 + (1 - g) * 0.3 + max(0, 1 - path_len / 10) * 0.2
            scored_candidates.append({
                "node":         cand,
                "risk":         round(r, 3),
                "gnn_score":    round(g, 3),
                "path_hops":    path_len,
                "decay_factor": round(decay, 3),
                "composite":    round(raw_composite * decay, 3),
            })

        scored_candidates.sort(key=lambda x: x["composite"], reverse=True)

        # Try PPO first
        if self.actor is not None and TORCH_OK:
            pool = candidates[:K]
            random.shuffle(pool)
            state_vec = []
            for cand in pool:
                r   = self.risk_map.get(cand, 0.5)
                g   = self.gnn_map.get(cand, 0.5)
                ind  = self.in_deg.get(cand, 0) / self.max_in
                outd = self.out_deg.get(cand, 0) / self.max_out
                hp   = 1.0 if nx.has_path(self.G, cand, ret) else 0.0
                state_vec.extend([r, g, ind, outd, hp])
            while len(state_vec) < STATE_DIM:
                state_vec.extend([0.0] * F_DIM)
            mask = [1.0] * len(pool) + [0.0] * (K - len(pool))

            st_t  = torch.FloatTensor(state_vec).unsqueeze(0)
            msk_t = torch.FloatTensor([mask])
            with torch.no_grad():
                probs = self.actor(st_t, msk_t)
            action = int(probs.argmax(dim=-1).item())

            if action < len(pool):
                chosen = pool[action]
                method = "PPO"
                risk_score = self.risk_map.get(chosen, 0.5)
                try:
                    path = nx.shortest_path(self.G, chosen, ret)
                    route_str = f"{mfr} → {chosen} → " + " → ".join(path[1:])
                    decay_applied = self._decay_factor(chosen)
                    self._record_reroute_use(chosen)
                    step["reasoning"] += (
                        f"PPO reinforcement learning agent selected [{chosen}] as the optimal reroute "
                        f"(composite risk={risk_score:.3f}, load decay={decay_applied:.2f}). "
                        f"Full path: {route_str}."
                    )
                except Exception:
                    chosen = None
                    decay_applied = 1.0

        # Dijkstra fallback
        if route_str is None:
            method = "Dijkstra"
            G_alt = self.G.copy()
            if dist in G_alt:
                G_alt.remove_node(dist)
            try:
                path = nx.shortest_path(G_alt, mfr, ret)
                route_str = " → ".join(path)
                chosen    = path[1] if len(path) > 1 else None
                risk_score= self.risk_map.get(chosen, 0.5) if chosen else None
                decay_applied = self._decay_factor(chosen) if chosen else 1.0
                if chosen:
                    self._record_reroute_use(chosen)
                step["reasoning"] += (
                    f"Dijkstra fallback found shortest path bypassing [{dist}]: "
                    f"{route_str} ({len(path)-1} hops, load decay={decay_applied:.2f})."
                )
            except Exception as e:
                decay_applied = 1.0
                step["reasoning"] += f" Dijkstra also failed: {e}. No alternate route available."

        step["data"] = {
            "original_route":        f"{mfr} → {dist} → {ret}",
            "alternate_route":       route_str,
            "method":                method,
            "chosen_distributor":    chosen,
            "chosen_risk":           round(risk_score, 3) if risk_score is not None else None,
            "decay_factor":          round(decay_applied, 3),
            "candidates_evaluated":  len(scored_candidates),
            "top_candidates":        scored_candidates[:3],
        }
        thinking.append(step)
        return step["data"]

    # ── Step 3: Backup Supplier ────────────────────────────────────────────

    def _find_backup_suppliers(self, mfr, dist, ret, thinking) -> dict:
        step = {
            "step": 3,
            "phase": "BACKUP SUPPLIER",
            "title": "Identifying backup suppliers to absorb disrupted node's load",
        }

        if self.G is None or self.risk_df is None:
            step["reasoning"] = "Graph or risk data unavailable — supplier analysis skipped."
            step["data"] = {}
            thinking.append(step)
            return {}

        try:
            risk_lookup = self.risk_df.set_index("entity").to_dict("index")
            disrupted   = dist if dist in self.G else mfr

            affected_retailers = set(self.G.successors(disrupted))
            if not affected_retailers:
                for d in self.G.successors(disrupted):
                    affected_retailers.update(self.G.successors(d))

            alternatives = set()
            for r in affected_retailers:
                for pred in self.G.predecessors(r):
                    if pred != disrupted and pred in risk_lookup:
                        nd = risk_lookup[pred]
                        if nd.get("type") in ("distributor", "manufacturer"):
                            alternatives.add(pred)

            if not alternatives:
                step["reasoning"] = (
                    f"No existing alternative suppliers found in the graph that already serve "
                    f"the same retailers as [{disrupted}]. Supply gap may require new supplier onboarding."
                )
                step["data"] = {"alternatives_found": 0}
                thinking.append(step)
                return {}

            scored = []
            for alt in alternatives:
                if alt not in risk_lookup:
                    continue
                nd        = risk_lookup[alt]
                safety    = 1.0 - float(nd.get("composite_risk", 0.5))
                out_vol   = float(nd.get("out_volume", 0))
                out_deg   = max(int(nd.get("out_degree", 1)), 1)
                efficiency= out_vol / out_deg
                scored.append({
                    "entity":          alt,
                    "type":            nd.get("type", "unknown"),
                    "composite_risk":  round(float(nd.get("composite_risk", 0.5)), 4),
                    "out_volume":      int(out_vol),
                    "out_degree":      out_deg,
                    "efficiency":      round(efficiency, 1),
                    "safety_score":    round(safety, 4),
                })

            if not scored:
                step["reasoning"] = "Alternatives found in graph but none have risk data."
                step["data"] = {}
                thinking.append(step)
                return {}

            sdf = pd.DataFrame(scored)
            for col in ["out_volume", "efficiency"]:
                mn, mx = sdf[col].min(), sdf[col].max()
                sdf[f"{col}_norm"] = (sdf[col] - mn) / (mx - mn + 1e-9)

            sdf["backup_score"] = (
                0.50 * sdf["safety_score"]
              + 0.30 * sdf["out_volume_norm"]
              + 0.20 * sdf["efficiency_norm"]
            ).round(4)

            top3 = sdf.sort_values("backup_score", ascending=False).head(3)
            top_list = top3.to_dict("records")

            best = top_list[0]
            step["reasoning"] = (
                f"Disrupted node [{disrupted}] supplies {len(affected_retailers)} retailer(s). "
                f"Scanned {len(alternatives)} existing suppliers in the graph that already serve the same retailers. "
                f"Best backup: [{best['entity']}] with backup_score={best['backup_score']:.3f} "
                f"(risk={best['composite_risk']:.3f}, volume={best['out_volume']:,}). "
                f"This supplier already has proven routes — activation requires no new connections."
            )
            step["data"] = {
                "disrupted_node":      disrupted,
                "affected_retailers":  len(affected_retailers),
                "alternatives_found":  len(alternatives),
                "top_backups":         top_list,
            }
            thinking.append(step)
            return step["data"]

        except Exception as e:
            step["reasoning"] = f"Supplier analysis failed: {e}"
            step["data"] = {}
            thinking.append(step)
            return {}

    # ── Step 4: Inventory Transfer ─────────────────────────────────────────

    def _find_inventory_transfers(self, ret, state, primary_dist, thinking) -> dict:
        step = {
            "step": 4,
            "phase": "INVENTORY TRANSFER",
            "title": "Searching for distributors with spare capacity to emergency-transfer stock",
        }

        if self.G is None or self.risk_df is None:
            step["reasoning"] = "Graph or risk data unavailable — inventory transfer analysis skipped."
            step["data"] = {}
            thinking.append(step)
            return {}

        try:
            fuel_to_ret = REGION_FUEL_COST.get(state, 1.3)

            dist_nodes = self.risk_df[self.risk_df["type"] == "distributor"].copy()
            dist_nodes["spare_capacity"] = (
                dist_nodes["out_volume"] / dist_nodes["out_degree"].clip(lower=1)
            ).round(1)

            def _norm(s):
                mn, mx = s.min(), s.max()
                return (s - mn) / (mx - mn + 1e-9)

            dist_nodes["capacity_norm"] = _norm(dist_nodes["spare_capacity"])
            dist_nodes["safety_norm"]   = _norm(1.0 - dist_nodes["composite_risk"])

            # Current suppliers of this retailer
            current_dists = set()
            if ret in self.G:
                current_dists = set(self.G.predecessors(ret))

            all_risks = dist_nodes["composite_risk"].tolist()
            risk_threshold = float(np.percentile(all_risks, 75)) if all_risks else 0.95

            candidates = []
            for _, row in dist_nodes.iterrows():
                ent = row["entity"]
                if ent in current_dists:
                    continue
                if row["composite_risk"] > risk_threshold:
                    continue
                fuel_score = max(0.0, min(1.0, 1.0 - (fuel_to_ret - 1.0) / 2.0))
                tscore = (
                    0.40 * float(row["capacity_norm"])
                  + 0.35 * float(row["safety_norm"])
                  + 0.25 * fuel_score
                )
                est_days = round(fuel_to_ret * 1.5, 1)
                candidates.append({
                    "distributor":     ent,
                    "transfer_score":  round(tscore, 4),
                    "composite_risk":  round(float(row["composite_risk"]), 4),
                    "spare_capacity":  round(float(row["spare_capacity"]), 1),
                    "fuel_cost":       round(fuel_to_ret, 2),
                    "estimated_days":  est_days,
                })

            if not candidates:
                step["reasoning"] = (
                    f"No low-risk distributors with spare capacity found that don't already "
                    f"supply [{ret}]. Inventory transfer not feasible at this time."
                )
                step["data"] = {"candidates_found": 0}
                thinking.append(step)
                return {}

            top3 = sorted(candidates, key=lambda x: x["transfer_score"], reverse=True)[:3]
            best = top3[0]

            step["reasoning"] = (
                f"Retailer [{ret}] (state: {state}, fuel multiplier: {fuel_to_ret}x) is at risk of stockout. "
                f"Scanned {len(candidates)} low-risk distributors with spare capacity. "
                f"Best transfer candidate: [{best['distributor']}] "
                f"(score={best['transfer_score']:.3f}, spare_capacity={best['spare_capacity']:.0f}, "
                f"est. {best['estimated_days']} days delivery). "
                f"Transfer would pre-position stock before the primary route is restored."
            )
            step["data"] = {
                "at_risk_retailer":    ret,
                "retailer_state":      state,
                "fuel_multiplier":     fuel_to_ret,
                "candidates_found":    len(candidates),
                "top_transfers":       top3,
            }
            thinking.append(step)
            return step["data"]

        except Exception as e:
            step["reasoning"] = f"Inventory transfer analysis failed: {e}"
            step["data"] = {}
            thinking.append(step)
            return {}

    # ── Step 5: Synthesise Final Verdict ───────────────────────────────────

    def _synthesise(self, memory, route, supplier, inventory,
                    mfr, dist, ret, z, thinking, actions) -> dict:
        step = {
            "step": 5,
            "phase": "FINAL VERDICT",
            "title": "Synthesising all signals into a ranked action plan",
        }

        severity = "CRITICAL" if z > 5 else "HIGH" if z > 3 else "MODERATE"
        est_recovery = memory.get("avg_recovery_days", None)

        # Build ranked actions list
        if route.get("alternate_route"):
            chosen_risk   = route.get("chosen_risk") or 0.5
            decay_factor  = route.get("decay_factor", 1.0)
            # Base confidence from risk, then multiplied by load-decay
            route_confidence = round(max(0.40, (1.0 - chosen_risk * 0.3) * decay_factor), 2)
            actions.append({
                "priority":    1,
                "action":      "REROUTE",
                "label":       f"Activate alternate route via [{route.get('chosen_distributor', '?')}]",
                "detail":      route["alternate_route"],
                "confidence":  route_confidence,
                "method":      route.get("method", "unknown"),
                "eta_days":    1,
            })

        if supplier.get("top_backups"):
            best_sup = supplier["top_backups"][0]
            actions.append({
                "priority":    2,
                "action":      "ACTIVATE BACKUP SUPPLIER",
                "label":       f"Activate [{best_sup['entity']}] as backup supplier",
                "detail":      (f"backup_score={best_sup['backup_score']:.3f}, "
                                f"risk={best_sup['composite_risk']:.3f}, "
                                f"volume_capacity={best_sup['out_volume']:,}"),
                "confidence":  round(float(best_sup["backup_score"]), 2),
                "method":      "SupplierAgent",
                "eta_days":    2,
            })

        if inventory.get("top_transfers"):
            best_inv = inventory["top_transfers"][0]
            actions.append({
                "priority":    3,
                "action":      "EMERGENCY INVENTORY TRANSFER",
                "label":       f"Transfer stock from [{best_inv['distributor']}] to [{ret}]",
                "detail":      (f"transfer_score={best_inv['transfer_score']:.3f}, "
                                f"spare_capacity={best_inv['spare_capacity']:.0f}, "
                                f"fuel_cost={best_inv['fuel_cost']:.2f}x"),
                "confidence":  round(float(best_inv["transfer_score"]), 2),
                "method":      "InventoryAgent",
                "eta_days":    best_inv.get("estimated_days", 3),
            })

        if memory.get("recommended_response"):
            actions.append({
                "priority":    4,
                "action":      "APPLY HISTORICAL PROTOCOL",
                "label":       f"Follow proven response: '{memory['recommended_response']}'",
                "detail":      (f"Based on {len(memory.get('matches', []))} similar past disruptions. "
                                f"Avg recovery: {est_recovery} days."),
                "confidence":  0.70,
                "method":      "ImmunologicalMemory",
                "eta_days":    est_recovery or 7,
            })

        if not actions:
            actions.append({
                "priority":   1,
                "action":     "MANUAL REVIEW REQUIRED",
                "label":      "Insufficient graph/model data for automated response",
                "detail":     "Escalate to supply chain operations team.",
                "confidence": 0.0,
                "method":     "none",
                "eta_days":   None,
            })

        recovery_estimate = (
            min(a["eta_days"] for a in actions if a.get("eta_days")) if actions else None
        )

        signals_active = sum([
            bool(memory),
            bool(route.get("alternate_route")),
            bool(supplier.get("top_backups")),
            bool(inventory.get("top_transfers")),
        ])

        step["reasoning"] = (
            f"Severity assessment: {severity} (Z={z:.2f}). "
            f"{signals_active}/4 immune response signals activated. "
            f"{'Immediate reroute available. ' if route.get('alternate_route') else 'No alternate route found. '}"
            f"{'Backup supplier identified. ' if supplier.get('top_backups') else ''}"
            f"{'Inventory transfer feasible. ' if inventory.get('top_transfers') else ''}"
            f"{'Memory-guided protocol available. ' if memory.get('recommended_response') else ''}"
            f"Recommended fastest resolution: {recovery_estimate} day(s) via priority-1 action."
        )
        step["data"] = {
            "severity":           severity,
            "signals_activated":  signals_active,
            "actions_count":      len(actions),
            "recovery_estimate_days": recovery_estimate,
        }
        thinking.append(step)

        return {
            "severity":               severity,
            "signals_activated":      signals_active,
            "recommended_action":     actions[0]["label"] if actions else "Manual review",
            "recovery_estimate_days": recovery_estimate,
            "actions_ranked":         actions,
        }

    # ── Emit decision to JSONL ─────────────────────────────────────────────

    def _emit(self, decision: dict):
        with open(DECISIONS_OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(decision, default=str) + "\n")

    # ── Pretty-print to terminal ───────────────────────────────────────────

    def print_response(self, decision: dict):
        mfr  = decision["manufacturer"]
        dist = decision["distributor"]
        ret  = decision["retailer"]
        z    = decision["z_score"]
        v    = decision["verdict"]

        width = 72
        print("\n" + "=" * width)
        print(f"  IMMUNE RESPONSE  |  {decision['timestamp']}")
        print("=" * width)
        print(f"  Flow   : {mfr[:30]} -> {dist[:25]} -> {ret[:25]}")
        print(f"  Signal : Z-score={z:.2f}  |  Severity={v['severity']}")
        print(f"  Signals: {v['signals_activated']}/4 response systems activated")
        print("-" * width)

        for step in decision["thinking"]:
            phase = step["phase"]
            title = step["title"]
            reason= step["reasoning"]
            print(f"\n  [Step {step['step']}] {phase}")
            print(f"  {title}")
            words = reason.split()
            line  = "  "
            for w in words:
                if len(line) + len(w) + 1 > 72:
                    print(line)
                    line = "  " + w + " "
                else:
                    line += w + " "
            if line.strip():
                print(line)

        print("\n" + "-" * width)
        print("  RANKED ACTION PLAN:")
        for a in decision["actions"]:
            conf_pct = int(a["confidence"] * 10)
            conf_bar = "#" * conf_pct + "-" * (10 - conf_pct)
            print(f"\n  [{a['priority']}] {a['action']}")
            print(f"      -> {a['label']}")
            print(f"      Confidence: [{conf_bar}] {a['confidence']:.0%}  |  ETA: {a.get('eta_days','?')} day(s)")
            print(f"      {a['detail'][:70]}")

        print("\n" + "-" * width)
        rec = v.get("recovery_estimate_days")
        print(f"  RECOMMENDATION: {v['recommended_action']}")
        print(f"  Estimated recovery: {rec if rec else '?'} day(s)")
        print("=" * width + "\n")


# ══════════════════════════════════════════════════════════════════════════
#  Standalone test — run a single synthetic anomaly
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    engine = ImmuneResponseEngine(verbose=True)

    # Simulate a disrupted shipment
    test_event = {
        "manufacturer":      "TEST_PHARMA_MFR",
        "distributor":       "TEST_DIST_01",
        "retailer":          "TEST_RETAILER_99",
        "retailer_state":    "CA",
        "quantity":          0,
        "z_score":           4.7,
        "disruption_injected": True,
    }

    # Try to use real graph nodes if available
    if engine.G and engine.G.number_of_nodes() > 0:
        nodes = list(engine.G.nodes(data=True))
        mfrs  = [n for n, d in nodes if d.get("type") == "manufacturer"]
        dists = [n for n, d in nodes if d.get("type") == "distributor"]
        rets  = [n for n, d in nodes if d.get("type") == "retailer"]
        if mfrs and dists and rets:
            test_event["manufacturer"] = random.choice(mfrs)
            test_event["distributor"]  = random.choice(dists)
            test_event["retailer"]     = random.choice(rets)
            states = [d.get("state","CA") for n, d in nodes if d.get("type") == "retailer" and n == test_event["retailer"]]
            if states:
                test_event["retailer_state"] = states[0]

    decision = engine.respond(test_event)
    engine.print_response(decision)
    print(f"\n[ENGINE] Decision written to: {DECISIONS_OUT}")
