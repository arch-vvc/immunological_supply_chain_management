"""
IR STAGES — dataset-agnostic analytics that consume the canonical IR
====================================================================
Where the capability gate stops being descriptive and runs for real. Flow
stages read output/ir/<dataset>/flows.csv; graph stages additionally read
nodes.csv + edges.csv. Every stage declares REQUIRES; the runner refuses to
run it unless the dataset's manifest satisfies every capability. Same code
on every dataset; a capability is never borrowed.

Seven stages, each returning proper metrics (not just counts):

  anomaly_detection    HAS_TRANSACTIONS   synthetic-injection benchmark ->
                                          precision / recall / F1 (any dataset)
  disruption_detection HAS_DISRUPTIONS    REAL late/on-time labels, temporal
                                          holdout -> precision / recall / F1
  safety_stock         HAS_LEAD_TIMES     King's SS -> $ buffer
  event_replay         HAS_DISRUPTIONS    walk-forward binomial surprise ->
                                          anomalous late-rate windows (alerts)
  value_at_risk        HAS_VALUE+DISRUPT  $ value riding on late flows
  graph_risk_routing   HAS_TOPOLOGY       centrality risk (mirrors Stage 4's
                                          composite) validated by volume/late
                                          concentration + knockout reroute test
  policy_gradient_routing HAS_TOPOLOGY     linear softmax policy, clipped policy-
                                          gradient (REINFORCE-style, NO critic) on a multi-
                                          step reroute CASCADE (capacity +
                                          load-dependent risk) vs myopic risk-
                                          greedy / Dijkstra / random baselines
                                          (mirrors the Stage 11 cascade env)

Metrics conventions:
  precision = TP/(TP+FP)   recall = TP/(TP+FN)   f1 = 2PR/(P+R)
  Every detector is reported against a NAIVE baseline (base rate) so the
  number can be judged, not taken on faith.

Pure stdlib. Run standalone:
    python3 src/ir_stages.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics
from collections import defaultdict
from datetime import date

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IR_DIR = os.path.join(ROOT, "output", "ir")

# Each stage names the capabilities it needs. The runner enforces these.
REQUIRES = {
    "anomaly_detection":    ["HAS_TRANSACTIONS"],
    "disruption_detection": ["HAS_DISRUPTIONS"],
    "safety_stock":         ["HAS_LEAD_TIMES"],
    "event_replay":         ["HAS_DISRUPTIONS"],
    "value_at_risk":        ["HAS_VALUE", "HAS_DISRUPTIONS"],
    "graph_risk_routing":   ["HAS_TOPOLOGY"],
    "policy_gradient_routing": ["HAS_TOPOLOGY"],
}

Z_95, HOLDING_RATE = 1.65, 0.25       # King's formula constants (mirror Stage 22)


# ── helpers ──────────────────────────────────────────────────────────────────

def _f(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _prf(tp, fp, fn):
    """precision, recall, f1 from a confusion count."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def _robust_flags(values, thresh):
    """Median/MAD robust z-score flags (mirrors anomaly_detection.robust_z)."""
    med   = statistics.median(values)
    scale = 1.4826 * statistics.median([abs(v - med) for v in values])
    if scale < 1e-9:
        sd = statistics.pstdev(values)
        scale = sd if sd > 1e-9 else 1.0
    return [1 if abs(v - med) / scale >= thresh else 0 for v in values]


def load_flows(dataset: str) -> list[dict]:
    path = os.path.join(IR_DIR, dataset, "flows.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no IR for '{dataset}' — run its adapter first ({path})")
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_capabilities(dataset: str) -> dict:
    with open(os.path.join(IR_DIR, dataset, "manifest.json")) as fh:
        return json.load(fh)["capabilities"]


def load_table(dataset: str, table: str) -> list[dict]:
    """nodes.csv / edges.csv companion loader for the graph stages."""
    path = os.path.join(IR_DIR, dataset, f"{table}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no IR table '{table}' for '{dataset}' ({path})")
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ── Stage: anomaly_detection — synthetic-injection precision/recall/F1 ───────

def anomaly_detection(flows: list[dict], contamination: float = 0.02,
                      thresh: float = 3.5, seed: int = 42) -> dict:
    """Inject synthetic volume spikes (8–20x) into a copy of the quantities,
    run the robust-z detector, and score it against the injected labels. This
    is the IR-native, dataset-agnostic version of anomaly_eval_injection.py:
    standard practice for evaluating an unlabeled detector (Emmott et al.)."""
    qtys = [q for q in (_f(r.get("qty")) for r in flows) if q is not None]
    n = len(qtys)
    if n < 50:
        return {"n_flows": n, "note": "too few quantities to benchmark"}

    rng = random.Random(seed)
    n_inject = max(1, int(n * contamination))
    inject = set(rng.sample(range(n), n_inject))
    aug, labels = list(qtys), [0] * n
    for i in inject:
        aug[i] = aug[i] * rng.uniform(8, 20) + 1.0     # +1 so a 0 qty still spikes
        labels[i] = 1

    preds = _robust_flags(aug, thresh)
    tp = sum(1 for i in range(n) if preds[i] and labels[i])
    fp = sum(1 for i in range(n) if preds[i] and not labels[i])
    fn = sum(1 for i in range(n) if not preds[i] and labels[i])
    p, r, f1 = _prf(tp, fp, fn)
    # Naive baseline: random flagging with the SAME flag budget k has expected
    # precision = contamination rate (labels are independent of a random draw).
    k = tp + fp
    rand_p = n_inject / n
    rand_r = k / n
    rand_f1 = round(2 * rand_p * rand_r / (rand_p + rand_r), 4) if (rand_p + rand_r) else 0.0
    return {
        "n_flows": n, "n_injected": n_inject,
        "precision": p, "recall": r, "f1": f1,
        "baseline_random_f1": rand_f1,
        "flagged": k, "threshold": thresh,
        "eval": "synthetic injection (2% volume spikes)",
    }


# ── Stage: disruption_detection — REAL labels, temporal holdout ──────────────

def disruption_detection(flows: list[dict], split: float = 0.7,
                         rate_thresh: float = 0.5) -> dict:
    """Predict LATE deliveries using each lane's historical late-rate learned
    on the earlier 70% of shipments, evaluated on the later 30%. Ground truth
    is the REAL status label the adapter derived from scheduled-vs-actual
    dates — no synthetic labels, no leakage (train strictly precedes test)."""
    rows = [(r.get("timestamp"),
             (r.get("src"), r.get("dst"), r.get("product")),
             1 if r.get("status") == "late" else 0)
            for r in flows
            if r.get("status") in ("late", "on_time") and r.get("timestamp")]
    rows.sort(key=lambda t: t[0])
    n = len(rows)
    if n < 50:
        return {"n_labeled": n, "note": "too few labeled flows to evaluate"}

    cut = int(n * split)
    train, test = rows[:cut], rows[cut:]
    late, tot = defaultdict(int), defaultdict(int)
    for _, lane, y in train:
        tot[lane] += 1
        late[lane] += y
    base = sum(y for *_, y in train) / len(train)   # global late-rate fallback

    tp = fp = fn = 0
    for _, lane, y in test:
        rate = late[lane] / tot[lane] if tot[lane] else base
        pred = 1 if rate >= rate_thresh else 0
        if pred and y:       tp += 1
        elif pred and not y: fp += 1
        elif not pred and y: fn += 1
    p, r, f1 = _prf(tp, fp, fn)
    test_base = sum(y for *_, y in test) / len(test)
    # Naive baseline: flag EVERYTHING late -> precision = base rate, recall = 1.
    base_f1 = round(2 * test_base / (1 + test_base), 4)
    return {
        "n_test": len(test), "base_late_rate": round(test_base, 4),
        "precision": p, "recall": r, "f1": f1,
        "precision_lift_over_base": round(p / test_base, 2) if test_base else None,
        "baseline_always_late_f1": base_f1,
        "method": "lane-history temporal holdout (70/30)",
    }


# ── Stage: safety_stock — King's SS -> $ buffer ──────────────────────────────

def _iso_to_date(s):
    try:
        y, m, d = s.split("-")
        return date(int(y), int(m), int(d))
    except Exception:
        return None


def safety_stock(flows: list[dict]) -> dict:
    """King's SS = z*sqrt(L*sd^2 + D^2*sL^2) per (src,dst,product) lane. IR-
    native, all-lanes capability demo (NOT the Stage-22 exposed-lane subset)."""
    lanes = defaultdict(list)
    for r in flows:
        qty = _f(r.get("qty"))
        po  = _iso_to_date(r.get("po_date") or "")
        act = _iso_to_date(r.get("actual_date") or "")
        if qty is None or po is None or act is None:
            continue
        lead = (act - po).days
        if lead <= 0:
            continue
        lanes[(r.get("src"), r.get("dst"), r.get("product"))].append(
            (act, qty, _f(r.get("value")), lead))

    total_buffer, sized = 0.0, 0
    for ships in lanes.values():
        if len(ships) < 3:
            continue
        ships.sort(key=lambda t: t[0])
        qtys  = [q for _, q, _, _ in ships]
        leads = [l for _, _, _, l in ships]
        span  = max((ships[-1][0] - ships[0][0]).days, 90)
        d_mean = sum(qtys) / span
        # SAMPLE std (statistics.stdev), matching scms_safety_stock.py — the
        # population estimator used before understates variance at n=3.
        d_std  = statistics.stdev(qtys) / math.sqrt(span) if len(qtys) > 1 else 0.0
        l_mean = statistics.mean(leads)
        l_std  = statistics.stdev(leads) if len(leads) > 1 else 0.0
        if d_mean <= 0:
            continue
        ss95   = Z_95 * math.sqrt(l_mean * d_std ** 2 + d_mean ** 2 * l_std ** 2)
        prices = [v / q for _, q, v, _ in ships if v and q > 0]
        total_buffer += ss95 * (statistics.median(prices) if prices else 0.0)
        sized += 1
    return {
        "n_lanes_sized": sized,
        "total_buffer_usd": round(total_buffer, 0),
        "holding_usd_per_yr": round(total_buffer * HOLDING_RATE, 0),
        "service_level": "95%",
    }


# ── Stage: event_replay — blind walk-forward late-rate surprise ─────────────

def event_replay(flows: list[dict], alpha: float = 1e-3, min_n: int = 10,
                 window_months: int = 3, min_prior_months: int = 12,
                 min_prior_n: int = 30) -> dict:
    """Blind WALK-FORWARD late-rate surprise test, per destination.

    Mirrors scms_event_replay.py exactly in what each window is allowed to
    know: a rolling `window_months` window ending at month m is tested against
    the destination's OWN late rate over the months strictly BEFORE the window
    (Laplace-smoothed), with an exact one-sided binomial tail. Nothing after
    the window — and no other destination — enters the baseline. The previous
    IR version tested every window against the whole-history global rate,
    which used the future and was not walk-forward despite claiming to be.
    """
    per_loc: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # loc -> ym -> [late, n]
    g_late = g_tot = 0
    for r in flows:
        if r.get("status") not in ("late", "on_time"):
            continue
        ts = r.get("timestamp") or ""
        if len(ts) < 7:
            continue
        y = 1 if r.get("status") == "late" else 0
        per_loc[r.get("dst")][ts[:7]][0] += y
        per_loc[r.get("dst")][ts[:7]][1] += 1
        g_late += y
        g_tot  += 1
    if g_tot == 0:
        return {"note": "no dated labeled flows"}

    def _binom_tail(k, n, p):
        """P[X >= k] for X ~ Binomial(n, p), summed in log space so large
        windows (DataCo has thousands of flows per destination-month) do not
        overflow the way math.comb(n, j) * p**j does."""
        if k <= 0:
            return 1.0
        if p <= 0.0:
            return 0.0 if k > 0 else 1.0
        if p >= 1.0:
            return 1.0
        lp, lq = math.log(p), math.log1p(-p)
        lg_n1 = math.lgamma(n + 1)
        terms = [lg_n1 - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * lp + (n - j) * lq
                 for j in range(k, n + 1)]
        m = max(terms)
        return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))

    def _month_index(ym):
        y, m = int(ym[:4]), int(ym[5:7])
        return y * 12 + (m - 1)

    alerts = []
    tested = 0
    for loc, months in per_loc.items():
        by_idx = {_month_index(ym): v for ym, v in months.items()}
        if not by_idx:
            continue
        first, last = min(by_idx), max(by_idx)
        for mi in range(first + min_prior_months, last + 1):
            win = range(mi - window_months + 1, mi + 1)
            n_win    = sum(by_idx[j][1] for j in win if j in by_idx)
            late_win = sum(by_idx[j][0] for j in win if j in by_idx)
            if n_win < min_n:
                continue
            prior = [j for j in by_idx if j < mi - window_months + 1]
            n_pri    = sum(by_idx[j][1] for j in prior)
            late_pri = sum(by_idx[j][0] for j in prior)
            if n_pri < min_prior_n:
                continue
            p0 = (late_pri + 1) / (n_pri + 2)                # Laplace-smoothed prior rate
            pval = _binom_tail(late_win, n_win, p0)
            tested += 1
            if pval < alpha:
                ym = f"{mi // 12:04d}-{mi % 12 + 1:02d}"
                alerts.append((pval, loc, ym, late_win, n_win, p0))
    alerts.sort(key=lambda t: (t[0], t[1], t[2]))
    fmt = [f"{loc} {ym}: {late}/{tot} late vs {p0:.1%} prior (p={pv:.1e})"
           for pv, loc, ym, late, tot, p0 in alerts]
    return {
        "windows_tested": tested, "n_alerts": len(alerts),
        "base_late_rate": round(g_late / g_tot, 4),            # reference only, not used in the test
        "top_alerts": fmt[:5],
        "alerts": fmt,
        "protocol": (f"walk-forward: {window_months}-month window vs the destination's own "
                     f"prior late rate (>= {min_prior_months} prior months, >= {min_prior_n} "
                     f"prior shipments), exact binomial tail, alpha={alpha:g}"),
    }


# ── Stage: value_at_risk — $ riding on late flows ────────────────────────────

def value_at_risk(flows: list[dict]) -> dict:
    late_val = total_val = 0.0
    n_late = 0
    for r in flows:
        v = _f(r.get("value"))
        if v is None:
            continue
        total_val += v
        if r.get("status") == "late":
            late_val += v
            n_late += 1
    return {
        "n_late_flows": n_late,
        "value_at_risk_usd": round(late_val, 0),
        "total_value_usd": round(total_val, 0),
        "late_value_share": round(late_val / total_val, 4) if total_val else 0.0,
    }


# ── Graph helpers (pure stdlib, shared by the two topology stages) ───────────

def _build_graph(nodes: list[dict], edges: list[dict]):
    """Adjacency / reverse adjacency over deduped (src,dst) pairs."""
    roles = {n["node_id"]: (n.get("role") or None) for n in nodes if n.get("node_id")}
    adj, radj = defaultdict(set), defaultdict(set)
    for e in edges:
        s, d = e.get("src"), e.get("dst")
        if s and d and s != d:
            adj[s].add(d)
            radj[d].add(s)
    node_list = sorted(set(roles) | set(adj) | set(radj))
    return roles, adj, radj, node_list


def _minmax(d: dict) -> dict:
    lo, hi = min(d.values()), max(d.values())
    span = (hi - lo) or 1.0
    return {k: (v - lo) / span for k, v in d.items()}


def _betweenness(adj, node_list, k=200, seed=42):
    """Brandes betweenness, sampled over k sources for tractability on the
    ARCOS-sized graph. Exact when |V| <= k."""
    from collections import deque
    bc = dict.fromkeys(node_list, 0.0)
    rng = random.Random(seed)
    sources = node_list if len(node_list) <= k else rng.sample(node_list, k)
    for s in sources:
        S, P = [], defaultdict(list)
        sigma, dist = defaultdict(float), {s: 0}
        sigma[s] = 1.0
        Q = deque([s])
        while Q:
            v = Q.popleft()
            S.append(v)
            for w in sorted(adj.get(v, ())):   # sorted: deterministic across processes
                if w not in dist:
                    dist[w] = dist[v] + 1
                    Q.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)
        delta = defaultdict(float)
        for w in reversed(S):
            for v in P[w]:
                delta[v] += sigma[v] / sigma[w] * (1 + delta[w])
            if w != s:
                bc[w] += delta[w]
    return bc


def _pagerank(adj, node_list, iters=50, damp=0.85):
    n = len(node_list)
    pr = dict.fromkeys(node_list, 1.0 / n)
    for _ in range(iters):
        new = dict.fromkeys(node_list, (1 - damp) / n)
        dangling = 0.0
        for v in node_list:
            out = adj.get(v)
            if out:
                share = pr[v] / len(out)
                for w in sorted(out):          # sorted: deterministic float order
                    if w in new:
                        new[w] += damp * share
            else:
                dangling += pr[v]
        if dangling:
            spread = damp * dangling / n
            for v in node_list:
                new[v] += spread
        pr = new
    return pr


def _risk_scores(adj, radj, node_list, seed=42):
    """Composite mirrors Stage 4 (risk_analysis.py):
    0.5*betweenness + 0.3*in_degree + 0.2*pagerank, each minmax-normalised."""
    bc = _minmax(_betweenness(adj, node_list, seed=seed))
    indeg = _minmax({v: float(len(radj.get(v, ()))) for v in node_list})
    pr = _minmax(_pagerank(adj, node_list))
    return {v: 0.5 * bc[v] + 0.3 * indeg[v] + 0.2 * pr[v] for v in node_list}


# ── Stage: graph_risk_routing — centrality risk + knockout reroute test ──────

def graph_risk_routing(flows: list[dict], nodes: list[dict], edges: list[dict],
                       seed: int = 42) -> dict:
    """Score every node with the Stage-4 composite, then VALIDATE the scores
    two ways instead of taking them on faith:
      1. concentration — share of flow quantity touching the top-risk decile,
         vs the median share of 20 random deciles (same size) as the naive
         baseline. If risk is just noise, the two match.
      2. late-rate lift — where real late/on-time labels exist, late rate of
         flows touching the top decile vs the global base rate.
    Plus the routing half: knock out each top-risk middle-tier node and
    measure what fraction of its destinations still have an alternate
    supplier (reroute feasibility), vs random knockouts."""
    roles, adj, radj, node_list = _build_graph(nodes, edges)
    if len(node_list) < 10:
        return {"note": "graph too small to score"}
    risk = _risk_scores(adj, radj, node_list, seed=seed)
    ranked = sorted(node_list, key=lambda v: risk[v], reverse=True)
    top = set(ranked[:max(1, len(node_list) // 10)])

    # 1. volume concentration vs random-decile baseline
    tot_q = top_q = 0.0
    for f in flows:
        q = _f(f.get("qty"))
        if q is None:
            continue
        tot_q += q
        if f.get("src") in top or f.get("dst") in top:
            top_q += q
    share = top_q / tot_q if tot_q else 0.0
    rng = random.Random(seed)
    draws = []
    for _ in range(20):
        rand_set = set(rng.sample(node_list, len(top)))
        rq = sum(q for f in flows
                 if (q := _f(f.get("qty"))) is not None
                 and (f.get("src") in rand_set or f.get("dst") in rand_set))
        draws.append(rq / tot_q if tot_q else 0.0)
    rand_share = statistics.median(draws)

    # 2. late-rate lift on top-decile flows (only if real labels exist)
    labeled = [f for f in flows if f.get("status") in ("late", "on_time")]
    late_lift = None
    if len(labeled) >= 100:
        base = sum(1 for f in labeled if f["status"] == "late") / len(labeled)
        touch = [f for f in labeled if f.get("src") in top or f.get("dst") in top]
        if base > 0 and len(touch) >= 30:
            top_rate = sum(1 for f in touch if f["status"] == "late") / len(touch)
            late_lift = round(top_rate / base, 2)

    # 3. knockout reroute feasibility, top-risk vs random middle-tier nodes
    mid = [v for v in node_list if adj.get(v) and radj.get(v)]
    def reroute(knock):
        dsts = adj.get(knock, set())
        if not dsts:
            return None
        return sum(1 for r in dsts if radj[r] - {knock}) / len(dsts)
    mid_set = set(mid)
    top_mid = [v for v in ranked if v in mid_set][:5]
    top_rates = [x for x in (reroute(v) for v in top_mid) if x is not None]
    rand_rates = ([x for v in (rng.choice(mid) for _ in range(20))
                   if (x := reroute(v)) is not None] if mid else [])
    return {
        "n_nodes": len(node_list), "n_edges": sum(len(s) for s in adj.values()),
        "top_risk_nodes": ranked[:3],
        "top_decile_volume_share": round(share, 4),
        "random_decile_volume_share": round(rand_share, 4),
        "concentration_lift": round(share / rand_share, 2) if rand_share else None,
        "late_rate_lift_top_decile": late_lift,
        "knockout_reroute_rate_top5": round(statistics.mean(top_rates), 4) if top_rates else None,
        "knockout_reroute_rate_random": round(statistics.mean(rand_rates), 4) if rand_rates else None,
    }


# ── Stage: policy_gradient_routing — cascade clipped policy gradient vs baselines ─

def policy_gradient_routing(flows: list[dict], nodes: list[dict], edges: list[dict],
                         seed: int = 42, pool_size: int = 4, d_steps: int = 6,
                         cap: int = 2, load_step: float = 0.15,
                         episodes: int = 2400, eval_episodes: int = 300,
                         clip: float = 0.2, gamma: float = 0.99,
                         lr: float = 0.05) -> dict:
    """IR-native Stage 11, cascade edition. The one-step task ('pick the
    lowest-risk valid candidate') is analytically solved by risk-greedy, so a
    learned policy could only tie it. Here an episode is D_STEPS sequential
    reroute demands (destinations of a disrupted middle-tier node) served
    from a TIGHT pool of alternate suppliers (pool_size x cap capacity vs
    d_steps demands), where every use raises a candidate's effective risk by
    load_step (config decay_per_use — clonal exhaustion) and not every
    candidate can serve every demand. Foresight now has value: burning the
    flexible low-risk candidate early corners the policy later (-20 per
    unserved demand). Linear softmax policy trained with a PPO-STYLE clipped
    surrogate on Monte-Carlo returns — no value function / critic, so this is
    clipped REINFORCE, not PPO; the stage is named accordingly. Batch-normalised
    Monte-Carlo returns — pure stdlib, runs on any IR dataset with topology.
    Baselines replay the SAME episodes with their own load state: myopic
    effective-risk greedy, Dijkstra (hops all 1 -> risk-blind random valid),
    and random-valid. Spread floor adapts to the mid-tier risk scale."""
    roles, adj, radj, node_list = _build_graph(nodes, edges)
    risk = _risk_scores(adj, radj, node_list, seed=seed)
    outdeg = _minmax({v: float(len(adj.get(v, ()))) for v in node_list})
    mid = [v for v in node_list if adj.get(v) and radj.get(v)
           and len(adj[v]) >= d_steps]
    if len(mid) < 5:
        return {"note": "not enough middle-tier nodes with >= d_steps destinations"}
    mid_range = max(risk[v] for v in mid) - min(risk[v] for v in mid)
    min_spread = max(0.01, 0.25 * mid_range)
    FAIL = -20.0

    rng = random.Random(seed)

    def build_episode():
        d = rng.choice(mid)
        demands = rng.sample(sorted(adj[d]), d_steps)
        alternates = sorted({c for r in demands for c in radj[r]} - {d})
        if len(alternates) < pool_size:
            return None
        pool = rng.sample(alternates, pool_size)
        base = [risk[c] for c in pool]
        if max(base) - min(base) < min_spread:
            return None
        valid = [[r in adj[c] for c in pool] for r in demands]
        if any(not any(row) for row in valid):      # serveable at t=0
            return None
        return pool, base, valid

    need = eval_episodes + max(800, episodes // 3)   # distinct specs; training cycles them
    specs, attempts = [], 0
    while len(specs) < need and attempts < need * 80:
        attempts += 1
        ep = build_episode()
        if ep:
            specs.append(ep)
    if len(specs) < eval_episodes + 200:
        return {"note": f"only {len(specs)} usable cascade episodes — too few to train"}
    train_specs, test_specs = specs[:-eval_episodes], specs[-eval_episodes:]

    W = [0.0] * 6   # [base_rel, eff_rel, valid, cap_remaining, outdeg, bias]

    def slot_feats(pool, base_rel, valid_row, loads, ok):
        effs = [min(1.0, risk[pool[i]] + load_step * loads[i])
                for i in range(len(pool))]
        elo, esp = min(effs), (max(effs) - min(effs)) or 1.0
        feats = [[base_rel[i], (effs[i] - elo) / esp,
                  1.0 if i in ok else 0.0, (cap - loads[i]) / cap,
                  outdeg[pool[i]], 1.0] for i in range(len(pool))]
        return feats, effs

    def masked_softmax(feats, ok):
        ls = {i: sum(w * x for w, x in zip(W, feats[i])) for i in ok}
        m = max(ls.values())
        ex = {i: math.exp(l - m) for i, l in ls.items()}
        s = sum(ex.values())
        return {i: e / s for i, e in ex.items()}

    # ── training: cycle the train specs, clipped policy gradient on MC returns ──
    ppo_epochs, batch_cap = 4, 256
    batch, played = [], 0
    order = list(range(len(train_specs)))
    rng.shuffle(order)
    cursor = 0
    while played < episodes:
        pool, base, valid = train_specs[order[cursor % len(train_specs)]]
        cursor += 1
        played += 1
        lo, span = min(base), (max(base) - min(base)) or 1.0
        base_rel = [(b - lo) / span for b in base]
        loads = [0] * len(pool)
        steps, rewards = [], []
        for t in range(d_steps):
            ok = [i for i in range(len(pool)) if valid[t][i] and loads[i] < cap]
            if not ok:
                steps.append(None)
                rewards.append(FAIL)
                continue
            feats, effs = slot_feats(pool, base_rel, valid[t], loads, ok)
            probs = masked_softmax(feats, ok)
            a = rng.choices(list(probs), weights=list(probs.values()))[0]
            steps.append((feats, ok, a, probs[a]))
            rewards.append(10.0 - 0.5 * 1 - 3.0 * effs[a])
            loads[a] += 1
        g, rets = 0.0, []
        for r in reversed(rewards):
            g = r + gamma * g
            rets.insert(0, g)
        for st, ret in zip(steps, rets):
            if st is not None:
                batch.append((*st, ret))
        if len(batch) >= batch_cap:
            all_rets = [b[4] for b in batch]
            mu = statistics.mean(all_rets)
            sd = statistics.pstdev(all_rets) or 1.0
            advs = [(r - mu) / sd for r in all_rets]
            for _ in range(ppo_epochs):
                grad = [0.0] * 6
                for (feats, ok, a, p_old, _), adv in zip(batch, advs):
                    p_new = masked_softmax(feats, ok)
                    ratio = p_new[a] / max(p_old, 1e-9)
                    if (adv >= 0 and ratio > 1 + clip) or (adv < 0 and ratio < 1 - clip):
                        continue
                    for k in range(6):
                        exp_k = sum(p_new[j] * feats[j][k] for j in ok)
                        grad[k] += ratio * adv * (feats[a][k] - exp_k)
                for k in range(6):
                    W[k] += lr * grad[k] / len(batch)
            batch = []

    # ── evaluation: every method replays the SAME episodes ──
    def replay(spec, method, mrng):
        pool, base, valid = spec
        lo, span = min(base), (max(base) - min(base)) or 1.0
        base_rel = [(b - lo) / span for b in base]
        loads = [0] * len(pool)
        total, risks, unserved = 0.0, [], 0
        for t in range(d_steps):
            ok = [i for i in range(len(pool)) if valid[t][i] and loads[i] < cap]
            if not ok:
                total += FAIL
                unserved += 1
                continue
            effs = [min(1.0, risk[pool[i]] + load_step * loads[i])
                    for i in range(len(pool))]
            if method == "pg":
                feats, _ = slot_feats(pool, base_rel, valid[t], loads, ok)
                probs = masked_softmax(feats, ok)
                a = max(probs, key=probs.get)
            elif method == "risk_greedy":
                a = min(ok, key=lambda i: effs[i])
            else:                       # dijkstra + random: hops all 1
                a = mrng.choice(ok)
            total += 10.0 - 0.5 * 1 - 3.0 * effs[a]
            risks.append(effs[a])
            loads[a] += 1
        return total, risks, unserved

    eval_rng = random.Random(seed + 1)
    methods = ("pg", "dijkstra", "risk_greedy", "random")
    totals = {m: [] for m in methods}
    risks_by = {m: [] for m in methods}
    unserved = {m: 0 for m in methods}
    for spec in test_specs:
        for m in methods:
            tot, rk, us = replay(spec, m, eval_rng)
            totals[m].append(tot)
            risks_by[m].extend(rk)
            unserved[m] += us
    avg_total = {m: round(statistics.mean(totals[m]), 3) for m in methods}
    avg_risk = {m: round(statistics.mean(risks_by[m]), 4) if risks_by[m] else None
                for m in methods}
    beats_g = 100.0 * statistics.mean(
        [1.0 if p > g else 0.0 for p, g in zip(totals["pg"], totals["risk_greedy"])])
    g_beats = 100.0 * statistics.mean(
        [1.0 if g > p else 0.0 for p, g in zip(totals["pg"], totals["risk_greedy"])])
    beats_d = 100.0 * statistics.mean(
        [1.0 if p > d else 0.0 for p, d in zip(totals["pg"], totals["dijkstra"])])
    if avg_total["pg"] > avg_total["risk_greedy"]:
        note = "PPO beats myopic risk-greedy — multi-step planning pays on this graph"
    elif avg_total["pg"] >= avg_total["random"]:
        note = ("greedy edges PPO: the load_step feedback makes effective-risk "
                "greedy implicitly load-balancing (a finding about the heuristic)")
    else:
        note = "PPO failed to clear the random baseline — treat as not learned"
    return {
        "n_train": len(train_specs), "n_test": len(test_specs),
        "cascade": f"{d_steps} demands, pool {pool_size}, cap {cap}, "
                   f"+{load_step}/use",
        "avg_total_reward": avg_total,
        "avg_route_risk": avg_risk,
        "unserved": unserved,
        "ppo_beats_greedy_pct": round(beats_g, 1),
        "greedy_beats_ppo_pct": round(g_beats, 1),
        "ppo_beats_dijkstra_pct": round(beats_d, 1),
        "note": note,
    }


STAGES = {
    "anomaly_detection":    anomaly_detection,
    "disruption_detection": disruption_detection,
    "safety_stock":         safety_stock,
    "event_replay":         event_replay,
    "value_at_risk":        value_at_risk,
}

# Topology stages take (flows, nodes, edges); the runner loads the extra
# tables lazily, only when a dataset's capabilities actually enable them.
GRAPH_STAGES = {
    "graph_risk_routing":   graph_risk_routing,
    "policy_gradient_routing": policy_gradient_routing,
}


# ── Gated runner ─────────────────────────────────────────────────────────────

def run(dataset: str) -> dict:
    caps  = load_capabilities(dataset)
    flows = load_flows(dataset)
    nodes = edges = None
    results = {}
    for name, fn in {**STAGES, **GRAPH_STAGES}.items():
        missing = [c for c in REQUIRES[name] if not caps.get(c)]
        if missing:
            results[name] = {"status": "skipped", "missing": missing}
        elif name in GRAPH_STAGES:
            if nodes is None:
                nodes = load_table(dataset, "nodes")
                edges = load_table(dataset, "edges")
            results[name] = {"status": "ran", "requires": REQUIRES[name],
                             **fn(flows, nodes, edges)}
        else:
            results[name] = {"status": "ran", "requires": REQUIRES[name], **fn(flows)}
    return results


if __name__ == "__main__":
    for ds in ("arcos", "scms", "dataco"):
        print("=" * 64)
        print(f"  GATED IR ANALYTICS — {ds}")
        print("=" * 64)
        try:
            res = run(ds)
        except FileNotFoundError as e:
            print(f"  [skip] {e}")
            continue
        for stage, out in res.items():
            if out["status"] == "skipped":
                print(f"  [SKIP] {stage:<20} needs {', '.join(out['missing'])}")
            else:
                detail = {k: v for k, v in out.items() if k not in ("status", "requires")}
                print(f"  [RAN ] {stage:<20} {detail}")
        print()
