"""
Stage 11 — PPO Recovery Routing Agent (multi-step cascade environment)
======================================================================
Trains a single-agent PPO policy to route around a disrupted distributor
across a CASCADE of demands, under capacity exhaustion.

WHY MULTI-STEP: the earlier one-step task ("pick the lowest-risk valid
candidate") is analytically solved by the risk-greedy heuristic — reward is
a known function of observable features, so PPO could only ever tie it
(and did, on every eval). The redesigned environment gives learning
something greedy cannot do:

  * An episode = D_STEPS sequential reroute demands (retailers of the
    disrupted distributor) served from the SAME candidate pool.
  * Each use of a candidate raises its effective risk by LOAD_STEP
    (config decay_per_use — the immune analogy is clonal exhaustion)
    and consumes capacity (CAP uses max, then the candidate is masked).
  * Not every candidate can serve every retailer, so a myopic policy that
    burns the flexible low-risk generalist on early demands gets cornered
    later — a foresighted policy reserves flexibility.

State  : K candidate slots x F_DIM features
         [pool-rel risk, pool-rel gnn, valid_for_demand,
          remaining_capacity, pool-rel effective risk, out_degree_norm]
Action : discrete — which candidate serves the current demand
Reward : 10 - 0.5*hops - 3*effective_risk per step; -20 invalid pick

Evaluated against three baselines replaying the SAME demand sequences:
  Random-valid (sanity floor), Dijkstra (min hops among valid, random
  tie-break), and Myopic risk-greedy (lowest CURRENT effective risk among
  valid — the strong heuristic that was unbeatable in the one-step task).

PPO with clipped surrogate (ε=0.2), Monte-Carlo returns (γ=0.99),
entropy bonus, and action masking (invalid/exhausted slots blocked).
"""

import sys, os, random, pickle, shutil, tempfile, subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque

# ── Paths ──────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH_F = os.path.join(BASE, "models",  "supplychain_graph.pkl")
RISK_F  = os.path.join(BASE, "output",  "risk_scores.csv")
GNN_F   = os.path.join(BASE, "output",  "gnn_risk_scores.csv")
OUT_MDL = os.path.join(BASE, "models",  "ppo_routing_agent.pth")
OUT_TXT = os.path.join(BASE, "output",  "ppo_routing_results.txt")
OUT_FIG = os.path.join(BASE, "output",  "figures", "fig9_ppo_training.png")

# ── Hyperparameters ────────────────────────────────────────────────────────
K           = 10       # max candidate distributors per episode pool
F_DIM       = 6        # features per candidate slot
STATE_DIM   = K * F_DIM
ACTION_DIM  = K
CLIP_EPS    = 0.2
GAMMA       = 0.99
LR          = 1e-3
K_EPOCHS    = 4        # PPO update epochs per batch
BATCH_SIZE  = 256      # transitions per update (episodes are D_STEPS long)
TOTAL_EPS   = 12000    # episodes (x D_STEPS decisions each)
EVAL_EPS    = 300      # evaluation episodes (each replayed by every method)
MIN_SPREAD  = 0.05     # episodes need real risk contrast across the pool
D_STEPS     = 6        # reroute demands per episode (the cascade)
# Candidates per episode. The headline run uses 4 (4x2=8 capacity vs 6
# demands, so capacity binds). Because that choice determines whether a
# learned policy can matter at all, the script also SWEEPS pool sizes 4/6/8
# (retraining each) and reports where PPO's advantage disappears, instead of
# asserting the setting that favours it. ISC_PPO_POOL selects the size for a
# sweep child; ISC_PPO_SWEEP_CHILD=1 makes a run write only its sweep JSON.
POOL_SIZE   = int(os.environ.get("ISC_PPO_POOL", "4"))
SWEEP_CHILD = os.environ.get("ISC_PPO_SWEEP_CHILD") == "1"
SWEEP_POOLS = (4, 6, 8)
CAP         = 2        # max uses per candidate within an episode
LOAD_STEP   = 0.15     # effective-risk increase per use (config decay_per_use)
SEED        = 42

# Reproducibility — training and evaluation scenarios are seeded
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 55)
print("  STAGE 11 — PPO ROUTING AGENT")
print("=" * 55)

# ── Load graph + risk scores ───────────────────────────────────────────────
for path in [GRAPH_F, RISK_F, GNN_F]:
    if not os.path.exists(path):
        print(f"[ERROR] Missing: {path}. Run earlier stages first.")
        sys.exit(1)

with open(GRAPH_F, "rb") as fh:
    G = pickle.load(fh)

risk_df = pd.read_csv(RISK_F)
gnn_df  = pd.read_csv(GNN_F)

# risk_scores.csv  : entity, risk_score (already 0-1ish via composite)
risk_map_raw = dict(zip(risk_df["entity"], risk_df["risk_score"]))
rmax = max(risk_map_raw.values()) if risk_map_raw else 1.0
risk_map = {k: v / rmax for k, v in risk_map_raw.items()}

# gnn_risk_scores.csv : entity, gnn_score
gnn_map_raw = dict(zip(gnn_df["entity"], gnn_df["gnn_score"]))
gmax = max(gnn_map_raw.values()) if gnn_map_raw else 1.0
gnn_map = {k: v / gmax for k, v in gnn_map_raw.items()}

# Node sets and degree stats
manufacturers = {n for n, d in G.nodes(data=True) if d.get("type") == "manufacturer"}
distributors  = {n for n, d in G.nodes(data=True) if d.get("type") == "distributor"}
retailers     = {n for n, d in G.nodes(data=True) if d.get("type") == "retailer"}

in_deg  = dict(G.in_degree())
out_deg = dict(G.out_degree())
max_in  = max(in_deg.values())  or 1
max_out = max(out_deg.values()) or 1

print(f"  Graph : {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"  Manufacturers: {len(manufacturers)}  "
      f"Distributors: {len(distributors)}  Retailers: {len(retailers)}")

# Distributors that can anchor an episode (enough retailers to sample demands)
eligible_disrupted = [d for d in sorted(distributors)
                      if sum(1 for r in G.successors(d) if r in retailers) >= D_STEPS]

# ── Entity-level train / test split ─────────────────────────────────────────
# The policy trains on episodes anchored at TRAIN distributors and is
# evaluated ONLY on episodes anchored at held-out TEST distributors it never
# saw, so the reported numbers measure generalisation to unseen disruption
# sites rather than recall of training scenarios. (Earlier versions drew
# training and evaluation episodes from the same population.)
_split_rng = random.Random(SEED)
_anchors = eligible_disrupted[:]
_split_rng.shuffle(_anchors)
_n_test = max(1, int(round(0.3 * len(_anchors))))
TEST_ANCHORS  = sorted(_anchors[:_n_test])
TRAIN_ANCHORS = sorted(_anchors[_n_test:])
print(f"  Episode anchors: {len(TRAIN_ANCHORS)} train / {len(TEST_ANCHORS)} held-out test distributors "
      f"(pool size {POOL_SIZE})")

# ── Cascade Environment ────────────────────────────────────────────────────
# Distributor→retailer paths in this graph are direct edges only (there are
# no distributor→distributor links), so validity is G.has_edge and hops is
# always 1 — the interesting structure is WHICH candidates can serve which
# demands, under capacity, not path length.
FAIL_REWARD = -20.0     # a demand no usable candidate can serve

class CascadeEnv:
    """One episode = D_STEPS reroute demands against a fixed candidate pool
    with per-candidate capacity CAP and load-dependent effective risk.
    `anchors` = the disrupted distributors this env may sample episodes from
    (train or held-out test set)."""

    def __init__(self, anchors):
        self.anchors = list(anchors)

    def reset(self):
        """Sample an episode spec. Returns True if usable, False to resample.
        The pool is drawn from the union of the sampled demands' OWN suppliers
        (not one manufacturer's distributor list) — this is what makes validity
        sparse enough for capacity to bind, matching the IR cascade stage."""
        if not self.anchors:
            return False
        disrupted = random.choice(self.anchors)
        self.disrupted = disrupted            # remembered so evaluation can cluster by anchor
        dis_retailers = [r for r in G.successors(disrupted) if r in retailers]
        demands = random.sample(dis_retailers, D_STEPS)

        alternates = sorted({s for r in demands for s in G.predecessors(r)
                             if s in distributors} - {disrupted})
        if len(alternates) < POOL_SIZE:
            return False
        pool = random.sample(alternates, POOL_SIZE)

        base = [risk_map.get(c, 0.5) for c in pool]
        if max(base) - min(base) < MIN_SPREAD:      # no risk contrast
            return False
        if len(pool) * CAP < D_STEPS + 1:           # capacity must bind, not choke
            return False

        valid = [[G.has_edge(c, r) for c in pool] for r in demands]
        if any(not any(row) for row in valid):      # every demand serveable at t=0
            return False

        self.pool, self.demands, self.valid = pool, demands, valid
        self.base = base
        lo, span = min(base), (max(base) - min(base)) or 1.0
        self.base_rel = [(b - lo) / span for b in base]
        g = [gnn_map.get(c, 0.5) for c in pool]
        glo, gspan = min(g), (max(g) - min(g)) or 1.0
        self.gnn_rel = [(v - glo) / gspan for v in g]
        self.outd = [out_deg.get(c, 0) / max_out for c in pool]
        self.loads = [0] * len(pool)
        self.t = 0
        return True

    def spec(self):
        """Snapshot the episode definition so training can cycle a fixed set
        of specs (seeing each several times is what makes credit assignment
        tractable — the IR cascade stage trains the same way)."""
        return (self.pool, self.demands, self.valid, self.base,
                self.base_rel, self.gnn_rel, self.outd)

    def load_spec(self, spec):
        (self.pool, self.demands, self.valid, self.base,
         self.base_rel, self.gnn_rel, self.outd) = spec
        self.loads = [0] * len(self.pool)
        self.t = 0

    def eff_risk(self, i, loads=None):
        loads = self.loads if loads is None else loads
        return min(1.0, self.base[i] + LOAD_STEP * loads[i])

    def usable(self, loads=None, t=None):
        """Slots that are valid for the current demand AND have capacity."""
        loads = self.loads if loads is None else loads
        t = self.t if t is None else t
        return [i for i in range(len(self.pool))
                if self.valid[t][i] and loads[i] < CAP]

    def build_state(self, loads=None, t=None):
        """(state, mask) for an arbitrary load vector — lets evaluation replay
        the same episode under each method's own load history."""
        loads = self.loads if loads is None else loads
        t = self.t if t is None else t
        effs = [self.eff_risk(i, loads) for i in range(len(self.pool))]
        elo, espan = min(effs), (max(effs) - min(effs)) or 1.0
        ok = set(self.usable(loads, t))
        feats = []
        for i in range(len(self.pool)):
            feats.append([
                self.base_rel[i],
                self.gnn_rel[i],
                1.0 if i in ok else 0.0,
                (CAP - loads[i]) / CAP,
                (effs[i] - elo) / espan,
                self.outd[i],
            ])
        state = [v for f in feats for v in f]
        while len(state) < STATE_DIM:
            state.extend([0.0] * F_DIM)
        mask = [1.0 if i in ok else 0.0 for i in range(len(self.pool))]
        mask += [0.0] * (K - len(self.pool))
        return np.array(state, dtype=np.float32), mask

    def step(self, action):
        """Serve the current demand with pool[action]. Returns (reward, done)."""
        eff = self.eff_risk(action)
        reward = 10.0 - 0.5 * 1 - 3.0 * eff        # hops always 1 (direct edge)
        self.loads[action] += 1
        self.t += 1
        return reward, self.t >= D_STEPS

    def forced_fail(self):
        """No usable candidate for this demand — it goes unserved."""
        self.t += 1
        return FAIL_REWARD, self.t >= D_STEPS

# ── Actor & Critic networks ────────────────────────────────────────────────
# SLOT-EQUIVARIANT scorer: one shared network scores each candidate slot from
# its own features plus a masked pool-context vector. A dense MLP over the
# concatenated K*F state must LEARN that candidate order is irrelevant — with
# ~70K transitions it never does, which is why the stdlib IR cascade (whose
# linear per-slot policy has this symmetry built in) beat this file's old MLP.
def _masked_ctx(x, mask):
    """(B,K,F), (B,K) -> (B,F) mean over real (unmasked) slots."""
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return (x * mask.unsqueeze(-1)).sum(dim=1) / denom


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(F_DIM * 2, 64), nn.ReLU(),
            nn.Linear(64, 32),        nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state, mask=None):
        x = state.view(-1, K, F_DIM)                    # (B, K, F)
        m = mask if mask is not None else torch.ones(x.shape[0], K)
        ctx = _masked_ctx(x, m).unsqueeze(1).expand(-1, K, -1)
        logits = self.score(torch.cat([x, ctx], dim=-1)).squeeze(-1)   # (B, K)
        if mask is not None:
            logits = logits + (1.0 - m) * (-1e9)
        return F.softmax(logits, dim=-1)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(F_DIM, 64), nn.ReLU(),
            nn.Linear(64, 32),    nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state, mask=None):
        x = state.view(-1, K, F_DIM)
        m = mask if mask is not None else torch.ones(x.shape[0], K)
        return self.net(_masked_ctx(x, m)).squeeze(-1)


actor  = Actor()
critic = Critic()

# ── Warm start is OPT-IN (ISC_CONTINUAL=1). Default trains from scratch so
#    every number is reproducible from a fresh clone and the results manifest
#    means something; fine-tuning from a prior checkpoint makes the result
#    depend on how many times the stage was run before on that machine.
if os.environ.get("ISC_CONTINUAL") == "1" and os.path.exists(OUT_MDL):
    try:
        ckpt = torch.load(OUT_MDL, weights_only=True)
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        print("  [CONTINUAL] ISC_CONTINUAL=1 — loaded previous PPO weights, fine-tuning")
    except Exception:
        print("  [CONTINUAL] Previous weights incompatible — training from scratch")
else:
    print("  Training from scratch (seed 42). Set ISC_CONTINUAL=1 to fine-tune a prior checkpoint.")

opt_a  = torch.optim.Adam(actor.parameters(),  lr=LR)
opt_c  = torch.optim.Adam(critic.parameters(), lr=LR)

# ── PPO update step ────────────────────────────────────────────────────────
def ppo_update(states, actions, old_log_probs, returns, advantages, masks):
    st  = torch.FloatTensor(np.array(states))
    ac  = torch.LongTensor(actions)
    olp = torch.FloatTensor(old_log_probs)
    ret = torch.FloatTensor(returns)
    adv = torch.FloatTensor(advantages)
    msk = torch.FloatTensor(np.array(masks))

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    for _ in range(K_EPOCHS):
        probs    = actor(st, msk)
        dist     = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(ac)
        entropy  = dist.entropy().mean()

        ratio = torch.exp(log_prob - olp)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
        a_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy

        values = critic(st, msk)
        c_loss = F.mse_loss(values, ret)

        opt_a.zero_grad(); a_loss.backward(); opt_a.step()
        opt_c.zero_grad(); c_loss.backward(); opt_c.step()

# ── Training loop ──────────────────────────────────────────────────────────
env      = CascadeEnv(TRAIN_ANCHORS)   # training episodes
eval_env = CascadeEnv(TEST_ANCHORS)    # held-out evaluation episodes

# Pre-generate a fixed pool of episode specs and CYCLE it (each spec is seen
# ~TOTAL_EPS/SPEC_POOL_N times). Fresh-every-time scenarios gave the policy
# one shot per situation; cycling is how the IR cascade stage learns.
SPEC_POOL_N = 1500
print(f"\n  Generating {SPEC_POOL_N} training episode specs ...")
spec_pool = []
while len(spec_pool) < SPEC_POOL_N:
    if env.reset():
        spec_pool.append(env.spec())
spec_order = list(range(SPEC_POOL_N))
random.shuffle(spec_order)

print(f"  Training PPO ({TOTAL_EPS} episode plays x {D_STEPS} demands, "
      f"batch={BATCH_SIZE} transitions) ...")
reward_history = []                    # per-episode TOTAL reward, running avg
running_avg    = deque(maxlen=200)

buf_s, buf_a, buf_lp, buf_ret, buf_m = [], [], [], [], []

ep = 0
while ep < TOTAL_EPS:
    env.load_spec(spec_pool[spec_order[ep % SPEC_POOL_N]])

    # Collect one episode. Forced fails (no usable candidate) yield a reward
    # with NO stored action — but that reward still flows into the returns of
    # the earlier decisions that caused the exhaustion.
    ep_steps = []          # (stored?, state, mask, action, log_p)
    ep_rewards = []
    done = False
    while not done:
        usable = env.usable()
        if not usable:
            reward, done = env.forced_fail()
            ep_steps.append((False, None, None, None, None))
            ep_rewards.append(reward)
            continue
        state, mask = env.build_state()
        st_t  = torch.FloatTensor(state).unsqueeze(0)
        msk_t = torch.FloatTensor([mask])
        with torch.no_grad():
            probs  = actor(st_t, msk_t)
            dist   = torch.distributions.Categorical(probs)
            action = dist.sample().item()
            log_p  = dist.log_prob(torch.tensor(action)).item()
        reward, done = env.step(action)
        ep_steps.append((True, state, mask, action, log_p))
        ep_rewards.append(reward)

    # Discounted Monte-Carlo returns over the full reward sequence
    rets, g = [], 0.0
    for r in reversed(ep_rewards):
        g = r + GAMMA * g
        rets.insert(0, g)
    for (stored, state, mask, action, log_p), ret in zip(ep_steps, rets):
        if stored:
            buf_s.append(state)
            buf_a.append(action)
            buf_lp.append(log_p)
            buf_ret.append(ret)
            buf_m.append(mask)

    running_avg.append(float(np.sum(ep_rewards)))
    reward_history.append(float(np.mean(running_avg)))
    ep += 1

    if ep % 500 == 0:
        print(f"    Episode {ep:4d}/{TOTAL_EPS}  avg_total_reward={np.mean(running_avg):.2f}")

    # Update every BATCH_SIZE transitions
    if len(buf_s) >= BATCH_SIZE:
        with torch.no_grad():
            vals = critic(torch.FloatTensor(np.array(buf_s)),
                          torch.FloatTensor(np.array(buf_m))).numpy()
        returns    = np.array(buf_ret, dtype=np.float32)
        advantages = returns - vals

        ppo_update(buf_s, buf_a, buf_lp, returns, advantages, buf_m)
        buf_s, buf_a, buf_lp, buf_ret, buf_m = [], [], [], [], []

if not SWEEP_CHILD:
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, OUT_MDL)
print(f"\n  Model saved → {OUT_MDL}")

# ── Evaluation: PPO vs baselines on IDENTICAL episode replays ──────────────
# Every method faces the SAME episodes (same pool, same demand sequence) and
# evolves its OWN load state — so the comparison is the sequencing policy:
#   Random-valid — uniform among usable candidates (sanity floor)
#   Dijkstra     — min hops among usable; hops are all 1 here, so this is the
#                  risk-blind baseline with random tie-break
#   Risk-greedy  — lowest CURRENT effective risk among usable (the myopic
#                  heuristic that was unbeatable in the one-step task)
print(f"\n  Evaluating PPO vs baselines ({EVAL_EPS} episodes x {D_STEPS} demands) ...")

actor.eval()
METHODS  = ["PPO", "Random", "Dijkstra", "Risk-greedy"]

def replay(env, method):
    """Run one episode spec under a method's own load history."""
    loads = [0] * len(env.pool)
    total, risks, fails = 0.0, [], 0
    for t in range(D_STEPS):
        ok = env.usable(loads, t)
        if not ok:
            total += FAIL_REWARD
            fails += 1
            continue
        if method == "PPO":
            state, mask = env.build_state(loads, t)
            st_t  = torch.FloatTensor(state).unsqueeze(0)
            msk_t = torch.FloatTensor([mask])
            with torch.no_grad():
                probs = actor(st_t, msk_t)
            i = int(probs.argmax(dim=-1).item())
            if i not in ok:                     # mask guards this; belt & braces
                i = random.choice(ok)
        elif method == "Risk-greedy":
            i = min(ok, key=lambda j: env.eff_risk(j, loads))
        else:                                   # Random and Dijkstra (hops all 1)
            i = random.choice(ok)
        eff = env.eff_risk(i, loads)
        total += 10.0 - 0.5 * 1 - 3.0 * eff
        risks.append(eff)
        loads[i] += 1
    return total, risks, fails

total_by = {m: [] for m in METHODS}
risk_by  = {m: [] for m in METHODS}
fails_by = {m: 0  for m in METHODS}

eval_done = 0
anchor_by = []                       # which held-out distributor each episode came from
while eval_done < EVAL_EPS:
    if not eval_env.reset():
        continue
    anchor_by.append(eval_env.disrupted)
    for m in METHODS:
        total, risks, fails = replay(eval_env, m)
        total_by[m].append(total)
        risk_by[m].extend(risks)
        fails_by[m] += fails
    eval_done += 1

avg_total = {m: float(np.mean(total_by[m])) for m in METHODS}
avg_risk  = {m: float(np.mean(risk_by[m]))  for m in METHODS}
ppo_beats_greedy_pct = 100.0 * float(np.mean(
    [p > g for p, g in zip(total_by["PPO"], total_by["Risk-greedy"])]))
greedy_beats_ppo_pct = 100.0 * float(np.mean(
    [g > p for p, g in zip(total_by["PPO"], total_by["Risk-greedy"])]))
ppo_safer_pct = 100.0 * float(np.mean(
    [p > d for p, d in zip(total_by["PPO"], total_by["Dijkstra"])]))

# ── Spread + paired uncertainty over the evaluation episodes ───────────────
# Every method replays the SAME episodes, so method differences are paired:
# bootstrap the per-episode reward difference, not two independent means.
std_total = {m: float(np.std(total_by[m], ddof=1)) for m in METHODS}
sem_total = {m: std_total[m] / float(np.sqrt(len(total_by[m]))) for m in METHODS}
std_risk  = {m: float(np.std(risk_by[m], ddof=1)) if len(risk_by[m]) > 1 else 0.0
             for m in METHODS}

def _paired_bootstrap(a, b, n_boot=5000, seed=SEED):
    """Mean of (a - b) with a 95% percentile bootstrap CI over episodes."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

delta_vs_greedy   = _paired_bootstrap(total_by["PPO"], total_by["Risk-greedy"])
delta_vs_dijkstra = _paired_bootstrap(total_by["PPO"], total_by["Dijkstra"])


def _cluster_bootstrap(a, b, clusters, n_boot=5000, seed=SEED):
    """Mean of (a - b) with a 95% CI from resampling ANCHORS with replacement.

    Every evaluation episode is anchored at one of the held-out distributors,
    and episodes from the same anchor share its retailer set and candidate
    pools — they are not independent draws. Resampling episodes (above)
    therefore understates uncertainty; resampling the anchors themselves is
    the honest interval, and with few anchors it is wide by construction.
    Returns (mean, lo, hi, {anchor: (mean_delta, n_episodes)})."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    cl = np.asarray(clusters)
    uniq = sorted(set(cl.tolist()))
    groups = [d[cl == u] for u in uniq]
    means = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        means[i] = np.concatenate([groups[j] for j in pick]).mean()
    per_anchor = {u: (float(g.mean()), int(len(g))) for u, g in zip(uniq, groups)}
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), per_anchor


cl_vs_greedy   = _cluster_bootstrap(total_by["PPO"], total_by["Risk-greedy"], anchor_by)
cl_vs_dijkstra = _cluster_bootstrap(total_by["PPO"], total_by["Dijkstra"],    anchor_by)
n_anchors_eval = len(set(anchor_by))

import json as _json
_sweep_record = {
    "pool_size": POOL_SIZE, "capacity_total": POOL_SIZE * CAP, "demands": D_STEPS,
    "capacity_binds": POOL_SIZE * CAP < 2 * D_STEPS,
    "avg_total_reward": avg_total, "unserved": {m: int(fails_by[m]) for m in METHODS},
    "ppo_minus_greedy": {"mean": delta_vs_greedy[0], "ci95": [delta_vs_greedy[1], delta_vs_greedy[2]]},
    "ppo_minus_dijkstra": {"mean": delta_vs_dijkstra[0], "ci95": [delta_vs_dijkstra[1], delta_vs_dijkstra[2]]},
    "ppo_minus_greedy_cluster":   {"ci95": [cl_vs_greedy[1], cl_vs_greedy[2]]},
    "ppo_minus_dijkstra_cluster": {"ci95": [cl_vs_dijkstra[1], cl_vs_dijkstra[2]]},
    "n_eval_anchors": n_anchors_eval,
    "ppo_safer_than_dijkstra_pct": ppo_safer_pct,
}
if SWEEP_CHILD:
    _sp = os.path.join(BASE, "output", f"ppo_sweep_pool{POOL_SIZE}.json")
    with open(_sp, "w") as _f:
        _json.dump(_sweep_record, _f, indent=2)
    print(f"  [sweep child] pool={POOL_SIZE}: PPO {avg_total['PPO']:.2f} vs greedy "
          f"{avg_total['Risk-greedy']:.2f}, unserved {fails_by['PPO']}/{fails_by['Risk-greedy']} -> {_sp}")
    sys.exit(0)

# ── Pool-size sweep: retrain at each size in a child process ───────────────
_sweep = {POOL_SIZE: _sweep_record}
for _k in SWEEP_POOLS:
    if _k == POOL_SIZE:
        continue
    _env = dict(os.environ, ISC_PPO_POOL=str(_k), ISC_PPO_SWEEP_CHILD="1")
    print(f"\n  Sweep: retraining with pool size {_k} ...")
    _r = subprocess.run([sys.executable, os.path.abspath(__file__)], env=_env,
                        cwd=BASE, capture_output=True, text=True)
    _sp = os.path.join(BASE, "output", f"ppo_sweep_pool{_k}.json")
    if _r.returncode == 0 and os.path.exists(_sp):
        with open(_sp) as _f:
            _sweep[_k] = _json.load(_f)
        os.remove(_sp)
    else:
        print(f"  [WARN] sweep child for pool {_k} failed:\n{_r.stdout[-600:]}{_r.stderr[-600:]}")
SWEEP_OUT = os.path.join(BASE, "output", "ppo_pool_sweep.json")
with open(SWEEP_OUT, "w") as _f:
    _json.dump({str(k): v for k, v in sorted(_sweep.items())}, _f, indent=2)
print(f"  Pool sweep saved → {SWEEP_OUT}")

def _sweep_lines():
    out = ["Pool-size sweep (each size retrained from scratch; same seed, same held-out anchors):",
           f"  {'pool':>4} {'capacity':>9} {'binds?':>7} {'PPO':>7} {'greedy':>7} {'dijkstra':>9} "
           f"{'unserved P/G/D':>15}  {'PPO−greedy [episode CI]':>26}  {'[anchor-cluster CI]':>20}"]
    for k in sorted(_sweep):
        r = _sweep[k]; a = r["avg_total_reward"]; u = r["unserved"]; d = r["ppo_minus_greedy"]
        c = r.get("ppo_minus_greedy_cluster", {}).get("ci95", [float("nan"), float("nan")])
        out.append(f"  {k:>4} {r['capacity_total']:>9} {'yes' if r['capacity_binds'] else 'no':>7} "
                   f"{a['PPO']:>7.2f} {a['Risk-greedy']:>7.2f} {a['Dijkstra']:>9.2f} "
                   f"{u['PPO']:>4}/{u['Risk-greedy']:>3}/{u['Dijkstra']:>3}      "
                   f"{d['mean']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]   "
                   f"[{c[0]:+.2f}, {c[1]:+.2f}]")
    wins = [k for k in sorted(_sweep) if _sweep[k].get("ppo_minus_greedy_cluster", _sweep[k]["ppo_minus_greedy"])["ci95"][0] > 0]
    ties = [k for k in sorted(_sweep) if k not in wins]
    if wins and ties:
        out.append(f"  Reading: PPO's edge over the myopic heuristic is significant only at pool "
                   f"size(s) {wins}, where capacity binds; at {ties} the CI includes zero — the "
                   "advantage is a property of the capacity-constrained regime, not of routing in general.")
    elif wins:
        out.append(f"  Reading: PPO beats the myopic heuristic at every pool size tested {wins}.")
    else:
        out.append(f"  Reading: PPO does not significantly beat the myopic heuristic at any pool size "
                   f"tested {ties}; the learned policy adds nothing the load-aware heuristic lacks here.")
    return out

header = (f"  {'Method':<14} {'Avg Total Reward':>17} {'SD':>6}  "
          f"{'Avg Eff. Risk':>13}  {'Unserved':>9}")
rows   = [f"  {m:<14} {avg_total[m]:>17.2f} {std_total[m]:>6.2f}  "
          f"{avg_risk[m]:>13.3f}  {fails_by[m]:>9}"
          for m in METHODS]

print(f"\n  ── Evaluation ({eval_done} episodes, all methods replay identical demands) ──")
print(header)
print("  " + "-" * 60)
for r in rows:
    print(r)
print(f"\n  PPO beats myopic Risk-greedy on total reward: "
      f"{ppo_beats_greedy_pct:.1f}% of episodes (greedy wins {greedy_beats_ppo_pct:.1f}%)")
print(f"  PPO strictly safer than Dijkstra: {ppo_safer_pct:.1f}% of episodes")

results_text = "\n".join([
    "PPO Routing Agent — Evaluation Report (multi-step cascade)",
    "===========================================================",
    f"Evaluated on {eval_done} HELD-OUT episodes of {D_STEPS} sequential reroute demands",
    f"(anchored at {n_anchors_eval} of the {len(TEST_ANCHORS)} held-out test distributors the policy never",
    f"trained on — the rest never yielded a usable pool under the episode filters;",
    f"trained on {len(TRAIN_ANCHORS)} others), served from one candidate pool of",
    "{} under capacity (CAP={} uses) and".format(POOL_SIZE, CAP),
    f"load-dependent effective risk (+{LOAD_STEP}/use — config decay_per_use).",
    "All methods replay the SAME episodes (same pool, same demand order)",
    "and evolve their own load state; pools require risk spread >= 0.05.",
    "",
    header.strip(),
    "-" * 60,
    *[r.strip() for r in rows],
    "",
    f"PPO beats myopic Risk-greedy      : {ppo_beats_greedy_pct:.1f}% of episodes "
    f"(greedy wins {greedy_beats_ppo_pct:.1f}%)",
    f"PPO strictly safer than Dijkstra : {ppo_safer_pct:.1f}% of episodes",
    "",
    f"Paired bootstrap 95% CI on total reward (5,000 resamples over {eval_done} episodes):",
    f"  PPO − Risk-greedy : {delta_vs_greedy[0]:+.2f}  [{delta_vs_greedy[1]:+.2f}, {delta_vs_greedy[2]:+.2f}]",
    f"  PPO − Dijkstra    : {delta_vs_dijkstra[0]:+.2f}  [{delta_vs_dijkstra[1]:+.2f}, {delta_vs_dijkstra[2]:+.2f}]",
    f"  (SD column above is across episodes; SEM for PPO = {sem_total['PPO']:.2f})",
    "",
    f"Cluster bootstrap 95% CI — resampling the {n_anchors_eval} held-out ANCHOR distributors that produced episodes, not episodes:",
    f"  PPO − Risk-greedy : {cl_vs_greedy[0]:+.2f}  [{cl_vs_greedy[1]:+.2f}, {cl_vs_greedy[2]:+.2f}]",
    f"  PPO − Dijkstra    : {cl_vs_dijkstra[0]:+.2f}  [{cl_vs_dijkstra[1]:+.2f}, {cl_vs_dijkstra[2]:+.2f}]",
    "  Episodes from the same anchor share its retailers and pools, so the",
    "  episode-level CI above overstates independence. The cluster CI is the",
    f"  honest one; with only {n_anchors_eval} clusters it is wide by construction.",
    "",
    "Per-anchor mean reward delta (held-out distributors):",
    f"  {'anchor':<40} {'episodes':>8} {'PPO−greedy':>11} {'PPO−dijkstra':>13}",
    *[f"  {str(a)[:39]:<40} {cl_vs_greedy[3][a][1]:>8} {cl_vs_greedy[3][a][0]:>+11.2f} {cl_vs_dijkstra[3][a][0]:>+13.2f}"
      for a in sorted(cl_vs_greedy[3])],
    "",
    "Baselines:",
    "  Random       — uniform among usable candidates (sanity floor).",
    "  Dijkstra     — min hops among usable; hops are all 1 on this graph, so",
    "                 this is the risk-blind baseline with random tie-break.",
    "  Risk-greedy  — lowest CURRENT effective risk among usable. This is the",
    "                 myopic heuristic that provably ties PPO in the one-step",
    "                 task; here it can exhaust the flexible low-risk candidate",
    "                 early and get cornered by later demands.",
    "",
    "How to read this:",
    "  'Unserved' counts demands no usable candidate could serve (-20 each).",
    "  The paired CI on PPO − Risk-greedy is the test: an interval above zero",
    "  means the learned policy plans across steps better than the load-aware",
    "  heuristic; an interval containing zero means it does not. The pool-size",
    "  sweep below shows in which regime that holds, so the headline setting",
    "  is not the only one reported.",
    "",
    *_sweep_lines(),
])
try:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
        tmp.write(results_text)
        _tmp = tmp.name
    shutil.copy2(_tmp, OUT_TXT)
    os.unlink(_tmp)
    print(f"  Results saved → {OUT_TXT}")
except Exception as _e:
    print(f"  [WARN] Could not save PPO results file: {_e}")

# Machine-readable sidecar for the results manifest / dashboard
import json as _json
OUT_STATS = os.path.splitext(OUT_TXT)[0].replace("_results", "_stats") + ".json"
_stats = {
    "n_eval_episodes": int(eval_done),
    "demands_per_episode": int(D_STEPS),
    "training_episodes": int(TOTAL_EPS),
    "seed": int(SEED),
    "avg_total_reward": avg_total,
    "sd_total_reward": std_total,
    "sem_total_reward": sem_total,
    "avg_eff_risk": avg_risk,
    "sd_eff_risk": std_risk,
    "unserved": {m: int(fails_by[m]) for m in METHODS},
    "ppo_beats_greedy_pct": ppo_beats_greedy_pct,
    "greedy_beats_ppo_pct": greedy_beats_ppo_pct,
    "ppo_safer_than_dijkstra_pct": ppo_safer_pct,
    "paired_delta_reward": {
        "ppo_minus_greedy":   {"mean": delta_vs_greedy[0],   "ci95": [delta_vs_greedy[1],   delta_vs_greedy[2]]},
        "ppo_minus_dijkstra": {"mean": delta_vs_dijkstra[0], "ci95": [delta_vs_dijkstra[1], delta_vs_dijkstra[2]]},
    },
    "paired_delta_reward_cluster": {
        "method": "anchor (held-out distributor) cluster bootstrap, 5000 resamples",
        "n_anchors": n_anchors_eval,
        "ppo_minus_greedy":   {"ci95": [cl_vs_greedy[1],   cl_vs_greedy[2]]},
        "ppo_minus_dijkstra": {"ci95": [cl_vs_dijkstra[1], cl_vs_dijkstra[2]]},
        "per_anchor": {str(a): {"n_episodes": cl_vs_greedy[3][a][1],
                                "ppo_minus_greedy": cl_vs_greedy[3][a][0],
                                "ppo_minus_dijkstra": cl_vs_dijkstra[3][a][0]} for a in sorted(cl_vs_greedy[3])},
    },
}
with open(OUT_STATS, "w") as _f:
    _json.dump(_stats, _f, indent=2)
print(f"  Stats saved   → {OUT_STATS}")

# ── Figure ─────────────────────────────────────────────────────────────────
BG   = "#0f0f1a"
BLUE = "#4fc3f7"
RED  = "#ff6b6b"
GREY = "#aaaaaa"

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(BG)
fig.suptitle("PPO Routing Agent — Stage 11",
             color="white", fontsize=14, y=1.02)

for ax in axes:
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333355")
    ax.tick_params(colors=GREY)

# Panel 1: Training reward curve
ax1 = axes[0]
ax1.plot(reward_history, color=BLUE, lw=1.0, alpha=0.8, label="Running avg reward")
ax1.axhline(0, color="#555", ls="--", lw=0.8)
# Rolling mean for smoother trend line
if len(reward_history) > 50:
    smooth = np.convolve(reward_history,
                         np.ones(50) / 50, mode="valid")
    ax1.plot(range(49, len(reward_history)), smooth,
             color="#ff6b6b", lw=1.8, label="Smoothed (50-ep)")
ax1.set_title("PPO Training Reward", color="white", fontsize=11, pad=8)
ax1.set_xlabel("Episode", color=GREY)
ax1.set_ylabel("Avg Reward (200-ep window)", color=GREY)
ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)

# Panel 2: avg route risk per method (hops annotated above each bar)
ax2     = axes[1]
colors  = {"PPO": BLUE, "Random": GREY, "Dijkstra": RED, "Risk-greedy": "#44dd88"}
x       = np.arange(len(METHODS))
bars    = ax2.bar(x, [avg_risk[m] for m in METHODS],
                  color=[colors[m] for m in METHODS], alpha=0.85, width=0.6)

for bar, m in zip(bars, METHODS):
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
             f"risk {avg_risk[m]:.3f}\nreward {avg_total[m]:.1f}",
             ha="center", va="bottom", color="white", fontsize=8.5)

ax2.set_xticks(x)
ax2.set_xticklabels(METHODS, color=GREY, fontsize=9)
ax2.set_ylim(0, max(avg_risk.values()) * 1.25)
ax2.set_ylabel("Avg Effective Risk of Served Demands (0–1)", color=GREY)
ax2.set_title(f"Cascade Quality vs Baselines ({eval_done} identical episodes)",
              color="white", fontsize=11, pad=8)

plt.tight_layout(pad=2.5)
os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
plt.savefig(OUT_FIG, dpi=130, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"  Figure saved → {OUT_FIG}")

print("\n  Stage 11 complete.")
print("  PPO evaluated against Random, fair Dijkstra, and Risk-greedy baselines")
print("  on identical scenarios — see ppo_routing_results.txt for the table.")
