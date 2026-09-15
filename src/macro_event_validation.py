"""
Stage 27 — Macro-Stress Crisis Validation (documented real events)
==================================================================
External validity check for the innate-immunity layer: does the macro
freight stress score (Stage 9) actually rise when documented real-world
supply chain crises hit — without ever being told about them?

METHOD (event study with placebo inference)
-------------------------------------------
For each documented crisis with a public start date inside the indicator
coverage (2017-2026):
  * event window : start .. start + 8 weeks   (the composite is built from
    monthly indicators carried forward weekly with a 4-week rolling mean, so the
    response is lagged and smeared — 8 weeks absorbs that)
  * pre window   : the 12 weeks before the start
  * delta        : mean(event window) - mean(pre window)

Significance is judged against a PLACEBO distribution: the same delta
computed at every eligible non-event week in the series. An event's
percentile within that null distribution says how unusual its rise is.
  DETECTED >= p90     PARTIAL >= p75     NOT SEEN otherwise

A null result is reported as a finding, not hidden: the composite is built
from five US-aggregate indicators, so crises whose freight impact bypassed
US aggregates (e.g. a 6-day canal blockage absorbed by schedule slack)
SHOULD look muted here. Which crises a US-macro lens sees is itself the
result.

Outputs:
    output/macro_event_validation.csv
    output/macro_event_validation_report.txt
    output/figures/fig11_macro_event_validation.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV  = os.path.join(BASE, "output", "macro_stress_scores.csv")
OUT_CSV = os.path.join(BASE, "output", "macro_event_validation.csv")
OUT_TXT = os.path.join(BASE, "output", "macro_event_validation_report.txt")
OUT_FIG = os.path.join(BASE, "output", "figures", "fig11_macro_event_validation.png")

EVENT_WEEKS = 8      # event window length
PRE_WEEKS   = 12     # baseline window length
EXCLUDE_W   = 20     # placebo starts this close to a real event are excluded

# Documented crises with public start dates (all inside 2017-2026 coverage).
EVENTS = [
    ("COVID-19 pandemic onset",      "2020-03-01",
     "WHO declaration 2020-03-11; US freight shock Feb-Apr 2020"),
    ("Suez Canal blockage",          "2021-03-23",
     "Ever Given aground 2021-03-23 to 03-29; ~$9.6B/day trade held"),
    ("US port congestion peak",      "2021-09-01",
     "record LA/Long Beach anchorage queues Sep-Oct 2021"),
    ("Russia invades Ukraine",       "2022-02-24",
     "energy/diesel and freight cost shock from 2022-02-24"),
    ("Red Sea shipping attacks",     "2023-12-15",
     "Houthi attacks; carriers reroute around the Cape from Dec 2023"),
]


def window_delta(series, dates, start, event_weeks, pre_weeks):
    """mean(event window) - mean(pre window); None if either window is empty."""
    ev = series[(dates >= start) & (dates < start + pd.Timedelta(weeks=event_weeks))]
    pre = series[(dates >= start - pd.Timedelta(weeks=pre_weeks)) & (dates < start)]
    if len(ev) == 0 or len(pre) == 0:
        return None, None, None
    return float(ev.mean() - pre.mean()), float(ev.max()), float(pre.mean())


def main():
    print("=" * 55)
    print("  STAGE 27 — MACRO-STRESS CRISIS VALIDATION")
    print("=" * 55)

    if not os.path.exists(IN_CSV):
        print(f"[ERROR] {IN_CSV} not found. Run Stage 9 first.")
        raise SystemExit(1)

    df = pd.read_csv(IN_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    comp_cols = [c for c in df.columns if c not in ("date", "stress_score", "stress_level")]
    dates, stress = df["date"], df["stress_score"]
    print(f"  Stress series: {len(df)} weeks  {dates.min().date()} -> {dates.max().date()}")

    # ── placebo null distribution ──────────────────────────────────────────
    event_starts = [pd.Timestamp(d) for _, d, _ in EVENTS]
    placebo = []
    for start in dates:
        if any(abs((start - e).days) < EXCLUDE_W * 7 for e in event_starts):
            continue
        d, _, _ = window_delta(stress, dates, start, EVENT_WEEKS, PRE_WEEKS)
        if d is not None:
            placebo.append(d)
    placebo = np.array(placebo)
    p90, p75 = np.percentile(placebo, 90), np.percentile(placebo, 75)
    print(f"  Placebo windows: {len(placebo)}  (delta p75={p75:+.4f}  p90={p90:+.4f})")

    # ── event studies ──────────────────────────────────────────────────────
    rows = []
    for name, start_s, note in EVENTS:
        start = pd.Timestamp(start_s)
        if start < dates.min() or start > dates.max():
            rows.append({"event": name, "start": start_s, "verdict": "OUT OF RANGE"})
            continue
        delta, ev_peak, pre_mean = window_delta(stress, dates, start, EVENT_WEEKS, PRE_WEEKS)
        pct = float((placebo < delta).mean() * 100)
        peak_pct = float((stress < ev_peak).mean() * 100)
        # which component moved most
        comp_deltas = {}
        for c in comp_cols:
            cd, _, _ = window_delta(df[c], dates, start, EVENT_WEEKS, PRE_WEEKS)
            comp_deltas[c] = cd if cd is not None else 0.0
        driver = max(comp_deltas, key=comp_deltas.get)
        verdict = ("DETECTED" if pct >= 90 else
                   "PARTIAL" if pct >= 75 else "NOT SEEN")
        rows.append({
            "event": name, "start": start_s,
            "pre_mean": round(pre_mean, 4), "window_peak": round(ev_peak, 4),
            "delta": round(delta, 4), "placebo_percentile": round(pct, 1),
            "peak_percentile_all_history": round(peak_pct, 1),
            "top_driver": driver, "driver_delta": round(comp_deltas[driver], 4),
            "verdict": verdict, "note": note,
        })

    res = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    res.to_csv(OUT_CSV, index=False)

    detected = [r for r in rows if r.get("verdict") == "DETECTED"]
    partial  = [r for r in rows if r.get("verdict") == "PARTIAL"]

    print(f"\n  {'Event':<28} {'Δ stress':>9} {'placebo %':>10} {'peak %ile':>10}  verdict")
    print("  " + "-" * 72)
    for r in rows:
        if r["verdict"] == "OUT OF RANGE":
            print(f"  {r['event']:<28} {'—':>9} {'—':>10} {'—':>10}  OUT OF RANGE")
        else:
            print(f"  {r['event']:<28} {r['delta']:>+9.4f} {r['placebo_percentile']:>9.1f}% "
                  f"{r['peak_percentile_all_history']:>9.1f}%  {r['verdict']}")

    # ── report ─────────────────────────────────────────────────────────────
    lines = [
        "STAGE 27 — MACRO-STRESS CRISIS VALIDATION (documented real events)",
        "=" * 68, "",
        "Question: does the Stage-9 stress composite rise at documented",
        "real-world crises it was never told about? Event-study deltas",
        f"(mean of {EVENT_WEEKS}-week event window minus mean of the prior",
        f"{PRE_WEEKS} weeks) are ranked against a placebo distribution of",
        f"{len(placebo)} non-event windows. DETECTED >= p90, PARTIAL >= p75.",
        "",
        f"{'Event':<28} {'pre':>6} {'peak':>6} {'delta':>8} {'plc%':>6} {'verdict':>9}  top driver",
        "-" * 92,
    ]
    for r in rows:
        if r["verdict"] == "OUT OF RANGE":
            lines.append(f"{r['event']:<28} {'—':>6} {'—':>6} {'—':>8} {'—':>6} {'OUT OF RANGE':>9}")
        else:
            lines.append(
                f"{r['event']:<28} {r['pre_mean']:>6.3f} {r['window_peak']:>6.3f} "
                f"{r['delta']:>+8.4f} {r['placebo_percentile']:>5.1f}% {r['verdict']:>9}  "
                f"{r['top_driver']} ({r['driver_delta']:+.3f})")
    lines += [
        "",
        f"Summary: {len(detected)}/{len(rows)} DETECTED, {len(partial)} PARTIAL.",
        "",
        *_reading(rows),
        "",
        "Caveat: with 5 events this is external validity evidence, not a",
        "powered statistical test. Placebo percentiles quantify how unusual",
        "each rise is within THIS series; they are not p-values from an",
        "independent sample.",
    ]
    with open(OUT_TXT, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Report saved -> {OUT_TXT}")


# What each crisis should do to a US-aggregate freight composite, used to
# phrase the reading of whichever verdicts the data actually produced.
_MECHANISM = {
    "COVID-19 pandemic onset": "US freight demand collapsed in Mar-Apr 2020: diesel and truck spot rates FELL while inventory-to-sales spiked, so a US freight-cost composite can read the onset as LOWER stress",
    "Suez Canal blockage": "a 6-day foreign chokepoint that global schedules largely absorbed; a US signal here is container/spot-rate spillover",
    "US port congestion peak": "a domestic port event; containerships-at-anchor is the direct indicator, but that series only enters the causal composite from mid-2022",
    "Russia invades Ukraine": "an energy-price shock atop already-elevated 2021-22 congestion, so the rise sits on a high base",
    "Red Sea shipping attacks": "an Asia-Europe rerouting crisis whose cost landed mostly outside US lanes",
}


def _reading(rows):
    """Verdict-driven interpretation: says what was and was not seen, why the
    mechanism makes that plausible, and never claims a null was expected only
    after seeing it was a null."""
    det = [r for r in rows if r["verdict"] == "DETECTED"]
    par = [r for r in rows if r["verdict"] == "PARTIAL"]
    nul = [r for r in rows if r["verdict"] not in ("DETECTED", "PARTIAL")]
    out = ["Reading the verdicts: the composite is five US-AGGREGATE indicators",
           "(diesel, truck spot rates, containerships at anchor, TSI,",
           "inventory-to-sales), each scored CAUSALLY as a percentile rank of its",
           "own history to that week (no full-series scaling), carried forward",
           "weekly and 4-week smoothed. Every window is compared with placebo",
           "windows from the same series."]
    for r in det:
        out.append(f"  DETECTED  {r['event']}: +{r['delta']:.3f} (placebo p{r['placebo_percentile']:.0f}), "
                   f"top driver {r['top_driver']} — {_MECHANISM.get(r['event'], 'mechanism not annotated')}.")
    for r in par:
        out.append(f"  PARTIAL   {r['event']}: +{r['delta']:.3f} (placebo p{r['placebo_percentile']:.0f}) — "
                   f"{_MECHANISM.get(r['event'], 'mechanism not annotated')}.")
    for r in nul:
        out.append(f"  NOT SEEN  {r['event']}: {r['delta']:+.3f} (placebo p{r['placebo_percentile']:.0f}) — "
                   f"{_MECHANISM.get(r['event'], 'mechanism not annotated')}.")
    out.append("A US-macro lens sees what moves US freight prices and capacity; it is")
    out.append("not a global crisis detector, and the nulls above are reported as such.")
    return out
    print(f"  CSV saved    -> {OUT_CSV}")

    # ── figure ─────────────────────────────────────────────────────────────
    BG, BLUE, RED, ORANGE, GREY = "#0f0f1a", "#4fc3f7", "#ff6b6b", "#ffaa44", "#aaaaaa"
    fig, axes = plt.subplots(2, 1, figsize=(15, 10),
                             gridspec_kw={"height_ratios": [3, 2]})
    fig.patch.set_facecolor(BG)
    fig.suptitle("Macro Stress vs Documented Crises — Stage 27",
                 color="white", fontsize=14, y=0.98)
    for ax in axes:
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333355")
        ax.tick_params(colors=GREY)

    ax1 = axes[0]
    ax1.plot(dates, stress, color=BLUE, lw=1.2, label="Weekly stress composite")
    ax1.axhline(0.65, color="red", ls="--", lw=0.7, alpha=0.5)
    ax1.axhline(0.40, color="orange", ls="--", lw=0.7, alpha=0.5)
    for r in rows:
        if r["verdict"] == "OUT OF RANGE":
            continue
        s = pd.Timestamp(r["start"])
        col = {"DETECTED": RED, "PARTIAL": ORANGE, "NOT SEEN": GREY}[r["verdict"]]
        ax1.axvspan(s, s + pd.Timedelta(weeks=EVENT_WEEKS), alpha=0.18, color=col)
        ax1.annotate(f"{r['event']}\n[{r['verdict']}]",
                     xy=(s, r["window_peak"]), xytext=(0, 22),
                     textcoords="offset points", ha="center",
                     fontsize=7.5, color=col,
                     arrowprops=dict(arrowstyle="-", color=col, lw=0.8))
    ax1.set_ylabel("Stress score (0-1)", color=GREY)
    ax1.set_ylim(0, 0.75)
    ax1.set_title("Stress timeline with 8-week event windows", color="white",
                  fontsize=11, pad=8)
    ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8.5)

    ax2 = axes[1]
    ev_names = [r["event"] for r in rows if r["verdict"] != "OUT OF RANGE"]
    ev_delta = [r["delta"] for r in rows if r["verdict"] != "OUT OF RANGE"]
    cols = [{"DETECTED": RED, "PARTIAL": ORANGE, "NOT SEEN": GREY}[r["verdict"]]
            for r in rows if r["verdict"] != "OUT OF RANGE"]
    x = np.arange(len(ev_names))
    ax2.bar(x, ev_delta, color=cols, alpha=0.9, width=0.55)
    ax2.axhline(p90, color=RED, ls="--", lw=1.0,
                label=f"placebo p90 ({p90:+.3f})")
    ax2.axhline(p75, color=ORANGE, ls="--", lw=1.0,
                label=f"placebo p75 ({p75:+.3f})")
    ax2.axhline(0, color="#555", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([n.replace(" ", "\n", 1) for n in ev_names],
                        color=GREY, fontsize=8)
    ax2.set_ylabel("Δ stress vs prior 12 weeks", color=GREY)
    ax2.set_title("Event-window rise vs placebo distribution", color="white",
                  fontsize=11, pad=8)
    ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8.5)

    plt.tight_layout(pad=2.5)
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    plt.savefig(OUT_FIG, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Figure saved -> {OUT_FIG}")
    print("\n  Stage 27 complete.")


if __name__ == "__main__":
    main()
