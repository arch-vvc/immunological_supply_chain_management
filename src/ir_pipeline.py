"""
IR PIPELINE — the dataset-independent runner (Stage 26)
=======================================================
Ties the plug-and-play layer together end to end:

    for each dataset:
        adapter  raw file        -> canonical IR (output/ir/<ds>/)
        manifest which caps exist -> capability gate
        ir_stages run ONLY the stages the caps allow (read IR, never raw)

Adding a new dataset = write one adapter and add it to ADAPTERS below. No
analytics code changes. A dataset whose raw file is absent is skipped with a
note, not an error.

Pure stdlib. Run standalone or via the pipeline:
    python3 src/ir_pipeline.py
    python3 main.py --only 26
"""

from __future__ import annotations

import json
import os

import arcos_adapter
import scms_adapter
import dataco_adapter
import ir_stages

# name -> adapter module (each exposes build() -> IR). Order is cosmetic.
ADAPTERS = {
    "arcos":  arcos_adapter,
    "scms":   scms_adapter,
    "dataco": dataco_adapter,
}


def run():
    print("=" * 62)
    print("  DATASET-INDEPENDENT IR PIPELINE")
    print("=" * 62)
    print("  One IR shape, one set of gated stages, N datasets.\n")

    built = []
    for name, adapter in ADAPTERS.items():
        try:
            ir = adapter.build()
        except SystemExit as e:            # adapter's own 'missing input' guard
            print(f"  [skip] {name}: {e}")
            continue
        out = ir.write()
        caps = [c for c, on in ir.capabilities().items() if on]
        print(f"  [built] {name:<7} {len(ir.flows):>7,} flows -> {os.path.relpath(out)}")
        print(f"          capabilities: {', '.join(caps)}")
        built.append(name)
    print()

    # Run the gated analytics off each IR.
    for name in built:
        print("-" * 62)
        print(f"  GATED ANALYTICS — {name}")
        print("-" * 62)
        res = ir_stages.run(name)
        for stage, o in res.items():
            if o["status"] == "skipped":
                print(f"    [SKIP] {stage:<18} needs {', '.join(o['missing'])} — not in this dataset")
            else:
                detail = {k: v for k, v in o.items() if k not in ("status", "requires")}
                print(f"    [RAN ] {stage:<18} {detail}")
        # Persist so the dashboard's IR tab can render results without rerunning.
        with open(os.path.join(ir_stages.IR_DIR, name, "stage_results.json"), "w") as fh:
            json.dump(res, fh, indent=2)
        print()

    if not built:
        print("  No datasets available — check data/raw/.")
        return
    print("=" * 62)
    print(f"  DONE — {len(built)} dataset(s) through one engine: {', '.join(built)}")
    print("  Same stage code ran on each; capabilities gated what executed.")
    print("=" * 62)


if __name__ == "__main__":
    run()
