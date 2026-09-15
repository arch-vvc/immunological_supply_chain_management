"""End-to-end smoke test: run Stages 1-6 in a throwaway copy of the repo and
assert every expected artifact appears.

Runs in an isolated temp directory so the committed output/ is never touched.
Skipped unless ISC_SMOKE=1 is set, because it takes a couple of minutes:

    ISC_SMOKE=1 pytest tests/test_pipeline_smoke.py -s
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import main as runner  # repo root is on sys.path via pyproject [tool.pytest] / tests/run_all.py

SMOKE_STAGES = [1, 2, 3, 4, 5, 6]
NEEDED_FILES = ["main.py", "config.yaml"]
NEEDED_DIRS  = ["src", "config", "data/raw", "data/supplementary"]


def _copy_skeleton(dst):
    for f in NEEDED_FILES:
        shutil.copy(os.path.join(ROOT, f), os.path.join(dst, f))
    for d in NEEDED_DIRS:
        shutil.copytree(os.path.join(ROOT, d), os.path.join(dst, d))
    for d in ("output/figures", "models", "data/processed"):
        os.makedirs(os.path.join(dst, d), exist_ok=True)


def test_stages_1_to_6_end_to_end():
    if os.environ.get("ISC_SMOKE") != "1":
        print("skipped: set ISC_SMOKE=1 to run the end-to-end smoke test")
        return
    tmp = tempfile.mkdtemp(prefix="isc_smoke_")
    try:
        _copy_skeleton(tmp)
        spec = ",".join(str(n) for n in SMOKE_STAGES)
        r = subprocess.run([sys.executable, "main.py", "--stages", spec],
                           cwd=tmp, capture_output=True, text=True)
        assert r.returncode == 0, "pipeline failed:\n" + r.stdout[-1500:] + r.stderr[-1500:]
        missing = []
        for n in SMOKE_STAGES:
            for rel in runner.EXPECTED_OUTPUTS[n]:
                p = os.path.join(tmp, rel)
                if not os.path.exists(p) or os.path.getsize(p) == 0:
                    missing.append(f"stage {n}: {rel}")
        assert not missing, "artifacts not produced:\n" + "\n".join(missing)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_stages_1_to_6_end_to_end()
    print("smoke test finished")
