"""Run the two script-style validation harnesses in src/ and require exit 0.

They print their own numbered PASS/FAIL lines; this wrapper makes them part of
`pytest tests/` without rewriting them.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(script):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "src", script)],
                       capture_output=True, text=True, cwd=ROOT)
    tail = "\n".join(r.stdout.splitlines()[-12:])
    assert r.returncode == 0, f"{script} failed (exit {r.returncode}):\n{tail}\n{r.stderr[-800:]}"
    return r.stdout


def test_ir_stages_harness():
    out = _run("test_ir_stages.py")
    assert "0 failed" in out, out.splitlines()[-3:]


def test_vendor_scorecard_harness():
    out = _run("test_vendor_scorecard.py")
    assert "ALL ASSERTIONS PASSED" in out, out.splitlines()[-3:]
