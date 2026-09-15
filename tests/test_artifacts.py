"""Every pipeline stage's expected artifacts must exist and be non-empty.

This is the "fresh clone" guard: models/ is gitignored, so a clone that has
not run the pipeline fails here with the exact list of what to regenerate.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import main as runner  # repo root is on sys.path via pyproject [tool.pytest] / tests/run_all.py


def _gaps(stages):
    return {n: runner.missing_outputs(n) for n in stages if runner.missing_outputs(n)}


def test_batch_stage_outputs_present():
    """Stages 1-6, 9, 12, 14-28: pure outputs, no trained model required."""
    stages = [n for n in runner.EXPECTED_OUTPUTS if n not in (7, 8, 10, 11, 13)]
    gaps = _gaps(stages)
    assert not gaps, "missing artifacts (re-run with `python3 main.py --stages N`):\n" + "\n".join(
        f"  stage {n}: {', '.join(files)}" for n, files in gaps.items())


def test_trained_model_outputs_present():
    """Stages 7, 8, 10, 11, 13 write models/ — absent on a fresh clone until run."""
    gaps = _gaps([7, 8, 10, 11, 13])
    assert not gaps, "trained artifacts missing (run `python3 main.py --stages 7-8,10-11,13`):\n" + "\n".join(
        f"  stage {n}: {', '.join(files)}" for n, files in gaps.items())


def test_check_command_agrees_with_helpers():
    """`main.py --check` must report the same gap count the helpers compute."""
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(ROOT, "main.py"), "--check"],
                       capture_output=True, text=True, cwd=ROOT)
    expected_gaps = len(_gaps(list(runner.EXPECTED_OUTPUTS)))
    assert (r.returncode == 0) == (expected_gaps == 0), r.stdout[-800:]
