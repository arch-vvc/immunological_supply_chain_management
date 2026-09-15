"""Unit tests for the pipeline runner's stage-selection and artifact helpers.

Run with pytest (`pytest tests/`) or without it (`python3 tests/run_all.py`).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import main as runner  # repo root is on sys.path via pyproject [tool.pytest] / tests/run_all.py


def test_parse_single_and_ranges():
    assert runner.parse_stage_spec("1-3,7,27-28") == [1, 2, 3, 7, 27, 28]


def test_parse_dedupes_and_sorts():
    assert runner.parse_stage_spec("5,3-5,3") == [3, 4, 5]


def test_parse_tolerates_whitespace_and_empty_parts():
    assert runner.parse_stage_spec(" 1 , 2 ,, 3 ") == [1, 2, 3]


def test_parse_rejects_unknown_stage():
    try:
        runner.parse_stage_spec("1,99")
    except ValueError as e:
        assert "99" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown stage")


def test_parse_rejects_reversed_range():
    try:
        runner.parse_stage_spec("6-2")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for reversed range")


def test_every_stage_has_an_expected_outputs_entry():
    nums = {s[0] for s in runner.STAGES}
    missing = sorted(nums - set(runner.EXPECTED_OUTPUTS))
    assert not missing, f"stages without EXPECTED_OUTPUTS: {missing}"


def test_training_stages_are_real_stages():
    nums = {s[0] for s in runner.ALL_STAGES}
    assert runner.TRAINING_STAGES <= nums


def test_missing_outputs_reports_absent_files(tmp_path=None):
    # Point one stage at a path that cannot exist and confirm it is reported.
    saved = runner.EXPECTED_OUTPUTS[1]
    try:
        runner.EXPECTED_OUTPUTS[1] = ["output/__definitely_not_here__.csv"]
        assert runner.missing_outputs(1) == ["output/__definitely_not_here__.csv"]
    finally:
        runner.EXPECTED_OUTPUTS[1] = saved


def test_pyproject_modules_are_import_safe():
    """Every module `pip install -e .` exposes must be safe to import: it has a
    `__main__` guard, or it is a pure library (no stage work at module scope).
    Stage scripts that run at import time must NOT be listed."""
    import re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = open(os.path.join(root, "pyproject.toml")).read()
    listed = re.findall(r'"([A-Za-z_0-9]+)"', text.split("py-modules")[1].split("]")[0])
    pure_libraries = {"domain_config", "ir_schema", "ir_stages", "isc_common",
                      "arcos_adapter", "scms_adapter", "dataco_adapter", "ai_agent"}
    bad = []
    for m in listed:
        path = os.path.join(root, "src", m + ".py")
        if not os.path.exists(path):
            bad.append(f"{m}: file missing"); continue
        src = open(path).read()
        if m not in pure_libraries and '__name__ == "__main__"' not in src \
                and "__name__ == '__main__'" not in src:
            bad.append(f"{m}: no __main__ guard (importing it would run the stage)")
    assert not bad, "\n".join(bad)


def test_stage_scripts_are_not_exposed_as_modules():
    """Modules with module-scope pipeline work must stay out of py-modules."""
    import re, glob
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = open(os.path.join(root, "pyproject.toml")).read()
    listed = set(re.findall(r'"([A-Za-z_0-9]+)"', text.split("py-modules")[1].split("]")[0]))
    pure_libraries = {"domain_config", "ir_schema", "ir_stages", "isc_common",
                      "arcos_adapter", "scms_adapter", "dataco_adapter", "ai_agent"}
    leaking = []
    for p in glob.glob(os.path.join(root, "src", "*.py")):
        m = os.path.basename(p)[:-3]
        if m in listed and m not in pure_libraries and "__name__ ==" not in open(p).read():
            leaking.append(m)
    assert not leaking, f"exposed but run at import time: {leaking}"
