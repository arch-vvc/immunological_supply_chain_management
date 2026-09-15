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


def test_pyproject_lists_every_src_module():
    """`pip install -e .` only exposes modules named in [tool.setuptools] py-modules."""
    import re, glob
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = open(os.path.join(root, "pyproject.toml")).read()
    listed = set(re.findall(r'"([A-Za-z_0-9]+)"', text.split("py-modules")[1].split("]")[0]))
    on_disk = {os.path.basename(p)[:-3] for p in glob.glob(os.path.join(root, "src", "*.py"))
               if not os.path.basename(p).startswith("test_")}
    assert on_disk <= listed, f"add to pyproject py-modules: {sorted(on_disk - listed)}"
