"""Dependency-free test runner: discovers test_*.py in this folder and calls
every `test_*` function. Same functions pytest collects, so `pytest tests/`
gives identical results when pytest is installed.

    python3 tests/run_all.py            # fast suite
    ISC_SMOKE=1 python3 tests/run_all.py  # + end-to-end smoke (slow)
"""
import importlib.util
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)          # so `import main` works without pytest


def main():
    passed = failed = 0
    for fname in sorted(os.listdir(HERE)):
        if not (fname.startswith("test_") and fname.endswith(".py")):
            continue
        spec = importlib.util.spec_from_file_location(fname[:-3], os.path.join(HERE, fname))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            t0 = time.time()
            try:
                fn()
                passed += 1
                print(f"  PASS  {fname}::{name}  ({time.time() - t0:.1f}s)")
            except Exception:
                failed += 1
                print(f"  FAIL  {fname}::{name}")
                traceback.print_exc(limit=3)
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
