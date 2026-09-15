#!/usr/bin/env python3
"""
Test runner for Cardiovascular Disease Pipeline test suite.
"""

from __future__ import annotations

import glob
import importlib
import inspect
import pathlib
import sys
import tempfile
import time
import traceback

# Add tests directory to sys.path so 'import pytest' resolves to our shim if pytest is not installed
sys.path.insert(0, str(pathlib.Path(__file__).parent / "tests"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    test_files = sorted(glob.glob("tests/test_*.py"))
    print("=" * 70)
    print(f" CARDIOVASCULAR ML PIPELINE: RUNNING {len(test_files)} TEST MODULES")
    print("=" * 70)

    total = 0
    passed = 0
    failed = 0
    start_time = time.time()

    for tf in test_files:
        mod_name = tf.replace("\\", ".").replace("/", ".").replace(".py", "")
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            print(f"\n[ERROR] Failed to import {mod_name}:")
            traceback.print_exc()
            failed += 1
            continue

        test_funcs = [
            getattr(mod, f)
            for f in dir(mod)
            if f.startswith("test_") and callable(getattr(mod, f))
        ]

        print(f"\nModule: {mod_name} ({len(test_funcs)} tests)")
        for fn in test_funcs:
            total += 1
            fn_name = fn.__name__
            try:
                sig = inspect.signature(fn)
                if "tmp_path" in sig.parameters:
                    with tempfile.TemporaryDirectory() as td:
                        fn(pathlib.Path(td))
                else:
                    fn()
                print(f"  ✓ {fn_name}")
                passed += 1
            except Exception:
                print(f"  ✗ {fn_name} FAILED:")
                traceback.print_exc()
                failed += 1

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    status_str = "ALL TESTS PASSED" if failed == 0 else f"{failed} TESTS FAILED"
    print(f" SUMMARY: {passed}/{total} Passed | {status_str} in {elapsed:.2f}s")
    print("=" * 70)

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
