#!/usr/bin/env python3
"""Run all tests in sequence. Returns nonzero exit if any fail."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path


def main() -> int:
    tests_dir = Path(__file__).parent
    test_files = sorted(tests_dir.glob("test_*.py"))
    failed = []
    for tf in test_files:
        print(f"\n{'='*60}\nRunning {tf.name}\n{'='*60}")
        result = subprocess.run(
            [sys.executable, str(tf)],
            capture_output=True, text=True,
        )
        sys.stdout.write(result.stdout)
        sys.stdout.write(result.stderr)
        if result.returncode != 0:
            failed.append(tf.name)
    print(f"\n{'='*60}")
    if failed:
        print(f"FAILED: {len(failed)} test file(s):")
        for f in failed:
            print(f"  - {f}")
        return 1
    print(f"All {len(test_files)} test files passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
