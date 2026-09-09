#!/usr/bin/env python3
"""Run the focused LayerKV prefetch overlap unit tests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run LayerKV KVC/expert prefetch overlap tests."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show individual pytest test names",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    test_files = [
        root / "test" / "registered" / "unit" / "layerkv" / "test_concurrent_perf.py",
        root / "test" / "registered" / "unit" / "layerkv" / "test_layer_prefetch_overlap.py",
        root / "test" / "registered" / "unit" / "layerkv" / "test_shared_admission.py",
        root / "test" / "registered" / "unit" / "layerkv" / "test_shared_expert.py",
    ]
    missing = [path for path in test_files if not path.is_file()]
    if missing:
        for path in missing:
            print(f"missing test file: {path}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(root / "python"))
    import pytest

    pytest_args = ["-v" if args.verbose else "-q"]
    pytest_args.extend(str(path) for path in test_files)
    return int(pytest.main(pytest_args))


if __name__ == "__main__":
    raise SystemExit(main())
