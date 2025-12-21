#!/usr/bin/env python3
"""
Validate CSV dataset against the EssaySchema.

Usage:
    python3 scripts/validate_data.py data/sample_sanitized.csv
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.data.schema import EssaySchema


def main():
    parser = argparse.ArgumentParser(description="Validate essay CSV file.")
    parser.add_argument("csv_path", type=str, help="Path to CSV file to validate")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv_path)
    except Exception as exc:
        print(f"Failed to read {args.csv_path}: {exc}", file=sys.stderr)
        sys.exit(2)

    ok, errors = EssaySchema.validate(df)
    if ok:
        print("Schema OK")
        sys.exit(0)
    else:
        print("Schema validation FAILED", file=sys.stderr)
        for e in errors:
            print(" -", e, file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
