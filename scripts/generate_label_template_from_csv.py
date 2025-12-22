#!/usr/bin/env python3
"""
Generate a pre-filled label template CSV for graders sampled from an essays CSV.

Outputs:
- docs/label_templates/label_template_prefilled.csv
Columns: essay_id,orig_score,labeler_id,score,notes,review_timestamp_utc

Usage:
    python3 scripts/generate_label_template_from_csv.py --input data/sample_sanitized.csv --out docs/label_templates/label_template_prefilled.csv --per-score 10 --seed 42
"""
from __future__ import annotations

import argparse
import os
import random
from typing import List

import pandas as pd


def sample_for_labeling(
    df: pd.DataFrame, per_score: int = 10, seed: int = 42
) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for s in range(1, 7):
        df_s = df[df["score"] == s]
        if df_s.empty:
            continue
        sample_n = min(per_score, len(df_s))
        sampled = df_s.sample(n=sample_n, random_state=seed)
        for _, r in sampled.iterrows():
            rows.append(
                {
                    "essay_id": r["essay_id"],
                    "orig_score": int(r["score"]),
                    "labeler_id": "",  # grader fills
                    "score": "",  # grader fills in
                    "notes": "",
                    "review_timestamp_utc": "",
                }
            )
    out = pd.DataFrame(
        rows,
        columns=[
            "essay_id",
            "orig_score",
            "labeler_id",
            "score",
            "notes",
            "review_timestamp_utc",
        ],
    )
    return out


def main(argv: List[str] | None = None):
    p = argparse.ArgumentParser(
        description="Generate label template CSV pre-filled with essay ids."
    )
    p.add_argument("--input", required=True, help="Essays CSV path")
    p.add_argument(
        "--out",
        default="docs/label_templates/label_template_prefilled.csv",
        help="Output CSV",
    )
    p.add_argument(
        "--per-score",
        type=int,
        default=10,
        help="Number of examples per score to include",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    df = pd.read_csv(args.input)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df = sample_for_labeling(df, per_score=args.per_score, seed=args.seed)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote label template to {args.out} (rows={len(out_df)})")


if __name__ == "__main__":
    main()
