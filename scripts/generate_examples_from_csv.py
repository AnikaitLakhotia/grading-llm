#!/usr/bin/env python3
"""
Generate labeling examples (positive/negative) per score (1..6) from an essays CSV.

Heuristic:
- For each score s in 1..6:
  - Filter essays with score == s
  - Compute median word count for that score
  - Positive examples = top 3 essays (by word count) at or above median (more developed)
  - Negative examples = bottom 3 essays (by word count) at or below median (less developed / shorter)
  - If not enough examples exist, fall back to random sampling without replacement.

Outputs:
- docs/examples/examples.csv with columns: essay_id, score, example_type, snippet, explanation

Usage:
    python3 scripts/generate_examples_from_csv.py --input data/sample_sanitized.csv --out docs/examples/examples.csv --snippet-chars 300 --seed 42
"""
from __future__ import annotations

import argparse
import os
import random
import textwrap
from typing import List

import pandas as pd


def _snippet(text: str, n_chars: int = 300) -> str:
    if not isinstance(text, str):
        return ""
    s = " ".join(text.strip().split())  # collapse whitespace/newlines
    if len(s) <= n_chars:
        return s
    # cut at nearest sentence boundary before n_chars if possible
    cut = s.rfind(".", 0, n_chars)
    if cut == -1:
        return s[: n_chars - 3] + "..."
    return s[: cut + 1]


def pick_examples_for_score(
    df_score: pd.DataFrame, score: int, n_pos: int = 3, n_neg: int = 3, seed: int = 42
) -> List[dict]:
    """
    Return list of example dicts for a given score.
    Each dict: {essay_id, score, example_type ('positive'/'negative'), snippet, explanation}
    """
    out = []
    if df_score.empty:
        # nothing to sample — return placeholders
        for i in range(n_pos):
            out.append(
                {
                    "essay_id": f"missing_{score}_pos_{i}",
                    "score": score,
                    "example_type": "positive",
                    "snippet": "",
                    "explanation": f"No real examples for score {score}; placeholder positive example.",
                }
            )
        for i in range(n_neg):
            out.append(
                {
                    "essay_id": f"missing_{score}_neg_{i}",
                    "score": score,
                    "example_type": "negative",
                    "snippet": "",
                    "explanation": f"No real examples for score {score}; placeholder negative example.",
                }
            )
        return out

    # ensure deterministic choices
    rng = random.Random(seed + score)

    df_score = df_score.copy()
    # ensure essay_word_count exists; if not, compute
    if "essay_word_count" not in df_score.columns:
        df_score["essay_word_count"] = (
            df_score["full_text"].fillna("").apply(lambda t: len(str(t).split()))
        )

    median = (
        int(df_score["essay_word_count"].median())
        if not df_score["essay_word_count"].isnull().all()
        else 0
    )

    # candidate positives: those with word_count >= median, sort desc
    pos_cands = df_score[df_score["essay_word_count"] >= median].sort_values(
        "essay_word_count", ascending=False
    )
    neg_cands = df_score[df_score["essay_word_count"] <= median].sort_values(
        "essay_word_count", ascending=True
    )

    # Select top N from candidates; if insufficient, sample from df_score (without replacement)
    def take_or_sample(
        cands: pd.DataFrame, n: int, fallback_df: pd.DataFrame
    ) -> List[int]:
        ids = cands.index.tolist()
        if len(ids) >= n:
            return ids[:n]
        # not enough: take all, then sample more from fallback (excluding already taken)
        taken = ids.copy()
        remaining = [i for i in fallback_df.index.tolist() if i not in taken]
        rng.shuffle(remaining)
        needed = n - len(taken)
        taken.extend(remaining[:needed])
        return taken

    pos_idx = take_or_sample(pos_cands, n_pos, df_score)
    neg_idx = take_or_sample(neg_cands, n_neg, df_score)

    for idx in pos_idx:
        row = df_score.loc[idx]
        out.append(
            {
                "essay_id": row.get("essay_id", f"idx_{idx}"),
                "score": int(score),
                "example_type": "positive",
                "snippet": _snippet(str(row.get("full_text", ""))),
                "explanation": f"Representative positive example for score {score} (word_count={row.get('essay_word_count')}).",
            }
        )
    for idx in neg_idx:
        row = df_score.loc[idx]
        out.append(
            {
                "essay_id": row.get("essay_id", f"idx_{idx}"),
                "score": int(score),
                "example_type": "negative",
                "snippet": _snippet(str(row.get("full_text", ""))),
                "explanation": f"Representative negative example for score {score} (word_count={row.get('essay_word_count')}).",
            }
        )

    return out


def main(argv: List[str] | None = None):
    p = argparse.ArgumentParser(description="Generate examples CSV from essays CSV.")
    p.add_argument(
        "--input",
        required=True,
        help="Path to input essays CSV (must have columns essay_id, score, full_text, essay_word_count)",
    )
    p.add_argument(
        "--out", default="docs/examples/examples.csv", help="Output examples CSV"
    )
    p.add_argument(
        "--snippet-chars", type=int, default=300, help="Max characters per snippet"
    )
    p.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    p.add_argument(
        "--n-pos", type=int, default=3, help="Number of positive examples per score"
    )
    p.add_argument(
        "--n-neg", type=int, default=3, help="Number of negative examples per score"
    )
    args = p.parse_args(argv)

    df = pd.read_csv(args.input)
    required = {"essay_id", "score", "full_text"}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        raise ValueError(f"Input CSV missing required columns: {missing}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for s in range(1, 7):
        df_s = df.loc[df["score"] == s]
        exs = pick_examples_for_score(
            df_s, s, n_pos=args.n_pos, n_neg=args.n_neg, seed=args.seed
        )
        for e in exs:
            # truncate snippet to requested length
            if e["snippet"]:
                e["snippet"] = (e["snippet"][: args.snippet_chars]).strip()
            rows.append(e)

    out_df = pd.DataFrame(
        rows, columns=["essay_id", "score", "example_type", "snippet", "explanation"]
    )
    out_df.to_csv(args.out, index=False)
    print(f"Wrote examples to {args.out} (rows={len(out_df)})")


if __name__ == "__main__":
    main()
