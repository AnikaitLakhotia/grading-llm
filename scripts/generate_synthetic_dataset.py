#!/usr/bin/env python3
"""
Generate a synthetic sanitized dataset for development and CI.

Usage:
    python3 scripts/generate_synthetic_dataset.py --out \
        data/sample_sanitized.csv --n 200 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime
from typing import List

SIMPLE_VOCAB = [
    "analysis",
    "argument",
    "evidence",
    "support",
    "conclude",
    "therefore",
    "however",
    "furthermore",
    "demonstrate",
    "illustrate",
    "contrast",
    "illustration",
    "perspective",
    "interpretation",
    "assertion",
    "motivation",
    "consequence",
    "context",
    "structure",
    "coherence",
]


def generate_sentence(
    word_pool: List[str], length: int, complexity: float = 0.5
) -> str:
    """Generate a single sentence with some punctuation variability."""
    words = [random.choice(word_pool) for _ in range(length)]
    s = " ".join(words).capitalize()

    if random.random() < complexity:
        parts = s.split()
        if len(parts) > 4:
            i = random.randint(2, max(2, len(parts) - 2))
            parts.insert(i, ",")
            s = " ".join(parts)

    return s + "."


def generate_essay_for_score(score: int, seed_offset: int = 0) -> str:
    """
    Heuristic generator that produces essays apparently matching rubric
    strength. Higher scores -> longer, more complex sentences and richer
    vocabulary.
    """
    random.seed(1000 + score * 100 + seed_offset)

    base_sentences = {6: 12, 5: 10, 4: 8, 3: 6, 2: 4, 1: 2}
    complexity = {6: 0.9, 5: 0.75, 4: 0.6, 3: 0.45, 2: 0.35, 1: 0.25}
    vocab_multiplier = {6: 3, 5: 2.5, 4: 2.0, 3: 1.5, 2: 1.2, 1: 1.0}

    sentences = []
    words = SIMPLE_VOCAB * int(vocab_multiplier[score])

    for _ in range(base_sentences[score]):
        length = random.randint(10, 20) if score >= 4 else random.randint(5, 12)
        sentences.append(generate_sentence(words, length, complexity=complexity[score]))

    body = " ".join(sentences)

    if score <= 2:
        body = body.replace(".", "")
        if random.random() < 0.5:
            body += " teh"
    elif score == 3:
        body = body.replace(" however", " but sometimes")

    if random.random() < 0.15:
        body += " Contact: student@example.edu or 555-123-4567."

    return body


def make_rows(n: int, seed: int = 42):
    random.seed(seed)

    rows = []
    prompt_names = ["prompt_oppose", "prompt_advocate", "prompt_explain"]
    assignments = ["argue_x", "explain_y", "analyze_z"]
    grade_levels = [9, 10, 11, 12]

    scores = []
    for s in range(1, 7):
        scores.extend([s] * (n // 6))

    while len(scores) < n:
        scores.append(random.randint(1, 6))

    random.shuffle(scores)

    for i in range(n):
        full_text = generate_essay_for_score(scores[i], seed_offset=i)

        rows.append(
            {
                "essay_id": f"e{100000 + i}",
                "score": int(scores[i]),
                "full_text": full_text,
                "assignment": random.choice(assignments),
                "prompt_name": random.choice(prompt_names),
                "grade_level": random.choice(grade_levels),
                "essay_word_count": len(full_text.split()),
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic essay dataset.")
    parser.add_argument(
        "--out",
        type=str,
        default="data/sample_sanitized.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Number of essays to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic RNG seed",
    )

    args = parser.parse_args()
    rows = make_rows(args.n, seed=args.seed)

    header = [
        "essay_id",
        "score",
        "full_text",
        "assignment",
        "prompt_name",
        "grade_level",
        "essay_word_count",
    ]

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Generated {len(rows)} essays to {args.out} at "
        f"{datetime.utcnow().isoformat()}"
    )


if __name__ == "__main__":
    main()
