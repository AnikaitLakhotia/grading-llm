"""Schema validation for essay dataset."""

from __future__ import annotations

import logging
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class SchemaValidationError(Exception):
    """Raised when schema validation fails."""


class EssaySchema:
    """Validates essay DataFrame rows."""

    REQUIRED_COLUMNS = {
        "essay_id": "unique identifier",
        "score": "int between 1 and 6",
        "full_text": "essay text",
        "essay_word_count": "integer word count of full_text",
    }

    OPTIONAL_COLUMNS = {
        "assignment",
        "prompt_name",
        "grade_level",
    }

    SCORE_MIN = 1
    SCORE_MAX = 6
    WORDCOUNT_TOLERANCE = 2

    @staticmethod
    def validate(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        errors: List[str] = []

        for col in EssaySchema.REQUIRED_COLUMNS:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")

        if errors:
            return False, errors

        if df["essay_id"].isnull().any():
            errors.append("Null essay_id values found.")

        if df["essay_id"].duplicated().any():
            dup_ids = df[df["essay_id"].duplicated()]["essay_id"].tolist()
            errors.append(
                f"Duplicate essay_id values found " f"(examples): {dup_ids[:5]}"
            )

        bad_scores = df.loc[
            ~df["score"].isin(
                range(
                    EssaySchema.SCORE_MIN,
                    EssaySchema.SCORE_MAX + 1,
                )
            ),
            "score",
        ]

        if not bad_scores.empty:
            errors.append(
                f"score values outside "
                f"[{EssaySchema.SCORE_MIN},"
                f"{EssaySchema.SCORE_MAX}] "
                f"(examples): "
                f"{list(bad_scores.unique()[:5])}"
            )

        short_texts = df.loc[
            df["full_text"].str.split().apply(lambda x: len(x) < 5),
            "essay_id",
        ].tolist()

        if short_texts:
            errors.append(
                "Some essays are very short (<5 words) "
                f"(examples): {short_texts[:5]}"
            )

        for _, row in df.iterrows():
            actual = len(str(row["full_text"]).split())
            declared = row["essay_word_count"]
            if abs(actual - int(declared)) > EssaySchema.WORDCOUNT_TOLERANCE:
                errors.append(
                    "essay_word_count mismatch for " f"essay_id {row['essay_id']}"
                )
                break

        return not errors, errors
