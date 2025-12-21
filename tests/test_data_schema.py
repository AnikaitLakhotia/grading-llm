import pandas as pd

from src.data.schema import EssaySchema


def test_schema_valid():
    df = pd.DataFrame(
        [
            {
                "essay_id": "e1",
                "score": 4,
                "full_text": (
                    "This is a valid essay with enough words " "to pass the short test."
                ),
                "assignment": "asm1",
                "prompt_name": "p1",
                "grade_level": 10,
                "essay_word_count": 15,
            }
        ]
    )

    ok, errors = EssaySchema.validate(df)
    assert ok, f"Expected valid schema but got errors: {errors}"


def test_schema_detects_bad_score_and_wordcount():
    df = pd.DataFrame(
        [
            {
                "essay_id": "e2",
                "score": 10,
                "full_text": "Short.",
                "essay_word_count": 1000,
            }
        ]
    )

    ok, errors = EssaySchema.validate(df)

    assert not ok
    assert any("score values outside" in e for e in errors)
    assert any("essay_word_count" in e or "very short" in e for e in errors)
