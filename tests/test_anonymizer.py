import re

import pandas as pd

from src.data.anonymizer import Anonymizer


def test_anonymize_email_and_phone_and_deterministic_anon_id():
    a = Anonymizer(salt="test-salt-123")

    text = (
        "Contact me at alice.smith@example.com or "
        "555-234-5678. Also say hi to John Smith."
    )

    anonymized = a.anonymize_text(text)

    assert "[EMAIL]" in anonymized
    assert "[PHONE]" in anonymized
    assert re.search(r"\[PERSON_[0-9a-f]{8}\]", anonymized)

    id1 = a.anon_id_for("e100")
    id2 = a.anon_id_for("e100")

    assert id1 == id2
    assert id1.startswith("ANON_")
    assert len(id1) == len("ANON_") + 8


def test_anonymize_df_adds_anon_id_and_replaces_text():
    a = Anonymizer(salt="another-salt")

    df = pd.DataFrame(
        [
            {
                "essay_id": "e200",
                "full_text": ("Reach me at bob@example.com " "and call 555-123-4567."),
            },
            {
                "essay_id": "e201",
                "full_text": "No pii here, just text.",
            },
        ]
    )

    out = a.anonymize_df(df)

    assert "anon_id" in out.columns
    assert out.loc[0, "anon_id"].startswith("ANON_")
    assert "[EMAIL]" in out.loc[0, "full_text"]
