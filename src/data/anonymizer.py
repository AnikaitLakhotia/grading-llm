"""Deterministic anonymizer for essay texts.

Features:
- Replaces common PII patterns (emails, phone numbers, SSNs).
- Creates deterministic anon_id from essay_id + salt (sha256).
- Heuristic replacement for two-word capitalized names with
  [PERSON_<hash>].
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class Anonymizer:
    salt: str = os.environ.get("ANON_SALT", "grading-llm-default-salt-v1")

    person_pattern: re.Pattern = re.compile(
        r"\b([A-Z][a-z]{1,20}\s+[A-Z][a-z]{1,20})\b"
    )
    email_pattern: re.Pattern = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
    phone_pattern: re.Pattern = re.compile(
        r"\b(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})" r"[-.\s]?\d{3}[-.\s]?\d{4}\b"
    )
    ssn_pattern: re.Pattern = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

    def _hash(self, value: str) -> str:
        h = hashlib.sha256()
        h.update(self.salt.encode("utf-8"))
        h.update(value.encode("utf-8"))
        return h.hexdigest()[:8]

    def anon_id_for(self, essay_id: str) -> str:
        """Return deterministic anonymized id."""
        return f"ANON_{self._hash(str(essay_id))}"

    def anonymize_text(self, text: str) -> str:
        """Replace PII patterns in text."""
        if text is None:
            return text

        s = str(text)
        s = self.email_pattern.sub("[EMAIL]", s)
        s = self.ssn_pattern.sub("[SSN]", s)
        s = self.phone_pattern.sub("[PHONE]", s)

        def _replace_person(match: re.Match) -> str:
            name = match.group(1)
            return f"[PERSON_{self._hash(name)}]"

        return self.person_pattern.sub(_replace_person, s)

    def anonymize_df(
        self,
        df: pd.DataFrame,
        essay_id_col: str = "essay_id",
        full_text_col: str = "full_text",
    ) -> pd.DataFrame:
        """Return anonymized DataFrame copy."""
        df_copy = df.copy()

        if essay_id_col not in df_copy.columns:
            raise KeyError(f"essay_id column '{essay_id_col}' not found")

        df_copy["anon_id"] = [self.anon_id_for(eid) for eid in df_copy[essay_id_col]]

        df_copy[full_text_col] = [
            self.anonymize_text(t) for t in df_copy[full_text_col]
        ]

        for col in ("student_name", "student_email", "name"):
            if col in df_copy.columns:
                logger.info("Dropping suspicious PII column: %s", col)
                df_copy = df_copy.drop(columns=[col])

        return df_copy
