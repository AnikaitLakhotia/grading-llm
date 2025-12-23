# src/baseline/classical.py
"""Classical baselines for essay grading (TF-IDF + LogisticRegression)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Model interface


class ModelInterface:
    """Simple interface for baseline models."""

    def fit(self, texts: Iterable[str], y: Iterable[int]) -> None:
        raise NotImplementedError

    def predict(self, texts: Iterable[str]) -> List[int]:
        raise NotImplementedError

    def predict_proba(self, texts: Iterable[str]) -> Any:
        raise NotImplementedError


@dataclass
class TFIDFLogistic(ModelInterface):
    """
    TF-IDF vectorizer + LogisticRegression (multinomial) baseline.
    This is a multiclass classifier; for ordinal tasks use ordinal-specific models later.
    """

    max_features: int = 20000
    ngram_range: tuple = (1, 2)
    solver: str = "lbfgs"
    random_state: int = 42

    def __post_init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features, ngram_range=self.ngram_range
        )
        # Try to construct a multinomial logistic regression when supported,
        # but fall back to older sklearn signatures if necessary.
        try:
            # Preferred constructor for newer sklearn versions
            self.clf = LogisticRegression(
                multi_class="multinomial",
                solver=self.solver,
                max_iter=1000,
                random_state=self.random_state,
            )
        except TypeError:
            # Older sklearn versions may not accept multi_class kwarg; fallback gracefully.
            self.clf = LogisticRegression(
                solver=self.solver, max_iter=1000, random_state=self.random_state
            )

        self.pipeline = Pipeline([("tfidf", self.vectorizer), ("clf", self.clf)])
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def fit(self, texts: Iterable[str], y: Iterable[int]) -> None:
        y_arr = self.label_encoder.fit_transform(list(y))
        self.pipeline.fit(list(texts), y_arr)
        self.is_fitted = True

    def predict(self, texts: Iterable[str]) -> List[int]:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        pred = self.pipeline.predict(list(texts))
        return list(self.label_encoder.inverse_transform(pred))

    def predict_proba(self, texts: Iterable[str]) -> Any:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        proba = self.pipeline.predict_proba(list(texts))
        return proba
