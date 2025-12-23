#!/usr/bin/env python3
"""
Headless baseline evaluation script.
Searches for data/processed.*.parquet; if missing, falls back to data/sample_sanitized.csv.
Runs TFIDFLogistic baseline and MockLLM PromptRunner and prints QWK and ±1 accuracy.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import glob
import json
import sys as _sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src.baseline.classical import TFIDFLogistic
from src.baseline.llm_client import MockLLMClient
from src.baseline.prompt_runner import PromptRunner
from src.eval.metrics import (
    classification_report_dict,
    confusion,
    qwk,
    within_one_accuracy,
)


def load_data():
    candidates = glob.glob("data/processed.*.parquet")
    if candidates:
        path = candidates[0]
        print("Using processed parquet:", path)
        df = pd.read_parquet(path)
    else:
        fallback = "data/sample_sanitized.csv"
        print("No processed parquet found; falling back to", fallback)
        df = pd.read_csv(fallback)
    return df


def main():
    df = load_data()
    df = df.dropna(subset=["full_text", "score"])
    df["score"] = df["score"].astype(int)
    df_small = df.sample(n=min(1000, len(df)), random_state=42).reset_index(drop=True)
    X = df_small["full_text"].astype(str).tolist()
    y = df_small["score"].astype(int).tolist()
    if len(set(y)) < 2:
        print("Not enough label variety to evaluate.", file=_sys.stderr)
        _sys.exit(2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF baseline
    clf = TFIDFLogistic(max_features=5000, ngram_range=(1, 2))
    clf.fit(X_train, y_train)
    preds_clf = clf.predict(X_test)
    print("TFIDF Logistic QWK:", qwk(y_test, preds_clf))
    print("TFIDF Logistic ±1 acc:", within_one_accuracy(y_test, preds_clf))

    # Mock LLM baseline
    mock = MockLLMClient()
    runner = PromptRunner(llm_client=mock)
    llm_outputs = [runner.run_single(text) for text in X_test]
    preds_llm = [int(o["score"]) for o in llm_outputs]
    print("Mock LLM QWK:", qwk(y_test, preds_llm))
    print("Mock LLM ±1 acc:", within_one_accuracy(y_test, preds_llm))

    # optional: print confusion and simple report
    cm_clf, labels = confusion(y_test, preds_clf, labels=sorted(set(y_test)))
    print("TFIDF confusion matrix (rows=true, cols=pred):")
    print(labels)
    print(cm_clf)
    print("TFIDF classification report (json):")
    print(json.dumps(classification_report_dict(y_test, preds_clf), indent=2))


if __name__ == "__main__":
    main()
