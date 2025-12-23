# tests/test_baseline.py
import json
import os

import numpy as np
from sklearn.model_selection import train_test_split

from src.baseline.classical import TFIDFLogistic
from src.baseline.llm_client import MockLLMClient
from src.baseline.prompt_runner import PromptRunner
from src.eval.metrics import qwk, within_one_accuracy


def _small_dataset():
    texts = [
        "This essay develops a strong thesis and uses evidence effectively.",
        "Simple short response. lacks detail",
        "An attempt at an argument with some structure but weak support.",
        "Masterful analysis with insightful critique and varied vocabulary.",
        "Poor grammar and unclear ideas, sentence fragments everywhere.",
    ]
    # Assign scores 6,2,3,6,1
    scores = [6, 2, 3, 6, 1]
    return texts, scores


def test_tfidf_logistic_smoke():
    texts, scores = _small_dataset()
    # For this tiny synthetic dataset we avoid stratified splitting to prevent
    # sklearn errors about too-few-members-per-class in the training fold.
    X_train, X_test, y_train, y_test = train_test_split(
        texts, scores, test_size=0.4, random_state=42
    )
    model = TFIDFLogistic(max_features=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    # predictions should be ints in 1..6
    assert all(1 <= int(p) <= 6 for p in preds)


def test_mock_llm_prompt_runner_output_parse():
    mock = MockLLMClient(salt="test-salt")
    runner = PromptRunner(llm_client=mock)
    essay = "A test essay with some content for scoring."
    out = runner.run_single(essay, prompt_context="Test prompt")
    assert isinstance(out, dict)
    assert "score" in out and 1 <= int(out["score"]) <= 6
    assert "feedback" in out and isinstance(out["feedback"], str)
    assert "confidence" in out and 0.0 <= float(out["confidence"]) <= 1.0


def test_metrics_qwk_and_within_one():
    y_true = [1, 2, 3, 4, 5, 6]
    y_pred_good = [1, 2, 3, 4, 5, 6]
    y_pred_ok = [1, 2, 2, 4, 6, 5]
    assert qwk(y_true, y_pred_good) == 1.0
    # For y_pred_ok: absolute diffs are [0,0,1,0,1,1] -> all are <=1 => within_one_accuracy == 1.0
    assert abs(within_one_accuracy(y_true, y_pred_ok) - 1.0) < 1e-9
