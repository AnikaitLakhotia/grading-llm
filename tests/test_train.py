# tests/test_train.py
import json
import os
import tempfile

import pandas as pd

from src.train.train import filename_from_cfg, load_config, run_train


def _make_small_fixture(tmpdir):
    # Create a tiny CSV with text and labels (1..6)
    rows = [
        {
            "essay_id": "e1",
            "full_text": "Strong thesis and clear evidence.",
            "final_score": 6,
        },
        {
            "essay_id": "e2",
            "full_text": "Developing ideas but weak support.",
            "final_score": 3,
        },
        {
            "essay_id": "e3",
            "full_text": "Poor structure and grammar.",
            "final_score": 2,
        },
        {
            "essay_id": "e4",
            "full_text": "Reasonable argument and examples.",
            "final_score": 5,
        },
        {"essay_id": "e5", "full_text": "Minimal content", "final_score": 1},
        {"essay_id": "e6", "full_text": "Adequate and coherent", "final_score": 4},
    ]
    df = pd.DataFrame(rows)
    path = os.path.join(tmpdir, "mini.csv")
    df.to_csv(path, index=False)
    return path


def test_train_smoke(tmp_path, monkeypatch):
    tmpdir = str(tmp_path)
    inpath = _make_small_fixture(tmpdir)

    cfg = {
        "data": {
            "input": inpath,
            "text_column": "full_text",
            "label_column": "final_score",
            "test_size": 0.33,
            "random_seed": 123,
        },
        "features": {"max_features": 100, "ngram_range": [1, 2]},
        "training": {"model_type": "logistic", "max_iter": 200},
        "output": {"out_dir": tmpdir, "prefix": "testtrain"},
    }

    # mimic argparse namespace
    class Args:
        input = inpath
        label_column = "final_score"
        test_size = 0.33
        seed = 123
        config = None

    args = Args()
    summary = run_train(cfg, args)
    assert "qwk" in summary and "model_path" in summary
    # model files exist
    assert os.path.exists(summary["model_path"])
    assert os.path.exists(summary["vectorizer_path"])
