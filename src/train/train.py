# src/train/train.py
"""
Reproducible training pipeline CLI.

Features:
- Loads processed dataset (Parquet or CSV)
- Extracts TF-IDF features (configurable)
- Trains a lightweight classifier (LogisticRegression or MLP)
- Reports metrics: QWK, ±1 accuracy, classification report
- Saves model + vectorizer with deterministic filename (seed + timestamp)
- Supports YAML config file (PyYAML) or direct CLI args

Usage:
  python -m src.train.train --config configs/default.yaml --input data/processed.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from src.eval.metrics import classification_report_dict, qwk, within_one_accuracy

try:
    import yaml
except Exception:
    yaml = None  # type: ignore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    logger.info("Seed set to %d", seed)


def load_config(path: str | None) -> Dict[str, Any]:
    if path is None:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML not installed. Install pyyaml to use config files.")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_data(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_feature_pipeline(cfg: Dict[str, Any]) -> TfidfVectorizer:
    feat_cfg = cfg.get("features", {})
    max_features = int(feat_cfg.get("max_features", 5000))
    ngram = tuple(feat_cfg.get("ngram_range", (1, 2)))
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram)
    return vectorizer


def build_model(cfg: Dict[str, Any], random_state: int = 42):
    tcfg = cfg.get("training", {})
    model_type = tcfg.get("model_type", "logistic")
    max_iter = int(tcfg.get("max_iter", 1000))
    if model_type == "logistic":
        try:
            clf = LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=max_iter,
                random_state=random_state,
            )
        except TypeError:
            clf = LogisticRegression(
                solver="lbfgs", max_iter=max_iter, random_state=random_state
            )
    elif model_type == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(256,), max_iter=max_iter, random_state=random_state
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return clf


def prepare_xy(
    df: pd.DataFrame, text_col: str, label_col: str
) -> Tuple[list[str], list[int]]:
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col]
    valid_mask = labels.notnull()
    texts = [t for t, v in zip(texts, valid_mask) if v]
    labels = labels[valid_mask].astype(int).tolist()
    return texts, labels


def filename_from_cfg(prefix: str, seed: int) -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = hashlib.sha256(f"{prefix}-{seed}-{now}".encode("utf-8")).hexdigest()[:8]
    return f"{prefix}.{now}.s{seed}.{digest}"


def run_train(config: Dict[str, Any], args: argparse.Namespace):
    cfg = dict(config)
    cfg.setdefault("data", {})
    cfg.setdefault("training", {})
    cfg.setdefault("features", {})
    cfg.setdefault("output", {})

    if args.input:
        cfg["data"]["input"] = args.input
    if args.label_column:
        cfg["data"]["label_column"] = args.label_column
    if args.test_size is not None:
        cfg["data"]["test_size"] = args.test_size
    if args.seed is not None:
        cfg["data"]["random_seed"] = args.seed

    seed = int(cfg["data"].get("random_seed", 42))
    set_seed(seed)

    in_path = cfg["data"].get("input")
    if in_path is None:
        raise ValueError("No input data path provided (cfg data.input or --input)")

    logger.info("Loading data from %s", in_path)
    df = load_data(in_path)

    label_col = cfg["data"].get("label_column", "final_score")
    if label_col not in df.columns and "score" in df.columns:
        label_col = "score"

    text_col = cfg["data"].get("text_column", "full_text")
    texts, labels = prepare_xy(df, text_col, label_col)
    if len(labels) < 2:
        raise ValueError("Not enough labeled examples to train (need >=2)")

    test_size = float(cfg["data"].get("test_size", 0.2))

    # --- robust stratify handling ---
    labels_arr = np.array(labels, dtype=int)
    unique, counts = np.unique(labels_arr, return_counts=True)
    if len(unique) <= 1 or counts.min() < 2:
        stratify_arg = None
        logger.info(
            "Stratification disabled: not enough samples per class for safe stratify."
        )
    else:
        stratify_arg = labels_arr
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=stratify_arg
    )
    # ---------------------------------

    logger.info("Train size: %d, Test size: %d", len(X_train), len(X_test))

    vectorizer = build_feature_pipeline(cfg)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = build_model(cfg, random_state=seed)
    logger.info("Training model...")
    clf.fit(X_train_vec, y_train)

    preds = clf.predict(X_test_vec)
    qwk_score = qwk(y_test, preds)
    within1 = within_one_accuracy(y_test, preds)
    creport = classification_report_dict(y_test, preds)

    logger.info("QWK: %.6f", qwk_score)
    logger.info("±1 acc: %.6f", within1)
    logger.info("Classification report keys: %s", list(creport.keys()))

    out_dir = cfg["output"].get("out_dir", "models")
    os.makedirs(out_dir, exist_ok=True)
    prefix = cfg["output"].get("prefix", "train")
    fname_base = filename_from_cfg(prefix, seed)
    model_path = os.path.join(out_dir, f"{fname_base}.model.joblib")
    vec_path = os.path.join(out_dir, f"{fname_base}.tfidf.joblib")

    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vec_path)
    logger.info("Saved model to %s and vectorizer to %s", model_path, vec_path)

    summary = {
        "model_path": model_path,
        "vectorizer_path": vec_path,
        "qwk": float(qwk_score),
        "within1": float(within1),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(
        os.path.join(out_dir, f"{fname_base}.summary.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(summary, fh, indent=2)

    logger.info("Training complete. Summary: %s", summary)
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Train a small classifier on essay text")
    p.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config (optional)",
    )
    p.add_argument(
        "--input", type=str, help="Path to processed dataset (overrides config)"
    )
    p.add_argument(
        "--label-column", type=str, help="Label column name (overrides config)"
    )
    p.add_argument(
        "--test-size", type=float, help="Test size fraction (overrides config)"
    )
    p.add_argument("--seed", type=int, help="Random seed (overrides config)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = {}
    if args.config:
        cfg = load_config(args.config) if yaml is not None else {}
    try:
        summary = run_train(cfg, args)
        print(json.dumps(summary, indent=2))
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
