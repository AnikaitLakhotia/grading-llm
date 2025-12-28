#!/usr/bin/env python3
# src/finetune/lora_finetune.py
"""
LoRA-capable fine-tune script for sequence classification (ordinal labels 1..6).

Usage (CPU, tiny test):
  python src/finetune/lora_finetune.py \
    --input data/sample_sanitized.csv \
    --text-col full_text \
    --label-col final_score \
    --model-name distilbert-base-uncased \
    --epochs 1 \
    --per-device-train-batch-size 8

To use LoRA (requires peft, accelerate, transformers >= 4.30):
  pip install transformers datasets accelerate peft
  python src/finetune/lora_finetune.py --use-lora --target-modules q,k,v,o --model-name distilbert-base-uncased ...

Notes:
- LoRA target modules vary by model architecture. For BERT-like models use target modules like:
    ["query", "value"] or ["q","v","k","o"] depending on implementation.
- This script will fall back to regular HF Trainer if PEFT fails or is not installed.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# repo metrics
from src.eval.metrics import qwk as compute_qwk
from src.eval.metrics import within_one_accuracy

# Optional imports (we handle absence gracefully)
try:
    import transformers
    from datasets import ClassLabel, Dataset, DatasetDict
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )
except Exception as e:
    transformers = None  # type: ignore
    AutoConfig = AutoModelForSequenceClassification = AutoTokenizer = None  # type: ignore
    Dataset = None  # type: ignore
    TrainingArguments = Trainer = None  # type: ignore
    DataCollatorWithPadding = None  # type: ignore

# peft is optional
PEFT_AVAILABLE = False
try:
    from peft import (  # type: ignore
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("lora_finetune")


@dataclass
class FinetuneArgs:
    input: str
    text_col: str
    label_col: str
    model_name: str
    output_dir: str
    epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    lr: float
    seed: int
    use_lora: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: Optional[List[str]]


def parse_args(argv: Optional[List[str]] = None) -> FinetuneArgs:
    p = argparse.ArgumentParser(
        description="LoRA-capable finetune for sequence classification"
    )
    p.add_argument(
        "--input", type=str, required=True, help="CSV or parquet with text + label"
    )
    p.add_argument("--text-col", type=str, default="full_text")
    p.add_argument("--label-col", type=str, default="final_score")
    p.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    p.add_argument("--output-dir", type=str, default="models/finetune")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per-device-train-batch-size", type=int, default=8)
    p.add_argument("--per-device-eval-batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--use-lora", action="store_true", help="Enable LoRA (requires peft)"
    )
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--target-modules",
        type=str,
        default="",
        help="Comma-separated target modules for LoRA (e.g. q,k,v,o)",
    )
    args = p.parse_args(argv)
    tmods = args.target_modules.split(",") if args.target_modules else None
    return FinetuneArgs(
        input=args.input,
        text_col=args.text_col,
        label_col=args.label_col,
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        lr=args.lr,
        seed=args.seed,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=tmods,
    )


def load_dataframe(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def prepare_datasets(
    df: pd.DataFrame, text_col: str, label_col: str, seed: int, test_size: float = 0.2
) -> Tuple[DatasetDict, Dict[int, int]]:
    # drop rows without label
    df = df.copy()
    if label_col not in df.columns and "score" in df.columns:
        label_col = "score"
    df = df.dropna(subset=[text_col, label_col])
    df[label_col] = df[label_col].astype(int)

    # Map labels to [0..k-1] for HF classification API
    labels_sorted = sorted(df[label_col].unique())
    label_to_int = {v: i for i, v in enumerate(labels_sorted)}
    df["label"] = df[label_col].map(label_to_int)

    ds = Dataset.from_pandas(df[[text_col, "label", "essay_id"]])
    ds = ds.train_test_split(test_size=test_size, seed=seed)
    # ensure dataset dict keys train/eval
    return ds, label_to_int


def tokenize_and_collate(
    tokenizer, dataset: Dataset, text_col: str, max_length: int = 512
):
    def preprocess(batch):
        return tokenizer(batch[text_col], truncation=True, max_length=max_length)

    tokenized = dataset.map(
        preprocess, batched=True, remove_columns=[text_col, "essay_id"]
    )
    return tokenized


def compute_metrics_for_trainer(p):
    # p is EvalPrediction with predictions and label_ids
    preds = p.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    y_pred = np.argmax(preds, axis=1)
    y_true = p.label_ids
    # y_pred and y_true are in 0..K-1; need to map back to original label values if needed outside.
    # For QWK / within_one, we compute on original ordinal values — but here we compute on int indices.
    # It's ok because mapping preserves order.
    q = compute_qwk(y_true.tolist(), y_pred.tolist())
    w1 = within_one_accuracy(y_true.tolist(), y_pred.tolist())
    return {"qwk": q, "within_one": w1, "accuracy": (y_pred == y_true).mean()}


def apply_lora_if_available(model, args: FinetuneArgs):
    if not PEFT_AVAILABLE:
        raise RuntimeError(
            "PEFT not available: install 'peft' to use LoRA (pip install peft)"
        )

    # Build LoraConfig
    # TaskType.SEQ_CLS is used for sequence classification
    try:
        task_type = TaskType.SEQ_CLS
    except Exception:
        # fallback string
        task_type = "SEQ_CLS"
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules or ["query", "value"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=task_type,
    )
    logger.info("Applying LoRA with config: %s", lora_config.__dict__)
    model = get_peft_model(model, lora_config)
    return model


def main_cli():
    args = parse_args()
    if transformers is None:
        print(
            "transformers/datasets not installed. Install with: pip install transformers datasets"
        )
        sys.exit(1)

    set_seed(args.seed)

    df = load_dataframe(args.input)
    ds, label_map = prepare_datasets(
        df, args.text_col, args.label_col, seed=args.seed, test_size=0.2
    )
    num_labels = len(label_map)
    logger.info("Number of labels: %d mapping: %s", num_labels, label_map)

    # tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    config = AutoConfig.from_pretrained(args.model_name, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, config=config
    )

    # Optionally apply LoRA
    if args.use_lora:
        try:
            model = apply_lora_if_available(model, args)
            logger.info("LoRA adapters applied successfully.")
        except Exception as exc:
            logger.warning(
                "LoRA application failed (%s). Falling back to standard model.", exc
            )

    # Tokenize
    tokenized = tokenize_and_collate(tokenizer, ds, args.text_col)
    train_ds = tokenized["train"]
    eval_ds = tokenized["test"]

    data_collator = DataCollatorWithPadding(tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
        save_total_limit=3,
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="qwk",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_for_trainer,
    )

    # Train
    logger.info("Starting training. Output dir: %s", args.output_dir)
    trainer.train()

    # Save model (and adapters if LoRA used)
    logger.info("Saving final model to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # If LoRA was used, save peft adapters explicitly
    if args.use_lora and PEFT_AVAILABLE:
        try:
            model.save_pretrained(
                args.output_dir
            )  # peft wraps model; this saves adapters too
            logger.info("Saved LoRA adapters to %s", args.output_dir)
        except Exception as exc:
            logger.warning("Could not save LoRA adapters automatically: %s", exc)

    logger.info("Training complete. Artifacts in: %s", args.output_dir)


if __name__ == "__main__":
    main_cli()
