"""
Training Script — BERTić Fine-Tuning for Bosnian AI Text Detection

BERTić is an Electra-based model (Clark et al., 2020) trained on 8B+ tokens of
Bosnian/Croatian/Montenegrin/Serbian text (Ljubešić & Lauc, 2021).
For classification we use ElectraForSequenceClassification via AutoModel.

Run:
    python train.py
    python train.py --output_dir output/run1 --num_epochs 6 --seed 42
"""

import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report
)

from config import cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_splits(data_dir: str) -> DatasetDict:
    train = pd.read_csv(f"{data_dir}/train.csv")
    val   = pd.read_csv(f"{data_dir}/val.csv")
    test  = pd.read_csv(f"{data_dir}/test.csv")

    for df, name in [(train, "train"), (val, "val"), (test, "test")]:
        logger.info(f"{name}: {len(df)} rows, class dist: {df['label'].value_counts().to_dict()}")

    return DatasetDict({
        "train": Dataset.from_pandas(train[["text", "label"]]),
        "val":   Dataset.from_pandas(val[["text", "label"]]),
        "test":  Dataset.from_pandas(test[["text", "label"]]),
    })


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------
def make_tokenize_fn(tokenizer, max_length: int):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
            padding=False,     # Dynamic padding via DataCollatorWithPadding
        )
    return tokenize


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy":  accuracy_score(labels, predictions),
        "f1":        f1_score(labels, predictions, average="macro"),
        "precision": precision_score(labels, predictions, average="macro"),
        "recall":    recall_score(labels, predictions, average="macro"),
    }


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------
def train_single_run(
    run_id: int,
    data_dir: str,
    output_dir: str,
    model_name: str,
    max_length: int,
    num_epochs: int,
    batch_size: int,
    grad_accum: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    label_smoothing: float,
    fp16: bool,
    seed: int,
) -> dict:
    """Run one complete training + evaluation cycle."""

    run_output = f"{output_dir}/run_{run_id}"
    Path(run_output).mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"  Training run {run_id}  (seed={seed})")
    logger.info(f"{'='*60}\n")

    # 1. Tokenizer  —  BERTić uses a cased WordPiece tokenizer (32k vocab)
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # 2. Datasets
    raw_datasets = load_splits(data_dir)
    tokenize_fn = make_tokenize_fn(tokenizer, max_length)
    tokenized = raw_datasets.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    # 3. Model  —  ElectraForSequenceClassification under the hood
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
        ignore_mismatched_sizes=True,
    )

    # 4. Training arguments
    training_args = TrainingArguments(
        output_dir=run_output,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        max_grad_norm=1.0,
        label_smoothing_factor=label_smoothing,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=fp16 and torch.cuda.is_available(),
        seed=seed,
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=0,
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 6. Train
    trainer.train()

    # 7. Evaluate on test set
    test_results = trainer.predict(tokenized["test"])
    preds = np.argmax(test_results.predictions, axis=-1)
    labels = test_results.label_ids

    results = {
        "run_id": run_id,
        "seed": seed,
        "accuracy":  float(accuracy_score(labels, preds)),
        "f1":        float(f1_score(labels, preds, average="macro")),
        "precision": float(precision_score(labels, preds, average="macro")),
        "recall":    float(recall_score(labels, preds, average="macro")),
        "report":    classification_report(
            labels, preds,
            target_names=["Human-written", "AI-generated"],
            output_dict=True,
        ),
    }

    logger.info(
        f"Run {run_id} test results — "
        f"Accuracy: {results['accuracy']:.4f}, "
        f"F1: {results['f1']:.4f}, "
        f"Precision: {results['precision']:.4f}, "
        f"Recall: {results['recall']:.4f}"
    )

    # Save per-run results
    with open(f"{run_output}/test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save best model
    trainer.save_model(f"{run_output}/best_model")
    tokenizer.save_pretrained(f"{run_output}/best_model")

    return results


# ---------------------------------------------------------------------------
# Multi-run training (as done in BERTić paper: 5 runs, report avg ± std)
# ---------------------------------------------------------------------------
def train_multiple_runs(
    data_dir: str,
    output_dir: str,
    num_runs: int = 5,
    **kwargs,
) -> dict:
    all_results = []
    for i in range(1, num_runs + 1):
        run_seed = kwargs.pop("seed", 42) + i - 1
        results = train_single_run(
            run_id=i,
            data_dir=data_dir,
            output_dir=output_dir,
            seed=run_seed,
            **kwargs,
        )
        all_results.append(results)
        kwargs["seed"] = run_seed  # restore for next pop

    # Aggregate
    metrics = ["accuracy", "f1", "precision", "recall"]
    summary = {}
    for m in metrics:
        vals = [r[m] for r in all_results]
        summary[m] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
            "min":  float(np.min(vals)),
            "max":  float(np.max(vals)),
            "per_run": vals,
        }

    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS (mean ± std across runs):")
    for m in metrics:
        s = summary[m]
        logger.info(f"  {m:<12} {s['mean']:.4f} ± {s['std']:.4f}")
    logger.info("="*60)

    with open(f"{output_dir}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Identify best run by F1 and copy its model as "final_model"
    best_run = max(all_results, key=lambda r: r["f1"])
    best_run_dir = f"{output_dir}/run_{best_run['run_id']}/best_model"
    final_model_dir = f"{output_dir}/final_model"
    import shutil
    if Path(final_model_dir).exists():
        shutil.rmtree(final_model_dir)
    shutil.copytree(best_run_dir, final_model_dir)
    logger.info(f"Best model (run {best_run['run_id']}, F1={best_run['f1']:.4f}) → {final_model_dir}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BERTić for Bosnian AI text detection")
    parser.add_argument("--data_dir",      default=cfg.data.processed_data_dir)
    parser.add_argument("--output_dir",    default=cfg.training.output_dir)
    parser.add_argument("--model_name",    default=cfg.model.model_name)
    parser.add_argument("--max_length",    type=int,   default=cfg.model.max_length)
    parser.add_argument("--num_epochs",    type=int,   default=cfg.training.num_train_epochs)
    parser.add_argument("--batch_size",    type=int,   default=cfg.training.per_device_train_batch_size)
    parser.add_argument("--grad_accum",    type=int,   default=cfg.training.gradient_accumulation_steps)
    parser.add_argument("--lr",            type=float, default=cfg.training.learning_rate)
    parser.add_argument("--weight_decay",  type=float, default=cfg.training.weight_decay)
    parser.add_argument("--warmup_ratio",  type=float, default=cfg.training.warmup_ratio)
    parser.add_argument("--label_smooth",  type=float, default=cfg.training.label_smoothing_factor)
    parser.add_argument("--fp16",          action="store_true", default=cfg.training.fp16)
    parser.add_argument("--num_runs",      type=int,   default=cfg.training.num_train_runs)
    parser.add_argument("--seed",          type=int,   default=cfg.training.seed)
    args = parser.parse_args()

    summary = train_multiple_runs(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_runs=args.num_runs,
        model_name=args.model_name,
        max_length=args.max_length,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        label_smoothing=args.label_smooth,
        fp16=args.fp16,
        seed=args.seed,
    )

    print("\nFinal summary:")
    print(json.dumps({k: {"mean": v["mean"], "std": v["std"]} for k, v in summary.items()}, indent=2))
