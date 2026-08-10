"""
Training Script — BERTić Fine-Tuning for Bosnian AI Text Detection

BERTić is an Electra-based model (Clark et al., 2020) trained on 8B+ tokens of
Bosnian/Croatian/Montenegrin/Serbian text (Ljubešić & Lauc, 2021).
For classification we use ElectraForSequenceClassification via AutoModel.

Compatible with both older (v4.x) and newer (v4.46+/v5.x) transformers:
  - evaluation_strategy → eval_strategy
  - Trainer(tokenizer=...) → Trainer(processing_class=...)
  - no set_format("torch")  (avoids torchvision VideoReader import conflict;
    DataCollatorWithPadding returns tensors anyway)

Run:
    python train.py
    python train.py --output_dir output/run1 --num_epochs 6 --seed 42
"""

import json
import inspect
import logging
import argparse
import shutil
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
# Version compatibility helpers
# ---------------------------------------------------------------------------
_TRAINING_ARGS_PARAMS = set(inspect.signature(TrainingArguments.__init__).parameters)
_TRAINER_PARAMS       = set(inspect.signature(Trainer.__init__).parameters)


def make_training_args(**kwargs) -> TrainingArguments:
    """
    Build TrainingArguments across transformers versions.
    Pass 'eval_strategy' — it is renamed to 'evaluation_strategy'
    automatically if running on an older transformers version.
    """
    if "eval_strategy" in kwargs and "eval_strategy" not in _TRAINING_ARGS_PARAMS:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return TrainingArguments(**kwargs)


def make_trainer(tokenizer=None, **kwargs) -> Trainer:
    """
    Build Trainer across transformers versions.
    Newer versions use processing_class=, older use tokenizer=.
    """
    if tokenizer is not None:
        if "processing_class" in _TRAINER_PARAMS:
            kwargs["processing_class"] = tokenizer
        else:
            kwargs["tokenizer"] = tokenizer
    return Trainer(**kwargs)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_splits(data_dir: str) -> DatasetDict:
    expected = [f"{data_dir}/train.csv", f"{data_dir}/val.csv", f"{data_dir}/test.csv"]
    missing = [p for p in expected if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing processed splits: {missing}\n"
            "Run data preparation first (it combines human + AI articles):\n"
            "    python data_preparation.py --input_dir data/raw --output_dir data/processed\n"
            "or pass --raw_dir data/raw to train.py to run it automatically."
        )

    train = pd.read_csv(f"{data_dir}/train.csv")
    val   = pd.read_csv(f"{data_dir}/val.csv")
    test  = pd.read_csv(f"{data_dir}/test.csv")

    for df, name in [(train, "train"), (val, "val"), (test, "test")]:
        # Drop rows with missing text/label to avoid tokenizer crashes
        df.dropna(subset=["text", "label"], inplace=True)
        df["label"] = df["label"].astype(int)
        logger.info(f"{name}: {len(df)} rows, class dist: {df['label'].value_counts().to_dict()}")

    return DatasetDict({
        "train": Dataset.from_pandas(train[["text", "label"]], preserve_index=False),
        "val":   Dataset.from_pandas(val[["text", "label"]],   preserve_index=False),
        "test":  Dataset.from_pandas(test[["text", "label"]],  preserve_index=False),
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
    # NOTE: no set_format("torch") here — it triggers a torchvision
    # VideoReader import conflict on Colab, and DataCollatorWithPadding
    # already converts batches to tensors.

    # 3. Model  —  ElectraForSequenceClassification under the hood
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
    )

    # 4. Training arguments (eval_strategy auto-renamed on old versions)
    training_args = make_training_args(
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
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        # Prepravka za 2. trening: eval loss po epohama bio je
        # 0.657 -> 0.527 -> 0.721 -> 0.917 (minimum u 2. epohi), a F1 se zna
        # popravljati i nakon što loss krene rasti jer model postaje presiguran.
        # Izbor po F1 zato bira kasniju, presigurnu epohu (histogram: AI stub u 1.0,
        # pristranost prema AI klasi). Loss bira bolje kalibrisan model.
        # Staro: metric_for_best_model="eval_f1", greater_is_better=True
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        fp16=fp16 and torch.cuda.is_available(),
        seed=seed,
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=0,
    )

    # 5. Trainer (processing_class= on new versions, tokenizer= on old)
    trainer = make_trainer(
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        # Patience 2 (uz eval po epohi) presijecao je runove prerano na malom,
        # šumnom datasetu i doprinosio varijansi 74.6–88.9% među runovima.
        # Staro: early_stopping_patience=2
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
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

    # Free GPU memory between runs (important on Colab T4)
    del trainer, model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


# ---------------------------------------------------------------------------
# Multi-run training (as done in BERTić paper: 5 runs, report avg ± std)
# ---------------------------------------------------------------------------
def train_multiple_runs(
    data_dir: str,
    output_dir: str,
    num_runs: int = 5,
    seed: int = 42,
    **kwargs,
) -> dict:
    base_seed = seed
    all_results = []
    for i in range(1, num_runs + 1):
        run_seed = base_seed + i - 1          # seeds: 42, 43, 44, 45, 46
        results = train_single_run(
            run_id=i,
            data_dir=data_dir,
            output_dir=output_dir,
            seed=run_seed,
            **kwargs,
        )
        all_results.append(results)

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
    parser.add_argument("--raw_dir",       default=None,
                        help="If given, run data_preparation on this dir first "
                             "(combines human + AI CSVs into train/val/test splits)")
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

    if args.raw_dir:
        from data_preparation import prepare_dataset
        logger.info(f"Preparing dataset from {args.raw_dir} → {args.data_dir}")
        prepare_dataset(input_dir=args.raw_dir, output_dir=args.data_dir)

    summary = train_multiple_runs(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_runs=args.num_runs,
        seed=args.seed,
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
    )

    print("\nFinal summary:")
    print(json.dumps({k: {"mean": v["mean"], "std": v["std"]} for k, v in summary.items()}, indent=2))
