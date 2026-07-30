"""
Inference & External Validation — Bosnian AI Text Detector

Aligned with the new data pipeline:
  - accepts CSV files OR raw <***> .txt dumps (buka.ba format) as input
  - applies the SAME text cleaning as training (clean_text from data_preparation)
  - uses the 'year' column from the data when present (falls back to filename)
  - per-source AND per-category breakdown (like Table 5 of the Turkish paper)
  - prediction columns are prefixed pred_* so they never overwrite dataset labels

Usage:
    # Single article
    python predict.py --model_dir output/final_model text --text "Predsjednik je danas izjavio..."

    # Batch CSV
    python predict.py --model_dir output/final_model batch \
        --input_csv data/external/buka_recent.csv --output_csv results/buka_pred.csv

    # Batch raw .txt dump (buka.ba format, parsed on the fly)
    python predict.py --model_dir output/final_model batch \
        --input_txt "BiHSviClanciPrviDio.txt" --output_csv results/bih_pred.csv

    # Yearly analysis over a directory of CSVs (uses 'year' column if present)
    python predict.py --model_dir output/final_model yearly \
        --input_dir data/external/ --output_dir results/yearly

    # Yearly analysis directly on the buka.ba folder structure
    python predict.py --model_dir output/final_model buka \
        --buka_dir "/content/drive/MyDrive/buka.ba" --output_dir results/buka
"""

import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from data_preparation import clean_text
from parse_raw_articles import parse_dump_file, build_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LABEL_NAMES = {0: "Human-written", 1: "AI-generated/rewritten"}
PRED_COLUMNS = ["pred_label", "pred_label_name", "prob_human", "prob_ai", "confidence"]


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------
class BosnianAIDetector:
    """
    Loads the fine-tuned BERTić model and predicts whether
    a Bosnian news article is human-written or AI-generated/rewritten.
    """

    def __init__(self, model_dir: str, device: str = "auto", max_length: int = 512):
        self.max_length = max_length
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Loading model from {model_dir} → device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model ready.")

    def predict(
        self,
        texts: list[str],
        batch_size: int = 16,
        clean: bool = True,
    ) -> list[dict]:
        """
        Predict for a list of texts.

        clean=True applies the same clean_text() used during training —
        keep this on unless texts were already cleaned, otherwise the
        input distribution won't match what the model saw in training.

        Returns: list of dicts with keys:
            pred_label (int), pred_label_name (str), confidence (float),
            prob_human (float), prob_ai (float)
        """
        if clean:
            texts = [clean_text(t) for t in texts]

        all_results = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = softmax(outputs.logits, dim=-1).cpu().numpy()

            for prob in probs:
                label = int(np.argmax(prob))
                all_results.append({
                    "pred_label":      label,
                    "pred_label_name": LABEL_NAMES[label],
                    "prob_human":      float(prob[0]),
                    "prob_ai":         float(prob[1]),
                    "confidence":      float(np.max(prob)),
                })

        return all_results

    def predict_single(self, text: str) -> dict:
        return self.predict([text])[0]

    def predict_dataframe(self, df: pd.DataFrame, text_col: str = "text",
                          batch_size: int = 16) -> pd.DataFrame:
        """Attach pred_* columns to a DataFrame."""
        results = self.predict(df[text_col].fillna("").tolist(), batch_size=batch_size)
        for key in PRED_COLUMNS:
            df[key] = [r[key] for r in results]
        return df


# ---------------------------------------------------------------------------
# Aggregation — summary tables like Table 5 of the Turkish paper
# ---------------------------------------------------------------------------
def summarize_predictions(
    df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """
    Aggregate predictions by the given columns (e.g. ['year', 'source'] or
    ['year', 'category']). Missing group columns are skipped automatically.
    """
    group_cols = [c for c in group_cols if c in df.columns]
    if not group_cols:
        group_cols = ["_all"]
        df = df.assign(_all="all")

    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n_total = len(sub)
        n_ai = int((sub["pred_label"] == 1).sum())
        rows.append({
            **dict(zip(group_cols, keys)),
            "n_articles":    n_total,
            "human_pct":     round(100 * (1 - n_ai / n_total), 1),
            "ai_pct":        round(100 * n_ai / n_total, 1),
            "mean_conf_pct": round(100 * sub["confidence"].mean(), 1),
        })
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def save_summaries(df: pd.DataFrame, output_dir: str) -> None:
    """Save per-year/source and per-year/category summaries + full predictions."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df.to_csv(out / "predictions_full.csv", index=False)

    summaries = {
        "summary_by_year_source":   ["year", "source"],
        "summary_by_year_category": ["year", "category"],
        "summary_by_year":          ["year"],
    }
    for name, cols in summaries.items():
        available = [c for c in cols if c in df.columns]
        if not available:
            continue
        s = summarize_predictions(df, available)
        s.to_csv(out / f"{name}.csv", index=False)
        logger.info(f"\n{name}:\n{s.to_string(index=False)}")

    logger.info(f"\nAll results saved to {out}/")


# ---------------------------------------------------------------------------
# External validation modes
# ---------------------------------------------------------------------------
def analyze_yearly_distribution(
    detector: BosnianAIDetector,
    input_dir: str,
    output_dir: str,
    batch_size: int = 16,
):
    """
    Apply detector to all CSV files in input_dir.
    Year is taken from the 'year' column when present; otherwise from the
    filename prefix (e.g. 2023_articles.csv → 2023).
    """
    all_dfs = []
    for csv_path in sorted(Path(input_dir).glob("*.csv")):
        df = pd.read_csv(csv_path)
        logger.info(f"Processing {csv_path.name}: {len(df)} articles")

        if "year" not in df.columns:
            year_from_name = csv_path.stem.split("_")[0]
            df["year"] = year_from_name

        df = detector.predict_dataframe(df, batch_size=batch_size)
        all_dfs.append(df)

    if not all_dfs:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    combined = pd.concat(all_dfs, ignore_index=True)
    save_summaries(combined, output_dir)
    return combined


def analyze_buka_folder(
    detector: BosnianAIDetector,
    buka_dir: str,
    output_dir: str,
    batch_size: int = 16,
    exclude_categories: set[str] | None = None,
):
    """
    Run the detector directly over the buka.ba folder structure
    (category subfolders with <***> .txt dumps). Produces the same
    summary tables, broken down by year and category.
    """
    exclude = exclude_categories or {"Karikature i stripovi"}
    base = Path(buka_dir)
    if not base.exists():
        raise FileNotFoundError(f"Directory not found: {base}")

    all_dfs = []
    for cat_dir in sorted(d for d in base.iterdir() if d.is_dir()):
        if cat_dir.name in exclude:
            logger.info(f"Skipping excluded category: {cat_dir.name}")
            continue
        articles = []
        for txt in sorted(cat_dir.glob("*.txt")):
            articles.extend(parse_dump_file(txt))
        if not articles:
            continue
        df = build_dataframe(articles, label=-1)  # -1 = unknown ground truth
        df = df.drop(columns=["label"])
        df["category"] = cat_dir.name
        all_dfs.append(df)
        logger.info(f"Category '{cat_dir.name}': {len(df)} articles")

    if not all_dfs:
        raise FileNotFoundError(f"No articles parsed from {base}")

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Predicting on {len(combined)} articles...")
    combined = detector.predict_dataframe(combined, batch_size=batch_size)
    save_summaries(combined, output_dir)
    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bosnian AI Text Detector — Inference")
    parser.add_argument("--model_dir", default="output/bertic_bosnian_detector/final_model",
                        help="Path to fine-tuned BERTić model")

    subparsers = parser.add_subparsers(dest="mode")

    # Single text
    sp = subparsers.add_parser("text", help="Predict a single article")
    sp.add_argument("--text", required=True, help="Article text (in Bosnian)")

    # Batch CSV or raw txt dump
    sp2 = subparsers.add_parser("batch", help="Predict from a CSV file or <***> .txt dump")
    sp2.add_argument("--input_csv",  default=None)
    sp2.add_argument("--input_txt",  default=None, help="Raw <***> dump instead of CSV")
    sp2.add_argument("--output_csv", default="predictions.csv")
    sp2.add_argument("--text_col",   default="text")
    sp2.add_argument("--batch_size", type=int, default=16)

    # Yearly analysis over CSVs
    sp3 = subparsers.add_parser("yearly", help="Yearly analysis across multiple CSV files")
    sp3.add_argument("--input_dir",  required=True, help="Dir with CSV files")
    sp3.add_argument("--output_dir", default="results/yearly")
    sp3.add_argument("--batch_size", type=int, default=16)

    # Direct analysis of the buka.ba folder structure
    sp4 = subparsers.add_parser("buka", help="Analyze the buka.ba folder structure directly")
    sp4.add_argument("--buka_dir",   required=True, help="Path to buka.ba folder")
    sp4.add_argument("--output_dir", default="results/buka")
    sp4.add_argument("--batch_size", type=int, default=16)
    sp4.add_argument("--exclude", nargs="*", default=None,
                     help="Category folders to skip (default: 'Karikature i stripovi')")

    args = parser.parse_args()
    detector = BosnianAIDetector(args.model_dir)

    if args.mode == "text":
        result = detector.predict_single(args.text)
        print(f"\nPrediction: {result['pred_label_name']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"P(human)={result['prob_human']:.4f}  P(AI)={result['prob_ai']:.4f}")

    elif args.mode == "batch":
        if args.input_txt:
            articles = parse_dump_file(Path(args.input_txt))
            df = build_dataframe(articles, label=-1).drop(columns=["label"])
        elif args.input_csv:
            df = pd.read_csv(args.input_csv)
        else:
            raise SystemExit("Provide --input_csv or --input_txt")

        df = detector.predict_dataframe(df, text_col=args.text_col,
                                        batch_size=args.batch_size)
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        n_ai = int((df["pred_label"] == 1).sum())
        print(f"\nResults saved to {args.output_csv}")
        print(f"AI-generated: {n_ai}/{len(df)} ({100*n_ai/len(df):.1f}%)")
        print(f"Mean confidence: {df['confidence'].mean():.1%}")

    elif args.mode == "yearly":
        analyze_yearly_distribution(
            detector,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )

    elif args.mode == "buka":
        analyze_buka_folder(
            detector,
            buka_dir=args.buka_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            exclude_categories=set(args.exclude) if args.exclude else None,
        )

    else:
        parser.print_help()
