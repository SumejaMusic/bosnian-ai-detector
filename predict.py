"""
Inference & External Validation — Bosnian AI Text Detector

Usage:
    # Single article
    python predict.py --text "Predsjednik je danas izjavio..." --model_dir output/final_model

    # Batch CSV (for external validation across multiple years)
    python predict.py --input_csv data/external/2023_articles.csv \
                      --output_csv results/2023_predictions.csv \
                      --model_dir output/final_model

    # Yearly analysis
    python predict.py --input_dir data/external/ --model_dir output/final_model
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LABEL_NAMES = {0: "Human-written", 1: "AI-generated/rewritten"}


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------
class BosniakAIDetector:
    """
    Loads the fine-tuned BERTić model and predicts whether
    a Bosnian news article is human-written or AI-generated.
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

    def predict(self, texts: list[str], batch_size: int = 16) -> list[dict]:
        """
        Predict for a list of texts.
        Returns: list of dicts with keys:
            label (int), label_name (str), confidence (float),
            prob_human (float), prob_ai (float)
        """
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
                    "label":       label,
                    "label_name":  LABEL_NAMES[label],
                    "prob_human":  float(prob[0]),
                    "prob_ai":     float(prob[1]),
                    "confidence":  float(np.max(prob)),
                })

        return all_results

    def predict_single(self, text: str) -> dict:
        return self.predict([text])[0]


# ---------------------------------------------------------------------------
# External validation — yearly analysis
# ---------------------------------------------------------------------------
def analyze_yearly_distribution(
    detector: BosniakAIDetector,
    input_dir: str,
    output_dir: str,
    batch_size: int = 16,
):
    """
    Apply detector to all CSV files in input_dir.
    Files should be named like: 2023_articles.csv, 2024_articles.csv, etc.
    Each CSV must have a 'text' column and optionally 'source'.
    Produces a summary table as in Table 5 of the Turkish paper.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    yearly_summary = []

    for csv_path in sorted(Path(input_dir).glob("*.csv")):
        year = csv_path.stem.split("_")[0]
        df = pd.read_csv(csv_path)
        logger.info(f"Processing {csv_path.name}: {len(df)} articles")

        results = detector.predict(df["text"].fillna("").tolist(), batch_size=batch_size)
        df["label"]      = [r["label"]      for r in results]
        df["label_name"] = [r["label_name"] for r in results]
        df["prob_human"] = [r["prob_human"] for r in results]
        df["prob_ai"]    = [r["prob_ai"]    for r in results]
        df["confidence"] = [r["confidence"] for r in results]

        # Per-source breakdown (if 'source' column exists)
        sources = df["source"].unique() if "source" in df.columns else ["all"]
        for source in sources:
            sub = df[df["source"] == source] if "source" in df.columns else df
            n_total  = len(sub)
            n_ai     = (sub["label"] == 1).sum()
            pct_human = 100 * (1 - n_ai / n_total)
            pct_ai    = 100 * n_ai / n_total
            mean_conf = sub["confidence"].mean() * 100

            yearly_summary.append({
                "year":          year,
                "source":        source,
                "n_articles":    n_total,
                "human_pct":     round(pct_human, 1),
                "ai_pct":        round(pct_ai, 1),
                "mean_conf_pct": round(mean_conf, 1),
            })
            logger.info(
                f"  {year} | {source}: "
                f"Human={pct_human:.1f}%, AI={pct_ai:.1f}%, "
                f"ConfI={mean_conf:.1f}%"
            )

        df.to_csv(f"{output_dir}/{csv_path.name}", index=False)

    summary_df = pd.DataFrame(yearly_summary)
    summary_df.to_csv(f"{output_dir}/yearly_summary.csv", index=False)
    with open(f"{output_dir}/yearly_summary.json", "w", encoding="utf-8") as f:
        json.dump(yearly_summary, f, indent=2, ensure_ascii=False)

    logger.info(f"\nYearly summary saved to {output_dir}/yearly_summary.csv")
    print("\n" + summary_df.to_string(index=False))
    return summary_df


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

    # Batch CSV
    sp2 = subparsers.add_parser("batch", help="Predict from a CSV file")
    sp2.add_argument("--input_csv",  required=True)
    sp2.add_argument("--output_csv", default="predictions.csv")
    sp2.add_argument("--text_col",   default="text")
    sp2.add_argument("--batch_size", type=int, default=16)

    # Yearly analysis
    sp3 = subparsers.add_parser("yearly", help="Yearly analysis across multiple CSV files")
    sp3.add_argument("--input_dir",  required=True, help="Dir with YYYY_*.csv files")
    sp3.add_argument("--output_dir", default="results/yearly")
    sp3.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()
    detector = BosniakAIDetector(args.model_dir)

    if args.mode == "text":
        result = detector.predict_single(args.text)
        print(f"\nPrediction: {result['label_name']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"P(human)={result['prob_human']:.4f}  P(AI)={result['prob_ai']:.4f}")

    elif args.mode == "batch":
        df = pd.read_csv(args.input_csv)
        results = detector.predict(df[args.text_col].fillna("").tolist(),
                                   batch_size=args.batch_size)
        for key in ["label", "label_name", "prob_human", "prob_ai", "confidence"]:
            df[key] = [r[key] for r in results]
        df.to_csv(args.output_csv, index=False)
        n_ai = sum(r["label"] == 1 for r in results)
        print(f"\nResults saved to {args.output_csv}")
        print(f"AI-generated: {n_ai}/{len(results)} ({100*n_ai/len(results):.1f}%)")

    elif args.mode == "yearly":
        analyze_yearly_distribution(
            detector,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )

    else:
        parser.print_help()
