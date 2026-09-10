"""
Chunked, resumable prediction for external validation (post-2022 buka.ba articles).

Why chunked + resumable:
  Colab can disconnect or run out of RAM on long jobs. This script
    1) loads and cleans the articles ONCE and caches them to work_dir/articles.csv,
    2) predicts in chunks and writes every finished chunk to work_dir on Google Drive,
    3) skips chunks that already exist, so after a disconnect you simply re-run
       the same command and it continues where it stopped,
    4) merges all chunks and prints the AI share per year / category at the end.

Speed-ups vs. naive prediction:
  - fp16 autocast on GPU
  - texts sorted by length inside each chunk (much less padding per batch)
  - large inference batch (no gradients, so memory allows it)

Run (Colab):
    python predict_chunked.py \
        --model_dir /content/drive/MyDrive/model3/final_model \
        --input_dir /content/drive/MyDrive/buka.ba \
        --work_dir  /content/drive/MyDrive/external_validation_model3 \
        --min_year 2023

    # or, if you already have a CSV with a 'text' column:
    python predict_chunked.py --model_dir ... --input_csv post2022.csv --work_dir ...
"""

import argparse
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CATEGORY_CANDIDATES = ["category", "rubrika", "RUBRIKA", "kategorija", "section"]


# ---------------------------------------------------------------------------
# 1. Load + clean articles (cached after the first run)
# ---------------------------------------------------------------------------
def load_articles(input_dir, input_csv, min_year, max_year, work_dir: Path) -> pd.DataFrame:
    cache = work_dir / "articles.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        logger.info(f"Loaded {len(df)} cached articles from {cache}")
        return df

    if input_csv:
        df = pd.read_csv(input_csv)
        logger.info(f"Loaded {len(df)} rows from {input_csv}")
    elif input_dir:
        from parse_raw_articles import parse_dump_file, build_dataframe
        files = sorted(Path(input_dir).glob("**/*.txt"))
        if not files:
            raise FileNotFoundError(f"No .txt dumps found under {input_dir}")
        dfs = []
        for p in files:
            articles = parse_dump_file(p)
            part = build_dataframe(articles, label=0)   # label is ignored below
            dfs.append(part)
        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Parsed {len(df)} articles from {len(files)} files under {input_dir}")
    else:
        raise ValueError("Pass either --input_dir or --input_csv")

    if "text" not in df.columns:
        raise ValueError(f"No 'text' column. Available columns: {list(df.columns)}")
    df = df.drop(columns=[c for c in ["label"] if c in df.columns])

    # Year filter (external validation = articles after the ChatGPT boundary)
    if min_year is not None or max_year is not None:
        if "year" not in df.columns:
            raise ValueError(
                f"--min_year/--max_year given but there is no 'year' column. "
                f"Available columns: {list(df.columns)}"
            )
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        before = len(df)
        if min_year is not None:
            df = df[df["year"] >= min_year]
        if max_year is not None:
            df = df[df["year"] <= max_year]
        logger.info(f"Year filter: kept {len(df)}/{before} articles")

    # Same cleaning as in training, otherwise the model sees a different input distribution
    from data_preparation import clean_text
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    df.insert(0, "article_id", np.arange(len(df)))

    work_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    logger.info(f"Cached {len(df)} cleaned articles → {cache}")
    return df


# ---------------------------------------------------------------------------
# 2. Model inference
# ---------------------------------------------------------------------------
class Predictor:
    def __init__(self, model_dir: str, max_length: int, batch_size: int):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            logger.warning("GPU not available — prediction will be VERY slow. "
                           "In Colab: Runtime → Change runtime type → T4 GPU.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        self.max_length = max_length
        self.batch_size = batch_size
        logger.info(f"Model loaded from {model_dir} on {self.device}")

    def predict_proba_ai(self, texts: list) -> np.ndarray:
        """Return P(AI) for every text, in the original order."""
        torch = self.torch
        order = np.argsort([len(t) for t in texts])       # sort by length → less padding
        probs = np.empty(len(texts), dtype=np.float32)

        for start in range(0, len(texts), self.batch_size):
            idx = order[start:start + self.batch_size]
            enc = self.tokenizer(
                [texts[i] for i in idx],
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=(self.device == "cuda")
            ):
                logits = self.model(**enc).logits.float()
            probs[idx] = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        return probs


# ---------------------------------------------------------------------------
# 3. Chunked, resumable loop
# ---------------------------------------------------------------------------
def run_chunks(df: pd.DataFrame, predict_fn, work_dir: Path, chunk_size: int, threshold: float):
    n_chunks = math.ceil(len(df) / chunk_size)
    done = [c for c in range(n_chunks) if (work_dir / f"chunk_{c:04d}.csv").exists()]
    logger.info(f"{len(df)} articles → {n_chunks} chunks of {chunk_size}; already done: {len(done)}")

    t0, processed = time.time(), 0
    for c in range(n_chunks):
        out = work_dir / f"chunk_{c:04d}.csv"
        if out.exists():
            continue
        part = df.iloc[c * chunk_size:(c + 1) * chunk_size]
        probs = predict_fn(part["text"].tolist())

        res = part.drop(columns=["text"]).copy()
        res["prob_ai"] = probs
        res["pred"] = (probs >= threshold).astype(int)

        tmp = work_dir / f"chunk_{c:04d}.tmp"
        res.to_csv(tmp, index=False)
        tmp.rename(out)                     # a chunk is either complete or absent

        processed += len(part)
        rate = processed / (time.time() - t0)
        remaining = sum(
            len(df.iloc[k * chunk_size:(k + 1) * chunk_size])
            for k in range(c + 1, n_chunks)
            if not (work_dir / f"chunk_{k:04d}.csv").exists()
        )
        logger.info(
            f"Chunk {c + 1}/{n_chunks} saved ({len(part)} articles, "
            f"{res['pred'].mean():.1%} AI) — {rate:.0f} art/s, "
            f"ETA {remaining / max(rate, 1e-9) / 60:.1f} min"
        )


# ---------------------------------------------------------------------------
# 4. Merge + summary
# ---------------------------------------------------------------------------
def merge_and_summarise(work_dir: Path, n_expected: int, chunk_size: int):
    n_chunks = math.ceil(n_expected / chunk_size)
    files = [work_dir / f"chunk_{c:04d}.csv" for c in range(n_chunks)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        logger.warning(f"{len(missing)} chunks still missing — re-run the same command to continue.")
        return None

    res = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    res.to_csv(work_dir / "predictions_all.csv", index=False)

    lines = [f"Articles: {len(res)}",
             f"Predicted AI: {res['pred'].sum()} ({res['pred'].mean():.2%})",
             f"Mean P(AI): {res['prob_ai'].mean():.4f}"]
    summary = {"n": int(len(res)), "ai_share": float(res["pred"].mean())}

    if "year" in res.columns:
        by_year = res.groupby("year")["pred"].agg(["count", "sum", "mean"])
        by_year.columns = ["articles", "predicted_ai", "ai_share"]
        by_year.to_csv(work_dir / "summary_by_year.csv")
        lines += ["", "By year:", by_year.to_string(float_format=lambda x: f"{x:.3f}")]

    cat_col = next((c for c in CATEGORY_CANDIDATES if c in res.columns), None)
    if cat_col:
        by_cat = res.groupby(cat_col)["pred"].agg(["count", "sum", "mean"]).sort_values("mean", ascending=False)
        by_cat.columns = ["articles", "predicted_ai", "ai_share"]
        by_cat.to_csv(work_dir / "summary_by_category.csv")
        lines += ["", f"By category ({cat_col}):", by_cat.to_string(float_format=lambda x: f"{x:.3f}")]

    text = "\n".join(lines)
    (work_dir / "summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        from config import cfg
        default_max_len = cfg.model.max_length
    except Exception:
        default_max_len = 512

    p = argparse.ArgumentParser(description="Chunked, resumable prediction for external validation")
    p.add_argument("--model_dir",  required=True)
    p.add_argument("--work_dir",   required=True, help="Put this on Google Drive so progress survives resets")
    p.add_argument("--input_dir",  default=None, help="Folder with <***> .txt dumps (e.g. buka.ba)")
    p.add_argument("--input_csv",  default=None, help="Alternatively: CSV with a 'text' column")
    p.add_argument("--min_year",   type=int, default=None)
    p.add_argument("--max_year",   type=int, default=None)
    p.add_argument("--chunk_size", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_length", type=int, default=default_max_len)
    p.add_argument("--threshold",  type=float, default=0.5)
    args = p.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    df = load_articles(args.input_dir, args.input_csv, args.min_year, args.max_year, work_dir)
    predictor = Predictor(args.model_dir, args.max_length, args.batch_size)
    run_chunks(df, predictor.predict_proba_ai, work_dir, args.chunk_size, args.threshold)
    merge_and_summarise(work_dir, len(df), args.chunk_size)
