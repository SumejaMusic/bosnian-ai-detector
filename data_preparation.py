"""
Data Preparation for Bosnian AI Text Detector

STEP-BY-STEP GUIDE:
1. Collect Bosnian newspaper articles from 1995–2022 (pre-AI era) → human-written label (0)
2. Rewrite each article with ChatGPT/GPT-4 using a structured prompt → AI-generated label (1)
3. Clean, deduplicate, balance, and split the dataset

Expected input CSV structure:
    text,label,source,year
    "Predsjednik je izjavio...",0,"oslobodjenje",2005
    "Predsjednik je izjavio...",1,"oslobodjenje_ai",2005

Run:
    python data_preparation.py --input_dir data/raw --output_dir data/processed
"""

import os
import re
import json
import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PROMPT for ChatGPT rewriting  (same principle as Turkish paper)
# Adjust [NEWSPAPER_NAME] when calling the API
# ---------------------------------------------------------------------------
REWRITE_PROMPT_TEMPLATE = """Ti si iskusan novinski urednik bosanskoga lista {newspaper_name}.
Tvoj zadatak je prepisati ili revidirati priloženi tekst vijesti u skladu s jezikom,
tonom i uređivačkim stilom lista {newspaper_name}. Slijedi sljedeća pravila:

• Ne mijenjaj značenje ni pod kojim uvjetima. Činjenice, datumi i vremena,
  numeričke vrijednosti te imena osoba i institucija moraju biti sačuvani u potpunosti.
• Ne mijenjaj dužinu teksta. Prepisani članak treba biti gotovo iste dužine kao original.
• Analiziraj i primijeni stil lista. Struktura rečenica, odabir riječi, razina formalnosti,
  naglasak i ritam vijesti trebaju odražavati uređivački jezik lista {newspaper_name}.
• Poboljšaj tečnost, smanji ponavljanja, pojednostavi formulacije i ispravi gramatičke
  i interpunkcijske greške. Rezultat treba biti uredničko prepisivanje, a ne kopija.
• Ne dodavaj nove informacije, tumačenja ni zaključke. Koristi samo informacije
  eksplicitno navedene u originalnom tekstu.
• Ne odstupaj od novinarskog jezika; izbjegavaj subjektivne ocjene ili komentare.
• Ako su prisutni citati, sačuvaj sav citirani materijal doslovno. Možeš revidirati
  samo rečenice koje uvode ili zaključuju citat kako bi odgovarale stilu lista.

Originalni tekst:
{article_text}

Prepiši/revidiraj tekst:"""


# ---------------------------------------------------------------------------
# ChatGPT rewrite helper  (requires openai package: pip install openai)
# ---------------------------------------------------------------------------
def rewrite_with_chatgpt(
    article_text: str,
    newspaper_name: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
) -> str:
    """
    Rewrite a Bosnian news article using ChatGPT.
    Returns the AI-rewritten text, or raises on API error.

    Usage:
        os.environ["OPENAI_API_KEY"] = "sk-..."
        ai_text = rewrite_with_chatgpt(original_text, "Oslobođenje")
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai  — needed for AI rewriting")

    client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
    prompt = REWRITE_PROMPT_TEMPLATE.format(
        newspaper_name=newspaper_name,
        article_text=article_text,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Minimal cleaning that preserves Bosnian diacritics.
    BERTić was trained with Unicode preserved (unlike original BERT that stripped diacritics).
    """
    if not isinstance(text, str):
        return ""

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove HTML artifacts that sometimes slip through scrapers
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)

    # Remove news-agency byline at end (e.g. "(FNA)" or "— FENA")
    text = re.sub(r"\s*[\(\-–—]\s*(FENA|SRNA|HINA|Beta|Tanjug|AP|Reuters|AFP)[^\)]*\)?\.?\s*$", "", text)

    # Collapse whitespace again after removals
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------------------------
# Main preparation pipeline
# ---------------------------------------------------------------------------
def load_raw_articles(input_dir: str) -> pd.DataFrame:
    """
    Load all CSV files from input_dir.
    Expected columns: text, label (0=human, 1=AI), source, year (optional)
    """
    dfs = []
    for csv_path in Path(input_dir).glob("**/*.csv"):
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from {csv_path}")
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"No CSV files found in {input_dir}.\n"
            "Create CSVs with columns: text, label, source, year"
        )

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total rows loaded: {len(combined)}")
    return combined


def prepare_dataset(
    input_dir: str,
    output_dir: str,
    min_length: int = 200,
    max_length: int = 50_000,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> dict:
    """
    Full preparation pipeline:
    1. Load raw CSVs
    2. Clean texts
    3. Filter by length
    4. Deduplicate
    5. Balance classes
    6. Stratified split
    7. Save train/val/test CSVs
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load
    df = load_raw_articles(input_dir)

    # 2. Clean
    df["text"] = df["text"].apply(clean_text)

    # 3. Filter by length (characters)
    df["text_len"] = df["text"].str.len()
    before = len(df)
    df = df[(df["text_len"] >= min_length) & (df["text_len"] <= max_length)]
    logger.info(f"Length filter: kept {len(df)}/{before} articles ({min_length}–{max_length} chars)")

    # 4. Deduplicate (exact match on cleaned text)
    before = len(df)
    df = df.drop_duplicates(subset="text")
    logger.info(f"Deduplication: removed {before - len(df)} duplicates")

    # 5. Balance classes (equal number of human vs AI articles)
    n_human = (df["label"] == 0).sum()
    n_ai    = (df["label"] == 1).sum()
    logger.info(f"Class distribution before balancing: human={n_human}, AI={n_ai}")

    n_min = min(n_human, n_ai)
    df_human = df[df["label"] == 0].sample(n_min, random_state=seed)
    df_ai    = df[df["label"] == 1].sample(n_min, random_state=seed)
    df = pd.concat([df_human, df_ai], ignore_index=True).sample(frac=1, random_state=seed)
    logger.info(f"After balancing: {len(df)} articles ({n_min} per class)")

    # 6. Stratified split  (train / val / test)
    test_ratio = 1.0 - train_ratio - val_ratio
    train_df, temp_df = train_test_split(
        df, test_size=(1.0 - train_ratio), stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_df["label"], random_state=seed
    )

    logger.info(
        f"Split — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}"
    )

    # 7. Save
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/val.csv",   index=False)
    test_df.to_csv(f"{output_dir}/test.csv",  index=False)

    # Save split statistics
    stats = {
        "total": len(df),
        "train": len(train_df),
        "val":   len(val_df),
        "test":  len(test_df),
        "class_distribution": {
            "train": train_df["label"].value_counts().to_dict(),
            "val":   val_df["label"].value_counts().to_dict(),
            "test":  test_df["label"].value_counts().to_dict(),
        },
    }
    with open(f"{output_dir}/stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info(f"Dataset saved to {output_dir}/")
    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Bosnian AI detector dataset")
    parser.add_argument("--input_dir",  default="data/raw",       help="Directory with raw CSV files")
    parser.add_argument("--output_dir", default="data/processed",  help="Where to save processed splits")
    parser.add_argument("--min_length", type=int, default=200)
    parser.add_argument("--max_length", type=int, default=50_000)
    parser.add_argument("--train_ratio", type=float, default=0.80)
    parser.add_argument("--val_ratio",   type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stats = prepare_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_length=args.min_length,
        max_length=args.max_length,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print("\nDataset statistics:")
    print(json.dumps(stats, indent=2))
