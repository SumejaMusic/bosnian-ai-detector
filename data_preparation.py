"""
Data Preparation for Bosnian AI Text Detector

STEP-BY-STEP GUIDE:
1. Collect Bosnian newspaper articles (pre-AI era) → human-written label (0)
2. Rewrite each article with ChatGPT/GPT-4 using a structured prompt → AI label (1)
3. Clean, deduplicate, keep complete human–AI pairs, and split BY PAIR

Supported inputs in data/raw/:
  A) CSV files with columns: text, label, source, year
     (+ src_idx in the AI CSV, linking each rewrite to its original)
  B) Raw .txt newspaper dumps where articles are separated by <***> and start
     with a metadata header block:

        <***>
        NOVINA: Dnevni Avaz
        DATUM: 06.05.2025
        RUBRIKA: N/A
        NADNASLOV: ...
        NASLOV: ...
        PODNASLOV: ...
        STRANA: N/A
        AUTOR(I): N/A

        Tekst članka...

     For .txt dumps, the label is inferred from the filename:
        *_human*.txt / *human*.txt  → label 0
        *_ai*.txt    / *ai*.txt     → label 1
     or can be forced with --txt_label.

Pairing / leakage prevention:
  Every AI article is a rewrite of a specific human article. If the original
  lands in train and its rewrite in test (or vice versa), the test set no longer
  measures generalisation. Each row therefore gets a `pair_id`:
    - CSV with a `src_idx` column → pair_id = src_idx
    - CSV without `src_idx`       → pair_id = row index in that file
  Only complete pairs (one human + one AI) are kept, and the train/val/test split
  is done on pair_ids, so an original and its rewrite always share a split.

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

from parse_raw_articles import parse_dump_file, build_dataframe

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
        ai_text = rewrite_with_chatgpt(original_text, "Dnevni Avaz")
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

# Metadata header lines that may leak into text if a dump was pasted into a CSV
METADATA_HEADER_RE = re.compile(
    r"^(NOVINA|PORTAL|DATUM|RUBRIKA|NADNASLOV|NASLOV|PODNASLOV|STRANA|AUTOR\(I\)|AUTORI|AUTOR|LINK)\s*:.*$",
    re.MULTILINE,
)
DELIMITER_RE = re.compile(r"<\*{3}>")


def clean_text(text: str) -> str:
    """
    Minimal cleaning that preserves Bosnian diacritics.
    BERTić was trained with Unicode preserved (unlike original BERT that stripped diacritics).
    """
    if not isinstance(text, str):
        return ""

    # Remove article delimiter and any metadata header lines that slipped through
    text = DELIMITER_RE.sub(" ", text)
    text = METADATA_HEADER_RE.sub(" ", text)

    # Remove HTML artifacts that sometimes slip through scrapers
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)

    # Remove news-agency byline at end (e.g. "(FENA)" or "— SRNA")
    text = re.sub(
        r"\s*[\(\-–—]\s*(FENA|SRNA|HINA|Beta|Tanjug|AP|Reuters|AFP|Avaz)[^\)]*\)?\.?\s*$",
        "", text,
    )

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------------------------
# Loading raw articles (CSV + raw .txt dumps)
# ---------------------------------------------------------------------------
def infer_label_from_filename(path: Path) -> Optional[int]:
    """Infer label from filename: '*ai*' → 1, '*human*' → 0."""
    name = path.stem.lower()
    # Check 'ai' patterns first, but avoid matching e.g. 'avaiz' by using word-ish boundaries
    if re.search(r"(^|[_\-])ai([_\-]|$)|_ai\b|ai_rewritten|chatgpt|gpt", name):
        return 1
    if re.search(r"human|original|ljudsk", name):
        return 0
    return None


def add_pair_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach a pair_id that links a human original with its AI rewrite.
      - existing 'pair_id' column  → kept as is
      - 'src_idx' column (AI CSV)  → pair_id = src_idx
      - otherwise (human CSV)      → pair_id = row index in the file
    Assumes src_idx in the AI CSV refers to the row index of the original in the
    human CSV. Verify with the Jaccard check before training.
    """
    if "pair_id" in df.columns:
        df["pair_id"] = pd.to_numeric(df["pair_id"], errors="coerce")
    elif "src_idx" in df.columns:
        df["pair_id"] = pd.to_numeric(df["src_idx"], errors="coerce")
    else:
        df["pair_id"] = np.arange(len(df))
    return df


def load_raw_articles(input_dir: str, txt_label: Optional[int] = None) -> pd.DataFrame:
    """
    Load all CSV and .txt dump files from input_dir.

    CSV expected columns: text, label (0=human, 1=AI), source, year (optional),
                          src_idx (AI CSV, optional but needed for pairing)
    TXT dumps: parsed via parse_raw_articles; label from filename or --txt_label
    """
    dfs = []

    # 1) CSV files
    for csv_path in Path(input_dir).glob("**/*.csv"):
        df = pd.read_csv(csv_path)
        df = add_pair_id(df)
        logger.info(f"Loaded {len(df)} rows from {csv_path}")
        dfs.append(df)

    # 2) Raw .txt dumps in the <***> format (no pairing information)
    for txt_path in Path(input_dir).glob("**/*.txt"):
        label = txt_label if txt_label is not None else infer_label_from_filename(txt_path)
        if label is None:
            logger.warning(
                f"Skipping {txt_path.name}: cannot infer label from filename. "
                "Rename to e.g. 'avaz_human.txt' / 'avaz_ai.txt' or pass --txt_label."
            )
            continue
        articles = parse_dump_file(txt_path)
        df = build_dataframe(articles, label=label)
        logger.info(f"Parsed {len(df)} articles from {txt_path.name} (label={label})")
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"No CSV or TXT files found in {input_dir}.\n"
            "Provide CSVs with columns text,label,source,year or raw <***> dumps."
        )

    combined = pd.concat(dfs, ignore_index=True)

    # Keep only the columns the pipeline needs (extra columns are fine but not required)
    required = {"text", "label"}
    missing = required - set(combined.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info(f"Total rows loaded: {len(combined)}")
    return combined


# ---------------------------------------------------------------------------
# Main preparation pipeline
# ---------------------------------------------------------------------------
def prepare_dataset(
    input_dir: str,
    output_dir: str,
    min_length: int = 200,
    max_length: int = 50_000,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
    txt_label: Optional[int] = None,
) -> dict:
    """
    Full preparation pipeline:
    1. Load raw CSVs and/or .txt dumps
    2. Clean texts
    3. Filter by length
    4. Deduplicate
    5. Keep complete human–AI pairs (this also balances the classes)
    6. Split on pair_id (original and rewrite always in the same split)
    7. Save train/val/test CSVs
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    test_ratio = 1.0 - train_ratio - val_ratio

    # 1. Load
    df = load_raw_articles(input_dir, txt_label=txt_label)
    df["label"] = df["label"].astype(int)

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

    n_human = (df["label"] == 0).sum()
    n_ai    = (df["label"] == 1).sum()
    logger.info(f"Class distribution before pairing: human={n_human}, AI={n_ai}")

    if n_human == 0 or n_ai == 0:
        raise ValueError(
            f"Both classes are required (human={n_human}, AI={n_ai}). "
            "Generate AI rewrites with rewrite_with_chatgpt() or add label=1 data."
        )

    paired = "pair_id" in df.columns and df["pair_id"].notna().any()

    if paired:
        # 5. Keep only complete pairs: exactly one human + one AI per pair_id.
        #    Pairs broken by the length filter or dedup are dropped entirely.
        df = df.dropna(subset=["pair_id"]).copy()
        df["pair_id"] = df["pair_id"].astype(int)
        df = df.drop_duplicates(subset=["pair_id", "label"])
        labels_per_pair = df.groupby("pair_id")["label"].nunique()
        complete_ids = np.array(sorted(labels_per_pair[labels_per_pair == 2].index))
        before = len(df)
        df = df[df["pair_id"].isin(complete_ids)]
        logger.info(
            f"Pairing: kept {len(df)}/{before} articles "
            f"({len(complete_ids)} complete human–AI pairs)"
        )
        if len(complete_ids) < 10:
            raise ValueError(
                f"Only {len(complete_ids)} complete pairs found — pair_id linking is "
                "probably wrong. Check that src_idx in the AI CSV matches row indices "
                "in the human CSV."
            )

        # 6. Group split on pair_id. Every pair has one human + one AI article,
        #    so each split is automatically balanced.
        train_ids, temp_ids = train_test_split(
            complete_ids, test_size=(1.0 - train_ratio), random_state=seed
        )
        val_ids, test_ids = train_test_split(
            temp_ids, test_size=test_ratio / (val_ratio + test_ratio), random_state=seed
        )
        train_df = df[df["pair_id"].isin(train_ids)].sample(frac=1, random_state=seed)
        val_df   = df[df["pair_id"].isin(val_ids)].sample(frac=1, random_state=seed)
        test_df  = df[df["pair_id"].isin(test_ids)].sample(frac=1, random_state=seed)

        # Sanity check: no pair crosses a split boundary
        tr, va, te = set(train_df["pair_id"]), set(val_df["pair_id"]), set(test_df["pair_id"])
        assert not (tr & va) and not (tr & te) and not (va & te), "Pair leakage between splits!"
        logger.info("Group split on pair_id: no pair crosses split boundaries")

    else:
        logger.warning(
            "No pair_id information — falling back to independent class balancing and "
            "a row-level stratified split. Originals and their rewrites may end up in "
            "different splits (leakage)."
        )
        n_min = min(n_human, n_ai)
        df_human = df[df["label"] == 0].sample(n_min, random_state=seed)
        df_ai    = df[df["label"] == 1].sample(n_min, random_state=seed)
        df = pd.concat([df_human, df_ai], ignore_index=True).sample(frac=1, random_state=seed)
        logger.info(f"After balancing: {len(df)} articles ({n_min} per class)")

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
        "total": len(train_df) + len(val_df) + len(test_df),
        "grouped_by_pair": bool(paired),
        "train": len(train_df),
        "val":   len(val_df),
        "test":  len(test_df),
        "class_distribution": {
            "train": {int(k): int(v) for k, v in train_df["label"].value_counts().items()},
            "val":   {int(k): int(v) for k, v in val_df["label"].value_counts().items()},
            "test":  {int(k): int(v) for k, v in test_df["label"].value_counts().items()},
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
    parser.add_argument("--input_dir",  default="data/raw",       help="Directory with raw CSV/.txt files")
    parser.add_argument("--output_dir", default="data/processed",  help="Where to save processed splits")
    parser.add_argument("--min_length", type=int, default=200)
    parser.add_argument("--max_length", type=int, default=50_000)
    parser.add_argument("--train_ratio", type=float, default=0.80)
    parser.add_argument("--val_ratio",   type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--txt_label", type=int, choices=[0, 1], default=None,
                        help="Force label for ALL .txt dumps (0=human, 1=AI). "
                             "If omitted, label is inferred from filename.")
    args = parser.parse_args()

    stats = prepare_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_length=args.min_length,
        max_length=args.max_length,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        txt_label=args.txt_label,
    )
    print("\nDataset statistics:")
    print(json.dumps(stats, indent=2))
