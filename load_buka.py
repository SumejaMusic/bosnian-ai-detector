"""
Loader for the buka.ba dataset.

Expected Google Drive structure:

    buka.ba/
    ├── BiH/
    │   ├── BiHSviClanciPrviDio.txt
    │   ├── BiHSviClanciDrugiDio.txt
    │   └── ...
    ├── Ekonomija/
    │   ├── EkonomijaSviClanciPrviDio.txt
    │   └── ...
    ├── Intervju/
    ├── Karikature i stripovi/
    └── ...

Each .txt file contains multiple articles separated by <***> with the
metadata header (NOVINA, DATUM, NASLOV, ...).

Output: one CSV per run, compatible with data_preparation.py:
    text, label, source, year, category, title, date

Usage (local / Colab after mounting Drive):
    python load_buka.py --input_dir "/content/drive/MyDrive/buka.ba" \
                        --output data/raw/buka_human.csv --label 0

    # Exclude non-news categories:
    python load_buka.py --input_dir "/content/drive/MyDrive/buka.ba" \
                        --output data/raw/buka_human.csv --label 0 \
                        --exclude "Karikature i stripovi"
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from parse_raw_articles import parse_dump_file, build_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Categories that are not regular news prose — excluded by default because
# they would teach the model genre differences instead of AI-vs-human signal
DEFAULT_EXCLUDE = {"Karikature i stripovi"}


def load_buka_dataset(
    input_dir: str,
    label: int = 0,
    exclude_categories: set[str] | None = None,
    include_title: bool = False,
) -> pd.DataFrame:
    """
    Walk category subfolders of the buka.ba dataset and parse every .txt dump.
    Returns a single DataFrame with a 'category' column added.
    """
    base = Path(input_dir)
    if not base.exists():
        raise FileNotFoundError(
            f"Directory not found: {base}\n"
            "In Colab: right-click the shared 'buka.ba' folder in Google Drive → "
            "'Organize' → 'Add shortcut' → My Drive, then mount Drive and use "
            "/content/drive/MyDrive/buka.ba"
        )

    exclude = exclude_categories if exclude_categories is not None else DEFAULT_EXCLUDE
    category_dirs = sorted(d for d in base.iterdir() if d.is_dir())

    if not category_dirs:
        # Maybe the txt files are directly in the base folder
        category_dirs = [base]

    all_dfs = []
    for cat_dir in category_dirs:
        category = cat_dir.name
        if category in exclude:
            logger.info(f"Skipping excluded category: {category}")
            continue

        txt_files = sorted(cat_dir.glob("*.txt"))
        if not txt_files:
            logger.warning(f"No .txt files in category '{category}' — skipping")
            continue

        cat_articles = []
        for txt in txt_files:
            cat_articles.extend(parse_dump_file(txt))

        if not cat_articles:
            logger.warning(f"Category '{category}': 0 articles parsed")
            continue

        df = build_dataframe(cat_articles, label=label, include_title=include_title)
        df["category"] = category
        # If NOVINA is missing in headers, fall back to 'buka_ba'
        df.loc[df["source"].isin(["", "unknown"]), "source"] = "buka_ba"
        all_dfs.append(df)
        logger.info(f"Category '{category}': {len(df)} articles from {len(txt_files)} files")

    if not all_dfs:
        raise FileNotFoundError(f"No articles parsed from {base}")

    combined = pd.concat(all_dfs, ignore_index=True)

    logger.info(f"\nTotal: {len(combined)} articles")
    logger.info(f"Per category:\n{combined['category'].value_counts().to_string()}")
    if combined["year"].notna().any():
        logger.info(f"Year range: {int(combined['year'].min())}–{int(combined['year'].max())}")
        n_ai_era = (combined["year"] >= 2023).sum()
        if label == 0 and n_ai_era:
            logger.warning(
                f"⚠️  {n_ai_era}/{len(combined)} articles are from 2023 or later. "
                "For the HUMAN training class use pre-2023 articles only; "
                "keep 2023+ for external validation with predict.py. "
                "Use --max_year 2022 to filter automatically."
            )
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load buka.ba dataset into a single CSV")
    parser.add_argument("--input_dir", required=True,
                        help="Path to the buka.ba folder (e.g. /content/drive/MyDrive/buka.ba)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--label", type=int, choices=[0, 1], default=0,
                        help="0 = human-written (default), 1 = AI-generated")
    parser.add_argument("--exclude", nargs="*", default=None,
                        help="Category folder names to skip "
                             "(default: 'Karikature i stripovi')")
    parser.add_argument("--include_title", action="store_true",
                        help="Prepend NASLOV to article text")
    parser.add_argument("--max_year", type=int, default=None,
                        help="Keep only articles with year <= max_year "
                             "(e.g. 2022 for a clean pre-AI human class)")
    parser.add_argument("--min_year", type=int, default=None,
                        help="Keep only articles with year >= min_year")
    args = parser.parse_args()

    exclude = set(args.exclude) if args.exclude is not None else None
    df = load_buka_dataset(
        args.input_dir,
        label=args.label,
        exclude_categories=exclude,
        include_title=args.include_title,
    )

    if args.max_year is not None:
        before = len(df)
        df = df[df["year"].notna() & (df["year"] <= args.max_year)]
        logger.info(f"Year filter (<= {args.max_year}): kept {len(df)}/{before}")
    if args.min_year is not None:
        before = len(df)
        df = df[df["year"].notna() & (df["year"] >= args.min_year)]
        logger.info(f"Year filter (>= {args.min_year}): kept {len(df)}/{before}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"Saved {len(df)} articles → {out}")
