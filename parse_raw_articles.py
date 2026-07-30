"""
Parser for raw Bosnian newspaper dumps.

Input format (articles separated by <***>):

    <***>
    NOVINA: Dnevni Avaz
    DATUM: 06.05.2025
    RUBRIKA: N/A
    NADNASLOV: POJAČANA KONTROLA
    NASLOV: U Tuzlanskom kantonu oduzeto 46 motornih vozila...
    PODNASLOV: POJAČANA KONTROLA
    STRANA: N/A
    AUTOR(I): N/A

    Policijski službenici Uprave policije MUP-a Tuzlanskog kantona su...

Output: CSV with columns compatible with data_preparation.py:
    text, label, source, year, title, date

Usage:
    # Human-written articles (label 0)
    python parse_raw_articles.py --input data/raw_txt/avaz_dump.txt \
                                 --output data/raw/avaz_human.csv --label 0

    # AI-rewritten articles in the same format (label 1)
    python parse_raw_articles.py --input data/raw_txt/avaz_ai.txt \
                                 --output data/raw/avaz_ai.csv --label 1

    # Parse a whole directory of .txt dumps
    python parse_raw_articles.py --input data/raw_txt/ \
                                 --output data/raw/human_articles.csv --label 0
"""

import re
import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Delimiter between articles in the dump
ARTICLE_DELIMITER = re.compile(r"^\s*<\*{3}>\s*$", re.MULTILINE)

# Metadata fields that appear in the header block
METADATA_KEYS = {
    "NOVINA", "DATUM", "RUBRIKA", "NADNASLOV", "NASLOV",
    "PODNASLOV", "STRANA", "AUTOR(I)", "AUTORI", "AUTOR",
}

# A metadata line looks like  "KEY: value"
METADATA_LINE = re.compile(r"^([A-ZČĆŽŠĐ()/ ]+?):\s*(.*)$")

# N/A-like empty values
NA_VALUES = {"", "N/A", "NA", "-", "n/a"}


def slugify_source(name: str) -> str:
    """'Dnevni Avaz' -> 'dnevni_avaz'"""
    name = name.strip().lower()
    # Replace Bosnian diacritics for a clean ASCII slug
    trans = str.maketrans("čćžšđ", "cczsd")
    name = name.translate(trans)
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return name or "unknown"


def parse_year(datum: str) -> int | None:
    """Extract a 4-digit year from DATUM (e.g. '06.05.2025')."""
    if not datum:
        return None
    m = re.search(r"(19|20)\d{2}", datum)
    return int(m.group(0)) if m else None


def parse_article_block(block: str) -> dict | None:
    """
    Parse one article block into {metadata..., 'text': body}.
    Returns None if the block has no usable body text.
    """
    lines = block.strip().splitlines()
    if not lines:
        return None

    meta: dict[str, str] = {}
    body_start = 0

    # Consume metadata lines from the top; the header ends at the first
    # line that is not a KEY: value pair (blank lines inside the header are skipped)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            body_start = i + 1
            continue
        m = METADATA_LINE.match(stripped)
        if m and m.group(1).strip() in METADATA_KEYS:
            key = m.group(1).strip()
            val = m.group(2).strip()
            meta[key] = "" if val in NA_VALUES else val
            body_start = i + 1
        else:
            body_start = i
            break

    body = "\n".join(lines[body_start:]).strip()
    if not body:
        return None

    return {
        "source_name": meta.get("NOVINA", ""),
        "date":        meta.get("DATUM", ""),
        "section":     meta.get("RUBRIKA", ""),
        "supertitle":  meta.get("NADNASLOV", ""),
        "title":       meta.get("NASLOV", ""),
        "subtitle":    meta.get("PODNASLOV", ""),
        "author":      meta.get("AUTOR(I)", meta.get("AUTORI", meta.get("AUTOR", ""))),
        "text":        body,
    }


def parse_dump_file(path: Path) -> list[dict]:
    """Parse one .txt dump file into a list of article dicts."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    blocks = ARTICLE_DELIMITER.split(raw)
    articles = []
    for block in blocks:
        parsed = parse_article_block(block)
        if parsed:
            articles.append(parsed)
    logger.info(f"{path.name}: parsed {len(articles)} articles from {len(blocks)} blocks")
    return articles


def build_dataframe(
    articles: list[dict],
    label: int,
    include_title: bool = False,
) -> pd.DataFrame:
    """
    Convert parsed articles to the CSV schema expected by data_preparation.py:
        text, label, source, year  (+ title, date kept for reference)

    include_title=False by default: following the Turkish paper's methodology,
    headlines are excluded because they can distort the narrative and leak
    formatting patterns instead of writing-style signal.
    """
    rows = []
    for a in articles:
        text = a["text"]
        if include_title and a["title"]:
            text = a["title"] + ". " + text

        rows.append({
            "text":   text,
            "label":  label,
            "source": slugify_source(a["source_name"]),
            "year":   parse_year(a["date"]),
            "title":  a["title"],
            "date":   a["date"],
        })

    df = pd.DataFrame(rows)
    return df


def parse_input(input_path: str, label: int, include_title: bool) -> pd.DataFrame:
    """Parse a single .txt file or every .txt file in a directory."""
    p = Path(input_path)
    if p.is_dir():
        files = sorted(p.glob("**/*.txt"))
        if not files:
            raise FileNotFoundError(f"No .txt files found in {p}")
    else:
        files = [p]

    all_articles = []
    for f in files:
        all_articles.extend(parse_dump_file(f))

    df = build_dataframe(all_articles, label=label, include_title=include_title)

    # Warn if "human" articles are from the AI era — they can't be trusted
    # as clean human-written training data (ChatGPT released Nov 2022)
    if label == 0 and df["year"].notna().any():
        ai_era = df[df["year"] >= 2023]
        if len(ai_era):
            logger.warning(
                f" {len(ai_era)}/{len(df)} articles labeled HUMAN are from year >= 2023. "
                "These may already contain AI-assisted writing and can contaminate "
                "the human class. Consider using pre-2023 articles for training, "
                "and keeping 2023+ articles for external validation (predict.py) instead."
            )

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse raw newspaper dump into training CSV")
    parser.add_argument("--input",  required=True, help=".txt file or directory of .txt dumps")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--label",  type=int, required=True, choices=[0, 1],
                        help="0 = human-written, 1 = AI-generated/rewritten")
    parser.add_argument("--include_title", action="store_true",
                        help="Prepend NASLOV to the article text (default: body only)")
    args = parser.parse_args()

    df = parse_input(args.input, label=args.label, include_title=args.include_title)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    logger.info(f"Saved {len(df)} articles → {out}")
    print(f"\nSources: {df['source'].value_counts().to_dict()}")
    print(f"Years:   {df['year'].value_counts().sort_index().to_dict()}")
    print(f"Label:   {args.label} ({'human' if args.label == 0 else 'AI'})")
