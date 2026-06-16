# Bosnian AI Text Detector — BERTić

Binary classifier that distinguishes **human-written** from **AI-generated/rewritten**
Bosnian news articles, using BERTić — the Electra transformer trained on 8 billion
tokens of Bosnian/Croatian/Montenegrin/Serbian text (Ljubešić & Lauc, 2021).

---

## Project structure

```
bosnian_ai_detector/
├── config.py              ← all hyperparameters in one place
├── data_preparation.py    ← clean, balance, split dataset
├── train.py               ← fine-tune BERTić (5 runs, report avg ± std)
├── predict.py             ← inference on new articles
├── requirements.txt
├── data/
│   ├── raw/               ← put your scraped CSVs here
│   │   ├── human_2005.csv
│   │   ├── human_2010.csv
│   │   └── ai_rewritten.csv
│   └── processed/         ← auto-generated train/val/test splits
└── output/
    └── bertic_bosnian_detector/
        ├── run_1/ … run_5/
        └── final_model/   ← best model across 5 runs
```

---

## Step-by-step guide

### STEP 0 — Install dependencies

```bash
pip install -r requirements.txt
```

---

### STEP 1 — Collect human-written articles (1995–2022)

Scrape Bosnian newspapers published **before ChatGPT** (November 2022).
Recommended sources:
- **Oslobođenje** (oslobodjenje.ba)
- **Dnevni avaz** (avaz.ba)
- **Klix** (klix.ba)

Save each batch as a CSV with these columns:

```
text,label,source,year
"Predsjednik je danas...",0,"oslobodjenje",2005
```

- `label = 0` → human-written
- Put all CSVs in `data/raw/`

> Tip: use newspaper3k or Scrapy. Articles from 1995–2022 are safe to label as
> human-written because LLMs were not yet in newsroom use.

---

### STEP 2 — Generate AI-rewritten versions

Use the `rewrite_with_chatgpt()` function in `data_preparation.py`:

```python
from data_preparation import rewrite_with_chatgpt
import os, pandas as pd

os.environ["OPENAI_API_KEY"] = "sk-..."

df = pd.read_csv("data/raw/human_articles.csv")
df["ai_text"] = df["text"].apply(
    lambda t: rewrite_with_chatgpt(t, newspaper_name="Oslobođenje")
)

# Save AI articles with label=1
ai_df = df[["ai_text", "source", "year"]].copy()
ai_df = ai_df.rename(columns={"ai_text": "text"})
ai_df["label"] = 1
ai_df.to_csv("data/raw/ai_rewritten.csv", index=False)
```

The Bosnian rewrite prompt in `data_preparation.py` (REWRITE_PROMPT_TEMPLATE)
mirrors the methodology of Ozdemir (2026) but adapted for Bosnian editorial style.

---

### STEP 3 — Prepare the dataset

```bash
python data_preparation.py \
  --input_dir  data/raw \
  --output_dir data/processed
```

This will:
- Clean texts (normalize whitespace, remove HTML, strip agency bylines)
- Filter by length (200–50 000 chars)
- Deduplicate
- Balance classes (equal human/AI)
- Stratified 80/10/10 split
- Save `train.csv`, `val.csv`, `test.csv`

---

### STEP 4 — Fine-tune BERTić

```bash
python train.py
```

Or with custom settings:
```bash
python train.py \
  --data_dir    data/processed \
  --output_dir  output/my_run \
  --num_epochs  6 \
  --batch_size  8 \
  --lr          2e-5 \
  --num_runs    5 \
  --fp16
```

**What happens:**
- Loads `CLASSLA/bcms-bertic` from HuggingFace (Electra discriminator)
- Adds a linear classification head on the [CLS] token
- Trains for up to 6 epochs with early stopping (patience=2, monitored by val F1)
- Runs 5 times with different seeds (as in BERTić paper evaluation protocol)
- Reports mean ± std across runs
- Copies the best model to `output/.../final_model`

**Why BERTić and not multilingual BERT?**
Both papers confirm this: language-specific models significantly outperform mBERT
on monolingual tasks. BERTić saw actual Bosnian/Croatian/Serbian text during
pre-training, making its representations far richer for this task.

---

### STEP 5 — Evaluate and predict

**Single article:**
```bash
python predict.py text \
  --model_dir output/bertic_bosnian_detector/final_model \
  --text "Predsjednik Federacije BiH izjavio je danas na konferenciji..."
```

**Batch prediction (CSV):**
```bash
python predict.py batch \
  --model_dir  output/bertic_bosnian_detector/final_model \
  --input_csv  data/external/2024_articles.csv \
  --output_csv results/2024_predictions.csv
```

**Yearly analysis (external validation):**
```bash
python predict.py yearly \
  --model_dir output/bertic_bosnian_detector/final_model \
  --input_dir data/external/ \
  --output_dir results/yearly/
```

CSV files in `data/external/` should be named `2023_articles.csv`,
`2024_articles.csv`, etc., each with a `text` column and optionally `source`.

---

## Key differences from the Turkish paper (Ozdemir 2026)

| Aspect | Turkish paper | This project |
|---|---|---|
| Base model | dbmdz/bert-base-turkish-cased (BERT) | CLASSLA/bcms-bertic (Electra) |
| Architecture | BERT encoder | Electra discriminator |
| Language | Turkish | Bosnian (BCMS macro-language) |
| Training data | 35 GB Turkish text | 8B+ BCMS tokens |
| Vocab | 32k WordPiece Turkish | 32k WordPiece BCMS (diacritics preserved) |
| Diacritics | dotted/undotted ı/İ | č, ć, š, ž, đ |

## Bosnian-specific notes

- BERTić preserves **all Unicode** including Bosnian diacritics (č, ć, š, ž, đ).
  Do NOT lowercase or strip diacritics — this degrades the model.
- The BCMS macro-language means BERTić also understands Croatian and Serbian text,
  so mixed-language corpora are handled naturally.
- If your newspaper archive uses both Latin and Cyrillic script, you can either
  transliterate Cyrillic to Latin before tokenising, or rely on BERTić's own
  training corpus which contains both scripts.

---

## Expected performance

Based on the Turkish paper results and BERTić's generally superior performance
over mBERT/cseBERT on BCMS tasks, you can expect:

- **F1 ≥ 0.95** on the held-out test set with a clean, balanced dataset
- High confidence (>95%) on in-distribution articles
- ~2–5% AI detection rate on post-2023 unseen articles (depending on actual LLM
  adoption by Bosnian outlets)
