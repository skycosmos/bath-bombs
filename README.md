# Count Bath Bombs

For each Amazon bath-bomb listing, decide whether it is a **countable bath bomb**
and, if so, **how many single units** it contains (`n_bomb_balls`). Rules-first,
fully config-driven, with a manual review UI.

Everything tunable — data paths, word lists, count patterns, priorities — lives
in **`config.yml`** (repo root). Nothing is hardcoded in the Python.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Pipelines

```bash
# 1-3) Read + consolidate data → purify → count. Writes output/filter_count/product_counts.csv
.venv/bin/python code/filter_count/run_pipeline.py --labeling-sample

# 4) Manual review / label UI (writes output/label_check/manual_labels.csv)
.venv/bin/streamlit run code/label_check/label_ui.py
```

### 1. Data (`data.py`)
Loads the **Amazon scrape** (primary) and the **Keepa export** (second source)
from the folders in `config.paths`, joins them on `asin` (Keepa fields under a
`keepa_` prefix), and keeps the configured columns.

### 2. Purification (`purity.py`)
Classifies each title against config word lists (`purity.lexicon`). Ladder,
first match wins:

```
craft_kit → bundle → substitute → (no bomb) unclassified
  → (ingredient before bomb) unclassified → pure
```

A listing is **pure** only if a bath-bomb phrase leads the title with nothing
disqualifying before it. Anything without bomb wording up front is excluded as
`unclassified`. Only `pure` listings are counted; every other class is an
`exclude_reason` (craft_kit / bundle / substitute / unclassified).

### 3. Counting (`counting.py`)
For pure listings, scans the text channels for count phrases (`counting.patterns`,
tie-broken by `pattern_priority`), then walks `resolution_order` — the first
signal yielding a value > 1 wins:

```
text (title>bullets>size>description) → number_of_items
  → keepa_number_of_items → keepa_package_quantity → label_unit_num
```

No count evidence → assume a single unit (low confidence). Outputs
`n_bomb_balls`, `count_source`, `count_confidence`.

### 4. Review UI (`code/label_check/label_ui.py`, `labeling.py`)
Streamlit console to check/correct predictions: product image, evidence and
candidate-count panels, optional scraped-page render, and a label form. Labels
save to `output/label_check/manual_labels.csv` (one row per ASIN, latest wins).
A stratified `labeling_sample.csv` (`--labeling-sample`) seeds the queue.

## Config map (`config.yml`)

| Section | Controls |
|---------|----------|
| `paths` | Amazon / Keepa / HTML folders, output + label CSVs |
| `amazon_columns`, `keepa` | Which source columns to keep / join |
| `purity.lexicon` | Every word set per exclusion class |
| `counting.patterns` / `pattern_priority` | Count regexes + tie-break priority |
| `counting.resolution_order` / `confidence` | Which signal wins, and its confidence |
| `labeling` | Sample size / seed, class + count label vocabularies |

## Layout

Both `code/` and `output/` split into the same two areas — **`filter_count`**
(the automated pipeline) and **`label_check`** (manual review):

```
config.yml                              # single source of truth
code/
  filter_count/                         # machine: read → filter/purify → count
    config.py  data.py  purity.py  counting.py  pipeline.py  run_pipeline.py
  label_check/                          # manual: label + check
    labeling.py  label_ui.py
output/
  filter_count/
    product_counts.csv                  # ← FINAL machine output (predictions)
    labeling_sample.csv                 # review queue for the UI
  label_check/
    manual_labels.csv                   # ← FINAL manual output (human labels)
```

## Results (current run)

25,357 listings → **11,645 pure** · excluded 13,712 (unclassified 6,763 ·
bundle 3,474 · substitute 2,482 · toiletry 629 · craft_kit 364). Keepa matched
100% of ASINs.
