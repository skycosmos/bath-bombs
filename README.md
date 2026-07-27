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
# 1-3) Read + consolidate data → purify → count.
#      Writes output/filter_count/parsed.csv (inputs) + results.csv (generated columns)
.venv/bin/python code/filter_count/run_pipeline.py --labeling-sample

# 4) Manual review / label UI (writes output/label_check/manual_labels.csv)
.venv/bin/streamlit run code/label_check/label_ui.py
```

### 1. Data (`data.py`)
Loads the **Amazon scrape** (primary) and the **Keepa export** (second source)
from the folders in `config.paths`, joins them on `asin` (Keepa fields under a
`keepa_` prefix), and keeps the configured columns.

### 2. Purification (`purity.py`)
Runs on the **title** (strict mode). Every rule's words live in `purity.lexicon`;
a listing is **pure** only if a bath-bomb phrase leads, with nothing disqualifying
before it. First split on whether a bomb phrase exists, then a first-match ladder:

```
NO bomb phrase →  no-bath-bomb   (always — even a lone "Bath Salts")

bomb phrase    → craft_kit → bundle → substitute-before-bomb
                 → ingredient-before-bomb (ingredients) → pure
```

**How the exclusion rules are defined** — each class is a word set in
`purity.lexicon`:

| class | fires on | words (examples) |
|---|---|---|
| `bomb_positive` | the countable unit → **pure** | bath bomb / fizzy / burst / blaster, shower bomb, aromabomb |
| `craft_kit` | supplies to *make* bombs | DIY, make-your-own, making kit, mould |
| `bundle` | a companion product sold with it, or a hidden surprise | soap, candle, lotion, bubble bath, cauldron, necklace/ring/toy; "surprise … inside" |
| `substitute` | a non-bomb bath form appearing **before** a bomb phrase (a *lone* one with no bomb phrase → `no-bath-bomb`) | bath salts, powder, beads, shower steamer, bath melt, fizz tablet |
| `ingredients` | raw materials appearing **before** a bomb phrase | citric acid, baking soda |

**Two mechanics decide close cases:**
- **Bomb-gating** — `craft_kit` and `bundle` only exclude when a bomb phrase is
  present; a lone kit or companion with no bomb wording → `no-bath-bomb`.
- **First-word adjudication** — `substitute` and `ingredient` exclude only when
  they appear *before* the bomb phrase. A leading bomb phrase wins: *"Bath Salts &
  Bath Bomb"* → substitute, but *"Bath Bomb with Citric Acid"* → pure.

Only `pure` listings are counted; every other class is an `exclude_reason`
(craft_kit / bundle / substitute / ingredients / no-bath-bomb).

### 3. Counting (`counting.py`)
For pure listings, four text channels (title, size, bullets, description) are
scanned for count phrases (`counting.patterns`, tie-broken by `pattern_priority`):
`set_of`, `pack_of`, **`pack_mult`** ("2 Pack of 6" → 12), `x_bombs`, `x_weight`
("6 × 40g"), `count_near_bomb` (incl. `pk`/`pc`/`ct`/singular `piece`),
`n_bath_bomb` ("6 Luxury Bath Bombs"), spelled `number_words`, `case_pack` — plus
the catalog fields (`number_of_items`, Keepa `numberOfItems` / `packageQuantity`,
label counts).

Resolution is by **weighted consensus** (`source_weights`), not first-wins: every
candidate value > 1 casts a vote weighted by source reliability (title 5 … the
overcounting `keepa_package_quantity` / `label_unit_num` = 1); the most-weight
value wins. A candidate of **1 never votes**, so a catalog "Pack of 1" can never
override a real text N. A sanity cap (`max_count`) drops barcodes / model numbers;
no evidence → assume a single unit (flagged for review).

**Confidence (`count_confidence`)** is **agreement-aware**, not a fixed per-source
tag. It begins at the winning source's base reliability (`counting.confidence`:
title/bullets `high`, size/description/`number_of_items`/Keepa `numberOfItems`
`medium`, the overcounting `keepa_package_quantity`/`label_unit_num`/`unit_num`
`low`), then adjusts on the `low → medium → high` ladder:

| situation | effect |
|---|---|
| ≥2 sources report the winning value | **+1 level** (corroborated) |
| the winner comes from a lone weight-1 source | **−1 level** |
| agreement is **catalog-only** (no text channel on the winner) | capped at `medium` — catalog fields share the "Pack of 1" overcount bias, so their mutual agreement isn't independent; `high` requires a text channel |
| unresolved conflict (`needs_review`) | forced to `low` |
| explicit "1" with no multi-count / no evidence at all | `single_default` (medium) / `assumed_single` (low) |

Outputs `n_bomb_balls`, `count_source`, `count_confidence`, and two flags:
`count_conflict` (≥2 distinct values > 1) and `needs_review` (an unresolved
conflict, or a no-evidence single) — plus `seller_counts_pack_as_one`.

### 4. Review UI (`code/label_check/label_ui.py`, `labeling.py`)
Streamlit console to check/correct predictions: product image, evidence and
candidate-count panels, optional scraped-page render, and a label form. A
**"Review queue" filter** lets you label only the rows counting flagged
`needs_review` (or `count_conflict`). Labels save to
`output/label_check/manual_labels.csv` (one row per ASIN, latest wins). A
stratified `labeling_sample.csv` (`--labeling-sample`) seeds the queue.

## Output — two files, joined on `asin`

The pipeline writes its columns split by origin:

- **`parsed.csv`** — the *parsed inputs* only: every raw column loaded from the
  Amazon scrape + Keepa (`title`, `size`, `feature`, `keepa_*`, …). Nothing
  generated here.
- **`results.csv`** — the *newly generated* columns only (the tables below): the
  purity verdict, the candidate counts (`cand_*`), and the resolved count.

Both keyed on `asin` (join to recombine). **Purification** (pipeline 2) sets:

| column | meaning |
|---|---|
| `is_pure_bath_bomb` | `True` if a countable bath bomb, else `False` |
| `exclude_reason` | why excluded — `craft_kit` / `bundle` / `substitute` / `ingredients` / `no-bath-bomb` (empty when pure) |
| `purity_source` | which rule fired (`rule_positive`, `rule_no_bomb`, `rule_bundle`, `rule_substitute`, `rule_craft_kit`, `rule_ingredient`) |

**Counting** (pipeline 3) fills these for pure rows (empty for excluded):

| column | meaning |
|---|---|
| `n_bomb_balls` | the resolved unit count |
| `count_source` | winning signal (`title` / `bullets` / … / `keepa_number_of_items` / `single_default` / `assumed_single`) |
| `count_confidence` | `high` / `medium` / `low` |
| `count_conflict` | `True` if ≥2 sources reported different counts > 1 |
| `needs_review` | `True` for unresolved conflicts or no-evidence singles (drives the review-queue filter) |
| `seller_counts_pack_as_one` | `True` if a catalog field read 1 while text gave N (the "Pack of 1" quirk) |

Intermediate `cand_*` columns hold each source's raw candidate value and the
pattern that produced it.

## Config map (`config.yml`)

| Section | Controls |
|---------|----------|
| `paths` | Amazon / Keepa / HTML folders, output + label CSVs |
| `amazon_columns`, `keepa` | Which source columns to keep / join |
| `purity.lexicon` | Every word set per exclusion class |
| `counting.patterns` / `pattern_priority` | Count regexes + tie-break priority |
| `counting.number_words` | Spelled-out numbers ("six" → 6) |
| `counting.source_weights` / `confidence` | Per-source vote weight (consensus) + confidence |
| `counting.max_count` / `conflict_needs_review` | Sanity cap + conflict-review gate |
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
    parsed.csv                          # parsed inputs (raw Amazon + Keepa)
    results.csv                         # ← FINAL machine output (generated columns)
    labeling_sample.csv                 # review queue for the UI
  label_check/
    manual_labels.csv                   # ← FINAL manual output (human labels)
```

## Results (current run)

25,357 listings → **10,637 pure** · excluded 14,720 (no-bath-bomb 8,804 ·
bundle 4,503 · substitute 1,078 · craft_kit 314 · ingredients 21). Keepa matched
100% of ASINs.

Of the 10,637 pure rows, counting resolves **43% high** / 15% medium / 43% low
confidence. It flags `count_conflict` on 852 (8%) and `needs_review` on 4,242
(40% — no-evidence singles + unresolved conflicts) for the review queue.
