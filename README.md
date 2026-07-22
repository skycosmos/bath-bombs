# Count Bath Bombs

Onboarding task: for each Amazon listing, decide whether it is a **pure finished
bath-bomb product** and, if so, count how many **single bomb units** it contains
(`n_bomb_balls`). Rules-only (title/text + catalog + HTML + Keepa), with a human
review/label loop for building and scoring gold.

## Pipelines at a glance

There are two pipelines: one that **produces counts**, one that **labels &
evaluates** them.

### 1. Counting pipeline — `scripts/run_pipeline.py` → `count_bath_bombs/pipeline.py`

Runs over all ~25k listings in stages, each adding columns to the frame:

| Stage | Module | What it does |
|-------|--------|--------------|
| Load | `pipeline.load_products` | Read the input CSV, keep the relevant columns |
| HTML extract | `html_extract.py` | Parse the scraped Amazon HTML per ASIN (title, bullets, description, "Number of Items"/"Unit Count"/"Package Quantity", size, weight). Cached per ASIN. Skippable with `--skip-html` |
| Keepa (2nd source) | `keepa.py` | Join the Keepa product export on `asin` → `keepa_*` fields (numberOfItems, packageQuantity, features, description, images, variations). Used as a **fallback / second signal** for counts and purity. Toggle with `keepa.enabled`; degrades to null columns if the source is unavailable |
| Purity | `purity.py` | Regex rules on the **title** (strict mode) → `is_pure_bath_bomb` ∈ {True, False, None}. Excludes kits/DIY/molds, mixed gift sets, non-bombs (salts/tablets/melts). Scope config toggles shower bombs / melts / fizz tablets |
| Candidates | `counts.py` | Extract every possible count from title/size/bullets/description + catalog fields ("Set of 6", "12 bombs", "3 x 5oz"), each tagged with the pattern that matched |
| Resolve | `resolve.py` | Pick the final `n_bomb_balls` via a priority ladder (text multi-count → number_of_items → Keepa numberOfItems → HTML details → single default), set `count_confidence`/`count_source`, mark `count_unable` when no number can be justified |

Outputs `output/product_counts.csv` (and an optional stratified
`labeling_sample.csv`).

### 2. Labeling & evaluation pipeline — `label_ui.py` · `seed_labels.py` · `eval_gold.py`

Turns human review into a scored gold set (see **Labeling / gold model** below):
review in the Streamlit console → labels land in `annotations.csv` → adjudicated
into `gold_labels.csv` → `eval_gold.py` scores predictions against the held-out
eval split.

## Counting SOP (annotated with live volumes)

The decision procedure, with each rule's share of the current **25,357-listing**
run. Purity branches are % of all listings; count branches are % of the **11,894
pure** products. First matching rule wins at every stage.

### Stage 1 — Purity: is this a countable bath bomb? (`purity.py`)

Judged from the **title** in strict mode. A six-rung ladder of mutually-exclusive
detectors; first match wins.

```mermaid
flowchart TD
  T["Title (strict mode)"] --> K{"CRAFT_KIT?<br/>kit, DIY, mould, baking soda, citric acid"}
  K -->|"yes · 556 (2.2%)"| EK["EXCLUDE: craft_kit"]
  K -->|no| B{"BUNDLE?<br/>bomb phrase AND (soap/candle/lotion/toy/<br/>necklace/ring, or surprise+inside)"}
  B -->|"yes · 3,439 (13.6%)"| EB["EXCLUDE: bundle"]
  B -->|no| SUB{"SUBSTITUTE first?<br/>shower steamer/melt/tablet/salt/powder/<br/>beads — wins if it precedes any bomb word"}
  SUB -->|"yes · 2,424 (9.6%)"| ESU["EXCLUDE: substitute"]
  SUB -->|no| TL{"TOILETRY?<br/>soap/shampoo/lotion/candle/body wash/…<br/>(only reachable with no bomb phrase)"}
  TL -->|"yes · 272 (1.1%)"| ETL["EXCLUDE: toiletry"]
  TL -->|no| P{"Bomb phrase present?"}
  P -->|"yes · 11,894 (46.9%)"| PURE["PURE — count it"]
  P -->|"no · 6,772 (26.7%)"| U["UNCLASSIFIED — flag for review"]
```

| Rule (`purity_source`) | `exclude_reason` | Fired | Share |
|---|---|---|---|
| `rule_positive` → PURE | — | 11,894 | 46.9% |
| `rule_unclassified` | `unclassified` | 6,772 | 26.7% |
| `rule_bundle` | `bundle` | 3,439 | 13.6% |
| `rule_substitute` | `substitute` | 2,424 | 9.6% |
| `rule_craft_kit` | `craft_kit` | 556 | 2.2% |
| `rule_toiletry` | `toiletry` | 272 | 1.1% |

**Gating modes:** CRAFT_KIT and TOILETRY exclude on any match; BUNDLE requires a
bomb phrase **and** a non-bomb item to co-occur (a bath bomb sold *with* something
else); SUBSTITUTE uses **first-word adjudication** — a shower-steamer/salt/tablet
term excludes only if it appears *before* any bomb phrase in the title
("Bath Bomb & Shower Steamer" stays PURE; "Shower Steamer, Bath Bomb Sampler" is
excluded). Substitute sub-families are gated by `scope.include_*` flags.

### Stage 2 — Detect candidate numbers (`counts.py`)

Five regexes scan four text fields (title, size, bullets, description) separately;
each field keeps **one** number — lowest-priority pattern, preferring values >1.
Catalog integers (`number_of_items`, HTML unit/package counts) are read as-is.

| Priority | Pattern | Example | Times fired |
|---|---|---|---|
| 0 | `set_of` | "Set of 6" | 1,293 |
| 1 | `n_x_bombs` | "6 x 5oz bombs" | 212 |
| 2 | `near_bomb/ball/fizz` | "12 bath bombs" | ~370 |
| 3 | `pack_of` | "Pack of 8" | 1,174 |
| 4 | `near_pack` | "4 pack" | 3,333 |
| 5 | `near_pcs/pieces/count` | "24 pcs" | 4,058 |

> ⚠️ The high-volume patterns (`near_pack`, `near_pcs`, `near_count`) are the
> **weakest/most generic** — they match any number next to "pack/pcs/count",
> which is where scent-count and size false positives creep in.

### Stage 3 — Choose the number (`resolve.py`)

A precedence ladder; first tier with a value >1 wins. Shares are of the 16,025 pure.

```mermaid
flowchart TD
  S["Pure product"] --> L1{"Text multi-count?<br/>title &gt; bullets &gt; size &gt; description"}
  L1 -->|"yes · high/medium"| C1["use it"]
  L1 -->|no| L2{"number_of_items &gt; 1?"}
  L2 -->|"yes · medium"| C2["use it"]
  L2 -->|no| L2b{"keepa numberOfItems / packageQuantity &gt; 1?"}
  L2b -->|"yes · medium"| C2b["use it (keepa_number_of_items)"]
  L2b -->|no| L3{"HTML details &gt; 1?"}
  L3 -->|"yes · medium"| C3["use it"]
  L3 -->|no| L4{"any signal == 1?"}
  L4 -->|"yes · 2,442 (15.2%) · med/low"| C4["n = 1 (single_default)"]
  L4 -->|"no · 6,973 (43.5%) · low"| C5["n = 1 (assumed_single) ⚠"]
```

Flags set alongside the number:
- **`seller_counts_pack_as_one`** — text says a multi-count but a catalog
  field says 1. Recorded only; **does not change the count**.
- **`count_unable`** — no number could be justified (no count wording and no
  catalog signal); surfaces the row for human review.

**Where the volume actually goes** (pure products): `assumed_single` **43.5%** is
the single largest source — i.e. nearly half of all counts are the "no evidence →
assume 1" fallback, the rule the eval flagged as the main error driver. Only
**36.3%** come from an explicit text count; **15.2%** are `single_default`.
Overall confidence: low 61.8% · high 21.5% · medium 8.6% · n/a 8.0%.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Recommended workflow

```bash
# 1) Rules + HTML + Keepa. Assigns a count unless the rules are unable to.
.venv/bin/python scripts/run_pipeline.py --labeling-sample

# Smoke without HTML:
.venv/bin/python scripts/run_pipeline.py --skip-html --labeling-sample

# 2) Review + label in the browser (thumbnail, evidence, scraped page render)
.venv/bin/streamlit run scripts/label_ui.py

# 2b) (optional) Seed confident TRAIN-split rows to pre-fill the UI for fast confirm
.venv/bin/python scripts/seed_labels.py --dry-run   # preview count
.venv/bin/python scripts/seed_labels.py             # write model_seed annotations

# 3) Score vs gold — defaults to the held-out EVAL split (honest)
.venv/bin/python scripts/eval_gold.py --rebuild             # rebuild gold from annotations, score eval
.venv/bin/python scripts/eval_gold.py --split all --report  # full report: confusion, F1, by-confidence/stratum, errors
.venv/bin/python scripts/eval_gold.py --split all --report --out output/gold_metrics.json
```

## Labeling / gold model

Labels flow through a multi-annotator store so agreement can be measured and the
model is never scored on labels it produced itself:

- `data/gold/annotations.csv` — every raw label, one row per (ASIN, annotator),
  tagged `source` (`human` / `model_seed`) and `split` (`eval` / `train`).
- `data/gold/gold_labels.csv` — **derived**, one adjudicated row per ASIN
  (human-only majority vote, ties → most recent). Consumed by `eval_gold.py`.
- **Held-out eval split** (`gold.eval_frac`, deterministic per ASIN) is **never
  seeded** from predictions — so `eval_gold.py --split eval` is an honest score.
- The Streamlit **Dashboard** tab shows inter-annotator agreement (purity/count
  agreement + Cohen's κ), conflicts to adjudicate, and human-vs-model by split.
  Use the sidebar *"Only eval rows needing a 2nd label"* filter to build agreement.

## Results (current run)

Counting pipeline over **25,357 listings** (rules + HTML + Keepa):

| Metric | Value |
|--------|-------|
| Pure bath bombs (count assigned) | **11,894** (46.9%) |
| Excluded | 6,691 — bundle 3,439 · substitute 2,424 · craft_kit 556 · toiletry 272 |
| Unclassified (purity undecided) | 6,772 |
| Unable to count (`count_unable`) | ~3,858 |

> Purity was retightened: the reorganized ladder now excludes shower steamers,
> bath salts/tablets/melts (SUBSTITUTE), and bath-bomb-plus-item sets (BUNDLE) that
> the earlier rules counted — moving ~4,200 listings from PURE to excluded. The
> 50-label gold slice predates this change; re-label before trusting the metrics
> below.

Model evaluation vs **50 human labels** (`eval_gold.py --report`):

| Split | Purity P / R / F1 | Count exact / ±1 | Count MAE |
|-------|-------------------|------------------|-----------|
| `eval` (honest, held-out) | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 | 0.0 |
| `all` (incl. train) | 0.90 / 0.78 / 0.84 | 0.78 / 0.78 | 2.06 |

Purity confusion (all): TP 18 · FP 2 · **FN 5** · TN 12. Calibration is sane —
`high`-confidence rows score 100% purity accuracy, `low` ≈ 78%.

Key findings driving the next iteration:
- **Recall < precision (0.78 vs 0.90): the rules over-exclude.** 5 pure products
  were wrongly marked not-pure — a target for better title/support rules.
- **Count errors are single-vs-multipack:** every count miss is `pred=1` vs a real
  6–16, i.e. the `assumed_single` default undercounts packs with no count wording.
  The Keepa `numberOfItems` join recovers ~2,100 of these (see below).

> Gold is still small (50 labels; 6 held out), so treat the `eval` row as
> directional — the priority is to grow and double-label the eval slice.
> Full breakdown lives in `output/gold_metrics.json`.

## Config knobs

| Key | Default | Meaning |
|-----|---------|---------|
| `purity.strict` | `true` | Title-primary exclude/include (fewer false kit excludes from feature text) |
| `scope.include_shower_bombs` | `true` | Count shower bombs |
| `scope.include_bath_melts` / `include_fizz_tablets` | `false` | Out of scope |
| `keepa.enabled` | `true` | Join the Keepa export (`paths.keepa_csv`) as a second source. Off → null `keepa_*` columns |

### Two-source comparison (Keepa vs amazon_web_scraping)

Both sources cover the same **25,357 ASINs (perfect 1:1 join)**. They are
complementary, so Keepa is wired as a fallback rather than a replacement:

| Field | amazon_web_scraping | Keepa | Wiring |
|---|---|---|---|
| `number_of_items` | 2,493 present | **12,714 present** (99.8% agree where both) | Keepa fills the count ladder → **+2,154 products counted, +2,103 multi-packs** |
| images | `image_num` count only, 20,880 | **imagesCSV URLs, all rows** (97% non-empty) | `keepa_main_image_url` / `keepa_image_count`; UI thumbnail fallback |
| variations | 2,627 | **all rows** (`variationCSV`) | `keepa_variation_count` |
| features | 62% non-empty | **75% non-empty** | added to bullets text for count regexes |
| description | **93% non-empty** | 51% | scrape stays primary; Keepa fallback |
| title | — | 59% exact match (different snapshot) | Keepa title only as last-resort fallback |

## Outputs

| Path | Role |
|------|------|
| `output/product_counts.csv` | Predictions (`n_bomb_balls`, `count_confidence`, `count_unable`, `keepa_*`, …) |
| `output/labeling_sample.csv` | Stratified sheet feeding the labeler |
| `data/gold/annotations.csv` | Raw multi-annotator labels (human + model_seed, with split) |
| `data/gold/gold_labels.csv` | Adjudicated human-only gold, derived from annotations |
