# Count Bath Bombs

Onboarding task: count how many single bath-bomb units are in each Amazon listing.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then paste OPENAI_API_KEY into .env (only needed for --enable-llm)
```

## Recommended workflow

```bash
# 1) Rules + HTML parse (LLM off by default). Assigns a count unless unable.
.venv/bin/python scripts/run_pipeline.py --labeling-sample

# Smoke without HTML:
.venv/bin/python scripts/run_pipeline.py --skip-html --labeling-sample

# 2) Manual labels (browser UI)
.venv/bin/streamlit run scripts/label_ui.py

# 3) Later: improve unsure/undecided rows only (needs_llm=True)
.venv/bin/python scripts/run_pipeline.py --enable-llm

# 4) Score vs gold
.venv/bin/python scripts/eval_gold.py
```

## Config knobs

| Key | Default | Meaning |
|-----|---------|---------|
| `purity.strict` | `true` | Title-primary exclude/include (fewer false kit excludes from feature text) |
| `scope.include_shower_bombs` | `true` | Count shower bombs |
| `scope.include_bath_melts` / `include_fizz_tablets` | `false` | Out of scope |
| `llm.enabled` | `false` | Stay off until you pass `--enable-llm` |

## Outputs

| Path | Role |
|------|------|
| `output/product_counts.csv` | Predictions (`n_bomb_balls`, `needs_llm`, …) |
| `output/needs_llm.csv` | Unsure / undecided rows for a later LLM pass |
| `output/labeling_sample.csv` | Stratified sheet feeding the labeler |
| `data/gold/gold_labels.csv` | Your manual labels (from the Streamlit UI) |
| `prompts/count_v1.md` | Frozen LLM prompt |
| `.env` | `OPENAI_API_KEY=...` (gitignored) |
