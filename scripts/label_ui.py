"""
Streamlit labeling UI for bath-bomb unit counts.

Launch:
  .venv/bin/streamlit run scripts/label_ui.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from count_bath_bombs.config import load_config
from count_bath_bombs.gold import GOLD_COLUMNS

st.set_page_config(page_title="Bath Bomb Labeler", layout="wide")


def _load_queue(cfg: dict) -> pd.DataFrame:
    sample_path = Path(cfg["paths"]["labeling_sample_csv"])
    pred_path = Path(cfg["paths"]["output_csv"])
    needs_path = Path(cfg["paths"].get("needs_llm_csv", ROOT / "output" / "needs_llm.csv"))

    source = st.sidebar.selectbox(
        "Queue source",
        ["labeling_sample", "needs_llm", "product_counts"],
        help="labeling_sample = stratified sheet; needs_llm = unsure/undecided rows",
    )
    path = {
        "labeling_sample": sample_path,
        "needs_llm": needs_path,
        "product_counts": pred_path,
    }[source]
    if not path.exists():
        st.error(f"Missing {path}. Run: python scripts/run_pipeline.py --skip-html --labeling-sample")
        st.stop()
    df = pd.read_csv(path, low_memory=False)
    if source == "product_counts" and len(df) > 2000:
        st.sidebar.warning("Full predictions are large; consider labeling_sample or needs_llm.")
    return df, source


def _load_gold(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        g = pd.read_csv(path)
        if len(g) == 0:
            return pd.DataFrame(columns=GOLD_COLUMNS)
        return g
    return pd.DataFrame(columns=GOLD_COLUMNS)


def _save_gold(path: Path, gold: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep stable column order
    cols = [c for c in GOLD_COLUMNS if c in gold.columns] + [
        c for c in gold.columns if c not in GOLD_COLUMNS
    ]
    gold[cols].drop_duplicates(subset=["asin"], keep="last").to_csv(path, index=False)


def main() -> None:
    cfg = load_config()
    gold_path = Path(cfg["paths"]["gold_csv"])

    st.title("Bath bomb unit labeler")
    st.caption("Type labels for verification. Saves to data/gold/gold_labels.csv")

    queue, source = _load_queue(cfg)
    gold = _load_gold(gold_path)
    labeled_asins = set(gold["asin"].astype(str)) if len(gold) else set()

    hide_labeled = st.sidebar.checkbox("Hide already labeled ASINs", value=True)
    only_needs_llm = st.sidebar.checkbox(
        "Only needs_llm rows",
        value=(source == "needs_llm"),
        disabled=(source == "needs_llm"),
    )

    work = queue.copy()
    work["asin"] = work["asin"].astype(str)
    if hide_labeled:
        work = work[~work["asin"].isin(labeled_asins)]
    if only_needs_llm and "needs_llm" in work.columns:
        work = work[work["needs_llm"].fillna(False)]

    st.sidebar.write(f"Queue size: **{len(work):,}**")
    st.sidebar.write(f"Gold labels: **{len(gold):,}** → `{gold_path}`")

    if work.empty:
        st.success("No rows left in this queue (or all labeled).")
        st.stop()

    if "idx" not in st.session_state:
        st.session_state.idx = 0
    st.session_state.idx = min(st.session_state.idx, len(work) - 1)

    col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 2])
    with col_nav1:
        if st.button("← Prev") and st.session_state.idx > 0:
            st.session_state.idx -= 1
            st.rerun()
    with col_nav2:
        if st.button("Next →") and st.session_state.idx < len(work) - 1:
            st.session_state.idx += 1
            st.rerun()
    with col_nav3:
        jump = st.number_input(
            "Row #",
            min_value=0,
            max_value=len(work) - 1,
            value=st.session_state.idx,
            step=1,
        )
        if jump != st.session_state.idx:
            st.session_state.idx = int(jump)
            st.rerun()

    row = work.iloc[st.session_state.idx]
    asin = str(row["asin"])

    left, right = st.columns([1.4, 1])
    with left:
        st.subheader(row.get("title") or "(no title)")
        st.write(f"**ASIN:** `{asin}`")
        if row.get("stratum"):
            st.write(f"**Stratum:** {row.get('stratum')}")
        meta = {
            "number_of_items": row.get("number_of_items"),
            "size": row.get("size"),
            "pred_pure": row.get("is_pure_bath_bomb"),
            "pred_n_balls": row.get("n_bomb_balls"),
            "count_source": row.get("count_source"),
            "count_confidence": row.get("count_confidence"),
            "exclude_reason": row.get("exclude_reason"),
            "seller_counts_pack_as_one": row.get("seller_counts_pack_as_one"),
            "needs_llm": row.get("needs_llm"),
        }
        st.json({k: (None if pd.isna(v) else v) for k, v in meta.items()})
        bullets = row.get("html_bullets") or row.get("feature")
        if isinstance(bullets, str) and bullets.strip():
            st.markdown("**Bullets / features**")
            st.write(bullets[:2000])

    with right:
        st.markdown("### Your label")
        existing = gold[gold["asin"].astype(str) == asin]
        def_pure = None
        def_n = None
        def_ex = ""
        def_notes = ""
        if len(existing):
            er = existing.iloc[-1]
            def_pure = er.get("is_pure_bath_bomb_gold")
            def_n = er.get("n_bomb_balls_gold")
            def_ex = str(er.get("exclude_reason_gold") or "")
            def_notes = str(er.get("notes") or "")

        pure_options = ["true", "false", "unsure"]
        pure_default = "unsure"
        if def_pure is not None and not (isinstance(def_pure, float) and pd.isna(def_pure)):
            s = str(def_pure).strip().lower()
            if s in {"1", "true", "yes"}:
                pure_default = "true"
            elif s in {"0", "false", "no"}:
                pure_default = "false"

        is_pure = st.radio("is_pure_bath_bomb", pure_options, index=pure_options.index(pure_default))
        n_default = 0
        if def_n is not None and not (isinstance(def_n, float) and pd.isna(def_n)):
            try:
                n_default = int(float(def_n))
            except ValueError:
                n_default = 0
        elif row.get("n_bomb_balls") is not None and not pd.isna(row.get("n_bomb_balls")):
            n_default = int(float(row.get("n_bomb_balls")))

        n_balls = st.number_input(
            "n_bomb_balls (0 = leave blank / N/A)",
            min_value=0,
            max_value=5000,
            value=n_default,
            step=1,
        )
        exclude = st.selectbox(
            "exclude_reason (if not pure)",
            ["", "kit", "mixed_set", "not_bath_bomb", "unclear"],
            index=["", "kit", "mixed_set", "not_bath_bomb", "unclear"].index(def_ex)
            if def_ex in {"", "kit", "mixed_set", "not_bath_bomb", "unclear"}
            else 0,
        )
        annotator = st.text_input("annotator", value=st.session_state.get("annotator", ""))
        st.session_state.annotator = annotator
        notes = st.text_area("notes", value=def_notes)

        if st.button("Save label", type="primary"):
            new = {
                "asin": asin,
                "stratum": row.get("stratum") if "stratum" in row else "",
                "is_pure_bath_bomb_gold": "" if is_pure == "unsure" else is_pure,
                "n_bomb_balls_gold": "" if n_balls == 0 and is_pure != "true" else (n_balls if n_balls > 0 else ""),
                "exclude_reason_gold": exclude,
                "notes": notes,
                "annotator": annotator,
            }
            # If pure=true and n_balls>0, store count; if pure=false clear count unless user set one
            if is_pure == "true" and n_balls == 0:
                st.warning("Pure=true but n_bomb_balls=0. Saving with blank count.")
                new["n_bomb_balls_gold"] = ""
            if is_pure == "false":
                new["n_bomb_balls_gold"] = ""

            gold2 = gold[gold["asin"].astype(str) != asin].copy() if len(gold) else gold.copy()
            gold2 = pd.concat([gold2, pd.DataFrame([new])], ignore_index=True)
            _save_gold(gold_path, gold2)
            st.success(f"Saved {asin}")
            if st.session_state.idx < len(work) - 1:
                st.session_state.idx += 1
            st.rerun()

    with st.expander("Recent gold labels"):
        st.dataframe(gold.tail(20), use_container_width=True)


if __name__ == "__main__":
    main()
