"""
Multi-annotator label store + adjudication, inter-annotator agreement (IAA),
and model-prediction seeding.

Design
------
- `annotations.csv` is the source of truth. It may hold MANY rows per ASIN
  (one per annotator), each tagged with:
    source : "human" | "model_seed"
    split  : "eval" | "train"   (deterministic, hash-based, stable per ASIN)
- `gold_labels.csv` is DERIVED by adjudicating human-only annotations into one
  row per ASIN (majority vote, ties broken by most-recent). It stays
  backward-compatible with scripts/eval_gold.py and evaluate_against_gold().

Rules that keep evaluation honest
---------------------------------
- The held-out **eval** split is never seeded from model predictions
  (`seed_from_predictions(..., only_train=True)`), so the model is never scored
  on labels it produced itself.
- IAA is measured only over ASINs a human labeled at least twice.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ANNOTATION_COLUMNS = [
    "asin",
    "stratum",
    "split",
    "source",
    "is_pure_bath_bomb_gold",
    "n_bomb_balls_gold",
    "exclude_reason_gold",
    "notes",
    "annotator",
    "ts",
]


# --------------------------------------------------------------------------- #
# Coercion helpers
# --------------------------------------------------------------------------- #
def to_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def pure_str(value: Any) -> str | None:
    """Normalise a purity label to 'true' / 'false' / None (unsure/blank)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "1.0"}:
        return "true"
    if s in {"0", "false", "no", "n", "0.0"}:
        return "false"
    return None


def now_ts() -> float:
    return time.time()


# --------------------------------------------------------------------------- #
# Split assignment — deterministic, stable, order-independent
# --------------------------------------------------------------------------- #
def assign_split(asin: str, eval_frac: float = 0.2) -> str:
    """Hash the ASIN into a stable [eval|train] bucket. No RNG, no run-order dep."""
    if eval_frac <= 0:
        return "train"
    if eval_frac >= 1:
        return "eval"
    h = int(hashlib.sha1(str(asin).encode("utf-8")).hexdigest(), 16) % 10000
    return "eval" if h < int(eval_frac * 10000) else "train"


# --------------------------------------------------------------------------- #
# Load / save / upsert
# --------------------------------------------------------------------------- #
def load_annotations(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        df = pd.read_csv(p)
    else:
        df = pd.DataFrame(columns=ANNOTATION_COLUMNS)
    for c in ANNOTATION_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[ANNOTATION_COLUMNS].copy()


def save_annotations(path: str | Path, ann: pd.DataFrame) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = ann.copy()
    for c in ANNOTATION_COLUMNS:
        if c not in out.columns:
            out[c] = None
    out[ANNOTATION_COLUMNS].to_csv(path, index=False)


def upsert_annotation(ann: pd.DataFrame, record: dict) -> pd.DataFrame:
    """Insert/replace a single (asin, annotator) annotation."""
    asin = str(record["asin"])
    who = str(record.get("annotator") or "")
    if ann is not None and len(ann):
        keep = ~(
            (ann["asin"].astype(str) == asin)
            & (ann["annotator"].astype(str) == who)
        )
        ann = ann[keep].copy()
    row = {c: record.get(c) for c in ANNOTATION_COLUMNS}
    new = pd.DataFrame([row], columns=ANNOTATION_COLUMNS)
    if ann is None or len(ann) == 0:
        return new
    return pd.concat([ann, new], ignore_index=True)


def upsert_many(ann: pd.DataFrame, records: Iterable[dict]) -> pd.DataFrame:
    for rec in records:
        ann = upsert_annotation(ann, rec)
    return ann


# --------------------------------------------------------------------------- #
# Adjudication  → one gold row per ASIN
# --------------------------------------------------------------------------- #
def _latest_first(sub: pd.DataFrame) -> pd.DataFrame:
    s = sub.copy()
    s["_ts"] = pd.to_numeric(s["ts"], errors="coerce")
    return s.sort_values("_ts", ascending=False, na_position="last")


def _majority(values: list, latest_order: list):
    """Most common value; ties broken by whichever appears first in latest_order."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    counts = {v: vals.count(v) for v in set(vals)}
    top = max(counts.values())
    leaders = {v for v, c in counts.items() if c == top}
    if len(leaders) == 1:
        return next(iter(leaders))
    for v in latest_order:
        if v in leaders:
            return v
    return next(iter(leaders))


GOLD_OUT_COLUMNS = [
    "asin",
    "stratum",
    "split",
    "is_pure_bath_bomb_gold",
    "n_bomb_balls_gold",
    "exclude_reason_gold",
    "annotator",
    "n_annotations",
    "notes",
]


def adjudicate(ann: pd.DataFrame, humans_only: bool = True) -> pd.DataFrame:
    """Collapse annotations to one adjudicated gold row per ASIN."""
    if ann is None or len(ann) == 0:
        return pd.DataFrame(columns=GOLD_OUT_COLUMNS)

    rows: list[dict] = []
    for asin, g in ann.groupby(ann["asin"].astype(str)):
        sub = g[g["source"].astype(str) == "human"] if humans_only else g
        if sub.empty:
            continue
        sub = _latest_first(sub)

        pur_latest = [pure_str(v) for v in sub["is_pure_bath_bomb_gold"]]
        purity = _majority(pur_latest, pur_latest)

        n_val: Any = ""
        if purity == "true":
            n_latest = [to_int(v) for v in sub["n_bomb_balls_gold"]]
            n_maj = _majority(n_latest, n_latest)
            n_val = "" if n_maj is None else n_maj

        exr = ""
        if purity == "false":
            for v in sub["exclude_reason_gold"]:
                if isinstance(v, str) and v.strip():
                    exr = v.strip()
                    break

        first = sub.iloc[0]
        annotators = sorted({str(a) for a in sub["annotator"].dropna().unique()})
        rows.append(
            {
                "asin": asin,
                "stratum": first.get("stratum"),
                "split": first.get("split"),
                "is_pure_bath_bomb_gold": "" if purity is None else purity,
                "n_bomb_balls_gold": n_val,
                "exclude_reason_gold": exr,
                "annotator": ", ".join(annotators),
                "n_annotations": len(annotators),
                "notes": first.get("notes"),
            }
        )
    return pd.DataFrame(rows, columns=GOLD_OUT_COLUMNS)


def rebuild_gold(
    annotations_path: str | Path,
    gold_path: str | Path,
    humans_only: bool = True,
) -> pd.DataFrame:
    """Derive gold_labels.csv from annotations and write it to disk."""
    ann = load_annotations(annotations_path)
    gold = adjudicate(ann, humans_only=humans_only)
    Path(gold_path).parent.mkdir(parents=True, exist_ok=True)
    gold.to_csv(gold_path, index=False)
    return gold


# --------------------------------------------------------------------------- #
# Inter-annotator agreement
# --------------------------------------------------------------------------- #
def _cohens_kappa(pairs: list[tuple]) -> float | None:
    n = len(pairs)
    if n == 0:
        return None
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    cats = sorted(set(a) | set(b))
    po = sum(1 for x, y in pairs if x == y) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def compute_iaa(ann: pd.DataFrame) -> dict:
    """Agreement over ASINs labeled by ≥2 distinct human annotators."""
    res: dict = {
        "n_double_labeled": 0,
        "purity_agreement": None,
        "purity_kappa": None,
        "count_agreement": None,
        "conflicts": [],
    }
    if ann is None or len(ann) == 0:
        return res
    hum = ann[ann["source"].astype(str) == "human"]
    if hum.empty:
        return res

    pur_flags: list[int] = []
    cnt_flags: list[int] = []
    pairs: list[tuple] = []
    conflicts: list[dict] = []
    n_double = 0

    for asin, g in hum.groupby(hum["asin"].astype(str)):
        g = _latest_first(g.drop_duplicates(subset=["annotator"]))
        if g["annotator"].nunique() < 2:
            continue
        n_double += 1

        purs = [pure_str(v) for v in g["is_pure_bath_bomb_gold"]]
        pv = [p for p in purs if p in ("true", "false")]
        ns = [to_int(v) for v in g["n_bomb_balls_gold"]]
        nv = [x for x in ns if x is not None]

        p_agree = c_agree = None
        if len(pv) >= 2:
            p_agree = len(set(pv)) == 1
            pur_flags.append(1 if p_agree else 0)
            pairs.append((pv[0], pv[1]))
        if len(nv) >= 2:
            c_agree = len(set(nv)) == 1
            cnt_flags.append(1 if c_agree else 0)

        if p_agree is False or c_agree is False:
            conflicts.append(
                {
                    "asin": asin,
                    "annotators": ", ".join(g["annotator"].astype(str)),
                    "purity": ", ".join(str(x) for x in purs),
                    "counts": ", ".join(
                        "-" if to_int(v) is None else str(to_int(v))
                        for v in g["n_bomb_balls_gold"]
                    ),
                }
            )

    res["n_double_labeled"] = n_double
    if pur_flags:
        res["purity_agreement"] = sum(pur_flags) / len(pur_flags)
    if cnt_flags:
        res["count_agreement"] = sum(cnt_flags) / len(cnt_flags)
    res["purity_kappa"] = _cohens_kappa(pairs)
    res["conflicts"] = conflicts
    return res


# --------------------------------------------------------------------------- #
# Seeding from model predictions
# --------------------------------------------------------------------------- #
def seed_candidates(
    pred: pd.DataFrame,
    eval_frac: float = 0.2,
    confidences: Iterable[str] = ("high",),
    only_train: bool = True,
    exclude_needs_llm: bool = True,
) -> list[dict]:
    """Model-prediction annotations for confident rows (train split by default)."""
    conf_set = {str(c).lower() for c in confidences}
    recs: list[dict] = []
    for _, r in pred.iterrows():
        asin = str(r.get("asin"))
        split = assign_split(asin, eval_frac)
        if only_train and split != "train":
            continue
        if exclude_needs_llm and bool(r.get("needs_llm")):
            continue
        if str(r.get("count_confidence")).lower() not in conf_set:
            continue
        purity = pure_str(r.get("is_pure_bath_bomb"))
        if purity not in ("true", "false"):
            continue
        n = to_int(r.get("n_bomb_balls"))
        exr = r.get("exclude_reason")
        recs.append(
            {
                "asin": asin,
                "stratum": r.get("stratum"),
                "split": split,
                "source": "model_seed",
                "is_pure_bath_bomb_gold": purity,
                "n_bomb_balls_gold": n if (purity == "true" and n) else "",
                "exclude_reason_gold": exr if (purity == "false" and isinstance(exr, str)) else "",
                "notes": "seeded from model prediction",
                "annotator": "model_seed",
                "ts": None,
            }
        )
    return recs
