"""Pipeline 1 — read the data sources and consolidate one working frame.

Amazon scrape (primary) + Keepa export (second source), joined on `asin`.
Both sources are folders in the config; the newest *.csv inside each is used.
Keepa fields are surfaced under a `keepa_` prefix so the scrape stays primary
and Keepa acts as a fallback / second signal downstream.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pandas as pd

# Working columns attached from Keepa (kept stable whether or not Keepa loads).
KEEPA_FIELDS = [
    "keepa_title", "keepa_brand", "keepa_manufacturer",
    "keepa_number_of_items", "keepa_package_quantity", "keepa_item_weight_g",
    "keepa_size", "keepa_features", "keepa_description",
    "keepa_image_count", "keepa_main_image_url", "keepa_image_urls",
    "keepa_variation_count", "keepa_present",
]


def pick_csv(folder_or_file: str | Path) -> Path:
    """Return a CSV path: the file itself, or the newest-named *.csv in a dir."""
    p = Path(folder_or_file)
    if p.is_file():
        return p
    if p.is_dir():
        csvs = sorted(p.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No .csv in {p}")
        return csvs[-1]  # lexicographic max — dated filenames sort chronologically
    raise FileNotFoundError(folder_or_file)


# --------------------------------------------------------------------------- #
# Keepa field extraction
# --------------------------------------------------------------------------- #
def _pos_int(value: Any) -> int | None:
    """Keepa uses -1 / 0 as 'unknown' sentinels; keep only positive ints."""
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _clean_text(value: Any) -> str | None:
    s = str(value or "").strip()
    return s if s and s.lower() != "nan" else None


def _features_text(value: Any) -> str | None:
    """Keepa stores features as a Python-list literal, e.g. "['a', 'b']"."""
    s = str(value or "").strip()
    if not s or s in ("nan", "[]"):
        return None
    try:
        items = ast.literal_eval(s)
        if isinstance(items, (list, tuple)):
            return " | ".join(str(x).strip() for x in items if str(x).strip()) or None
    except (ValueError, SyntaxError):
        pass
    return s


def _count_csv(value: Any) -> int | None:
    s = str(value or "").strip()
    if not s or s.lower() == "nan":
        return None
    return len([x for x in s.split(",") if x]) or None


def _images(value: Any, prefix: str) -> tuple[int | None, str | None, str | None]:
    s = str(value or "").strip()
    if not s or s.lower() == "nan":
        return None, None, None
    tokens = [t for t in s.split(",") if t]
    if not tokens:
        return None, None, None
    urls = [prefix + t for t in tokens]
    return len(tokens), urls[0], " | ".join(urls)


def _keepa_row(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    n_images, main_url, all_urls = _images(row.get("imagesCSV"), prefix)
    return {
        "keepa_title": _clean_text(row.get("title")),
        "keepa_brand": _clean_text(row.get("brand")),
        "keepa_manufacturer": _clean_text(row.get("manufacturer")),
        "keepa_number_of_items": _pos_int(row.get("numberOfItems")),
        "keepa_package_quantity": _pos_int(row.get("packageQuantity")),
        "keepa_item_weight_g": _pos_int(row.get("itemWeight")),
        "keepa_size": _clean_text(row.get("size")),
        "keepa_features": _features_text(row.get("features")),
        "keepa_description": _clean_text(row.get("description")),
        "keepa_image_count": n_images,
        "keepa_main_image_url": main_url,
        "keepa_image_urls": all_urls,
        "keepa_variation_count": _count_csv(row.get("variationCSV")),
        "keepa_present": True,
    }


def _empty_keepa() -> dict[str, Any]:
    row = {f: None for f in KEEPA_FIELDS}
    row["keepa_present"] = False
    return row


def _attach_keepa(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    kcfg = cfg.get("keepa", {})
    if not kcfg.get("enabled", False):
        return df.join(pd.DataFrame([_empty_keepa()] * len(df), index=df.index))

    prefix = kcfg.get("image_url_prefix", "")
    wanted = set(kcfg.get("columns", []))
    try:
        kpath = pick_csv(cfg["paths"]["keepa_dir"])
    except FileNotFoundError:
        print("[keepa] source not found — attaching empty columns")
        return df.join(pd.DataFrame([_empty_keepa()] * len(df), index=df.index))

    kdf = pd.read_csv(kpath, usecols=lambda c: c in wanted, dtype={"asin": str}, low_memory=False)
    by_asin = {str(r["asin"]): _keepa_row(r, prefix) for r in kdf.to_dict("records")}
    rows = [by_asin.get(str(a), _empty_keepa()) for a in df["asin"].astype(str)]
    keepa_df = pd.DataFrame(rows, index=df.index)
    print(f"[keepa] matched {int(keepa_df['keepa_present'].sum()):,}/{len(df):,} ASINs from {kpath.name}")
    return df.join(keepa_df)


def load_data(cfg: dict) -> pd.DataFrame:
    """Load + consolidate the Amazon scrape and Keepa export into one frame."""
    amz_path = pick_csv(cfg["paths"]["amazon_dir"])
    df = pd.read_csv(amz_path, low_memory=False)
    cols = cfg["amazon_columns"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Amazon CSV missing columns: {missing}")
    df = df[cols].copy()
    print(f"[amazon] {len(df):,} listings from {amz_path.name}")
    return _attach_keepa(df, cfg)
