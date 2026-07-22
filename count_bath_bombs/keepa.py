"""Second product source: Keepa product-data export (one row per ASIN).

Complements the amazon_web_scraping CSV. Keepa is the stronger source for
item counts (numberOfItems / packageQuantity), images (imagesCSV) and
variations; the scrape stays stronger for product descriptions. All Keepa
fields are surfaced under a `keepa_` prefix and joined on `asin`, so the
scrape remains the primary source and Keepa acts as a fallback / second
signal downstream (counts, resolver, purity, label UI images).
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pandas as pd

IMAGE_URL_PREFIX = "https://m.media-amazon.com/images/I/"

# Only these columns are read from the (large) Keepa CSV.
KEEPA_SOURCE_COLUMNS = [
    "asin",
    "title",
    "brand",
    "manufacturer",
    "numberOfItems",
    "packageQuantity",
    "itemWeight",
    "size",
    "features",
    "description",
    "imagesCSV",
    "variationCSV",
]

# Columns attached to the working frame (keeps output schema stable when Keepa
# is disabled or the source file is absent).
KEEPA_FIELDS = [
    "keepa_title",
    "keepa_brand",
    "keepa_manufacturer",
    "keepa_number_of_items",
    "keepa_package_quantity",
    "keepa_item_weight_g",
    "keepa_size",
    "keepa_features",
    "keepa_description",
    "keepa_image_count",
    "keepa_main_image_url",
    "keepa_image_urls",
    "keepa_variation_count",
    "keepa_present",
]


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
            joined = " | ".join(str(x).strip() for x in items if str(x).strip())
            return joined or None
    except (ValueError, SyntaxError):
        pass
    return s


def _count_csv(value: Any) -> int | None:
    s = str(value or "").strip()
    if not s or s.lower() == "nan":
        return None
    n = len([x for x in s.split(",") if x])
    return n or None


def _images(value: Any, prefix: str) -> tuple[int | None, str | None, str | None]:
    s = str(value or "").strip()
    if not s or s.lower() == "nan":
        return None, None, None
    tokens = [t for t in s.split(",") if t]
    if not tokens:
        return None, None, None
    urls = [prefix + t for t in tokens]
    return len(tokens), urls[0], " | ".join(urls)


def extract_keepa_row(row: dict[str, Any], image_url_prefix: str = IMAGE_URL_PREFIX) -> dict[str, Any]:
    n_images, main_url, all_urls = _images(row.get("imagesCSV"), image_url_prefix)
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


def _empty_row() -> dict[str, Any]:
    row = {f: None for f in KEEPA_FIELDS}
    row["keepa_present"] = False
    return row


def attach_keepa(
    df: pd.DataFrame,
    csv_path: str | Path | None,
    image_url_prefix: str = IMAGE_URL_PREFIX,
) -> pd.DataFrame:
    """Left-join Keepa fields onto `df` by `asin`.

    Missing/absent source degrades gracefully to null `keepa_*` columns
    (mirrors the html_missing_file fallback), so the pipeline still runs
    when the Dropbox-hosted Keepa file is not hydrated locally.
    """
    path = Path(csv_path) if csv_path else None
    if path is None or not path.exists():
        print(f"[keepa] source not found, attaching empty columns: {path}")
        empty = pd.DataFrame([_empty_row() for _ in range(len(df))], index=df.index)
        return df.join(empty)

    kdf = pd.read_csv(
        path,
        usecols=lambda c: c in KEEPA_SOURCE_COLUMNS,
        dtype={"asin": str},
        low_memory=False,
    )
    by_asin = {
        str(rec["asin"]): extract_keepa_row(rec, image_url_prefix)
        for rec in kdf.to_dict("records")
    }
    rows = [by_asin.get(str(asin), _empty_row()) for asin in df["asin"].astype(str)]
    keepa_df = pd.DataFrame(rows, index=df.index)

    matched = int(keepa_df["keepa_present"].sum())
    print(f"[keepa] matched {matched:,}/{len(df):,} ASINs from {path.name}")
    return df.join(keepa_df)
