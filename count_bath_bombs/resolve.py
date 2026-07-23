from __future__ import annotations

from typing import Any

import pandas as pd


def _first_multi(*values: int | None) -> int | None:
    for v in values:
        if v is not None and v > 1:
            return int(v)
    return None


def _cand_count_guess(row: pd.Series) -> tuple[int | None, str | None, str]:
    """Best numeric guess + source + confidence (without purity gating)."""
    title_n = row.get("cand_title")
    bullets_n = row.get("cand_bullets")
    size_n = row.get("cand_size")
    desc_n = row.get("cand_description")
    noi = row.get("cand_number_of_items")
    html_noi = row.get("cand_html_number_of_items")
    html_unit = row.get("cand_html_unit_count")
    html_pkg = row.get("cand_html_package_qty")
    keepa_noi = row.get("cand_keepa_number_of_items")
    keepa_pkg = row.get("cand_keepa_package_qty")
    label_n = row.get("cand_label_unit_num")

    text_multi = _first_multi(title_n, bullets_n, size_n, desc_n)
    if text_multi is not None:
        source = (
            "title"
            if title_n and title_n > 1
            else (
                "bullets"
                if bullets_n and bullets_n > 1
                else ("size" if size_n and size_n > 1 else "description")
            )
        )
        conf = "high" if source in {"title", "bullets"} else "medium"
        return text_multi, source, conf

    if noi is not None and noi > 1:
        return int(noi), "number_of_items", "medium"

    # Keepa numberOfItems agrees ~99.8% with the scrape where both exist and
    # covers ~10k rows the scrape leaves blank — strong second catalog signal.
    keepa_multi = _first_multi(keepa_noi, keepa_pkg)
    if keepa_multi is not None:
        return keepa_multi, "keepa_number_of_items", "medium"

    html_multi = _first_multi(html_noi, html_unit, html_pkg, label_n)
    if html_multi is not None:
        return html_multi, "html_details", "medium"

    text_one = title_n == 1 or bullets_n == 1 or size_n == 1
    if noi == 1 or html_noi == 1 or html_unit == 1 or text_one or label_n == 1 or keepa_noi == 1:
        conf = "medium" if (noi == 1 or text_one) else "low"
        return 1, "single_default", conf

    # Pure singles often omit pack language — assume 1 but flag low confidence.
    return 1, "assumed_single", "low"


def resolve_row(row: pd.Series) -> dict[str, Any]:
    """Count only pure bath bombs; every other listing is excluded (no count)."""
    is_pure = row.get("is_pure_bath_bomb")
    title_n = row.get("cand_title")
    noi = row.get("cand_number_of_items")
    html_noi = row.get("cand_html_number_of_items")
    html_unit = row.get("cand_html_unit_count")
    html_pkg = row.get("cand_html_package_qty")
    keepa_noi = row.get("cand_keepa_number_of_items")
    keepa_pkg = row.get("cand_keepa_package_qty")
    unit_num = row.get("cand_unit_num")

    text_multi = _first_multi(
        title_n,
        row.get("cand_bullets"),
        row.get("cand_size"),
        row.get("cand_description"),
    )
    catalog_ones = [
        v for v in (noi, html_noi, html_unit, html_pkg, keepa_noi, keepa_pkg, unit_num) if v == 1
    ]
    seller_pack_as_one = bool(text_multi and catalog_ones)

    # Only pure bath bombs get a unit count; everything else is excluded.
    if is_pure is not True:
        return {
            "n_bomb_balls": None,
            "count_confidence": "n/a",
            "count_source": None,
            "seller_counts_pack_as_one": seller_pack_as_one,
            "count_unable": False,
        }

    n, source, conf = _cand_count_guess(row)
    return {
        "n_bomb_balls": int(n) if n is not None else None,
        "count_confidence": conf,
        "count_source": source,
        "seller_counts_pack_as_one": seller_pack_as_one,
        "count_unable": n is None,
    }


def apply_resolver(df: pd.DataFrame) -> pd.DataFrame:
    rows = [resolve_row(row) for _, row in df.iterrows()]
    resolved = pd.DataFrame(rows, index=df.index)
    return pd.concat([df, resolved], axis=1)
