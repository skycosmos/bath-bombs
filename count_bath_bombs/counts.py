from __future__ import annotations

import re
from typing import Any

import pandas as pd

# Count next to bomb-ish nouns
COUNT_NEAR_BOMB = re.compile(
    r"(?P<n>\d+)\s*[-\s]?(?P<unit>pack|count|pcs|pieces|bombs?|balls?|fizz(?:y|ies|ers?)?|blasters?)",
    re.IGNORECASE,
)
SET_OF = re.compile(r"\bset of\s+(?P<n>\d+)\b", re.IGNORECASE)
PACK_OF = re.compile(r"\bpack of\s+(?P<n>\d+)\b", re.IGNORECASE)
N_COUNT = re.compile(r"\b(?P<n>\d+)\s*count\b", re.IGNORECASE)
X_BOMBS = re.compile(
    r"\b(?P<n>\d+)\s*(?:x|×)\s*(?:\d+(?:\.\d+)?\s*(?:oz|ounce|oza|g|gram)s?\s+)?(?:bath\s+)?(?:bombs?|balls?|fizz)",
    re.IGNORECASE,
)


def _to_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def parse_counts_from_text(text: str) -> list[tuple[int, str]]:
    """Return list of (count, pattern_name) from free text."""
    if not text or not isinstance(text, str):
        return []
    found: list[tuple[int, str]] = []

    for m in SET_OF.finditer(text):
        found.append((int(m.group("n")), "set_of"))
    for m in X_BOMBS.finditer(text):
        found.append((int(m.group("n")), "n_x_bombs"))
    for m in COUNT_NEAR_BOMB.finditer(text):
        n = int(m.group("n"))
        unit = m.group("unit").lower()
        # Skip bare "pack of 1" handled separately; "1 count" is weak alone
        if unit in {"pack"} and n == 1:
            continue
        found.append((n, f"near_{unit}"))
    for m in N_COUNT.finditer(text):
        n = int(m.group("n"))
        found.append((n, "n_count"))

    pack_of_matches = list(PACK_OF.finditer(text))
    for m in pack_of_matches:
        n = int(m.group("n"))
        if n > 1:
            found.append((n, "pack_of"))

    return found


def best_text_count(text: str) -> tuple[int | None, str | None]:
    hits = parse_counts_from_text(text)
    if not hits:
        return None, None
    # Prefer counts tied to bombs / set_of / multi pack over generic "1 count"
    priority = {
        "set_of": 0,
        "n_x_bombs": 1,
        "near_bombs": 2,
        "near_bomb": 2,
        "near_balls": 2,
        "near_ball": 2,
        "near_fizzies": 2,
        "near_fizzy": 2,
        "near_fizzers": 2,
        "near_blasters": 2,
        "near_blaster": 2,
        "pack_of": 3,
        "near_pack": 4,
        "near_count": 5,
        "near_pcs": 5,
        "near_pieces": 5,
        "n_count": 6,
    }
    # Prefer larger multi-counts when same priority (e.g. 5 Count vs Pack of 1 already skipped)
    hits_sorted = sorted(
        hits,
        key=lambda x: (priority.get(x[1], 9), -x[0] if x[0] > 1 else 999),
    )
    # If best is 1 and there exists a >1 elsewhere, prefer >1
    multi = [h for h in hits if h[0] > 1]
    if multi:
        multi_sorted = sorted(multi, key=lambda x: (priority.get(x[1], 9), -x[0]))
        return multi_sorted[0]
    return hits_sorted[0]


def extract_candidates(row: pd.Series) -> dict[str, Any]:
    title = str(row.get("title") or "")
    feature = str(row.get("feature") or "")
    desc = str(row.get("product_description") or "")
    size = str(row.get("size") or "")
    bullets = str(row.get("html_bullets") or "")
    html_desc = str(row.get("html_description") or "")
    keepa_features = str(row.get("keepa_features") or "")
    keepa_desc = str(row.get("keepa_description") or "")

    title_n, title_pat = best_text_count(title)
    size_n, size_pat = best_text_count(size)
    bullets_n, bullets_pat = best_text_count(bullets + "\n" + feature + "\n" + keepa_features)
    desc_n, desc_pat = best_text_count(desc + "\n" + html_desc + "\n" + keepa_desc)

    number_of_items = _to_int(row.get("number_of_items"))
    html_number_of_items = _to_int(row.get("html_number_of_items"))
    html_unit_count = _to_int(row.get("html_unit_count"))
    html_package_qty = _to_int(row.get("html_item_package_quantity"))
    keepa_number_of_items = _to_int(row.get("keepa_number_of_items"))
    keepa_package_qty = _to_int(row.get("keepa_package_quantity"))
    unit_num = _to_int(row.get("unit_num")) if str(row.get("unit_text") or "").lower() in {
        "count",
        "each",
        "unit",
        "units",
    } else None
    label_unit_num = None
    label_unit = str(row.get("label_unit") or "").lower()
    if "count" in label_unit:
        label_unit_num = _to_int(row.get("label_unit_num"))

    return {
        "cand_title": title_n,
        "cand_title_pattern": title_pat,
        "cand_size": size_n,
        "cand_size_pattern": size_pat,
        "cand_bullets": bullets_n,
        "cand_bullets_pattern": bullets_pat,
        "cand_description": desc_n,
        "cand_description_pattern": desc_pat,
        "cand_number_of_items": number_of_items,
        "cand_html_number_of_items": html_number_of_items,
        "cand_html_unit_count": html_unit_count,
        "cand_html_package_qty": html_package_qty,
        "cand_keepa_number_of_items": keepa_number_of_items,
        "cand_keepa_package_qty": keepa_package_qty,
        "cand_unit_num": unit_num,
        "cand_label_unit_num": label_unit_num,
    }


def apply_candidates(df: pd.DataFrame) -> pd.DataFrame:
    rows = [extract_candidates(row) for _, row in df.iterrows()]
    cand = pd.DataFrame(rows, index=df.index)
    return pd.concat([df, cand], axis=1)
