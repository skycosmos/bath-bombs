"""Counting: extract every candidate number from a listing, then pick the winner.

`extract_candidates(row)` scans the text channels + catalog fields and returns
the `cand_*` signals; `resolve_row(row)` turns those (plus the purity verdict)
into the final `n_bomb_balls` via a priority ladder. Both are per-row; the
pipeline runs them in a single pass alongside purity.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

# --------------------------------------------------------------------------- #
# Candidate extraction — regexes over the text channels
# --------------------------------------------------------------------------- #
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

    for m in PACK_OF.finditer(text):
        n = int(m.group("n"))
        if n > 1:
            found.append((n, "pack_of"))

    return found


_TEXT_PRIORITY = {
    "set_of": 0,
    "n_x_bombs": 1,
    "near_bombs": 2, "near_bomb": 2, "near_balls": 2, "near_ball": 2,
    "near_fizzies": 2, "near_fizzy": 2, "near_fizzers": 2,
    "near_blasters": 2, "near_blaster": 2,
    "pack_of": 3,
    "near_pack": 4,
    "near_count": 5, "near_pcs": 5, "near_pieces": 5,
    "n_count": 6,
}


def best_text_count(text: str) -> tuple[int | None, str | None]:
    hits = parse_counts_from_text(text)
    if not hits:
        return None, None
    # Prefer any value >1 over a bare 1; within that, lowest-priority pattern wins.
    multi = [h for h in hits if h[0] > 1]
    if multi:
        return sorted(multi, key=lambda x: (_TEXT_PRIORITY.get(x[1], 9), -x[0]))[0]
    return sorted(hits, key=lambda x: (_TEXT_PRIORITY.get(x[1], 9), 999))[0]


def extract_candidates(row) -> dict[str, Any]:
    title = str(row.get("title") or "")
    feature = str(row.get("feature") or "")
    desc = str(row.get("product_description") or "")
    size = str(row.get("size") or "")
    keepa_features = str(row.get("keepa_features") or "")
    keepa_desc = str(row.get("keepa_description") or "")

    title_n, title_pat = best_text_count(title)
    size_n, size_pat = best_text_count(size)
    bullets_n, bullets_pat = best_text_count(feature + "\n" + keepa_features)
    desc_n, desc_pat = best_text_count(desc + "\n" + keepa_desc)

    unit_num = _to_int(row.get("unit_num")) if str(row.get("unit_text") or "").lower() in {
        "count", "each", "unit", "units",
    } else None
    label_unit_num = None
    if "count" in str(row.get("label_unit") or "").lower():
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
        "cand_number_of_items": _to_int(row.get("number_of_items")),
        "cand_keepa_number_of_items": _to_int(row.get("keepa_number_of_items")),
        "cand_keepa_package_qty": _to_int(row.get("keepa_package_quantity")),
        "cand_unit_num": unit_num,
        "cand_label_unit_num": label_unit_num,
    }


# --------------------------------------------------------------------------- #
# Resolution — pick the final count from the candidates (pure items only)
# --------------------------------------------------------------------------- #
def _first_multi(*values: int | None) -> int | None:
    for v in values:
        if v is not None and v > 1:
            return int(v)
    return None


def _count_guess(row) -> tuple[int | None, str | None, str]:
    """Best numeric guess + source + confidence via the priority ladder."""
    title_n = row.get("cand_title")
    bullets_n = row.get("cand_bullets")
    size_n = row.get("cand_size")
    desc_n = row.get("cand_description")
    noi = row.get("cand_number_of_items")
    keepa_noi = row.get("cand_keepa_number_of_items")
    keepa_pkg = row.get("cand_keepa_package_qty")
    label_n = row.get("cand_label_unit_num")

    text_multi = _first_multi(title_n, bullets_n, size_n, desc_n)
    if text_multi is not None:
        source = (
            "title" if title_n and title_n > 1
            else "bullets" if bullets_n and bullets_n > 1
            else "size" if size_n and size_n > 1
            else "description"
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

    if label_n is not None and label_n > 1:
        return int(label_n), "label_unit_num", "medium"

    text_one = title_n == 1 or bullets_n == 1 or size_n == 1
    if noi == 1 or text_one or label_n == 1 or keepa_noi == 1:
        conf = "medium" if (noi == 1 or text_one) else "low"
        return 1, "single_default", conf

    # Pure singles often omit pack language — assume 1 but flag low confidence.
    return 1, "assumed_single", "low"


def resolve_row(row) -> dict[str, Any]:
    """Count only pure bath bombs; every other listing is excluded (no count)."""
    is_pure = row.get("is_pure_bath_bomb")
    noi = row.get("cand_number_of_items")
    keepa_noi = row.get("cand_keepa_number_of_items")
    keepa_pkg = row.get("cand_keepa_package_qty")
    unit_num = row.get("cand_unit_num")

    text_multi = _first_multi(
        row.get("cand_title"), row.get("cand_bullets"),
        row.get("cand_size"), row.get("cand_description"),
    )
    catalog_ones = [v for v in (noi, keepa_noi, keepa_pkg, unit_num) if v == 1]
    seller_pack_as_one = bool(text_multi and catalog_ones)

    if is_pure is not True:
        return {
            "n_bomb_balls": None,
            "count_confidence": "n/a",
            "count_source": None,
            "seller_counts_pack_as_one": seller_pack_as_one,
            "count_unable": False,
        }

    n, source, conf = _count_guess(row)
    return {
        "n_bomb_balls": int(n) if n is not None else None,
        "count_confidence": conf,
        "count_source": source,
        "seller_counts_pack_as_one": seller_pack_as_one,
        "count_unable": n is None,
    }
