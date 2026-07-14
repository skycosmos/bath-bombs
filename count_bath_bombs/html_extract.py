from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

DETAIL_KEYS = {
    "number of items": "html_number_of_items",
    "unit count": "html_unit_count",
    "item package quantity": "html_item_package_quantity",
    "size": "html_size",
    "item weight": "html_item_weight",
}


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    # Amazon accessibility marks often leave "‏ : ‎"
    text = text.replace("‏", "").replace("‎", "")
    text = re.sub(r"\s*:\s*", ": ", text)
    return text.strip(" :")


def _parse_number(text: str) -> float | None:
    if not text:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", text.replace(",", ""))
    return float(m.group(1)) if m else None


def extract_from_html(
    html: str,
    max_bullets: int = 12,
    max_description_chars: int = 1200,
) -> dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    title_el = soup.select_one("#productTitle")
    title = _clean(title_el.get_text(" ", strip=True)) if title_el else None

    details: dict[str, str] = {}

    for table_sel in (
        "#productDetails_techSpec_section_1",
        "#productDetails_detailBullets_sections1",
        "#productDetails_techSpec_section_2",
    ):
        table = soup.select_one(table_sel)
        if not table:
            continue
        for tr in table.select("tr"):
            th, td = tr.select_one("th"), tr.select_one("td")
            if th and td:
                details[_clean(th.get_text(" ", strip=True)).lower()] = _clean(
                    td.get_text(" ", strip=True)
                )

    for li in soup.select("#detailBullets_feature_div li"):
        raw = _clean(li.get_text(" ", strip=True))
        if ":" in raw:
            k, v = raw.split(":", 1)
            details[k.strip().lower()] = v.strip()

    for row in soup.select("#productOverview_feature_div tr"):
        cells = row.select("td")
        if len(cells) >= 2:
            details[_clean(cells[0].get_text(" ", strip=True)).lower()] = _clean(
                cells[1].get_text(" ", strip=True)
            )

    bullets: list[str] = []
    for li in soup.select("#feature-bullets li"):
        t = _clean(li.get_text(" ", strip=True))
        if not t or "videos" in t.lower():
            continue
        bullets.append(t)
        if len(bullets) >= max_bullets:
            break

    desc_el = soup.select_one("#productDescription")
    description = _clean(desc_el.get_text(" ", strip=True)) if desc_el else ""
    if len(description) > max_description_chars:
        description = description[:max_description_chars]

    out: dict[str, Any] = {
        "html_title": title,
        "html_bullets": " | ".join(bullets),
        "html_description": description,
        "html_number_of_items": None,
        "html_unit_count": None,
        "html_item_package_quantity": None,
        "html_size": None,
        "html_item_weight": None,
        "html_parse_ok": True,
    }
    for key_substr, field in DETAIL_KEYS.items():
        for dk, dv in details.items():
            if key_substr in dk:
                if field.startswith("html_") and field in {
                    "html_number_of_items",
                    "html_unit_count",
                    "html_item_package_quantity",
                }:
                    out[field] = _parse_number(dv)
                else:
                    out[field] = dv
                break
    return out


def load_or_extract_asin(
    asin: str,
    html_dir: Path,
    cache_dir: Path,
    max_bullets: int,
    max_description_chars: int,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{asin}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    html_path = html_dir / f"{asin}.html"
    if not html_path.exists():
        result = {
            "html_title": None,
            "html_bullets": None,
            "html_description": None,
            "html_number_of_items": None,
            "html_unit_count": None,
            "html_item_package_quantity": None,
            "html_size": None,
            "html_item_weight": None,
            "html_parse_ok": False,
            "html_missing_file": True,
        }
        cache_path.write_text(json.dumps(result), encoding="utf-8")
        return result

    try:
        html = html_path.read_text(encoding="utf-8", errors="ignore")
        result = extract_from_html(html, max_bullets, max_description_chars)
        result["html_missing_file"] = False
    except Exception as exc:  # noqa: BLE001 — cache failures per ASIN
        result = {
            "html_title": None,
            "html_bullets": None,
            "html_description": None,
            "html_number_of_items": None,
            "html_unit_count": None,
            "html_item_package_quantity": None,
            "html_size": None,
            "html_item_weight": None,
            "html_parse_ok": False,
            "html_missing_file": False,
            "html_error": str(exc),
        }
    cache_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def attach_html_extracts(
    df,
    html_dir: str | Path,
    cache_dir: str | Path,
    max_bullets: int = 12,
    max_description_chars: int = 1200,
    limit: int | None = None,
):
    html_dir = Path(html_dir)
    cache_dir = Path(cache_dir)
    asins = df["asin"].astype(str).tolist()
    if limit is not None:
        asins = asins[:limit]

    records = []
    for asin in tqdm(asins, desc="HTML extract"):
        records.append(
            load_or_extract_asin(
                asin, html_dir, cache_dir, max_bullets, max_description_chars
            )
        )

    html_df = pd.DataFrame.from_records(records, index=df.index[: len(records)])
    if limit is not None:
        out = df.copy()
        for col in html_df.columns:
            out.loc[out.index[:limit], col] = html_df[col].values
        return out
    return df.join(html_df)
