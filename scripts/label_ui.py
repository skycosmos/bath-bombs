"""
Streamlit review + labeling console for bath-bomb unit counts.

Human reviewers look at each listing's evidence (title, catalog fields,
extracted candidate counts, bullets, description), then confirm or correct the
rule prediction. Labels are written to data/gold/gold_labels.csv and consumed
by scripts/eval_gold.py.

Launch:
  .venv/bin/streamlit run scripts/label_ui.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from count_bath_bombs import annotations as A
from count_bath_bombs.config import load_config
from count_bath_bombs.gold import (
    CLASS_LABELS,
    COUNT_LABELS,
    class_label_for_row,
    count_label_for_row,
)

st.set_page_config(page_title="Bath Bomb Review Console", layout="wide")

PURE_OPTIONS = ["true", "false", "unsure"]
EXCLUDE_OPTIONS = ["", "craft_kit", "bundle", "substitute", "toiletry", "unclassified"]
AMAZON_URL = "https://www.amazon.com/dp/{asin}"


# --------------------------------------------------------------------------- #
# Data loading / persistence
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _read_csv(path_str: str, mtime: float) -> pd.DataFrame:
    """Cached read keyed on path + mtime so edits are picked up on rerun."""
    return pd.read_csv(path_str, low_memory=False)


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return _read_csv(str(path), path.stat().st_mtime)


def _ensure_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee class_label / count_label columns for filtering + badges."""
    if "class_label" not in df.columns and "is_pure_bath_bomb" in df.columns:
        df = df.copy()
        df["class_label"] = [class_label_for_row(r) for _, r in df.iterrows()]
        df["count_label"] = [count_label_for_row(r) for _, r in df.iterrows()]
    return df


def _load_queue(cfg: dict) -> tuple[pd.DataFrame, str]:
    sample_path = Path(cfg["paths"]["labeling_sample_csv"])
    pred_path = Path(cfg["paths"]["output_csv"])

    source = st.sidebar.selectbox(
        "Queue source",
        ["labeling_sample", "product_counts"],
        help="labeling_sample = stratified review sheet · product_counts = full predictions",
    )
    path = {
        "labeling_sample": sample_path,
        "product_counts": pred_path,
    }[source]
    queue = _read(path)
    if queue is None:
        st.error(
            f"Missing or empty: `{path}`.\n\n"
            "Run: `python scripts/run_pipeline.py --labeling-sample`"
        )
        st.stop()

    # Enrich thin queues (e.g. labeling_sample) with the full evidence columns.
    full = _read(pred_path)
    if full is not None and "asin" in full.columns:
        extra = [c for c in full.columns if c not in queue.columns]
        if extra:
            queue = queue.merge(full[["asin"] + extra], on="asin", how="left")
    return _ensure_labels(queue), source


# --------------------------------------------------------------------------- #
# Scraped-page HTML parser — used only by the optional "view Amazon page" panel
# below (the counting pipeline no longer parses HTML).
# --------------------------------------------------------------------------- #
_DETAIL_KEYS = {
    "number of items": "html_number_of_items",
    "unit count": "html_unit_count",
    "item package quantity": "html_item_package_quantity",
    "size": "html_size",
    "item weight": "html_item_weight",
}


def _clean_html_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    # Amazon accessibility marks often leave stray RLM/LRM characters.
    text = text.replace("‏", "").replace("‎", "")
    text = re.sub(r"\s*:\s*", ": ", text)
    return text.strip(" :")


def _parse_number(text: str) -> float | None:
    if not text:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", text.replace(",", ""))
    return float(m.group(1)) if m else None


def extract_from_html(html: str, max_bullets: int = 12, max_description_chars: int = 1200) -> dict:
    """Parse a scraped Amazon product page into a clean dict (Reader view)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    title_el = soup.select_one("#productTitle")
    title = _clean_html_text(title_el.get_text(" ", strip=True)) if title_el else None

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
                details[_clean_html_text(th.get_text(" ", strip=True)).lower()] = _clean_html_text(
                    td.get_text(" ", strip=True)
                )

    for li in soup.select("#detailBullets_feature_div li"):
        raw = _clean_html_text(li.get_text(" ", strip=True))
        if ":" in raw:
            k, v = raw.split(":", 1)
            details[k.strip().lower()] = v.strip()

    for row in soup.select("#productOverview_feature_div tr"):
        cells = row.select("td")
        if len(cells) >= 2:
            details[_clean_html_text(cells[0].get_text(" ", strip=True)).lower()] = _clean_html_text(
                cells[1].get_text(" ", strip=True)
            )

    bullets: list[str] = []
    for li in soup.select("#feature-bullets li"):
        t = _clean_html_text(li.get_text(" ", strip=True))
        if not t or "videos" in t.lower():
            continue
        bullets.append(t)
        if len(bullets) >= max_bullets:
            break

    desc_el = soup.select_one("#productDescription")
    description = _clean_html_text(desc_el.get_text(" ", strip=True)) if desc_el else ""
    if len(description) > max_description_chars:
        description = description[:max_description_chars]

    out: dict = {
        "html_title": title,
        "html_bullets": " | ".join(bullets),
        "html_description": description,
        "html_number_of_items": None,
        "html_unit_count": None,
        "html_item_package_quantity": None,
        "html_size": None,
        "html_item_weight": None,
    }
    for key_substr, field in _DETAIL_KEYS.items():
        for dk, dv in details.items():
            if key_substr in dk:
                if field in {"html_number_of_items", "html_unit_count", "html_item_package_quantity"}:
                    out[field] = _parse_number(dv)
                else:
                    out[field] = dv
                break
    return out


@st.cache_data(show_spinner=False, max_entries=8)
def _load_local_html(path_str: str, mtime: float) -> str:
    return Path(path_str).read_text(encoding="utf-8", errors="ignore")


def _sanitize_amazon_html(html: str) -> str:
    """Make a scraped Amazon page safe/static to render in an iframe:
    strip scripts (no redirects / frame-busting / heavy JS) and add a <base>
    so the page's relative CSS/image URLs resolve against amazon.com."""
    html = re.sub(r"<script\b[^>]*>.*?</script>", "", html, flags=re.I | re.S)
    html = re.sub(r"<noscript\b[^>]*>.*?</noscript>", "", html, flags=re.I | re.S)
    html = re.sub(r"<meta[^>]+http-equiv=['\"]?refresh['\"]?[^>]*>", "", html, flags=re.I)
    base = '<base href="https://www.amazon.com/" target="_blank">'
    if re.search(r"<head[^>]*>", html, flags=re.I):
        html = re.sub(r"(<head[^>]*>)", r"\1" + base, html, count=1, flags=re.I)
    else:
        html = base + html
    return html


@st.cache_data(show_spinner=False, max_entries=256)
def _quick_media(path_str: str, mtime: float) -> dict:
    """Fast regex scan of a snapshot for a thumbnail + price (no full HTML parse)."""
    try:
        raw = Path(path_str).read_text(encoding="utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return {"image": None, "price": None}
    img = None
    tag = re.search(r"<meta[^>]*og:image[^>]*>", raw, re.I)
    if tag:
        c = re.search(r'content=["\']([^"\']+)', tag.group(0), re.I)
        if c and c.group(1).startswith("http"):
            img = c.group(1)
    if not img:
        tag = re.search(r'<img[^>]*id=["\']landingImage["\'][^>]*>', raw, re.I)
        if tag:
            s = re.search(r'\bsrc=["\'](https?://[^"\']+)', tag.group(0), re.I)
            if s:
                img = s.group(1)
    if not img:
        m = re.search(r'"(https://m\.media-amazon\.com/images/I/[^"\']+\.jpg)"', raw)
        if m:
            img = m.group(1)
    price = None
    for m in re.finditer(r'class=["\']a-offscreen["\']>\s*([^<]{1,16})<', raw, re.I):
        txt = m.group(1).strip()
        if re.search(r"[$£€]|\d", txt):
            price = txt
            break
    return {"image": img, "price": price}


@st.cache_data(show_spinner=False, max_entries=32)
def _reader_fields(path_str: str, mtime: float) -> dict:
    """Extract a clean product card from a scraped Amazon page."""
    from bs4 import BeautifulSoup

    raw = Path(path_str).read_text(encoding="utf-8", errors="ignore")
    base = extract_from_html(raw, max_bullets=15, max_description_chars=1600)
    soup = BeautifulSoup(raw, "lxml")

    img = None
    og = soup.select_one('meta[property="og:image"]')
    if og and og.get("content"):
        img = og["content"]
    if not img:
        el = soup.select_one("#landingImage") or soup.select_one("#imgTagWrapperId img")
        if el:
            img = el.get("data-old-hires") or el.get("src")

    price = None
    for sel in (
        "#corePriceDisplay_desktop_feature_div .a-offscreen",
        "#corePrice_feature_div .a-offscreen",
        ".a-price .a-offscreen",
        "#priceblock_ourprice",
        "#priceblock_dealprice",
        "#price_inside_buybox",
    ):
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            price = el.get_text(strip=True)
            break

    byline = soup.select_one("#bylineInfo")
    brand = byline.get_text(" ", strip=True) if byline else None

    return {
        "title": base.get("html_title"),
        "image": img,
        "price": price,
        "brand": brand,
        "bullets": [b for b in (base.get("html_bullets") or "").split(" | ") if b],
        "description": base.get("html_description") or "",
        "details": {
            "Number of items": base.get("html_number_of_items"),
            "Unit count": base.get("html_unit_count"),
            "Item package quantity": base.get("html_item_package_quantity"),
            "Size": base.get("html_size"),
            "Item weight": base.get("html_item_weight"),
        },
    }


def _reader_card(path: Path) -> None:
    fields = _reader_fields(str(path), path.stat().st_mtime)
    img_col, fact_col = st.columns([1, 2])
    with img_col:
        if fields["image"]:
            st.image(fields["image"], use_container_width=True)
        else:
            st.caption("No product image found in snapshot.")
    with fact_col:
        if fields["title"]:
            st.markdown(f"**{fields['title']}**")
        if fields["brand"]:
            st.caption(fields["brand"])
        if fields["price"]:
            st.markdown(f"### {fields['price']}")
        det = {k: v for k, v in fields["details"].items() if v not in (None, "")}
        if det:
            st.table(pd.DataFrame(list(det.items()), columns=["Detail", "Value"]))
    if fields["bullets"]:
        st.markdown("**About this item**")
        for b in fields["bullets"]:
            st.markdown(f"- {b}")
    if fields["description"]:
        with st.expander("Product description"):
            st.write(fields["description"])


def _scraped_page_panel(asin: str, html_dir: str | Path) -> None:
    path = Path(html_dir) / f"{asin}.html"
    exists = path.exists()
    mode = st.radio(
        "Amazon product page",
        ["Off", "📖 Reader", "🖼 Full page"],
        horizontal=True,
        index=0,
        key=f"view_{asin}",
        help="Reader = clean extracted card · Full page = raw scraped HTML (heavier).",
    )
    st.caption(
        f"Local snapshot: {'✅ available' if exists else '❌ not found'} · "
        f"[open live on Amazon ↗]({AMAZON_URL.format(asin=asin)})"
    )
    if mode == "Off":
        return
    if not exists:
        st.warning("No local scraped HTML for this ASIN.")
        return
    if mode == "📖 Reader":
        with st.spinner("Building reader view…"):
            _reader_card(path)
    else:
        try:
            raw = _load_local_html(str(path), path.stat().st_mtime)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read HTML: {exc}")
            return
        with st.spinner("Rendering scraped page…"):
            components.html(_sanitize_amazon_html(raw), height=850, scrolling=True)


def _persist_annotation(
    annotations_path: str | Path,
    gold_path: str | Path,
    record: dict,
) -> None:
    """Write one human annotation, then re-derive the gold file."""
    record = dict(record)
    record["ts"] = A.now_ts()
    annots = A.load_annotations(annotations_path)
    annots = A.upsert_annotation(annots, record)
    A.save_annotations(annotations_path, annots)
    A.rebuild_gold(annotations_path, gold_path)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _clean(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return v


def _as_pure_str(value) -> str:
    v = _clean(value)
    if v is None:
        return "unsure"
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "1.0"}:
        return "true"
    if s in {"0", "false", "no", "n", "0.0"}:
        return "false"
    return "unsure"


def _as_int(value) -> int | None:
    v = _clean(value)
    if v is None:
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _highlight_count(title: str, row: pd.Series) -> str:
    """Bold the count the rules detected in the title, to guide the reviewer's eye."""
    n = _as_int(row.get("cand_title"))
    if not n:
        return title
    return re.sub(rf"(?<!\d)({n})(?!\d)", r"**\1**", title, count=1)


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def _evidence_panel(row: pd.Series) -> None:
    st.markdown("#### 🔍 Rule prediction")
    pred = _as_pure_str(row.get("is_pure_bath_bomb"))
    n_pred = _as_int(row.get("n_bomb_balls"))
    c1, c2, c3 = st.columns(3)
    c1.metric("is_pure", pred)
    c2.metric("n_bomb_balls", "—" if n_pred is None else n_pred)
    c3.metric("confidence", str(_clean(row.get("count_confidence")) or "—"))
    tags = []
    if _clean(row.get("count_source")):
        tags.append(f"source=`{row.get('count_source')}`")
    if _clean(row.get("exclude_reason")):
        tags.append(f"exclude=`{row.get('exclude_reason')}`")
    if bool(_clean(row.get("seller_counts_pack_as_one"))):
        tags.append("⚠️ seller lists multi-pack as 1 item")
    if tags:
        st.caption(" · ".join(tags))

    st.markdown("#### 🧮 Candidate counts (what the rules saw)")
    cand_rows = []
    cand_map = [
        ("title", "cand_title", "cand_title_pattern"),
        ("size", "cand_size", "cand_size_pattern"),
        ("bullets", "cand_bullets", "cand_bullets_pattern"),
        ("description", "cand_description", "cand_description_pattern"),
        ("number_of_items", "cand_number_of_items", None),
        ("keepa_number_of_items", "cand_keepa_number_of_items", None),
        ("keepa_package_qty", "cand_keepa_package_qty", None),
        ("unit_num", "cand_unit_num", None),
        ("label_unit_num", "cand_label_unit_num", None),
    ]
    for field, ncol, patcol in cand_map:
        n = _as_int(row.get(ncol))
        if n is None:
            continue
        cand_rows.append(
            {"field": field, "count": n, "pattern": _clean(row.get(patcol)) if patcol else ""}
        )
    if cand_rows:
        st.dataframe(pd.DataFrame(cand_rows), hide_index=True, use_container_width=True)
    else:
        st.caption("No numeric count candidates found in any field.")


def _catalog_panel(row: pd.Series) -> None:
    fields = [
        "brand", "manufacturer", "price", "per_unit_price_text",
        "unit_text", "number_of_items", "size", "item_weight",
        "product_dimensions", "variation_quantity", "badge_label", "image_num",
        "keepa_number_of_items", "keepa_package_quantity", "keepa_item_weight_g",
        "keepa_image_count", "keepa_variation_count",
    ]
    data = {f: _clean(row.get(f)) for f in fields if _clean(row.get(f)) is not None}
    if data:
        st.markdown("#### 📦 Catalog fields")
        st.json(data)


def _text_panel(row: pd.Series) -> None:
    bullets = _clean(row.get("feature")) or _clean(row.get("keepa_features"))
    if isinstance(bullets, str) and bullets.strip():
        with st.expander("Bullets / features", expanded=True):
            for b in str(bullets).split(" | "):
                b = b.strip()
                if b:
                    st.markdown(f"- {b}")
    desc = _clean(row.get("product_description")) or _clean(row.get("keepa_description"))
    if isinstance(desc, str) and desc.strip():
        with st.expander("Product description"):
            st.write(str(desc)[:4000])


# --------------------------------------------------------------------------- #
# Review tab
# --------------------------------------------------------------------------- #
def _apply_filters(queue: pd.DataFrame, my_asins: set) -> pd.DataFrame:
    st.sidebar.markdown("### Filters")
    hide_mine = st.sidebar.checkbox("Hide ASINs I've already labeled", value=True)

    st.sidebar.markdown("**Classification**")
    class_vals = [c for c in CLASS_LABELS if c in set(queue.get("class_label", pd.Series()).dropna())]
    pick_class = st.sidebar.multiselect(
        "Classification label", class_vals, default=[],
        help="pure / craft_kit / bundle / substitute / toiletry / unclassified",
    )

    st.sidebar.markdown("**Counting**")
    count_vals = [c for c in COUNT_LABELS if c in set(queue.get("count_label", pd.Series()).dropna())]
    pick_count = st.sidebar.multiselect(
        "Count label", count_vals, default=[],
        help="multi_pack / single / pack_as_one / extreme_count / count_unable (pure items only)",
    )
    conf_vals = (
        sorted(queue["count_confidence"].dropna().unique().tolist())
        if "count_confidence" in queue else []
    )
    pick_conf = st.sidebar.multiselect("Count confidence", conf_vals, default=[]) if conf_vals else []

    search = st.sidebar.text_input("Find ASIN / title contains", value="").strip().lower()

    work = queue.copy()
    work["asin"] = work["asin"].astype(str)

    if hide_mine:
        work = work[~work["asin"].isin(my_asins)]
    if pick_class and "class_label" in work.columns:
        work = work[work["class_label"].isin(pick_class)]
    if pick_count and "count_label" in work.columns:
        work = work[work["count_label"].isin(pick_count)]
    if pick_conf:
        work = work[work["count_confidence"].isin(pick_conf)]
    if search:
        title = work.get("title", pd.Series("", index=work.index)).astype(str).str.lower()
        work = work[work["asin"].str.lower().str.contains(search) | title.str.contains(search)]
    return work.reset_index(drop=True)


def _navigate(n_rows: int) -> int:
    st.session_state.setdefault("idx", 0)
    st.session_state.idx = max(0, min(st.session_state.idx, n_rows - 1))
    c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
    if c1.button("← Prev", use_container_width=True) and st.session_state.idx > 0:
        st.session_state.idx -= 1
        st.rerun()
    if c2.button("Next →", use_container_width=True) and st.session_state.idx < n_rows - 1:
        st.session_state.idx += 1
        st.rerun()
    if c3.button("Skip ⏭", use_container_width=True) and st.session_state.idx < n_rows - 1:
        st.session_state.idx += 1
        st.rerun()
    jump = c4.number_input(
        f"Go to row (0–{n_rows - 1})", min_value=0, max_value=n_rows - 1,
        value=st.session_state.idx, step=1, label_visibility="collapsed",
    )
    if jump != st.session_state.idx:
        st.session_state.idx = int(jump)
        st.rerun()
    return st.session_state.idx


def _review_tab(
    work: pd.DataFrame,
    annots: pd.DataFrame,
    annotations_path: Path,
    gold_path: Path,
    annotator: str,
    html_dir: str | Path,
) -> None:
    if work.empty:
        st.success("🎉 Nothing left in this queue with the current filters — "
                   "clear a filter or switch queue source to keep reviewing.")
        return

    # Clamp the cursor before showing progress so an exhausted queue never
    # produces a progress value > 1.0.
    cur = min(max(0, st.session_state.get("idx", 0)), len(work) - 1)
    st.progress((cur + 1) / len(work), text=f"Row {cur + 1} of {len(work)} in queue")

    idx = _navigate(len(work))
    row = work.iloc[idx]
    asin = str(row["asin"])

    left, right = st.columns([1.5, 1])

    with left:
        media_path = Path(html_dir) / f"{asin}.html" if html_dir else None
        media = (
            _quick_media(str(media_path), media_path.stat().st_mtime)
            if media_path and media_path.exists() else {"image": None, "price": None}
        )
        thumb, head = st.columns([1, 4])
        with thumb:
            image_url = media["image"] or _clean(row.get("keepa_main_image_url"))
            if image_url:
                st.image(image_url, use_container_width=True)
        with head:
            title = _clean(row.get("title")) or "(no title)"
            st.markdown(f"### {_highlight_count(title, row)}")
            if media["price"]:
                st.markdown(f"**{media['price']}**")
        badges = [f"`{asin}`", f"[🔗 Amazon]({AMAZON_URL.format(asin=asin)})"]
        if _clean(row.get("class_label")):
            badges.append(f"class: **{row.get('class_label')}**")
        if _clean(row.get("count_label")) and row.get("count_label") != "n/a":
            badges.append(f"count: **{row.get('count_label')}**")
        st.markdown(" · ".join(badges))
        _annotation_status(annots, asin)
        _evidence_panel(row)
        _catalog_panel(row)
        _text_panel(row)

    with right:
        _label_form(row, asin, annots, annotations_path, gold_path, annotator)

    st.divider()
    _scraped_page_panel(asin, html_dir)


def _annotation_status(annots: pd.DataFrame, asin: str) -> None:
    if annots is None or len(annots) == 0:
        return
    sub = annots[annots["asin"].astype(str) == asin]
    if sub.empty:
        return
    names = ", ".join(sorted({str(a) for a in sub["annotator"].dropna().unique()}))
    if names:
        st.caption(f"👥 {sub['annotator'].nunique()} label(s): {names}")


def _advance_and_save(
    annotations_path: Path,
    gold_path: Path,
    record: dict,
    n_rows: int,
) -> None:
    _persist_annotation(annotations_path, gold_path, record)
    if st.session_state.get("idx", 0) < n_rows - 1:
        st.session_state.idx += 1
    st.rerun()


def _prefill(annots: pd.DataFrame, asin: str, annotator: str, row: pd.Series):
    """Defaults: my own prior label → any existing annotation → rule prediction."""
    er = None
    if annots is not None and len(annots):
        sub = annots[annots["asin"].astype(str) == asin].copy()
        mine = sub[sub["annotator"].astype(str) == str(annotator)]
        if len(mine):
            er = mine.iloc[-1]
        elif len(sub):
            sub["_ts"] = pd.to_numeric(sub["ts"], errors="coerce")
            er = sub.sort_values("_ts", na_position="first").iloc[-1]
    if er is not None:
        return (
            _as_pure_str(er.get("is_pure_bath_bomb_gold")),
            _as_int(er.get("n_bomb_balls_gold")),
            str(_clean(er.get("exclude_reason_gold")) or ""),
            str(_clean(er.get("notes")) or ""),
            _clean(er.get("annotator")),
        )
    return (
        _as_pure_str(row.get("is_pure_bath_bomb")),
        _as_int(row.get("n_bomb_balls")),
        str(_clean(row.get("exclude_reason")) or ""),
        "",
        None,
    )


def _label_form(
    row: pd.Series,
    asin: str,
    annots: pd.DataFrame,
    annotations_path: Path,
    gold_path: Path,
    annotator: str,
) -> None:
    st.markdown("### ✍️ Your label")
    def_pure, def_n, def_ex, def_notes, prev_by = _prefill(annots, asin, annotator, row)
    def_ex = def_ex if def_ex in EXCLUDE_OPTIONS else ""
    if prev_by and str(prev_by) == str(annotator):
        st.info("You already labeled this — saving overwrites your label.")

    n_rows = st.session_state.get("_n_rows", 1)

    # --- One-click accept of the rule prediction ---
    pred_pure = _as_pure_str(row.get("is_pure_bath_bomb"))
    pred_n = _as_int(row.get("n_bomb_balls"))
    if st.button(
        f"✅ Accept prediction ({pred_pure}"
        + (f", n={pred_n}" if pred_pure == "true" and pred_n else "")
        + ")",
        type="primary",
        use_container_width=True,
        disabled=not annotator,
        help="Save the rule prediction as your label and advance" if annotator
        else "Enter your annotator name in the sidebar first",
    ):
        record = {
            "asin": asin,
            "stratum": _clean(row.get("class_label")) or _clean(row.get("stratum")) or "",
            "is_pure_bath_bomb_gold": "" if pred_pure == "unsure" else pred_pure,
            "n_bomb_balls_gold": pred_n if (pred_pure == "true" and pred_n) else "",
            "exclude_reason_gold": str(_clean(row.get("exclude_reason")) or "") if pred_pure == "false" else "",
            "notes": "accepted rule prediction",
            "annotator": annotator,
        }
        _advance_and_save(annotations_path, gold_path, record, n_rows)

    st.divider()

    # Widgets are keyed by ASIN so each product gets a fresh form seeded from its
    # own defaults — no values carry over to the next row after a submit.
    with st.form(f"label_form_{asin}", clear_on_submit=False):
        is_pure = st.radio(
            "is_pure_bath_bomb", PURE_OPTIONS, index=PURE_OPTIONS.index(def_pure),
            horizontal=True, key=f"pure_{asin}",
        )
        n_balls = st.number_input(
            "n_bomb_balls (only for pure=true)",
            min_value=0, max_value=5000, value=def_n or 0, step=1, key=f"n_{asin}",
        )
        exclude = st.selectbox(
            "exclude_reason (only for pure=false)",
            EXCLUDE_OPTIONS, index=EXCLUDE_OPTIONS.index(def_ex), key=f"ex_{asin}",
        )
        notes = st.text_area("notes", value=def_notes, height=80, key=f"notes_{asin}")
        submitted = st.form_submit_button(
            "💾 Save & next", type="primary", use_container_width=True, disabled=not annotator
        )

    if not annotator:
        st.warning("Enter an **annotator name** in the sidebar before saving.")

    if submitted:
        n_out = n_balls if (is_pure == "true" and n_balls > 0) else ""
        if is_pure == "true" and n_balls == 0:
            st.warning("pure=true but n_bomb_balls=0 — saved with a blank count.")
        record = {
            "asin": asin,
            "stratum": _clean(row.get("class_label")) or _clean(row.get("stratum")) or "",
            "is_pure_bath_bomb_gold": "" if is_pure == "unsure" else is_pure,
            "n_bomb_balls_gold": n_out,
            "exclude_reason_gold": exclude if is_pure == "false" else "",
            "notes": notes,
            "annotator": annotator,
        }
        _advance_and_save(annotations_path, gold_path, record, n_rows)


# --------------------------------------------------------------------------- #
# Dashboard tab
# --------------------------------------------------------------------------- #
def _pct(x) -> str:
    return "—" if x is None else f"{x:.1%}"


def _num(x) -> str:
    return "—" if x is None else f"{x:.2f}"


def _agreement(sub: pd.DataFrame) -> tuple[int, float | None, int, float | None]:
    """(n purity-labeled, purity agree, n count-labeled, count exact) vs the rules."""
    if sub is None or len(sub) == 0:
        return 0, None, 0, None
    pg = sub["is_pure_bath_bomb_gold"].map(_as_pure_str)
    pp = sub["is_pure_bath_bomb"].map(_as_pure_str)
    mask = pg.isin(["true", "false"])
    n_pure = int(mask.sum())
    p_agree = float((pg[mask] == pp[mask]).mean()) if n_pure else None
    both = sub[(pg == "true")].copy()
    both["_ng"] = both["n_bomb_balls_gold"].map(_as_int)
    both["_np"] = both["n_bomb_balls"].map(_as_int)
    both = both[both["_ng"].notna() & both["_np"].notna()]
    n_cnt = len(both)
    c_exact = float((both["_ng"] == both["_np"]).mean()) if n_cnt else None
    return n_pure, p_agree, n_cnt, c_exact


def _dashboard_tab(
    cfg: dict,
    annots: pd.DataFrame,
    gold: pd.DataFrame,
    queue: pd.DataFrame,
) -> None:
    st.markdown("### 📊 Progress")
    c1, c2, c3 = st.columns(3)
    c1.metric("Gold ASINs", gold["asin"].nunique() if len(gold) else 0)
    c2.metric("Annotations", len(annots))
    c3.metric("Annotators", annots["annotator"].nunique() if len(annots) else 0)

    if len(annots) == 0:
        st.info("No labels yet. Switch to the Review tab to start.")
        return

    # --- Inter-annotator agreement ---
    st.markdown("### 🤝 Inter-annotator agreement")
    iaa = A.compute_iaa(annots)
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Double-labeled ASINs", iaa["n_double_labeled"])
    a2.metric("Purity agreement", _pct(iaa["purity_agreement"]))
    a3.metric("Purity κ (Cohen)", _num(iaa["purity_kappa"]))
    a4.metric("Count agreement", _pct(iaa["count_agreement"]))
    if iaa["conflicts"]:
        st.markdown("**Disagreements to adjudicate**")
        st.dataframe(pd.DataFrame(iaa["conflicts"]), hide_index=True, use_container_width=True)

    # --- Agreement vs the rules ---
    st.markdown("### 🎯 Human gold vs rule prediction")
    full = _read(Path(cfg["paths"]["output_csv"]))
    if full is not None and len(gold):
        m = gold.merge(full[["asin", "is_pure_bath_bomb", "n_bomb_balls"]], on="asin", how="inner")
        n_p, pa, n_c, ce = _agreement(m)
        r1, r2, r3 = st.columns(3)
        r1.metric("Purity labeled", n_p)
        r2.metric("Purity agree", _pct(pa))
        r3.metric("Count exact", _pct(ce))

    # --- Breakdowns + downloads ---
    with st.expander("Breakdowns & raw data"):
        st.markdown("**Annotations by annotator**")
        st.dataframe(
            annots.groupby(annots["annotator"].fillna("(blank)")).size().rename("labels").reset_index(),
            hide_index=True, use_container_width=True,
        )
        st.markdown("**Adjudicated gold**")
        st.dataframe(gold, use_container_width=True, hide_index=True)
        d1, d2 = st.columns(2)
        d1.download_button(
            "⬇️ gold_labels.csv", gold.to_csv(index=False).encode("utf-8"),
            file_name="gold_labels.csv", mime="text/csv",
        )
        d2.download_button(
            "⬇️ annotations.csv", annots.to_csv(index=False).encode("utf-8"),
            file_name="annotations.csv", mime="text/csv",
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    cfg = load_config()
    gold_path = Path(cfg["paths"]["gold_csv"])
    annotations_path = Path(
        cfg["paths"].get("annotations_csv", ROOT / "data" / "gold" / "annotations.csv")
    )

    st.title("🛁 Bath Bomb Review Console")
    st.caption("Review the evidence, then confirm or correct the rule prediction. "
               f"Annotations → `{annotations_path}` · gold → `{gold_path}`.")

    annotator = st.sidebar.text_input("👤 Annotator name", value=st.session_state.get("annotator", ""))
    st.session_state.annotator = annotator.strip()
    annotator = st.session_state.annotator
    if not annotator:
        st.sidebar.warning("Set your name to enable saving.")

    queue, source = _load_queue(cfg)
    annots = A.load_annotations(annotations_path)
    gold = A.adjudicate(annots)

    my_asins = (
        set(annots[annots["annotator"].astype(str) == annotator]["asin"].astype(str))
        if annotator and len(annots) else set()
    )
    labeled_asins = set(annots["asin"].astype(str)) if len(annots) else set()

    work = _apply_filters(queue, my_asins)
    st.session_state["_n_rows"] = len(work)

    st.sidebar.markdown("---")
    sc1, sc2 = st.sidebar.columns(2)
    sc1.metric("In queue", len(work))
    sc2.metric("My labels", len(my_asins))
    remaining = queue["asin"].nunique() - len(labeled_asins & set(queue["asin"].astype(str)))
    st.sidebar.metric("Queue unlabeled", max(0, remaining))

    html_dir = cfg["paths"].get("html_dir", "")

    review, dash = st.tabs(["📝 Review & label", "📊 Dashboard"])
    with review:
        _review_tab(work, annots, annotations_path, gold_path, annotator, html_dir)
    with dash:
        _dashboard_tab(cfg, annots, gold, queue)


if __name__ == "__main__":
    main()
