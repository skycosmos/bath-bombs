"""Pipeline 3 — counting: how many bomb units are in a pure listing.

Regex patterns, their tie-break priority, the resolution order, and the
confidence tags all come from config (`counting.*`). This module only holds the
mechanics: scan the text channels for candidates, then walk the resolution
ladder and pick the winner. Non-pure listings get no count.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

# Resolution signal -> candidate column produced by extract().
_SIGNAL_COL = {
    "number_of_items": "cand_number_of_items",
    "keepa_number_of_items": "cand_keepa_number_of_items",
    "keepa_package_quantity": "cand_keepa_package_qty",
    "label_unit_num": "cand_label_unit_num",
}


def _to_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


class Counter:
    """Compiled-once counter built from config; applied per row."""

    def __init__(self, cfg: dict):
        c = cfg["counting"]
        self.channels: dict[str, list[str]] = c["text_channels"]
        self.patterns = {name: re.compile(rx, re.IGNORECASE) for name, rx in c["patterns"].items()}
        self.priority: dict[str, int] = c["pattern_priority"]
        self.order: list[str] = c["resolution_order"]
        self.conf: dict[str, str] = c["confidence"]
        self.default_single = int(c.get("default_single", 1))

    # -- candidate extraction ------------------------------------------------ #
    def _parse(self, text: str) -> list[tuple[int, str]]:
        if not text or not isinstance(text, str):
            return []
        found: list[tuple[int, str]] = []
        for name, rx in self.patterns.items():
            for m in rx.finditer(text):
                n = int(m.group("n"))
                if name == "count_near_bomb":
                    unit = m.group("unit").lower()
                    if unit == "pack" and n == 1:
                        continue
                    found.append((n, f"near_{unit}"))
                elif name == "pack_of":
                    if n > 1:
                        found.append((n, "pack_of"))
                elif name == "x_bombs":
                    found.append((n, "n_x_bombs"))
                else:  # set_of, n_count
                    found.append((n, name))
        return found

    def best_text_count(self, text: str) -> tuple[int | None, str | None]:
        hits = self._parse(text)
        if not hits:
            return None, None
        multi = [h for h in hits if h[0] > 1]
        if multi:
            return sorted(multi, key=lambda x: (self.priority.get(x[1], 9), -x[0]))[0]
        return sorted(hits, key=lambda x: (self.priority.get(x[1], 9), 999))[0]

    def candidates(self, row) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, cols in self.channels.items():
            text = "\n".join(str(row.get(c) or "") for c in cols)
            n, pat = self.best_text_count(text)
            out[f"cand_{name}"] = n
            out[f"cand_{name}_pattern"] = pat
        out["cand_number_of_items"] = _to_int(row.get("number_of_items"))
        out["cand_keepa_number_of_items"] = _to_int(row.get("keepa_number_of_items"))
        out["cand_keepa_package_qty"] = _to_int(row.get("keepa_package_quantity"))
        out["cand_unit_num"] = (
            _to_int(row.get("unit_num"))
            if str(row.get("unit_text") or "").lower() in {"count", "each", "unit", "units"}
            else None
        )
        out["cand_label_unit_num"] = (
            _to_int(row.get("label_unit_num"))
            if "count" in str(row.get("label_unit") or "").lower()
            else None
        )
        return out

    # -- resolution ---------------------------------------------------------- #
    def _text_winner(self, cand: dict) -> tuple[int | None, str | None]:
        """Best text multi-count across channels, in title>bullets>size>description order."""
        for ch in ("title", "bullets", "size", "description"):
            v = cand.get(f"cand_{ch}")
            if v is not None and v > 1:
                return int(v), ch
        return None, None

    def resolve(self, is_pure, cand: dict) -> dict[str, Any]:
        text_multi, _ = self._text_winner(cand)
        catalog_ones = [
            cand.get(k) for k in
            ("cand_number_of_items", "cand_keepa_number_of_items", "cand_keepa_package_qty", "cand_unit_num")
            if cand.get(k) == 1
        ]
        seller_pack = bool(text_multi and catalog_ones)

        if is_pure is not True:
            return {
                "n_bomb_balls": None, "count_confidence": "n/a", "count_source": None,
                "seller_counts_pack_as_one": seller_pack, "count_unable": False,
            }

        # Walk the configured resolution ladder — first signal with a value >1 wins.
        for signal in self.order:
            if signal == "text":
                n, ch = self._text_winner(cand)
                if n is not None:
                    return self._done(n, ch, self.conf.get(f"text_{ch}", "medium"), seller_pack)
            else:
                v = cand.get(_SIGNAL_COL.get(signal, ""))
                if v is not None and v > 1:
                    return self._done(int(v), signal, self.conf.get(signal, "medium"), seller_pack)

        # No multi-count. Fall back to an explicit "1", else assume a single.
        noi = cand.get("cand_number_of_items")
        keepa_noi = cand.get("cand_keepa_number_of_items")
        label_n = cand.get("cand_label_unit_num")
        text_one = cand.get("cand_title") == 1 or cand.get("cand_bullets") == 1 or cand.get("cand_size") == 1
        if noi == 1 or text_one or label_n == 1 or keepa_noi == 1:
            conf = self.conf.get("single_default", "medium") if (noi == 1 or text_one) else "low"
            return self._done(self.default_single, "single_default", conf, seller_pack)
        return self._done(self.default_single, "assumed_single", self.conf.get("assumed_single", "low"), seller_pack)

    def _done(self, n, source, conf, seller_pack) -> dict[str, Any]:
        return {
            "n_bomb_balls": int(n), "count_confidence": conf, "count_source": source,
            "seller_counts_pack_as_one": seller_pack, "count_unable": False,
        }


def build_counter(cfg: dict) -> Counter:
    return Counter(cfg)
