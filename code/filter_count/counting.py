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

# Number sources -> the candidate column produced by candidates().
_SRC_COL = {
    "title": "cand_title", "bullets": "cand_bullets", "size": "cand_size", "description": "cand_description",
    "number_of_items": "cand_number_of_items", "keepa_number_of_items": "cand_keepa_number_of_items",
    "keepa_package_quantity": "cand_keepa_package_qty", "label_unit_num": "cand_label_unit_num",
    "unit_num": "cand_unit_num",
}
_TEXT = ("title", "bullets", "size", "description")
_CATALOG = ("number_of_items", "keepa_number_of_items", "keepa_package_quantity", "unit_num")

# Confidence ladder — agreement-aware adjustment steps up/down this list.
_CONF_LEVELS = ("low", "medium", "high")


def _step_conf(level: str, delta: int) -> str:
    try:
        i = _CONF_LEVELS.index(level)
    except ValueError:
        return level
    return _CONF_LEVELS[max(0, min(len(_CONF_LEVELS) - 1, i + delta))]


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
        self.conf: dict[str, str] = c["confidence"]
        self.weights: dict[str, int] = c.get("source_weights", {})
        self.conflict_needs_review = bool(c.get("conflict_needs_review", True))
        self.max_count = int(c.get("max_count", 1000))
        self.default_single = int(c.get("default_single", 1))

        # Spelled-out numbers ("six" -> 6). Two matchers: a word next to a
        # bomb/pack/piece noun, and a word confirmed by a digit ("Eight (8)").
        self.number_words = {str(k).lower(): int(v) for k, v in (c.get("number_words") or {}).items()}
        self._word_unit_re = self._word_paren_re = None
        if self.number_words:
            alt = "|".join(re.escape(w) for w in sorted(self.number_words, key=len, reverse=True))
            self._word_unit_re = re.compile(
                rf"\b(?P<w>{alt})\s+(?:(?:bath|shower)\s+)?"
                r"(?:pack|count|pcs|pieces?|bombs?|balls?|fizz(?:y|ies|ers?)?|blasters?)\b",
                re.IGNORECASE,
            )
            self._word_paren_re = re.compile(rf"\b(?P<w>{alt})\s*\((?P<n>\d+)\)", re.IGNORECASE)

    def _cap(self, v):
        """Drop implausibly large counts (barcodes, model numbers)."""
        return int(v) if (v is not None and 0 < v <= self.max_count) else None

    # -- candidate extraction ------------------------------------------------ #
    def _parse(self, text: str) -> list[tuple[int, str]]:
        if not text or not isinstance(text, str):
            return []
        found: list[tuple[int, str]] = []
        for name, rx in self.patterns.items():
            for m in rx.finditer(text):
                if name == "pack_mult":               # "2 Pack of 6" -> 2 x 6 = 12
                    found.append((int(m.group("a")) * int(m.group("b")), "pack_mult"))
                    continue
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
                elif name == "x_weight":
                    found.append((n, "n_x_weight"))
                elif name == "n_bath_bomb":
                    found.append((n, "near_bombs"))   # same strength as "N bombs"
                else:  # set_of, n_count
                    found.append((n, name))

        # Spelled-out numbers.
        if self._word_unit_re is not None:
            for m in self._word_unit_re.finditer(text):
                found.append((self.number_words[m.group("w").lower()], "word_count"))
            for m in self._word_paren_re.finditer(text):
                found.append((int(m.group("n")), "word_paren"))
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
            n = self._cap(n)
            out[f"cand_{name}"] = n
            out[f"cand_{name}_pattern"] = pat if n is not None else None
        out["cand_number_of_items"] = self._cap(_to_int(row.get("number_of_items")))
        out["cand_keepa_number_of_items"] = self._cap(_to_int(row.get("keepa_number_of_items")))
        out["cand_keepa_package_qty"] = self._cap(_to_int(row.get("keepa_package_quantity")))
        out["cand_unit_num"] = (
            self._cap(_to_int(row.get("unit_num")))
            if str(row.get("unit_text") or "").lower() in {"count", "each", "unit", "units"}
            else None
        )
        out["cand_label_unit_num"] = (
            self._cap(_to_int(row.get("label_unit_num")))
            if "count" in str(row.get("label_unit") or "").lower()
            else None
        )
        return out

    # -- resolution (weighted consensus) ------------------------------------- #
    def resolve(self, is_pure, cand: dict) -> dict[str, Any]:
        # (source, value) for every source that reported a positive number.
        reports = [
            (s, int(cand[c])) for s, c in _SRC_COL.items()
            if cand.get(c) is not None and _to_int(cand[c]) is not None
        ]
        multi = [(s, v) for s, v in reports if v > 1]
        ones = {s for s, v in reports if v == 1}
        # "N vs 1": a text multi-count exists AND a catalog field reads 1.
        seller_pack = bool(any(s in _TEXT for s, v in multi) and (ones & set(_CATALOG)))
        base = {"seller_counts_pack_as_one": seller_pack}

        if is_pure is not True:
            return {"n_bomb_balls": None, "count_confidence": "n/a", "count_source": None,
                    "count_unable": False, "count_conflict": False, "needs_review": False, **base}

        # No multi-count anywhere -> a single (an explicit "1") or an assumption.
        if not multi:
            text_one = any(s in ("title", "bullets", "size") for s in ones)
            if ones:
                conf = self.conf.get("single_default", "medium") if (text_one or "number_of_items" in ones) else "low"
                return self._done(self.default_single, "single_default", conf, base,
                                  conflict=False, needs_review=(conf == "low"))
            return self._done(self.default_single, "assumed_single",
                              self.conf.get("assumed_single", "low"), base,
                              conflict=False, needs_review=True)

        # Weighted consensus over the values >1 (1s never vote -> N-vs-1 accepts N).
        tally: dict[int, int] = {}
        best: dict[int, tuple[int, str]] = {}   # value -> (best single weight, source)
        for s, v in multi:
            w = self.weights.get(s, 1)
            tally[v] = tally.get(v, 0) + w
            if v not in best or w > best[v][0]:
                best[v] = (w, s)
        winner = max(
            tally,
            key=lambda v: (tally[v], any(s in _TEXT for s, vv in multi if vv == v), v),
        )
        conflict = len(tally) >= 2
        # A conflict only needs review when the winner does NOT clearly dominate
        # (a lone garbage catalog value vs a text consensus should not flag it).
        others = [tally[v] for v in tally if v != winner]
        dominant = (not others) or tally[winner] >= 2 * max(others)
        needs_review = conflict and not dominant and self.conflict_needs_review
        src = best[winner][1]

        # Confidence starts from the winning source's reliability, then adjusts
        # for corroboration: >=2 independent sources on the winning value boost
        # it a level; a lone weight-1 source demotes it. An unresolved conflict
        # (needs_review) always lands at low.
        base_conf = self.conf.get(f"text_{src}" if src in _TEXT else src, "medium")
        if needs_review:
            conf = "low"
        else:
            backers = [s for s, v in multi if v == winner]    # sources on the winning value
            text_backed = any(s in _TEXT for s in backers)
            if len(backers) >= 2:                             # corroborated -> boost
                conf = _step_conf(base_conf, +1)
            elif best[winner][0] <= 1:                        # lone weight-1 source -> demote
                conf = _step_conf(base_conf, -1)
            else:
                conf = base_conf
            # Catalog fields share the package-count ("Pack of 1") bias, so their
            # mutual agreement is not independent -> it cannot reach "high".
            if conf == "high" and not text_backed:
                conf = "medium"
        return self._done(winner, src, conf, base, conflict=conflict, needs_review=needs_review)

    def _done(self, n, source, conf, base, *, conflict, needs_review) -> dict[str, Any]:
        return {
            "n_bomb_balls": int(n), "count_confidence": conf, "count_source": source,
            "count_unable": False, "count_conflict": conflict, "needs_review": needs_review, **base,
        }


def build_counter(cfg: dict) -> Counter:
    return Counter(cfg)
