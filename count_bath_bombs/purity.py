from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

# --- Exclude: not finished pure bath bombs ---
KIT_PATTERNS = [
    r"\bcraft kit\b",
    r"\bdiy\b",
    r"\bmake your own\b",
    r"\bmake[- ]your[- ]own\b",
    r"\bstarter kit\b",
    r"\bbath bomb kit\b",
    r"\bbath fizzie kit\b",
    r"\bmoulds?\b",
    r"\bmolds?\b",
    r"\bbook\s*&\s*kit\b",
    r"\bbaking soda\b",
    r"\bcitric acid\b",
]

MIXED_COMPANION = re.compile(
    r"\b(?:soap slices?|hand soaps?|bar soaps?|candles?|lotions?|shampoo|"
    r"body butter|bath creamers?)\b",
    re.I,
)
GIFTISH = re.compile(r"\bgift (?:set|pack)\b|\bwrapped gift\b", re.I)

NOT_BOMB_PATTERNS = [
    r"\bdiffuser\b",
    r"\bpedicure\b",
    r"\bbath beads?\b",
    r"\bbath salts?\b",
    r"\bbath powder\b",
    r"\bcleansing lotion\b",
    r"\bdry shampoo\b",
    r"\bbubble bath\b(?!.*bomb)",
]

BOMB_POSITIVE = [
    r"\bbath bombs?\b",
    r"\bbath fizz(?:y|ies|er|ers)?\b",
    r"\bbath balls?\b",
    r"\bbath blasters?\b",
    r"\bshower bombs?\b",
    r"\baromabombs?\b",
]


def _compile(patterns: list[str]) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


KIT_RE = _compile(KIT_PATTERNS)
NOT_BOMB_RE = _compile(NOT_BOMB_PATTERNS)
BOMB_POS_RE = _compile(BOMB_POSITIVE)
TABLET_RE = re.compile(r"\b(?:fizz\s+)?tablets?\b|\bbath drops?\b", re.I)
MELT_RE = re.compile(r"\bbath melts?\b", re.I)


@dataclass
class PurityResult:
    is_pure_bath_bomb: bool | None
    exclude_reason: str | None
    needs_review: bool
    purity_source: str


def _title_text(row: pd.Series) -> str:
    return str(row.get("title") or row.get("html_title") or "")


def _support_text(row: pd.Series) -> str:
    """Bullets/features/description — used sparingly in strict mode."""
    parts = [
        str(row.get("feature") or ""),
        str(row.get("product_description") or ""),
        str(row.get("html_bullets") or ""),
        str(row.get("html_description") or ""),
    ]
    return "\n".join(parts)


def classify_purity(row: pd.Series, scope: dict, purity_cfg: dict | None = None) -> PurityResult:
    """
    purity.strict (default True):
      - Kit / not-bomb / positive bomb language judged primarily from TITLE.
      - Mixed-set companion products may use bullets only if title looks gift-like
        or already mentions a companion. Avoids false excludes from brand/feature text.
    purity.strict False (lenient): title + support text for all checks.
    """
    purity_cfg = purity_cfg or {}
    strict = bool(purity_cfg.get("strict", True))

    title = _title_text(row)
    support = _support_text(row)
    exclude_text = title if strict else f"{title}\n{support}"
    positive_text = title if strict else f"{title}\n{support}"

    if KIT_RE.search(exclude_text):
        return PurityResult(False, "kit", False, "rule_kit")

    # Mixed sets
    title_companion = MIXED_COMPANION.search(title)
    support_companion = MIXED_COMPANION.search(support)
    giftish_title = GIFTISH.search(title)

    if title_companion:
        return PurityResult(False, "mixed_set", False, "rule_mixed")
    if giftish_title and support_companion:
        return PurityResult(False, "mixed_set", False, "rule_mixed_gift_bullets")
    if not strict and support_companion and (giftish_title or BOMB_POS_RE.search(title)):
        return PurityResult(False, "mixed_set", False, "rule_mixed_lenient")
    if giftish_title and not BOMB_POS_RE.search(title):
        return PurityResult(False, "mixed_set", True, "rule_mixed_weak")

    if NOT_BOMB_RE.search(exclude_text) and not BOMB_POS_RE.search(title):
        return PurityResult(False, "not_bath_bomb", False, "rule_not_bomb")

    if not scope.get("include_fizz_tablets", False) and TABLET_RE.search(title):
        if not BOMB_POS_RE.search(title):
            return PurityResult(False, "not_bath_bomb", False, "rule_tablet")

    if not scope.get("include_bath_melts", False) and MELT_RE.search(title):
        if not BOMB_POS_RE.search(title):
            return PurityResult(False, "not_bath_bomb", False, "rule_melt")

    if not scope.get("include_shower_bombs", True):
        if re.search(r"\bshower bombs?\b", title, re.I) and not re.search(
            r"\bbath bombs?\b", title, re.I
        ):
            return PurityResult(False, "not_bath_bomb", False, "rule_shower")

    if BOMB_POS_RE.search(positive_text):
        return PurityResult(True, None, False, "rule_positive")

    return PurityResult(None, "unclear", True, "rule_unclear")


def apply_purity(
    df: pd.DataFrame,
    scope: dict,
    purity_cfg: dict | None = None,
) -> pd.DataFrame:
    out = df.copy()
    results = [classify_purity(row, scope, purity_cfg) for _, row in out.iterrows()]
    out["is_pure_bath_bomb"] = [r.is_pure_bath_bomb for r in results]
    out["exclude_reason"] = [r.exclude_reason for r in results]
    out["needs_review"] = [r.needs_review for r in results]
    out["purity_source"] = [r.purity_source for r in results]
    return out
