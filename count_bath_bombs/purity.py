from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

# --------------------------------------------------------------------------- #
# Lexicons
#
# Detection runs on the TITLE in strict mode (default); lenient mode also scans
# the support text (bullets/description). The ladder in classify_purity():
#
#   1. CRAFT_KIT    kit/DIY/mould/baking soda/…          → exclude "craft_kit"
#   2. BUNDLE       BOMB ∧ (TOILETRY∪HIDDEN ∨ surprise∧inside) → exclude "bundle"
#   3a SUBSTITUTE   steamer/salt/melt/tablet/…           → exclude "substitute"  (first-word adj vs bomb)
#   3b TOILETRY     soap/shampoo/lotion/…                → exclude "toiletry"    (word present ⇒ exclude)
#   4. PURE         bath bomb / bath fizzy / …           → is_pure=True
#   5. UNCLASSIFIED everything else                      → is_pure=None, needs_review=True
#
# SUBSTITUTE and TOILETRY lexicons are mutually exclusive (no word in both).
# --------------------------------------------------------------------------- #

# 1. Craft kits / DIY — not a finished product.
CRAFT_KIT_PATTERNS = [
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

# 4. Positive: a countable, molded bath-bomb unit. "bath fizz*" counts as a bomb.
BOMB_POSITIVE = [
    r"\bbath bombs?\b",
    r"\bbath fizz(?:y|ies|er|ers)?\b",
    r"\bbath balls?\b",
    r"\bbath bursts?\b",
    r"\bbath blasters?\b",
    r"\bshower bombs?\b",
    r"\baromabombs?\b",
]

# 3b. TOILETRY: accompanying products. With a bomb phrase these become BUNDLE
# (step 2); reaching step 3b implies no bomb phrase, so presence ⇒ exclude.
# Guarded soap: match soap(s) but not soap-free / soapberry-nut-wort / "Soap Co".
_SOAP = r"\bsoaps?\b(?!\s*-?\s*free)(?!\s*&?\s*co(?:mpany|\.)?\b)"
TOILETRY_PATTERNS = [
    _SOAP,
    r"\bcandles?\b",
    r"\blotions?\b",
    r"\bcleansing lotion\b",
    r"\bshampoos?\b",
    r"\bdry shampoo\b",
    r"\bbody butters?\b",
    r"\bbath creamers?\b",
    r"\bbody wash\b",
    r"\bshower gel\b",
    r"\bconditioners?\b",
    r"\bdiffusers?\b",
    r"\bpedicure\b",
]

# 2. BUNDLE hidden items + surprise context.
HIDDEN_ITEM_PATTERNS = [r"\bnecklaces?\b", r"\brings?\b", r"\btoys?\b"]

# 3a. SUBSTITUTE: product forms used instead of a bath bomb.
#   - "always" families are excluded regardless of scope.
#   - scope-gated families are excluded only when their include_* flag is False.
SUBSTITUTE_ALWAYS = [
    r"\bbath salts?\b",
    r"\bbath powder\b",
    r"\bbath beads?\b",
    r"\bbubble bath\b(?!.*bomb)",
]
SUBSTITUTE_SHOWER = r"\bshower (?:steamer|melt|tablet|disc|fizzy|fizzies|fizzers?)s?\b"
SUBSTITUTE_MELT = r"\bbath melts?\b"
SUBSTITUTE_TABLET = r"\b(?:fizz )?tablets?\b|\bbath drops?\b"


def _compile(patterns: list[str]) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


CRAFT_KIT_RE = _compile(CRAFT_KIT_PATTERNS)
BOMB_POS_RE = _compile(BOMB_POSITIVE)
TOILETRY_RE = _compile(TOILETRY_PATTERNS)
HIDDEN_ITEM_RE = _compile(HIDDEN_ITEM_PATTERNS)
SURPRISE_RE = re.compile(r"\bsurprise\b", re.IGNORECASE)
INSIDE_RE = re.compile(r"\b(?:inside|hidden)\b", re.IGNORECASE)


def _substitute_re(scope: dict) -> re.Pattern[str]:
    """Active substitute patterns for the current scope (built per call)."""
    parts = list(SUBSTITUTE_ALWAYS)
    if not scope.get("include_shower_steamers", False):
        parts.append(SUBSTITUTE_SHOWER)
    if not scope.get("include_bath_melts", False):
        parts.append(SUBSTITUTE_MELT)
    if not scope.get("include_fizz_tablets", False):
        parts.append(SUBSTITUTE_TABLET)
    if not scope.get("include_shower_bombs", True):
        parts.append(r"\bshower bombs?\b")
    return _compile(parts)


@dataclass
class PurityResult:
    is_pure_bath_bomb: bool | None
    exclude_reason: str | None
    needs_review: bool
    purity_source: str


def _title_text(row: pd.Series) -> str:
    return str(row.get("title") or row.get("html_title") or row.get("keepa_title") or "")


def _support_text(row: pd.Series) -> str:
    """Bullets/features/description — used only in lenient mode."""
    parts = [
        str(row.get("feature") or ""),
        str(row.get("product_description") or ""),
        str(row.get("html_bullets") or ""),
        str(row.get("html_description") or ""),
        str(row.get("keepa_features") or ""),
        str(row.get("keepa_description") or ""),
    ]
    return "\n".join(parts)


def classify_purity(row: pd.Series, scope: dict, purity_cfg: dict | None = None) -> PurityResult:
    purity_cfg = purity_cfg or {}
    strict = bool(purity_cfg.get("strict", True))

    title = _title_text(row)
    text = title if strict else f"{title}\n{_support_text(row)}"

    bomb = BOMB_POS_RE.search(text)
    bomb_pos = bomb.start() if bomb else None

    # 1. Craft kit — unconditional.
    if CRAFT_KIT_RE.search(text):
        return PurityResult(False, "craft_kit", False, "rule_craft_kit")

    # 2. Bundle — a bath bomb sold together with a non-bomb item.
    if bomb_pos is not None:
        has_item = (
            TOILETRY_RE.search(text)
            or HIDDEN_ITEM_RE.search(text)
            or (SURPRISE_RE.search(text) and INSIDE_RE.search(text))
        )
        if has_item:
            return PurityResult(False, "bundle", False, "rule_bundle")

    # 3a. Substitute — first-word adjudication vs a bomb phrase.
    sub = _substitute_re(scope).search(text)
    if sub is not None and (bomb_pos is None or sub.start() < bomb_pos):
        return PurityResult(False, "substitute", False, "rule_substitute")

    # 3b. Toiletry — reaches here only without a bomb phrase, so presence ⇒ exclude.
    if TOILETRY_RE.search(text):
        return PurityResult(False, "toiletry", False, "rule_toiletry")

    # 4. Pure bath bomb.
    if bomb_pos is not None:
        return PurityResult(True, None, False, "rule_positive")

    # 5. Unclassified — route to review.
    return PurityResult(None, "unclassified", True, "rule_unclassified")


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
