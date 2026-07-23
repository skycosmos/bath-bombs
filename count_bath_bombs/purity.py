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
#   1. CRAFT_KIT   DIY / make-your-own / mould               → exclude "craft_kit"
#   2. BUNDLE      BOMB ∧ (companion∪hidden ∨ surprise∧inside) → exclude "bundle"
#   3. SUBSTITUTE  steamer/salt/melt/tablet/…                 → exclude "substitute"  (first-word adj)
#   4. INGREDIENT  citric acid / baking soda                 → exclude "toiletry"    (first-word adj)
#   5. TOILETRY    soap/shampoo/lotion/…  (no bomb phrase)    → exclude "toiletry"
#   6. PURE        bath bomb / bath fizzy / …                → is_pure=True
#   7. UNCLASSIFIED everything else                          → exclude "unclassified" (is_pure=False)
#
# Notes:
#  - "Bath Bomb Kit" is a finished gift set, NOT a craft kit — only real DIY
#    markers (make your own / DIY / craft kit / making kit / mould) exclude.
#  - soap is a normal companion word (guarded against "Soap Co" / "soap-free" /
#    "soapberry"), so soap + a bomb phrase → bundle.
#  - INGREDIENT and SUBSTITUTE use first-word adjudication: the term excludes
#    only if it appears before any bomb phrase in the title.
# --------------------------------------------------------------------------- #

# 1. Craft kits / DIY — supplies to *make* bombs, not finished bombs.
CRAFT_KIT_PATTERNS = [
    r"\bcraft kit\b",
    r"\bdiy\b",
    r"\bmake your own\b",
    r"\bmake[- ]your[- ]own\b",
    r"\bmaking kit\b",
    r"\bmoulds?\b",
    r"\bmolds?\b",
    r"\bbook\s*&\s*kit\b",
]

# 6. Positive: a countable, molded bath-bomb unit. "bath fizz*" counts as a bomb.
BOMB_POSITIVE = [
    r"\bbath bombs?\b",
    r"\bbath fizz(?:y|ies|er|ers)?\b",
    r"\bbath bursts?\b",
    r"\bbath blasters?\b",
    r"\bshower bombs?\b",
    r"\baromabombs?\b",
]

# Guarded soap: match soap(s) but not soap-free / soapberry-nut-wort / "Soap (&) Co".
_SOAP = r"\bsoaps?\b(?!\s*-?\s*free)(?!\s*&?\s*co(?:mpany|\.)?\b)"

# Companions bundled *with* a bath bomb. Soap is a normal companion (guarded so
# brand/ingredient uses like "Soap Co" / "soap-free" don't fire).
COMPANION_PATTERNS = [
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

# Toiletry-only words: exclude a *standalone* product (no bomb phrase) but do NOT
# trigger a bundle — "Cauldron Bath Bomb" / "Magic Bath Balls … Bath Bombs" stay
# PURE, while a lone "Bath Ball" / "Witch Cauldron" is a toiletry.
TOILETRY_ONLY_PATTERNS = [
    r"\bbath balls?\b",
    r"\bcauldrons?\b",
]

# 2. BUNDLE hidden items + surprise context.
HIDDEN_ITEM_PATTERNS = [r"\bnecklaces?\b", r"\brings?\b", r"\btoys?\b"]

# 4. INGREDIENT: raw materials ("Citric Acid ... for Bath Bombs") — reported as toiletry.
INGREDIENT_PATTERNS = [r"\bcitric acid\b", r"\bbaking soda\b"]

# 3. SUBSTITUTE: product forms used instead of a bath bomb.
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
# Bundle triggers = companions (incl. soap) + hidden items (NOT toiletry-only words).
BUNDLE_ITEM_RE = _compile(COMPANION_PATTERNS + HIDDEN_ITEM_PATTERNS)
# Toiletry (no-bomb exclusion) = companions + toiletry-only words.
TOILETRY_RE = _compile(COMPANION_PATTERNS + TOILETRY_ONLY_PATTERNS)
INGREDIENT_RE = _compile(INGREDIENT_PATTERNS)
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
    return str(row.get("title") or row.get("keepa_title") or "")


def _support_text(row: pd.Series) -> str:
    """Bullets/features/description — used only in lenient mode."""
    parts = [
        str(row.get("feature") or ""),
        str(row.get("product_description") or ""),
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

    # 1. Craft kit — supplies to make bombs (unconditional).
    if CRAFT_KIT_RE.search(text):
        return PurityResult(False, "craft_kit", False, "rule_craft_kit")

    # 2. Bundle — a bath bomb sold together with a non-bomb item.
    if bomb_pos is not None:
        has_item = BUNDLE_ITEM_RE.search(text) or (
            SURPRISE_RE.search(text) and INSIDE_RE.search(text)
        )
        if has_item:
            return PurityResult(False, "bundle", False, "rule_bundle")

    # 3. Substitute — first-word adjudication vs a bomb phrase.
    sub = _substitute_re(scope).search(text)
    if sub is not None and (bomb_pos is None or sub.start() < bomb_pos):
        return PurityResult(False, "substitute", False, "rule_substitute")

    # 4. Ingredient — raw citric acid / baking soda; first-word adjudication → toiletry.
    ing = INGREDIENT_RE.search(text)
    if ing is not None and (bomb_pos is None or ing.start() < bomb_pos):
        return PurityResult(False, "toiletry", False, "rule_ingredient")

    # 5. Toiletry — soap/shampoo/lotion/… with no bomb phrase (soap + bomb → PURE).
    if bomb_pos is None and TOILETRY_RE.search(text):
        return PurityResult(False, "toiletry", False, "rule_toiletry")

    # 6. Pure bath bomb.
    if bomb_pos is not None:
        return PurityResult(True, None, False, "rule_positive")

    # 7. Unclassified — no bomb phrase and no other signal; excluded, flag for review.
    return PurityResult(False, "unclassified", True, "rule_unclassified")
