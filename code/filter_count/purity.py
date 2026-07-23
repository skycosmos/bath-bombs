"""Pipeline 2 — purification: is a listing a countable bath bomb?

All word lists live in config (`purity.lexicon`). The ladder below is the fixed
decision logic; the words that fire each rule come entirely from the config.

Ladder (first match wins):
  1. craft_kit   DIY / make-your-own / mould              -> exclude "craft_kit"
  2. bundle      bomb + companion/hidden (or surprise+inside) -> exclude "bundle"
  3. substitute  steamer / salt / melt / tablet / ...      -> exclude "substitute"
  4. ingredient  citric acid / baking soda                 -> exclude "toiletry"
  5. toiletry    soap/shampoo/... with no bomb phrase       -> exclude "toiletry"
  6. pure        a bomb phrase is present                   -> is_pure = True
  7. unclassified nothing matched                          -> exclude "unclassified"

A `bomb_positive` phrase in the title rescues a listing from rules 3-5 (via
first-word adjudication), but not from craft_kit or bundle.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


def _compile(patterns) -> re.Pattern[str]:
    """Compile a list of regex fragments (or a single string) into one pattern."""
    if isinstance(patterns, str):
        patterns = [patterns]
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


@dataclass
class PurityResult:
    is_pure_bath_bomb: bool | None
    exclude_reason: str | None
    needs_review: bool
    purity_source: str


class Purifier:
    """Compiled-once classifier built from config; applied per row."""

    def __init__(self, cfg: dict):
        lex = cfg["purity"]["lexicon"]
        scope = cfg.get("scope", {})
        self.strict = bool(cfg["purity"].get("strict", True))

        self.bomb = _compile(lex["bomb_positive"])
        self.craft_kit = _compile(lex["craft_kit"])
        self.bundle_item = _compile(list(lex["companions"]) + list(lex["hidden_items"]))
        self.toiletry = _compile(list(lex["companions"]) + list(lex["toiletry_only"]))
        self.ingredient = _compile(lex["ingredients"])
        self.surprise = _compile(lex["surprise"])
        self.inside = _compile(lex["inside"])

        # Substitute patterns active for the current scope.
        subs = list(lex["substitute_always"])
        if not scope.get("include_shower_steamers", False):
            subs.append(lex["substitute_shower"])
        if not scope.get("include_bath_melts", False):
            subs.append(lex["substitute_melt"])
        if not scope.get("include_fizz_tablets", False):
            subs.append(lex["substitute_tablet"])
        if not scope.get("include_shower_bombs", True):
            subs.append(lex["substitute_shower_bomb"])
        self.substitute = _compile(subs)

    def _text(self, row) -> str:
        title = str(row.get("title") or row.get("keepa_title") or "")
        if self.strict:
            return title
        support = "\n".join(str(row.get(c) or "") for c in
                            ("feature", "product_description", "keepa_features", "keepa_description"))
        return f"{title}\n{support}"

    def classify(self, row) -> PurityResult:
        text = self._text(row)
        bomb = self.bomb.search(text)
        bomb_pos = bomb.start() if bomb else None

        # 1. Craft kit (unconditional).
        if self.craft_kit.search(text):
            return PurityResult(False, "craft_kit", False, "rule_craft_kit")

        # 2. Bundle — a bomb sold with a non-bomb item.
        if bomb_pos is not None:
            if self.bundle_item.search(text) or (self.surprise.search(text) and self.inside.search(text)):
                return PurityResult(False, "bundle", False, "rule_bundle")

        # 3. Substitute — excludes only if it appears before any bomb phrase.
        sub = self.substitute.search(text)
        if sub is not None and (bomb_pos is None or sub.start() < bomb_pos):
            return PurityResult(False, "substitute", False, "rule_substitute")

        # 4. Ingredient — raw materials before a bomb phrase -> toiletry.
        ing = self.ingredient.search(text)
        if ing is not None and (bomb_pos is None or ing.start() < bomb_pos):
            return PurityResult(False, "toiletry", False, "rule_ingredient")

        # 5. Toiletry — companion word with no bomb phrase.
        if bomb_pos is None and self.toiletry.search(text):
            return PurityResult(False, "toiletry", False, "rule_toiletry")

        # 6. Pure.
        if bomb_pos is not None:
            return PurityResult(True, None, False, "rule_positive")

        # 7. Unclassified.
        return PurityResult(False, "unclassified", True, "rule_unclassified")


def build_purifier(cfg: dict) -> Purifier:
    return Purifier(cfg)
