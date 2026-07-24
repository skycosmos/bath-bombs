"""Pipeline 2 — purification: is a listing a countable bath bomb?

All word lists live in config (`purity.lexicon`). The ladder below is the fixed
decision logic; the words that fire each rule come entirely from the config.

A listing is PURE only if a bath-bomb phrase leads the title — nothing that
signals a different product precedes it. First split on whether the title has a
bomb phrase at all, then run the bomb-present ladder:

  NO bomb phrase -> exclude "no-bath-bomb"   (always, even for salts/steamers)
  bomb phrase present (ladder, first match wins):
      1. craft_kit   DIY / make-your-own / mould          -> exclude "craft_kit"
      2. bundle      companion, or "surprise ... inside"  -> exclude "bundle"
      3. substitute  steamer/salt/melt/tablet BEFORE bomb -> exclude "substitute"
      4. ingredient  citric acid / baking soda BEFORE bomb -> exclude "ingredients"
      5. pure        nothing above fired                  -> is_pure = True
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
        self.strict = bool(cfg["purity"].get("strict", True))

        self.bomb = _compile(lex["bomb_positive"])
        self.craft_kit = _compile(lex["craft_kit"])
        self.companion = _compile(lex["companions"])  # incl. hidden items (necklace/ring/toy)
        self.ingredient = _compile(lex["ingredients"])
        self.substitute = _compile(lex["substitute"])
        self.surprise = _compile(lex["surprise"])
        self.inside = _compile(lex["inside"])

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

        # No bomb wording anywhere -> "no-bath-bomb" (always, even for a lone
        # substitute like "Bath Salts" — substitute only applies vs a bomb phrase).
        if bomb_pos is None:
            return PurityResult(False, "no-bath-bomb", True, "rule_no_bomb")

        # A bomb phrase is present. Ladder, first match wins.
        # 1. Craft kit — a kit to MAKE bombs (DIY / mould / making kit).
        if self.craft_kit.search(text):
            return PurityResult(False, "craft_kit", False, "rule_craft_kit")

        # 2. Bundle — a bomb sold with a companion item, or a surprise hidden inside.
        if self.companion.search(text) or (self.surprise.search(text) and self.inside.search(text)):
            return PurityResult(False, "bundle", False, "rule_bundle")

        # 3. Substitute — a substitute form appearing BEFORE the bomb phrase.
        sub = self.substitute.search(text)
        if sub is not None and sub.start() < bomb_pos:
            return PurityResult(False, "substitute", False, "rule_substitute")

        # 4. Ingredient wording BEFORE the bomb phrase -> "ingredients".
        ing = self.ingredient.search(text)
        if ing is not None and ing.start() < bomb_pos:
            return PurityResult(False, "ingredients", True, "rule_ingredient")

        # 5. A bomb phrase leads and nothing above fired -> pure.
        return PurityResult(True, None, False, "rule_positive")


def build_purifier(cfg: dict) -> Purifier:
    return Purifier(cfg)
