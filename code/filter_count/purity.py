"""Pipeline 2 — purification: is a listing a countable bath bomb?

All word lists live in config (`purity.lexicon`). The ladder below is the fixed
decision logic; the words that fire each rule come entirely from the config.

A listing is PURE only if a bath-bomb phrase leads the title — nothing that
signals a different product precedes it. Ladder (first match wins):
  1. craft_kit    DIY / make-your-own / mould                -> exclude "craft_kit"
  2. bundle       bomb + companion (or "surprise ... inside") -> exclude "bundle"
  3. substitute   steamer / salt / melt / tablet, at/before a bomb phrase -> "substitute"
  4. no bomb      no bomb phrase anywhere in the title        -> exclude "unclassified"
  5. ingredient   citric acid / baking soda before the bomb   -> exclude "unclassified"
  6. pure         a bomb phrase leads and nothing above fired -> is_pure = True
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

        # 1. Craft kit (unconditional).
        if self.craft_kit.search(text):
            return PurityResult(False, "craft_kit", False, "rule_craft_kit")

        # 2. Bundle — a bomb sold with a companion item, or a surprise hidden inside.
        if bomb_pos is not None:
            if self.companion.search(text) or (self.surprise.search(text) and self.inside.search(text)):
                return PurityResult(False, "bundle", False, "rule_bundle")

        # 3. Substitute — a substitute form at/before any bomb phrase (or no bomb).
        sub = self.substitute.search(text)
        if sub is not None and (bomb_pos is None or sub.start() < bomb_pos):
            return PurityResult(False, "substitute", False, "rule_substitute")

        # 4. No bomb wording anywhere -> unclassified.
        if bomb_pos is None:
            return PurityResult(False, "unclassified", True, "rule_no_bomb")

        # 5. Ingredient wording before the bomb phrase -> unclassified.
        ing = self.ingredient.search(text)
        if ing is not None and ing.start() < bomb_pos:
            return PurityResult(False, "unclassified", True, "rule_ingredient")

        # 6. A bomb phrase leads and nothing above fired -> pure.
        return PurityResult(True, None, False, "rule_positive")


def build_purifier(cfg: dict) -> Purifier:
    return Purifier(cfg)
