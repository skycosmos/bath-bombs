from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

PROMPT_SECTION = {
    "system": "## System",
    "schema": "## Schema",
    "user": "## User template",
}


def load_prompt(path: str | Path) -> dict[str, str]:
    text = Path(path).read_text(encoding="utf-8")
    parts: dict[str, str] = {"system": "", "schema": "", "user": ""}
    current = None
    lines: list[str] = []
    for line in text.splitlines():
        if line.strip() in PROMPT_SECTION.values():
            if current:
                parts[current] = "\n".join(lines).strip()
            current = {v: k for k, v in PROMPT_SECTION.items()}[line.strip()]
            lines = []
        else:
            if current:
                lines.append(line)
    if current:
        parts[current] = "\n".join(lines).strip()
    system = parts["system"]
    if parts["schema"]:
        system = system + "\n\n" + parts["schema"]
    return {"system": system.strip(), "user_template": parts["user"]}


def render_user_prompt(template: str, row: dict[str, Any]) -> str:
    bullets = row.get("html_bullets") or row.get("feature") or ""
    if isinstance(bullets, str) and " | " in bullets:
        bullets = "\n".join(f"- {b}" for b in bullets.split(" | ") if b)
    description = row.get("html_description") or row.get("product_description") or ""
    return template.format(
        asin=row.get("asin", ""),
        title=row.get("title") or row.get("html_title") or "",
        number_of_items=row.get("html_number_of_items") or row.get("number_of_items") or "",
        unit_count=row.get("html_unit_count") or "",
        item_package_quantity=row.get("html_item_package_quantity") or "",
        size=row.get("html_size") or row.get("size") or "",
        item_weight=row.get("html_item_weight") or row.get("item_weight") or "",
        bullets=bullets or "(none)",
        description=(description or "")[:1200] or "(none)",
    )


def cache_key(prompt_version: str, model: str, temperature: float, seed: int, user_prompt: str) -> str:
    payload = json.dumps(
        {
            "prompt_version": prompt_version,
            "model": model,
            "temperature": temperature,
            "seed": seed,
            "user_prompt": user_prompt,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def call_llm_row(
    row: dict[str, Any],
    *,
    prompt_path: str | Path,
    prompt_version: str,
    model: str,
    temperature: float,
    seed: int,
    max_tokens: int,
    cache_dir: str | Path,
    log_path: str | Path,
    client=None,
) -> dict[str, Any]:
    """Return structured LLM fields; uses disk cache. client=None → dry-run stub."""
    prompt = load_prompt(prompt_path)
    user_prompt = render_user_prompt(prompt["user_template"], row)
    key = cache_key(prompt_version, model, temperature, seed, user_prompt)

    cache_dir = Path(cache_dir) / prompt_version
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{row.get('asin')}_{key[:16]}.json"
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    record: dict[str, Any] = {
        "asin": row.get("asin"),
        "prompt_version": prompt_version,
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "cache_key": key,
        "ts": time.time(),
    }

    if client is None:
        parsed = {
            "is_pure_bath_bomb": None,
            "n_bomb_balls": None,
            "exclude_reason": "unclear",
            "evidence": [],
            "confidence": "low",
            "llm_dry_run": True,
        }
        record["dry_run"] = True
        record["response"] = parsed
    else:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        parsed = _parse_json_response(content)
        parsed["llm_dry_run"] = False
        record["response_raw"] = content
        record["response"] = parsed
        if hasattr(response, "usage") and response.usage:
            record["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }

    cache_payload = {
        "asin": row.get("asin"),
        "cache_key": key,
        "prompt_version": prompt_version,
        "model": model,
        **parsed,
    }
    cache_file.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return cache_payload


def apply_llm_hard_cases(df, cfg: dict, client=None):
    import pandas as pd

    llm_cfg = cfg["llm"]
    if not llm_cfg.get("enabled", False):
        df = df.copy()
        df["llm_applied"] = False
        return df

    if "needs_llm" in df.columns and llm_cfg.get("only_needs_llm", llm_cfg.get("only_hard_cases", True)):
        targets = df.index[df["needs_llm"].fillna(False)].tolist()
    elif llm_cfg.get("only_hard_cases", True) and "is_hard_case" in df.columns:
        targets = df.index[df["is_hard_case"].fillna(False)].tolist()
    else:
        targets = df.index.tolist()

    out = df.copy()
    out["llm_is_pure_bath_bomb"] = None
    out["llm_n_bomb_balls"] = None
    out["llm_exclude_reason"] = None
    out["llm_confidence"] = None
    out["llm_applied"] = False

    for idx in targets:
        row = out.loc[idx].to_dict()
        result = call_llm_row(
            row,
            prompt_path=llm_cfg["prompt_path"],
            prompt_version=llm_cfg["prompt_version"],
            model=llm_cfg["model"],
            temperature=float(llm_cfg.get("temperature", 0)),
            seed=int(llm_cfg.get("seed", 0)),
            max_tokens=int(llm_cfg.get("max_tokens", 400)),
            cache_dir=cfg["paths"]["llm_cache_dir"],
            log_path=cfg["paths"]["llm_log"],
            client=client,
        )
        out.at[idx, "llm_is_pure_bath_bomb"] = result.get("is_pure_bath_bomb")
        out.at[idx, "llm_n_bomb_balls"] = result.get("n_bomb_balls")
        out.at[idx, "llm_exclude_reason"] = result.get("exclude_reason")
        out.at[idx, "llm_confidence"] = result.get("confidence")
        out.at[idx, "llm_applied"] = True

        if result.get("is_pure_bath_bomb") is not None:
            out.at[idx, "is_pure_bath_bomb"] = result.get("is_pure_bath_bomb")
            out.at[idx, "exclude_reason"] = result.get("exclude_reason")
            out.at[idx, "purity_source"] = f"llm:{llm_cfg['prompt_version']}"
        if result.get("n_bomb_balls") is not None:
            out.at[idx, "n_bomb_balls"] = result.get("n_bomb_balls")
            out.at[idx, "count_source"] = f"llm:{llm_cfg['prompt_version']}"
            out.at[idx, "count_confidence"] = result.get("confidence")
            out.at[idx, "is_hard_case"] = False
            out.at[idx, "needs_llm"] = False
            out.at[idx, "count_unable"] = False

    return out
