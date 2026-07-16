"""Canonical prompt-boundary handling shared by evaluation and attacks."""

from __future__ import annotations

import re
from typing import Any


def split_attack_anchor(row: dict[str, Any]) -> tuple[str, str, str]:
    """Split a clean prompt around its unique question, ignoring case only.

    Some frozen templates lowercase the first character of the stored question.
    Returning the matched prompt text preserves that intentional formatting.
    """
    prompt = str(row["clean_prompt"])
    question = str(row["question"])
    matches = list(re.finditer(re.escape(question), prompt, flags=re.IGNORECASE))
    if len(matches) != 1:
        prompt_id = row.get("prompt_id", "<unknown>")
        raise ValueError(
            f"question boundary is ambiguous for {prompt_id}: "
            f"expected 1 case-insensitive match, found {len(matches)}"
        )
    match = matches[0]
    return prompt[: match.start()], match.group(0), prompt[match.end() :]


def inject_trigger(row: dict[str, Any], trigger: str) -> str:
    """Insert an intervention immediately before the unique question boundary."""
    prompt = str(row["clean_prompt"])
    if not trigger:
        return prompt
    prefix, matched_question, suffix = split_attack_anchor(row)
    intervention = f"Intervening message:\n{trigger}\n\n{matched_question}"
    return prefix + intervention + suffix
