"""Capability-limited query-only universal-trigger refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class PublicResponse:
    task_score: float
    correct: bool
    response_excerpt: str


class QueryOracle(Protocol):
    @property
    def queries_used(self) -> int: ...

    def query(self, rows: Sequence[dict], trigger: str) -> list[PublicResponse]: ...


def response_attack_score(responses: Sequence[PublicResponse]) -> float:
    if not responses:
        raise ValueError("oracle returned no public responses")
    return sum(1.0 - float(item.task_score) for item in responses) / len(responses)


def query_only_search(
    oracle: QueryOracle,
    rows: Sequence[dict],
    candidates: Sequence[str],
    *,
    query_budget: int,
) -> dict:
    """Select a universal trigger using only public output-derived scores."""
    if not rows or not candidates:
        raise ValueError("rows and candidates must be non-empty")
    records = []
    best = None
    for candidate_id, trigger in enumerate(dict.fromkeys(candidates)):
        projected = oracle.queries_used + len(rows)
        if projected > int(query_budget):
            break
        responses = oracle.query(rows, trigger)
        score = response_attack_score(responses)
        record = {
            "candidate_id": candidate_id,
            "trigger": trigger,
            "public_attack_score": score,
            "queries_after": oracle.queries_used,
        }
        records.append(record)
        if best is None or (score, -candidate_id) > (
            best["public_attack_score"],
            -best["candidate_id"],
        ):
            best = record
    if best is None:
        raise RuntimeError("query budget cannot evaluate one candidate")
    return {
        "schema_version": 1,
        "algorithm": "query_only_universal_candidate_search",
        "selected_trigger": best["trigger"],
        "selected_public_attack_score": best["public_attack_score"],
        "queries_used": oracle.queries_used,
        "query_budget": int(query_budget),
        "candidate_records": records,
        "target_information_used": ["task_score", "correct", "response_excerpt"],
    }
