"""Adversarial-benign trigger construction for availability red teaming."""

from __future__ import annotations

from typing import Any

from src.experiments.phase3_attacks import optimize_universal_trigger


def optimize_adversarial_benign_trigger(
    loaded: Any,
    rows: list[dict[str, Any]],
    detector: dict[str, Any],
    *,
    seed: int,
    trigger_tokens: int,
    steps: int,
    candidates_per_step: int,
    rho_weight: float,
    task_preservation_weight: float,
    hazard_weight: float,
) -> dict[str, Any]:
    """Maximize hazard while minimizing teacher-forced benign task loss.

    The shared HotFlip optimizer minimizes its objective. A negative task
    weight changes ``-task_weight * loss`` into a positive loss penalty, while
    a negative hazard weight rewards high frozen-detector hazard. This is an
    availability attack, not an evasion attack.
    """
    if min(rho_weight, task_preservation_weight, hazard_weight) <= 0:
        raise ValueError("Phase 5 objective weights must be positive magnitudes")
    record = optimize_universal_trigger(
        loaded,
        rows,
        seed=int(seed),
        trigger_tokens=int(trigger_tokens),
        steps=int(steps),
        rho_weight=float(rho_weight),
        task_weight=-float(task_preservation_weight),
        hazard_weight=-float(hazard_weight),
        detector=detector,
        candidates_per_step=int(candidates_per_step),
        progress_label=f"adversarial-benign seed={seed}",
    )
    record.update(
        {
            "attack_family": "adversarial_benign_false_positive",
            "intent": "preserve benign task loss while increasing frozen-detector hazard",
            "selection_split": "validation",
        }
    )
    return record


def select_benign_candidate(
    candidates: list[dict[str, Any]],
    *,
    maximum_task_score_drop: float,
) -> dict[str, Any]:
    """Select on validation only, prioritizing feasible false-block pressure."""
    if not candidates:
        raise ValueError("no adversarial-benign candidates were evaluated")
    feasible = [
        row
        for row in candidates
        if float(row["validation_task_score_drop"]) <= maximum_task_score_drop
    ]
    pool = feasible or candidates
    selected = max(
        pool,
        key=lambda row: (
            float(row["validation_hard_block_rate"]),
            float(row["validation_window_fpr"]),
            -float(row["validation_task_score_drop"]),
            -int(row["seed"]),
        ),
    )
    return {
        **selected,
        "benignity_constraint_satisfied": bool(feasible),
        "maximum_validation_task_score_drop": float(maximum_task_score_drop),
    }
