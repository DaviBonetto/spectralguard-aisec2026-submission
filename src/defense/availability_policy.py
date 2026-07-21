"""Streaming availability policies for SpectralGuard decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


PASS = "PASS"
FLAG_CONTINUE = "FLAG-CONTINUE"
HARD_BLOCK = "HARD-BLOCK"


@dataclass(frozen=True)
class PolicyOutcome:
    """Token-window decisions and session-level availability outcomes."""

    decisions: tuple[str, ...]
    first_flag_window: int | None
    first_block_window: int | None

    @property
    def survived(self) -> bool:
        return self.first_block_window is None


def algorithm2(scores: Iterable[float], *, threshold: float) -> PolicyOutcome:
    """Apply the original fail-closed policy: one alert blocks the session."""
    decisions: list[str] = []
    first_block: int | None = None
    for index, score in enumerate(scores):
        if first_block is not None or float(score) >= float(threshold):
            if first_block is None:
                first_block = index
            decisions.append(HARD_BLOCK)
        else:
            decisions.append(PASS)
    return PolicyOutcome(tuple(decisions), first_block, first_block)


def algorithm2b(
    scores: Iterable[float],
    *,
    threshold: float,
    k: int,
    w: int,
) -> PolicyOutcome:
    """Apply a k-of-w graduated PASS/FLAG-CONTINUE/HARD-BLOCK policy."""
    if not 1 < int(k) <= int(w):
        raise ValueError("graduated policy requires 1 < k <= w")
    alerts: list[bool] = []
    decisions: list[str] = []
    first_flag: int | None = None
    first_block: int | None = None
    for index, score in enumerate(scores):
        alert = float(score) >= float(threshold)
        alerts.append(alert)
        recent = sum(alerts[max(0, index - int(w) + 1) : index + 1])
        if first_block is not None or recent >= int(k):
            if first_block is None:
                first_block = index
            decisions.append(HARD_BLOCK)
        elif alert:
            if first_flag is None:
                first_flag = index
            decisions.append(FLAG_CONTINUE)
        else:
            decisions.append(PASS)
    return PolicyOutcome(tuple(decisions), first_flag, first_block)


def outcome_record(outcome: PolicyOutcome) -> dict[str, float | int | bool | None]:
    """Convert an outcome into stable session-level metrics."""
    total = len(outcome.decisions)
    flags = sum(decision == FLAG_CONTINUE for decision in outcome.decisions)
    blocks = sum(decision == HARD_BLOCK for decision in outcome.decisions)
    return {
        "window_count": total,
        "flag_windows": flags,
        "blocked_windows": blocks,
        "flag_window_rate": flags / total if total else 0.0,
        "blocked_window_rate": blocks / total if total else 0.0,
        "survived": outcome.survived,
        "first_flag_window": outcome.first_flag_window,
        "first_block_window": outcome.first_block_window,
    }
