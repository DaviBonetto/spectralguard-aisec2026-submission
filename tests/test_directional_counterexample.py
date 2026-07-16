"""Tests for the Phase 2 structural completeness counterexample."""

import pytest

from src.attacks.directional_counterexample import wilson_interval


def test_wilson_extremes_are_not_degenerate() -> None:
    low_zero, high_zero = wilson_interval(0, 1000)
    low_one, high_one = wilson_interval(1000, 1000)
    assert low_zero == 0.0
    assert 0.003 < high_zero < 0.004
    assert 0.996 < low_one < 0.997
    assert high_one == 1.0


@pytest.mark.parametrize("successes,total", [(-1, 10), (11, 10), (0, 0)])
def test_wilson_rejects_invalid_counts(successes: int, total: int) -> None:
    with pytest.raises(ValueError):
        wilson_interval(successes, total)
