import math

import pytest

from src.eval.phase2_theory_audit import audit, horizon_denominator


def test_horizon_denominator_rejects_unstable_values() -> None:
    with pytest.raises(ValueError):
        horizon_denominator(1.0)


def test_legacy_horizons_do_not_share_fixed_constants() -> None:
    config = {
        "legacy_horizon_claims": {
            "benign_rho": 0.98,
            "benign_horizon": 1.17e6,
            "attacked_rho": 0.32,
            "attacked_horizon": 45,
        }
    }
    foundation = {
        "model_id": "example/model",
        "model_revision": "a" * 40,
        "A_log_sha256": "b" * 64,
        "layers": [{"parameter": "A_log", "abs_A_max": 16.0}],
    }
    report = audit(config, foundation)
    legacy = report["legacy_horizon_audit"]
    assert legacy["claims_reconcile_with_fixed_constants"] is False
    assert legacy["implied_numerator_ratio"] > 400
    assert legacy["attacked_horizon_if_constants_fixed_from_benign"] == pytest.approx(
        20744.76, rel=1e-4
    )


def test_near_critical_approximation_is_not_used_at_point_32() -> None:
    exact = horizon_denominator(0.32)
    assert exact == pytest.approx(1.139434283)
    assert not math.isclose(exact, 0.68, rel_tol=0.1)
