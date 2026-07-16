"""Reproduce Phase 2 norm measurements and legacy-horizon arithmetic."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

from src.utils.provenance import package_versions, sha256_file


def horizon_denominator(rho: float) -> float:
    """Return the exact positive denominator log(1/rho)."""
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must lie strictly between zero and one")
    return math.log(1.0 / rho)


def audit(
    config: dict[str, object], foundation: dict[str, object]
) -> dict[str, object]:
    claims = config["legacy_horizon_claims"]
    assert isinstance(claims, dict)
    benign_rho = float(claims["benign_rho"])
    attacked_rho = float(claims["attacked_rho"])
    benign_horizon = float(claims["benign_horizon"])
    attacked_horizon = float(claims["attacked_horizon"])
    benign_denominator = horizon_denominator(benign_rho)
    attacked_denominator = horizon_denominator(attacked_rho)
    benign_numerator = benign_horizon * benign_denominator
    attacked_numerator = attacked_horizon * attacked_denominator
    attacked_from_benign_constants = benign_numerator / attacked_denominator
    benign_from_attacked_constants = attacked_numerator / benign_denominator

    layer_norms = [
        {
            "parameter": row["parameter"],
            "A_operator_norm_2": row["abs_A_max"],
        }
        for row in foundation["layers"]
    ]
    return {
        "schema_version": 1,
        "model_id": foundation["model_id"],
        "model_revision": foundation["model_revision"],
        "A_log_sha256": foundation["A_log_sha256"],
        "A_operator_norm_definition": (
            "For A=-diag(exp(A_log)), ||A||_2=max_i exp(A_log_i)."
        ),
        "layer_count": len(layer_norms),
        "layer_A_operator_norm_2": layer_norms,
        "global_A_operator_norm_2_max": max(
            float(row["A_operator_norm_2"]) for row in layer_norms
        ),
        "legacy_horizon_audit": {
            "benign_rho": benign_rho,
            "benign_horizon": benign_horizon,
            "benign_exact_denominator": benign_denominator,
            "benign_implied_numerator": benign_numerator,
            "attacked_rho": attacked_rho,
            "attacked_horizon": attacked_horizon,
            "attacked_exact_denominator": attacked_denominator,
            "attacked_implied_numerator": attacked_numerator,
            "implied_numerator_ratio": benign_numerator / attacked_numerator,
            "attacked_horizon_if_constants_fixed_from_benign": (
                attacked_from_benign_constants
            ),
            "benign_horizon_if_constants_fixed_from_attacked": (
                benign_from_attacked_constants
            ),
            "claims_reconcile_with_fixed_constants": math.isclose(
                benign_numerator, attacked_numerator, rel_tol=1e-9
            ),
        },
        "near_critical_audit": [
            {
                "eta": eta,
                "rho": 1.0 - eta,
                "exact_denominator": -math.log1p(-eta),
                "first_order_denominator": eta,
                "relative_error": abs((-math.log1p(-eta) - eta) / -math.log1p(-eta)),
            }
            for eta in (0.01, 0.02, 0.68)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    foundation_path = Path(config["inputs"]["foundation_measurement"])
    foundation = json.loads(foundation_path.read_text(encoding="utf-8"))
    if config["model"]["id"] != foundation["model_id"]:
        raise ValueError("configured model ID does not match foundation evidence")
    if config["model"]["revision"] != foundation["model_revision"]:
        raise ValueError("configured revision does not match foundation evidence")
    report = audit(config, foundation)
    report.update(
        {
            "seed": int(config["seed"]),
            "config_path": args.config.as_posix(),
            "config_sha256": sha256_file(args.config),
            "foundation_measurement": foundation_path.as_posix(),
            "foundation_measurement_sha256": sha256_file(foundation_path),
            "package_versions": package_versions(),
        }
    )
    destination = Path(config["output"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(destination)


if __name__ == "__main__":
    main()
