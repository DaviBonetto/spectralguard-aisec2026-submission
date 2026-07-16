"""Construct a reproducible boundary case for radius-only SSM monitors.

This experiment is deliberately model-level. It proves that a readout
direction can erase retained information while leaving a diagonal transition
and its spectral radius unchanged. It does not assert that every constructed
readout is reachable from a discrete token in a trained Mamba checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.utils.provenance import package_versions, sha256_file


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("require 0 <= successes <= total and total > 0")
    p_hat = successes / total
    denominator = 1.0 + z**2 / total
    center = (p_hat + z**2 / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            p_hat * (1.0 - p_hat) / total + z**2 / (4.0 * total**2)
        )
        / denominator
    )
    lower = 0.0 if successes == 0 else max(0.0, center - radius)
    upper = 1.0 if successes == total else min(1.0, center + radius)
    return lower, upper


def _validate(config: dict[str, Any]) -> None:
    if int(config["n_systems"]) < 200:
        raise ValueError("n_systems must be at least 200")
    if int(config["dimension"]) < 2:
        raise ValueError("dimension must be at least 2")
    if int(config["horizon"]) < 1:
        raise ValueError("horizon must be positive")
    if not 0 < float(config["rho_collapse_max"]) < float(
        config["rho_monitor_threshold"]
    ):
        raise ValueError("collapse radii must remain below the monitor threshold")
    if not float(config["rho_monitor_threshold"]) < float(
        config["rho_benign_min"]
    ) < 1:
        raise ValueError("benign radii must remain above the monitor threshold")
    evidence = Path(config["architecture_evidence"])
    if not evidence.is_file():
        raise FileNotFoundError(evidence)


def _summary_row(
    family: str, successes: int, detections: int, total: int
) -> dict[str, Any]:
    return {
        "attack_family": family,
        "n": total,
        "attack_successes": successes,
        "attack_success_rate": successes / total,
        "attack_success_wilson_95": list(wilson_interval(successes, total)),
        "monitor_detections": detections,
        "monitor_detection_rate": detections / total,
        "monitor_detection_wilson_95": list(wilson_interval(detections, total)),
    }


def run(config_path: str | Path) -> dict[str, Any]:
    """Execute the randomized witness suite and write deterministic evidence."""
    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    _validate(config)

    seed = int(config["seed"])
    total = int(config["n_systems"])
    dimension = int(config["dimension"])
    horizon = int(config["horizon"])
    monitor_threshold = float(config["rho_monitor_threshold"])
    drop_threshold = float(config["task_output_drop_threshold"])
    rng = np.random.default_rng(seed)

    directional_successes = 0
    directional_detections = 0
    collapse_successes = 0
    collapse_detections = 0
    max_readout_norm_change = 0.0

    for _ in range(total):
        dominant = int(rng.integers(0, dimension))
        orthogonal = (dominant + int(rng.integers(1, dimension))) % dimension
        rho_benign = float(
            rng.uniform(config["rho_benign_min"], config["rho_benign_max"])
        )
        rho_collapse = float(
            rng.uniform(config["rho_collapse_min"], config["rho_collapse_max"])
        )

        h0 = np.zeros(dimension)
        h0[dominant] = 1.0
        c_benign = np.zeros(dimension)
        c_benign[dominant] = 1.0
        c_directional = np.zeros(dimension)
        c_directional[orthogonal] = 1.0

        benign_state = h0 * rho_benign**horizon
        benign_output = float(c_benign @ benign_state)
        directional_output = float(c_directional @ benign_state)
        collapse_output = rho_collapse**horizon
        directional_drop = 1.0 - abs(directional_output) / abs(benign_output)
        collapse_drop = 1.0 - abs(collapse_output) / abs(benign_output)

        directional_successes += int(directional_drop >= drop_threshold)
        collapse_successes += int(collapse_drop >= drop_threshold)
        directional_detections += int(rho_benign <= monitor_threshold)
        collapse_detections += int(rho_collapse <= monitor_threshold)
        max_readout_norm_change = max(
            max_readout_norm_change,
            abs(np.linalg.norm(c_benign) - np.linalg.norm(c_directional)),
        )

    rows = [
        _summary_row(
            "spectral_collapse_control",
            collapse_successes,
            collapse_detections,
            total,
        ),
        _summary_row(
            "directional_readout_counterexample",
            directional_successes,
            directional_detections,
            total,
        ),
    ]
    evidence_path = Path(config["architecture_evidence"])
    payload = {
        "schema_version": 1,
        "experiment": "phase2_directional_counterexample",
        "scope": (
            "seeded model-level structural counterexample; "
            "not a token-reachability claim"
        ),
        "seed": seed,
        "config_path": config_file.as_posix(),
        "config_sha256": sha256_file(config_file),
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "architecture_evidence": evidence_path.as_posix(),
        "architecture_evidence_sha256": sha256_file(evidence_path),
        "eigenvector_rotation_applicable_to_diagonal_A": False,
        "construction": {
            "dimension": dimension,
            "horizon": horizon,
            "rho_monitor_threshold": monitor_threshold,
            "task_output_drop_threshold": drop_threshold,
            "transition_A_held_exactly_fixed_for_directional_family": True,
            "readout_C_unit_norm_before_and_after": True,
            "max_absolute_directional_rho_change": 0.0,
            "max_absolute_readout_norm_change": max_readout_norm_change,
        },
        "results": rows,
        "limitations": [
            "The witness operates on model parameters, not discrete token strings.",
            "Input-conditioned reachability of the constructed C direction is not assumed or proved.",
            "The result falsifies unconditional radius-only completeness but does not estimate natural-world attack prevalence.",
        ],
        "package_versions": package_versions(),
    }

    output_json = Path(config["output_json"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    output_csv = Path(config["output_csv"])
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "attack_family",
            "n",
            "attack_successes",
            "attack_success_rate",
            "monitor_detections",
            "monitor_detection_rate",
            "attack_success_wilson_95",
            "monitor_detection_wilson_95",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["attack_success_wilson_95"] = json.dumps(
                row["attack_success_wilson_95"]
            )
            csv_row["monitor_detection_wilson_95"] = json.dumps(
                row["monitor_detection_wilson_95"]
            )
            writer.writerow(csv_row)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase2_directional.yaml")
    args = parser.parse_args()
    result = run(args.config)
    for row in result["results"]:
        print(
            f"{row['attack_family']}: "
            f"attack_success={row['attack_success_rate']:.3f}, "
            f"monitor_detection={row['monitor_detection_rate']:.3f}, "
            f"N={row['n']}"
        )


if __name__ == "__main__":
    main()
