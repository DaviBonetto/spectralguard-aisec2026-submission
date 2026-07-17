"""Migrate an existing Phase 3 BF16 probe to the spectral-stability policy."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def main() -> None:
    from src.experiments.phase3_common import atomic_write_json
    from src.experiments.phase3_modeling import apply_bf16_precision_policy

    path = Path("Experiments/manifests/precision-mamba_130m.json")
    if not path.is_file():
        raise FileNotFoundError(f"missing completed precision probe: {path}")
    payload = apply_bf16_precision_policy(json.loads(path.read_text(encoding="utf-8")))
    if not payload["bf16_acceptable"]:
        raise RuntimeError(
            "the existing spectral precision measurement genuinely exceeds the threshold"
        )
    atomic_write_json(path, payload)
    print(
        {
            "precision_policy": payload["precision_policy"],
            "max_relative_rho_error": payload["max_relative_rho_error"],
            "threshold": payload["spectral_relative_error_threshold"],
            "bf16_acceptable": payload["bf16_acceptable"],
        }
    )
    print("PRECISION GATE MIGRATION: FINISHED")


if __name__ == "__main__":
    main()
