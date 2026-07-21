"""Verify the deterministic Phase 1 checkpoint result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_REVISION = "1e76775f628fbf1350fbe4dbb3d971ba64af25a1"
EXPECTED_A_LOG_SHA256 = (
    "fc6ce3892302656ffe48f79f7906831ec0514ff70ce6faaa67b2646be9336e74"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("results/foundation/mamba_130m_kappa.json"),
    )
    args = parser.parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    checks = {
        "immutable model revision": payload["model_revision"] == EXPECTED_REVISION,
        "24 A_log tensors": payload["A_log_tensor_count"] == 24,
        "A_log content hash": payload["A_log_sha256"] == EXPECTED_A_LOG_SHA256,
        "eigenvector kappa": payload["eigenvector_condition_kappa_2"] == 1.0,
        "every layer kappa": all(
            row["eigenvector_condition_kappa_2"] == 1.0 for row in payload["layers"]
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit("Foundation result verification failed: " + ", ".join(failed))
    print("PASS: kappa_2(V)=1.0 for all 24 layers at the pinned revision.")


if __name__ == "__main__":
    main()
