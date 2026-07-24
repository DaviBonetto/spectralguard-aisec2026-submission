"""Fail closed unless the committed Phase 2 evidence is internally consistent."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _sha256(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _load(path: str) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main() -> None:
    foundation_path = Path("results/foundation/mamba_130m_kappa.json")
    theory_config = Path("configs/phase2_theory.yaml")
    directional_config = Path("configs/phase2_directional.yaml")
    foundation = _load(str(foundation_path))
    theory = _load("results/theory/phase2_theory_audit.json")
    directional = _load("results/theory/directional_counterexample.json")

    assert theory["model_revision"] == foundation["model_revision"]
    assert theory["A_log_sha256"] == foundation["A_log_sha256"]
    assert theory["config_sha256"] == _sha256(theory_config)
    assert theory["foundation_measurement_sha256"] == _sha256(foundation_path)

    assert theory["layer_count"] == 24
    assert math.isclose(
        theory["global_A_operator_norm_2_max"],
        foundation["global_abs_A_max"],
        rel_tol=1e-15,
    )
    horizon = theory["legacy_horizon_audit"]
    assert horizon["claims_reconcile_with_fixed_constants"] is False
    assert horizon["implied_numerator_ratio"] > 460

    assert directional["model_revision"] == foundation["model_revision"]
    assert directional["config_sha256"] == _sha256(directional_config)
    assert directional["architecture_evidence_sha256"] == _sha256(foundation_path)

    assert directional["eigenvector_rotation_applicable_to_diagonal_A"] is False
    construction = directional["construction"]
    assert construction["transition_A_held_exactly_fixed_for_directional_family"]
    assert construction["readout_C_unit_norm_before_and_after"]
    assert construction["max_absolute_directional_rho_change"] == 0.0
    assert construction["max_absolute_readout_norm_change"] == 0.0

    rows = {row["attack_family"]: row for row in directional["results"]}
    collapse = rows["spectral_collapse_control"]
    counterexample = rows["directional_readout_counterexample"]
    assert collapse["n"] >= 200 and counterexample["n"] >= 200
    assert collapse["attack_success_rate"] == 1.0
    assert collapse["monitor_detection_rate"] == 1.0
    assert counterexample["attack_success_rate"] == 1.0
    assert counterexample["monitor_detection_rate"] == 0.0
    assert counterexample["monitor_detection_wilson_95"][1] < 0.004

    paper_log = Path("paper/main.log")
    if paper_log.is_file():
        log = paper_log.read_text(encoding="utf-8", errors="replace")
        assert "Overfull" not in log
        assert "undefined references" not in log.lower()
        assert "undefined citations" not in log.lower()

    print("PASS: Phase 2 evidence, scope, statistics, and paper build are consistent.")


if __name__ == "__main__":
    main()
