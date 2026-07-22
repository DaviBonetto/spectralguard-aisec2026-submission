"""Verify the compact, tracked Phase 5 publication evidence without a GPU."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "phase5"


def _sha256(path: Path) -> str:
    payload = path.read_bytes()
    if path.suffix.lower() in {".csv", ".json"}:
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def verify() -> dict[str, bool]:
    provenance = json.loads((EVIDENCE / "provenance.json").read_text(encoding="utf-8"))
    acceptance = json.loads(
        (EVIDENCE / "manifests" / "phase5_acceptance.json").read_text(
            encoding="utf-8"
        )
    )
    analysis = json.loads(
        (EVIDENCE / "summaries" / "analysis_report.json").read_text(
            encoding="utf-8"
        )
    )
    outcome = json.loads((EVIDENCE / "outcome.json").read_text(encoding="utf-8"))
    sessions = pd.read_csv(EVIDENCE / "summaries" / "session_survival.csv")
    tradeoff = pd.read_csv(EVIDENCE / "summaries" / "availability_tradeoff.csv")
    benignity = pd.read_csv(EVIDENCE / "summaries" / "adversarial_benign.csv")
    overhead = pd.read_csv(EVIDENCE / "summaries" / "runtime_overhead.csv")

    published_hashes = provenance["published_artifacts"]
    hash_ok = all(_sha256(ROOT / path) == digest for path, digest in published_hashes.items())
    selected = sessions[
        (sessions["condition"] == "clean") & (sessions["policy"] == "algorithm2b")
    ].iloc[0]
    original = sessions[
        (sessions["condition"] == "clean") & (sessions["policy"] == "algorithm2")
    ].iloc[0]
    malicious = sessions[
        (sessions["condition"] == "malicious_whitebox")
        & (sessions["policy"] == "algorithm2b")
    ].iloc[0]

    checks = {
        "published_hashes": hash_ok,
        "execution_acceptance": acceptance["status"] == "PASS"
        and all(bool(value) for value in acceptance["checks"].values()),
        "analysis_complete": analysis["status"] == "PASS",
        "session_matrix": len(sessions) == 8
        and set(sessions["condition"])
        == {"clean", "matched_benign_control", "adversarial_benign", "malicious_whitebox"}
        and set(sessions["policy"]) == {"algorithm2", "algorithm2b"}
        and bool((sessions["sessions"] == 320).all()),
        "validation_selection_unique": int(tradeoff["selected"].astype(bool).sum()) == 1,
        "heldout_benignity": len(benignity) == 3
        and bool((benignity["prompts"] == 300).all()),
        "finite_overhead": bool(np.isfinite(overhead["overhead_ratio"]).all())
        and int(overhead["sessions"].iloc[0]) == 60,
        "frozen_detector": _sha256(EVIDENCE / "manifests" / "frozen_detector.json")
        == _sha256(ROOT / "results" / "phase4" / "manifests" / "frozen_detector.json"),
        "negative_outcome_disclosed": outcome["scientific_outcome"] == "NEGATIVE"
        and outcome["availability_objective_achieved"] is False
        and abs(float(selected["survival_rate"]) - 0.025) < 1e-12
        and abs(float(original["survival_rate"])) < 1e-12
        and abs(float(malicious["hard_block_rate"]) - 1.0) < 1e-12,
        "portable_paths": all(
            not str(path).startswith(("/", "\\")) and ":" not in str(path)
            for path in analysis["outputs"]
        ),
    }
    return {key: bool(value) for key, value in checks.items()}


if __name__ == "__main__":
    result = verify()
    if not all(result.values()):
        raise SystemExit(f"PHASE 5 RESULTS: FAIL {result}")
    print("PHASE 5 RESULTS: PASS (complete negative availability result)")
