"""Semantic acceptance gate for completed Phase 5 availability evidence."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.experiments.phase3_common import atomic_write_json, read_jsonl
from src.experiments.phase5_common import Phase5Paths, load_phase5_config
from src.utils.provenance import sha256_file


def validate_phase5_acceptance() -> dict:
    config = load_phase5_config()
    paths = Phase5Paths.create(config)
    automatic = json.loads(
        paths.child("manifests", "automatic_phase5_run.json").read_text(encoding="utf-8")
    )
    preflight = json.loads(paths.child("manifests", "preflight.json").read_text(encoding="utf-8"))
    analysis = json.loads(
        paths.child("summaries", "analysis_report.json").read_text(encoding="utf-8")
    )
    traces = read_jsonl(paths.child("raw", "session_traces.jsonl"))
    prompts = read_jsonl(paths.child("raw", "adversarial_benign_prompts.jsonl"))
    candidates = json.loads(
        paths.child("attacks", "adversarial_benign_candidates.json").read_text(encoding="utf-8")
    )
    selection = json.loads(
        paths.child("attacks", "selected_adversarial_benign.json").read_text(
            encoding="utf-8"
        )
    )
    sessions = pd.read_csv(paths.child("summaries", "session_survival.csv"))
    benignity = pd.read_csv(paths.child("summaries", "adversarial_benign.csv"))
    overhead = pd.read_csv(paths.child("summaries", "runtime_overhead.csv"))
    tradeoff = pd.read_csv(paths.child("summaries", "availability_tradeoff.csv"))
    expected = (
        int(config["dataset"]["validation_sessions"])
        + int(config["dataset"]["test_sessions"])
    ) * 4
    trace_pairs = {(row["session_id"], row["condition"]) for row in traces}
    prompt_pairs = {(row["prompt_id"], row["condition"]) for row in prompts}
    detector_source = paths.root / config["monitor"]["detector_source"]
    frozen_detector = paths.child("manifests", "frozen_detector.json")
    checks = {
        "h100_run_complete": automatic["status"] == "COMPLETE",
        "frozen_detector_identical": sha256_file(detector_source)
        == sha256_file(frozen_detector)
        == preflight["detector_sha256"],
        "complete_unique_session_matrix": len(traces) == len(trace_pairs) == expected,
        "complete_unique_prompt_matrix": len(prompts)
        == len(prompt_pairs)
        == int(config["adversarial_benign"]["test_prompts"]) * 3,
        "five_distinct_candidate_seeds": {int(row["seed"]) for row in candidates}
        == set(config["seeds"]),
        "selection_is_validation_only": selection["selection_split"] == "validation"
        and analysis["selection_policy"]["selected_on"] == "validation",
        "no_detector_refit": analysis["detector_refit"] is False,
        "both_policies_reported": set(sessions["policy"])
        == {"algorithm2", "algorithm2b"},
        "four_conditions_reported": set(sessions["condition"])
        == {
            "clean",
            "matched_benign_control",
            "adversarial_benign",
            "malicious_whitebox",
        },
        "session_confidence_intervals": bool(
            sessions[["survival_ci_low", "survival_ci_high"]].notna().all().all()
            and (sessions["survival_ci_low"] <= sessions["survival_rate"]).all()
            and (sessions["survival_rate"] <= sessions["survival_ci_high"]).all()
        ),
        "heldout_benignity_reported": len(benignity) == 3
        and benignity[["mean_task_score", "task_score_drop_from_clean"]].notna().all().all(),
        "finite_overhead": bool(np.isfinite(overhead["overhead_ratio"]).all())
        and bool(
            (
                overhead["sessions"]
                >= min(
                    int(config["overhead"]["sessions"]),
                    int(config["dataset"]["validation_sessions"]),
                )
            ).all()
        ),
        "validation_tradeoff_complete": len(tradeoff)
        == 1
        + len(config["graduated_policy"]["k_grid"])
        * len(config["graduated_policy"]["w_grid"]),
        "portable_report_paths": all(
            not str(path).startswith(("/", "\\")) and ":" not in str(path)
            for path in analysis["outputs"]
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    atomic_write_json(paths.child("manifests", "phase5_acceptance.json"), payload)
    return payload
