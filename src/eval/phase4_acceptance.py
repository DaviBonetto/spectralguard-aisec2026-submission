"""Semantic acceptance gate for completed Phase 4 evidence."""

from __future__ import annotations

import json

import pandas as pd

from src.experiments.phase4_common import Phase4Paths, load_phase4_config
from src.experiments.phase3_common import atomic_write_json, read_jsonl


def validate_phase4_acceptance() -> dict:
    config = load_phase4_config()
    paths = Phase4Paths.create(config)
    automatic = json.loads(
        paths.child("manifests", "automatic_phase4_run.json").read_text(encoding="utf-8")
    )
    attack = pd.read_csv(paths.child("summaries", "attack_comparison.csv"))
    attack_by_task = pd.read_csv(
        paths.child("summaries", "attack_comparison_by_task.csv")
    )
    defense = pd.read_csv(paths.child("summaries", "worst_case_defense.csv"))
    query = pd.read_csv(paths.child("summaries", "query_budget.csv"))
    analysis = json.loads(
        paths.child("summaries", "analysis_report.json").read_text(encoding="utf-8")
    )
    raw = read_jsonl(paths.child("raw", "phase4_confirmatory.jsonl"))
    pairs = {(row["prompt_id"], row["condition"]) for row in raw}
    expected = int(config["dataset"]["frozen_test_examples"])
    confidence_columns = [
        "f1_ci_low",
        "f1_ci_high",
        "auc_ci_low",
        "auc_ci_high",
    ]
    checks = {
        "h100_run_complete": automatic["status"] == "COMPLETE",
        "same_protocol_three_attacks": set(attack["condition"])
        == {"blackbox_transfer", "blackbox_query", "whitebox_joint"},
        "minimum_n_300": bool((attack["n"] >= 300).all()),
        "conditional_asr_with_coverage": bool(
            attack[["conditional_asr", "coverage", "absolute_asr"]].notna().all().all()
        ),
        "query_budget_respected": int(query["target_queries"].max())
        <= int(query["budget"].max()),
        "transfer_is_zero_query": int(
            query.loc[query["method"] == "blackbox_transfer", "target_queries"].iloc[0]
        )
        == 0,
        "frozen_detector_no_refit": bool((~defense["threshold_refit"].astype(bool)).all()),
        "f1_auc_finite": bool(defense[["f1", "auc"]].notna().all().all()),
        "complete_unique_raw_matrix": len(raw) == len(pairs) == expected * 4,
        "per_task_attack_table": bool(
            len(attack_by_task) == 3 * 4
            and set(attack_by_task["condition"])
            == {"blackbox_transfer", "blackbox_query", "whitebox_joint"}
        ),
        "defense_confidence_intervals": bool(
            defense[confidence_columns].notna().all().all()
            and (defense["f1_ci_low"] <= defense["f1"]).all()
            and (defense["f1"] <= defense["f1_ci_high"]).all()
            and (defense["auc_ci_low"] <= defense["auc"]).all()
            and (defense["auc"] <= defense["auc_ci_high"]).all()
        ),
        "portable_report_paths": all(
            not str(path).startswith(("/", "\\")) and ":" not in str(path)
            for path in analysis["outputs"]
        ),
        "low_coverage_disclosed": any(
            "low coverage" in limitation for limitation in analysis["limitations"]
        ),
    }
    payload = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    atomic_write_json(paths.child("manifests", "phase4_acceptance.json"), payload)
    return payload
