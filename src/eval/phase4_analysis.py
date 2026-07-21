"""Leakage-safe Phase 4 statistics, tables, and CPU export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src.experiments.phase3_common import create_research_zip, read_jsonl
from src.experiments.phase4_common import Phase4Paths, load_phase4_config
from src.utils.provenance import sha256_file


ATTACKS = ("blackbox_transfer", "blackbox_query", "whitebox_joint")


def _bootstrap(values: np.ndarray, *, seed: int, resamples: int) -> tuple[float, float]:
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    estimates = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def _metric_bootstrap(
    labels: np.ndarray,
    values: np.ndarray,
    *,
    metric: str,
    seed: int,
    resamples: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(resamples):
        sample = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[sample]
        sampled_values = values[sample]
        if metric == "f1":
            estimates.append(
                f1_score(sampled_labels, sampled_values.astype(bool), zero_division=0)
            )
        elif len(np.unique(sampled_labels)) == 2:
            estimates.append(roc_auc_score(sampled_labels, sampled_values))
    if not estimates:
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def analyze_phase4() -> dict[str, Any]:
    config = load_phase4_config()
    paths = Phase4Paths.create(config)
    source = paths.child("raw", "phase4_confirmatory.jsonl")
    frame = pd.DataFrame(read_jsonl(source))
    expected = int(config["dataset"]["frozen_test_examples"])
    counts = frame.groupby("condition")["prompt_id"].nunique().to_dict()
    if counts != {condition: expected for condition in ("clean", *ATTACKS)}:
        raise RuntimeError(f"incomplete Phase 4 condition matrix: {counts}")
    if frame.duplicated(["prompt_id", "condition"]).any():
        raise RuntimeError("duplicate Phase 4 prompt/condition pair")

    clean = frame[frame["condition"] == "clean"][["prompt_id", "correct"]].rename(
        columns={"correct": "clean_correct"}
    )
    attack_rows = []
    task_attack_rows = []
    defense_rows = []
    resamples = int(config["analysis"]["bootstrap_resamples"])
    seed = int(config["seeds"][0])
    for condition in ATTACKS:
        attacked = frame[frame["condition"] == condition].merge(
            clean, on="prompt_id", validate="one_to_one"
        )
        eligible = attacked["clean_correct"].astype(bool).to_numpy()
        success = (eligible & ~attacked["correct"].astype(bool).to_numpy()).astype(float)
        conditional = success[eligible]
        low, high = _bootstrap(conditional, seed=seed, resamples=resamples)
        attack_rows.append(
            {
                "condition": condition,
                "n": len(attacked),
                "clean_correct_n": int(eligible.sum()),
                "coverage": float(eligible.mean()),
                "conditional_asr": float(conditional.mean()) if len(conditional) else float("nan"),
                "conditional_asr_ci_low": low,
                "conditional_asr_ci_high": high,
                "absolute_asr": float(success.mean()),
            }
        )
        for task, task_frame in attacked.groupby("task", sort=True):
            task_eligible = task_frame["clean_correct"].astype(bool).to_numpy()
            task_success = (
                task_eligible & ~task_frame["correct"].astype(bool).to_numpy()
            ).astype(float)
            task_conditional = task_success[task_eligible]
            task_low, task_high = _bootstrap(
                task_conditional, seed=seed, resamples=resamples
            )
            task_attack_rows.append(
                {
                    "condition": condition,
                    "task": task,
                    "n": len(task_frame),
                    "clean_correct_n": int(task_eligible.sum()),
                    "coverage": float(task_eligible.mean()),
                    "conditional_asr": (
                        float(task_conditional.mean())
                        if len(task_conditional)
                        else float("nan")
                    ),
                    "conditional_asr_ci_low": task_low,
                    "conditional_asr_ci_high": task_high,
                    "absolute_asr": float(task_success.mean()),
                }
            )
        pair = pd.concat(
            [
                frame[frame["condition"] == "clean"],
                frame[frame["condition"] == condition],
            ],
            ignore_index=True,
        )
        labels = (pair["condition"] != "clean").astype(int).to_numpy()
        probability = pair["spectralguard_probability"].astype(float).to_numpy()
        prediction = pair["spectralguard_prediction"].astype(bool).to_numpy()
        f1_low, f1_high = _metric_bootstrap(
            labels,
            prediction,
            metric="f1",
            seed=seed,
            resamples=resamples,
        )
        auc_low, auc_high = _metric_bootstrap(
            labels,
            probability,
            metric="auc",
            seed=seed,
            resamples=resamples,
        )
        defense_rows.append(
            {
                "condition": condition,
                "detector": "spectralguard_phase3_frozen_seed42",
                "n": len(pair),
                "f1": float(f1_score(labels, prediction, zero_division=0)),
                "f1_ci_low": f1_low,
                "f1_ci_high": f1_high,
                "auc": float(roc_auc_score(labels, probability)),
                "auc_ci_low": auc_low,
                "auc_ci_high": auc_high,
                "threshold_refit": False,
                "threshold_selected_on": "phase3_validation",
            }
        )

    attack_table = pd.DataFrame(attack_rows)
    task_attack_table = pd.DataFrame(task_attack_rows)
    defense_table = pd.DataFrame(defense_rows)
    attack_path = paths.child("summaries", "attack_comparison.csv")
    task_attack_path = paths.child("summaries", "attack_comparison_by_task.csv")
    defense_path = paths.child("summaries", "worst_case_defense.csv")
    attack_table.to_csv(attack_path, index=False, lineterminator="\n")
    task_attack_table.to_csv(task_attack_path, index=False, lineterminator="\n")
    defense_table.to_csv(defense_path, index=False, lineterminator="\n")

    query = json.loads(paths.child("attacks", "blackbox_query.json").read_text(encoding="utf-8"))
    query_table = pd.DataFrame(
        [
            {
                "method": "blackbox_transfer",
                "target_queries": 0,
                "budget": int(query["query_budget"]),
                "candidate_queries_after_median": 0.0,
                "candidate_queries_after_p95": 0.0,
            },
            {
                "method": "blackbox_query",
                "target_queries": int(query["queries_used"]),
                "budget": int(query["query_budget"]),
                "candidate_queries_after_median": float(
                    np.median(
                        [record["queries_after"] for record in query["candidate_records"]]
                    )
                ),
                "candidate_queries_after_p95": float(
                    np.quantile(
                        [record["queries_after"] for record in query["candidate_records"]],
                        0.95,
                    )
                ),
            },
        ]
    )
    query_path = paths.child("summaries", "query_budget.csv")
    query_table.to_csv(query_path, index=False, lineterminator="\n")

    worst = defense_table.loc[defense_table["f1"].idxmin()].to_dict()
    report = {
        "schema_version": 1,
        "status": "PASS",
        "primary_unit": "prompt_id",
        "selection_policy": "craft and select on frozen validation; test evaluated once",
        "test_prompts": expected,
        "conditions": ["clean", *ATTACKS],
        "worst_case_adaptive": worst,
        "blackbox_target_information": query["target_information_used"],
        "clasp_scope": (
            "BOE features are exported for future common-protocol scoring; Phase 3 did not "
            "publish a serializable frozen CLASP-style tree ensemble, so no invalid refit or "
            "cross-protocol Phase 4 CLASP number is reported."
        ),
        "limitations": [
            (
                f"Only {int(attack_table['clean_correct_n'].min())}/{expected} clean "
                "prompts were answered correctly by the target; conditional ASR therefore "
                "has low coverage and must be reported with its interval and absolute ASR."
            ),
            "No Phase 4 CLASP number is reported because no frozen serializable comparator was available.",
        ],
        "outputs": [
            path.relative_to(paths.root).as_posix()
            for path in (attack_path, task_attack_path, defense_path, query_path)
        ],
    }
    report_path = paths.child("summaries", "analysis_report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def export_phase4_final() -> Path:
    config = load_phase4_config()
    paths = Phase4Paths.create(config)
    report = paths.child("summaries", "analysis_report.json")
    if not report.is_file():
        raise RuntimeError("run analyze_phase4 before export")
    from src.eval.phase4_acceptance import validate_phase4_acceptance

    acceptance = validate_phase4_acceptance()
    if acceptance["status"] != "PASS":
        raise RuntimeError("Phase 4 semantic acceptance failed")
    files = [
        path
        for folder in ("attacks", "summaries", "manifests", "figures", "logs")
        for path in paths.child(folder).rglob("*")
        if path.is_file()
    ]
    files.extend((paths.root / "configs" / "phase4.yaml", paths.root / "requirements-lightning.txt"))
    return create_research_zip(
        paths.root / config["exports"]["final_zip"],
        base=paths.root,
        files=files,
        metadata={
            "bundle": "phase4_final",
            "config_sha256": sha256_file(paths.root / "configs" / "phase4.yaml"),
        },
    )
