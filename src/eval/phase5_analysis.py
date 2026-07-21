"""Leakage-safe Phase 5 availability analysis and publication export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.defense.availability_policy import algorithm2, algorithm2b, outcome_record
from src.experiments.phase3_common import create_research_zip, read_jsonl
from src.experiments.phase5_common import Phase5Paths, load_phase5_config
from src.utils.provenance import sha256_file


SESSION_CONDITIONS = (
    "clean",
    "matched_benign_control",
    "adversarial_benign",
    "malicious_whitebox",
)
BENIGN_CONDITIONS = ("clean", "matched_benign_control", "adversarial_benign")


def _bootstrap(values: np.ndarray, *, seed: int, resamples: int) -> tuple[float, float]:
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    estimates = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def _apply_policy(row: pd.Series, policy: str, k: int | None, w: int | None) -> dict:
    scores = row["hazard_scores"]
    threshold = float(row["threshold"])
    if policy == "algorithm2":
        outcome = algorithm2(scores, threshold=threshold)
    else:
        outcome = algorithm2b(scores, threshold=threshold, k=int(k), w=int(w))
    return outcome_record(outcome)


def _policy_metrics(
    frame: pd.DataFrame,
    *,
    policy: str,
    k: int | None = None,
    w: int | None = None,
) -> dict[str, float]:
    outcomes = pd.DataFrame(
        [_apply_policy(row, policy, k, w) for _, row in frame.iterrows()]
    )
    return {
        "sessions": float(len(outcomes)),
        "survival_rate": float(outcomes["survived"].astype(float).mean()),
        "hard_block_rate": float((~outcomes["survived"].astype(bool)).mean()),
        "flag_window_rate": float(outcomes["flag_window_rate"].mean()),
        "blocked_window_rate": float(outcomes["blocked_window_rate"].mean()),
    }


def _select_policy(validation: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    clean = validation[validation["condition"] == "clean"]
    attack = validation[validation["condition"] == "malicious_whitebox"]
    original_clean = _policy_metrics(clean, policy="algorithm2")
    original_attack = _policy_metrics(attack, policy="algorithm2")
    rows = [
        {
            "policy": "algorithm2",
            "k": np.nan,
            "w": np.nan,
            "clean_survival_rate": original_clean["survival_rate"],
            "attack_hard_block_rate": original_attack["hard_block_rate"],
            "attack_capture_drop": 0.0,
            "capture_constraint_satisfied": True,
        }
    ]
    limit = float(config["graduated_policy"]["maximum_attack_capture_drop"])
    for w in config["graduated_policy"]["w_grid"]:
        for k in config["graduated_policy"]["k_grid"]:
            if int(k) > int(w):
                continue
            clean_metrics = _policy_metrics(clean, policy="algorithm2b", k=k, w=w)
            attack_metrics = _policy_metrics(attack, policy="algorithm2b", k=k, w=w)
            drop = original_attack["hard_block_rate"] - attack_metrics["hard_block_rate"]
            rows.append(
                {
                    "policy": "algorithm2b",
                    "k": int(k),
                    "w": int(w),
                    "clean_survival_rate": clean_metrics["survival_rate"],
                    "attack_hard_block_rate": attack_metrics["hard_block_rate"],
                    "attack_capture_drop": drop,
                    "capture_constraint_satisfied": bool(drop <= limit + 1e-12),
                }
            )
    table = pd.DataFrame(rows)
    candidates = table[
        (table["policy"] == "algorithm2b")
        & table["capture_constraint_satisfied"].astype(bool)
    ]
    constraint_satisfied = not candidates.empty
    if candidates.empty:
        candidates = table[table["policy"] == "algorithm2b"]
    selected = candidates.sort_values(
        ["clean_survival_rate", "attack_hard_block_rate", "k", "w"],
        ascending=[False, False, True, True],
    ).iloc[0]
    selection = {
        "policy": "algorithm2b",
        "k": int(selected["k"]),
        "w": int(selected["w"]),
        "selected_on": "validation",
        "capture_constraint_satisfied": bool(constraint_satisfied),
        "maximum_attack_capture_drop": limit,
    }
    table["selected"] = (
        (table["policy"] == "algorithm2b")
        & (table["k"] == selection["k"])
        & (table["w"] == selection["w"])
    )
    return table, selection


def analyze_phase5() -> dict[str, Any]:
    config = load_phase5_config()
    paths = Phase5Paths.create(config)
    traces = pd.DataFrame(read_jsonl(paths.child("raw", "session_traces.jsonl")))
    prompts = pd.DataFrame(read_jsonl(paths.child("raw", "adversarial_benign_prompts.jsonl")))
    expected_validation = int(config["dataset"]["validation_sessions"])
    expected_test = int(config["dataset"]["test_sessions"])
    counts = traces.groupby(["split", "condition"])["session_id"].nunique().to_dict()
    expected_counts = {
        **{
            ("validation", condition): expected_validation
            for condition in SESSION_CONDITIONS
        },
        **{("test", condition): expected_test for condition in SESSION_CONDITIONS},
    }
    if counts != expected_counts:
        raise RuntimeError(f"incomplete Phase 5 session matrix: {counts}")
    if traces.duplicated(["session_id", "condition"]).any():
        raise RuntimeError("duplicate Phase 5 session-condition pair")
    prompt_expected = int(config["adversarial_benign"]["test_prompts"])
    prompt_counts = prompts.groupby("condition")["prompt_id"].nunique().to_dict()
    if prompt_counts != {condition: prompt_expected for condition in BENIGN_CONDITIONS}:
        raise RuntimeError(f"incomplete Phase 5 benignity matrix: {prompt_counts}")

    validation = traces[traces["split"] == "validation"]
    test = traces[traces["split"] == "test"]
    tradeoff, selection = _select_policy(validation, config)
    tradeoff_path = paths.child("summaries", "availability_tradeoff.csv")
    tradeoff.to_csv(tradeoff_path, index=False, lineterminator="\n")

    resamples = int(config["analysis"]["bootstrap_resamples"])
    seed = int(config["seeds"][0])
    session_rows = []
    for condition in SESSION_CONDITIONS:
        subset = test[test["condition"] == condition]
        for policy, k, w in (
            ("algorithm2", None, None),
            ("algorithm2b", selection["k"], selection["w"]),
        ):
            outcomes = pd.DataFrame(
                [_apply_policy(row, policy, k, w) for _, row in subset.iterrows()]
            )
            survival = outcomes["survived"].astype(float).to_numpy()
            low, high = _bootstrap(survival, seed=seed, resamples=resamples)
            session_rows.append(
                {
                    "condition": condition,
                    "policy": policy,
                    "k": k,
                    "w": w,
                    "sessions": len(subset),
                    "survival_rate": float(survival.mean()),
                    "survival_ci_low": low,
                    "survival_ci_high": high,
                    "hard_block_rate": float(1.0 - survival.mean()),
                    "flag_window_rate": float(outcomes["flag_window_rate"].mean()),
                    "blocked_window_rate": float(outcomes["blocked_window_rate"].mean()),
                    "window_fpr": (
                        float(outcomes["blocked_window_rate"].mean())
                        if condition != "malicious_whitebox"
                        else np.nan
                    ),
                    "selected_on": "validation" if policy == "algorithm2b" else "frozen_original",
                }
            )
    session_table = pd.DataFrame(session_rows)
    session_path = paths.child("summaries", "session_survival.csv")
    session_table.to_csv(session_path, index=False, lineterminator="\n")

    prompt_rows = []
    for condition, subset in prompts.groupby("condition", sort=True):
        scores = subset["task_score"].astype(float).to_numpy()
        low, high = _bootstrap(scores, seed=seed, resamples=resamples)
        prompt_rows.append(
            {
                "condition": condition,
                "prompts": len(subset),
                "mean_task_score": float(scores.mean()),
                "mean_task_score_ci_low": low,
                "mean_task_score_ci_high": high,
                "accuracy": float(subset["correct"].astype(float).mean()),
            }
        )
    benignity = pd.DataFrame(prompt_rows)
    clean_score = float(
        benignity.loc[benignity["condition"] == "clean", "mean_task_score"].iloc[0]
    )
    benignity["task_score_drop_from_clean"] = clean_score - benignity["mean_task_score"]
    benignity_path = paths.child("summaries", "adversarial_benign.csv")
    benignity.to_csv(benignity_path, index=False, lineterminator="\n")

    timed = validation[
        validation["baseline_batch_seconds"].notna()
        & validation["monitored_batch_seconds"].notna()
    ].copy()
    timed["baseline_seconds_per_session"] = (
        timed["baseline_batch_seconds"] / timed["batch_size"]
    )
    timed["monitored_seconds_per_session"] = (
        timed["monitored_batch_seconds"] / timed["batch_size"]
    )
    baseline = float(timed["baseline_seconds_per_session"].mean())
    monitored = float(timed["monitored_seconds_per_session"].mean())
    overhead = pd.DataFrame(
        [
            {
                "sessions": len(timed),
                "baseline_seconds_per_session": baseline,
                "monitored_seconds_per_session": monitored,
                "overhead_ratio": monitored / baseline,
                "overhead_percent": 100.0 * (monitored / baseline - 1.0),
                "peak_vram_gib": float(timed["peak_vram_gib"].max()),
            }
        ]
    )
    overhead_path = paths.child("summaries", "runtime_overhead.csv")
    overhead.to_csv(overhead_path, index=False, lineterminator="\n")

    figure_path = paths.child("figures", "availability_frontier.pdf")
    plt.figure(figsize=(5.4, 3.5))
    candidates = tradeoff[tradeoff["policy"] == "algorithm2b"]
    plt.scatter(
        candidates["attack_hard_block_rate"],
        candidates["clean_survival_rate"],
        c=candidates["capture_constraint_satisfied"].astype(int),
        cmap="coolwarm",
        alpha=0.8,
    )
    chosen = candidates[candidates["selected"]].iloc[0]
    plt.scatter(
        [chosen["attack_hard_block_rate"]],
        [chosen["clean_survival_rate"]],
        marker="*",
        s=170,
        color="black",
        label=f"selected k={selection['k']}, w={selection['w']}",
    )
    plt.xlabel("Validation adversarial-benign hard-block rate")
    plt.ylabel("Validation clean session survival")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()

    report = {
        "schema_version": 1,
        "status": "PASS",
        "primary_unit": "session_id",
        "selection_policy": selection,
        "test_sessions_per_condition": expected_test,
        "test_prompts_per_condition": prompt_expected,
        "detector_refit": False,
        "threshold_selected_on": config["monitor"]["detector_frozen_on"],
        "claims_policy": (
            "Report the availability gain only if the validation-selected policy satisfies "
            "the frozen attack-capture constraint; otherwise report a negative result."
        ),
        "limitations": [
            "Sessions are deterministic concatenations of held-out benchmark prompts, not live user logs.",
            "The adversarial-benign optimizer is white-box and bounded by the frozen compute budget.",
        ],
        "outputs": [
            path.relative_to(paths.root).as_posix()
            for path in (
                tradeoff_path,
                session_path,
                benignity_path,
                overhead_path,
                figure_path,
            )
        ],
    }
    report_path = paths.child("summaries", "analysis_report.json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def export_phase5_final() -> Path:
    config = load_phase5_config()
    paths = Phase5Paths.create(config)
    from src.eval.phase5_acceptance import validate_phase5_acceptance

    acceptance = validate_phase5_acceptance()
    if acceptance["status"] != "PASS":
        raise RuntimeError(f"Phase 5 semantic acceptance failed: {acceptance['checks']}")
    files = [
        path
        for folder in ("attacks", "summaries", "manifests", "figures", "logs")
        for path in paths.child(folder).rglob("*")
        if path.is_file()
    ]
    files.extend(
        (paths.root / "configs" / "phase5.yaml", paths.root / "requirements-lightning.txt")
    )
    return create_research_zip(
        paths.root / config["exports"]["final_zip"],
        base=paths.root,
        files=files,
        metadata={
            "bundle": "phase5_final",
            "config_sha256": sha256_file(paths.root / "configs" / "phase5.yaml"),
        },
    )
