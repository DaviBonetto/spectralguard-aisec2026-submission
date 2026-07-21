"""Fail-closed Phase 3 acceptance matrix for requirements 3.1--3.18."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.experiments.phase3_common import (
    Phase3Paths,
    atomic_write_json,
    load_phase3_config,
)


def _artifact_path(paths: Phase3Paths, name: str) -> Path:
    folder = "figures" if name.endswith(".pdf") else "summaries"
    return paths.child(folder, name)


def _validate_artifact(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty acceptance artifact: {path}")
    if path.suffix == ".pdf":
        if not path.read_bytes().startswith(b"%PDF"):
            raise ValueError(f"invalid PDF artifact: {path}")
        return {"rows": None, "columns": None}
    frame = pd.read_csv(path)
    if frame.empty or not len(frame.columns):
        raise ValueError(f"empty CSV acceptance artifact: {path}")
    if (
        frame.astype(str)
        .apply(
            lambda column: column.str.contains(
                "TODO|PLACEHOLDER|PENDING", case=False
            ).any()
        )
        .any()
    ):
        raise ValueError(f"placeholder text in acceptance artifact: {path}")
    return {"rows": len(frame), "columns": list(frame.columns)}


def validate_phase3_acceptance() -> dict[str, Any]:
    config = load_phase3_config()
    paths = Phase3Paths.create(config)
    specification = yaml.safe_load(
        (paths.root / "configs" / "phase3_acceptance.yaml").read_text(encoding="utf-8")
    )
    records = []
    for item, details in specification["items"].items():
        artifacts = []
        for name in details["required_artifacts"]:
            path = _artifact_path(paths, name)
            artifacts.append({"name": name, **_validate_artifact(path)})
        records.append({"item": str(item), "status": "PASS", "artifacts": artifacts})

    lexical = pd.read_csv(_artifact_path(paths, "lexical_control.csv"))
    gpt2_lexical = lexical[lexical["model_key"] == "gpt2_xl"]
    if gpt2_lexical.empty or not (
        gpt2_lexical["test_auc"] < gpt2_lexical["auc_ceiling"]
    ).all():
        raise ValueError("3.1 failed: GPT-2 lexical AUC is not below its ceiling")
    table10 = pd.read_csv(_artifact_path(paths, "table10_unified.csv"))
    if len(table10) < 2 or (table10["n"] < 1000).any():
        raise ValueError("3.2 failed: unified detector comparison lacks N >= 1000")

    threshold = pd.read_csv(_artifact_path(paths, "threshold_provenance.csv"))
    if set(threshold["seed"]) != set(config["seeds"]):
        raise ValueError("threshold provenance does not cover all five seeds")
    if set(threshold["threshold_selected_on"]) != {"validation"}:
        raise ValueError("test leakage: a threshold was not selected on validation")
    headline = pd.read_csv(_artifact_path(paths, "headline_multiseed.csv"))
    if set(headline["seed"]) != set(config["seeds"]):
        raise ValueError("headline table does not cover all five seeds")
    controls = pd.read_csv(_artifact_path(paths, "four_task_controls.csv"))
    if set(controls["task"]) != set(config["dataset"]["task_splits"]):
        raise ValueError(
            "four-task control table does not contain exactly the frozen tasks"
        )
    matching = pd.read_csv(_artifact_path(paths, "control_matching_audit.csv"))
    if not (matching["token_length"] > 0).all():
        raise ValueError("matched-control audit has invalid token lengths")
    pareto = pd.read_csv(_artifact_path(paths, "pareto_frontier.csv"))
    expected_points = {
        item["name"] for item in config["attacks"]["pareto"]["tradeoff_points"]
    }
    expected_restarts = int(config["attacks"]["pareto"]["restarts_per_cell"])
    if set(pareto["tradeoff_point"]) != expected_points:
        raise ValueError("Pareto trade-off points drifted")
    if not (
        pareto.groupby("tradeoff_point")["restart"].nunique() == expected_restarts
    ).all():
        raise ValueError("Pareto cells do not contain five independent restarts")
    paired = pd.read_csv(_artifact_path(paths, "mcnemar_tests.csv"))
    required_paired = {
        "n",
        "paired_difference_ci_low",
        "paired_difference_ci_high",
        "bootstrap_resamples",
    }
    if not required_paired.issubset(paired.columns):
        raise ValueError("3.5 failed: paired adaptive bootstrap evidence is absent")
    if (paired["n"] < 300).any() or (paired["bootstrap_resamples"] < 2000).any():
        raise ValueError("3.5 failed: adaptive N or bootstrap count is insufficient")
    table10_ci = pd.read_csv(_artifact_path(paths, "table10_bootstrap_ci.csv"))
    required_ci = ["paired_drop_ci_low", "paired_drop_ci_high"]
    if not set(required_ci).issubset(table10_ci.columns) or table10_ci[required_ci].isna().any().any():
        raise ValueError("3.6 failed: a Table 10 confidence interval is absent")
    regression = pd.read_csv(_artifact_path(paths, "rho_accuracy_regression.csv"))
    if "mcfadden_pseudo_r2" not in regression or regression["mcfadden_pseudo_r2"].isna().any():
        raise ValueError("3.7 failed: McFadden pseudo-R2 is absent")
    transfer = pd.read_csv(_artifact_path(paths, "cross_model_transfer_wilson.csv"))
    if (transfer["n"] < 200).any():
        raise ValueError("3.8 failed: a transfer cell has N < 200")
    clasp = pd.read_csv(_artifact_path(paths, "clasp_common_protocol.csv"))
    if not clasp["not_official_pretrained_weights"].astype(bool).all():
        raise ValueError("CLASP comparison scope is mislabeled")
    bh = pd.read_csv(_artifact_path(paths, "multiple_comparisons_bh.csv"))
    if not {"p_value", "p_value_bh", "rejected_at_fdr"}.issubset(bh.columns):
        raise ValueError("BH correction columns are absent")

    report = {
        "schema_version": 1,
        "status": "PASS",
        "passed_items": len(records),
        "required_items": len(specification["items"]),
        "items": records,
    }
    if report["passed_items"] != 18:
        raise ValueError("acceptance matrix must contain exactly 18 passed items")
    atomic_write_json(paths.child("manifests", "phase3_acceptance.json"), report)
    return report
