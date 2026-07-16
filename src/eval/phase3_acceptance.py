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
