"""Cell-oriented, resumable Phase 3 experiment orchestration.

Every expensive public function performs its own preflight, writes atomically, and
can be rerun after a Lightning session interruption.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.experiments.phase3_attacks import (
    load_official_trigger_pool,
    match_benign_control,
    official_attack_families,
    optimize_universal_trigger,
)
from src.experiments.phase3_common import (
    BudgetLedger,
    CheckpointStore,
    Phase3Paths,
    atomic_write_json,
    atomic_write_jsonl,
    create_research_zip,
    load_phase3_config,
    preflight,
    read_jsonl,
)
from src.experiments.phase3_dataset import (
    load_frozen_dataset,
    materialize_dataset,
    snapshot_official_artifacts,
)
from src.experiments.phase3_defense import fit_spectral_detector
from src.experiments.phase3_modeling import (
    evaluate_prompt,
    load_pinned_model,
    model_spec,
    precision_probe,
)
from src.utils.provenance import sha256_file

ESTIMATED_MINUTES = {
    "mamba_130m": 2,
    "mamba_790m": 55,
    "mamba_1_4b": 25,
    "mamba_2_8b": 75,
    "pythia_1_4b": 20,
    "pythia_2_8b": 30,
    "gpt2_xl": 35,
    "zamba2_2_7b": 35,
}


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False


def _config_and_paths() -> tuple[dict[str, Any], Phase3Paths]:
    config = load_phase3_config()
    return config, Phase3Paths.create(config)


def _checkpoint(config: dict[str, Any], paths: Phase3Paths) -> CheckpointStore:
    return CheckpointStore(paths, sha256_file(paths.root / "configs" / "phase3.yaml"))


def prepare_phase3(
    *,
    mode: str = "dry_run",
    confirmation: str = "",
    materialize_assets: bool = False,
) -> dict[str, Any]:
    """Validate first; optionally download only the frozen datasets/artifacts."""
    config, paths = _config_and_paths()
    report = preflight(config, mode=mode, confirmation=confirmation)
    outputs: list[Path] = []
    if materialize_assets:
        dataset_path, audit_path = materialize_dataset(config, paths)
        outputs.extend(
            (dataset_path, audit_path, snapshot_official_artifacts(config, paths))
        )
    matrix = {
        "schema_version": 2,
        "mode": mode,
        "tasks": config["dataset"]["task_splits"],
        "seeds": config["seeds"],
        "experiments": config["experiments"],
        "pareto_optimizations": len(config["attacks"]["pareto"]["tradeoff_points"])
        * int(config["attacks"]["pareto"]["restarts_per_cell"]),
        "note": "Five seeds fit detectors; five independent restarts optimize each attack recipe.",
    }
    matrix_path = atomic_write_json(paths.child("manifests", "run_matrix.json"), matrix)
    outputs.append(matrix_path)
    _checkpoint(config, paths).mark(f"prepare-{mode}", outputs)
    return {
        "preflight": report,
        "run_matrix": matrix,
        "outputs": [str(x) for x in outputs],
    }


def _stratified_calibration(
    rows: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    config = load_phase3_config()
    allowed_tasks = set(config["attacks"]["gradient_guided"]["calibration_tasks"])
    eligible = [
        row
        for row in rows
        if row["split"] == "calibration" and row["task"] in allowed_tasks
    ]
    tasks = sorted({row["task"] for row in eligible})
    selected = []
    while len(selected) < count:
        progressed = False
        for task in tasks:
            candidates = [
                row for row in eligible if row["task"] == task and row not in selected
            ]
            if candidates and len(selected) < count:
                selected.append(candidates[0])
                progressed = True
        if not progressed:
            break
    if len(selected) != count:
        raise ValueError(
            f"only {len(selected)} eligible attack-calibration rows; require {count}"
        )
    return selected


def run_official_hispa_selection_cell(*, confirmation: str) -> dict[str, Any]:
    """Select one official HiSPA trigger on calibration only; never inspect test."""
    config, paths = _config_and_paths()
    preflight(config, mode="full", confirmation=confirmation)
    output = paths.child("raw", "official_hispa_selection.json")
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    rows = _stratified_calibration(
        load_frozen_dataset(paths),
        int(config["attacks"]["gradient_guided"]["calibration_prompts"]),
    )
    spec = model_spec(config, "mamba_790m")
    scores = []
    with load_pinned_model(spec, dtype_name="bfloat16") as loaded:
        candidates = official_attack_families(
            paths,
            count=int(
                config["attacks"]["hispa_original"]["calibration_candidate_count"]
            ),
            tokenizer=loaded.tokenizer,
        )
        for candidate in candidates:
            values = [
                evaluate_prompt(
                    loaded,
                    row=row,
                    condition=f"hispa_candidate_{candidate['candidate_id']}",
                    trigger=candidate["text"],
                    max_new_tokens=None,
                    collect_spectrum=True,
                )
                for row in rows
            ]
            scores.append(
                {
                    **candidate,
                    "calibration_n": len(values),
                    "calibration_task_score": float(
                        np.mean([item["task_score"] for item in values])
                    ),
                    "calibration_rho": float(
                        np.mean(
                            [
                                item["spectral"]["rho_mean_over_layers"]
                                for item in values
                            ]
                        )
                    ),
                }
            )
    selected = min(
        scores,
        key=lambda item: (item["calibration_task_score"], item["calibration_rho"]),
    )
    payload = {
        "schema_version": 1,
        "selected_on": "calibration",
        "source_model": spec["key"],
        "selected": selected,
        "candidates": scores,
    }
    atomic_write_json(output, payload)
    _checkpoint(config, paths).mark("official-hispa-selection", [output])
    return payload


def _detector_payload(paths: Phase3Paths) -> dict[str, Any] | None:
    path = paths.child("manifests", "spectral_detector_seed42.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def run_attack_optimization_cell(recipe: str, *, confirmation: str) -> dict[str, Any]:
    """Run five true HotFlip restarts for one frozen recipe, or 25 for Pareto."""
    config, paths = _config_and_paths()
    preflight(config, mode="full", confirmation=confirmation)
    allowed = {"gradient_guided", "adaptive_v2", "adaptive_v3", "pareto"}
    if recipe not in allowed:
        raise KeyError(f"recipe must be one of {sorted(allowed)}")
    detector = _detector_payload(paths) if recipe in {"adaptive_v3", "pareto"} else None
    if recipe == "adaptive_v3" and detector is None:
        raise RuntimeError("fit_detector_cell() must finish before adaptive_v3")
    output = paths.child("raw", f"attack_{recipe}.json")
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    attack = config["attacks"]["gradient_guided"]
    rows = _stratified_calibration(
        load_frozen_dataset(paths), int(attack["calibration_prompts"])
    )
    if recipe == "pareto":
        points = config["attacks"]["pareto"]["tradeoff_points"]
        restarts = int(config["attacks"]["pareto"]["restarts_per_cell"])
    else:
        weights = {
            "gradient_guided": {
                "task_weight": 1.0,
                "rho_weight": 0.5,
                "hazard_weight": 0.0,
            },
            "adaptive_v2": {
                "task_weight": 1.0,
                "rho_weight": 1.0,
                "hazard_weight": 0.0,
            },
            "adaptive_v3": {
                "task_weight": 1.0,
                "rho_weight": 0.5,
                "hazard_weight": 1.0,
            },
        }[recipe]
        points = [{"name": recipe, **weights}]
        restarts = int(attack["restarts"])
    records = []
    spec = model_spec(config, "mamba_790m")
    ledger = BudgetLedger(paths, config["execution"]["max_gpu_hours_per_session"])
    estimate = 45 if recipe == "pareto" else 12
    with ledger.measured(f"attack-{recipe}", estimated_minutes=estimate, used_gpu=True):
        with load_pinned_model(spec, dtype_name="bfloat16") as loaded:
            for point in points:
                point_detector = detector if float(point["hazard_weight"]) > 0 else None
                if point_detector is None and float(point["hazard_weight"]) > 0:
                    raise RuntimeError(
                        "detector-aware Pareto point requires fit_detector_cell()"
                    )
                for restart in range(restarts):
                    seed = int(config["seeds"][restart % len(config["seeds"])])
                    record = optimize_universal_trigger(
                        loaded,
                        rows,
                        seed=seed,
                        trigger_tokens=int(attack["trigger_tokens"]),
                        steps=int(attack["optimization_steps"]),
                        candidates_per_step=int(attack["candidates_per_step"]),
                        task_weight=float(point["task_weight"]),
                        rho_weight=float(point["rho_weight"]),
                        hazard_weight=float(point["hazard_weight"]),
                        detector=point_detector,
                    )
                    record.update(
                        {
                            "recipe": recipe,
                            "tradeoff_point": point["name"],
                            "restart": restart,
                        }
                    )
                    records.append(record)
    best = min(records, key=lambda item: item["final_objective"])
    payload = {
        "schema_version": 1,
        "recipe": recipe,
        "source_model": spec["key"],
        "selected_on": "calibration",
        "restart_count": len(records),
        "best": best,
        "restarts": records,
    }
    atomic_write_json(output, payload)
    _checkpoint(config, paths).mark(f"attack-{recipe}", [output])
    return payload


def _experiment_spec(config: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    try:
        return next(
            item for item in config["experiments"] if item["id"] == experiment_id
        )
    except StopIteration as exc:
        raise KeyError(f"unknown experiment {experiment_id!r}") from exc


def _rows_for_experiment(
    rows: list[dict[str, Any]], experiment: dict[str, Any]
) -> list[dict[str, Any]]:
    test = [row for row in rows if row["split"] == "test"][: int(experiment["test_n"])]
    if experiment["id"] in {"E01_main_mamba", "E02_gpt2_control", "E04_adaptive"}:
        return [
            row for row in rows if row["split"] in {"calibration", "validation"}
        ] + test
    return test


def _load_attack_text(paths: Phase3Paths, recipe: str) -> str:
    path = paths.child("raw", f"attack_{recipe}.json")
    if not path.is_file():
        raise FileNotFoundError(f"run_attack_optimization_cell({recipe!r}) first")
    return str(json.loads(path.read_text(encoding="utf-8"))["best"]["trigger_text"])


def _conditions_for(experiment_id: str, paths: Phase3Paths) -> dict[str, str]:
    selection = paths.child("raw", "official_hispa_selection.json")
    if not selection.is_file():
        raise FileNotFoundError("run_official_hispa_selection_cell() first")
    official = json.loads(selection.read_text(encoding="utf-8"))["selected"]["text"]
    conditions = {"clean": "", "hispa_original": official}
    if experiment_id != "E02_gpt2_control":
        conditions["gradient_guided"] = _load_attack_text(paths, "gradient_guided")
    if experiment_id == "E04_adaptive":
        conditions["adaptive_v2"] = _load_attack_text(paths, "adaptive_v2")
        adaptive_v3 = paths.child("raw", "attack_adaptive_v3.json")
        if adaptive_v3.is_file():
            conditions["adaptive_v3"] = _load_attack_text(paths, "adaptive_v3")
    return conditions


def run_smoke_cell(*, confirmation: str) -> dict[str, Any]:
    """Exercise real tokenization, spectrum, BOE, generation, scoring and export paths."""
    config, paths = _config_and_paths()
    preflight(config, mode="full", confirmation=confirmation)
    rows = [row for row in load_frozen_dataset(paths) if row["split"] == "calibration"][
        :2
    ]
    trigger = official_attack_families(paths, count=1)[0]["text"]
    output = paths.child("raw", "smoke.jsonl")
    values = []
    with load_pinned_model(
        model_spec(config, "mamba_130m"), dtype_name="bfloat16"
    ) as loaded:
        for row in rows:
            for condition, text in {"clean": "", "hispa_original": trigger}.items():
                values.append(
                    evaluate_prompt(
                        loaded,
                        row=row,
                        condition=condition,
                        trigger=text,
                        max_new_tokens=None,
                        collect_spectrum=True,
                        collect_boe=True,
                    )
                )
    atomic_write_jsonl(output, values)
    _checkpoint(config, paths).mark("smoke", [output])
    return {"status": "complete", "path": str(output), "rows": len(values)}


def run_model_cell(experiment_id: str, *, confirmation: str) -> dict[str, Any]:
    """Run one preregistered experiment, atomically resuming prompt/condition pairs."""
    config, paths = _config_and_paths()
    preflight(config, mode="full", confirmation=confirmation)
    experiment = _experiment_spec(config, experiment_id)
    spec = model_spec(config, experiment["model"])
    rows = _rows_for_experiment(load_frozen_dataset(paths), experiment)
    conditions = _conditions_for(experiment_id, paths)
    output = paths.child("raw", f"{experiment_id}.jsonl")
    existing = read_jsonl(output) if output.is_file() else []
    completed = {(row["prompt_id"], row["condition"]) for row in existing}
    selected = list(existing)
    set_reproducible_seed(int(config["seeds"][0]))
    ledger = BudgetLedger(paths, config["execution"]["max_gpu_hours_per_session"])
    with ledger.measured(
        experiment_id, estimated_minutes=ESTIMATED_MINUTES[spec["key"]], used_gpu=True
    ):
        with load_pinned_model(spec, dtype_name="bfloat16") as loaded:
            control = match_benign_control(
                loaded,
                conditions["hispa_original"],
                candidates=load_official_trigger_pool(paths),
            )
            conditions["benign_control"] = control["control_text"]
            control_path = atomic_write_json(
                paths.child("manifests", f"control_match_{experiment_id}.json"), control
            )
            test_diagnostic_ids = {
                row["prompt_id"] for row in rows if row["split"] == "test"
            }
            test_diagnostic_ids = set(
                list(test_diagnostic_ids)[
                    : int(config["protocol"]["transformer_diagnostic_examples"])
                ]
            )
            for row in rows:
                for condition, trigger in conditions.items():
                    if (row["prompt_id"], condition) in completed:
                        continue
                    selected.append(
                        evaluate_prompt(
                            loaded,
                            row=row,
                            condition=condition,
                            trigger=trigger,
                            max_new_tokens=None,
                            collect_spectrum=spec["architecture"] == "mamba",
                            collect_boe=spec["architecture"] == "mamba",
                            collect_transformer_diagnostics=spec["architecture"]
                            == "transformer"
                            and row["prompt_id"] in test_diagnostic_ids,
                        )
                    )
                    completed.add((row["prompt_id"], condition))
                    if (
                        len(selected)
                        % int(config["execution"]["checkpoint_every_examples"])
                        == 0
                    ):
                        atomic_write_jsonl(output, selected)
            atomic_write_jsonl(output, selected)
    _checkpoint(config, paths).mark(experiment_id, [output, control_path])
    return {
        "status": "complete",
        "path": str(output),
        "rows": len(selected),
        "sha256": sha256_file(output),
        "budget_remaining_hours": ledger.remaining_hours(),
    }


def fit_detector_cell() -> dict[str, Any]:
    """Fit five detectors with frozen calibration/validation/test separation."""
    config, paths = _config_and_paths()
    source = paths.child("raw", "E04_adaptive.jsonl")
    if not source.is_file():
        raise FileNotFoundError("run E04_adaptive through adaptive_v2 before fitting")
    frame = pd.DataFrame(read_jsonl(source))
    frame = frame[
        frame["condition"].isin(
            {
                "clean",
                "benign_control",
                "hispa_original",
                "gradient_guided",
                "adaptive_v2",
            }
        )
    ].copy()
    frame["label"] = (~frame["condition"].isin({"clean", "benign_control"})).astype(int)
    outputs = []
    summaries = []
    for seed in config["seeds"]:
        payload, predictions = fit_spectral_detector(
            frame,
            seed=int(seed),
            threshold_points=int(config["protocol"]["detector_threshold_grid_points"]),
        )
        model_path = atomic_write_json(
            paths.child("manifests", f"spectral_detector_seed{seed}.json"), payload
        )
        prediction_path = paths.child(
            "summaries", f"spectral_predictions_seed{seed}.csv"
        )
        predictions.to_csv(prediction_path, index=False, lineterminator="\n")
        outputs.extend((model_path, prediction_path))
        summaries.append(payload)
    _checkpoint(config, paths).mark("fit-detectors", outputs)
    return {
        "status": "complete",
        "detectors": summaries,
        "outputs": [str(path) for path in outputs],
    }


def run_precision_gate(
    *, confirmation: str, model_key: str = "mamba_130m"
) -> dict[str, Any]:
    config, paths = _config_and_paths()
    preflight(config, mode="full", confirmation=confirmation)
    rows = [row for row in load_frozen_dataset(paths) if row["split"] == "calibration"][
        :2
    ]
    payload = precision_probe(model_spec(config, model_key), rows, max_new_tokens=16)
    output = atomic_write_json(
        paths.child("manifests", f"precision-{model_key}.json"), payload
    )
    if not payload["bf16_acceptable"]:
        raise RuntimeError(f"BF16 precision gate failed; inspect {output}")
    return payload


def export_experiment_bundle() -> Path:
    config, paths = _config_and_paths()
    required = [
        paths.child("raw", f"{item['id']}.jsonl") for item in config["experiments"]
    ]
    required += [
        paths.child("raw", f"attack_{name}.json")
        for name in ("gradient_guided", "adaptive_v2", "adaptive_v3", "pareto")
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"experiment evidence is incomplete: {missing}")
    files = [
        path
        for folder in ("raw", "datasets", "manifests", "checkpoints", "logs")
        for path in paths.child(folder).rglob("*")
        if path.is_file()
    ]
    files.extend(
        (
            paths.root / "configs" / "phase3.yaml",
            paths.root / "configs" / "phase3_acceptance.yaml",
            paths.root / "requirements-lightning.txt",
        )
    )
    return create_research_zip(
        paths.root / config["exports"]["experiments_zip"],
        base=paths.root,
        files=files,
        metadata={
            "bundle": "phase3_experiments",
            "config_sha256": sha256_file(paths.root / "configs" / "phase3.yaml"),
        },
    )
