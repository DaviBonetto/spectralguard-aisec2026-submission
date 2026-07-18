"""Cell-oriented, resumable Phase 3 experiment orchestration.

Every expensive public function performs its own preflight, writes atomically, and
can be rerun after a Lightning session interruption.
"""

from __future__ import annotations

import json
import hashlib
import math
import random
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import defaultdict
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
    apply_bf16_precision_policy,
    evaluate_prompt,
    evaluate_prompts_batch,
    load_pinned_model,
    model_spec,
    precision_probe,
)
from src.experiments.phase3_prompts import inject_trigger
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


class SessionTimeboxReached(RuntimeError):
    """Raised before the export reserve would be consumed by another GPU batch."""


def _require_session_time(deadline_monotonic: float | None, label: str) -> None:
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise SessionTimeboxReached(
            f"three-hour session timebox reached before {label}; progress is checkpointed"
        )


def _evaluate_batch_with_backoff(
    loaded: Any,
    *,
    rows: list[dict[str, Any]],
    condition: str,
    trigger: str,
    collect_spectrum: bool,
    collect_boe: bool,
    oom_backoff: bool,
) -> list[dict[str, Any]]:
    """Halve only an OOMing batch; all other failures remain fail-closed."""
    try:
        return evaluate_prompts_batch(
            loaded,
            rows=rows,
            condition=condition,
            trigger=trigger,
            max_new_tokens=None,
            collect_spectrum=collect_spectrum,
            collect_boe=collect_boe,
        )
    except torch.cuda.OutOfMemoryError:
        if not oom_backoff or len(rows) == 1:
            raise
        torch.cuda.empty_cache()
        midpoint = len(rows) // 2
        return _evaluate_batch_with_backoff(
            loaded,
            rows=rows[:midpoint],
            condition=condition,
            trigger=trigger,
            collect_spectrum=collect_spectrum,
            collect_boe=collect_boe,
            oom_backoff=oom_backoff,
        ) + _evaluate_batch_with_backoff(
            loaded,
            rows=rows[midpoint:],
            condition=condition,
            trigger=trigger,
            collect_spectrum=collect_spectrum,
            collect_boe=collect_boe,
            oom_backoff=oom_backoff,
        )


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


def run_pre_gpu_automated_audit() -> dict[str, Any]:
    """Replace the former human input gate with executable, recorded checks."""
    config, paths = _config_and_paths()
    preflight_report = preflight(config, mode="dry_run")
    commands = (
        [sys.executable, "-m", "ruff", "check", "src", "tests"],
        [sys.executable, "-m", "pytest", "-q"],
    )
    checks = []
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=paths.root,
            check=False,
            capture_output=True,
            text=True,
        )
        checks.append(
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            }
        )
        if completed.returncode:
            raise RuntimeError(
                f"automated pre-GPU audit failed: {' '.join(command)}\n"
                f"{completed.stdout}\n{completed.stderr}"
            )
    expected = ["01_phase3_experiments.ipynb", "02_phase3_analysis.ipynb"]
    tracked = subprocess.run(
        ["git", "ls-files", "notebooks/*.ipynb"],
        cwd=paths.root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if [Path(path).name for path in tracked] != expected:
        raise RuntimeError("automated audit requires exactly the two frozen notebooks")
    for relative in tracked:
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=paths.root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        payload = json.loads(committed)
        for cell in payload["cells"]:
            if cell["cell_type"] == "code" and (
                cell.get("execution_count") is not None or cell.get("outputs")
            ):
                raise RuntimeError(f"committed notebook is not clean: {relative}")
    report = {
        "schema_version": 1,
        "status": "PASS",
        "preflight": preflight_report,
        "checks": checks,
        "notebooks": expected,
    }
    output = atomic_write_json(
        paths.child("manifests", "automated_pre_gpu_audit.json"), report
    )
    _checkpoint(config, paths).mark("automated-pre-gpu-audit", [output])
    return report


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
    progress = paths.child("raw", "official_hispa_selection_progress.jsonl")
    scores = read_jsonl(progress) if progress.is_file() else []
    completed_candidates = {item["candidate_id"] for item in scores}
    candidate_count = int(
        config["attacks"]["hispa_original"]["calibration_candidate_count"]
    )
    estimated_minutes = max(
        2.0, 15.0 * (candidate_count - len(completed_candidates)) / candidate_count
    )
    ledger = BudgetLedger(paths, config["execution"]["max_gpu_hours_total"])
    with ledger.measured(
        "official-hispa-selection",
        estimated_minutes=estimated_minutes,
        used_gpu=True,
    ):
        with load_pinned_model(spec, dtype_name="bfloat16") as loaded:
            candidates = official_attack_families(
                paths,
                count=candidate_count,
                tokenizer=loaded.tokenizer,
            )
            for candidate in candidates:
                if candidate["candidate_id"] in completed_candidates:
                    continue
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
                atomic_write_jsonl(progress, scores)
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
    _checkpoint(config, paths).mark("official-hispa-selection", [output, progress])
    return payload


def _detector_payload(paths: Phase3Paths) -> dict[str, Any] | None:
    path = paths.child("manifests", "spectral_detector_seed42.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def run_attack_optimization_cell(
    recipe: str,
    *,
    confirmation: str,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
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
    progress = paths.child("raw", f"attack_{recipe}_progress.jsonl")
    records = read_jsonl(progress) if progress.is_file() else []
    completed_restarts = {
        (item["tradeoff_point"], int(item["restart"])) for item in records
    }
    spec = model_spec(config, "mamba_790m")
    ledger = BudgetLedger(paths, config["execution"]["max_gpu_hours_total"])
    full_estimate = 45 if recipe == "pareto" else 12
    total_restarts = len(points) * restarts
    remaining_restarts = total_restarts - len(completed_restarts)
    estimate = max(2.0, full_estimate * remaining_restarts / total_restarts)
    with ledger.measured(f"attack-{recipe}", estimated_minutes=estimate, used_gpu=True):
        with load_pinned_model(spec, dtype_name="bfloat16") as loaded:
            for point in points:
                point_detector = detector if float(point["hazard_weight"]) > 0 else None
                if point_detector is None and float(point["hazard_weight"]) > 0:
                    raise RuntimeError(
                        "detector-aware Pareto point requires fit_detector_cell()"
                    )
                for restart in range(restarts):
                    if (point["name"], restart) in completed_restarts:
                        continue
                    _require_session_time(
                        deadline_monotonic, f"attack-{recipe}/{point['name']}/{restart}"
                    )
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
                    atomic_write_jsonl(progress, records)
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
    _checkpoint(config, paths).mark(f"attack-{recipe}", [output, progress])
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
    test_rows = [row for row in rows if row["split"] == "test"]
    tasks = sorted({row["task"] for row in test_rows})
    requested = int(experiment["test_n"])
    test = []
    while len(test) < requested:
        progressed = False
        for task in tasks:
            candidates = [
                row for row in test_rows if row["task"] == task and row not in test
            ]
            if candidates and len(test) < requested:
                test.append(candidates[0])
                progressed = True
        if not progressed:
            break
    if len(test) != requested:
        raise ValueError(
            f"stratified test selection produced {len(test)} of {requested} rows"
        )
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


def _sort_rows_for_transformer_batch(
    loaded: Any, rows: list[dict[str, Any]], trigger: str
) -> list[dict[str, Any]]:
    """Return a stable input-length ordering to minimize masked padding."""
    return sorted(
        rows,
        key=lambda row: len(
            loaded.tokenizer(
                inject_trigger(row, trigger),
                truncation=True,
                max_length=int(row.get("max_input_tokens", 768)),
            )["input_ids"]
        ),
    )


def run_smoke_cell(*, confirmation: str) -> dict[str, Any]:
    """Exercise real tokenization, spectrum, BOE, generation, scoring and export paths."""
    config, paths = _config_and_paths()
    preflight(config, mode="full", confirmation=confirmation)
    rows = [row for row in load_frozen_dataset(paths) if row["split"] == "calibration"][
        :2
    ]
    trigger = official_attack_families(paths, count=1)[0]["text"]
    output = paths.child("raw", "smoke.jsonl")
    if output.is_file() and len(read_jsonl(output)) == 4:
        return {"status": "already_complete", "path": str(output), "rows": 4}
    values = []
    ledger = BudgetLedger(paths, config["execution"]["max_gpu_hours_total"])
    with ledger.measured("smoke", estimated_minutes=2, used_gpu=True):
        with load_pinned_model(
            model_spec(config, "mamba_130m"), dtype_name="bfloat16"
        ) as loaded:
            for row in rows:
                for condition, text in {
                    "clean": "",
                    "hispa_original": trigger,
                }.items():
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


def run_batch_equivalence_gate(*, confirmation: str) -> dict[str, Any]:
    """Prove that restored batch-one E04 evidence can be mixed with batching."""
    config, paths = _config_and_paths()
    preflight(config, mode="full", confirmation=confirmation)
    output = paths.child("manifests", "batch_equivalence_gate.json")
    if output.is_file():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("passed"):
            return payload
    source = paths.child("raw", "E04_adaptive.jsonl")
    if not source.is_file():
        payload = {
            "schema_version": 1,
            "passed": True,
            "status": "NOT_REQUIRED_NO_RESTORED_ROWS",
            "comparisons": [],
        }
        atomic_write_json(output, payload)
        return payload
    existing = read_jsonl(source)
    if existing and all(int(row.get("evaluation_batch_size", 1)) > 1 for row in existing):
        payload = {
            "schema_version": 1,
            "passed": True,
            "status": "NOT_REQUIRED_ALREADY_BATCHED",
            "comparisons": [],
        }
        atomic_write_json(output, payload)
        return payload

    batching = config["execution"]["batching"]
    desired_conditions = list(batching["bridge_conditions"])
    per_condition = int(batching["bridge_rows_per_condition"])
    baseline_by_pair = {
        (row["prompt_id"], row["condition"]): row for row in existing
    }
    dataset_by_id = {
        row["prompt_id"]: row for row in load_frozen_dataset(paths)
    }
    conditions = _conditions_for("E04_adaptive", paths)
    comparisons = []
    spec = model_spec(config, "mamba_790m")
    with load_pinned_model(spec, dtype_name="bfloat16") as loaded:
        for condition in desired_conditions:
            candidates = [
                dataset_by_id[prompt_id]
                for prompt_id, observed_condition in baseline_by_pair
                if observed_condition == condition and prompt_id in dataset_by_id
            ]
            candidates.sort(key=lambda row: (row["task"], row["prompt_id"]))
            grouped_candidates: dict[tuple[int, int], list[dict[str, Any]]] = (
                defaultdict(list)
            )
            for row in candidates:
                token_count = len(
                    loaded.tokenizer(
                        inject_trigger(row, conditions[condition]),
                        truncation=True,
                        max_length=int(row.get("max_input_tokens", 768)),
                    )["input_ids"]
                )
                grouped_candidates[(int(row["max_new_tokens"]), token_count)].append(
                    row
                )
            eligible = [
                rows for rows in grouped_candidates.values() if len(rows) >= per_condition
            ]
            if not eligible:
                raise RuntimeError(
                    f"batch bridge lacks {per_condition} equal-length {condition} rows"
                )
            selected = max(eligible, key=len)[:per_condition]
            grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for row in selected:
                grouped[int(row["max_new_tokens"])].append(row)
            recomputed = []
            for rows in grouped.values():
                recomputed.extend(
                    _evaluate_batch_with_backoff(
                        loaded,
                        rows=rows,
                        condition=condition,
                        trigger=conditions[condition],
                        collect_spectrum=True,
                        collect_boe=True,
                        oom_backoff=True,
                    )
                )
            for current in recomputed:
                baseline = baseline_by_pair[(current["prompt_id"], condition)]
                baseline_rho = float(baseline["spectral"]["rho_mean_over_layers"])
                current_rho = float(current["spectral"]["rho_mean_over_layers"])
                baseline_nll = float(baseline["prompt_nll"])
                comparisons.append(
                    {
                        "prompt_id": current["prompt_id"],
                        "task": current["task"],
                        "condition": condition,
                        "task_score_equal": math.isclose(
                            float(baseline["task_score"]),
                            float(current["task_score"]),
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        ),
                        "correct_equal": bool(baseline["correct"])
                        == bool(current["correct"]),
                        "rho_relative_error": abs(current_rho - baseline_rho)
                        / max(abs(baseline_rho), 1e-12),
                        "prompt_nll_relative_error": abs(
                            float(current["prompt_nll"]) - baseline_nll
                        )
                        / max(abs(baseline_nll), 1e-12),
                        "batch_size": int(current["evaluation_batch_size"]),
                    }
                )
    rho_tolerance = float(batching["spectral_relative_tolerance"])
    nll_tolerance = float(batching["prompt_nll_relative_tolerance"])
    passed = bool(comparisons) and all(
        row["task_score_equal"]
        and row["correct_equal"]
        and row["rho_relative_error"] <= rho_tolerance
        and row["prompt_nll_relative_error"] <= nll_tolerance
        for row in comparisons
    )
    payload = {
        "schema_version": 1,
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
        "reuse_decision": "RESTORED_ROWS_COMPATIBLE" if passed else "BLOCK_MIXED_BATCHES",
        "rho_relative_tolerance": rho_tolerance,
        "prompt_nll_relative_tolerance": nll_tolerance,
        "max_rho_relative_error": max(
            (row["rho_relative_error"] for row in comparisons), default=None
        ),
        "max_prompt_nll_relative_error": max(
            (row["prompt_nll_relative_error"] for row in comparisons), default=None
        ),
        "comparisons": comparisons,
    }
    atomic_write_json(output, payload)
    if not passed:
        raise RuntimeError(f"batch equivalence gate failed; inspect {output}")
    _checkpoint(config, paths).mark("batch-equivalence-gate", [output])
    return payload


def run_model_cell(
    experiment_id: str,
    *,
    confirmation: str,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
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
    chunk_directory = paths.child("raw", "chunks", experiment_id)
    chunk_directory.mkdir(parents=True, exist_ok=True)
    for chunk_path in sorted(chunk_directory.glob("*.jsonl")):
        for value in read_jsonl(chunk_path):
            pair = (value["prompt_id"], value["condition"])
            if pair not in completed:
                selected.append(value)
                completed.add(pair)
    set_reproducible_seed(int(config["seeds"][0]))
    desired_conditions = set(conditions) | {"benign_control"}
    expected_pairs = {
        (row["prompt_id"], condition)
        for row in rows
        for condition in desired_conditions
    }
    if expected_pairs.issubset(completed):
        selected.sort(key=lambda row: (row["prompt_id"], row["condition"]))
        atomic_write_jsonl(output, selected)
        for chunk_path in chunk_directory.glob("*.jsonl"):
            chunk_path.unlink()
        return {
            "status": "already_complete",
            "path": str(output),
            "rows": len(selected),
            "sha256": sha256_file(output),
        }
    missing_fraction = len(expected_pairs - completed) / len(expected_pairs)
    estimated_minutes = max(2.0, ESTIMATED_MINUTES[spec["key"]] * missing_fraction)
    ledger = BudgetLedger(paths, config["execution"]["max_gpu_hours_total"])
    with ledger.measured(
        experiment_id, estimated_minutes=estimated_minutes, used_gpu=True
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
                sorted(test_diagnostic_ids)[
                    : int(config["protocol"]["transformer_diagnostic_examples"])
                ]
            )
            work: dict[tuple[str, str, int, int, bool], list[dict[str, Any]]] = (
                defaultdict(list)
            )
            for row in rows:
                for condition in conditions:
                    if (row["prompt_id"], condition) in completed:
                        continue
                    diagnostic = bool(
                        spec["role"] == "historical_control"
                        and row["prompt_id"] in test_diagnostic_ids
                    )
                    key = (
                        condition,
                        row["task"],
                        int(row.get("max_input_tokens", 768)),
                        int(row["max_new_tokens"]),
                        diagnostic,
                    )
                    work[key].append(row)
            batching = config["execution"]["batching"]
            configured_batch_size = int(
                batching["initial_batch_sizes"][spec["key"]]
            )
            missing_total = sum(len(values) for values in work.values())
            processed_new = 0
            progress_interval = max(1, missing_total // 20)
            next_progress = progress_interval
            experiment_started = time.monotonic()
            print(
                f"{experiment_id}: {missing_total} missing pairs; "
                f"configured batch={configured_batch_size}",
                flush=True,
            )
            for key in sorted(work):
                condition, _task, _input_limit, _output_limit, diagnostic = key
                trigger = conditions[condition]
                group_rows = work[key]
                if spec["architecture"] in {"mamba", "hybrid"}:
                    exact_length_groups: dict[int, list[dict[str, Any]]] = (
                        defaultdict(list)
                    )
                    for row in group_rows:
                        token_count = len(
                            loaded.tokenizer(
                                inject_trigger(row, trigger),
                                truncation=True,
                                max_length=int(row.get("max_input_tokens", 768)),
                            )["input_ids"]
                        )
                        exact_length_groups[token_count].append(row)
                    grouped_batches = [
                        values
                        for _, values in sorted(exact_length_groups.items())
                    ]
                else:
                    # Transformer masks make length-sorted batching exact while
                    # avoiding the worst-case padding cost of dataset order.
                    group_rows = _sort_rows_for_transformer_batch(
                        loaded, group_rows, trigger
                    )
                    grouped_batches = [group_rows]
                batch_size = 1 if diagnostic else configured_batch_size
                for homogeneous_rows in grouped_batches:
                    for start in range(0, len(homogeneous_rows), batch_size):
                        _require_session_time(
                            deadline_monotonic, f"{experiment_id}/{condition}"
                        )
                        batch_rows = homogeneous_rows[start : start + batch_size]
                        if diagnostic:
                            values = [
                                evaluate_prompt(
                                    loaded,
                                    row=row,
                                    condition=condition,
                                    trigger=trigger,
                                    max_new_tokens=None,
                                    collect_spectrum=False,
                                    collect_boe=False,
                                    collect_transformer_diagnostics=True,
                                )
                                for row in batch_rows
                            ]
                        else:
                            values = _evaluate_batch_with_backoff(
                                loaded,
                                rows=batch_rows,
                                condition=condition,
                                trigger=trigger,
                                collect_spectrum=spec["architecture"] == "mamba",
                                collect_boe=spec["architecture"] == "mamba",
                                oom_backoff=bool(batching["oom_backoff"]),
                            )
                        selected.extend(values)
                        processed_new += len(values)
                        completed.update(
                            (value["prompt_id"], value["condition"])
                            for value in values
                        )
                        chunk_key = "|".join(
                            f"{value['prompt_id']}:{value['condition']}"
                            for value in values
                        )
                        chunk_name = hashlib.sha256(
                            chunk_key.encode("utf-8")
                        ).hexdigest()[:24]
                        atomic_write_jsonl(
                            chunk_directory / f"{chunk_name}.jsonl", values
                        )
                        if (
                            processed_new >= next_progress
                            or processed_new == missing_total
                        ):
                            elapsed = max(time.monotonic() - experiment_started, 1e-9)
                            rate = processed_new / elapsed
                            eta_minutes = (missing_total - processed_new) / max(rate, 1e-9) / 60
                            print(
                                f"{experiment_id}: {processed_new}/{missing_total} "
                                f"({100 * processed_new / missing_total:.1f}%), "
                                f"{rate:.2f} pairs/s, ETA {eta_minutes:.1f} min",
                                flush=True,
                            )
                            next_progress += progress_interval
            selected.sort(key=lambda row: (row["prompt_id"], row["condition"]))
            atomic_write_jsonl(output, selected)
            for chunk_path in chunk_directory.glob("*.jsonl"):
                chunk_path.unlink()
    batched_rows = [row for row in selected if "evaluation_batch_size" in row]
    telemetry = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "model_key": spec["key"],
        "configured_batch_size": int(
            config["execution"]["batching"]["initial_batch_sizes"][spec["key"]]
        ),
        "batched_rows": len(batched_rows),
        "batch_sizes_observed": sorted(
            {int(row["evaluation_batch_size"]) for row in batched_rows}
        ),
        "max_peak_vram_gib": max(
            (float(row["peak_vram_gib"]) for row in batched_rows), default=None
        ),
        "median_examples_per_second": float(
            np.median([row["examples_per_second"] for row in batched_rows])
        )
        if batched_rows
        else None,
        "completed_pairs": len(completed),
        "expected_pairs": len(expected_pairs),
    }
    telemetry_path = atomic_write_json(
        paths.child("manifests", f"batch_telemetry_{experiment_id}.json"), telemetry
    )
    _checkpoint(config, paths).mark(
        experiment_id, [output, control_path, telemetry_path]
    )
    return {
        "status": "complete",
        "path": str(output),
        "rows": len(selected),
        "sha256": sha256_file(output),
        "budget_remaining_hours": ledger.remaining_hours(),
    }


def materialize_derived_experiment_cell(experiment_id: str) -> dict[str, Any]:
    """Reuse an exact frozen subset from a larger completed experiment."""
    config, paths = _config_and_paths()
    experiment = _experiment_spec(config, experiment_id)
    source_id = experiment.get("derived_from")
    if not source_id:
        raise ValueError(f"{experiment_id} is not declared as a derived experiment")
    source = paths.child("raw", f"{source_id}.jsonl")
    if not source.is_file():
        raise FileNotFoundError(f"source experiment is incomplete: {source}")
    selected_rows = _rows_for_experiment(load_frozen_dataset(paths), experiment)
    selected_ids = {row["prompt_id"] for row in selected_rows}
    source_rows = read_jsonl(source)
    values = [
        {**row, "derived_experiment_id": experiment_id, "derived_from": source_id}
        for row in source_rows
        if row["split"] == "test" and row["prompt_id"] in selected_ids
    ]
    conditions = {row["condition"] for row in values}
    required_conditions = {
        "clean",
        "benign_control",
        "hispa_original",
        "gradient_guided",
    }
    if conditions != required_conditions:
        raise RuntimeError(
            f"derived source conditions drifted: {conditions} != {required_conditions}"
        )
    expected = len(selected_ids) * len(conditions)
    if len(values) != expected or len(selected_ids) != int(experiment["test_n"]):
        raise RuntimeError(
            f"derived experiment is incomplete: {len(values)} rows, expected {expected}"
        )
    output = atomic_write_jsonl(paths.child("raw", f"{experiment_id}.jsonl"), values)
    _checkpoint(config, paths).mark(experiment_id, [source, output])
    return {
        "status": "complete",
        "path": str(output),
        "rows": len(values),
        "derived_from": source_id,
        "gpu_seconds": 0,
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
    output = paths.child("manifests", f"precision-{model_key}.json")
    if output.is_file():
        existing = apply_bf16_precision_policy(
            json.loads(output.read_text(encoding="utf-8"))
        )
        if existing.get("bf16_acceptable"):
            atomic_write_json(output, existing)
            _checkpoint(config, paths).mark("precision-gate", [output])
            return existing
    ledger = BudgetLedger(paths, config["execution"]["max_gpu_hours_total"])
    with ledger.measured("precision-gate", estimated_minutes=3, used_gpu=True):
        payload = precision_probe(
            model_spec(config, model_key), rows, max_new_tokens=16
        )
    atomic_write_json(output, payload)
    if not payload["bf16_acceptable"]:
        raise RuntimeError(f"BF16 precision gate failed; inspect {output}")
    _checkpoint(config, paths).mark("precision-gate", [output])
    return payload


def run_automatic_phase3() -> dict[str, Any]:
    """Execute Notebook 1 end-to-end with fail-closed gates and safe resume."""
    config, paths = _config_and_paths()
    confirmation = str(config["execution"]["full_confirmation"])
    wall_seconds = float(config["execution"]["session_wall_clock_hours"]) * 3600.0
    reserve_seconds = float(config["execution"]["export_reserve_minutes"]) * 60.0
    session_started = time.monotonic()
    deadline_monotonic = session_started + wall_seconds - reserve_seconds
    report_path = paths.child("manifests", "automatic_phase3_run.json")
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "RUNNING",
        "stages": [],
        "gpu_budget_hours_total": float(config["execution"]["max_gpu_hours_total"]),
        "session_wall_clock_hours": float(
            config["execution"]["session_wall_clock_hours"]
        ),
        "export_reserve_minutes": float(
            config["execution"]["export_reserve_minutes"]
        ),
    }

    def persist() -> None:
        atomic_write_json(report_path, report)

    def stage(name: str, function: Any, *args: Any, **kwargs: Any) -> Any:
        if report["status"] == "TIMEBOX_HANDOFF":
            return None
        print(f"\n=== {name} ===", flush=True)
        try:
            result = function(*args, **kwargs)
        except Exception as exc:
            if isinstance(exc, SessionTimeboxReached):
                report["status"] = "TIMEBOX_HANDOFF"
                report["stopped_stage"] = name
                report["reason"] = str(exc)
                persist()
                handoff = export_phase3_handoff_bundle()
                report["handoff_archive"] = str(handoff)
                report["session_elapsed_seconds"] = (
                    time.monotonic() - session_started
                )
                persist()
                print(
                    f"\nTIMEBOX HANDOFF READY: download {handoff}", flush=True
                )
                return {"status": "TIMEBOX_HANDOFF", "path": str(handoff)}
            report["status"] = "FAILED"
            report["failed_stage"] = name
            report["error"] = f"{type(exc).__name__}: {exc}"
            persist()
            raise
        report["stages"].append({"name": name, "status": "PASS"})
        persist()
        return result

    persist()
    stage("automated_pre_gpu_audit", run_pre_gpu_automated_audit)
    stage(
        "frozen_assets_and_h100_preflight",
        prepare_phase3,
        mode="full",
        confirmation=confirmation,
        materialize_assets=True,
    )
    stage("smoke", run_smoke_cell, confirmation=confirmation)
    stage("precision_gate", run_precision_gate, confirmation=confirmation)
    stage(
        "official_hispa_selection",
        run_official_hispa_selection_cell,
        confirmation=confirmation,
    )
    for recipe in ("gradient_guided", "adaptive_v2"):
        stage(
            f"attack_{recipe}",
            run_attack_optimization_cell,
            recipe,
            confirmation=confirmation,
            deadline_monotonic=deadline_monotonic,
        )
    stage(
        "batch_equivalence_gate",
        run_batch_equivalence_gate,
        confirmation=confirmation,
    )
    stage(
        "E04_adaptive_pre_detector",
        run_model_cell,
        "E04_adaptive",
        confirmation=confirmation,
        deadline_monotonic=deadline_monotonic,
    )
    stage("fit_detector", fit_detector_cell)
    for recipe in ("adaptive_v3", "pareto"):
        stage(
            f"attack_{recipe}",
            run_attack_optimization_cell,
            recipe,
            confirmation=confirmation,
            deadline_monotonic=deadline_monotonic,
        )
    stage(
        "E04_adaptive_final",
        run_model_cell,
        "E04_adaptive",
        confirmation=confirmation,
        deadline_monotonic=deadline_monotonic,
    )
    for experiment in config["experiments"]:
        if experiment["id"] == "E04_adaptive":
            continue
        if experiment.get("derived_from"):
            stage(
                experiment["id"],
                materialize_derived_experiment_cell,
                experiment["id"],
            )
        else:
            stage(
                experiment["id"],
                run_model_cell,
                experiment["id"],
                confirmation=confirmation,
                deadline_monotonic=deadline_monotonic,
            )
    if report["status"] == "TIMEBOX_HANDOFF":
        return report
    report["status"] = "READY_FOR_EXPORT"
    persist()
    bundle = stage("export_experiment_bundle", export_experiment_bundle)
    report["status"] = "COMPLETE"
    report["session_elapsed_seconds"] = time.monotonic() - session_started
    report["experiments_zip"] = str(bundle)
    report["next_notebook"] = "notebooks/02_phase3_analysis.ipynb"
    persist()
    print(f"\nCOMPLETE: download {bundle}", flush=True)
    return report


def export_phase3_handoff_bundle() -> Path:
    """Create a restore-compatible tarball for an interrupted paid session."""
    _config, paths = _config_and_paths()

    def count_jsonl(path: Path) -> int:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    destination = paths.child("exports", "phase3_handoff.tar.gz")
    runtime_files = [
        path
        for path in paths.runtime.rglob("*")
        if path.is_file()
        and "handoff-account1-audit" not in path.parts
        and "exports" not in path.relative_to(paths.runtime).parts
        and not path.name.endswith(".tmp")
    ]
    file_sizes = {
        path.relative_to(paths.runtime).as_posix(): path.stat().st_size
        for path in runtime_files
    }
    jsonl_rows = {
        relative: count_jsonl(path)
        for path in runtime_files
        if (relative := path.relative_to(paths.runtime).as_posix()).endswith(".jsonl")
    }
    budget_path = paths.child("manifests", "budget.json")
    report_path = paths.child("manifests", "automatic_phase3_run.json")
    status = {
        "schema_version": 2,
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=paths.root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "files": file_sizes,
        "jsonl_rows": jsonl_rows,
        "checkpoints": sorted(
            path.name for path in paths.child("checkpoints").glob("*.json")
        ),
        "budget": json.loads(budget_path.read_text(encoding="utf-8")),
        "automatic_report": json.loads(report_path.read_text(encoding="utf-8")),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with tempfile.TemporaryDirectory(prefix="spectralguard-handoff-") as directory:
        status_path = Path(directory) / "phase3-status.json"
        status_path.write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with tarfile.open(temporary, "w:gz") as archive:
            archive.add(status_path, arcname="phase3-status.json", recursive=False)
            for path in runtime_files:
                archive.add(
                    path,
                    arcname=(Path("Experiments") / path.relative_to(paths.runtime)).as_posix(),
                    recursive=False,
                )
    temporary.replace(destination)
    return destination


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
