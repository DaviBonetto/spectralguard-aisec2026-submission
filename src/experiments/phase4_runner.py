"""Resumable, time-boxed Phase 4 H100 orchestration."""

from __future__ import annotations

import gc
import json
import random
import subprocess
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from src.attacks.phase4_blackbox import PublicResponse, query_only_search
from src.attacks.phase4_whitebox import load_frozen_spectralguard
from src.experiments.phase3_attacks import (
    attack_objective_and_gradient,
    optimize_universal_trigger,
)
from src.experiments.phase3_common import (
    Phase3Paths,
    atomic_write_json,
    atomic_write_jsonl,
    create_research_zip,
    load_phase3_config,
    read_jsonl,
)
from src.experiments.phase3_dataset import load_frozen_dataset, materialize_dataset
from src.experiments.phase3_modeling import (
    evaluate_prompt,
    evaluate_prompts_batch,
    load_pinned_model,
)
from src.experiments.phase3_prompts import inject_trigger
from src.experiments.phase4_common import (
    Phase4Paths,
    load_phase4_config,
    mark_checkpoint,
    phase4_preflight,
)
from src.utils.provenance import sha256_file


class Phase4Timebox(RuntimeError):
    pass


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _phase3_assets() -> tuple[dict[str, Any], Phase3Paths, list[dict[str, Any]]]:
    config = load_phase3_config()
    paths = Phase3Paths.create(config)
    dataset_path = paths.child("datasets", "phase3_four_tasks.jsonl")
    if not dataset_path.is_file():
        materialize_dataset(config, paths)
    return config, paths, load_frozen_dataset(paths)


def _stratified(rows: Sequence[dict[str, Any]], split: str, count: int) -> list[dict[str, Any]]:
    pool = sorted(
        (row for row in rows if row["split"] == split), key=lambda row: row["prompt_id"]
    )
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        groups[row["task"]].append(row)
    selected = []
    tasks = sorted(groups)
    cursor = {task: 0 for task in tasks}
    while len(selected) < count:
        progressed = False
        for task in tasks:
            index = cursor[task]
            if index < len(groups[task]) and len(selected) < count:
                selected.append(groups[task][index])
                cursor[task] += 1
                progressed = True
        if not progressed:
            break
    if len(selected) != count:
        raise ValueError(f"split {split} exposes only {len(selected)} rows")
    return selected


def run_phase4_cpu_audit() -> dict[str, Any]:
    config = load_phase4_config()
    paths = Phase4Paths.create(config)
    commands = (
        [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"],
        [sys.executable, "-m", "pytest", "-q"],
    )
    checks = []
    for command in commands:
        result = subprocess.run(
            command,
            cwd=paths.root,
            capture_output=True,
            text=True,
            check=False,
        )
        checks.append({"command": command, "returncode": result.returncode})
        if result.returncode:
            raise RuntimeError(result.stdout + "\n" + result.stderr)
    tracked = subprocess.run(
        ["git", "ls-files", "notebooks/*.ipynb"],
        cwd=paths.root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    expected = [
        "notebooks/01_phase3_experiments.ipynb",
        "notebooks/02_phase3_analysis.ipynb",
        "notebooks/03_phase4_attacks.ipynb",
        "notebooks/04_phase4_analysis.ipynb",
    ]
    if [path.replace("\\", "/") for path in tracked] != expected:
        raise RuntimeError(f"tracked notebook registry drifted: {tracked}")
    report = {"schema_version": 1, "status": "PASS", "checks": checks}
    output = atomic_write_json(paths.child("manifests", "cpu_audit.json"), report)
    mark_checkpoint(paths, "cpu_audit", [output])
    return report


def _mutated_candidates(tokenizer: Any, records: Sequence[dict[str, Any]]) -> list[str]:
    candidates = []
    for record in records:
        ids = list(record["trigger_token_ids"])
        variants = (ids, ids[1:] + ids[:1], list(reversed(ids)))
        for variant in variants:
            text = tokenizer.decode(variant, skip_special_tokens=True).strip()
            if text:
                candidates.append(text)
    return list(dict.fromkeys(candidates))


class ProductionQueryOracle:
    """Expose only output-derived values and an auditable query counter."""

    __slots__ = ("__evaluate", "__queries")

    def __init__(self, evaluate: Any) -> None:
        object.__setattr__(self, "_ProductionQueryOracle__evaluate", evaluate)
        object.__setattr__(self, "_ProductionQueryOracle__queries", 0)

    @property
    def queries_used(self) -> int:
        return int(object.__getattribute__(self, "_ProductionQueryOracle__queries"))

    def query(self, rows: Sequence[dict], trigger: str) -> list[PublicResponse]:
        evaluate = object.__getattribute__(self, "_ProductionQueryOracle__evaluate")
        output = evaluate(list(rows), trigger)
        object.__setattr__(
            self, "_ProductionQueryOracle__queries", self.queries_used + len(rows)
        )
        return [
            PublicResponse(
                task_score=float(row["task_score"]),
                correct=bool(row["correct"]),
                response_excerpt=str(row["response_excerpt"]),
            )
            for row in output
        ]


def _evaluate_public(loaded: Any, rows: list[dict[str, Any]], trigger: str) -> list[dict]:
    return [
        evaluate_prompt(
            loaded,
            row=row,
            condition="query_candidate",
            trigger=trigger,
            max_new_tokens=None,
            collect_spectrum=False,
        )
        for row in rows
    ]


def _hazard(spectral: dict[str, Any], detector: dict[str, Any]) -> float:
    features = np.asarray([layer["rho_mean"] for layer in spectral["layers"]])
    mean = np.asarray(detector["feature_mean"])
    scale = np.maximum(np.asarray(detector["feature_scale"]), 1e-12)
    weights = np.asarray(detector["weights"])
    logit = float((((features - mean) / scale) * weights).sum() + detector["bias"])
    return float(1.0 / (1.0 + np.exp(-np.clip(logit, -60, 60))))


def _require_deadline(deadline: float, label: str) -> None:
    if time.monotonic() >= deadline:
        raise Phase4Timebox(f"Phase 4 export reserve reached before {label}")


def _whitebox_validation_objective(
    loaded: Any,
    rows: list[dict[str, Any]],
    trigger: str,
    detector: dict[str, Any],
) -> float:
    values = []
    for row in rows:
        result = evaluate_prompt(
            loaded,
            row=row,
            condition="whitebox_validation",
            trigger=trigger,
            max_new_tokens=None,
            collect_spectrum=True,
        )
        values.append(
            float(result["task_score"]) + _hazard(result["spectral"], detector)
        )
    return float(np.mean(values))


def _evaluate_condition(
    loaded: Any,
    rows: list[dict[str, Any]],
    condition: str,
    trigger: str,
    *,
    detector: dict[str, Any],
    deadline: float,
    batch_size: int,
    accumulated: list[dict[str, Any]],
    output: Path,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        prompt = inject_trigger(row, trigger)
        length = len(loaded.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        buckets[(row["task"], int(row["max_new_tokens"]), length)].append(row)
    results = []
    for key in sorted(buckets):
        group = buckets[key]
        for start in range(0, len(group), batch_size):
            if time.monotonic() >= deadline:
                raise Phase4Timebox("Phase 4 export reserve reached")
            chunk = group[start : start + batch_size]
            try:
                values = evaluate_prompts_batch(
                    loaded,
                    rows=chunk,
                    condition=condition,
                    trigger=trigger,
                    max_new_tokens=None,
                    collect_spectrum=True,
                    collect_boe=True,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                values = []
                for row in chunk:
                    values.append(
                        evaluate_prompt(
                            loaded,
                            row=row,
                            condition=condition,
                            trigger=trigger,
                            max_new_tokens=None,
                            collect_spectrum=True,
                            collect_boe=True,
                        )
                    )
            for value in values:
                value["spectralguard_probability"] = _hazard(value["spectral"], detector)
                value["spectralguard_prediction"] = bool(
                    value["spectralguard_probability"] >= detector["threshold"]
                )
            results.extend(values)
            accumulated.extend(values)
            atomic_write_jsonl(output, accumulated)
            print(f"{condition}: {len(results)}/{len(rows)}", flush=True)
    return results


def _write_handoff(paths: Phase4Paths, status: dict[str, Any]) -> Path:
    destination = paths.root / load_phase4_config()["exports"]["handoff_tar"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    status_path = atomic_write_json(paths.child("manifests", "handoff_status.json"), status)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        archive.add(status_path, arcname="phase4/handoff_status.json")
        for path in paths.runtime.rglob("*"):
            if path.is_file() and not path.name.endswith(".tmp"):
                archive.add(path, arcname=f"phase4/{path.relative_to(paths.runtime).as_posix()}")
    temporary.replace(destination)
    return destination


def run_automatic_phase4() -> dict[str, Any]:
    config = load_phase4_config()
    paths = Phase4Paths.create(config)
    started = time.monotonic()
    deadline = started + float(config["execution"]["session_wall_clock_hours"]) * 3600 - float(
        config["execution"]["export_reserve_minutes"]
    ) * 60
    report_path = paths.child("manifests", "automatic_phase4_run.json")
    report: dict[str, Any] = {"schema_version": 1, "status": "RUNNING", "stages": []}

    def persist() -> None:
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_write_json(report_path, report)

    try:
        run_phase4_cpu_audit()
        report["stages"].append("cpu_audit")
        phase4_preflight(require_h100=True)
        report["stages"].append("h100_preflight")
        phase3_config, _phase3_paths, dataset = _phase3_assets()
        validation = _stratified(
            dataset, "validation", int(config["dataset"]["validation_crafting_examples"])
        )
        whitebox_crafting = [
            row for row in validation if row["task"] == "associative_recall"
        ][:4]
        if len(whitebox_crafting) != 4:
            raise RuntimeError("Phase 4 requires four short white-box crafting prompts")
        test = _stratified(dataset, "test", int(config["dataset"]["frozen_test_examples"]))
        detector = load_frozen_spectralguard(
            paths.root / config["white_box"]["detector_source"],
            seed=int(config["white_box"]["detector_seed"]),
        )
        atomic_write_json(paths.child("manifests", "frozen_detector.json"), detector)

        surrogate_path = paths.child("attacks", "surrogate_triggers.json")
        surrogate_records = (
            json.loads(surrogate_path.read_text(encoding="utf-8"))
            if surrogate_path.is_file()
            else []
        )
        with load_pinned_model(config["models"]["surrogate"]) as surrogate:
            for restart, seed in enumerate(config["seeds"]):
                if any(int(item["seed"]) == int(seed) for item in surrogate_records):
                    continue
                _require_deadline(deadline, f"surrogate restart {restart}")
                _seed(int(seed))
                record = optimize_universal_trigger(
                    surrogate,
                    validation[:4],
                    seed=int(seed),
                    trigger_tokens=int(config["black_box"]["trigger_tokens"]),
                    steps=int(config["black_box"]["surrogate_steps"]),
                    rho_weight=0.5,
                    task_weight=1.0,
                    candidates_per_step=int(config["black_box"]["candidates_per_step"]),
                )
                record["restart"] = restart
                surrogate_records.append(record)
                atomic_write_json(surrogate_path, surrogate_records)
            query_candidates = _mutated_candidates(surrogate.tokenizer, surrogate_records)
        transfer_trigger = min(surrogate_records, key=lambda item: item["final_objective"])[
            "trigger_text"
        ]
        report["stages"].append("surrogate_crafting")
        persist()

        with load_pinned_model(config["models"]["target"]) as target:
            blackbox_path = paths.child("attacks", "blackbox_query.json")
            if blackbox_path.is_file():
                blackbox = json.loads(blackbox_path.read_text(encoding="utf-8"))
            else:
                _require_deadline(deadline, "black-box target queries")
                oracle = ProductionQueryOracle(
                    lambda rows, trigger: _evaluate_public(target, rows, trigger)
                )
                blackbox = query_only_search(
                    oracle,
                    validation,
                    query_candidates,
                    query_budget=int(config["black_box"]["target_query_budget"]),
                )
                atomic_write_json(blackbox_path, blackbox)
            report["stages"].append("blackbox_query_only")
            persist()

            proof_path = paths.child("manifests", "joint_gradient_gate.json")
            if not proof_path.is_file():
                trigger_ids = surrogate_records[0]["trigger_token_ids"]
                _, task_gradient, _ = attack_objective_and_gradient(
                    target,
                    validation[:1],
                    trigger_ids,
                    rho_weight=0.5,
                    task_weight=1.0,
                    hazard_weight=0.0,
                    detector=None,
                )
                _, joint_gradient, _ = attack_objective_and_gradient(
                    target,
                    validation[:1],
                    trigger_ids,
                    rho_weight=0.5,
                    task_weight=1.0,
                    hazard_weight=1.0,
                    detector=detector,
                )
                proof = {
                    "schema_version": 1,
                    "status": "PASS",
                    "task_gradient_norm": float(task_gradient.norm().cpu()),
                    "joint_gradient_norm": float(joint_gradient.norm().cpu()),
                    "detector_gradient_contribution_norm": float(
                        (joint_gradient - task_gradient).norm().cpu()
                    ),
                }
                if min(proof[key] for key in proof if key.endswith("_norm")) <= 0:
                    raise RuntimeError(f"actual joint-gradient gate failed: {proof}")
                atomic_write_json(proof_path, proof)
            whitebox_path = paths.child("attacks", "whitebox_joint.json")
            whitebox_records = (
                json.loads(whitebox_path.read_text(encoding="utf-8"))
                if whitebox_path.is_file()
                else []
            )
            for hazard_weight in config["white_box"]["lambda_grid"]:
                for seed in config["seeds"]:
                    if any(
                        int(item["seed"]) == int(seed)
                        and float(item["lambda"]) == float(hazard_weight)
                        for item in whitebox_records
                    ):
                        continue
                    _require_deadline(deadline, f"white-box lambda={hazard_weight} seed={seed}")
                    _seed(int(seed))
                    record = optimize_universal_trigger(
                        target,
                        whitebox_crafting,
                        seed=int(seed),
                        trigger_tokens=int(config["white_box"]["trigger_tokens"]),
                        steps=int(config["white_box"]["steps"]),
                        rho_weight=float(config["white_box"]["rho_weight"]),
                        task_weight=float(config["white_box"]["task_weight"]),
                        hazard_weight=float(hazard_weight),
                        detector=detector,
                        candidates_per_step=int(config["white_box"]["candidates_per_step"]),
                        progress_label=f"whitebox lambda={hazard_weight} seed={seed}",
                    )
                    record["lambda"] = float(hazard_weight)
                    record["validation_objective"] = _whitebox_validation_objective(
                        target,
                        validation[4:],
                        record["trigger_text"],
                        detector,
                    )
                    whitebox_records.append(record)
                    atomic_write_json(whitebox_path, whitebox_records)
            whitebox_trigger = min(
                whitebox_records, key=lambda item: item["validation_objective"]
            )["trigger_text"]
            report["stages"].append("whitebox_joint")
            persist()

            conditions = {
                "clean": "",
                "blackbox_transfer": transfer_trigger,
                "blackbox_query": blackbox["selected_trigger"],
                "whitebox_joint": whitebox_trigger,
            }
            output = paths.child("raw", "phase4_confirmatory.jsonl")
            existing = read_jsonl(output) if output.is_file() else []
            completed = {(row["prompt_id"], row["condition"]) for row in existing}
            values = list(existing)
            for condition, trigger in conditions.items():
                missing = [row for row in test if (row["prompt_id"], condition) not in completed]
                if missing:
                    _evaluate_condition(
                        target,
                        missing,
                        condition,
                        trigger,
                        detector=detector,
                        deadline=deadline,
                        batch_size=int(config["execution"]["initial_batch_size"]),
                        accumulated=values,
                        output=output,
                    )
            mark_checkpoint(paths, "confirmatory", [output])
            report["stages"].append("confirmatory_evaluation")

        report.update({"status": "COMPLETE"})
        persist()
        files = [path for path in paths.runtime.rglob("*") if path.is_file()]
        files.append(paths.root / "configs" / "phase4.yaml")
        bundle = create_research_zip(
            paths.root / config["exports"]["experiments_zip"],
            base=paths.root,
            files=files,
            metadata={
                "bundle": "phase4_experiments",
                "config_sha256": sha256_file(paths.root / "configs" / "phase4.yaml"),
            },
        )
        report["experiments_zip"] = str(bundle)
        persist()
        print(f"COMPLETE: download {bundle}", flush=True)
        return report
    except Phase4Timebox as exc:
        report.update({"status": "TIMEBOX_HANDOFF", "reason": str(exc)})
        persist()
        handoff = _write_handoff(paths, report)
        report["handoff_tar"] = str(handoff)
        persist()
        print(f"TIMEBOX HANDOFF READY: download {handoff}", flush=True)
        return report
    except Exception as exc:
        report.update(
            {
                "status": "FAILED_WITH_HANDOFF",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        persist()
        handoff = _write_handoff(paths, report)
        print(f"FAILURE HANDOFF READY: {handoff}", flush=True)
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
