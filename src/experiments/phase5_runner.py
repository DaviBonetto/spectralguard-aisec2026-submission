"""Resumable H100 orchestration for Phase 5 availability experiments."""

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

from src.attacks.phase5_adversarial_benign import (
    optimize_adversarial_benign_trigger,
    select_benign_candidate,
)
from src.defense.availability_policy import algorithm2
from src.experiments.phase3_attacks import load_official_trigger_pool, match_benign_control
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
    MambaSpectralCollector,
    evaluate_prompts_batch,
    load_pinned_model,
)
from src.experiments.phase3_prompts import inject_trigger
from src.experiments.phase5_common import (
    Phase5Paths,
    load_phase5_config,
    mark_checkpoint,
    phase5_preflight,
)
from src.utils.provenance import sha256_file


class Phase5Timebox(RuntimeError):
    """Raised only after reserving enough time for an atomic handoff."""


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _phase3_assets() -> tuple[dict[str, Any], Phase3Paths, list[dict[str, Any]]]:
    config = load_phase3_config()
    paths = Phase3Paths.create(config)
    if not paths.child("datasets", "phase3_four_tasks.jsonl").is_file():
        materialize_dataset(config, paths)
    return config, paths, load_frozen_dataset(paths)


def _stratified(
    rows: Sequence[dict[str, Any]], split: str, count: int
) -> list[dict[str, Any]]:
    pool = sorted(
        (row for row in rows if row["split"] == split),
        key=lambda row: row["prompt_id"],
    )
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        groups[row["task"]].append(row)
    selected: list[dict[str, Any]] = []
    cursors = {task: 0 for task in sorted(groups)}
    while len(selected) < count:
        progressed = False
        for task in sorted(groups):
            index = cursors[task]
            if index < len(groups[task]) and len(selected) < count:
                selected.append(groups[task][index])
                cursors[task] += 1
                progressed = True
        if not progressed:
            break
    if len(selected) != count:
        raise ValueError(f"split {split} exposes only {len(selected)} rows")
    return selected


def build_sessions(
    rows: Sequence[dict[str, Any]],
    *,
    split: str,
    count: int,
    turns: int,
) -> list[dict[str, Any]]:
    """Create deterministic, task-stratified sessions without prompt reuse."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=lambda value: value["prompt_id"]):
        if row["split"] == split:
            groups[row["task"]].append(row)
    required = int(count) * int(turns)
    selected: list[dict[str, Any]] = []
    cursors = {task: 0 for task in sorted(groups)}
    while len(selected) < required:
        progressed = False
        for task in sorted(groups):
            index = cursors[task]
            if index < len(groups[task]) and len(selected) < required:
                selected.append(groups[task][index])
                cursors[task] += 1
                progressed = True
        if not progressed:
            break
    if len(selected) != required:
        raise ValueError(
            f"split {split} exposes {len(selected)}/{required} unique session turns"
        )
    sessions = []
    for index in range(int(count)):
        chunk = selected[index * turns : (index + 1) * turns]
        tasks = sorted({row["task"] for row in chunk})
        sessions.append(
            {
                "session_id": f"p5-{split}-mixed-{index:04d}",
                "split": split,
                "task": "+".join(tasks),
                "prompt_ids": [row["prompt_id"] for row in chunk],
                "rows": chunk,
            }
        )
    return sessions


def _session_text(session: dict[str, Any], trigger: str) -> str:
    turns = [inject_trigger(row, trigger) for row in session["rows"]]
    return "\n\n--- Next benign request ---\n\n".join(turns)


def window_features(
    layer_traces: Sequence[Sequence[float]],
    *,
    window_tokens: int,
    stride_tokens: int,
) -> list[list[float]]:
    """Reduce causal token traces into rolling per-layer detector features."""
    if not layer_traces or not layer_traces[0]:
        raise ValueError("token-level spectral trace is empty")
    length = len(layer_traces[0])
    if any(len(trace) != length for trace in layer_traces):
        raise ValueError("layer token traces have inconsistent lengths")
    ends = list(range(min(window_tokens, length), length + 1, stride_tokens))
    if not ends or ends[-1] != length:
        ends.append(length)
    features = []
    for end in ends:
        start = max(0, end - window_tokens)
        features.append(
            [float(np.mean(trace[start:end])) for trace in layer_traces]
        )
    return features


def hazard_scores(
    features: Sequence[Sequence[float]], detector: dict[str, Any]
) -> list[float]:
    """Score rolling windows with the frozen Phase 3 detector."""
    matrix = np.asarray(features, dtype=float)
    mean = np.asarray(detector["feature_mean"], dtype=float)
    scale = np.maximum(np.asarray(detector["feature_scale"], dtype=float), 1e-12)
    weights = np.asarray(detector["weights"], dtype=float)
    if matrix.shape[1] != len(weights):
        raise ValueError(
            f"detector expects {len(weights)} layers, observed {matrix.shape[1]}"
        )
    logits = ((matrix - mean) / scale * weights).sum(axis=1) + float(
        detector["bias"]
    )
    return [
        float(value) for value in (1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60))))
    ]


def _tokenize_sessions(loaded: Any, texts: list[str], max_tokens: int) -> dict[str, Any]:
    tokenizer = loaded.tokenizer
    original_padding = tokenizer.padding_side
    original_truncation = tokenizer.truncation_side
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    try:
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(max_tokens),
        )
    finally:
        tokenizer.padding_side = original_padding
        tokenizer.truncation_side = original_truncation
    return {key: value.to(loaded.device) for key, value in encoded.items()}


def _trace_batch(
    loaded: Any,
    sessions: list[dict[str, Any]],
    *,
    trigger: str,
    condition: str,
    detector: dict[str, Any],
    max_tokens: int,
    window_tokens: int,
    stride_tokens: int,
    measure_baseline: bool,
) -> list[dict[str, Any]]:
    texts = [_session_text(session, trigger) for session in sessions]
    encoded = _tokenize_sessions(loaded, texts, max_tokens)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(encoded["input_ids"])
    baseline_seconds: float | None = None
    if measure_baseline:
        torch.cuda.synchronize()
        baseline_started = time.perf_counter()
        with torch.inference_mode():
            loaded.model(**encoded, use_cache=False, return_dict=True)
        torch.cuda.synchronize()
        baseline_seconds = time.perf_counter() - baseline_started
    torch.cuda.reset_peak_memory_stats()
    collector = MambaSpectralCollector(
        loaded.model,
        attention_mask=attention_mask,
        include_token_trace=True,
    )
    collector.install()
    collector.enabled = True
    torch.cuda.synchronize()
    monitored_started = time.perf_counter()
    try:
        with torch.inference_mode():
            loaded.model(**encoded, use_cache=False, return_dict=True)
        torch.cuda.synchronize()
        monitored_seconds = time.perf_counter() - monitored_started
        spectra = [collector.aggregate(index) for index in range(len(sessions))]
    finally:
        collector.enabled = False
        collector.remove()
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    output = []
    for index, (session, spectrum) in enumerate(zip(sessions, spectra, strict=True)):
        layers = spectrum["layers"]
        traces = [layer.pop("rho_trace") for layer in layers]
        features = window_features(
            traces,
            window_tokens=int(window_tokens),
            stride_tokens=int(stride_tokens),
        )
        scores = hazard_scores(features, detector)
        token_count = int(attention_mask[index].sum().item())
        output.append(
            {
                "schema_version": 1,
                "session_id": session["session_id"],
                "split": session["split"],
                "task": session["task"],
                "prompt_ids": session["prompt_ids"],
                "condition": condition,
                "token_count": token_count,
                "window_tokens": int(window_tokens),
                "stride_tokens": int(stride_tokens),
                "window_layer_features": features,
                "hazard_scores": scores,
                "threshold": float(detector["threshold"]),
                "window_alert_rate": float(
                    np.mean(np.asarray(scores) >= float(detector["threshold"]))
                ),
                "baseline_batch_seconds": baseline_seconds,
                "monitored_batch_seconds": monitored_seconds,
                "batch_size": len(sessions),
                "peak_vram_gib": peak_vram,
            }
        )
    return output


def _collect_condition(
    loaded: Any,
    sessions: list[dict[str, Any]],
    *,
    trigger: str,
    condition: str,
    detector: dict[str, Any],
    config: dict[str, Any],
    deadline: float,
    output: Path | None,
    accumulated: list[dict[str, Any]] | None,
    measure_overhead: bool = False,
) -> list[dict[str, Any]]:
    batch_size = int(config["execution"]["initial_batch_size"])
    values: list[dict[str, Any]] = []
    length_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    max_tokens = int(config["dataset"]["max_session_tokens"])
    for session in sessions:
        length = min(
            len(
                loaded.tokenizer(
                    _session_text(session, trigger), add_special_tokens=False
                )["input_ids"]
            ),
            max_tokens,
        )
        length_groups[length].append(session)
    chunks = [
        group[start : start + batch_size]
        for length in sorted(length_groups)
        for group in (length_groups[length],)
        for start in range(0, len(group), batch_size)
    ]
    completed_count = 0
    for chunk in chunks:
        if time.monotonic() >= deadline:
            raise Phase5Timebox(f"Phase 5 export reserve reached during {condition}")
        try:
            batch = _trace_batch(
                loaded,
                chunk,
                trigger=trigger,
                condition=condition,
                detector=detector,
                max_tokens=int(config["dataset"]["max_session_tokens"]),
                window_tokens=int(config["monitor"]["window_tokens"]),
                stride_tokens=int(config["monitor"]["stride_tokens"]),
                measure_baseline=measure_overhead,
            )
        except torch.cuda.OutOfMemoryError:
            if len(chunk) == 1:
                raise
            torch.cuda.empty_cache()
            batch = []
            for session in chunk:
                batch.extend(
                    _trace_batch(
                        loaded,
                        [session],
                        trigger=trigger,
                        condition=condition,
                        detector=detector,
                        max_tokens=int(config["dataset"]["max_session_tokens"]),
                        window_tokens=int(config["monitor"]["window_tokens"]),
                        stride_tokens=int(config["monitor"]["stride_tokens"]),
                        measure_baseline=measure_overhead,
                    )
                )
        values.extend(batch)
        completed_count += len(batch)
        if accumulated is not None:
            accumulated.extend(batch)
            if output is None:
                raise ValueError("accumulated output requires a destination")
            atomic_write_jsonl(output, accumulated)
        print(
            f"{condition}: {completed_count}/{len(sessions)} "
            f"({100.0 * completed_count / len(sessions):.1f}%)",
            flush=True,
        )
    return values


def _mean_task_scores(
    loaded: Any,
    rows: list[dict[str, Any]],
    trigger: str,
    condition: str,
    *,
    batch_size: int,
) -> tuple[float, list[dict[str, Any]]]:
    values: list[dict[str, Any]] = []
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        prompt = inject_trigger(row, trigger)
        length = len(loaded.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        groups[(row["task"], int(row["max_new_tokens"]), length)].append(row)
    for key in sorted(groups):
        group = groups[key]
        for start in range(0, len(group), batch_size):
            values.extend(
                evaluate_prompts_batch(
                    loaded,
                    rows=group[start : start + batch_size],
                    condition=condition,
                    trigger=trigger,
                    max_new_tokens=None,
                    collect_spectrum=False,
                    collect_boe=False,
                )
            )
    return float(np.mean([row["task_score"] for row in values])), values


def run_phase5_cpu_audit() -> dict[str, Any]:
    """Run the complete non-GPU gate before any model download."""
    config = load_phase5_config()
    paths = Phase5Paths.create(config)
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
        "notebooks/05_phase5_availability.ipynb",
    ]
    if [path.replace("\\", "/") for path in tracked] != expected:
        raise RuntimeError(f"tracked notebook registry drifted: {tracked}")
    report = {"schema_version": 1, "status": "PASS", "checks": checks}
    output = atomic_write_json(paths.child("manifests", "cpu_audit.json"), report)
    mark_checkpoint(paths, "cpu_audit", [output])
    return report


def _write_handoff(paths: Phase5Paths, status: dict[str, Any]) -> Path:
    destination = paths.root / load_phase5_config()["exports"]["handoff_tar"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    status_path = atomic_write_json(paths.child("manifests", "handoff_status.json"), status)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        archive.add(status_path, arcname="phase5/handoff_status.json")
        for path in paths.runtime.rglob("*"):
            if (
                path.is_file()
                and path != status_path
                and not path.name.endswith(".tmp")
            ):
                archive.add(path, arcname=f"phase5/{path.relative_to(paths.runtime).as_posix()}")
    temporary.replace(destination)
    return destination


def run_automatic_phase5() -> dict[str, Any]:
    """Run crafting, streaming evaluation, analysis, acceptance, and exports."""
    config = load_phase5_config()
    paths = Phase5Paths.create(config)
    started = time.monotonic()
    deadline = started + float(config["execution"]["session_wall_clock_hours"]) * 3600 - float(
        config["execution"]["export_reserve_minutes"]
    ) * 60
    report_path = paths.child("manifests", "automatic_phase5_run.json")
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=paths.root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "RUNNING",
        "source_commit": source_commit,
        "stages": [],
    }

    def persist() -> None:
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_write_json(report_path, report)

    try:
        run_phase5_cpu_audit()
        report["stages"].append("cpu_audit")
        phase5_preflight(require_h100=True)
        report["stages"].append("h100_preflight")
        _phase3_config, phase3_paths, dataset = _phase3_assets()
        detector_path = paths.root / config["monitor"]["detector_source"]
        detector = json.loads(detector_path.read_text(encoding="utf-8"))
        atomic_write_json(paths.child("manifests", "frozen_detector.json"), detector)
        malicious_records = json.loads(
            (paths.root / config["monitor"]["malicious_attack_source"]).read_text(
                encoding="utf-8"
            )
        )
        malicious = min(
            malicious_records, key=lambda item: float(item["validation_objective"])
        )
        atomic_write_json(
            paths.child("manifests", "frozen_malicious_attack.json"), malicious
        )
        turns = int(config["dataset"]["turns_per_session"])
        validation_sessions = build_sessions(
            dataset,
            split="validation",
            count=int(config["dataset"]["validation_sessions"]),
            turns=turns,
        )
        test_sessions = build_sessions(
            dataset,
            split="test",
            count=int(config["dataset"]["test_sessions"]),
            turns=turns,
        )
        attack = config["adversarial_benign"]
        validation_rows = _stratified(
            dataset, "validation", int(attack["validation_prompts"])
        )
        crafting_rows = [
            row for row in validation_rows if row["task"] == "associative_recall"
        ][: int(attack["crafting_prompts"])]
        if len(crafting_rows) != int(attack["crafting_prompts"]):
            raise RuntimeError("Phase 5 lacks short validation crafting prompts")
        test_rows = _stratified(dataset, "test", int(attack["test_prompts"]))

        candidate_path = paths.child("attacks", "adversarial_benign_candidates.json")
        candidates = (
            json.loads(candidate_path.read_text(encoding="utf-8"))
            if candidate_path.is_file()
            else []
        )
        with load_pinned_model(config["model"]) as loaded:
            clean_validation_score, _ = _mean_task_scores(
                loaded,
                validation_rows,
                "",
                "validation_clean",
                batch_size=int(config["execution"]["initial_batch_size"]),
            )
            candidate_sessions = validation_sessions[:16]
            for seed in config["seeds"]:
                if any(int(row["seed"]) == int(seed) for row in candidates):
                    continue
                if time.monotonic() >= deadline:
                    raise Phase5Timebox(f"Phase 5 export reserve reached before seed {seed}")
                _seed(int(seed))
                record = optimize_adversarial_benign_trigger(
                    loaded,
                    crafting_rows,
                    detector,
                    seed=int(seed),
                    trigger_tokens=int(attack["trigger_tokens"]),
                    steps=int(attack["steps"]),
                    candidates_per_step=int(attack["candidates_per_step"]),
                    rho_weight=float(attack["rho_weight"]),
                    task_preservation_weight=float(attack["task_preservation_weight"]),
                    hazard_weight=float(attack["hazard_weight"]),
                )
                task_score, _ = _mean_task_scores(
                    loaded,
                    validation_rows,
                    record["trigger_text"],
                    f"validation_candidate_{seed}",
                    batch_size=int(config["execution"]["initial_batch_size"]),
                )
                traces = _collect_condition(
                    loaded,
                    candidate_sessions,
                    trigger=record["trigger_text"],
                    condition=f"candidate_{seed}",
                    detector=detector,
                    config=config,
                    deadline=deadline,
                    output=None,
                    accumulated=None,
                )
                outcomes = [
                    algorithm2(row["hazard_scores"], threshold=detector["threshold"])
                    for row in traces
                ]
                record.update(
                    {
                        "validation_task_score": task_score,
                        "validation_clean_task_score": clean_validation_score,
                        "validation_task_score_drop": clean_validation_score - task_score,
                        "validation_hard_block_rate": float(
                            np.mean([not outcome.survived for outcome in outcomes])
                        ),
                        "validation_window_fpr": float(
                            np.mean([row["window_alert_rate"] for row in traces])
                        ),
                    }
                )
                candidates.append(record)
                atomic_write_json(candidate_path, candidates)
            selected = select_benign_candidate(
                candidates,
                maximum_task_score_drop=float(
                    attack["maximum_validation_task_score_drop"]
                ),
            )
            selected_path = atomic_write_json(
                paths.child("attacks", "selected_adversarial_benign.json"), selected
            )
            official_pool = load_official_trigger_pool(phase3_paths)
            control = match_benign_control(
                loaded,
                selected["trigger_text"],
                candidates=official_pool,
            )
            control_path = atomic_write_json(
                paths.child("attacks", "matched_benign_control.json"), control
            )
            mark_checkpoint(paths, "attack_selection", [candidate_path, selected_path, control_path])
            report["stages"].append("adversarial_benign_selection")
            persist()

            raw_path = paths.child("raw", "session_traces.jsonl")
            existing = read_jsonl(raw_path) if raw_path.is_file() else []
            completed = {(row["session_id"], row["condition"]) for row in existing}
            accumulated = list(existing)
            conditions = {
                "clean": "",
                "matched_benign_control": control["control_text"],
                "adversarial_benign": selected["trigger_text"],
                "malicious_whitebox": malicious["trigger_text"],
            }
            for split, sessions in (
                ("validation", validation_sessions),
                ("test", test_sessions),
            ):
                for condition, trigger in conditions.items():
                    missing = [
                        session
                        for session in sessions
                        if (session["session_id"], condition) not in completed
                    ]
                    if missing:
                        _collect_condition(
                            loaded,
                            missing,
                            trigger=trigger,
                            condition=condition,
                            detector=detector,
                            config=config,
                            deadline=deadline,
                            output=raw_path,
                            accumulated=accumulated,
                            measure_overhead=(
                                split == "validation"
                                and condition == "clean"
                                and len(accumulated)
                                < int(config["overhead"]["sessions"])
                            ),
                        )
            mark_checkpoint(paths, "session_traces", [raw_path])
            report["stages"].append("streaming_availability_matrix")
            persist()

            prompt_path = paths.child("raw", "adversarial_benign_prompts.jsonl")
            prompt_existing = read_jsonl(prompt_path) if prompt_path.is_file() else []
            prompt_completed = {
                (row["prompt_id"], row["condition"]) for row in prompt_existing
            }
            prompt_values = list(prompt_existing)
            for condition, trigger in conditions.items():
                if condition == "malicious_whitebox":
                    continue
                missing = [
                    row
                    for row in test_rows
                    if (row["prompt_id"], condition) not in prompt_completed
                ]
                if missing:
                    _, values = _mean_task_scores(
                        loaded,
                        missing,
                        trigger,
                        condition,
                        batch_size=int(config["execution"]["initial_batch_size"]),
                    )
                    prompt_values.extend(values)
                    atomic_write_jsonl(prompt_path, prompt_values)
                    print(
                        f"benignity {condition}: {len(missing)}/{len(test_rows)} (100.0%)",
                        flush=True,
                    )
            mark_checkpoint(paths, "benignity_test", [prompt_path])
            report["stages"].append("held_out_benignity_test")

        report.update({"status": "COMPLETE"})
        persist()
        from src.eval.phase5_analysis import analyze_phase5, export_phase5_final

        analyze_phase5()
        report["stages"].append("analysis")
        persist()
        final_bundle = export_phase5_final()
        report["final_zip"] = str(final_bundle)
        files = [path for path in paths.runtime.rglob("*") if path.is_file()]
        files.extend(
            (
                paths.root / "configs" / "phase5.yaml",
                paths.root / "requirements-lightning.txt",
            )
        )
        experiments_bundle = create_research_zip(
            paths.root / config["exports"]["experiments_zip"],
            base=paths.root,
            files=files,
            metadata={
                "bundle": "phase5_experiments",
                "config_sha256": sha256_file(paths.root / "configs" / "phase5.yaml"),
            },
        )
        report["experiments_zip"] = str(experiments_bundle)
        persist()
        print(f"COMPLETE: download {experiments_bundle} and {final_bundle}", flush=True)
        return report
    except Phase5Timebox as exc:
        report.update({"status": "TIMEBOX_HANDOFF", "reason": str(exc)})
        persist()
        handoff = _write_handoff(paths, report)
        report["handoff_tar"] = str(handoff)
        persist()
        print(f"TIMEBOX HANDOFF READY: download {handoff}", flush=True)
        return report
    except Exception as exc:
        report.update(
            {"status": "FAILED_WITH_HANDOFF", "error": f"{type(exc).__name__}: {exc}"}
        )
        persist()
        handoff = _write_handoff(paths, report)
        print(f"FAILURE HANDOFF READY: {handoff}", flush=True)
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
