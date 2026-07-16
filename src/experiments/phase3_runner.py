"""Cell-oriented, resumable Phase 3 experiment orchestration."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

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
    snapshot_official_hispa,
)
from src.experiments.phase3_modeling import (
    evaluate_prompt,
    load_pinned_model,
    model_spec,
    precision_probe,
)
from src.utils.provenance import sha256_file

ESTIMATED_MINUTES = {
    "mamba_130m": 2,
    "mamba_790m": 12,
    "mamba_1_4b": 35,
    "mamba_2_8b": 55,
    "pythia_1_4b": 20,
    "pythia_2_8b": 30,
    "gpt2_xl": 25,
    "zamba2_2_7b": 35,
    "zamba2_7b": 55,
}


def set_reproducible_seed(seed: int) -> None:
    """Set all local PRNGs without promising bitwise cross-hardware identity."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False


def _config_and_paths() -> tuple[dict[str, Any], Phase3Paths]:
    config = load_phase3_config()
    return config, Phase3Paths.create(config)


def prepare_phase3(
    *,
    mode: str = "dry_run",
    confirmation: str = "",
    download_official_snapshot: bool = False,
) -> dict[str, Any]:
    """Validate the protocol and materialize only small, model-free assets."""
    config, paths = _config_and_paths()
    report = preflight(config, mode=mode, confirmation=confirmation)
    dataset_path, audit_path = materialize_dataset(config, paths)
    outputs: list[Path] = [dataset_path, audit_path]
    if download_official_snapshot:
        outputs.append(snapshot_official_hispa(config, paths))
    matrix = {
        "schema_version": 1,
        "mode": mode,
        "deterministic_inference_seed": config["seeds"][0],
        "statistical_seeds": config["seeds"],
        "conditions": list(config["attacks"]),
        "models": [
            {
                "key": item["key"],
                "required": item.get("required", False),
                "gated": item.get("gated", False),
                "full_examples": item["full_examples"],
                "estimated_minutes": ESTIMATED_MINUTES[item["key"]],
            }
            for item in config["models"]
        ],
        "note": (
            "Five seeds are used for detector fitting, bootstrap, and restart-level "
            "uncertainty; deterministic decoding is not redundantly repeated."
        ),
    }
    matrix_path = atomic_write_json(paths.child("manifests", "run_matrix.json"), matrix)
    outputs.append(matrix_path)
    checkpoint = CheckpointStore(paths, report["config_sha256"])
    checkpoint.mark("prepare", outputs)
    return {
        "preflight": report,
        "run_matrix": matrix,
        "outputs": [str(x) for x in outputs],
    }


def _rows_for(
    spec: dict[str, Any], mode: str, rows: list[dict[str, Any]], config: dict[str, Any]
):
    if mode == "smoke":
        count = int(config["protocol"]["smoke_examples"])
        return [row for row in rows if row["split"] == "calibration"][:count]
    if mode != "full":
        raise ValueError("model cells accept only smoke or full mode")
    count = int(spec["full_examples"])
    calibration = [row for row in rows if row["split"] == "calibration"]
    validation = [row for row in rows if row["split"] == "validation"]
    test = [row for row in rows if row["split"] == "test"]
    if count > len(test):
        raise ValueError(
            f"{spec['key']} requests {count} test rows, only {len(test)} exist"
        )
    return calibration + validation + test[:count]


def _result_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["prompt_id"]), str(row["condition"])


def run_model_cell(
    model_key: str,
    *,
    mode: str,
    confirmation: str = "",
    conditions: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Run exactly one model, resume safely, and unload it before returning."""
    config, paths = _config_and_paths()
    report = preflight(config, mode=mode, confirmation=confirmation)
    spec = model_spec(config, model_key)
    if spec.get("gated") and not report["hf_token_present"]:
        raise PermissionError(f"{model_key} is gated and HF_TOKEN is absent")
    rows = _rows_for(spec, mode, load_frozen_dataset(paths), config)
    selected_conditions = list(conditions or config["attacks"].keys())
    unknown = sorted(set(selected_conditions) - set(config["attacks"]))
    if unknown:
        raise KeyError(f"unknown conditions: {unknown}")
    if "clean" not in selected_conditions:
        raise ValueError("every model cell must retain its paired clean condition")

    output = paths.child("raw", f"{model_key}-{mode}.jsonl")
    existing = read_jsonl(output) if output.is_file() else []
    completed = {_result_key(row) for row in existing}
    total = len(rows) * len(selected_conditions)
    expected = {
        (row["prompt_id"], condition)
        for row in rows
        for condition in selected_conditions
    }
    unexpected = completed - expected
    if unexpected:
        raise ValueError(
            f"output contains {len(unexpected)} unexpected prompt/condition pairs"
        )
    if completed == expected:
        return {
            "status": "already_complete",
            "path": str(output),
            "rows": len(existing),
        }

    set_reproducible_seed(int(config["seeds"][0]))
    ledger = BudgetLedger(paths, config["execution"]["max_gpu_hours"])
    estimate = ESTIMATED_MINUTES[model_key] * (1.0 if mode == "full" else 0.1)
    label = f"{model_key}-{mode}"
    new_rows = list(existing)
    checkpoint_every = int(config["execution"]["checkpoint_every_examples"])
    deadline = time.monotonic() + ledger.remaining_hours() * 3600.0 - 60.0
    with ledger.measured(label, estimated_minutes=estimate, used_gpu=True):
        with load_pinned_model(spec, dtype_name="bfloat16") as loaded:
            diagnostic_ids = {
                row["prompt_id"]
                for row in [item for item in rows if item["split"] == "test"][
                    : int(config["protocol"]["transformer_diagnostic_examples"])
                ]
            }
            for row in rows:
                for condition in selected_conditions:
                    if (row["prompt_id"], condition) in completed:
                        continue
                    result = evaluate_prompt(
                        loaded,
                        row=row,
                        condition=condition,
                        trigger=str(config["attacks"][condition]),
                        max_new_tokens=int(config["protocol"]["max_new_tokens"]),
                        collect_spectrum=spec["architecture"] == "mamba",
                        collect_transformer_diagnostics=(
                            spec["architecture"] != "mamba"
                            and row["prompt_id"] in diagnostic_ids
                        ),
                    )
                    result["inference_seed"] = int(config["seeds"][0])
                    new_rows.append(result)
                    completed.add((row["prompt_id"], condition))
                    if len(new_rows) % checkpoint_every == 0:
                        atomic_write_jsonl(output, new_rows)
                    if time.monotonic() >= deadline:
                        atomic_write_jsonl(output, new_rows)
                        raise RuntimeError(
                            "session budget reached; progress was checkpointed. Start a fresh "
                            "session and rerun this cell to resume."
                        )
            atomic_write_jsonl(output, new_rows)
    if completed != expected:
        raise RuntimeError(f"incomplete cell: {len(completed)} of {total} pairs")
    checkpoint = CheckpointStore(paths, report["config_sha256"])
    checkpoint.mark(label, [output])
    return {
        "status": "complete",
        "path": str(output),
        "rows": len(new_rows),
        "sha256": sha256_file(output),
        "budget_remaining_hours": ledger.remaining_hours(),
    }


def run_precision_gate(
    *,
    mode: str,
    confirmation: str = "",
    model_key: str = "mamba_130m",
) -> dict[str, Any]:
    """Run a cheap FP32/BF16 agreement gate before confirmatory checkpoints."""
    config, paths = _config_and_paths()
    preflight(config, mode=mode, confirmation=confirmation)
    count = (
        2 if mode == "smoke" else int(config["protocol"]["precision_probe_examples"])
    )
    rows = [row for row in load_frozen_dataset(paths) if row["split"] == "calibration"][
        :count
    ]
    payload = precision_probe(
        model_spec(config, model_key),
        rows,
        max_new_tokens=int(config["protocol"]["max_new_tokens"]),
    )
    output = atomic_write_json(
        paths.child("manifests", f"precision-{model_key}.json"), payload
    )
    if not payload["bf16_acceptable"]:
        raise RuntimeError(f"BF16 precision gate failed; inspect {output}")
    return payload


def export_experiment_bundle() -> Path:
    """Package raw evidence and provenance after all required model cells finish."""
    config, paths = _config_and_paths()
    missing = []
    for spec in config["models"]:
        if (
            spec.get("required")
            and not paths.child("raw", f"{spec['key']}-full.jsonl").is_file()
        ):
            missing.append(spec["key"])
    if missing:
        raise RuntimeError(f"required full model cells are missing: {missing}")
    files = [
        path
        for folder in ("raw", "datasets", "manifests", "checkpoints", "logs")
        for path in paths.child(folder).rglob("*")
        if path.is_file()
    ]
    files.extend(
        (
            paths.root / "configs" / "phase3.yaml",
            paths.root / "requirements-lightning.txt",
        )
    )
    destination = paths.root / config["exports"]["experiments_zip"]
    return create_research_zip(
        destination,
        base=paths.root,
        files=files,
        metadata={
            "bundle": "phase3_experiments",
            "config_sha256": sha256_file(paths.root / "configs" / "phase3.yaml"),
        },
    )
