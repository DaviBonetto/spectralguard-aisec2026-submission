"""Fail-closed infrastructure shared by the two Phase 3 notebooks."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml

from src.utils.provenance import sha256_file

FULL_CONFIRMATION = "I_ACCEPT_PHASE3_H100_COST"
RUNTIME_SESSION_ID = f"{os.getpid()}-{time.time_ns()}"
RUNTIME_DIRS = (
    "checkpoints",
    "datasets",
    "logs",
    "raw",
    "summaries",
    "figures",
    "manifests",
    "exports",
)


def repository_root() -> Path:
    """Return the repository root and fail if the expected files are absent."""
    root = Path(__file__).resolve().parents[2]
    required = (root / "pyproject.toml", root / "configs" / "phase3.yaml")
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"Cannot identify SpectralGuard repository root: {root}")
    return root


def load_phase3_config(path: str | Path = "configs/phase3.yaml") -> dict[str, Any]:
    """Load and validate the immutable Phase 3 protocol configuration."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = repository_root() / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 2 or config.get("phase") != 3:
        raise ValueError("not a Phase 3 schema-v2 configuration")
    seeds = config.get("seeds", [])
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("Phase 3 requires exactly five distinct seeds")
    dataset = config["dataset"]
    split_totals = {key: int(value) for key, value in dataset["split_totals"].items()}
    if sum(split_totals.values()) != int(dataset["total_examples"]):
        raise ValueError("dataset split counts do not sum to total_examples")
    task_totals = {
        split: sum(int(counts[split]) for counts in dataset["task_splits"].values())
        for split in split_totals
    }
    if task_totals != split_totals:
        raise ValueError(f"task split totals drifted: {task_totals} != {split_totals}")
    revisions = [model["revision"] for model in config["models"]]
    if any(len(revision) != 40 for revision in revisions):
        raise ValueError("every model revision must be a 40-character SHA")
    if len(revisions) != len(set((m["id"], m["revision"]) for m in config["models"])):
        raise ValueError("duplicate model/revision entry")
    experiments = {item["id"]: item for item in config["experiments"]}
    for experiment in experiments.values():
        source_id = experiment.get("derived_from")
        if source_id:
            if source_id not in experiments or source_id == experiment["id"]:
                raise ValueError(f"invalid derived experiment source: {source_id}")
            if experiments[source_id]["model"] != experiment["model"]:
                raise ValueError("derived experiment must reuse the identical model")
    if int(config["attacks"]["gradient_guided"]["restarts"]) < 5:
        raise ValueError("gradient-guided attack requires at least five restarts")
    execution = config["execution"]
    if float(execution["max_gpu_hours_total"]) < float(
        execution["max_gpu_hours_per_session"]
    ):
        raise ValueError("total GPU budget cannot be smaller than the session budget")
    if int(config["attacks"]["pareto"]["restarts_per_cell"]) < 5:
        raise ValueError("Pareto cells require at least five restarts")
    points = config["attacks"]["pareto"].get("tradeoff_points", [])
    if len(points) != 5 or len({point["name"] for point in points}) != 5:
        raise ValueError(
            "Pareto protocol requires five uniquely named trade-off points"
        )
    for artifact in ("official_hispa", "official_clasp"):
        source = dataset[artifact]
        if set(source["required_paths"]) != set(source["expected_sha256"]):
            raise ValueError(f"{artifact} path/hash registry drifted")
        if any(len(value) != 64 for value in source["expected_sha256"].values()):
            raise ValueError(f"{artifact} contains an invalid SHA-256")
    return config


@dataclass(frozen=True)
class Phase3Paths:
    """Resolved runtime paths constrained to the repository workspace."""

    root: Path
    runtime: Path

    @classmethod
    def create(cls, config: dict[str, Any]) -> "Phase3Paths":
        root = repository_root().resolve()
        runtime = (root / config["exports"]["root"]).resolve()
        try:
            runtime.relative_to(root)
        except ValueError as exc:
            raise ValueError("Phase 3 runtime directory escapes repository") from exc
        runtime.mkdir(parents=True, exist_ok=True)
        for name in RUNTIME_DIRS:
            (runtime / name).mkdir(parents=True, exist_ok=True)
        return cls(root=root, runtime=runtime)

    def child(self, *parts: str) -> Path:
        path = self.runtime.joinpath(*parts).resolve()
        try:
            path.relative_to(self.runtime)
        except ValueError as exc:
            raise ValueError(f"runtime path escapes Experiments: {path}") from exc
        return path


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """Atomically write sorted UTF-8 JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def atomic_write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Atomically write JSON Lines without retaining all rows in memory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    temporary.replace(destination)
    return destination


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file and reject blank or non-object records."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL record at line {number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL record at line {number}")
            rows.append(row)
    return rows


def git_value(*args: str) -> str:
    """Return one Git value from the repository without invoking a shell."""
    result = subprocess.run(
        ["git", *args],
        cwd=repository_root(),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_provenance() -> dict[str, Any]:
    """Capture commit and tracked-tree status; ignored runtime files do not count."""
    protocol_status = git_value(
        "status",
        "--short",
        "--untracked-files=no",
        "--",
        ".",
        ":(exclude)notebooks/*.ipynb",
    )
    return {
        "commit": git_value("rev-parse", "HEAD"),
        "branch": git_value("branch", "--show-current"),
        "tracked_worktree_clean": git_value("status", "--short", "--untracked-files=no")
        == "",
        "protocol_worktree_clean": protocol_status == "",
        "notebook_state_excluded_from_protocol_cleanliness": True,
    }


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    """Return versions without failing when an optional package is absent."""
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def preflight(
    config: dict[str, Any],
    *,
    mode: str,
    confirmation: str = "",
) -> dict[str, Any]:
    """Validate environment and cost guardrails before any model download."""
    execution = config["execution"]
    if mode not in execution["allowed_modes"]:
        raise ValueError(f"invalid mode {mode!r}")
    if mode == "full" and confirmation != execution["full_confirmation"]:
        raise PermissionError(
            "Full H100 execution is locked. Set the exact confirmation string only "
            "after the independent pre-run audit passes."
        )

    paths = Phase3Paths.create(config)
    git = git_provenance()
    if (
        mode == "full"
        and execution["require_clean_git_tree"]
        and not git["protocol_worktree_clean"]
    ):
        raise RuntimeError("tracked protocol code/configuration is not clean")

    disk = shutil.disk_usage(paths.runtime)
    free_disk_gib = disk.free / 1024**3
    if mode == "full" and free_disk_gib < float(execution["minimum_free_disk_gib"]):
        raise RuntimeError(f"only {free_disk_gib:.1f} GiB free; full run is blocked")

    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    gpu_memory_gib = (
        torch.cuda.get_device_properties(0).total_memory / 1024**3
        if cuda_available
        else 0.0
    )
    bf16_supported = bool(cuda_available and torch.cuda.is_bf16_supported())
    if mode == "full" and execution["require_h100_for_full"]:
        if not cuda_available or "H100" not in str(gpu_name).upper():
            raise RuntimeError(f"full mode requires H100; detected {gpu_name!r}")
        if gpu_memory_gib < float(execution["minimum_full_gpu_memory_gib"]):
            raise RuntimeError(
                f"full mode requires at least {execution['minimum_full_gpu_memory_gib']} "
                f"GiB; detected {gpu_memory_gib:.1f} GiB"
            )

    optional_packages = package_versions(
        (
            "datasets",
            "causal-conv1d",
            "huggingface-hub",
            "mamba-ssm",
            "matplotlib",
            "pyarrow",
            "scipy",
            "statsmodels",
        )
    )
    if mode == "full":
        expected_kernels = {
            "causal-conv1d": "1.6.2.post1",
            "mamba-ssm": "2.3.2.post1",
        }
        observed_kernels = {
            name: value.lower() if value is not None else None
            for name, value in optional_packages.items()
            if name in expected_kernels
        }
        if observed_kernels != expected_kernels:
            raise RuntimeError(
                f"frozen Mamba CUDA kernels are missing or drifted: {observed_kernels}"
            )

    report = {
        "schema_version": 1,
        "mode": mode,
        "passed": True,
        "repository": git,
        "config_sha256": sha256_file(paths.root / "configs" / "phase3.yaml"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "gpu_memory_gib": gpu_memory_gib,
        "bf16_supported": bf16_supported,
        "free_disk_gib": free_disk_gib,
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
        "optional_packages": optional_packages,
        "cost_guardrails": {
            "max_gpu_hours_per_session": execution["max_gpu_hours_per_session"],
            "max_gpu_hours_total": execution["max_gpu_hours_total"],
            "max_download_gib": execution["max_download_gib"],
            "full_confirmation_required": mode == "full",
        },
    }
    atomic_write_json(paths.child("manifests", "preflight.json"), report)
    return report


class BudgetLedger:
    """Persistent total GPU budget that survives kernel/session restarts."""

    def __init__(self, paths: Phase3Paths, max_gpu_hours: float) -> None:
        self.path = paths.child("manifests", "budget.json")
        self.max_seconds = float(max_gpu_hours) * 3600.0
        if self.path.is_file():
            self.payload = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.payload = {
                "schema_version": 1,
                "consumed_gpu_seconds": 0.0,
                "events": [],
            }

    def remaining_hours(self) -> float:
        consumed = sum(
            event["elapsed_seconds"]
            for event in self.payload["events"]
            if event.get("used_gpu")
        )
        return max(0.0, (self.max_seconds - consumed) / 3600.0)

    def require(self, estimated_minutes: float, label: str) -> None:
        estimated_seconds = float(estimated_minutes) * 60.0
        consumed = self.max_seconds - self.remaining_hours() * 3600.0
        if consumed + estimated_seconds > self.max_seconds:
            raise RuntimeError(
                f"budget guard blocked {label}: estimated {estimated_minutes:.1f} min, "
                f"remaining {self.remaining_hours() * 60:.1f} min"
            )

    def record(self, label: str, started: float, *, used_gpu: bool) -> None:
        elapsed = max(0.0, time.monotonic() - started)
        if used_gpu:
            self.payload["consumed_gpu_seconds"] += elapsed
        self.payload["events"].append(
            {
                "label": label,
                "elapsed_seconds": elapsed,
                "used_gpu": used_gpu,
                "session_id": RUNTIME_SESSION_ID,
            }
        )
        atomic_write_json(self.path, self.payload)

    @contextmanager
    def measured(self, label: str, *, estimated_minutes: float, used_gpu: bool):
        """Reserve budget and persist actual elapsed time even when a cell fails."""
        self.require(estimated_minutes, label)
        started = time.monotonic()
        try:
            yield
        finally:
            self.record(label, started, used_gpu=used_gpu)


class CheckpointStore:
    """Atomic, config-bound completion markers for resumable notebook cells."""

    def __init__(self, paths: Phase3Paths, config_sha256: str) -> None:
        self.directory = paths.child("checkpoints")
        self.config_sha256 = config_sha256

    def path_for(self, stage: str) -> Path:
        safe = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in stage
        )
        return self.directory / f"{safe}.json"

    def completed(self, stage: str) -> bool:
        path = self.path_for(stage)
        if not path.is_file():
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (
            payload.get("status") == "complete"
            and payload.get("config_sha256") == self.config_sha256
        )

    def mark(self, stage: str, outputs: Iterable[str | Path]) -> Path:
        output_records = []
        for output in outputs:
            path = Path(output)
            if not path.is_file():
                raise FileNotFoundError(path)
            output_records.append(
                {
                    "path": path.as_posix(),
                    "sha256": sha256_file(path),
                    "size": path.stat().st_size,
                }
            )
        return atomic_write_json(
            self.path_for(stage),
            {
                "schema_version": 1,
                "stage": stage,
                "status": "complete",
                "config_sha256": self.config_sha256,
                "outputs": output_records,
            },
        )


def build_manifest(base: Path, included: Iterable[Path]) -> dict[str, Any]:
    """Build a sorted file manifest and reject files outside the declared base."""
    records = []
    base = base.resolve()
    for path in sorted({Path(item).resolve() for item in included}):
        relative = path.relative_to(base)
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {
                "path": relative.as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"schema_version": 1, "file_count": len(records), "files": records}


def create_research_zip(
    destination: str | Path,
    *,
    base: Path,
    files: Iterable[Path],
    metadata: dict[str, Any],
) -> Path:
    """Create an atomic ZIP with a manifest and without secrets or model weights."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    prohibited_suffixes = {".bin", ".pt", ".pth", ".safetensors", ".env"}
    selected = sorted({Path(path).resolve() for path in files})
    for path in selected:
        if (
            path.suffix.lower() in prohibited_suffixes
            or "token" in path.name.casefold()
        ):
            raise ValueError(f"prohibited export file: {path}")
    manifest = build_manifest(base, selected)
    payload = {**metadata, "manifest": manifest}
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary_directory:
        temporary_root = Path(temporary_directory)
        manifest_path = atomic_write_json(temporary_root / "MANIFEST.json", payload)
        checksums_path = temporary_root / "SHA256SUMS.txt"
        checksums_path.write_text(
            "".join(
                f"{record['sha256']}  {record['path']}\n"
                for record in manifest["files"]
            ),
            encoding="utf-8",
        )
        temporary_zip = temporary_root / destination.name
        with zipfile.ZipFile(
            temporary_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            archive.write(manifest_path, "MANIFEST.json")
            archive.write(checksums_path, "SHA256SUMS.txt")
            for path in selected:
                archive.write(
                    path, path.resolve().relative_to(base.resolve()).as_posix()
                )
        shutil.copy2(temporary_zip, destination)
    return destination
