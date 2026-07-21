"""Validated configuration, paths, checkpoints, and preflight for Phase 5."""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from src.experiments.phase3_common import atomic_write_json
from src.utils.provenance import sha256_file


def repository_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    if not (root / "configs" / "phase5.yaml").is_file():
        raise RuntimeError("Phase 5 repository root is unavailable")
    return root


def load_phase5_config(path: str | Path = "configs/phase5.yaml") -> dict[str, Any]:
    """Load the frozen Phase 5 protocol and reject unsafe drift."""
    source = Path(path)
    if not source.is_absolute():
        source = repository_root() / source
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("phase") != 5:
        raise ValueError("not a Phase 5 schema-v1 configuration")
    execution = config["execution"]
    if not 0 < float(execution["session_wall_clock_hours"]) <= 2.7:
        raise ValueError("Phase 5 paid session must not exceed 2.7 hours")
    if not 0 < float(execution["export_reserve_minutes"]) < 15:
        raise ValueError("Phase 5 export reserve must be between zero and 15 minutes")
    if len(config["seeds"]) != 5 or len(set(config["seeds"])) != 5:
        raise ValueError("Phase 5 requires five distinct attack seeds")
    dataset = config["dataset"]
    if int(dataset["test_sessions"]) < 300:
        raise ValueError("Phase 5 requires at least 300 held-out benign sessions")
    if int(dataset["turns_per_session"]) < 2:
        raise ValueError("Phase 5 sessions must contain multiple benign turns")
    policy = config["graduated_policy"]
    if min(policy["k_grid"]) <= 1 or max(policy["k_grid"]) > max(policy["w_grid"]):
        raise ValueError("invalid Phase 5 k-of-w grid")
    attack = config["adversarial_benign"]
    if int(attack["test_prompts"]) < 300 or int(attack["restarts"]) < 5:
        raise ValueError("Phase 5 red-team is underspecified")
    if len(str(config["model"]["revision"])) != 40:
        raise ValueError("Phase 5 model revision must be an immutable SHA")
    return config


@dataclass(frozen=True)
class Phase5Paths:
    """Resolved Phase 5 runtime paths constrained to the repository."""

    root: Path
    runtime: Path

    @classmethod
    def create(cls, config: dict[str, Any]) -> "Phase5Paths":
        root = repository_root().resolve()
        runtime = (root / config["exports"]["runtime_root"]).resolve()
        runtime.relative_to(root)
        for folder in (
            "checkpoints",
            "raw",
            "attacks",
            "manifests",
            "summaries",
            "figures",
            "logs",
        ):
            (runtime / folder).mkdir(parents=True, exist_ok=True)
        (root / "Experiments" / "exports").mkdir(parents=True, exist_ok=True)
        return cls(root, runtime)

    def child(self, *parts: str) -> Path:
        target = self.runtime.joinpath(*parts).resolve()
        target.relative_to(self.runtime)
        return target


def protocol_git_status() -> str:
    completed = subprocess.run(
        [
            "git",
            "status",
            "--short",
            "--untracked-files=no",
            "--",
            ".",
            ":(exclude)notebooks/*.ipynb",
        ],
        cwd=repository_root(),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def phase5_preflight(*, require_h100: bool) -> dict[str, Any]:
    """Fail before paid work if the frozen environment or evidence drifted."""
    config = load_phase5_config()
    paths = Phase5Paths.create(config)
    dirty = protocol_git_status()
    if dirty:
        raise RuntimeError(f"tracked Phase 5 protocol is dirty:\n{dirty}")
    detector = paths.root / config["monitor"]["detector_source"]
    if not detector.is_file():
        raise FileNotFoundError(f"frozen detector is missing: {detector}")
    malicious_attack = paths.root / config["monitor"]["malicious_attack_source"]
    if not malicious_attack.is_file():
        raise FileNotFoundError(
            f"frozen malicious attack is missing: {malicious_attack}"
        )
    cuda = torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if cuda else None
    memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if cuda else 0
    free_disk = shutil.disk_usage(paths.root).free / 1024**3
    if require_h100 and (
        not cuda
        or "H100" not in str(gpu).upper()
        or memory < float(config["execution"]["minimum_gpu_memory_gib"])
    ):
        raise RuntimeError(f"Phase 5 full run requires an >=75 GiB H100; found {gpu}")
    if require_h100:
        expected_python = tuple(
            int(value) for value in str(config["execution"]["required_python"]).split(".")
        )
        if sys.version_info[:2] != expected_python:
            raise RuntimeError(
                f"Phase 5 requires Python {expected_python}; found {sys.version_info[:2]}"
            )
        if not torch.__version__.startswith(config["execution"]["required_torch_prefix"]):
            raise RuntimeError(f"Phase 5 requires Torch 2.8.x; found {torch.__version__}")
        if not str(torch.version.cuda).startswith(
            config["execution"]["required_cuda_major"] + "."
        ):
            raise RuntimeError(f"Phase 5 requires CUDA 12.x; found {torch.version.cuda}")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Phase 5 requires H100 BF16 support")
        expected_kernels = {
            "causal-conv1d": "1.6.2.post1",
            "mamba-ssm": "2.3.2.post1",
        }
        observed = {name: importlib.metadata.version(name) for name in expected_kernels}
        if observed != expected_kernels:
            raise RuntimeError(f"frozen CUDA kernel versions drifted: {observed}")
        if free_disk < float(config["execution"]["minimum_free_disk_gib"]):
            raise RuntimeError(
                f"Phase 5 requires {config['execution']['minimum_free_disk_gib']} GiB "
                f"free; found {free_disk:.1f} GiB"
            )
    report = {
        "schema_version": 1,
        "status": "PASS",
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=paths.root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "config_sha256": sha256_file(paths.root / "configs" / "phase5.yaml"),
        "detector_sha256": sha256_file(detector),
        "malicious_attack_sha256": sha256_file(malicious_attack),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu,
        "gpu_memory_gib": memory,
        "bf16": bool(cuda and torch.cuda.is_bf16_supported()),
        "free_disk_gib": free_disk,
    }
    atomic_write_json(paths.child("manifests", "preflight.json"), report)
    return report


def checkpoint_complete(paths: Phase5Paths, stage: str) -> bool:
    path = paths.child("checkpoints", f"{stage}.json")
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("config_sha256") == sha256_file(
        paths.root / "configs" / "phase5.yaml"
    ) and all(Path(item["path"]).is_file() for item in payload.get("outputs", []))


def mark_checkpoint(paths: Phase5Paths, stage: str, outputs: list[Path]) -> None:
    atomic_write_json(
        paths.child("checkpoints", f"{stage}.json"),
        {
            "schema_version": 1,
            "stage": stage,
            "config_sha256": sha256_file(paths.root / "configs" / "phase5.yaml"),
            "outputs": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in outputs
            ],
        },
    )
