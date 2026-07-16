"""Deterministic seeding and immutable run-manifest helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
TRACKED_PACKAGES = (
    "accelerate",
    "numpy",
    "pandas",
    "PyYAML",
    "safetensors",
    "scikit-learn",
    "torch",
    "transformers",
)


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch using one recorded integer."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)


def package_versions() -> dict[str, str]:
    """Return exact installed versions for the experiment dependency set."""
    return {name: importlib.metadata.version(name) for name in TRACKED_PACKAGES}


def sha256_file(path: str | Path) -> str:
    """Hash an existing file without loading it entirely into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_run_manifest(
    output: str | Path,
    *,
    experiment: str,
    seed: int,
    config_path: str | Path,
    model_id: str,
    model_revision: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write the mandatory provenance record for one experiment run."""
    config = Path(config_path)
    if not config.is_file():
        raise FileNotFoundError(config)
    if not REVISION_RE.fullmatch(model_revision):
        raise ValueError("model_revision must be an immutable 40-character SHA")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "seed": seed,
        "config_path": config.as_posix(),
        "config_sha256": sha256_file(config),
        "model_id": model_id,
        "model_revision": model_revision,
        "package_versions": package_versions(),
        "extra": extra or {},
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return destination
