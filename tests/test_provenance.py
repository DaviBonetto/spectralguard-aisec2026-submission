"""Provenance and public-tree hygiene tests."""

import json
import random
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils.provenance import seed_everything, write_run_manifest


def test_public_tree_contains_no_local_or_placeholder_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    tracked = [path for path in tracked if (root / path).is_file()]
    assert tracked
    assert not any(path.endswith(".gitkeep") or " " in path for path in tracked)
    blocked = (
        re.compile(r"[A-Za-z]:[\\/]+Users[\\/]", re.I),
        re.compile(r"/(?:home|Users)/[^/\s]+/", re.I),
        re.compile(r"@(?:gmail|outlook|hotmail|yahoo)[.]", re.I),
    )
    for relative in tracked:
        content = (root / relative).read_bytes().decode("utf-8", errors="ignore")
        assert not any(pattern.search(content) for pattern in blocked), relative


def test_seed_everything_is_repeatable() -> None:
    seed_everything(42)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    seed_everything(42)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert first == second


def test_manifest_requires_revision_sha(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("seed: 42\n", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable"):
        write_run_manifest(
            tmp_path / "run.json",
            experiment="test",
            seed=42,
            config_path=config,
            model_id="example/model",
            model_revision="main",
        )


def test_manifest_records_required_provenance(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("seed: 42\n", encoding="utf-8")
    destination = write_run_manifest(
        tmp_path / "run.json",
        experiment="test",
        seed=42,
        config_path=config,
        model_id="example/model",
        model_revision="a" * 40,
    )
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["seed"] == 42
    assert payload["config_sha256"]
    assert payload["model_revision"] == "a" * 40
    assert payload["package_versions"]["torch"]
