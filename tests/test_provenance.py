import json
import random

import numpy as np
import pytest
import torch

from src.utils.provenance import seed_everything, write_run_manifest


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
