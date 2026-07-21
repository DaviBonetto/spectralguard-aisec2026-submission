"""Protocol, attack-isolation, and evidence tests for Phase 4."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch

from src.eval.phase4_analysis import _metric_bootstrap

from src.attacks.phase4_blackbox import PublicResponse, query_only_search
from src.attacks.phase4_whitebox import differentiable_hazard, prove_joint_gradient
from src.experiments.phase4_common import load_phase4_config


class TrapOracle:
    def __init__(self) -> None:
        self._queries = 0

    @property
    def queries_used(self) -> int:
        return self._queries

    def __getattr__(self, name: str):
        if name in {"model", "gradient", "rho", "hazard", "hidden_states", "logits"}:
            raise AssertionError(f"black-box attacker touched forbidden target field {name}")
        raise AttributeError(name)

    def query(self, rows, trigger):
        self._queries += len(rows)
        score = 0.0 if trigger == "best" else 1.0
        return [PublicResponse(score, score > 0, "public") for _ in rows]


def test_phase4_protocol_is_frozen_and_fits_paid_window():
    config = load_phase4_config()
    assert config["dataset"]["frozen_test_examples"] == 300
    assert config["execution"]["session_wall_clock_hours"] == 2.4
    assert config["white_box"]["detector_frozen_on"] == "phase3_validation"
    assert config["models"]["target"]["revision"] == "96c48e0292b63f5346b6d30061af2551f7101e26"


def test_query_only_search_uses_only_public_oracle_and_respects_budget():
    oracle = TrapOracle()
    result = query_only_search(
        oracle,
        [{"prompt_id": "a"}, {"prompt_id": "b"}],
        ["weak", "best", "unused"],
        query_budget=4,
    )
    assert result["selected_trigger"] == "best"
    assert result["queries_used"] == 4
    assert result["target_information_used"] == [
        "task_score",
        "correct",
        "response_excerpt",
    ]


def test_joint_whitebox_objective_has_task_and_detector_gradients():
    features = torch.tensor([0.4, 0.6], requires_grad=True)
    task_loss = (features.square()).sum()
    detector = {
        "feature_mean": [0.5, 0.5],
        "feature_scale": [0.1, 0.2],
        "weights": [1.0, -0.5],
        "bias": 0.2,
    }
    hazard = differentiable_hazard(features, detector)
    assert hazard.requires_grad
    proof = prove_joint_gradient(
        task_loss, features, detector, rho_weight=0.5, hazard_weight=1.0
    )
    assert min(proof.values()) > 0


def test_phase4_notebooks_are_clean_and_every_code_cell_finishes():
    for name in ("03_phase4_attacks.ipynb", "04_phase4_analysis.ipynb"):
        committed = subprocess.run(
            ["git", "show", f"HEAD:notebooks/{name}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        notebook = json.loads(committed)
        assert notebook["nbformat"] == 4
        code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert code
        for cell in code:
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            source = "".join(cell["source"])
            assert "print('Finished')" in source or 'print("Finished")' in source


def test_phase4_export_paths_are_ignored_runtime_files():
    config = load_phase4_config()
    for key in ("experiments_zip", "handoff_tar", "final_zip"):
        assert config["exports"][key].startswith("Experiments/")


def test_cuda_smoke_cannot_stop_early_on_default_eos():
    source = Path("scripts/install_lightning_mamba_kernels.py").read_text(
        encoding="utf-8"
    )
    assert "attention_mask=attention_mask" in source
    assert "eos_token_id=None" in source


def test_differentiable_spectrum_uses_memory_bounded_exact_reduction():
    delta = torch.rand(2, 5, 7, requires_grad=True)
    decay = torch.rand(7, 3, requires_grad=True)
    original = torch.exp(-delta[..., None] * decay[None, None, ...]).amax(
        dim=(-1, -2)
    )
    minimum = (delta * decay.amin(dim=-1)[None, None, :]).amin(dim=-1)
    bounded = torch.exp(-minimum)
    assert torch.allclose(original, bounded)


def test_phase4_metric_bootstrap_is_finite_and_contains_point_estimate():
    labels = torch.tensor([0, 0, 0, 1, 1, 1]).numpy()
    prediction = torch.tensor([0, 0, 1, 1, 1, 0]).numpy()
    low, high = _metric_bootstrap(
        labels,
        prediction,
        metric="f1",
        seed=42,
        resamples=500,
    )
    point = 2 * 2 / (2 * 2 + 1 + 1)
    assert 0 <= low <= point <= high <= 1
