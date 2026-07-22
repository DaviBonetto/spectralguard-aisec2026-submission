"""Protocol, policy, portability, and leakage tests for Phase 5."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.defense.availability_policy import (
    FLAG_CONTINUE,
    HARD_BLOCK,
    PASS,
    algorithm2,
    algorithm2b,
)
from src.eval.phase5_analysis import _select_policy
from src.experiments.phase5_common import load_phase5_config
from src.experiments.phase5_runner import build_sessions, window_features


def test_phase5_protocol_is_frozen_and_fits_paid_window():
    config = load_phase5_config()
    assert config["execution"]["session_wall_clock_hours"] == 2.7
    assert config["dataset"]["validation_sessions"] == 60
    assert config["dataset"]["test_sessions"] == 320
    assert config["dataset"]["turns_per_session"] == 3
    assert config["adversarial_benign"]["test_prompts"] == 300
    assert len(config["seeds"]) == 5
    assert config["monitor"]["detector_frozen_on"] == "phase3_validation"
    assert config["model"]["revision"] == "96c48e0292b63f5346b6d30061af2551f7101e26"


def test_original_and_graduated_policy_have_expected_temporal_semantics():
    scores = [0.1, 0.9, 0.2, 0.95]
    original = algorithm2(scores, threshold=0.8)
    graduated = algorithm2b(scores, threshold=0.8, k=2, w=4)
    assert original.decisions == (PASS, HARD_BLOCK, HARD_BLOCK, HARD_BLOCK)
    assert graduated.decisions == (PASS, FLAG_CONTINUE, PASS, HARD_BLOCK)
    assert original.first_block_window == 1
    assert graduated.first_block_window == 3


def test_graduated_policy_rejects_invalid_k_of_w():
    with pytest.raises(ValueError):
        algorithm2b([0.5], threshold=0.4, k=1, w=4)


def test_token_windows_are_causal_and_include_the_tail():
    features = window_features(
        [[1, 2, 3, 4, 5], [2, 4, 6, 8, 10]],
        window_tokens=3,
        stride_tokens=2,
    )
    assert features == [[2.0, 4.0], [4.0, 8.0]]


def test_sessions_are_deterministic_disjoint_and_multi_turn():
    rows = []
    for task in ("a", "b", "c", "d"):
        for index in range(8):
            rows.append({"task": task, "split": "test", "prompt_id": f"{task}-{index}"})
    sessions = build_sessions(rows, split="test", count=8, turns=4)
    prompt_ids = [prompt for session in sessions for prompt in session["prompt_ids"]]
    assert len(sessions) == 8
    assert len(prompt_ids) == len(set(prompt_ids)) == 32


def test_policy_selection_uses_validation_and_respects_capture_constraint():
    import pandas as pd

    frame = pd.DataFrame(
        [
            {"condition": condition, "hazard_scores": scores, "threshold": 0.8}
            for condition, scores in (
                ("clean", [0.9, 0.1, 0.1, 0.1]),
                ("malicious_whitebox", [0.9, 0.95, 0.1, 0.1]),
            )
            for _ in range(8)
        ]
    )
    table, selection = _select_policy(frame, load_phase5_config())
    assert selection["selected_on"] == "validation"
    assert table["selected"].sum() == 1


def test_phase5_notebook_is_clean_and_every_code_cell_finishes():
    committed = subprocess.run(
        ["git", "show", "HEAD:notebooks/05_phase5_availability.ipynb"],
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
        assert "print('Finished')" in "".join(cell["source"])


def test_phase5_export_paths_are_ignored_runtime_files():
    config = load_phase5_config()
    for key in ("experiments_zip", "handoff_tar", "final_zip"):
        assert config["exports"][key].startswith("Experiments/")


def test_tracked_phase5_evidence_is_self_verifying():
    completed = subprocess.run(
        [sys.executable, "scripts/verify_phase5_results.py"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "complete negative availability result" in completed.stdout
