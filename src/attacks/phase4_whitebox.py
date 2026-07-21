"""Joint white-box objective checks and frozen detector loading."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def load_frozen_spectralguard(path: str | Path, *, seed: int) -> dict[str, Any]:
    frame = pd.read_csv(path)
    row = frame[frame["seed"].astype(int) == int(seed)]
    if len(row) != 1:
        raise ValueError(f"expected one frozen detector row for seed {seed}")
    record = row.iloc[0]
    if record["threshold_selected_on"] != "validation":
        raise ValueError("detector threshold was not frozen on validation")
    return {
        "seed": int(seed),
        "feature_names": ast.literal_eval(record["feature_names"]),
        "feature_mean": ast.literal_eval(record["feature_mean"]),
        "feature_scale": ast.literal_eval(record["feature_scale"]),
        "weights": ast.literal_eval(record["weights"]),
        "bias": float(record["bias"]),
        "threshold": float(record["threshold"]),
        "threshold_selected_on": "validation",
    }


def differentiable_hazard(features: torch.Tensor, detector: dict[str, Any]) -> torch.Tensor:
    mean = features.new_tensor(detector["feature_mean"])
    scale = features.new_tensor(detector["feature_scale"]).clamp_min(1e-12)
    weights = features.new_tensor(detector["weights"])
    bias = features.new_tensor(float(detector["bias"]))
    if features.shape[-1] != weights.numel():
        raise ValueError("white-box feature count does not match frozen detector")
    return torch.sigmoid((((features - mean) / scale) * weights).sum() + bias)


def prove_joint_gradient(
    task_loss: torch.Tensor,
    spectral_features: torch.Tensor,
    detector: dict[str, Any],
    *,
    rho_weight: float,
    hazard_weight: float,
) -> dict[str, float]:
    """Prove both base-task and frozen-detector paths reach the attack variable."""
    hazard = differentiable_hazard(spectral_features, detector)
    task_grad = torch.autograd.grad(task_loss, spectral_features, retain_graph=True)[0]
    hazard_grad = torch.autograd.grad(hazard, spectral_features, retain_graph=True)[0]
    joint = -task_loss + float(rho_weight) * spectral_features.mean() + float(
        hazard_weight
    ) * hazard
    joint_grad = torch.autograd.grad(joint, spectral_features)[0]
    norms = {
        "task_gradient_norm": float(task_grad.norm().detach()),
        "detector_gradient_norm": float(hazard_grad.norm().detach()),
        "joint_gradient_norm": float(joint_grad.norm().detach()),
    }
    if not all(value > 0 for value in norms.values()):
        raise RuntimeError(f"joint white-box gradient is disconnected: {norms}")
    return norms
