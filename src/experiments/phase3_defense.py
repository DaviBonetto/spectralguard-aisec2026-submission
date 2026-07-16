"""SpectralGuard and CLASP-style common-protocol detector fitting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def spectral_feature_names(frame: pd.DataFrame) -> list[str]:
    first = next(value for value in frame["spectral"] if isinstance(value, dict))
    return [f"layer_{int(layer['layer'])}_rho_mean" for layer in first["layers"]]


def spectral_matrix(frame: pd.DataFrame) -> np.ndarray:
    rows = []
    expected_layers = None
    for payload in frame["spectral"]:
        if not isinstance(payload, dict):
            raise ValueError("missing spectral payload")
        values = [float(layer["rho_mean"]) for layer in payload["layers"]]
        expected_layers = expected_layers or len(values)
        if len(values) != expected_layers:
            raise ValueError("inconsistent layer count within detector frame")
        rows.append(values)
    return np.asarray(rows, dtype=float)


def boe_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    payloads = list(frame["boe_features"])
    if not payloads or not all(isinstance(value, dict) for value in payloads):
        raise ValueError("missing CLASP-style BOE payload")
    names = sorted(payloads[0])
    if any(sorted(payload) != names for payload in payloads):
        raise ValueError("inconsistent BOE feature schema")
    return np.asarray(
        [[float(payload[name]) for name in names] for payload in payloads]
    ), names


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray, points: int):
    best = (-1.0, 0.5)
    for threshold in np.linspace(0.0, 1.0, int(points)):
        score = f1_score(y_true, probabilities >= threshold, zero_division=0)
        if score > best[0]:
            best = (float(score), float(threshold))
    return best[1], best[0]


def fit_spectral_detector(
    frame: pd.DataFrame,
    *,
    seed: int,
    threshold_points: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit on calibration, choose threshold on validation, predict test once."""
    required = {"calibration", "validation", "test"}
    if set(frame["split"]) != required:
        raise ValueError("detector frame must contain all three frozen splits")
    train = frame[frame["split"] == "calibration"]
    validation = frame[frame["split"] == "validation"]
    test = frame[frame["split"] == "test"]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000, random_state=int(seed), class_weight="balanced"
        ),
    )
    model.fit(spectral_matrix(train), train["label"])
    validation_probability = model.predict_proba(spectral_matrix(validation))[:, 1]
    threshold, validation_f1 = select_threshold(
        validation["label"].to_numpy(), validation_probability, threshold_points
    )
    test_probability = model.predict_proba(spectral_matrix(test))[:, 1]
    scaler = model.named_steps["standardscaler"]
    logistic = model.named_steps["logisticregression"]
    payload = {
        "schema_version": 1,
        "detector": "spectralguard_multilayer_logistic",
        "seed": int(seed),
        "feature_names": spectral_feature_names(frame),
        "feature_mean": scaler.mean_.tolist(),
        "feature_scale": scaler.scale_.tolist(),
        "weights": logistic.coef_[0].tolist(),
        "bias": float(logistic.intercept_[0]),
        "threshold": threshold,
        "threshold_selected_on": "validation",
        "validation_f1": validation_f1,
        "test_f1": float(f1_score(test["label"], test_probability >= threshold)),
        "test_auc": float(roc_auc_score(test["label"], test_probability)),
        "n_calibration": len(train),
        "n_validation": len(validation),
        "n_test": len(test),
    }
    predictions = test[["prompt_id", "condition", "task", "label"]].copy()
    predictions["probability"] = test_probability
    predictions["prediction"] = test_probability >= threshold
    predictions["seed"] = int(seed)
    return payload, predictions


def fit_clasp_style_detector(
    frame: pd.DataFrame,
    *,
    seed: int,
    threshold_points: int,
    xgboost_params: dict[str, Any],
    device: str = "cuda",
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Reimplement CLASP's BOE+XGBoost design under the same frozen splits."""
    from xgboost import XGBClassifier

    train = frame[frame["split"] == "calibration"]
    validation = frame[frame["split"] == "validation"]
    test = frame[frame["split"] == "test"]
    x_train, names = boe_matrix(train)
    x_validation, validation_names = boe_matrix(validation)
    x_test, test_names = boe_matrix(test)
    if names != validation_names or names != test_names:
        raise ValueError("CLASP BOE schema drifted across splits")
    allowed = {
        key: value for key, value in xgboost_params.items() if key not in {"verbosity"}
    }
    allowed.update({"random_state": int(seed), "n_jobs": -1, "device": device})
    model = XGBClassifier(**allowed)
    model.fit(
        x_train,
        train["label"],
        eval_set=[(x_validation, validation["label"])],
        verbose=False,
    )
    validation_probability = model.predict_proba(x_validation)[:, 1]
    threshold, validation_f1 = select_threshold(
        validation["label"].to_numpy(), validation_probability, threshold_points
    )
    test_probability = model.predict_proba(x_test)[:, 1]
    payload = {
        "schema_version": 1,
        "detector": "clasp_style_boe_xgboost_reimplementation",
        "seed": int(seed),
        "official_design_elements": [
            "block_output_embeddings",
            "block_statistics",
            "xgboost",
        ],
        "not_official_pretrained_weights": True,
        "feature_count": len(names),
        "threshold": threshold,
        "threshold_selected_on": "validation",
        "validation_f1": validation_f1,
        "test_f1": float(f1_score(test["label"], test_probability >= threshold)),
        "test_auc": float(roc_auc_score(test["label"], test_probability)),
        "n_calibration": len(train),
        "n_validation": len(validation),
        "n_test": len(test),
    }
    predictions = test[["prompt_id", "condition", "task", "label"]].copy()
    predictions["probability"] = test_probability
    predictions["prediction"] = test_probability >= threshold
    predictions["seed"] = int(seed)
    return payload, predictions
