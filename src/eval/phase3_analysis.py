"""Leakage-aware statistics and publication exports for Phase 3."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_confint

from src.experiments.phase3_common import (
    Phase3Paths,
    atomic_write_json,
    create_research_zip,
    load_phase3_config,
    read_jsonl,
)
from src.utils.provenance import sha256_file

SPECTRAL_FEATURES = (
    "rho_min_over_layers",
    "rho_mean_over_layers",
    "rho_max_over_layers",
    "B_norm_mean_over_layers",
    "C_norm_mean_over_layers",
)


def _load_full_results(config: dict[str, Any], paths: Phase3Paths) -> pd.DataFrame:
    rows = []
    for spec in config["models"]:
        path = paths.child("raw", f"{spec['key']}-full.jsonl")
        if spec.get("required") and not path.is_file():
            raise FileNotFoundError(f"required model result is absent: {path}")
        if path.is_file():
            rows.extend(read_jsonl(path))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no Phase 3 full results found")
    keys = ["prompt_id", "model_key", "condition"]
    if frame.duplicated(keys).any():
        raise ValueError("duplicate model/prompt/condition results")
    return frame


def _wilson(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    low, high = proportion_confint(successes, total, alpha=alpha, method="wilson")
    return float(low), float(high)


def _bootstrap_drop(
    clean: np.ndarray,
    attacked: np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(clean), size=(resamples, len(clean)))
    values = clean[indices].mean(axis=1) - attacked[indices].mean(axis=1)
    return tuple(float(x) for x in np.quantile(values, [0.025, 0.975]))


def _threshold_by_validation(y_true: np.ndarray, probability: np.ndarray, points: int):
    best = (-1.0, 0.5)
    for threshold in np.linspace(0.0, 1.0, points):
        score = f1_score(y_true, probability >= threshold, zero_division=0)
        if score > best[0]:
            best = (float(score), float(threshold))
    return best[1], best[0]


def _spectral_matrix(frame: pd.DataFrame) -> np.ndarray:
    rows = []
    for payload in frame["spectral"]:
        if not isinstance(payload, dict):
            raise ValueError("missing spectrum in Mamba analysis row")
        rows.append([float(payload[name]) for name in SPECTRAL_FEATURES])
    return np.asarray(rows, dtype=float)


def _condition_metrics(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    records = []
    test = frame[frame["split"] == "test"]
    for (model, condition), group in test.groupby(
        ["model_key", "condition"], sort=True
    ):
        successes = int(group["correct"].sum())
        low, high = _wilson(successes, len(group))
        clean = test[(test["model_key"] == model) & (test["condition"] == "clean")][
            ["prompt_id", "correct"]
        ].rename(columns={"correct": "clean_correct"})
        paired = group[["prompt_id", "correct"]].merge(
            clean, on="prompt_id", validate="one_to_one"
        )
        attacked = paired["correct"].astype(float).to_numpy()
        baseline = paired["clean_correct"].astype(float).to_numpy()
        ci_low, ci_high = _bootstrap_drop(
            baseline,
            attacked,
            seed=int(config["seeds"][0]),
            resamples=int(config["protocol"]["bootstrap_resamples"]),
        )
        attack_successes = int((paired["clean_correct"] & ~paired["correct"]).sum())
        records.append(
            {
                "model_key": model,
                "condition": condition,
                "n": len(group),
                "accuracy": successes / len(group),
                "accuracy_wilson_low": low,
                "accuracy_wilson_high": high,
                "paired_accuracy_drop": float(baseline.mean() - attacked.mean()),
                "paired_drop_ci_low": ci_low,
                "paired_drop_ci_high": ci_high,
                "attack_success_rate": attack_successes / len(group),
            }
        )
    return pd.DataFrame(records)


def _fit_detectors(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detector_rows = []
    lexical_rows = []
    positive_conditions = {
        name for name in config["attacks"] if name.startswith("hispa_")
    }
    mamba = frame[frame["architecture"] == "mamba"].copy()
    mamba = mamba[
        mamba["condition"].isin(positive_conditions | {"clean", "benign_control"})
    ]
    mamba["label"] = mamba["condition"].isin(positive_conditions).astype(int)
    for model_key, model_frame in mamba.groupby("model_key", sort=True):
        train = model_frame[model_frame["split"] == "calibration"]
        validation = model_frame[model_frame["split"] == "validation"]
        test = model_frame[model_frame["split"] == "test"]
        for seed in config["seeds"]:
            spectral_model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000, random_state=int(seed), class_weight="balanced"
                ),
            )
            spectral_model.fit(_spectral_matrix(train), train["label"])
            validation_probability = spectral_model.predict_proba(
                _spectral_matrix(validation)
            )[:, 1]
            threshold, validation_f1 = _threshold_by_validation(
                validation["label"].to_numpy(),
                validation_probability,
                int(config["protocol"]["detector_threshold_grid_points"]),
            )
            test_probability = spectral_model.predict_proba(_spectral_matrix(test))[
                :, 1
            ]
            detector_rows.append(
                {
                    "model_key": model_key,
                    "seed": int(seed),
                    "threshold": threshold,
                    "threshold_selected_on": "validation",
                    "validation_f1": validation_f1,
                    "test_f1": f1_score(test["label"], test_probability >= threshold),
                    "test_auc": roc_auc_score(test["label"], test_probability),
                    "n_calibration": len(train),
                    "n_validation": len(validation),
                    "n_test": len(test),
                }
            )
            lexical_model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000, random_state=int(seed), class_weight="balanced"
                ),
            )
            lexical_columns = ["input_tokens", "prompt_nll"]
            lexical_model.fit(train[lexical_columns], train["label"])
            lexical_probability = lexical_model.predict_proba(test[lexical_columns])[
                :, 1
            ]
            lexical_rows.append(
                {
                    "model_key": model_key,
                    "seed": int(seed),
                    "test_auc": roc_auc_score(test["label"], lexical_probability),
                    "auc_ceiling": float(config["protocol"]["lexical_auc_ceiling"]),
                }
            )
    return pd.DataFrame(detector_rows), pd.DataFrame(lexical_rows)


def _transformer_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    diagnostic = frame[
        frame["transformer_diagnostics"].map(lambda value: isinstance(value, dict))
    ]
    for (model, condition), group in diagnostic.groupby(
        ["model_key", "condition"], sort=True
    ):
        entropy = [
            value["attention_entropy_mean"]
            for value in group["transformer_diagnostics"]
        ]
        ranks = [
            value["kv_stable_rank_mean"] for value in group["transformer_diagnostics"]
        ]
        records.append(
            {
                "model_key": model,
                "condition": condition,
                "n": len(group),
                "attention_entropy_mean": float(np.nanmean(entropy)),
                "kv_stable_rank_mean": float(np.nanmean(ranks)),
            }
        )
    return pd.DataFrame(records)


def analyze_phase3() -> dict[str, Any]:
    """Generate all statistics derivable from the frozen reconstruction run."""
    config = load_phase3_config()
    paths = Phase3Paths.create(config)
    frame = _load_full_results(config, paths)
    condition_metrics = _condition_metrics(frame, config)
    detector, lexical = _fit_detectors(frame, config)
    transformer = _transformer_diagnostics(frame)

    outputs = {}
    for name, table in {
        "condition_metrics": condition_metrics,
        "detector_multi_seed": detector,
        "lexical_control": lexical,
        "transformer_diagnostics": transformer,
    }.items():
        path = paths.child("summaries", f"{name}.csv")
        table.to_csv(path, index=False, lineterminator="\n")
        outputs[name] = str(path)

    lexical_pass = (
        bool((lexical["test_auc"] < lexical["auc_ceiling"]).all())
        if not lexical.empty
        else False
    )
    effects = condition_metrics[
        ~condition_metrics["condition"].isin(["clean", "benign_control"])
    ]["paired_accuracy_drop"].abs()
    smallest = float(effects[effects > 0].min()) if (effects > 0).any() else 0.0
    power_n = None
    if 0 < smallest < 1:
        baseline = 0.5
        alternative = min(0.999, baseline + smallest)
        effect_size = 2 * (
            math.asin(math.sqrt(alternative)) - math.asin(math.sqrt(baseline))
        )
        power_n = float(
            NormalIndPower().solve_power(effect_size, power=0.8, alpha=0.05)
        )
    report = {
        "schema_version": 1,
        "raw_rows": len(frame),
        "models": sorted(frame["model_key"].unique()),
        "seeds": config["seeds"],
        "bootstrap_resamples": config["protocol"]["bootstrap_resamples"],
        "threshold_policy": "fit on calibration; select threshold on validation; report once on test",
        "multiple_comparisons": config["protocol"]["multiple_comparisons"],
        "lexical_auc_gate_passed": lexical_pass,
        "smallest_observed_absolute_accuracy_drop": smallest,
        "approximate_n_per_group_for_80pct_power": power_n,
        "scope_warning": (
            "This reconstruction bundle evaluates frozen HiSPA-style interventions. "
            "It does not relabel them as gradient-guided, adaptive-v2/v3, or CLASP results."
        ),
        "outputs": outputs,
    }
    report_path = atomic_write_json(
        paths.child("summaries", "analysis_report.json"), report
    )
    report["report_path"] = str(report_path)
    return report


def export_final_bundle() -> Path:
    """Create the final download ZIP without silently claiming unsupported experiments."""
    config = load_phase3_config()
    paths = Phase3Paths.create(config)
    report_path = paths.child("summaries", "analysis_report.json")
    if not report_path.is_file():
        raise RuntimeError("run analyze_phase3() before final export")
    files = [
        path
        for folder in ("summaries", "figures", "manifests", "logs")
        for path in paths.child(folder).rglob("*")
        if path.is_file()
    ]
    files.extend(
        (
            paths.root / "configs" / "phase3.yaml",
            paths.root / "requirements-lightning.txt",
        )
    )
    destination = paths.root / config["exports"]["final_zip"]
    return create_research_zip(
        destination,
        base=paths.root,
        files=files,
        metadata={
            "bundle": "phase3_final",
            "config_sha256": sha256_file(paths.root / "configs" / "phase3.yaml"),
        },
    )
