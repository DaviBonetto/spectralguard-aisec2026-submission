"""Leakage-aware statistics and publication exports for Phase 3."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2, ttest_rel, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestPower
from statsmodels.stats.proportion import proportion_confint

from src.experiments.phase3_common import (
    Phase3Paths,
    atomic_write_json,
    create_research_zip,
    load_phase3_config,
    read_jsonl,
)
from src.experiments.phase3_defense import (
    fit_clasp_style_detector,
    fit_spectral_detector,
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
    for experiment in config["experiments"]:
        path = paths.child("raw", f"{experiment['id']}.jsonl")
        if not path.is_file():
            raise FileNotFoundError(f"required experiment result is absent: {path}")
        rows.extend(read_jsonl(path))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no Phase 3 full results found")
    keys = ["prompt_id", "model_key", "condition"]
    if frame.duplicated(keys).any():
        # The confirmatory model deliberately appears in main and transfer runs.
        frame = frame.drop_duplicates(keys, keep="first").reset_index(drop=True)
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
        "hispa_original",
        "gradient_guided",
        "adaptive_v2",
        "adaptive_v3",
    }
    labeled = frame[
        frame["condition"].isin(positive_conditions | {"clean", "benign_control"})
    ].copy()
    labeled["label"] = labeled["condition"].isin(positive_conditions).astype(int)
    for model_key, model_frame in labeled.groupby("model_key", sort=True):
        if set(model_frame["split"]) != {"calibration", "validation", "test"}:
            continue
        train = model_frame[model_frame["split"] == "calibration"]
        validation = model_frame[model_frame["split"] == "validation"]
        test = model_frame[model_frame["split"] == "test"]
        for seed in config["seeds"]:
            if model_frame["architecture"].iloc[0] == "mamba":
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
                        "test_f1": f1_score(
                            test["label"], test_probability >= threshold
                        ),
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


def _save(paths: Phase3Paths, name: str, table: pd.DataFrame) -> str:
    if table.empty:
        raise RuntimeError(f"required artifact {name} would be empty")
    path = paths.child("summaries", name)
    table.to_csv(path, index=False, lineterminator="\n")
    return str(path)


def _main_detector_frame(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[
        (frame["model_key"] == "mamba_2_8b")
        & frame["condition"].isin(
            {"clean", "benign_control", "hispa_original", "gradient_guided"}
        )
    ].copy()
    selected["label"] = (
        ~selected["condition"].isin({"clean", "benign_control"})
    ).astype(int)
    if set(selected["split"]) != {"calibration", "validation", "test"}:
        raise ValueError("main detector frame lacks calibration, validation, or test")
    return selected


def _official_xgboost_params(paths: Phase3Paths) -> dict[str, Any]:
    source = paths.child(
        "datasets", "official_clasp", "archive", "xgboost_hyperparam.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))

    def find(value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict) and any(
            key in value for key in ("n_estimators", "max_depth", "learning_rate")
        ):
            return value
        children = (
            value.values()
            if isinstance(value, dict)
            else value
            if isinstance(value, list)
            else []
        )
        for child in children:
            result = find(child)
            if result is not None:
                return result
        return None

    result = find(payload)
    if result is None:
        raise ValueError("official CLASP snapshot has no XGBoost parameter map")
    return result


def analyze_phase3() -> dict[str, Any]:
    """Generate real, schema-checked evidence for every Phase 3 requirement."""
    config = load_phase3_config()
    paths = Phase3Paths.create(config)
    frame = _load_full_results(config, paths)
    condition_metrics = _condition_metrics(frame, config)
    _, lexical = _fit_detectors(frame, config)
    transformer = _transformer_diagnostics(frame)
    outputs: dict[str, str] = {}

    outputs["gpt2_diagnostics.csv"] = _save(
        paths,
        "gpt2_diagnostics.csv",
        transformer[transformer["model_key"] == "gpt2_xl"],
    )
    outputs["lexical_control.csv"] = _save(paths, "lexical_control.csv", lexical)

    main = _main_detector_frame(frame)
    spectral_rows, spectral_predictions = [], []
    clasp_rows, clasp_predictions = [], []
    official_params = _official_xgboost_params(paths)
    for seed in config["seeds"]:
        payload, prediction = fit_spectral_detector(
            main,
            seed=int(seed),
            threshold_points=int(config["protocol"]["detector_threshold_grid_points"]),
        )
        spectral_rows.append(payload)
        spectral_predictions.append(prediction)
        clasp_payload, clasp_prediction = fit_clasp_style_detector(
            main,
            seed=int(seed),
            threshold_points=int(config["protocol"]["detector_threshold_grid_points"]),
            xgboost_params=official_params,
            device="cpu",
        )
        clasp_rows.append(clasp_payload)
        clasp_predictions.append(clasp_prediction)
    spectral = pd.DataFrame(spectral_rows)
    predictions = pd.concat(spectral_predictions, ignore_index=True)
    clasp = pd.DataFrame(clasp_rows)
    clasp_test = pd.concat(clasp_predictions, ignore_index=True)

    def classifier_row(
        name: str, prediction: pd.DataFrame, auc: float
    ) -> dict[str, Any]:
        precision, recall, f1, _ = precision_recall_fscore_support(
            prediction["label"],
            prediction["prediction"],
            average="binary",
            zero_division=0,
        )
        return {
            "detector": name,
            "n": len(prediction),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc_mean": auc,
        }

    table10 = pd.DataFrame(
        [
            classifier_row(
                "SpectralGuard", predictions, float(spectral["test_auc"].mean())
            ),
            classifier_row(
                "CLASP-style BOE+XGBoost reimplementation",
                clasp_test,
                float(clasp["test_auc"].mean()),
            ),
        ]
    )
    outputs["table10_unified.csv"] = _save(paths, "table10_unified.csv", table10)
    provenance = spectral[
        [
            "seed",
            "threshold",
            "threshold_selected_on",
            "n_calibration",
            "n_validation",
            "n_test",
        ]
    ]
    outputs["threshold_provenance.csv"] = _save(
        paths, "threshold_provenance.csv", provenance
    )

    clean = main[(main["split"] == "test") & (main["condition"] == "clean")][
        ["prompt_id", "correct"]
    ].rename(columns={"correct": "clean_correct"})
    attacked = main[
        (main["split"] == "test")
        & main["condition"].isin({"hispa_original", "gradient_guided"})
    ][["prompt_id", "condition", "correct"]].merge(
        clean, on="prompt_id", validate="many_to_one"
    )
    attacked["attack_success"] = attacked["clean_correct"] & ~attacked["correct"]
    strata = predictions.merge(
        attacked[["prompt_id", "condition", "attack_success"]],
        on=["prompt_id", "condition"],
    )
    stratified = (
        strata.groupby(["seed", "attack_success"])
        .apply(
            lambda group: pd.Series(
                {
                    "n": len(group),
                    "f1": f1_score(
                        group["label"], group["prediction"], zero_division=0
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    outputs["attack_success_stratified_f1.csv"] = _save(
        paths, "attack_success_stratified_f1.csv", stratified
    )

    adaptive = frame[(frame["model_key"] == "mamba_790m") & (frame["split"] == "test")]
    v2 = adaptive[adaptive["condition"] == "adaptive_v2"][
        ["prompt_id", "correct", "task_score"]
    ].rename(columns={"correct": "v2_correct", "task_score": "v2_task_score"})
    v3 = adaptive[adaptive["condition"] == "adaptive_v3"][
        ["prompt_id", "correct", "task_score"]
    ].rename(columns={"correct": "v3_correct", "task_score": "v3_task_score"})
    paired = v2.merge(v3, on="prompt_id", validate="one_to_one")
    paired["v2_only_correct"] = paired["v2_correct"] & ~paired["v3_correct"]
    paired["v3_only_correct"] = paired["v3_correct"] & ~paired["v2_correct"]
    outputs["adaptive_v2_v3_paired.csv"] = _save(
        paths, "adaptive_v2_v3_paired.csv", paired
    )
    b, c = int(paired["v2_only_correct"].sum()), int(paired["v3_only_correct"].sum())
    statistic = (abs(b - c) - 1) ** 2 / (b + c) if b + c else 0.0
    mcnemar = pd.DataFrame(
        [
            {
                "comparison": "adaptive_v2_vs_v3_correctness",
                "b": b,
                "c": c,
                "statistic": statistic,
                "p_value": float(chi2.sf(statistic, 1)) if b + c else 1.0,
            }
        ]
    )
    outputs["mcnemar_tests.csv"] = _save(paths, "mcnemar_tests.csv", mcnemar)
    outputs["table10_bootstrap_ci.csv"] = _save(
        paths, "table10_bootstrap_ci.csv", condition_metrics
    )

    regressions = []
    for condition, group in main[main["split"] == "test"].groupby("condition"):
        x = (
            group["spectral"]
            .map(lambda value: value["rho_mean_over_layers"])
            .to_numpy(float)
        )
        y = group["correct"].to_numpy(float)
        slope, intercept = np.polyfit(x, y, 1)
        logit_slope, logit_p = math.nan, math.nan
        if np.unique(y).size == 2:
            try:
                fitted = sm.Logit(y, sm.add_constant(x)).fit(disp=False)
                logit_slope, logit_p = float(fitted.params[1]), float(fitted.pvalues[1])
            except (ValueError, np.linalg.LinAlgError):
                pass
        regressions.append(
            {
                "condition": condition,
                "n": len(group),
                "linear_slope": slope,
                "linear_intercept": intercept,
                "pearson_r": float(np.corrcoef(x, y)[0, 1]) if np.std(y) else 0.0,
                "logit_slope": logit_slope,
                "logit_p_value": logit_p,
            }
        )
    outputs["rho_accuracy_regression.csv"] = _save(
        paths, "rho_accuracy_regression.csv", pd.DataFrame(regressions)
    )
    transfer_n = int(config["protocol"]["transfer_test_examples"])
    per_task = transfer_n // len(config["dataset"]["task_splits"])
    transfer_ids = set(
        frame[(frame["model_key"] == "mamba_2_8b") & (frame["split"] == "test")]
        .drop_duplicates("prompt_id")
        .groupby("task", group_keys=False)
        .head(per_task)["prompt_id"]
    )
    transfer_frame = frame[
        frame["model_key"].isin({"mamba_1_4b", "mamba_2_8b", "pythia_1_4b"})
        & frame["prompt_id"].isin(transfer_ids)
    ]
    transfer_metrics = _condition_metrics(transfer_frame, config)
    outputs["cross_model_transfer_wilson.csv"] = _save(
        paths,
        "cross_model_transfer_wilson.csv",
        transfer_metrics,
    )
    outputs["hispa_head_to_head.csv"] = _save(
        paths,
        "hispa_head_to_head.csv",
        condition_metrics[
            condition_metrics["condition"].isin({"hispa_original", "gradient_guided"})
        ],
    )
    controls = (
        frame[frame["split"] == "test"]
        .groupby(["task", "condition"], as_index=False)
        .agg(
            n=("prompt_id", "size"),
            accuracy=("correct", "mean"),
            task_score=("task_score", "mean"),
        )
    )
    outputs["four_task_controls.csv"] = _save(paths, "four_task_controls.csv", controls)
    matches = [
        {
            "experiment": path.stem.removeprefix("control_match_"),
            **json.loads(path.read_text(encoding="utf-8")),
        }
        for path in paths.child("manifests").glob("control_match_*.json")
    ]
    outputs["control_matching_audit.csv"] = _save(
        paths, "control_matching_audit.csv", pd.DataFrame(matches)
    )
    outputs["headline_multiseed.csv"] = _save(paths, "headline_multiseed.csv", spectral)

    pareto_payload = json.loads(
        paths.child("raw", "attack_pareto.json").read_text(encoding="utf-8")
    )
    pareto = pd.DataFrame(
        [
            {
                **row["final_diagnostics"],
                "tradeoff_point": row["tradeoff_point"],
                "restart": row["restart"],
                "seed": row["seed"],
            }
            for row in pareto_payload["restarts"]
        ]
    )
    outputs["pareto_frontier.csv"] = _save(paths, "pareto_frontier.csv", pareto)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5.5, 4.0))
    for name, group in pareto.groupby("tradeoff_point"):
        axis.scatter(group["rho_mean"], group["task_loss"], label=name, alpha=0.8)
    axis.set(
        xlabel="Mean spectral radius",
        ylabel="Task loss",
        title="Calibration Pareto restarts",
    )
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure_path = paths.child("figures", "pareto_frontier_band.pdf")
    figure.savefig(figure_path)
    plt.close(figure)
    outputs["pareto_frontier_band.pdf"] = str(figure_path)

    test_rows, effect_rows = [], []
    clean_scores = main[(main["split"] == "test") & (main["condition"] == "clean")][
        ["prompt_id", "task_score", "spectral"]
    ]
    for condition in ("benign_control", "hispa_original", "gradient_guided"):
        other = main[(main["split"] == "test") & (main["condition"] == condition)][
            ["prompt_id", "task_score", "spectral"]
        ]
        joined = clean_scores.merge(
            other, on="prompt_id", suffixes=("_clean", "_other"), validate="one_to_one"
        )
        difference = joined["task_score_clean"] - joined["task_score_other"]
        try:
            statistic_w, p_w = wilcoxon(difference)
        except ValueError:
            statistic_w, p_w = 0.0, 1.0
        statistic_t, p_t = ttest_rel(
            joined["task_score_clean"], joined["task_score_other"]
        )
        test_rows.append(
            {
                "comparison": f"clean_vs_{condition}",
                "test": "paired_wilcoxon",
                "statistic": statistic_w,
                "p_value": p_w,
                "n": len(joined),
            }
        )
        rho_clean = (
            joined["spectral_clean"]
            .map(lambda value: value["rho_mean_over_layers"])
            .to_numpy(float)
        )
        rho_other = (
            joined["spectral_other"]
            .map(lambda value: value["rho_mean_over_layers"])
            .to_numpy(float)
        )
        delta = rho_clean - rho_other
        cohen_d = float(delta.mean() / delta.std(ddof=1)) if delta.std(ddof=1) else 0.0
        rng = np.random.default_rng(int(config["seeds"][0]))
        samples = rng.integers(
            0,
            len(delta),
            size=(int(config["protocol"]["cohen_d_bootstrap_resamples"]), len(delta)),
        )
        boot = delta[samples]
        denominators = boot.std(axis=1, ddof=1)
        boot_d = np.divide(
            boot.mean(axis=1),
            denominators,
            out=np.zeros_like(denominators),
            where=denominators > 0,
        )
        ci_low, ci_high = np.quantile(boot_d, [0.025, 0.975])
        effect_rows.append(
            {
                "comparison": f"clean_vs_{condition}",
                "n": len(delta),
                "cohen_d_paired": cohen_d,
                "cohen_d_ci_low": float(ci_low),
                "cohen_d_ci_high": float(ci_high),
                "mean_delta_rho": float(delta.mean()),
                "ttest_statistic": float(statistic_t),
                "ttest_p_value": float(p_t),
            }
        )
    tests = pd.DataFrame(test_rows)
    rejected, adjusted, _, _ = multipletests(
        tests["p_value"],
        alpha=float(config["protocol"]["false_discovery_rate"]),
        method="fdr_bh",
    )
    bh = tests.assign(p_value_bh=adjusted, rejected_at_fdr=rejected)
    outputs["multiple_comparisons_bh.csv"] = _save(
        paths, "multiple_comparisons_bh.csv", bh
    )
    outputs["nonparametric_tests.csv"] = _save(paths, "nonparametric_tests.csv", tests)
    effects = pd.DataFrame(effect_rows)
    effects["bootstrap_resamples"] = int(
        config["protocol"]["cohen_d_bootstrap_resamples"]
    )
    outputs["cohen_d_bootstrap.csv"] = _save(paths, "cohen_d_bootstrap.csv", effects)
    common = pd.concat(
        [
            spectral.assign(detector="SpectralGuard"),
            clasp.assign(detector="CLASP-style BOE+XGBoost reimplementation"),
        ],
        ignore_index=True,
    )
    outputs["clasp_common_protocol.csv"] = _save(
        paths, "clasp_common_protocol.csv", common
    )
    outputs["mamba_2_8b_sign_flip.csv"] = _save(
        paths,
        "mamba_2_8b_sign_flip.csv",
        effects[["comparison", "n", "mean_delta_rho", "cohen_d_paired"]],
    )
    observed = effects["cohen_d_paired"].abs().replace(0, np.nan).min()
    required_n = (
        float(
            TTestPower().solve_power(effect_size=float(observed), power=0.8, alpha=0.05)
        )
        if pd.notna(observed)
        else math.inf
    )
    power = pd.DataFrame(
        [
            {
                "smallest_observed_absolute_effect": observed,
                "target_power": 0.8,
                "alpha": 0.05,
                "required_n_per_group": required_n,
                "available_test_n": int(config["dataset"]["split_totals"]["test"]),
            }
        ]
    )
    outputs["statistical_power.csv"] = _save(paths, "statistical_power.csv", power)

    report = {
        "schema_version": 2,
        "raw_rows": len(frame),
        "models": sorted(frame["model_key"].unique()),
        "seeds": config["seeds"],
        "lexical_auc_gate_passed": bool(
            (lexical["test_auc"] < lexical["auc_ceiling"]).all()
        ),
        "threshold_policy": "fit calibration; select threshold validation; evaluate test once",
        "multiple_comparisons": "Benjamini-Hochberg",
        "clasp_scope": "BOE+XGBoost reimplementation; no claim of official pretrained weights",
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
    from src.eval.phase3_acceptance import validate_phase3_acceptance

    acceptance = validate_phase3_acceptance()
    if acceptance["status"] != "PASS":
        raise RuntimeError("Phase 3 acceptance gate did not pass")
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
