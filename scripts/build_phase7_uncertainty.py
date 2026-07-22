"""Build the Phase 7 detector-uncertainty table from frozen Phase 3 evidence.

SpectralGuard is deterministic on the frozen split, so its uncertainty is a
cluster bootstrap over test prompt IDs.  The CLASP-style comparator changes
with its training seed; its interval is a descriptive t interval over the five
fits on that same split.  These intervals answer different questions and must
not be presented as interchangeable error bars.
"""

from __future__ import annotations

import ast
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "results" / "phase3" / "summaries" / "clasp_common_protocol.csv"
RAW = ROOT / "Experiments" / "raw" / "E01_main_mamba.jsonl"
OUTPUT = ROOT / "results" / "phase3" / "summaries" / "detector_uncertainty.csv"
BENIGN = {"clean", "benign_control"}
ATTACKED = {"hispa_original", "gradient_guided"}


def _spectralguard_predictions(detector: dict[str, str]) -> pd.DataFrame:
    mean = np.asarray(ast.literal_eval(detector["feature_mean"]), dtype=float)
    scale = np.asarray(ast.literal_eval(detector["feature_scale"]), dtype=float)
    weights = np.asarray(ast.literal_eval(detector["weights"]), dtype=float)
    bias = float(detector["bias"])
    threshold = float(detector["threshold"])
    rows: list[dict[str, object]] = []
    with RAW.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("split") != "test" or row.get("condition") not in BENIGN | ATTACKED:
                continue
            features = np.asarray(
                [float(layer["rho_mean"]) for layer in row["spectral"]["layers"]],
                dtype=float,
            )
            if len(features) != len(weights):
                raise ValueError("frozen detector and test feature dimensions differ")
            logit = float((((features - mean) / np.maximum(scale, 1e-12)) * weights).sum() + bias)
            probability = 1.0 / (1.0 + math.exp(-logit))
            rows.append(
                {
                    "prompt_id": row["prompt_id"],
                    "label": int(row["condition"] in ATTACKED),
                    "probability": probability,
                    "prediction": int(probability >= threshold),
                }
            )
    frame = pd.DataFrame(rows)
    if len(frame) != 4000 or frame["prompt_id"].nunique() != 1000:
        raise ValueError("unexpected frozen test matrix")
    return frame


def _cluster_bootstrap(frame: pd.DataFrame, *, seed: int, resamples: int) -> dict[str, tuple[float, float, float]]:
    groups = {key: value for key, value in frame.groupby("prompt_id", sort=True)}
    prompt_ids = np.asarray(sorted(groups))
    rng = np.random.default_rng(seed)
    f1_values, auc_values = [], []
    for _ in range(resamples):
        sampled = rng.choice(prompt_ids, size=len(prompt_ids), replace=True)
        draw = pd.concat([groups[prompt_id] for prompt_id in sampled], ignore_index=True)
        f1_values.append(f1_score(draw["label"], draw["prediction"]))
        auc_values.append(roc_auc_score(draw["label"], draw["probability"]))
    return {
        "f1": (
            float(f1_score(frame["label"], frame["prediction"])),
            *[float(value) for value in np.quantile(f1_values, [0.025, 0.975])],
        ),
        "auc": (
            float(roc_auc_score(frame["label"], frame["probability"])),
            *[float(value) for value in np.quantile(auc_values, [0.025, 0.975])],
        ),
    }


def build() -> pd.DataFrame:
    common = pd.read_csv(COMMON)
    spectral = common[common["detector"] == "SpectralGuard"].copy()
    clasp = common[common["detector"].str.startswith("CLASP-style")].copy()
    if len(spectral) != 5 or len(clasp) != 5:
        raise ValueError("expected five frozen fits for each detector")
    detector = next(csv.DictReader(COMMON.open(newline="", encoding="utf-8")))
    bootstrap = _cluster_bootstrap(
        _spectralguard_predictions(detector), seed=42, resamples=2000
    )
    records: list[dict[str, object]] = []
    for metric, (point, low, high) in bootstrap.items():
        records.append(
            {
                "detector": "SpectralGuard",
                "metric": metric,
                "point_estimate": point,
                "interval_low": low,
                "interval_high": high,
                "interval_kind": "cluster bootstrap over fixed test prompt IDs",
                "resamples": 2000,
                "cluster_unit": "prompt_id",
                "seed_fits": 5,
                "seed_sd": 0.0,
                "interpretation": "sampling uncertainty on the fixed test set; the deterministic fits are identical",
            }
        )
    for metric, column in (("f1", "test_f1"), ("auc", "test_auc")):
        values = clasp[column].to_numpy(dtype=float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        half_width = float(t.ppf(0.975, len(values) - 1) * sd / math.sqrt(len(values)))
        records.append(
            {
                "detector": "CLASP-style BOE+XGBoost reimplementation",
                "metric": metric,
                "point_estimate": mean,
                "interval_low": mean - half_width,
                "interval_high": mean + half_width,
                "interval_kind": "descriptive t interval over training seeds on one shared split",
                "resamples": 0,
                "cluster_unit": "not applicable",
                "seed_fits": len(values),
                "seed_sd": sd,
                "interpretation": "seed-to-seed training variation, not independent sampling uncertainty",
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    if not RAW.is_file():
        raise FileNotFoundError(
            "The private Phase 3 raw JSONL is required to rebuild the fixed-test bootstrap table"
        )
    table = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False, lineterminator="\n")
    print(f"PHASE 7 UNCERTAINTY: PASS ({OUTPUT.relative_to(ROOT)})")


if __name__ == "__main__":
    main()
