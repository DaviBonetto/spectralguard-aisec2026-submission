"""Build the publication figures for Phase 8 from tracked evidence only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
MANIFEST = ROOT / "results" / "foundation" / "phase8_figure_manifest.json"

# Okabe--Ito palette: colorblind safe and still distinguishable in grayscale
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
GREY = "#5B6573"
LIGHT = "#EDF2F7"
INK = "#17212B"


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 7.4,
            "axes.labelsize": 7.4,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.7,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.35,
            "lines.markersize": 4.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def clean_axis(ax: plt.Axes, grid: str | None = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color="#D8DEE8", linewidth=0.55, alpha=0.85, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=8.5,
        va="top",
    )


def save(fig: plt.Figure, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return path


def build_overview() -> Path:
    fig, ax = plt.subplots(figsize=(3.35, 2.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    nodes = [
        (0.02, 0.66, "Input\nsession", SKY),
        (0.35, 0.66, "Hidden-state\npoisoning", ORANGE),
        (0.68, 0.66, "Layer-wise\n$\\rho(\\bar A_t)$", BLUE),
        (0.02, 0.25, "Frozen\ndetector", PURPLE),
        (0.35, 0.25, "Response\npolicy", GREEN),
        (0.68, 0.25, "Utility +\navailability", RED),
    ]
    width = 0.28
    height = 0.22
    for x, y, label, color in nodes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.015,rounding_size=0.018",
            linewidth=1.1,
            edgecolor=color,
            facecolor=LIGHT,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontweight="bold")
    arrows = [
        ((0.305, 0.77), (0.345, 0.77)),
        ((0.635, 0.77), (0.675, 0.77)),
        ((0.82, 0.65), (0.16, 0.48)),
        ((0.305, 0.36), (0.345, 0.36)),
        ((0.635, 0.36), (0.675, 0.36)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, linewidth=0.95, color=GREY))
    ax.text(0.50, 0.055, "score $\\neq$ robust, usable defense", ha="center", color=RED, fontweight="bold")
    ax.text(0.50, -0.01, "test detection, utility, adaptation, and availability", ha="center", color=GREY, fontsize=7)
    return save(fig, "evaluation_pipeline.pdf")


def build_theory() -> Path:
    evidence = pd.read_csv(ROOT / "results" / "theory" / "directional_counterexample_table.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.25), gridspec_kw={"wspace": 0.32})

    ax = axes[0]
    rho = np.linspace(0.1, 0.995, 500)
    for eps, color, label in ((0.1, BLUE, "$\\epsilon=0.1$"), (0.01, ORANGE, "$\\epsilon=0.01$")):
        horizon = np.ceil(np.log(1 / eps) / np.log(1 / rho))
        ax.plot(rho, horizon, color=color, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Frozen spectral radius $\\rho$")
    ax.set_ylabel("Certified forgetting upper bound")
    ax.set_title("Exact diagonal certificate")
    ax.legend(frameon=False, loc="upper left")
    clean_axis(ax, "both")
    panel_label(ax, "a")

    ax = axes[1]
    order = ["spectral_collapse_control", "directional_readout_counterexample"]
    selected = evidence.set_index("attack_family").loc[order]
    y = np.arange(2)
    rates = selected["monitor_detection_rate"].to_numpy(float)
    intervals = [json.loads(value) for value in selected["monitor_detection_wilson_95"]]
    low = np.array([value[0] for value in intervals])
    high = np.array([value[1] for value in intervals])
    ax.errorbar(
        rates,
        y,
        xerr=np.vstack([rates - low, high - rates]),
        fmt="o",
        color=INK,
        ecolor=INK,
        capsize=2.5,
        zorder=3,
    )
    ax.scatter(rates, y, s=35, c=[GREEN, RED], edgecolor="white", linewidth=0.7, zorder=4)
    ax.set_yticks(y, ["Spectral-collapse\ncontrol", "Readout\ncounterexample"])
    ax.set_xlim(-0.04, 1.04)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Radius-monitor detection rate")
    ax.set_title("Same attack success; opposite detection")
    ax.invert_yaxis()
    ax.text(0.98, 0, "1,000/1,000", ha="right", va="bottom", color=GREEN, fontweight="bold")
    ax.text(0.02, 1, "0/1,000", ha="left", va="bottom", color=RED, fontweight="bold")
    clean_axis(ax, "x")
    panel_label(ax, "b")
    return save(fig, "theory_counterexample.pdf")


def build_detection() -> Path:
    uncertainty = pd.read_csv(ROOT / "results" / "phase3" / "summaries" / "detector_uncertainty.csv")
    fits = pd.read_csv(ROOT / "results" / "phase3" / "summaries" / "clasp_common_protocol.csv")
    worst = pd.read_csv(ROOT / "results" / "phase4" / "summaries" / "worst_case_defense.csv")
    paired = pd.read_csv(ROOT / "results" / "phase3" / "summaries" / "adaptive_v2_v3_paired.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw={"wspace": 0.38})

    ax = axes[0]
    sg = uncertainty[(uncertainty.detector == "SpectralGuard") & (uncertainty.metric == "f1")].iloc[0]
    clasp = fits[fits.detector.str.startswith("CLASP-style")].sort_values("seed")
    ax.hlines(1, sg.interval_low, sg.interval_high, color=BLUE, linewidth=4, alpha=0.75)
    ax.scatter([sg.point_estimate], [1], marker="D", color=BLUE, s=30, zorder=3, label="SG bootstrap CI")
    jitter = np.linspace(-0.11, 0.11, len(clasp))
    ax.scatter(clasp.test_f1, jitter, marker="o", facecolors="none", edgecolors=ORANGE, s=27, label="CLASP seed fits")
    ax.vlines(clasp.test_f1.mean(), -0.16, 0.16, color=ORANGE, linewidth=2.2)
    ax.set_yticks([0, 1], ["CLASP-style\n5 seed fits", "SpectralGuard\ncluster bootstrap"])
    ax.set_xlim(0.77, 0.985)
    ax.set_xlabel("Held-out F1")
    ax.set_title("Nonexchangeable uncertainty")
    clean_axis(ax, "x")
    panel_label(ax, "a")

    ax = axes[1]
    labels = ["Transfer", "Query-only", "Joint\nwhite-box"]
    x = np.arange(3)
    for offset, metric, color, marker in ((-0.08, "f1", BLUE, "o"), (0.08, "auc", GREEN, "s")):
        values = worst[metric].to_numpy(float)
        lows = worst[f"{metric}_ci_low"].to_numpy(float)
        highs = worst[f"{metric}_ci_high"].to_numpy(float)
        ax.errorbar(
            x + offset,
            values,
            yerr=np.vstack([values - lows, highs - values]),
            fmt=marker,
            color=color,
            capsize=2.2,
            label=metric.upper(),
        )
    ax.set_xticks(x, labels)
    ax.set_ylim(0.66, 0.95)
    ax.set_ylabel("Frozen-detector score")
    ax.set_title("Adaptive escalation")
    ax.legend(frameon=False, ncol=2, loc="lower left")
    clean_axis(ax, "y")
    panel_label(ax, "b")

    ax = axes[2]
    values = np.array([paired.v2_correct.mean(), paired.v3_correct.mean()]) * 100
    ax.bar([0, 1], values, color=[ORANGE, RED], width=0.56, edgecolor=INK, linewidth=0.55)
    ax.set_xticks([0, 1], ["Adaptive v2", "Detector-aware\nv3"])
    ax.set_ylim(0, 2.25)
    ax.set_ylabel("Attacked-task accuracy (%)")
    ax.set_title("Floor-saturated utility")
    for index, value in enumerate(values):
        ax.text(index, value + 0.08, f"{value:.2f}%", ha="center", fontweight="bold")
    ax.text(0.5, 2.03, "paired $\\Delta=-0.33$ pp\n95% CI $[-2.33,1.67]$", ha="center", va="top", color=GREY)
    clean_axis(ax, "y")
    panel_label(ax, "c")
    return save(fig, "detection_adaptation.pdf")


def build_availability() -> Path:
    sessions = pd.read_csv(ROOT / "results" / "phase5" / "summaries" / "session_survival.csv")
    overhead = pd.read_csv(ROOT / "results" / "phase5" / "summaries" / "runtime_overhead.csv").iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.65), gridspec_kw={"wspace": 0.42})
    condition_order = ["clean", "matched_benign_control", "adversarial_benign", "malicious_whitebox"]
    labels = ["Clean", "Matched\nbenign", "Adversarial\nbenign", "Malicious\nwhite-box"]

    ax = axes[0]
    for policy, color, marker, label in (("algorithm2", GREY, "o", "Fail closed"), ("algorithm2b", GREEN, "s", "Graduated")):
        data = sessions[sessions.policy == policy].set_index("condition").loc[condition_order]
        x = np.arange(4)
        values = data.survival_rate.to_numpy(float) * 100
        low = data.survival_ci_low.to_numpy(float) * 100
        high = data.survival_ci_high.to_numpy(float) * 100
        ax.errorbar(x, values, yerr=np.vstack([values - low, high - values]), fmt=marker, color=color, capsize=2, label=label)
    ax.set_xticks(np.arange(4), labels, rotation=32, ha="right")
    ax.set_ylim(-0.25, 5.1)
    ax.set_ylabel("Session survival (%)")
    ax.set_title("Held-out availability")
    ax.legend(frameon=False, loc="upper right")
    clean_axis(ax, "y")
    panel_label(ax, "a")

    ax = axes[1]
    alg2 = sessions[sessions.policy == "algorithm2"].set_index("condition").loc[condition_order]
    alg2b = sessions[sessions.policy == "algorithm2b"].set_index("condition").loc[condition_order]
    x = np.arange(4)
    width = 0.34
    ax.bar(x - width / 2, alg2.blocked_window_rate * 100, width, color=GREY, label="Fail closed")
    ax.bar(x + width / 2, alg2b.blocked_window_rate * 100, width, color=GREEN, label="Graduated")
    ax.set_xticks(x, labels, rotation=32, ha="right")
    ax.set_ylim(60, 102)
    ax.set_ylabel("Blocked windows (%)")
    ax.set_title("Alerts remain saturated")
    clean_axis(ax, "y")
    panel_label(ax, "b")

    ax = axes[2]
    values = [float(overhead.baseline_seconds_per_session), float(overhead.monitored_seconds_per_session)]
    ax.bar([0, 1], values, width=0.58, color=[SKY, RED], edgecolor=INK, linewidth=0.55)
    ax.set_xticks([0, 1], ["Baseline", "Monitored"])
    ax.set_ylabel("Seconds per session")
    ax.set_title("Runtime cost")
    ax.text(1, values[1] + 0.004, f"{overhead.overhead_ratio:.2f}$\\times$", ha="center", fontweight="bold", color=RED)
    ax.text(0.5, 0.113, f"+{overhead.overhead_percent:.1f}%", ha="center", color=GREY)
    ax.set_ylim(0, 0.125)
    clean_axis(ax, "y")
    panel_label(ax, "c")
    return save(fig, "availability_cost.pdf")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    configure()
    outputs = [build_overview(), build_theory(), build_detection(), build_availability()]
    sources = {
        "evaluation_pipeline.pdf": [
            "configs/phase3.yaml",
            "configs/phase4.yaml",
            "configs/phase5.yaml",
        ],
        "theory_counterexample.pdf": ["results/theory/directional_counterexample_table.csv"],
        "detection_adaptation.pdf": [
            "results/phase3/summaries/detector_uncertainty.csv",
            "results/phase3/summaries/clasp_common_protocol.csv",
            "results/phase3/summaries/adaptive_v2_v3_paired.csv",
            "results/phase4/summaries/worst_case_defense.csv",
        ],
        "availability_cost.pdf": [
            "results/phase5/summaries/session_survival.csv",
            "results/phase5/summaries/runtime_overhead.csv",
        ],
    }
    manifest = {
        "schema_version": 1,
        "generator": "scripts/build_phase8_figures.py",
        "style": "Okabe-Ito, vector PDF, embedded TrueType fonts",
        "figures": {
            output.name: {
                "path": output.relative_to(ROOT).as_posix(),
                "sha256": sha256(output),
                "bytes": output.stat().st_size,
                "format": "vector_pdf",
                "sources": [
                    {"path": source, "sha256": sha256(ROOT / source)}
                    for source in sources[output.name]
                ],
            }
            for output in outputs
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"PHASE 8 FIGURES: PASS ({len(outputs)} vector PDFs)")


if __name__ == "__main__":
    main()
