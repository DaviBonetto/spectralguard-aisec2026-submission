"""Build acceptance-grade Phase 8 figures from tracked evidence only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
MANIFEST = ROOT / "results" / "foundation" / "phase8_figure_manifest.json"

# Okabe--Ito: color-blind safe, high contrast, and legible in grayscale.
BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
INK = "#16212C"
MID = "#536171"
GRID = "#D9E0E8"
PALE_BLUE = "#EAF4FA"
PALE_ORANGE = "#FFF4D8"
PALE_GREEN = "#E7F6F1"
PALE_RED = "#FCECE6"


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.labelsize": 8.2,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.55,
            "lines.markersize": 5.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def save(fig: plt.Figure, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return path


def clean_axis(ax: plt.Axes, grid: str | None = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.6, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.11, 1.08, label, transform=ax.transAxes, weight="bold", fontsize=9.2, va="top")


def rounded(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float],
            title: str, subtitle: str, edge: str, face: str,
            title_size: float = 7.7, sub_size: float = 6.2) -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.010,rounding_size=0.016",
            linewidth=1.2, edgecolor=edge, facecolor=face,
        )
    )
    ax.text(x + w / 2, y + h - 0.042, title, weight="bold", fontsize=title_size,
            va="top", ha="center", linespacing=1.15, color=INK)
    ax.text(x + w / 2, y + 0.045, subtitle, fontsize=sub_size, va="bottom",
            ha="center", color=MID, linespacing=1.22)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = MID) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10,
                                 linewidth=1.2, color=color, shrinkA=0, shrinkB=0))


def build_overview() -> Path:
    """Two-column DURA overview: threat, observable, decision, and four outcomes."""
    fig, ax = plt.subplots(figsize=(7.0, 3.28))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    rounded(ax, (0.010, 0.560), (0.220, 0.310), "1  ATTACK",
            "hidden-state attack\ntrigger / state / task", ORANGE, PALE_ORANGE,
            title_size=7.8, sub_size=6.1)
    rounded(ax, (0.250, 0.560), (0.255, 0.310), "2  TELEMETRY OPTIONS",
            "spectral radius → SpectralGuard\nBOE → CLASP-style comparator", BLUE, PALE_BLUE,
            title_size=7.1, sub_size=5.7)
    rounded(ax, (0.525, 0.560), (0.225, 0.310), "3  FROZEN FITS",
            "calibrate / validate / test\nno final-test refit", PURPLE, "#F8ECF5",
            title_size=7.8, sub_size=6.1)
    rounded(ax, (0.770, 0.560), (0.220, 0.310), "4  POLICY",
            "spectral alerts only\nfail-closed / k-of-w", GREEN, PALE_GREEN,
            title_size=7.8, sub_size=6.1)

    arrow(ax, (0.232, 0.715), (0.248, 0.715), MID)
    arrow(ax, (0.507, 0.715), (0.523, 0.715), MID)
    arrow(ax, (0.752, 0.715), (0.768, 0.715), MID)

    ax.text(0.5, 0.450, "DURA evaluation: a detector score is only one coordinate", ha="center",
            weight="bold", fontsize=9.8)
    cards = [
        (0.010, 0.220, "D", "DETECTION", "SpectralGuard: 0.806\nBOE: 0.969", BLUE, PALE_BLUE),
        (0.250, 0.255, "U", "UTILITY", "Clean: 13/300 (4.33%)\nAttacked: 5/300 (1.67%)", ORANGE, PALE_ORANGE),
        (0.525, 0.225, "R", "ROBUSTNESS", "Transfer F1: 0.810\nWhite-box F1: 0.746", RED, PALE_RED),
        (0.770, 0.220, "A", "AVAILABILITY", "Fail-closed: 0/320\nGraduated: 8/320", GREEN, PALE_GREEN),
    ]
    for x, w, letter, title, value, edge, face in cards:
        ax.add_patch(FancyBboxPatch((x, 0.09), w, 0.29,
                    boxstyle="round,pad=0.010,rounding_size=0.016",
                    linewidth=0.95, edgecolor=edge, facecolor=face))
        ax.add_patch(Circle((x + 0.035, 0.323), 0.024, facecolor=edge, edgecolor="none"))
        ax.text(x + 0.035, 0.323, letter, ha="center", va="center", color="white", weight="bold", fontsize=8.0)
        ax.text(x + 0.070, 0.323, title, va="center", weight="bold", fontsize=7.5)
        ax.text(x + w / 2, 0.205, value, ha="center", va="center", weight="bold",
                fontsize=6.3, linespacing=1.30)

    ax.text(0.5, 0.018,
            "A useful monitor must be evaluated jointly: separability alone does not establish preserved behavior or service.",
            ha="center", fontsize=7.4, color=MID)
    return save(fig, "evaluation_pipeline.pdf")


def build_theory() -> Path:
    evidence = pd.read_csv(ROOT / "results" / "theory" / "directional_counterexample_table.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.52), gridspec_kw={"wspace": 0.46})

    ax = axes[0]
    rho = np.linspace(0.08, 0.995, 500)
    for eps, color, style, label in (
        (0.1, BLUE, "-", r"$\epsilon=0.1$"),
        (0.01, ORANGE, "--", r"$\epsilon=0.01$"),
    ):
        horizon = np.ceil(np.log(1 / eps) / np.log(1 / rho))
        ax.plot(rho, horizon, color=color, linestyle=style, label=label)
    ax.set_yscale("log")
    ax.set_xlabel(r"spectral radius $\rho$")
    ax.set_ylabel("forgetting upper bound")
    ax.set_title("What radius certifies", weight="bold")
    ax.text(.50, .90, r"$\left\lceil\frac{\log(1/\epsilon)}{\log(1/\rho)}\right\rceil$",
            transform=ax.transAxes, fontsize=7.2, ha="center")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, .82))
    clean_axis(ax, "both")
    panel(ax, "a")

    ax = axes[1]
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color=GRID, lw=0.7)
    ax.axvline(0, color=GRID, lw=0.7)
    u = np.array([0.80, 0.60])
    v = np.array([-0.60, 0.80])
    ax.arrow(0, 0, *(0.72 * u), width=0.012, head_width=0.08, color=GREEN,
             length_includes_head=True, zorder=3)
    ax.arrow(0, 0, *(0.96 * u), width=0.008, head_width=0.075, color=BLUE,
             linestyle="--", length_includes_head=True, zorder=2)
    ax.arrow(0, 0, *(0.86 * v), width=0.010, head_width=0.08, color=RED,
             length_includes_head=True, zorder=3)
    s = .14
    corner = np.vstack([s * u, s * (u + v), s * v])
    ax.plot(corner[:, 0], corner[:, 1], color=INK, lw=1.0)
    ax.text(.46, .30, r"state $h$", color=GREEN, weight="bold")
    ax.text(.73, .65, r"$C=h^\top/\|h\|$", color=BLUE, weight="bold", fontsize=7.0)
    ax.text(-.82, .62, r"$C'\perp h$", color=RED, weight="bold")
    ax.text(.18, -.23, r"$Ch=\|h\|$", color=BLUE, fontsize=7.2)
    ax.text(-.82, -.23, r"$C'h=0$", color=RED, fontsize=7.2)
    ax.text(0, -.92, "same $\\bar A$, same $\\rho$\ndifferent output", ha="center",
            fontsize=7.1, bbox={"boxstyle": "round,pad=.25", "facecolor": "white", "edgecolor": GRID})
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("What radius misses", weight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel(ax, "b")

    ax = axes[2]
    order = ["spectral_collapse_control", "directional_readout_counterexample"]
    selected = evidence.set_index("attack_family").loc[order]
    y = np.arange(2)
    rates = selected["monitor_detection_rate"].to_numpy(float)
    intervals = [json.loads(value) for value in selected["monitor_detection_wilson_95"]]
    low = np.array([value[0] for value in intervals])
    high = np.array([value[1] for value in intervals])
    ax.errorbar(rates, y, xerr=np.vstack([rates - low, high - rates]), fmt="none",
                ecolor=INK, capsize=2.5, zorder=2)
    ax.scatter([rates[0]], [y[0]], s=48, c=GREEN, marker="s", edgecolor="white", linewidth=0.8, zorder=3)
    ax.scatter([rates[1]], [y[1]], s=48, c=RED, marker="o", edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yticks(y, ["Radius\ncollapse", "Readout\nrotation"])
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks([0, .5, 1])
    ax.set_xlabel("detection rate")
    ax.set_title("Constructive test", weight="bold")
    ax.invert_yaxis()
    ax.text(.99, .13, "1,000/1,000\n[0.9962, 1]", ha="right", va="top",
            color=GREEN, weight="bold", fontsize=6.6)
    ax.text(.01, 1, "0/1,000\n[0, 0.0038]", ha="left", va="bottom",
            color=RED, weight="bold", fontsize=6.6)
    clean_axis(ax, "x")
    panel(ax, "c")
    return save(fig, "theory_counterexample.pdf")


def build_detection() -> Path:
    uncertainty = pd.read_csv(ROOT / "results" / "phase3" / "summaries" / "detector_uncertainty.csv")
    fits = pd.read_csv(ROOT / "results" / "phase3" / "summaries" / "clasp_common_protocol.csv")
    worst = pd.read_csv(ROOT / "results" / "phase4" / "summaries" / "worst_case_defense.csv")
    paired = pd.read_csv(ROOT / "results" / "phase3" / "summaries" / "adaptive_v2_v3_paired.csv")
    attack_tasks = pd.read_csv(ROOT / "results" / "phase4" / "summaries" / "attack_comparison_by_task.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55), gridspec_kw={"wspace": 0.46})

    ax = axes[0]
    sg = uncertainty[(uncertainty.detector == "SpectralGuard") & (uncertainty.metric == "f1")].iloc[0]
    clasp = fits[fits.detector.str.startswith("CLASP-style")].sort_values("seed")
    ax.errorbar([sg.point_estimate], [1],
                xerr=[[sg.point_estimate - sg.interval_low], [sg.interval_high - sg.point_estimate]],
                fmt="D", color=BLUE, capsize=2.5, zorder=3)
    jitter = np.linspace(-.11, .11, len(clasp))
    ax.scatter(clasp.test_f1, jitter, marker="o", facecolors="white", edgecolors=ORANGE, s=31)
    ax.vlines(clasp.test_f1.mean(), -.18, .18, color=ORANGE, linewidth=2.5)
    ax.scatter([clasp.test_f1.mean()], [0], marker="D", color=ORANGE, s=31, zorder=3)
    ax.set_yticks([0, 1], ["BOE+XGBoost", "SpectralGuard"])
    ax.set_xlim(.76, 1.02)
    ax.set_xlabel("held-out F1")
    ax.set_title("Shared protocol", weight="bold")
    ax.text(1.015, 1, "0.806 [0.795, 0.818]", ha="right", va="bottom", fontsize=6.5)
    ax.text(1.015, 0, "mean 0.969; SD 0.0027", ha="right", va="bottom", fontsize=6.5)
    clean_axis(ax, "x")
    panel(ax, "a")

    ax = axes[1]
    labels = ["Transfer", "Query", "Joint\nwhite-box"]
    x = np.arange(3)
    for metric, color, marker, style, name in (
        ("f1", BLUE, "o", "--", "F1"),
        ("auc", GREEN, "s", "-", "ROC-AUC"),
    ):
        values = worst[metric].to_numpy(float)
        lows = worst[f"{metric}_ci_low"].to_numpy(float)
        highs = worst[f"{metric}_ci_high"].to_numpy(float)
        ax.errorbar(x, values, yerr=np.vstack([values - lows, highs - values]),
                    fmt=marker, linestyle=style, color=color, capsize=2.2, label=name)
    ax.set_xticks(x, labels)
    ax.set_ylim(.66, .96)
    ax.set_ylabel("F1 / ROC-AUC")
    ax.set_title("Adaptive escalation", weight="bold")
    ax.legend(frameon=False, ncol=2, loc="lower left")
    clean_axis(ax)
    panel(ax, "b")

    ax = axes[2]
    clean_correct = int(attack_tasks[attack_tasks.condition == "blackbox_transfer"].clean_correct_n.sum())
    counts = np.array([clean_correct, int(paired.v2_correct.sum()), int(paired.v3_correct.sum())])
    values = counts / 300 * 100
    ax.bar([0, 1, 2], values, color=[MID, ORANGE, RED], width=.62,
           edgecolor=INK, linewidth=.6)
    ax.set_xticks([0, 1, 2], ["Clean", "Adapt.\nv2", "Aware\nv3"])
    ax.tick_params(axis="x", labelsize=7.2)
    ax.set_ylim(0, 5.25)
    ax.set_ylabel("task accuracy (%)")
    ax.set_title("Utility at the floor", weight="bold")
    for index, (value, count) in enumerate(zip(values, counts)):
        ax.text(index, value + .12, f"{count}/300\n{value:.2f}%", ha="center",
                weight="bold", fontsize=6.8)
    clean_axis(ax)
    panel(ax, "c")
    return save(fig, "detection_adaptation.pdf")


def build_availability() -> Path:
    sessions = pd.read_csv(ROOT / "results" / "phase5" / "summaries" / "session_survival.csv")
    overhead = pd.read_csv(ROOT / "results" / "phase5" / "summaries" / "runtime_overhead.csv").iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55), gridspec_kw={"wspace": 0.48})
    order = ["clean", "matched_benign_control", "adversarial_benign", "malicious_whitebox"]
    labels = ["Clean", "Matched", "Adv. benign", "Malicious"]

    ax = axes[0]
    for offset, policy, color, marker, label in (
        (-.10, "algorithm2", MID, "o", "Fail-closed"),
        (.10, "algorithm2b", GREEN, "s", "Graduated"),
    ):
        data = sessions[sessions.policy == policy].set_index("condition").loc[order]
        y = np.arange(4) + offset
        values = data.survival_rate.to_numpy(float) * 100
        low = data.survival_ci_low.to_numpy(float) * 100
        high = data.survival_ci_high.to_numpy(float) * 100
        ax.errorbar(values, y, xerr=np.vstack([values - low, high - values]), fmt=marker,
                    color=color, capsize=2.2, label=label)
        counts = (data.survival_rate.to_numpy(float) * data.sessions.to_numpy(int)).round().astype(int)
        for x_value, y_value, count in zip(values, y, counts):
            ax.text(max(x_value + 2.5, 2.5), y_value, f"{count}/320", va="center",
                    fontsize=6.4, color=color)
    ax.set_yticks(np.arange(4), labels)
    ax.set_xlim(-2, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("session survival (%)")
    ax.set_title("Held-out sessions", weight="bold")
    ax.legend(frameon=False, loc="lower right")
    clean_axis(ax, "x")
    ax.invert_yaxis()
    panel(ax, "a")

    ax = axes[1]
    fail = sessions[sessions.policy == "algorithm2"].set_index("condition").loc[order]
    grad = sessions[sessions.policy == "algorithm2b"].set_index("condition").loc[order]
    x = np.arange(4)
    width = .34
    fail_values = fail.blocked_window_rate.to_numpy(float) * 100
    grad_values = grad.blocked_window_rate.to_numpy(float) * 100
    ax.bar(x - width / 2, fail_values, width, color=MID, label="Fail-closed", hatch="//")
    ax.bar(x + width / 2, grad_values, width, color=GREEN, label="Graduated")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("blocked windows (%)")
    ax.set_title("Alert saturation", weight="bold")
    for positions, values in ((x - width / 2, fail_values), (x + width / 2, grad_values)):
        for x_value, value in zip(positions, values):
            ax.text(x_value, value + 1.2, f"{value:.1f}", ha="center", fontsize=5.9)
    clean_axis(ax)
    panel(ax, "b")

    ax = axes[2]
    values = [float(overhead.baseline_seconds_per_session), float(overhead.monitored_seconds_per_session)]
    ax.bar([0, 1], values, width=.58, color=[SKY, RED], edgecolor=INK, linewidth=.6)
    ax.set_xticks([0, 1], ["Baseline", "Monitored"])
    ax.set_ylabel("seconds / session")
    ax.set_title("End-to-end cost", weight="bold")
    for index, value in enumerate(values):
        ax.text(index, value + .003, f"{value:.3f} s", ha="center", weight="bold", fontsize=6.8)
    delta = values[1] - values[0]
    ax.text(.5, .138, f"$\\Delta$=+{delta:.3f} s\n{overhead.overhead_ratio:.2f}$\\times$ (+{overhead.overhead_percent:.1f}%)",
            ha="center", va="top", color=RED, fontsize=6.8)
    ax.set_ylim(0, .145)
    clean_axis(ax)
    panel(ax, "c")
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
            "results/phase3/summaries/detector_uncertainty.csv",
            "results/phase3/summaries/adaptive_v2_v3_paired.csv",
            "results/phase4/summaries/worst_case_defense.csv",
            "results/phase4/summaries/attack_comparison_by_task.csv",
            "results/phase5/summaries/session_survival.csv",
        ],
        "theory_counterexample.pdf": ["results/theory/directional_counterexample_table.csv"],
        "detection_adaptation.pdf": [
            "results/phase3/summaries/detector_uncertainty.csv",
            "results/phase3/summaries/clasp_common_protocol.csv",
            "results/phase3/summaries/adaptive_v2_v3_paired.csv",
            "results/phase4/summaries/worst_case_defense.csv",
            "results/phase4/summaries/attack_comparison_by_task.csv",
        ],
        "availability_cost.pdf": [
            "results/phase5/summaries/session_survival.csv",
            "results/phase5/summaries/runtime_overhead.csv",
        ],
    }
    manifest = {
        "schema_version": 2,
        "generator": "scripts/build_phase8_figures.py",
        "style": "Okabe-Ito palette; vector PDF; embedded TrueType fonts; conclusion-first panels",
        "figures": {
            output.name: {
                "path": output.relative_to(ROOT).as_posix(),
                "sha256": sha256(output),
                "bytes": output.stat().st_size,
                "format": "vector_pdf",
                "sources": [
                    {"path": source, "sha256": sha256(ROOT / source)} for source in sources[output.name]
                ],
            }
            for output in outputs
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"PHASE 8 FIGURES: PASS ({len(outputs)} publication-scale vector PDFs)")


if __name__ == "__main__":
    main()
