"""Verify the lightweight, tracked Phase 3 evidence without GPU access."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "phase3"


def _csv(name: str) -> pd.DataFrame:
    path = RESULTS / "summaries" / name
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"empty Phase 3 table: {path}")
    return frame


def _build_denylist() -> tuple[str, ...]:
    """Return tokens that must not appear in tracked result files.

    Generic machine-path patterns are always included.  Private terms
    (e.g., author names) are loaded exclusively from a local, git-ignored
    source so they never exist as literals in the published repository:

    - Environment variable SPECTRALGUARD_PRIVATE_DENYLIST (colon-separated)
    - File .spectralguard_denylist at the repository root (one term per line)

    Both sources are optional; absence is not an error.
    """
    tokens: list[str] = [
        # Generic absolute-path prefixes — cannot reveal specific identity
        "C:" + "\\Users\\",
        "/team" + "space/",
        # Generic email / URL patterns
        "@" + "gmail.com",
        "@" + "hotmail.com",
        "github.com/",
    ]

    # Load private terms from environment variable (colon-separated)
    env_val = os.environ.get("SPECTRALGUARD_PRIVATE_DENYLIST", "")
    if env_val.strip():
        tokens.extend(t.strip() for t in env_val.split(":") if t.strip())

    # Load private terms from local untracked file
    denylist_file = ROOT / ".spectralguard_denylist"
    if denylist_file.is_file():
        for line in denylist_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                tokens.append(line)

    return tuple(tokens)


def verify() -> None:
    provenance = json.loads((RESULTS / "provenance.json").read_text(encoding="utf-8"))
    if provenance["confirmatory_raw_rows"] != 19300:
        raise ValueError("unexpected confirmatory row count")

    acceptance = json.loads(
        (RESULTS / "manifests" / "phase3_acceptance.json").read_text(
            encoding="utf-8"
        )
    )
    if (acceptance["status"], acceptance["passed_items"], acceptance["required_items"]) != (
        "PASS",
        18,
        18,
    ):
        raise ValueError("Phase 3 acceptance matrix is not 18/18 PASS")

    lexical = _csv("lexical_control.csv")
    gpt2 = lexical[lexical["model_key"] == "gpt2_xl"]
    if gpt2.empty or not (gpt2["test_auc"] < gpt2["auc_ceiling"]).all():
        raise ValueError("GPT-2 lexical leakage gate failed")

    transfer = _csv("cross_model_transfer_wilson.csv")
    if len(transfer) != 12 or (transfer["n"] < 200).any():
        raise ValueError("cross-model transfer does not contain 12 cells with N >= 200")

    paired = _csv("mcnemar_tests.csv")
    if int(paired.loc[0, "n"]) < 300 or int(paired.loc[0, "bootstrap_resamples"]) < 2000:
        raise ValueError("adaptive paired test is underpowered or lacks bootstrap evidence")
    if paired[["paired_difference_ci_low", "paired_difference_ci_high"]].isna().any().any():
        raise ValueError("adaptive paired confidence interval is absent")

    regression = _csv("rho_accuracy_regression.csv")
    if regression["mcfadden_pseudo_r2"].isna().any():
        raise ValueError("McFadden pseudo-R2 is absent")

    threshold = _csv("threshold_provenance.csv")
    if set(threshold["threshold_selected_on"]) != {"validation"}:
        raise ValueError("threshold leakage detected")

    pareto = _csv("pareto_frontier.csv")
    if not (pareto.groupby("tradeoff_point")["restart"].nunique() == 5).all():
        raise ValueError("Pareto cells do not contain five independent restarts")
    figure = RESULTS / "figures" / "pareto_frontier_band.pdf"
    if not figure.read_bytes().startswith(b"%PDF"):
        raise ValueError("Pareto PDF is absent or invalid")

    forbidden = _build_denylist()
    for path in RESULTS.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
            text = path.read_text(encoding="utf-8")
            if any(token in text for token in forbidden):
                raise ValueError(f"identity or absolute-path leak in {path}")

    print("PHASE 3 TRACKED EVIDENCE: PASS (18/18 semantic gates)")


if __name__ == "__main__":
    verify()
