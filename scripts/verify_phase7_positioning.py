"""Verify Phase 7 claims, statistical labels, and manuscript completeness."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper" / "main.tex"
UNCERTAINTY = ROOT / "results" / "phase3" / "summaries" / "detector_uncertainty.csv"
LITERATURE = ROOT / "results" / "foundation" / "phase7_literature_audit.json"


def main() -> None:
    tex = TEX.read_text(encoding="utf-8")
    normalized = " ".join(tex.split())
    assert r"\title{Detection Is Not Defense: Stress-Testing Spectral Monitoring for Hidden-State Poisoning}" in tex
    for marker in (
        r"\section{Introduction}",
        r"\section{SpectralGuard Defense}",
        r"\section{Adaptive Threat Model}",
        r"\section{Conclusion}",
        r"\label{tab:detectors}",
        r"\label{tab:adaptive}",
        r"\label{tab:availability}",
    ):
        assert marker in tex, marker
    required = (
        "cluster bootstrap over the fixed test prompt IDs",
        "seed-to-seed training variation on the same shared split",
        "not independent sampling uncertainty",
        "floor-saturated regime",
        "make no detector-superiority claim",
        "no general priority claim",
        "26,725.9 recorded GPU-seconds",
    )
    for phrase in required:
        assert phrase in normalized, phrase
    banned = (
        "first session-level",
        "first study of session",
        "production-ready",
        "spectral monitoring is unconditionally complete",
        "CLASP reproduction",
        "equally powerful",
        "Transformer resistance",
    )
    lowered = normalized.lower()
    for phrase in banned:
        assert phrase.lower() not in lowered, phrase
    windows_user_root = r"[A-Za-z]:[\\/]+" + "Users" + r"[\\/]"
    teamspace_root = "/" + "teamspace/"
    zeus_home = "/" + "home/" + "zeus/"
    assert not re.search(windows_user_root, tex, flags=re.I)
    assert teamspace_root not in tex and zeus_home not in tex

    table = pd.read_csv(UNCERTAINTY)
    assert set(table["metric"]) == {"f1", "auc"}
    assert set(table["detector"]) == {
        "SpectralGuard",
        "CLASP-style BOE+XGBoost reimplementation",
    }
    spectral_f1 = table[(table["detector"] == "SpectralGuard") & (table["metric"] == "f1")].iloc[0]
    assert abs(float(spectral_f1["point_estimate"]) - 0.8061420345489443) < 1e-12
    assert float(spectral_f1["interval_low"]) < 0.806 < float(spectral_f1["interval_high"])
    clasp_f1 = table[(table["detector"].str.startswith("CLASP-style")) & (table["metric"] == "f1")].iloc[0]
    assert 0.0026 < float(clasp_f1["seed_sd"]) < 0.0028
    assert "not independent sampling uncertainty" in clasp_f1["interpretation"]

    audit = json.loads(LITERATURE.read_text(encoding="utf-8"))
    assert len(audit["primary_sources"]) >= 5
    assert "no general first-study claim" in audit["result"]
    print("PHASE 7 POSITIONING: PASS")


if __name__ == "__main__":
    main()
