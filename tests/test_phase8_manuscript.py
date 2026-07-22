"""Regression tests for the Phase 8 paper and publication figures."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_phase8_manuscript_gate_passes() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/verify_phase8_manuscript.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "PHASE 8 MANUSCRIPT: PASS" in completed.stdout


def test_phase8_figure_manifest_is_complete_and_portable() -> None:
    manifest = json.loads(
        (ROOT / "results" / "foundation" / "phase8_figure_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(manifest["figures"]) == 4
    for record in manifest["figures"].values():
        assert record["format"] == "vector_pdf"
        assert len(record["sha256"]) == 64
        for source in record["sources"]:
            assert not Path(source["path"]).is_absolute()
            assert len(source["sha256"]) == 64


def test_phase8_has_no_raster_publication_figures() -> None:
    raster_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    assert [
        path
        for path in (ROOT / "paper" / "figures").iterdir()
        if path.suffix.lower() in raster_suffixes
    ] == []
