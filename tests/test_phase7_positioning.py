"""Submission-facing regression tests for Phase 7 scientific positioning."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_phase7_positioning_gate_passes() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/verify_phase7_positioning.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "PHASE 7 POSITIONING: PASS" in completed.stdout


def test_phase7_literature_audit_forbids_unverified_priority_claims() -> None:
    audit = json.loads(
        (ROOT / "results" / "foundation" / "phase7_literature_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(audit["primary_sources"]) >= 5
    assert "no general first-study claim" in audit["result"]
    assert audit["prohibited_claim"].startswith("This is the first")


def test_raw_research_archives_are_physically_outside_repository() -> None:
    archives = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not ({".git", ".venv"} & set(path.relative_to(ROOT).parts))
        and (path.suffix == ".zip" or path.name.endswith(".tar.gz"))
    ]
    assert archives == []
