"""Submission-facing regression test for Phase 6 ethics compliance."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_phase6_ethics_gate_passes() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/verify_phase6_ethics.py"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "PHASE 6 ETHICS: PASS" in completed.stdout
