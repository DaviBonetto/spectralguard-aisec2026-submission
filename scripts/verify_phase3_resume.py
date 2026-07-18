"""Verify a restored Phase 3 handoff before enabling a paid GPU runtime."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNTIME = REPOSITORY_ROOT / "Experiments"
REQUIRED_COMPLETED_STAGES = {
    "automated_pre_gpu_audit",
    "frozen_assets_and_h100_preflight",
    "smoke",
    "precision_gate",
    "official_hispa_selection",
    "attack_gradient_guided",
    "attack_adaptive_v2",
    "batch_equivalence_gate",
    "E04_adaptive_pre_detector",
    "fit_detector",
    "attack_adaptive_v3",
    "attack_pareto",
    "E04_adaptive_final",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing required resume artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def verify(*, expected_sha256: str, expected_e01_rows: int) -> dict[str, Any]:
    restore = _read_json(RUNTIME / "manifests" / "handoff_restore.json")
    observed_sha256 = str(restore.get("archive_sha256", "")).lower()
    if observed_sha256 != expected_sha256.lower():
        raise RuntimeError(
            f"restored handoff SHA-256 mismatch: {observed_sha256} != "
            f"{expected_sha256.lower()}"
        )

    automatic = _read_json(RUNTIME / "manifests" / "automatic_phase3_run.json")
    if automatic.get("status") != "TIMEBOX_HANDOFF":
        raise RuntimeError(f"unexpected prior run status: {automatic.get('status')}")
    if automatic.get("stopped_stage") != "E01_main_mamba":
        raise RuntimeError(
            f"unexpected stopped stage: {automatic.get('stopped_stage')}"
        )
    completed = {
        str(stage["name"])
        for stage in automatic.get("stages", [])
        if stage.get("status") == "PASS"
    }
    missing_stages = sorted(REQUIRED_COMPLETED_STAGES - completed)
    if missing_stages:
        raise RuntimeError(f"handoff is missing completed stages: {missing_stages}")

    chunk_directory = RUNTIME / "raw" / "chunks" / "E01_main_mamba"
    chunks = sorted(chunk_directory.glob("*.jsonl"))
    if not chunks:
        raise RuntimeError("restored handoff contains no E01_main_mamba chunks")
    e01_rows = sum(_count_jsonl(path) for path in chunks)
    if e01_rows != expected_e01_rows:
        raise RuntimeError(
            f"E01_main_mamba resume count mismatch: {e01_rows} != {expected_e01_rows}"
        )

    e04 = RUNTIME / "raw" / "E04_adaptive.jsonl"
    if not e04.is_file() or _count_jsonl(e04) != 4800:
        raise RuntimeError("completed E04_adaptive evidence is missing or incomplete")

    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(f"tracked worktree is not clean:\n{status}")

    report = {
        "status": "READY_FOR_H100",
        "archive_sha256": observed_sha256,
        "source_commit": restore.get("source_commit"),
        "completed_prior_stages": len(completed),
        "e04_rows": _count_jsonl(e04),
        "e01_main_mamba_rows": e01_rows,
        "e01_main_mamba_total": 6000,
        "e01_main_mamba_completion_fraction": e01_rows / 6000,
    }
    print(json.dumps(report, indent=2))
    print("PHASE 3 RESUME VERIFICATION: READY FOR H100")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-e01-rows", type=int, default=3507)
    args = parser.parse_args()
    verify(
        expected_sha256=args.expected_sha256,
        expected_e01_rows=args.expected_e01_rows,
    )


if __name__ == "__main__":
    main()
