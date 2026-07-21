"""Safely restore a Phase 4 timebox handoff into Experiments/phase4."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.phase4_common import (  # noqa: E402
    Phase4Paths,
    load_phase4_config,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_restored(paths: Phase4Paths) -> dict[str, object]:
    config = load_phase4_config()
    status = json.loads(
        paths.child("manifests", "handoff_status.json").read_text(encoding="utf-8")
    )
    source_commit = str(status.get("source_commit", ""))
    if len(source_commit) != 40:
        raise ValueError("Phase 4 handoff has no immutable source commit")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
        cwd=paths.root,
        check=False,
    )
    if ancestry.returncode:
        raise ValueError("Phase 4 handoff source is not an ancestor of checked-out code")

    whitebox = json.loads(
        paths.child("attacks", "whitebox_joint.json").read_text(encoding="utf-8")
    )
    observed_attacks = {
        (float(item["lambda"]), int(item["seed"])) for item in whitebox
    }
    expected_attacks = {
        (float(weight), int(seed))
        for weight in config["white_box"]["lambda_grid"]
        for seed in config["seeds"]
    }
    if observed_attacks != expected_attacks or len(whitebox) != len(expected_attacks):
        raise ValueError("Phase 4 handoff does not contain the exact white-box grid")

    raw_path = paths.child("raw", "phase4_confirmatory.jsonl")
    rows = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pairs = [(str(row["prompt_id"]), str(row["condition"])) for row in rows]
    if len(pairs) != len(set(pairs)):
        raise ValueError("Phase 4 handoff contains duplicate prompt-condition rows")
    conditions = {
        "clean", "blackbox_transfer", "blackbox_query", "whitebox_joint"
    }
    if not set(condition for _, condition in pairs).issubset(conditions):
        raise ValueError("Phase 4 handoff contains an unknown condition")
    expected_rows = int(config["dataset"]["frozen_test_examples"]) * len(conditions)
    if not 0 < len(rows) <= expected_rows:
        raise ValueError("Phase 4 handoff has an invalid confirmatory row count")
    return {
        "source_commit": source_commit,
        "whitebox_attacks": f"{len(whitebox)}/{len(expected_attacks)}",
        "confirmatory_rows": f"{len(rows)}/{expected_rows}",
        "remaining_rows": expected_rows - len(rows),
    }


def restore(source: Path, expected_sha256: str) -> None:
    if sha256(source) != expected_sha256.lower():
        raise ValueError("Phase 4 handoff SHA-256 mismatch")
    paths = Phase4Paths.create(load_phase4_config())
    with tarfile.open(source, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            if member.issym() or member.islnk() or not member.name.startswith("phase4/"):
                raise ValueError(f"unsafe Phase 4 handoff member: {member.name}")
            destination = (paths.runtime / Path(member.name).relative_to("phase4")).resolve()
            destination.relative_to(paths.runtime)
        archive.extractall(paths.runtime.parent, filter="data")
    report = _verify_restored(paths)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("PHASE 4 HANDOFF RESTORE: FINISHED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    restore(args.archive, args.sha256)
