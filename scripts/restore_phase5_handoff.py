"""Safely restore a Phase 5 timebox handoff into Experiments/phase5."""

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

from src.experiments.phase5_common import Phase5Paths, load_phase5_config  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_restored(paths: Phase5Paths) -> dict[str, object]:
    config = load_phase5_config()
    status_path = paths.child("manifests", "handoff_status.json")
    if not status_path.is_file():
        raise ValueError("Phase 5 handoff has no status manifest")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    source_commit = str(status.get("source_commit", status.get("commit", "")))
    if source_commit:
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
            cwd=paths.root,
            check=False,
        )
        if ancestry.returncode:
            raise ValueError("Phase 5 handoff source is not an ancestor of checked-out code")
    raw_path = paths.child("raw", "session_traces.jsonl")
    rows = []
    if raw_path.is_file():
        rows = [
            json.loads(line)
            for line in raw_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    pairs = [(str(row["session_id"]), str(row["condition"])) for row in rows]
    if len(pairs) != len(set(pairs)):
        raise ValueError("Phase 5 handoff contains duplicate session-condition rows")
    expected = (
        int(config["dataset"]["validation_sessions"])
        + int(config["dataset"]["test_sessions"])
    ) * 4
    if len(rows) > expected:
        raise ValueError("Phase 5 handoff contains too many session rows")
    return {
        "source_commit": source_commit or "legacy-compatible",
        "session_rows": f"{len(rows)}/{expected}",
        "remaining_session_rows": expected - len(rows),
    }


def restore(source: Path, expected_sha256: str) -> None:
    if sha256(source) != expected_sha256.lower():
        raise ValueError("Phase 5 handoff SHA-256 mismatch")
    paths = Phase5Paths.create(load_phase5_config())
    with tarfile.open(source, "r:gz") as archive:
        for member in archive.getmembers():
            if member.issym() or member.islnk() or not member.name.startswith("phase5/"):
                raise ValueError(f"unsafe Phase 5 handoff member: {member.name}")
            destination = (
                paths.runtime / Path(member.name).relative_to("phase5")
            ).resolve()
            destination.relative_to(paths.runtime)
        archive.extractall(paths.runtime.parent, filter="data")
    print(json.dumps(_verify_restored(paths), indent=2, sort_keys=True))
    print("PHASE 5 HANDOFF RESTORE: FINISHED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    restore(args.archive, args.sha256)
