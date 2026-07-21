"""Safely restore and validate a resumable Phase 3 Lightning handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_TOP_LEVEL = {"Experiments", "phase3-status.json"}
PROHIBITED_SUFFIXES = {".bin", ".env", ".pt", ".pth", ".safetensors"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _validate_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    if not members:
        raise ValueError("handoff archive is empty")
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe archive path: {member.name}")
        if not path.parts or path.parts[0] not in ALLOWED_TOP_LEVEL:
            raise ValueError(f"unexpected archive member: {member.name}")
        if member.issym() or member.islnk() or member.isdev():
            raise ValueError(f"links and devices are prohibited: {member.name}")
        if path.suffix.lower() in PROHIBITED_SUFFIXES:
            raise ValueError(f"model weights or secrets are prohibited: {member.name}")
    return members


def _status_from_archive(archive: tarfile.TarFile) -> dict[str, Any]:
    member = archive.getmember("phase3-status.json")
    handle = archive.extractfile(member)
    if handle is None:
        raise ValueError("phase3-status.json is unreadable")
    return json.loads(handle.read().decode("utf-8"))


def _assert_source_is_ancestor(source_commit: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"handoff commit {source_commit} is not an ancestor of the checked-out code"
        )


def _validate_restored(status: dict[str, Any]) -> dict[str, Any]:
    runtime = REPOSITORY_ROOT / "Experiments"
    for relative, expected_size in status["files"].items():
        if relative == "README.md":
            continue
        path = runtime / relative
        if not path.is_file() or path.stat().st_size != int(expected_size):
            raise RuntimeError(f"restored file mismatch: {relative}")
        if path.suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"restored JSON is invalid: {relative}") from exc

    observed_rows: dict[str, int] = {}
    chunk_pairs: dict[str, set[tuple[str, str]]] = {}
    for relative, expected_rows in status["jsonl_rows"].items():
        path = runtime / relative
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        observed_rows[relative] = len(rows)
        if len(rows) != int(expected_rows):
            raise RuntimeError(f"JSONL row-count mismatch: {relative}")
        if relative == "raw/E04_adaptive.jsonl":
            pairs = [(row["prompt_id"], row["condition"]) for row in rows]
            if len(pairs) != len(set(pairs)):
                raise RuntimeError(
                    "E04 handoff contains duplicate prompt-condition pairs"
                )
        parts = PurePosixPath(relative).parts
        if len(parts) == 4 and parts[:2] == ("raw", "chunks"):
            experiment_id = parts[2]
            pairs = chunk_pairs.setdefault(experiment_id, set())
            for row in rows:
                pair = (str(row["prompt_id"]), str(row["condition"]))
                if pair in pairs:
                    raise RuntimeError(
                        f"{experiment_id} handoff contains duplicate "
                        "prompt-condition pairs across chunks"
                    )
                pairs.add(pair)

    precision = json.loads(
        (runtime / "manifests" / "precision-mamba_130m.json").read_text(
            encoding="utf-8"
        )
    )
    if not precision.get("bf16_acceptable"):
        raise RuntimeError("restored BF16 precision gate is not acceptable")
    return {
        "source_commit": status["commit"],
        "e04_rows": observed_rows.get("raw/E04_adaptive.jsonl", 0),
        "e04_pre_detector_reference_rows": 4000,
        "e04_pre_detector_completion_fraction": min(
            1.0, observed_rows.get("raw/E04_adaptive.jsonl", 0) / 4000
        ),
        "consumed_gpu_seconds": status["budget"]["consumed_gpu_seconds"],
        "automatic_status": status["automatic_report"]["status"],
        "stopped_stage": status["automatic_report"].get("stopped_stage"),
        "chunk_rows": {
            experiment_id: len(pairs)
            for experiment_id, pairs in sorted(chunk_pairs.items())
        },
    }


def restore(archive_path: Path, expected_sha256: str | None) -> dict[str, Any]:
    archive_path = archive_path.resolve()
    observed_sha256 = _sha256(archive_path)
    if expected_sha256 and observed_sha256 != expected_sha256.lower():
        raise RuntimeError(
            f"handoff SHA-256 mismatch: {observed_sha256} != {expected_sha256.lower()}"
        )
    runtime = REPOSITORY_ROOT / "Experiments"
    generated = [
        path
        for path in runtime.rglob("*")
        if path.is_file() and path.name != "README.md"
    ]
    if generated:
        raise RuntimeError(
            "Experiments already contains generated state; restore into a fresh clone"
        )
    with tarfile.open(archive_path, "r:gz") as archive:
        members = _validate_members(archive)
        status = _status_from_archive(archive)
        _assert_source_is_ancestor(str(status["commit"]))
        generated_members = [
            member
            for member in members
            if member.name not in {"phase3-status.json", "Experiments/README.md"}
        ]
        archive.extractall(REPOSITORY_ROOT, members=generated_members, filter="data")
    report = {
        "schema_version": 1,
        "archive": archive_path.name,
        "archive_sha256": observed_sha256,
        **_validate_restored(status),
    }
    destination = runtime / "manifests" / "handoff_restore.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    report = restore(args.archive, args.sha256)
    print(json.dumps(report, indent=2))
    print("PHASE 3 HANDOFF RESTORE: FINISHED")


if __name__ == "__main__":
    main()
