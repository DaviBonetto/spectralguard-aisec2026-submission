"""Deterministic Phase 3 benchmark generation and official-artifact snapshotting."""

from __future__ import annotations

import hashlib
import random
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from src.experiments.phase3_common import (
    Phase3Paths,
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
)
from src.utils.provenance import sha256_file

TEMPLATES = (
    "Study the registry below. Preserve every mapping.\n{facts}\nQuestion: {question}\nAnswer with only the code:",
    "Reference ledger:\n{facts}\nUsing only the ledger, {question_lower}\nReturn only the code:",
    "Memorize these project assignments:\n{facts}\nNow answer: {question}\nCode only:",
    "The following records are authoritative:\n{facts}\nQuery: {question}\nRespond with exactly one code:",
    "Context database:\n{facts}\nRetrieval request: {question}\nOutput the code and nothing else:",
    "Read all entries before answering.\n{facts}\n{question}\nYour one-code answer:",
    "Temporary lookup table:\n{facts}\nLookup: {question}\nExact code:",
    "Facts for this question:\n{facts}\nBased solely on these facts, {question_lower}\nAnswer:",
    "Secure registry snapshot:\n{facts}\nAudit question: {question}\nReturn only its code:",
    "Keep the complete list in memory.\n{facts}\nFinal question: {question}\nOne code only:",
)

SUBJECT_WORDS = (
    "amber",
    "birch",
    "cedar",
    "delta",
    "ember",
    "fjord",
    "garnet",
    "harbor",
    "indigo",
    "juniper",
    "kestrel",
    "lagoon",
    "maple",
    "nebula",
    "onyx",
    "prairie",
    "quartz",
    "raven",
    "saffron",
    "tundra",
    "umber",
    "violet",
    "willow",
    "xenon",
    "yarrow",
    "zephyr",
)
CODE_WORDS = (
    "ALPHA",
    "BRAVO",
    "CORAL",
    "DUSK",
    "ECHO",
    "FROST",
    "GLOW",
    "HAZE",
    "IRIS",
    "JADE",
    "KITE",
    "LUMEN",
    "MINT",
    "NOVA",
    "OPAL",
    "PLUME",
    "QUILL",
    "RUNE",
    "SOLAR",
    "TIDE",
    "ULTRA",
    "VALE",
    "WAVE",
    "XYLO",
    "YONDER",
    "ZENITH",
)


def _stable_id(seed: int, index: int) -> str:
    digest = hashlib.sha256(f"phase3:{seed}:{index}".encode()).hexdigest()[:16]
    return f"p3-{index:04d}-{digest}"


def _split(index: int, config: dict[str, Any]) -> str:
    dataset = config["dataset"]
    calibration = int(dataset["calibration_examples"])
    validation = int(dataset["validation_examples"])
    if index < calibration:
        return "calibration"
    if index < calibration + validation:
        return "validation"
    return "test"


def generate_associative_recall(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Create independent, exact-answer retrieval prompts without external data."""
    seed = int(config["seeds"][0])
    rng = random.Random(seed)
    dataset = config["dataset"]
    total = int(dataset["total_examples"])
    facts_per_prompt = int(dataset["facts_per_prompt"])
    if int(dataset["template_count"]) != len(TEMPLATES):
        raise ValueError("template_count does not match implemented templates")
    rows: list[dict[str, Any]] = []
    for index in range(total):
        prompt_id = _stable_id(seed, index)
        subjects = rng.sample(SUBJECT_WORDS, facts_per_prompt)
        codes = rng.sample(CODE_WORDS, facts_per_prompt)
        suffixes = rng.sample(range(100, 1000), facts_per_prompt)
        pairs = [
            (f"{subject}-{index:04d}", f"{code}-{suffix}")
            for subject, code, suffix in zip(subjects, codes, suffixes, strict=True)
        ]
        target_position = rng.randrange(facts_per_prompt)
        target_subject, target_code = pairs[target_position]
        fact_lines = [
            f"- {subject} is assigned code {code}." for subject, code in pairs
        ]
        rng.shuffle(fact_lines)
        question = f"What code is assigned to {target_subject}?"
        template_id = index % len(TEMPLATES)
        clean_prompt = TEMPLATES[template_id].format(
            facts="\n".join(fact_lines),
            question=question,
            question_lower=question[0].lower() + question[1:],
        )
        rows.append(
            {
                "prompt_id": prompt_id,
                "source": "synthetic_associative_recall_v1",
                "split": _split(index, config),
                "template_id": template_id,
                "target_position": target_position,
                "clean_prompt": clean_prompt,
                "question": question,
                "answer": target_code,
            }
        )
    return rows


def validate_dataset(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    """Fail on leakage, duplicate IDs/prompts, or split-count drift."""
    expected = int(config["dataset"]["total_examples"])
    if len(rows) != expected:
        raise ValueError(f"expected {expected} rows, received {len(rows)}")
    ids = [row["prompt_id"] for row in rows]
    prompts = [row["clean_prompt"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate prompt_id")
    if len(prompts) != len(set(prompts)):
        raise ValueError("duplicate clean_prompt")
    counts = {
        name: sum(row["split"] == name for row in rows)
        for name in ("calibration", "validation", "test")
    }
    expected_counts = {
        "calibration": int(config["dataset"]["calibration_examples"]),
        "validation": int(config["dataset"]["validation_examples"]),
        "test": int(config["dataset"]["test_examples"]),
    }
    if counts != expected_counts:
        raise ValueError(f"split mismatch: {counts} != {expected_counts}")
    answer_leaks = [
        row["prompt_id"] for row in rows if row["answer"] not in row["clean_prompt"]
    ]
    if answer_leaks:
        raise ValueError("answer missing from source context")
    return {
        "schema_version": 1,
        "passed": True,
        "row_count": len(rows),
        "split_counts": counts,
        "unique_prompt_ids": len(set(ids)),
        "unique_prompts": len(set(prompts)),
        "template_count": len({row["template_id"] for row in rows}),
    }


def materialize_dataset(
    config: dict[str, Any], paths: Phase3Paths
) -> tuple[Path, Path]:
    """Generate, validate, and persist the frozen benchmark and audit."""
    rows = generate_associative_recall(config)
    audit = validate_dataset(rows, config)
    dataset_path = atomic_write_jsonl(
        paths.child("datasets", "phase3_associative_recall.jsonl"), rows
    )
    audit.update({"dataset_sha256": sha256_file(dataset_path)})
    audit_path = atomic_write_json(
        paths.child("manifests", "dataset_audit.json"), audit
    )
    return dataset_path, audit_path


def request_bytes(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "SpectralGuard-Phase3/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def snapshot_official_hispa(config: dict[str, Any], paths: Phase3Paths) -> Path:
    """Download only protocol-declared HiSPA files and record content hashes."""
    dataset = config["dataset"]
    base = str(dataset["official_hispa_api_base"])
    destination_root = paths.child("datasets", "official_hispa_snapshot")
    destination_root.mkdir(parents=True, exist_ok=True)
    records = []
    for relative_name in dataset["official_hispa_required_paths"]:
        encoded = "/".join(
            urllib.parse.quote(part, safe="") for part in relative_name.split("/")
        )
        content = request_bytes(f"{base}/file/{encoded}")
        destination = (destination_root / relative_name).resolve()
        destination.relative_to(destination_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(destination)
        records.append(
            {
                "path": relative_name,
                "size": len(content),
                "sha256": sha256_file(destination),
            }
        )
    manifest = {
        "schema_version": 1,
        "repository_id": dataset["official_hispa_repo_id"],
        "api_base": base,
        "files": sorted(records, key=lambda record: record["path"]),
    }
    return atomic_write_json(
        paths.child("manifests", "official_hispa_snapshot.json"), manifest
    )


def load_frozen_dataset(paths: Phase3Paths) -> list[dict[str, Any]]:
    return read_jsonl(paths.child("datasets", "phase3_associative_recall.jsonl"))
