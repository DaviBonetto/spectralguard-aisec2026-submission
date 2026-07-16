"""Frozen four-task Phase 3 benchmark and official-artifact snapshots."""

from __future__ import annotations

import hashlib
import json
import random
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from huggingface_hub import hf_hub_download

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

SUBJECT_WORDS = tuple(
    "amber birch cedar delta ember fjord garnet harbor indigo juniper kestrel lagoon maple "
    "nebula onyx prairie quartz raven saffron tundra umber violet willow xenon yarrow zephyr".split()
)
CODE_WORDS = tuple(
    "ALPHA BRAVO CORAL DUSK ECHO FROST GLOW HAZE IRIS JADE KITE LUMEN MINT NOVA OPAL PLUME "
    "QUILL RUNE SOLAR TIDE ULTRA VALE WAVE XYLO YONDER ZENITH".split()
)


def _stable_id(task: str, source_id: str) -> str:
    digest = hashlib.sha256(f"phase3-v2:{task}:{source_id}".encode()).hexdigest()[:16]
    return f"p3-{task}-{digest}"


def _split_sequence(task: str, config: dict[str, Any]) -> list[str]:
    counts = config["dataset"]["task_splits"][task]
    return [
        name
        for name in ("calibration", "validation", "test")
        for _ in range(int(counts[name]))
    ]


def _generation(config: dict[str, Any], task: str) -> dict[str, Any]:
    payload = config["generation"][task]
    return {
        "scorer": payload["scorer"],
        "max_new_tokens": int(payload["max_new_tokens"]),
        "max_input_tokens": int(config["protocol"]["max_input_tokens"]),
    }


def generate_associative_recall(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate the pre-registered synthetic memory task."""
    task = "associative_recall"
    seed = int(config["seeds"][0])
    rng = random.Random(seed)
    facts_per_prompt = int(config["dataset"][task]["facts_per_prompt"])
    if int(config["dataset"][task]["template_count"]) != len(TEMPLATES):
        raise ValueError("associative template_count drifted")
    rows = []
    for index, split in enumerate(_split_sequence(task, config)):
        subjects = rng.sample(SUBJECT_WORDS, facts_per_prompt)
        codes = rng.sample(CODE_WORDS, facts_per_prompt)
        suffixes = rng.sample(range(100, 1000), facts_per_prompt)
        pairs = [
            (f"{subject}-{index:04d}", f"{code}-{suffix}")
            for subject, code, suffix in zip(subjects, codes, suffixes, strict=True)
        ]
        target_subject, target_code = pairs[rng.randrange(facts_per_prompt)]
        facts = [f"- {subject} is assigned code {code}." for subject, code in pairs]
        rng.shuffle(facts)
        question = f"What code is assigned to {target_subject}?"
        template_id = index % len(TEMPLATES)
        prompt = TEMPLATES[template_id].format(
            facts="\n".join(facts),
            question=question,
            question_lower=question[0].lower() + question[1:],
        )
        rows.append(
            {
                "prompt_id": _stable_id(task, str(index)),
                "task": task,
                "source": "synthetic_associative_recall_v2",
                "source_id": str(index),
                "split": split,
                "template_id": template_id,
                "clean_prompt": prompt,
                "question": question,
                "answer": target_code,
                **_generation(config, task),
            }
        )
    return rows


def _shuffle_take(
    records: list[dict[str, Any]], task: str, config: dict[str, Any], offset: int
):
    records = list(records)
    random.Random(int(config["seeds"][0]) + offset).shuffle(records)
    required = len(_split_sequence(task, config))
    if len(records) < required:
        raise ValueError(f"{task} exposes {len(records)} rows, requires {required}")
    return list(zip(records[:required], _split_sequence(task, config), strict=True))


def _longbench_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    task = "longbench_qasper"
    source = config["dataset"]["sources"]["longbench"]
    archive_path = hf_hub_download(
        source["repo_id"],
        source["file"],
        repo_type="dataset",
        revision=source["revision"],
    )
    with ZipFile(archive_path) as archive:
        member = source["member"]
        if member not in archive.namelist() or ".." in Path(member).parts:
            raise ValueError("pinned LongBench member is absent or unsafe")
        with archive.open(member) as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    rows = []
    for record, split in _shuffle_take(records, task, config, 11):
        source_id = str(record["_id"])
        question = str(record["input"])
        rows.append(
            {
                "prompt_id": _stable_id(task, source_id),
                "task": task,
                "source": f"{source['repo_id']}@{source['revision']}",
                "source_id": source_id,
                "split": split,
                "template_id": 0,
                "clean_prompt": (
                    "Read the research document and answer using only supported information.\n\n"
                    f"DOCUMENT:\n{record['context']}\n\nQUESTION: {question}\nCONCISE ANSWER:"
                ),
                "question": f"QUESTION: {question}",
                "answer": list(record["answers"]),
                **_generation(config, task),
            }
        )
    return rows


def _parquet_records(source: dict[str, Any]) -> list[dict[str, Any]]:
    from datasets import load_dataset

    url = (
        f"https://huggingface.co/datasets/{source['repo_id']}/resolve/"
        f"{source['revision']}/{source['file']}"
    )
    return list(load_dataset("parquet", data_files={"test": url}, split="test"))


def _gsm8k_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    task = "gsm8k"
    source = config["dataset"]["sources"]["gsm8k"]
    rows = []
    for index, (record, split) in enumerate(
        _shuffle_take(_parquet_records(source), task, config, 23)
    ):
        answer = str(record["answer"])
        match = re.search(r"####\s*([^\n]+)", answer)
        if not match:
            raise ValueError(f"GSM8K row {index} lacks final answer marker")
        question = str(record["question"])
        rows.append(
            {
                "prompt_id": _stable_id(task, str(index)),
                "task": task,
                "source": f"{source['repo_id']}@{source['revision']}",
                "source_id": str(index),
                "split": split,
                "template_id": 0,
                "clean_prompt": (
                    "Solve the following grade-school problem. Show concise reasoning and end with "
                    f"FINAL: <number>.\n\nPROBLEM: {question}\nSOLUTION:"
                ),
                "question": f"PROBLEM: {question}",
                "answer": match.group(1).strip(),
                **_generation(config, task),
            }
        )
    return rows


def _humaneval_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    task = "humaneval"
    source = config["dataset"]["sources"]["humaneval"]
    rows = []
    for record, split in _shuffle_take(_parquet_records(source), task, config, 37):
        source_id = str(record["task_id"])
        question = f"Complete the function {record['entry_point']} without changing its signature."
        rows.append(
            {
                "prompt_id": _stable_id(task, source_id),
                "task": task,
                "source": f"{source['repo_id']}@{source['revision']}",
                "source_id": source_id,
                "split": split,
                "template_id": 0,
                "clean_prompt": (
                    "Complete the Python function. Return only the indented function body.\n\n"
                    f"{record['prompt']}\n{question}\nBODY:"
                ),
                "question": question,
                "answer": str(record["canonical_solution"]),
                "function_prompt": str(record["prompt"]),
                "entry_point": str(record["entry_point"]),
                "test_code": str(record["test"]),
                **_generation(config, task),
            }
        )
    return rows


def generate_four_task_benchmark(config: dict[str, Any]) -> list[dict[str, Any]]:
    return (
        generate_associative_recall(config)
        + _longbench_rows(config)
        + _gsm8k_rows(config)
        + _humaneval_rows(config)
    )


def validate_dataset(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    expected = int(config["dataset"]["total_examples"])
    if len(rows) != expected:
        raise ValueError(f"expected {expected} rows, received {len(rows)}")
    ids = [row["prompt_id"] for row in rows]
    prompts = [row["clean_prompt"] for row in rows]
    if len(ids) != len(set(ids)) or len(prompts) != len(set(prompts)):
        raise ValueError("duplicate prompt ID or clean prompt")
    counts = {
        task: {
            split: sum(row["task"] == task and row["split"] == split for row in rows)
            for split in ("calibration", "validation", "test")
        }
        for task in config["dataset"]["task_splits"]
    }
    expected_counts = {
        task: {split: int(value) for split, value in split_counts.items()}
        for task, split_counts in config["dataset"]["task_splits"].items()
    }
    if counts != expected_counts:
        raise ValueError(f"task/split mismatch: {counts} != {expected_counts}")
    return {
        "schema_version": 2,
        "passed": True,
        "row_count": len(rows),
        "task_split_counts": counts,
        "unique_prompt_ids": len(set(ids)),
        "unique_prompts": len(set(prompts)),
    }


def materialize_dataset(
    config: dict[str, Any], paths: Phase3Paths
) -> tuple[Path, Path]:
    rows = generate_four_task_benchmark(config)
    audit = validate_dataset(rows, config)
    dataset_path = atomic_write_jsonl(
        paths.child("datasets", "phase3_four_tasks.jsonl"), rows
    )
    audit["dataset_sha256"] = sha256_file(dataset_path)
    audit_path = atomic_write_json(
        paths.child("manifests", "dataset_audit.json"), audit
    )
    return dataset_path, audit_path


def request_bytes(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "SpectralGuard-Phase3/2.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _snapshot(config: dict[str, Any], paths: Phase3Paths, key: str) -> dict[str, Any]:
    specification = config["dataset"][key]
    destination_root = paths.child("datasets", key)
    destination_root.mkdir(parents=True, exist_ok=True)
    records = []
    for relative_name in specification["required_paths"]:
        encoded = "/".join(
            urllib.parse.quote(part, safe="") for part in relative_name.split("/")
        )
        content = request_bytes(f"{specification['api_base']}/file/{encoded}")
        destination = (destination_root / relative_name).resolve()
        destination.relative_to(destination_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(destination)
        observed_hash = sha256_file(destination)
        expected_hash = specification["expected_sha256"][relative_name]
        if observed_hash != expected_hash:
            destination.unlink(missing_ok=True)
            raise ValueError(
                f"official artifact hash drift for {key}/{relative_name}: "
                f"{observed_hash} != {expected_hash}"
            )
        records.append(
            {"path": relative_name, "size": len(content), "sha256": observed_hash}
        )
    return {
        "repository_id": specification["repo_id"],
        "api_base": specification["api_base"],
        "files": sorted(records, key=lambda item: item["path"]),
    }


def snapshot_official_artifacts(config: dict[str, Any], paths: Phase3Paths) -> Path:
    manifest = {
        "schema_version": 2,
        "official_hispa": _snapshot(config, paths, "official_hispa"),
        "official_clasp": _snapshot(config, paths, "official_clasp"),
    }
    return atomic_write_json(
        paths.child("manifests", "official_artifacts.json"), manifest
    )


def load_frozen_dataset(paths: Phase3Paths) -> list[dict[str, Any]]:
    return read_jsonl(paths.child("datasets", "phase3_four_tasks.jsonl"))
