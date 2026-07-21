"""Task-specific, deterministic Phase 3 response scoring."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
import textwrap
from collections import Counter
from typing import Any

BLOCKED_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "vars",
}
BLOCKED_NODES = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)


def normalize_code_answer(text: str) -> str:
    match = re.search(r"\b([A-Z]{3,10}-\d{3})\b", text.upper())
    return match.group(1) if match else ""


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def qa_token_f1(prediction: str, references: list[str]) -> float:
    predicted = _tokens(prediction)
    if not predicted:
        return 0.0
    scores = []
    for reference in references:
        expected = _tokens(reference)
        overlap = sum((Counter(predicted) & Counter(expected)).values())
        if not overlap:
            scores.append(0.0)
            continue
        precision = overlap / len(predicted)
        recall = overlap / len(expected)
        scores.append(2 * precision * recall / (precision + recall))
    return max(scores, default=0.0)


def normalize_number(text: str) -> str:
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text.replace("$", ""))
    if not matches:
        return ""
    candidate = matches[-1].replace(",", "")
    try:
        return f"{float(candidate):.12g}"
    except ValueError:
        return ""


def _validate_generated_body(body: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(
            "def _candidate():\n"
            + "\n".join(f"    {line}" for line in body.splitlines())
        )
    except SyntaxError as exc:
        return False, f"syntax:{exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, BLOCKED_NODES):
            return False, f"blocked_node:{type(node).__name__}"
        if isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            return False, f"blocked_name:{node.id}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, f"blocked_dunder:{node.attr}"
    return True, "validated"


def restricted_humaneval(
    row: dict[str, Any], response: str, timeout_seconds: float = 3.0
):
    raw_body = (
        response.removeprefix("```python").removeprefix("```").removesuffix("```")
    )
    body = textwrap.dedent(raw_body).strip()
    valid, reason = _validate_generated_body(body)
    if not valid:
        return 0.0, reason
    indented_body = "\n".join(
        "    " + line if line.strip() else "" for line in body.splitlines()
    )
    source = str(row["function_prompt"]) + indented_body + "\n" + str(row["test_code"])
    source += f"\ncheck({row['entry_point']})\n"
    wrapper = r"""
import json, os, sys
if os.name == "posix":
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
payload = json.loads(sys.stdin.read())
namespace = {"__builtins__": {"abs": abs, "all": all, "any": any, "bool": bool,
"dict": dict, "enumerate": enumerate, "float": float, "int": int, "len": len,
"list": list, "map": map, "max": max, "min": min, "range": range, "reversed": reversed,
"round": round, "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
"zip": zip, "AssertionError": AssertionError, "Exception": Exception, "__import__": __import__}}
exec(payload["source"], namespace, namespace)
"""
    try:
        with tempfile.TemporaryDirectory() as temporary_directory:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", "-c", wrapper],
                input=json.dumps({"source": source}),
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
                cwd=temporary_directory,
            )
    except subprocess.TimeoutExpired:
        return 0.0, "timeout"
    if completed.returncode == 0:
        return 1.0, "passed"
    return 0.0, f"failed:{completed.stderr[-160:]}"


def score_response(row: dict[str, Any], response: str) -> dict[str, Any]:
    scorer = row["scorer"]
    if scorer == "exact_code":
        predicted = normalize_code_answer(response)
        target = normalize_code_answer(str(row["answer"]))
        score, detail = float(predicted == target), predicted
    elif scorer == "qa_token_f1":
        score, detail = qa_token_f1(response, list(row["answer"])), response[:200]
    elif scorer == "gsm8k_numeric":
        predicted = normalize_number(response)
        target = normalize_number(str(row["answer"]))
        score, detail = float(bool(predicted) and predicted == target), predicted
    elif scorer == "humaneval_restricted_pass_at_1":
        score, detail = restricted_humaneval(row, response)
    else:
        raise ValueError(f"unsupported scorer: {scorer}")
    threshold = 0.5 if scorer == "qa_token_f1" else 1.0
    return {
        "task_score": float(score),
        "correct": bool(score >= threshold),
        "score_detail": detail,
        "scorer": scorer,
        "correctness_threshold": threshold,
    }
