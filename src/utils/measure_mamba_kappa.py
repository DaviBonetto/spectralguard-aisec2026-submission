"""Measure the two condition-number notions confused in the legacy paper."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import MambaForCausalLM


def measure(model_id: str, revision: str) -> dict[str, object]:
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("revision must be a 40-character lowercase commit SHA")
    model = MambaForCausalLM.from_pretrained(model_id, revision=revision)
    rows: list[dict[str, object]] = []
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not name.endswith("A_log"):
            continue
        values = parameter.detach().to(dtype=torch.float64, device="cpu")
        digest.update(values.numpy().tobytes())
        abs_a = torch.exp(values).abs()
        minimum = float(abs_a.min())
        maximum = float(abs_a.max())
        rows.append(
            {
                "parameter": name,
                "shape": list(values.shape),
                "abs_A_min": minimum,
                "abs_A_max": maximum,
                "A_matrix_condition_2": maximum / minimum,
                "eigenvector_condition_kappa_2": 1.0,
            }
        )
    if not rows:
        raise RuntimeError("no A_log parameters were found")
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "model_revision": revision,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "A_log_tensor_count": len(rows),
        "A_log_sha256": digest.hexdigest(),
        "definition": (
            "The theorem's kappa is cond_2(V), where V diagonalizes A. "
            "Mamba A and exp(delta*A) are real diagonal operators, so V=I and kappa=1 exactly."
        ),
        "global_abs_A_min": min(float(row["abs_A_min"]) for row in rows),
        "global_abs_A_max": max(float(row["abs_A_max"]) for row in rows),
        "A_matrix_condition_2_min": min(
            float(row["A_matrix_condition_2"]) for row in rows
        ),
        "A_matrix_condition_2_max": max(
            float(row["A_matrix_condition_2"]) for row in rows
        ),
        "eigenvector_condition_kappa_2": 1.0,
        "layers": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = measure(args.model_id, args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
