"""Official HiSPA replay and discrete gradient-guided universal attacks."""

from __future__ import annotations

import json
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch

from src.experiments.phase3_common import Phase3Paths
from src.experiments.phase3_prompts import split_attack_anchor


def load_official_trigger_pool(paths: Phase3Paths) -> list[dict[str, Any]]:
    """Load the official labeled HiSPA/benign trigger pool from its hashed snapshot."""
    source = paths.child("datasets", "official_hispa", "Mamba_triggers_scores.json")
    if not source.is_file():
        raise FileNotFoundError("official HiSPA snapshot must be materialized first")
    payload = json.loads(source.read_text(encoding="utf-8"))
    records = []
    for index, (text, values) in enumerate(payload.items()):
        label, score = values
        records.append(
            {
                "official_index": index,
                "text": text,
                "is_hispa": bool(label),
                "official_score": float(score),
            }
        )
    if not any(row["is_hispa"] for row in records) or not any(
        not row["is_hispa"] for row in records
    ):
        raise ValueError("official trigger pool lacks one class")
    return records


def official_attack_families(
    paths: Phase3Paths,
    count: int = 16,
    *,
    tokenizer: Any | None = None,
) -> list[dict[str, Any]]:
    pool = load_official_trigger_pool(paths)
    attacks = [row for row in pool if row["is_hispa"]]
    if tokenizer is not None:
        benign_lengths = {
            len(_token_ids(tokenizer, row["text"]))
            for row in pool
            if not row["is_hispa"] and row["text"].strip()
        }
        attacks = [
            row
            for row in attacks
            if len(_token_ids(tokenizer, row["text"])) in benign_lengths
        ]
    if len(attacks) < count:
        raise ValueError(
            f"official snapshot exposes only {len(attacks)} HiSPA triggers"
        )
    return [
        {**row, "candidate_id": f"official_{index:02d}"}
        for index, row in enumerate(attacks[:count])
    ]


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def match_benign_control(
    loaded: Any,
    attack_text: str,
    *,
    candidates: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Match a benign official trigger by exact token length then standalone NLL."""
    tokenizer = loaded.tokenizer
    attack_ids = _token_ids(tokenizer, attack_text)
    benign = [row for row in candidates if not row["is_hispa"] and row["text"].strip()]
    exact = [
        row
        for row in benign
        if len(_token_ids(tokenizer, row["text"])) == len(attack_ids)
    ]
    neutral_words = (
        " note",
        " context",
        " reference",
        " information",
        " statement",
        " benign",
    )
    neutral_ids = []
    for word in neutral_words:
        ids = _token_ids(tokenizer, word)
        if len(ids) == 1 and _token_ids(tokenizer, tokenizer.decode(ids)) == ids:
            neutral_ids.append(ids[0])
    for offset in range(len(neutral_ids)):
        ids = [
            neutral_ids[(offset + index) % len(neutral_ids)]
            for index in range(len(attack_ids))
        ]
        text = tokenizer.decode(ids)
        if _token_ids(tokenizer, text) == ids:
            exact.append(
                {
                    "text": text,
                    "official_index": None,
                    "is_hispa": False,
                    "control_source": "generated_neutral_token_cycle",
                }
            )
    if not exact:
        raise RuntimeError(
            f"could not construct a benign {len(attack_ids)}-token control"
        )

    def standalone_nll(text: str) -> float:
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(loaded.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = loaded.model(
                **encoded, labels=encoded["input_ids"], use_cache=False
            )
        return float(output.loss.detach().float().cpu())

    attack_nll = standalone_nll(attack_text)
    ranked = sorted(
        (
            (abs(candidate_nll - attack_nll), candidate_nll, row)
            for row in exact
            for candidate_nll in (standalone_nll(row["text"]),)
        ),
        key=lambda item: (
            item[0],
            item[2]["official_index"] is None,
            item[2]["official_index"] or -1,
        ),
    )
    difference, control_nll, selected = ranked[0]
    return {
        "attack_text": attack_text,
        "control_text": selected["text"],
        "token_length": len(attack_ids),
        "attack_nll": attack_nll,
        "control_nll": control_nll,
        "absolute_nll_difference": difference,
        "control_official_index": selected["official_index"],
        "control_source": selected.get(
            "control_source", "official_hispa_snapshot_benign"
        ),
    }


@dataclass
class DifferentiableSpectrum:
    """Differentiable Mamba spectral summaries captured from mixer inputs."""

    model: Any

    def __post_init__(self) -> None:
        self.handles: list[Any] = []
        self.layer_values: list[torch.Tensor] = []

    def _layers(self) -> list[Any]:
        candidates = (
            getattr(getattr(self.model, "backbone", None), "layers", None),
            getattr(
                getattr(getattr(self.model, "model", None), "backbone", None),
                "layers",
                None,
            ),
        )
        for candidate in candidates:
            if candidate is not None:
                return list(candidate)
        raise TypeError("could not locate Mamba backbone layers")

    def _hook(self, mixer: Any):
        def callback(_module: Any, args: tuple[Any, ...]) -> None:
            input_states = args[0]
            projected = mixer.in_proj(input_states).transpose(1, 2)
            hidden_states, _ = projected.chunk(2, dim=1)
            length = hidden_states.shape[-1]
            hidden_states = mixer.act(mixer.conv1d(hidden_states)[..., :length])
            parameters = mixer.x_proj(hidden_states.transpose(1, 2))
            time_step, _, _ = torch.split(
                parameters,
                [mixer.time_step_rank, mixer.ssm_state_size, mixer.ssm_state_size],
                dim=-1,
            )
            delta = torch.nn.functional.softplus(mixer.dt_proj(time_step).float())
            decay = torch.exp(mixer.A_log.float())
            # max(exp(-delta * decay)) == exp(-min(delta * decay)). Reduce one
            # state channel at a time so the 2.8B white-box graph never
            # materializes the enormous [batch, seq, inner, state] tensor.
            minimum_decay = decay.amin(dim=-1)
            minimum = (delta * minimum_decay[None, None, :]).amin(dim=-1)
            self.layer_values.append(torch.exp(-minimum))

        return callback

    def install(self) -> None:
        if self.handles:
            raise RuntimeError("differentiable collector already installed")
        for layer in self._layers():
            self.handles.append(
                layer.mixer.register_forward_pre_hook(self._hook(layer.mixer))
            )

    def clear(self) -> None:
        self.layer_values.clear()

    def features(self) -> torch.Tensor:
        if not self.layer_values:
            raise RuntimeError("no differentiable spectral features captured")
        return torch.stack([value.mean() for value in self.layer_values])

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@contextmanager
def frozen_parameters(model: Any):
    original = [parameter.requires_grad for parameter in model.parameters()]
    try:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, flag in zip(model.parameters(), original, strict=True):
            parameter.requires_grad_(flag)


def _target_text(row: dict[str, Any]) -> str:
    if isinstance(row["answer"], list):
        return str(row["answer"][0])
    return str(row["answer"])


def _attack_parts(tokenizer: Any, row: dict[str, Any], trigger_ids: list[int]):
    prefix, anchor, suffix = split_attack_anchor(row)
    prefix_ids = _token_ids(tokenizer, prefix)
    suffix_ids = _token_ids(tokenizer, anchor + suffix)
    target_ids = _token_ids(tokenizer, _target_text(row))
    input_ids = prefix_ids + trigger_ids + suffix_ids + target_ids
    labels = [-100] * (
        len(prefix_ids) + len(trigger_ids) + len(suffix_ids)
    ) + target_ids
    return input_ids, labels, slice(len(prefix_ids), len(prefix_ids) + len(trigger_ids))


def _detector_hazard(features: torch.Tensor, detector: dict[str, Any] | None):
    if detector is None:
        return features.new_zeros(())
    mean = features.new_tensor(detector["feature_mean"])
    scale = features.new_tensor(detector["feature_scale"]).clamp_min(1e-8)
    weights = features.new_tensor(detector["weights"])
    bias = features.new_tensor(float(detector["bias"]))
    return torch.sigmoid(((features - mean) / scale * weights).sum() + bias)


def attack_objective_and_gradient(
    loaded: Any,
    rows: list[dict[str, Any]],
    trigger_ids: list[int],
    *,
    rho_weight: float,
    task_weight: float,
    hazard_weight: float,
    detector: dict[str, Any] | None,
) -> tuple[float, torch.Tensor, dict[str, float]]:
    """Return the universal objective and mean trigger-embedding gradient."""
    if loaded.spec["architecture"] != "mamba":
        raise TypeError(
            "gradient-guided spectral optimization requires a Mamba source model"
        )
    embedding = loaded.model.get_input_embeddings()
    collector = DifferentiableSpectrum(loaded.model)
    collector.install()
    gradients = []
    values = []
    losses = []
    rhos = []
    hazards = []
    try:
        with frozen_parameters(loaded.model):
            for row in rows:
                collector.clear()
                ids, labels, trigger_slice = _attack_parts(
                    loaded.tokenizer, row, trigger_ids
                )
                token_tensor = torch.tensor([ids], device=loaded.device)
                label_tensor = torch.tensor([labels], device=loaded.device)
                inputs = embedding(token_tensor).detach().requires_grad_(True)
                output = loaded.model(
                    inputs_embeds=inputs,
                    labels=label_tensor,
                    use_cache=False,
                    return_dict=True,
                )
                features = collector.features()
                hazard = _detector_hazard(features, detector)
                objective = (
                    float(rho_weight) * features.mean()
                    - float(task_weight) * output.loss
                    + float(hazard_weight) * hazard
                )
                objective.backward()
                gradients.append(inputs.grad[0, trigger_slice, :].detach())
                values.append(float(objective.detach().cpu()))
                losses.append(float(output.loss.detach().cpu()))
                rhos.append(float(features.mean().detach().cpu()))
                hazards.append(float(hazard.detach().cpu()))
    finally:
        collector.remove()
    return (
        float(np.mean(values)),
        torch.stack(gradients).mean(dim=0),
        {
            "task_loss": float(np.mean(losses)),
            "rho_mean": float(np.mean(rhos)),
            "hazard": float(np.mean(hazards)),
        },
    )


def _allowed_vocabulary(
    tokenizer: Any, device: torch.device, embedding_rows: int
) -> torch.Tensor:
    allowed = []
    special = set(tokenizer.all_special_ids)
    for token_id in range(min(len(tokenizer), int(embedding_rows))):
        if token_id in special:
            continue
        text = tokenizer.decode([token_id])
        if text.strip() and text.isprintable():
            allowed.append(token_id)
    if len(allowed) < 100:
        raise RuntimeError("tokenizer exposes too few safe attack candidates")
    return torch.tensor(allowed, device=device, dtype=torch.long)


def optimize_universal_trigger(
    loaded: Any,
    calibration_rows: list[dict[str, Any]],
    *,
    seed: int,
    trigger_tokens: int,
    steps: int,
    rho_weight: float,
    task_weight: float,
    hazard_weight: float = 0.0,
    detector: dict[str, Any] | None = None,
    candidates_per_step: int = 32,
    progress_label: str | None = None,
) -> dict[str, Any]:
    """Optimize a discrete universal trigger with HotFlip embedding gradients."""
    if not calibration_rows:
        raise ValueError("calibration rows are empty")
    rng = random.Random(int(seed))
    embedding_weight = loaded.model.get_input_embeddings().weight.detach()
    allowed = _allowed_vocabulary(
        loaded.tokenizer, loaded.device, embedding_weight.shape[0]
    )
    initial = rng.sample(allowed.detach().cpu().tolist(), int(trigger_tokens))
    trigger_ids = list(initial)
    history = []
    for step in range(int(steps)):
        value, gradient, diagnostics = attack_objective_and_gradient(
            loaded,
            calibration_rows,
            trigger_ids,
            rho_weight=rho_weight,
            task_weight=task_weight,
            hazard_weight=hazard_weight,
            detector=detector,
        )
        position = step % len(trigger_ids)
        scores = embedding_weight[allowed] @ gradient[position]
        order = torch.argsort(scores)
        replacement = None
        for candidate_index in order[: int(candidates_per_step)]:
            candidate = int(allowed[candidate_index])
            if candidate != trigger_ids[position]:
                replacement = candidate
                break
        if replacement is None:
            raise RuntimeError("HotFlip could not find a replacement token")
        trigger_ids[position] = replacement
        history.append(
            {
                "step": step,
                "position": position,
                "objective": value,
                **diagnostics,
            }
        )
        if progress_label:
            print(
                f"{progress_label}: {step + 1}/{int(steps)} "
                f"({100.0 * (step + 1) / int(steps):.1f}%)",
                flush=True,
            )
    final_value, _, final_diagnostics = attack_objective_and_gradient(
        loaded,
        calibration_rows,
        trigger_ids,
        rho_weight=rho_weight,
        task_weight=task_weight,
        hazard_weight=hazard_weight,
        detector=detector,
    )
    trigger_text = loaded.tokenizer.decode(trigger_ids, skip_special_tokens=True)
    if not trigger_text.strip() or not math.isfinite(final_value):
        raise RuntimeError("optimized trigger is empty or non-finite")
    return {
        "schema_version": 1,
        "algorithm": "discrete_hotflip_universal",
        "seed": int(seed),
        "trigger_token_ids": trigger_ids,
        "trigger_text": trigger_text,
        "trigger_tokens": len(trigger_ids),
        "steps": int(steps),
        "rho_weight": float(rho_weight),
        "task_weight": float(task_weight),
        "hazard_weight": float(hazard_weight),
        "candidates_per_step": int(candidates_per_step),
        "detector_aware": detector is not None,
        "final_objective": final_value,
        "final_diagnostics": final_diagnostics,
        "history": history,
    }
