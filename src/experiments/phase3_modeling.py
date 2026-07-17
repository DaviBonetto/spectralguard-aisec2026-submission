"""Pinned-model evaluation and architecture-specific diagnostic extraction."""

from __future__ import annotations

import gc
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch

from src.experiments.phase3_prompts import inject_trigger
from src.experiments.phase3_scoring import normalize_code_answer, score_response


def normalize_answer(text: str) -> str:
    """Normalize an exact-code answer without accepting arbitrary substrings."""
    return normalize_code_answer(text)


def model_spec(config: dict[str, Any], key: str) -> dict[str, Any]:
    matches = [model for model in config["models"] if model["key"] == key]
    if len(matches) != 1:
        raise KeyError(f"unknown or duplicate model key: {key}")
    return matches[0]


def _dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


@dataclass
class LoadedModel:
    spec: dict[str, Any]
    tokenizer: Any
    model: Any
    device: torch.device
    dtype_name: str


def require_mamba_fast_path() -> None:
    """Fail closed instead of silently running the sequential Mamba backend."""
    from transformers.models.mamba import modeling_mamba

    if not modeling_mamba.is_fast_path_available:
        raise RuntimeError(
            "Mamba CUDA fast path is unavailable; run "
            "scripts/install_lightning_mamba_kernels.py before using the H100"
        )


@contextmanager
def load_pinned_model(
    spec: dict[str, Any],
    *,
    dtype_name: str = "bfloat16",
) -> Iterator[LoadedModel]:
    """Load one immutable checkpoint and guarantee GPU-memory cleanup."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "model execution requires CUDA; use dry_run for CPU validation"
        )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if spec["architecture"] == "mamba":
        require_mamba_fast_path()

    token = os.environ.get("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(
        spec["id"], revision=spec["revision"], token=token, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {}
    if spec["architecture"] == "transformer":
        model_kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        spec["id"],
        revision=spec["revision"],
        torch_dtype=_dtype(dtype_name),
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        token=token,
        **model_kwargs,
    )
    model.eval()
    loaded = LoadedModel(
        spec=spec,
        tokenizer=tokenizer,
        model=model,
        device=torch.device("cuda:0"),
        dtype_name=dtype_name,
    )
    try:
        yield loaded
    finally:
        del loaded
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class MambaSpectralCollector:
    """Recompute input-dependent diagonal spectra from mixer inputs.

    The collector registers pre-hooks on Mamba mixers. It deliberately
    reconstructs the selection projection instead of relying on x_proj hooks,
    because fused CUDA kernels bypass the Python x_proj module call.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.handles: list[Any] = []
        self.records: dict[int, list[dict[str, Any]]] = {}
        self.enabled = False

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

    def _hook(self, layer_index: int, mixer: Any):
        def callback(_module: Any, args: tuple[Any, ...]) -> None:
            if not self.enabled:
                return
            input_states = args[0]
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=input_states.device.type,
                    enabled=False,
                ),
            ):
                projected = mixer.in_proj(input_states).transpose(1, 2)
                hidden_states, _gate = projected.chunk(2, dim=1)
                sequence_length = hidden_states.shape[-1]
                hidden_states = mixer.act(
                    mixer.conv1d(hidden_states)[..., :sequence_length]
                )
                parameters = mixer.x_proj(hidden_states.transpose(1, 2))
                time_step, b_vector, c_vector = torch.split(
                    parameters,
                    [mixer.time_step_rank, mixer.ssm_state_size, mixer.ssm_state_size],
                    dim=-1,
                )
                delta = torch.nn.functional.softplus(mixer.dt_proj(time_step).float())
                q = torch.exp(mixer.A_log.float())
                discrete_abs = torch.exp(-delta[:, :, :, None] * q[None, None, :, :])
                rho_channel = discrete_abs.amax(dim=-1)
                rho_global = rho_channel.amax(dim=-1)
                b_norm = torch.linalg.vector_norm(b_vector.float(), dim=-1)
                c_norm = torch.linalg.vector_norm(c_vector.float(), dim=-1)
                for batch_index in range(input_states.shape[0]):
                    rho = rho_global[batch_index]
                    record = {
                        "layer": layer_index,
                        "rho_min": float(rho.amin().cpu()),
                        "rho_mean": float(rho.mean().cpu()),
                        "rho_max": float(rho.amax().cpu()),
                        "rho_std": float(rho.std(unbiased=False).cpu()),
                        "delta_min": float(delta[batch_index].amin().cpu()),
                        "delta_mean": float(delta[batch_index].mean().cpu()),
                        "delta_max": float(delta[batch_index].amax().cpu()),
                        "B_norm_mean": float(b_norm[batch_index].mean().cpu()),
                        "C_norm_mean": float(c_norm[batch_index].mean().cpu()),
                    }
                    self.records.setdefault(batch_index, []).append(record)

        return callback

    def install(self) -> None:
        if self.handles:
            raise RuntimeError("collector already installed")
        for index, layer in enumerate(self._layers()):
            mixer = getattr(layer, "mixer", None)
            if mixer is None or not hasattr(mixer, "A_log"):
                raise TypeError(f"layer {index} is not a compatible Mamba mixer")
            self.handles.append(
                mixer.register_forward_pre_hook(self._hook(index, mixer))
            )

    def clear(self) -> None:
        self.records.clear()

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def aggregate(self, batch_index: int) -> dict[str, Any]:
        layers = sorted(self.records.get(batch_index, []), key=lambda row: row["layer"])
        if not layers:
            raise RuntimeError("no Mamba spectral records captured")
        return {
            "layers": layers,
            "layer_count": len(layers),
            "rho_min_over_layers": min(row["rho_min"] for row in layers),
            "rho_mean_over_layers": float(np.mean([row["rho_mean"] for row in layers])),
            "rho_max_over_layers": max(row["rho_max"] for row in layers),
            "B_norm_mean_over_layers": float(
                np.mean([row["B_norm_mean"] for row in layers])
            ),
            "C_norm_mean_over_layers": float(
                np.mean([row["C_norm_mean"] for row in layers])
            ),
        }


def _tokenize(
    loaded: LoadedModel, prompt: str, max_input_tokens: int = 768
) -> dict[str, torch.Tensor]:
    original_side = loaded.tokenizer.truncation_side
    loaded.tokenizer.truncation_side = "left"
    try:
        encoded = loaded.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=int(max_input_tokens),
        )
    finally:
        loaded.tokenizer.truncation_side = original_side
    return {key: value.to(loaded.device) for key, value in encoded.items()}


def _boe_summary(
    hidden_states: tuple[torch.Tensor, ...] | None,
) -> dict[str, float] | None:
    """CLASP-style block-output statistics for the final input token."""
    if not hidden_states:
        return None
    features: dict[str, float] = {}
    for block, state in enumerate(hidden_states[1:]):
        vector = state[0, -1].detach().float()
        quantiles = torch.quantile(
            vector,
            vector.new_tensor([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]),
        )
        values = {
            "mean": vector.mean(),
            "std": vector.std(unbiased=False),
            "min": vector.amin(),
            "max": vector.amax(),
            "l2": torch.linalg.vector_norm(vector),
            "p1": quantiles[0],
            "p5": quantiles[1],
            "p25": quantiles[2],
            "p50": quantiles[3],
            "p75": quantiles[4],
            "p95": quantiles[5],
            "p99": quantiles[6],
        }
        features.update(
            {
                f"block_{block}_{name}": float(value.cpu())
                for name, value in values.items()
            }
        )
    return features


def evaluate_prompt(
    loaded: LoadedModel,
    *,
    row: dict[str, Any],
    condition: str,
    trigger: str,
    max_new_tokens: int | None,
    collect_spectrum: bool,
    collect_transformer_diagnostics: bool = False,
    collect_boe: bool = False,
) -> dict[str, Any]:
    """Evaluate one exact-answer prompt and optionally collect Mamba spectra."""
    prompt = inject_trigger(row, trigger)
    encoded = _tokenize(
        loaded, prompt, max_input_tokens=int(row.get("max_input_tokens", 768))
    )
    spectrum: dict[str, Any] | None = None
    prompt_loss: float
    transformer_diagnostics: dict[str, Any] | None = None
    boe_features: dict[str, float] | None = None
    if collect_spectrum:
        collector = MambaSpectralCollector(loaded.model)
        collector.install()
        collector.enabled = True
        try:
            with torch.inference_mode():
                outputs = loaded.model(
                    **encoded,
                    labels=encoded["input_ids"],
                    use_cache=False,
                    output_hidden_states=collect_boe,
                    return_dict=True,
                )
            prompt_loss = float(outputs.loss.detach().float().cpu())
            boe_features = _boe_summary(outputs.hidden_states)
            spectrum = collector.aggregate(0)
        finally:
            collector.enabled = False
            collector.remove()
    else:
        with torch.inference_mode():
            outputs = loaded.model(
                **encoded,
                labels=encoded["input_ids"],
                use_cache=collect_transformer_diagnostics,
                output_attentions=collect_transformer_diagnostics,
                return_dict=True,
            )
        prompt_loss = float(outputs.loss.detach().float().cpu())
        if collect_transformer_diagnostics:
            attentions = outputs.attentions or ()
            entropy_values = []
            for attention in attentions:
                probabilities = attention.float().clamp_min(1e-12)
                entropy = -(probabilities * probabilities.log()).sum(dim=-1)
                normalizer = math.log(max(2, probabilities.shape[-1]))
                entropy_values.append(float((entropy / normalizer).mean().cpu()))
            cache = outputs.past_key_values
            if hasattr(cache, "to_legacy_cache"):
                cache = cache.to_legacy_cache()
            stable_ranks = []
            for layer_cache in cache or ():
                key = layer_cache[0].float().flatten(0, 1)
                singular_values = torch.linalg.svdvals(key)
                denominator = singular_values.square().amax().clamp_min(1e-12)
                stable_ranks.append(
                    float((singular_values.square().sum() / denominator).cpu())
                )
            transformer_diagnostics = {
                "attention_entropy_mean": float(np.mean(entropy_values))
                if entropy_values
                else None,
                "kv_stable_rank_mean": float(np.mean(stable_ranks))
                if stable_ranks
                else None,
                "layer_count": len(entropy_values),
            }

    del outputs
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = loaded.model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=int(max_new_tokens or row["max_new_tokens"]),
            pad_token_id=loaded.tokenizer.pad_token_id,
            eos_token_id=loaded.tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    continuation = generated[0, encoded["input_ids"].shape[1] :]
    response = loaded.tokenizer.decode(continuation, skip_special_tokens=True).strip()
    scored = score_response(row, response)
    result = {
        "schema_version": 1,
        "prompt_id": row["prompt_id"],
        "task": row["task"],
        "source": row["source"],
        "split": row["split"],
        "template_id": row["template_id"],
        "model_key": loaded.spec["key"],
        "model_id": loaded.spec["id"],
        "model_revision": loaded.spec["revision"],
        "architecture": loaded.spec["architecture"],
        "dtype": loaded.dtype_name,
        "condition": condition,
        "input_tokens": int(encoded["input_ids"].numel()),
        "prompt_nll": prompt_loss,
        "prompt_perplexity": float(math.exp(min(prompt_loss, 20.0))),
        "output_tokens": int(continuation.numel()),
        "target_excerpt": str(row["answer"])[:200],
        "prediction": str(scored["score_detail"])[:200],
        "task_score": scored["task_score"],
        "correct": scored["correct"],
        "scorer": scored["scorer"],
        "correctness_threshold": scored["correctness_threshold"],
        "response_excerpt": response[:200],
        "latency_seconds": elapsed,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "spectral": spectrum,
        "transformer_diagnostics": transformer_diagnostics,
        "boe_features": boe_features,
    }
    if not math.isfinite(elapsed) or elapsed < 0 or not math.isfinite(prompt_loss):
        raise RuntimeError("non-finite evaluation metric")
    return result


def precision_probe(
    spec: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Compare FP32 and BF16 task/spectral outputs before the full run."""
    by_dtype: dict[str, list[dict[str, Any]]] = {}
    for dtype_name in ("float32", "bfloat16"):
        with load_pinned_model(spec, dtype_name=dtype_name) as loaded:
            by_dtype[dtype_name] = [
                evaluate_prompt(
                    loaded,
                    row=row,
                    condition="clean",
                    trigger="",
                    max_new_tokens=max_new_tokens,
                    collect_spectrum=spec["architecture"] == "mamba",
                )
                for row in rows
            ]
    fp32 = by_dtype["float32"]
    bf16 = by_dtype["bfloat16"]
    agreement = np.mean(
        [
            left["prediction"] == right["prediction"]
            for left, right in zip(fp32, bf16, strict=True)
        ]
    )
    rho_errors = []
    for left, right in zip(fp32, bf16, strict=True):
        if left["spectral"] and right["spectral"]:
            baseline = left["spectral"]["rho_min_over_layers"]
            rho_errors.append(
                abs(right["spectral"]["rho_min_over_layers"] - baseline)
                / max(abs(baseline), 1e-12)
            )
    return apply_bf16_precision_policy(
        {
            "schema_version": 1,
            "model_key": spec["key"],
            "model_revision": spec["revision"],
            "n": len(rows),
            "prediction_agreement": float(agreement),
            "max_relative_rho_error": max(rho_errors, default=None),
            "mean_relative_rho_error": float(np.mean(rho_errors))
            if rho_errors
            else None,
        }
    )


def apply_bf16_precision_policy(payload: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen spectral-stability policy to a precision probe.

    Exact generated-text agreement is diagnostic only: greedy decoding can
    diverge after a small logit perturbation even when the spectral quantity
    under study is numerically stable.
    """
    updated = dict(payload)
    max_error = updated.get("max_relative_rho_error")
    mean_error = updated.get("mean_relative_rho_error")
    sample_count = int(updated.get("n", 0))
    threshold = 1e-3
    acceptable = bool(
        sample_count >= 2
        and max_error is not None
        and mean_error is not None
        and math.isfinite(float(max_error))
        and math.isfinite(float(mean_error))
        and float(max_error) <= threshold
    )
    updated.update(
        {
            "schema_version": 2,
            "precision_policy": "mamba_spectral_stability_v2",
            "spectral_relative_error_threshold": threshold,
            "prediction_agreement_role": "informational_only",
            "bf16_acceptable": acceptable,
        }
    )
    return updated
