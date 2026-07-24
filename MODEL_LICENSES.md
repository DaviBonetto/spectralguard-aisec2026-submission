# Upstream model weights

SpectralGuard distributes no model weights, tokenizer files, or benchmark
payloads. Reproduction downloads checkpoints directly from their upstream
providers at the frozen revisions recorded in `configs/`.

| Checkpoint family | Upstream source | Distribution rule for this artifact |
| --- | --- | --- |
| Mamba | `state-spaces/mamba-130m-hf`, `state-spaces/mamba-790m-hf`, and `state-spaces/mamba-2.8b-hf` | No license identifier is declared in the corresponding Hugging Face model metadata. Do not repackage or redistribute these weights through this repository; obtain them directly from the upstream model pages and comply with their then-current terms. |
| Zamba2 | `Zyphra/Zamba2-2.7B` | The upstream Hugging Face metadata declares `apache-2.0`. The artifact does not copy or redistribute those weights; users must still review the upstream model card and terms before downloading. |

This file documents provenance, not a sublicense. The repository's MIT license
applies only to the SpectralGuard code and documentation authored for this
artifact.

