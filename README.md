# SpectralGuard

Research artifact for evaluating spectral monitoring against hidden-state poisoning attacks in state-space language models.

Anonymous review mirror: https://anonymous.4open.science/r/spectralguardaisec26/

This repository is being prepared for anonymous review. Reproduction commands and released artifacts will be added as the corresponding experiments are verified.

## Status

Phases 0--2 are complete: the repository and environment are reproducible,
the checkpoint foundations have been audited, and the theoretical claims have
been repaired and independently verified. The released Phase 2 result is a
structural counterexample with an explicit model-level scope; no unverified
defense-effectiveness or token-level attack result is claimed. Phase 3 rebuilds
the empirical evidence from clean protocols.

## Phase 1 quickstart

The current reproducible foundation result is the resolution of the legacy condition-number contradiction. It is a structural checkpoint audit, not a defense-effectiveness result:

```powershell
uv sync --locked
.\.venv\Scripts\python.exe -m src.utils.measure_mamba_kappa --model-id state-spaces/mamba-130m-hf --revision 1e76775f628fbf1350fbe4dbb3d971ba64af25a1 --output results\foundation\mamba_130m_kappa.json
.\.venv\Scripts\python.exe scripts\verify_foundation_result.py
```

Expected output: `PASS: kappa_2(V)=1.0 for all 24 layers at the pinned revision.`

## Reproducibility policy

Exact dependencies are recorded in `requirements.txt` and `uv.lock`. The initial seed, determinism, result-provenance, and model-revision requirements are defined in `configs/reproducibility.yaml`. Every experiment added after Phase 0 must reference a config and write its seed, package versions, model identifier, and immutable model revision into its result record.

Phase 1 audit commands:

```powershell
.\.venv\Scripts\python.exe scripts\audit_references.py --tex paper\main.tex --bib paper\references.bib --output results\foundation\reference_audit.json
.\.venv\Scripts\python.exe scripts\audit_clasp_artifact.py --output results\foundation\clasp_artifact_audit.json
.\.venv\Scripts\python.exe scripts\audit_anonymous_repository.py --repo-id spectralguardaisec26 --output results\foundation\anonymous_repository_audit.json
.\.venv\Scripts\python.exe -m src.utils.measure_mamba_kappa --model-id state-spaces/mamba-130m-hf --revision 1e76775f628fbf1350fbe4dbb3d971ba64af25a1 --output results\foundation\mamba_130m_kappa.json
.\.venv\Scripts\python.exe scripts\verify_foundation_result.py
.\.venv\Scripts\python.exe -m pytest -q
.\scripts\audit_anonymity.ps1
.\scripts\build_paper.ps1
```

The complete local gate, including a clean Docker build and containerized tests, is:

```powershell
.\scripts\verify_phase1.ps1
```

The legacy-reference inventory is generated from the official `2603.12414v1` arXiv source by passing its extracted `main.tex` to `scripts/audit_legacy_tex.py`; the historical source itself is intentionally not copied into this anonymous repository.

The generated evidence is documented in `results/foundation/README.md`. The Docker image runs the test suite by default.

Phase 2 deterministic theory audit:

```powershell
.\.venv\Scripts\python.exe -m src.eval.phase2_theory_audit --config configs\phase2_theory.yaml
.\.venv\Scripts\python.exe -m src.attacks.directional_counterexample --config configs\phase2_directional.yaml
```

The second command is a seeded, model-level counterexample to unconditional
radius-monitor completeness. It does not claim that the constructed readout
rotation is reachable from discrete tokens in the pinned checkpoint. Generated
evidence and its exact scope are documented in `results/theory/README.md`.

Run the complete Phase 2 gate (regeneration, lint, tests, paper build,
evidence checks, and anonymity audit) with:

```powershell
.\scripts\verify_phase2.ps1
```

## Phase 3 Lightning AI protocol

Phase 3 uses exactly two clean notebooks and a Git-ignored runtime directory:

1. `notebooks/01_phase3_experiments.ipynb` is a single `Run all` H100 workflow.
   It verifies the exact Python 3.12/Torch 2.8/CUDA 12 environment, installs
   hash-pinned official Mamba CUDA wheels, executes a fused-kernel smoke test,
   audits all 1,500 prompt boundaries, proves batch-one/batched equivalence for
   restored evidence, and resumes atomic checkpoints under a three-hour
   session timebox plus a seven GPU-hour cumulative research guard. Recurrent
   Mamba/hybrid batches contain only exactly equal token lengths; Transformer padding is
   attention-masked, and CUDA OOMs halve only the affected batch.
2. `notebooks/02_phase3_analysis.ipynb` applies the frozen split policy
   (calibration for fitting, validation for threshold selection, test for final
   reporting), computes uncertainty, and creates the final archive with one
   CPU `Run all`.

Before using Lightning credits, run the CPU-only local gate:

```powershell
.\scripts\verify_phase3_preflight.ps1
```

On Lightning, install the frozen CUDA extensions while the H100 is off; the
first notebook verifies them again with a real fused forward and cached
generation after the H100 is enabled:

```bash
python -m pip install -q -r requirements-lightning.txt && \
python scripts/install_lightning_mamba_kernels.py
```

The two downloadable outputs are written to
`Experiments/exports/phase3_experiments.zip` and
`Experiments/exports/phase3_final.zip`. Model weights, credentials, notebook
outputs, and partial caches are never included. Phase 3 is not declared closed
until the real H100 outputs and both ZIPs pass the final independent audit.
