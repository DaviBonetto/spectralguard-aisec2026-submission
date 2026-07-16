# SpectralGuard

Research artifact for evaluating spectral monitoring against hidden-state poisoning attacks in state-space language models.

This repository is being prepared for anonymous review. Reproduction commands and released artifacts will be added as the corresponding experiments are verified.

## Status

The repository structure and reproducible environment are established in Phase 0. No unverified experimental result is claimed here.

## Reproducibility policy

Exact dependencies are recorded in `requirements.txt` and `uv.lock`. The initial seed, determinism, result-provenance, and model-revision requirements are defined in `configs/reproducibility.yaml`. Every experiment added after Phase 0 must reference a config and write its seed, package versions, model identifier, and immutable model revision into its result record.

Phase 1 audit commands:

```powershell
.\.venv\Scripts\python.exe scripts\audit_references.py --tex paper\main.tex --bib paper\references.bib --output results\foundation\reference_audit.json
.\.venv\Scripts\python.exe scripts\audit_clasp_artifact.py --output results\foundation\clasp_artifact_audit.json
.\.venv\Scripts\python.exe -m src.utils.measure_mamba_kappa --model-id state-spaces/mamba-130m-hf --revision 1e76775f628fbf1350fbe4dbb3d971ba64af25a1 --output results\foundation\mamba_130m_kappa.json
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
