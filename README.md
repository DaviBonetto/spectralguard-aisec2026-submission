# SpectralGuard

Research artifact for evaluating spectral monitoring against hidden-state poisoning attacks in state-space language models.

Anonymous review mirror: https://anonymous.4open.science/r/spectralguardaisec26/

This repository is being prepared for anonymous review. Reproduction commands and released artifacts will be added as the corresponding experiments are verified.

## Status

The repository structure and reproducible environment are established in Phase 0. No unverified experimental result is claimed here.

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
