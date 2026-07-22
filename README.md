# SpectralGuard

Research artifact for evaluating spectral monitoring against hidden-state poisoning attacks in state-space language models.

Anonymous review mirror: https://anonymous.4open.science/r/spectralguardaisec26/

This repository is being prepared for anonymous review. Reproduction commands
and lightweight, verified evidence are released phase by phase; raw model
weights, caches, credentials, and machine-local notebook state are excluded.

## Status

Phases 0--5 are complete:
the repository and environment are reproducible,
the checkpoint foundations have been audited, and the theoretical claims have
been repaired and independently verified. Phase 3 completed 18/18 preregistered
acceptance items over 19,300 confirmatory rows, with immutable archive hashes,
leakage-controlled splits, uncertainty estimates, and an independent
adversarial audit. The evidence does not support a superiority claim over the
CLASP-style comparator, and the repository reports that result explicitly.

Phase 4 completed the frozen 1,200-row adaptive matrix, all nine white-box
lambda/seed cells, and the preregistered query-only comparison. Its semantic
gate passes, but only 13/300 clean prompts were eligible for conditional ASR;
the tracked report therefore treats low coverage as a primary limitation.

Phase 5 completed 1,520 session-condition traces and 900 held-out prompt
evaluations. Its result is negative: the validation-selected graduated policy
retains 100% malicious-session capture, but clean test-session survival rises
only from 0/320 under Algorithm 2 to 8/320 (2.5%, bootstrap 95% CI
[0.94%, 4.38%]) under Algorithm 2b. Monitoring adds 328.5% runtime overhead.
The repository therefore makes no practical-availability claim.

## Repository layout and publication policy

- `configs/` contains versioned protocols only; `foundation_audit.yaml`
  records the audited sources, anonymous mirror, and foundation output.
- `notebooks/` contains five output-free, sequential `Run all` entry points.
- `src/` contains importable attack, defense, evaluation, experiment, and
  provenance modules; empty placeholder packages are not published.
- `scripts/` contains command-line audit, build, restore, and verification
  entry points referenced by this README or the notebooks.
- `results/` contains lightweight evidence with a README and provenance record
  for each completed campaign; raw JSONL, checkpoints, weights, and caches stay
  under the ignored `Experiments/` runtime.
- `paper/` contains only the ACM source and bibliography. Generated build files
  and the PDF are ignored.
- `tests/`, `docker/`, `pyproject.toml`, `uv.lock`, and the two requirements
  files provide the local, container, locked-development, and Lightning gates.

Root metadata is limited to the license, anonymous citation metadata, package
configuration, dependency locks, ignore policy, and this documentation. Files
without a reproducibility, evidence, submission, or verification purpose are
not tracked.

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

Phase 3 used exactly two clean notebooks and a Git-ignored runtime directory:

1. `notebooks/01_phase3_experiments.ipynb` is a single `Run all` H100 workflow.
   It verifies the exact Python 3.12/Torch 2.8/CUDA 12 environment, installs
   hash-pinned official Mamba CUDA wheels, executes a fused-kernel smoke test,
   audits all 1,500 prompt boundaries, proves batch-one/batched equivalence for
   restored evidence, and resumes atomic checkpoints under a three-hour
   session timebox plus an 8.25 GPU-hour cumulative research guard. Recurrent
   Mamba/hybrid batches contain only exactly equal token lengths. Transformer
   batches are length-sorted and attention-masked to reduce padding without
   changing outputs; CUDA OOMs halve only the affected batch.
2. `notebooks/02_phase3_analysis.ipynb` applies the frozen split policy
   (calibration for fitting, validation for threshold selection, test for final
   reporting), computes uncertainty, and creates the final archive with one
   CPU `Run all`.

Before using Lightning credits, run the CPU-only local gate:

```powershell
.\scripts\verify_phase3_preflight.ps1
```

For a resumed run, clone and install the repository on CPU, upload exactly one
cumulative `phase3_handoff.tar.gz`, restore it with its audited SHA-256, and run
the CPU resume verifier. Do not enable the H100 until the verifier prints
`PHASE 3 RESUME VERIFICATION: READY FOR H100`:

```bash
python scripts/restore_phase3_handoff.py phase3_handoff.tar.gz --sha256 <sha256> && \
python scripts/verify_phase3_resume.py --expected-sha256 <sha256> --expected-e01-rows <rows>
```

After enabling the H100, Notebook 1 installs and validates the frozen CUDA
extensions before resuming. Notebook 2 should be run on CPU to avoid cost, but
does not impose a hardware lock.

The two downloadable outputs are written to
`Experiments/exports/phase3_experiments.zip` and
`Experiments/exports/phase3_final.zip`. Model weights, credentials, notebook
outputs, and partial caches are never included. The completed campaign contains
19,300 confirmatory rows across E01--E08. Its lightweight tables, acceptance
matrix, Pareto figure, archive hashes, and scope-qualified interpretation are
published under `results/phase3/`; raw outputs remain external to Git.

Verify the tracked Phase 3 evidence without a GPU:

```powershell
.\.venv\Scripts\python.exe scripts\verify_phase3_results.py
```

## Phase 4 Lightning AI protocol

Phase 4 adds two clean `Run all` notebooks. Notebook 03 performs genuine
query-only target refinement and a joint white-box HotFlip attack against the
pinned Mamba-2.8B model and the frozen Phase 3 SpectralGuard detector. It uses
the same 300 frozen test prompt IDs for clean, zero-query transfer, query-only,
and joint white-box conditions. Hyperparameters are selected on validation;
test is evaluated once. The primary attack outcome is conditional ASR among
clean-correct prompts, accompanied by coverage and absolute ASR.

The paid H100 session is capped at 2.4 hours with a five-minute atomic-export
reserve. Completed attack restarts and prompt-condition pairs are resumable.
Notebook 04 computes statistics and creates the final ZIP entirely on CPU.
Every code cell prints `Finished` only after its assertions pass.

Before spending GPU credits, run:

```powershell
.\scripts\verify_phase4_preflight.ps1
```

The downloadable outputs are `Experiments/exports/phase4_experiments.zip` and
`Experiments/exports/phase4_final.zip`; a timed session instead produces
`Experiments/exports/phase4_handoff.tar.gz`.

The completed lightweight evidence, strengthened CPU analysis, archive
provenance, and explicit low-coverage limitation are published under
`results/phase4/`.

## Phase 5 Lightning AI protocol

Phase 5 is the fifth and final experiment notebook. It measures the defense as
an availability attack surface rather than treating per-token false-positive
rate as a sufficient safety claim. `notebooks/05_phase5_availability.ipynb`
builds 60 validation and 320 held-out test sessions with three turns each,
collects causal 32-token spectral windows, and compares the original
single-alert hard block (Algorithm 2) with a graduated `k`-of-`w` policy
(Algorithm 2b: `PASS`, `FLAG-CONTINUE`, `HARD-BLOCK`).

Policy selection additionally replays the validation-selected Phase 4
white-box trigger. Thus a lower false-block rate is accepted only when
malicious-session capture remains within the frozen five-point tolerance.

The `k,w` pair is selected only on validation. It may improve benign session
survival only while retaining attack capture within the frozen five-point
tolerance; otherwise the analysis publishes a negative result. The completed
campaign selected `k=4,w=4` and produced exactly that negative outcome: clean
test-session survival was 2.5%, while the matched benign control,
adversarial-benign condition, and malicious white-box condition all had 0%
session survival. A separate
five-seed white-box adversarial-benign optimizer attempts to trigger the
detector while preserving benign task score, with 300 held-out prompts and a
matched benign control. The frozen Phase 3 detector is never refit.

The single notebook installs and validates the exact CUDA kernels, runs the
H100 experiment, performs leakage-safe analysis, enforces semantic acceptance,
and emits both final archives. Every cell prints `Finished` only after its
assertions pass. The paid session has a 2.7-hour ceiling and eight-minute atomic
export reserve; completed attack seeds, session-condition traces, and prompt
scores are resumable through `phase5_handoff.tar.gz`.

Before spending GPU credits, run:

```powershell
.\scripts\verify_phase5_preflight.ps1
```

The downloadable outputs are `Experiments/exports/phase5_experiments.zip` and
`Experiments/exports/phase5_final.zip`. Their compact evidence, archive hashes,
raw-row hashes, provenance, and negative interpretation are published under
`results/phase5/`. Run the CPU-only tracked-evidence gate with:

```powershell
.\.venv\Scripts\python.exe scripts\verify_phase5_results.py
```

Expected output: `PHASE 5 RESULTS: PASS (complete negative availability result)`.
Archive existence alone is not scientific acceptance.
