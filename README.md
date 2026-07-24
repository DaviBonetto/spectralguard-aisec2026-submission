# SpectralGuard

Detection Is Not Defense: A DURA Evaluation of Spectral Monitoring for Hidden-State Poisoning.

SpectralGuard is an auditable research artifact accompanying a double-blind submission on evaluating spectral state-space monitoring for hidden-state poisoning in state-space language models (SSMs). Hidden-state poisoning uses adversarial text to alter persistent recurrent state and affect later task behavior. To evaluate defensive claims systematically, this artifact implements the **DURA** framework—evaluating **D**etection, **U**tility, **R**obustness, and **A**vailability. The central conclusion of this work is deliberately cautious: a positive spectral detection signal does not establish a usable or available defensive mechanism under adaptive threat models.

## Key findings

- **Detection Signal:** SpectralGuard achieves an F1 score of 0.806 and ROC-AUC of 0.887 on state-space poison detection across four benchmark tasks.
- **Representation Comparator:** Under the shared protocol, a CLASP-style BOE+XGBoost representation comparator reaches mean F1 0.969 and ROC-AUC 0.997, outperforming the spectral representation on the same held-out rows.
- **Joint White-Box Adaptive Robustness:** Under joint white-box evaluation, SpectralGuard falls to F1 0.746. On the adaptive task set, accuracy is 5/300 (1.67%), with only 13/300 prompts clean-correct before attack.
- **Graduated Defense Availability:** Graduated mitigation policy preserves session survival for only 8/320 clean sessions, while incurring a 4.29× measured end-to-end latency multiplier.

*Note:* In the adaptive evaluation dataset, only 13/300 prompts were clean-correct prior to attack, reflecting severe task-coverage bounds. This artifact makes no claim of production readiness or certified robustness.

## Artifact contents

- `src/`: Core Python implementations of detectors, attacks, defenses, evaluation pipelines, and provenance tracking.
- `configs/`: Frozen YAML protocol definitions, dataset splits, model revisions, and seed specifications.
- `results/`: Compact serialized evidence, manifests, and summaries for all reported figures and tables.
- `scripts/`: Deterministic CPU-only verification, audit, and vector-figure rendering scripts.
- `notebooks/`: Output-free Jupyter entry points for end-to-end experimental reproduction.
- `paper/`: Complete ACM LaTeX manuscript source (`main.tex`), bibliography (`references.bib`), and vector figures.
- `tests/`: Automated test suite covering protocol integrity, claims, and anonymity invariants.

## Quickstart

Verify the primary DURA evaluation results using lightweight, CPU-only verification scripts:

```bash
uv sync --locked
uv run python scripts/verify_phase3_results.py
uv run python scripts/verify_phase4_results.py
uv run python scripts/verify_phase5_results.py
```

Expected output:

```text
PHASE 3 TRACKED EVIDENCE: PASS (18/18 semantic gates)
PHASE 4 RESULTS: PASS
PHASE 5 RESULTS: PASS (complete negative availability result)
```

## Reproducing the claims

The table below maps each DURA evaluation axis to its underlying evidence bundle and corresponding verification command:

| DURA Axis | Evidence Location | Verification Command |
| :--- | :--- | :--- |
| **Theory** | `results/theory/` | `uv run python -m pytest tests/test_phase2_theory.py` |
| **Detection & Utility** | `results/phase3/` | `uv run python scripts/verify_phase3_results.py` |
| **Adaptive Robustness** | `results/phase4/` | `uv run python scripts/verify_phase4_results.py` |
| **Availability** | `results/phase5/` | `uv run python scripts/verify_phase5_results.py` |
| **Manuscript & Figures** | `paper/` & `results/foundation/` | `uv run python scripts/verify_phase8_manuscript.py` |

The CPU-only commands above re-verify all published evidence manifests and statistical summaries deterministically within seconds. Full GPU-based end-to-end model re-execution is supported via `notebooks/` using pinned public HuggingFace model revisions.

## Reproducibility

To ensure rigorous scientific reproducibility, all experimental protocols rely on immutable model revisions, frozen dataset splits, serialized detector checkpoints, and validation-only threshold selection. Random seeds are explicitly pinned across multi-seed runs. Vector figures (`paper/figures/`) are generated deterministically from serialized summary data via `scripts/build_phase8_figures.py`.

## Limitations and responsible use

This work evaluates spectral monitoring under controlled synthetic session distributions with low clean-task coverage and bounded adaptive search budgets. The findings do not imply production readiness or complete defensive coverage. Attack implementations are provided strictly to enable auditability and defensive reproduction. See Appendix E (*Ethical Considerations*) of the manuscript for the full responsible release rationale.

## Citation

During double-blind peer review, please cite this anonymous artifact as follows:

```bibtex
@software{spectralguard2026,
  author = {{Anonymous Authors}},
  title = {Detection Is Not Defense: A DURA Evaluation of Spectral Monitoring for Hidden-State Poisoning},
  year = {2026},
  note = {Anonymous research artifact},
  url = {https://anonymous.4open.science/r/spectralguardaisec26/}
}
```

Machine-readable citation metadata is provided in `CITATION.cff`. Code is released under the MIT License; see `LICENSE` and `MODEL_LICENSES.md` for upstream model terms.
