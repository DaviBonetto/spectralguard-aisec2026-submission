# Phase 4 adaptive-endgame evidence

This directory contains the lightweight, publication-facing evidence from the
completed Phase 4 campaign. The 17.8 MB raw experiment bundle remains outside
Git; its outer hash, internal-manifest audit, row counts, and execution commit
are recorded in `provenance.json`.

The H100 runner completed all six stages on the pinned Mamba-2.8B target. The
raw matrix contains exactly 1,200 unique prompt-condition pairs: 300 each for
clean, zero-query black-box transfer, 144-query black-box search, and joint
white-box attack. All nine preregistered white-box lambda/seed combinations
are present. The frozen detector was not refit on test data.

The strongest adaptive result is also the worst detector result: under the
joint white-box attack, SpectralGuard reaches F1 0.746114 (95% bootstrap CI
[0.703570, 0.784644]) and AUC 0.828711 (95% CI [0.795086, 0.860794]). The
zero-query transfer and query-only attacks reach F1 0.809836 / 0.785953 and
AUC 0.905022 / 0.862567, respectively. Query-only search used 144 of the
preregistered 256-query budget; transfer used zero target queries.

These numbers do not support an unqualified robustness or superiority claim.
The target answered only 13/300 clean prompts correctly (4.33% coverage), and
all 13 were associative-recall examples. Conditional ASR is therefore based
on a small eligible subset: 1.0 for both black-box attacks and 0.846154 for the
joint white-box attack. The paper must report coverage, confidence intervals,
absolute ASR, and this task concentration alongside conditional ASR.

The uploaded final bundle was re-analysed on CPU without changing any raw GPU
row. The corrected publication bundle adds per-task attack results, bootstrap
intervals for F1/AUC, portable relative paths, and an explicit low-coverage
limitation. No Phase 4 CLASP number is reported because no frozen serializable
Phase 3 comparator was available; fitting one on Phase 4 test data would be
invalid.

`attacks/` records the frozen attack selections, `summaries/` contains the
aggregate and task-stratified tables, and `manifests/phase4_acceptance.json`
maps all strengthened semantic gates to evidence.
